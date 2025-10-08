import logging
import pandas as pd
import numpy as np
import itertools
import operator

from pipeline_execution import PipelineExecutor
from score_lookup import ScoreLookup

np.random.seed(42)

class GlassBoxOptimizer:
    """
    A glass-box optimizer that chooses between two ranking methods:
      - Similarity-only (wS=1, wU=0)
      - Prediction-only (wS=0, wU=1)

    Selection is learned from the first K optimize() calls (each with its own initial_params),
    using a pilot that measures 'iterations to pass' per action. There is NO cap in the pilot:
    for each action, we evaluate candidates until the pass threshold is met; if none pass,
    the action's k is (#attempts + 1) as a penalty.

    After PILOT_LEARN_CALLS pilots, the best action is frozen and reused for all future optimize() calls.

    IMPORTANT: Pilot evaluations DO NOT increment self.rank_iter. After choosing the best action,
    we set self.rank_iter = pilot_k_for_chosen_action so rank_iter reflects the best action cost only.
    The subsequent optimistic and exhaustive searches then add their own iterations to self.rank_iter.
    """

    # How many optimize() calls to use for learning the best action before freezing
    PILOT_LEARN_CALLS = 8

    def __init__(self, dataset_name, model_type, metric_type, pipeline_type, pipeline_order, filename_train, filename_test, new_components):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.metric_type = metric_type
        self.pipeline_type = pipeline_type
        self.pipeline_order = pipeline_order
        self.filename_train = filename_train
        self.filename_test = filename_test
        self.new_components = new_components

        self.fail = 0
        self.pass_ = 0
        self.fail_with_fallback = 0
        self.rank_iter = 0
        self.rank_f = 0
        self.ranges = {}
        self.base_strategies = pipeline_order

        self.executor_pass = PipelineExecutor(
            pipeline_type=self.pipeline_type,
            dataset_name=self.dataset_name,
            metric_type=self.metric_type,
            pipeline_ord=self.pipeline_order
        )

        self.pasing_hist_data = pd.read_csv(self.filename_train)
        self.profiles = self.executor_pass.get_header(self.filename_train)
        self.score_lookup = ScoreLookup(pipeline_order, metric_type)

        # ---- Learning state across calls (two actions only) ----
        # Track cumulative attempts (k) and counts per action over the first N calls
        # {action_name: {"sum_k": float, "count": int}}
        self._pilot_stats = {
            "similarity": {"sum_k": 0.0, "count": 0},
            "prediction": {"sum_k": 0.0, "count": 0},
        }
        self._pilot_calls_done = 0
        # Once determined, store chosen weights here
        self._chosen_weights = None  # tuple (wS, wU)

        # Expose per-call pilot result
        self.pilot_choice = None  # "similarity" or "prediction"
        self.pilot_k = None       # #evaluations needed by the chosen action in the pilot (best case for this call)

    # ---------------- helpers ----------------

    def set_ranges(self):
        for strategy in self.pipeline_order:
            self.ranges[strategy] = list(np.unique(self.pasing_hist_data[strategy]))

    @staticmethod
    def _action_to_weights(name: str):
        if name == "similarity":
            return 1.0, 0.0
        # "prediction"
        return 0.0, 1.0

    def _build_ranked_candidates(self, cur_params_opt, new_components, wS, wU):
        new_components=self.new_components
        return self.executor_pass.evaluate_interventions_pred_and_similarity(
            [int(v) for v in cur_params_opt.values()],
            self.filename_train,
            new_components=list(new_components),
            wS=wS,
            wU=wU
        )

    def _apply_candidate(self, component, strategy, pos, cur_params_dict):
        """
        Build (order, vec) for a candidate; also return a unique key for deduping.
        """
        original_order = self.pipeline_order.copy()
        if component in original_order:
            idx = original_order.index(component)
            new_params = cur_params_dict.copy()
            new_params[original_order[idx]] = int(strategy)
            intervened_order = original_order
        else:
            if pos is None:
                return None, None, None
            pos = int(pos)
            intervened_order = original_order[:pos] + [component] + original_order[pos:]
            items = list(cur_params_dict.items())
            items = items[:pos] + [(component, int(strategy))] + items[pos:]
            new_params = dict(items)
        vec = [int(new_params[s]) for s in intervened_order]
        key = tuple((name, new_params[name]) for name in intervened_order)
        return intervened_order, vec, key

    def _eval_pipeline_pass(self, order, vec, f_goal, count_iter: bool = True):
        """
        Executes the pipeline once and returns True if pass (util <= f_goal).
        Optionally increments global iteration counter.
        """
        try:
            util = self.executor_pass.current_par_lookup(order, vec)
        except Exception as e:
            logging.warning(f"[EVAL] evaluation error: {e}")
            return False
        if count_iter:
            self.rank_iter += 1
        return (util <= f_goal)

    def _pilot_iterations_to_pass_for_action(self, cur_params_opt, new_components, f_goal, action_name):
        """
        Pilot for one action on the current initial_params (NO CAP):
        - Walk this action’s ranked list and evaluate each fresh candidate in order.
        - Evaluations here DO NOT increment self.rank_iter (count_iter=False).
        - Return k = #evaluations until pass.
        - If we reach the end without passing, return k = (#evaluations attempted) + 1 as penalty.
        """
        new_components=self.new_components
        wS, wU = self._action_to_weights(action_name)
        ranked = self._build_ranked_candidates(cur_params_opt, new_components, wS, wU)

        attempts = 0
        tried_keys = set()  # per-call dedupe
        for cand in ranked:
            component, strategy, similarity, y_pred, pos, _ = cand
            order, vec, key = self._apply_candidate(component, int(strategy), pos, cur_params_opt)
            if key is None or key in tried_keys:
                continue
            tried_keys.add(key)
            attempts += 1
            if self._eval_pipeline_pass(order, vec, f_goal, count_iter=False):
                return attempts  # success in 'attempts' evaluations

        # No candidate from this action passed; penalize by +1 beyond all attempts we made
        return attempts + 1

    def _learn_or_get_weights(self, cur_params_opt, new_components, f_goal):
        """
        Use the first PILOT_LEARN_CALLS calls to learn which action is best (similarity vs prediction).
        Returns chosen (wS, wU) for THIS call. If frozen, skips pilot and returns the frozen choice.
        """
        # If we already chose after N pilots, just return it
        new_components=self.new_components
        if self._chosen_weights is not None:
            # For transparency on this call, expose pilot_* based on frozen choice
            if self._chosen_weights == (1.0, 0.0):
                self.pilot_choice, self.pilot_k = "similarity", None
            else:
                self.pilot_choice, self.pilot_k = "prediction", None
            return self._chosen_weights

        # Otherwise: run a pilot on THIS call (two actions only)
        actions = ["similarity", "prediction"]
        ks = {}
        for name in actions:
            k = self._pilot_iterations_to_pass_for_action(cur_params_opt, new_components, f_goal, name)
            ks[name] = k
            # Update cumulative stats
            self._pilot_stats[name]["sum_k"] += float(k)
            self._pilot_stats[name]["count"] += 1
            logging.info(f"[PILOT] action={name} → attempts_to_pass={k}")

        # Pick best-so-far by lowest average k (for THIS call's choice + freezing after N calls)
        best_name = None
        best_avg = None
        for name in actions:
            c = self._pilot_stats[name]["count"]
            s = self._pilot_stats[name]["sum_k"]
            if c > 0:
                avg_k = s / c
                if best_avg is None or avg_k < best_avg:
                    best_avg = avg_k
                    best_name = name

        # Record the chosen action and its k for THIS call
        self.pilot_choice = best_name
        self.pilot_k = int(ks[best_name]) if best_name in ks else None
        wS_star, wU_star = self._action_to_weights(best_name)
        logging.info(f"[PILOT] best-so-far action={best_name} (avg attempts={best_avg:.3f}, "
                     f"this_call_k={self.pilot_k})")

        # Increment number of pilot calls done & freeze if we hit the limit
        self._pilot_calls_done += 1
        if self._pilot_calls_done >= self.PILOT_LEARN_CALLS:
            self._chosen_weights = (wS_star, wU_star)
            logging.info(f"[PILOT] Finalized action after {self.PILOT_LEARN_CALLS} calls: {best_name} "
                         f"(wS={wS_star}, wU={wU_star})")

        return (wS_star, wU_star)

    # ---------------- main entry ----------------

    def optimize(self, init_params, f_goal, new_components=None, max_depth=None, top_n_actions=120):
        """
        init_params: list of ints for current base pipeline strategies in self.pipeline_order.
        f_goal: pass threshold (lower is better).
        new_components: optional list of modules allowed for insertion (with strategy ranges present in executor).
        """

        new_components=self.new_components

        # Reset per-call counters
        self.rank_iter = 0
        self.optimisitic_iter = 0
        self.exhaustive_iter = 0
        self.rank_f = 0
        self.pilot_choice = None
        self.pilot_k = None

        # Map initial params into dict keyed by component name
        cur_params_opt = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params)}
        cur_param_check = init_params[:len(self.base_strategies)]

        # Initial evaluation (for logging / early pass); counts as 1 iteration only if already passing
        opt_f = self.executor_pass.current_par_lookup(self.base_strategies, cur_param_check)
        logging.info(f'Evaluating {[int(v) for v in cur_params_opt.values()]} -- Initial Utility {opt_f} -- Target Utility {f_goal}')

        if self.pipeline_type == 'ml':
            self.set_ranges()

        seen = set()
        if opt_f <= f_goal:
            # Already meets threshold; report iterations as 1 for this init param
            self.rank_iter = 1
            self.rank_f = opt_f
            return

        # Determine search depth bounds
        max_iter_size = len(self.pipeline_order) + len(set(new_components))
        if max_depth is None:
            max_depth = max_iter_size
        else:
            max_depth = min(max_depth, max_iter_size)

        found = False

        # ---- Choose ranking method for THIS call (learn across first N calls; then freeze) ----
        wS_star, wU_star = self._learn_or_get_weights(cur_params_opt, new_components, f_goal)

        # Seed rank_iter with the best action's pilot cost ONLY
        if self.pilot_k is not None:
            self.rank_iter = int(self.pilot_k)

        # Build final ranking with the chosen method
        self.ranked_interventions = self._build_ranked_candidates(cur_params_opt, new_components, wS_star, wU_star)
        logging.info(f"[CHOICE] Using weights wS={wS_star}, wU={wU_star} for ranking this run. "
                     f"(pilot_choice={self.pilot_choice}, pilot_k={self.pilot_k})")

        # ---- Main search loop (unchanged) ----
        for k in range(1, max_depth + 1):
            logging.info(f"[OPTIMIZE] Trying combo size k={k}")

            if k == 1:
                found, opt_f, cur_params_opt = self.optimistic_search(seen, cur_params_opt, f_goal, opt_f)
            else:
                found, opt_f, cur_params_opt = self.exhaustive_search_similarity_guided(
                    iter_size=k,
                    seen=seen,
                    cur_params_opt=cur_params_opt,
                    f_goal=f_goal,
                    opt_f=opt_f,
                    new_components=new_components,
                    top_n_actions=top_n_actions,
                    early_stop=True
                )

            logging.info(f"Total iterations so far (this init_param): {self.rank_iter}")
            if found:
                break

        if not found:
            self.fail += 1

    # ---------------- existing methods (unchanged logic) ----------------

    def optimistic_search(self, seen, cur_params_opt, f_goal, opt_f):
        logging.info("[INFO] Running optimistic search with combined interventions...")
        original_order = self.pipeline_order.copy()
        for component, strategy, similarity, uti, pos, _ in self.ranked_interventions:
            logging.info(f"[SEARCH] Trying intervention: {component} → {strategy}")

            if component in original_order:
                idx = original_order.index(component)
                cur_params = cur_params_opt.copy()
                cur_params[original_order[idx]] = strategy
                intervened_order = original_order.copy()
            else:
                if pos > len(cur_params_opt):
                    logging.warning(f"Skipping {component}: optimal_position {pos} out of range.")
                    continue
                intervened_order = original_order[:pos] + [component] + original_order[pos:]
                cur_params_items = list(cur_params_opt.items())
                cur_params_items = cur_params_items[:pos] + [(component, strategy)] + cur_params_items[pos:]
                cur_params = dict(cur_params_items)

            if len(cur_params) != len(intervened_order):
                logging.warning(f"[SKIP] Misaligned parameter length for {component}")
                continue

            cur_f = self.executor_pass.current_par_lookup(intervened_order, [int(v) for v in cur_params.values()])
            self.rank_iter += 1
            self.optimisitic_iter += 1
            logging.info(f"[TRY] {component}={strategy}, pipeline: {cur_params}, Utility={cur_f:.4f}, Best={opt_f:.4f}")

            if cur_f <= f_goal:
                logging.info("✅ Target achieved 🎯")
                self.rank_f = cur_f
                self.pass_ += 1
                logging.info(f"required optimistic iterations: {self.optimisitic_iter}")
                logging.info(f"passing pipeline: order={intervened_order}, vec={[int(v) for v in cur_params.values()]}")
                return True, cur_f, cur_params
            elif cur_f < opt_f:
                opt_f = cur_f
                cur_params_opt = cur_params.copy()
                original_order = intervened_order
            else:
                seen.add(tuple(cur_params.items()))
        return False, opt_f, cur_params_opt

    def exhaustive_search_similarity_guided(self,
                                            iter_size,
                                            seen,
                                            cur_params_opt,
                                            f_goal,
                                            opt_f,
                                            new_components,
                                            top_n_actions=100,
                                            early_stop=True,
                                            use_fused=False):
        new_components=self.new_components
        
        logging.info("[INFO] Running optimistic search with combined interventions...]")
        original_order = self.pipeline_order[:]
        baseline_vec = [int(cur_params_opt[s]) for s in original_order]

        actions = []
        base_set = set(original_order)
        for comp, strat, sim, util, pos, _ in self.ranked_interventions:
            if comp in base_set:
                actions.append(("change", comp, int(strat), None, float(sim)))
            else:
                actions.append(("insert", comp, int(strat), int(pos), float(sim)))

        actions.sort(key=lambda a: a[4], reverse=True)
        if top_n_actions is not None and top_n_actions > 0:
            actions = actions[:top_n_actions]

        if iter_size <= 0:
            return False, opt_f, cur_params_opt
        if iter_size > len(actions):
            iter_size = len(actions)

        def _conflict_free(combo):
            seen_change = set()
            seen_insert = set()
            for kind, comp, strat, pos, sim in combo:
                if kind == "change":
                    if comp in seen_change:
                        return False
                    seen_change.add(comp)
                    if comp not in base_set:
                        return False
                else:  # insert
                    if comp in seen_insert:
                        return False
                    if comp in base_set:
                        return False
                    seen_insert.add(comp)
            return True

        def _apply_combo_concurrently(base_order, base_params_dict, combo):
            changes = [(comp, strat) for (kind, comp, strat, pos, sim) in combo if kind == "change"]
            inserts = [(comp, strat, pos, sim) for (kind, comp, strat, pos, sim) in combo if kind == "insert"]
            inserts_sorted = sorted(inserts, key=lambda t: (t[2], -t[3]))

            order = base_order[:]
            params = base_params_dict.copy()

            shift = 0
            for comp, strat, pos, _sim in inserts_sorted:
                ins_at = max(0, min(pos + shift, len(order)))
                order = order[:ins_at] + [comp] + order[ins_at:]
                left = list(params.items())[:ins_at]
                right = list(params.items())[ins_at:]
                params = dict(left + [(comp, int(strat))] + right)
                shift += 1

            for comp, strat in changes:
                if comp in order:
                    params[comp] = int(strat)

            vec = [int(params[s]) for s in order]
            return order, vec, params

        best_params_dict = cur_params_opt
        best_val = opt_f
        found = False

        for combo in itertools.combinations(actions, iter_size):
            if not _conflict_free(combo):
                continue

            eval_order, eval_vec, eval_params_dict = _apply_combo_concurrently(original_order, cur_params_opt, combo)
            key = tuple((name, eval_params_dict[name]) for name in eval_order)
            if key in seen:
                continue
            seen.add(key)
            try:
                cur_f = self.executor_pass.current_par_lookup(eval_order, eval_vec)
            except Exception as e:
                logging.warning(f"[SIM-GUIDED][k={iter_size}] Skipping combo due to eval error: {e}")
                continue

            self.rank_iter += 1
            self.exhaustive_iter += 1
            logging.info(f"[SIM-GUIDED][k={iter_size}] combo={combo}, utility={cur_f:.6f}")

            if early_stop and (cur_f <= f_goal):
                logging.info("✅ Target achieved 🎯")
                logging.info(f"passing pipeline: order={eval_order}, vec={eval_vec}")
                self.rank_f = cur_f
                self.pass_ += 1
                logging.info(f"required exhaustive iterations: {self.exhaustive_iter}")
                return True, cur_f, eval_params_dict

            if cur_f < best_val:
                # (optional) adopt best found so far:
                # best_val = cur_f
                # best_params_dict = eval_params_dict.copy()
                pass

        return found, best_val, best_params_dict

    def rank_individual_intervention(self, cur_strategy, idx, prof_name, val):
        if self.coefs_profile[idx] > 0:
            current_param_value = self.ranges[cur_strategy][0] if self.param_coeff[prof_name][val] > 0 else self.ranges[cur_strategy][-1]
        else:
            current_param_value = self.ranges[cur_strategy][0] if self.param_coeff[prof_name][val] < 0 else self.ranges[cur_strategy][-1]
        return current_param_value
    
    def similarity_based_ranking(self, new_comp, cur_par):
        combined_rank = self.executor_pass.evaluate_combined_intervention(new_comp, cur_par)
        return combined_rank

    def rank_intervention_combination(self, profile_name):
        score_map = {}
        lst = []
        for row in self.historical_data:
            score_val = sum([row[i] * self.param_coeff[profile_name][i] for i in range(len(self.base_strategies))])
            lst.append(score_val, row)
            score_map[score_val] = row

        sorted_param = sorted(score_map.items(), key=operator.itemgetter(0))
        sorted_params_lst = sorted(lst, key=lambda x: x[0])
        return sorted_param, sorted_params_lst

    def write_quartiles(self, csv_writer, algorithm, metric, quartiles, f_goal, f_goals):
        if self.model_type != 'reg':
            base = round(1 - f_goal, 2)
        else:
            base = round(1 - (f_goal - min(f_goals)) / min(f_goals), 2)
        for i, q in enumerate(quartiles, 1):
            csv_writer.writerow([base, algorithm, f"{metric} q{i}", round(q, 5)])
