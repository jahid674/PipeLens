import logging
import pandas as pd
import numpy as np
import itertools
import operator
from pipeline_component.swapping_handler import SwapHandler
from pipeline_execution import PipelineExecutor
from rank_method_selector import GaussianTSSelector
#from score_lookup import ScoreLookup
np.random.seed(42)

class GlassBoxOptimizer:
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

        #self.historical_data_pd = pd.read_csv(self.filename_test)
        #self.historical_data = self.historical_data_pd.values.tolist()

        self.executor_pass = PipelineExecutor(
            pipeline_type=self.pipeline_type,
            dataset_name=self.dataset_name,
            metric_type=self.metric_type,
            pipeline_ord=self.pipeline_order,
            execution_type='fail'
        )

        self.ranked_interventions = self.executor_pass.evaluate_interventions_pred_and_similarity(
            [int(v) for v in cur_params_opt.values()],
            self.filename_train,
            new_components=list(new_components)
        )
        self.fixed_data = self.executor_pass.get_injected_data()

        self.pasing_hist_data = pd.read_csv(self.filename_train)
        #self.coefs_profile, self.profile_ranking, self.param_coeff, self.param_rank = self.executor_pass.rank_profile_parameter(self.filename_train)
        self.profiles = self.executor_pass.get_header(self.filename_train)
        #print(self.profiles)
        #self.score_lookup = ScoreLookup(pipeline_order, metric_type)
    
    def set_ranges(self):
        for strategy in self.pipeline_order:
            self.ranges[strategy] = list(np.unique(self.pasing_hist_data[strategy]))

    def optimize(self, init_params, f_goal, max_depth=None, top_n_actions=120):

        new_components = self.new_components

        self.rank_iter = 0
        self.optimisitic_iter = 0
        self.exhaustive_iter = 0
        self.rank_f = 0
        cur_params_opt = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params)}
        print('cur_params_opt',cur_params_opt)
        cur_param_check = init_params[:len(self.base_strategies)]

        opt_f = self.executor_pass.current_par_lookup(self.base_strategies, cur_param_check, fixed_data=self.fixed_data)
        logging.info(f'Evaluating {[int(v) for v in cur_params_opt.values()]} -- Initial Utility {opt_f} -- Target Utility {f_goal:.2f}')

        if self.pipeline_type == 'ml':
            self.set_ranges()

        seen = set()
        if opt_f < f_goal:
            self.rank_iter = 1
            self.rank_f = opt_f
            return

        #seen.add(tuple(cur_params_opt.items()))

        max_iter_size = len(self.pipeline_order) + len(set(new_components))
        if max_depth is None:
            max_depth = max_iter_size
        else:
            max_depth = min(max_depth, max_iter_size)

        found = False

        #logging.info(f"Ranked interventions: {self.ranked_interventions}")

        for k in range(1, max_depth + 1):
            logging.info(f"[OPTIMIZE] Trying combo size k={k}")

            if k == 1:
                ## single-intervention path
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

            logging.info(f"Total iterations so far: {self.rank_iter}")
            
            if found:
                break

        if not found:
            self.fail += 1


    '''def optimistic_search(self, seen, cur_params_opt, f_goal, opt_f):
        for profile_index in self.profile_ranking:
            logging.info(f'first loop profile = {self.profiles[profile_index]}')
            #need correction here
            #distinguish between profile and p
            profile_name = self.profiles[profile_index]
            coef_rank = self.param_rank[profile_name]
            logging.info(f'Current Profile name {profile_name} : corresponding ranking parameter: {coef_rank}')
            for val in coef_rank:
                cur_strategy = self.base_strategies[val]
                logging.info(f'Strategy selected  : {cur_strategy}')
                current_param_value=self.rank_individual_intervention(cur_strategy, profile_index, profile_name, val)
                logging.info(f'order of parameter  value: {current_param_value}')                
                logging.warn(f'current Iteration before parameter selection {self.rank_iter}')
                cur_params = cur_params_opt.copy()
                cur_params[cur_strategy] = current_param_value
                logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')

                if tuple(cur_params.items()) in seen:
                    continue
                seen.add(tuple(cur_params.items()))
                cur_f  = self.score_lookup.utility_look_up(self.historical_data_pd,list(cur_params.values()))
                #cur_f = self.executor_pass.current_par_lookup(cur_params)

                self.rank_iter += 1
                logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                logging.warn(f'Current iteration after parameter selection {self.rank_iter}')
                logging.info(f'updated utility after parameter selection {cur_f}')

                if cur_f <= f_goal:
                    self.rank_f = cur_f
                    self.pass_ += 1
                    logging.error("Target achieved")
                    return True, cur_f, cur_params
                elif cur_f < opt_f:
                    opt_f = cur_f
                    cur_params_opt = cur_params.copy()
        return False, opt_f, cur_params_opt'''
    
    def optimistic_search(self, seen, cur_params_opt, f_goal, opt_f):
        logging.info("[INFO] Running optimistic search with combined interventions...")
        #ranked_interventions = self.executor_pass.evaluate_interventions([int(v) for v in cur_params_opt.values()], self.filename_train, new_components=['outlier', 'whitespace', 'punctuation', 'stopword'])
        original_order = self.pipeline_order.copy()
        for component, strategy, similarity, uti, pos, _ in self.ranked_interventions:
            logging.info(f"[SEARCH] Trying intervention: {component} → {strategy}")

            if component in original_order:
                idx = original_order.index(component)
                cur_params = cur_params_opt.copy()
                cur_params[original_order[idx]] = strategy
                intervened_order = original_order.copy()
            else:
                if component in ["swapping"]:
                    pipeline = [m for m in original_order if m != "swapping"]
                    handler = SwapHandler(strategy=strategy,config={"verbose": True})
                    intervened_order, new_params= handler.apply_with_params(pipeline, [int(v) for v in cur_params.values()])
                    cur_params = {strategy: selection for strategy, selection in zip(intervened_order, new_params)}
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

            #if tuple(cur_params) in seen:
            #    continue
            #seen.add(tuple(cur_params))
            prev_order = self.pipeline_order

            cur_f = self.executor_pass.current_par_lookup(intervened_order, [int(v) for v in cur_params.values()], fixed_data=self.fixed_data)
            logging.info(f'current utility {cur_f:.2f}')
            self.rank_iter += 1
            self.optimisitic_iter += 1
            logging.info(f"[TRY] {component}={strategy},pipeline: {cur_params}, Utility={cur_f:.2f}, Best={opt_f:.2f}")

            if cur_f <= f_goal:
                logging.info("✅ Target achieved 🎯")
                self.rank_f = cur_f
                self.pass_ += 1
                logging.info(f"required optimistic iterations: {self.optimisitic_iter}")
                logging.info(f"passing pipeline: order={intervened_order}, vec={[int(v) for v in cur_params.values()]}")
                return True, cur_f, cur_params
            elif cur_f < opt_f:
                opt_f = cur_f
                print('cur_F',cur_f)
                cur_params_opt = cur_params.copy()
                original_order = intervened_order
            elif cur_f > opt_f:
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
        
        logging.info("[INFO] Running optimistic search with combined interventions...")
        original_order = self.pipeline_order[:]
        baseline_vec = [int(cur_params_opt[s]) for s in original_order]


        actions = []
        base_set = set(original_order)
        for comp, strat, sim, util, pos, _ in self.ranked_interventions:
            if comp in base_set:
                actions.append(("change", comp, int(strat), None, float(sim)))
            elif comp != "swappping":
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
                # Insert its param at the same index
                left = list(params.items())[:ins_at]
                right = list(params.items())[ins_at:]
                params = dict(left + [(comp, int(strat))] + right)
                shift += 1

            # Apply changes to whatever existing components are still in the order
            for comp, strat in changes:
                if comp in order:
                    params[comp] = int(strat)

            vec = [int(params[s]) for s in order]
            return order, vec, params

        #Try all size-k combinations, guided by similarity order
        best_params_dict = cur_params_opt
        best_val = opt_f
        found = False

        for combo in itertools.combinations(actions, iter_size):
            if not _conflict_free(combo):
                continue

            eval_order, eval_vec, eval_params_dict = _apply_combo_concurrently(original_order, cur_params_opt, combo)
            print('eval_order', eval_order)
            print('eval_vec', eval_vec)
            print('original_order', original_order) 
            key = tuple((name, eval_params_dict[name]) for name in eval_order)
            if key in seen:
                continue
            seen.add(key)
            try:
                cur_f = self.executor_pass.current_par_lookup(eval_order, eval_vec, fixed_data=self.fixed_data)
            except Exception as e:
                logging.warning(f"[SIM-GUIDED][k={iter_size}] Skipping combo due to eval error: {e}")
                continue

            self.rank_iter += 1
            self.exhaustive_iter += 1
            logging.info(f"[SIM-GUIDED][k={iter_size}] combo={combo}, utility={cur_f:.2f}")

            if early_stop and (cur_f <= f_goal):
                logging.info("✅ Target achieved 🎯")
                logging.info(f"passing pipeline: order={eval_order}, vec={eval_vec}")
                self.rank_f = cur_f
                self.pass_ += 1
                logging.info(f"required exhaustive iterations: {self.exhaustive_iter}")
                return True, cur_f, eval_params_dict

            if cur_f < best_val:
                #best_val = cur_f
                print('cur_F',cur_f)
                #cur_params_opt = eval_params_dict.copy()
                #original_order = eval_order

        return found, best_val, best_params_dict

    
    def rank_individual_intervention(self, cur_strategy, idx, prof_name, val):
        if self.coefs_profile[idx] > 0:
            current_param_value = self.ranges[cur_strategy][0] if self.param_coeff[prof_name][val] > 0 else self.ranges[cur_strategy][-1]
        else:
            current_param_value = self.ranges[cur_strategy][0] if self.param_coeff[prof_name][val] < 0 else self.ranges[cur_strategy][-1]
        
        return current_param_value
    
    def similarity_based_ranking(self, new_comp, cur_par):
        combined_rank=self.executor_pass.evaluate_combined_intervention(new_comp, cur_par)
        return combined_rank


    
    def rank_intervention_combination(self,profile_name):
        score_map = {}
        lst = []
        for row in self.historical_data:
            score_val = sum([row[i] * self.param_coeff[profile_name][i] for i in range(len(self.base_strategies))])
            lst.append(score_val, row)
            score_map[score_val] = row

        sorted_param = sorted(score_map.items(), key=operator.itemgetter(0))
        sorted_params_lst = sorted(lst, key=lambda x: x[0])

        return sorted_param,sorted_params_lst

    def write_quartiles(self, csv_writer, algorithm, metric, quartiles, f_goal, f_goals):
        if self.model_type != 'reg':
            base = round(1 - f_goal, 2)
        else:
            base = round(1 - (f_goal - min(f_goals)) / min(f_goals), 2)
        for i, q in enumerate(quartiles, 1):
            csv_writer.writerow([base, algorithm, f"{metric} q{i}", round(q, 5)])



