import itertools
import logging
import random
from typing import Iterable, List, Optional, Tuple, Dict
from pipeline_execution import PipelineExecutor


class GridSearch:
    """
    Staged grid search with strict front-guard ordering + evaluation-based counting.

    HARD REQUIREMENTS (from you):
      ✅ Fix invalid_value and missing_value positions in EVERY candidate:
            invalid_value -> position 0
            missing_value -> position 1
            model -> last
      ✅ Every evaluation counts as 1 iteration (even if invalid/crash).
      ✅ Log EVERY evaluation with vector + pipeline order.
      ✅ Stage order:
            (1) SUBSET via deletions: start delete-1, then delete-2, then delete-3, ...
            (2) PARAM changes: allow k sequential param changes, start k=1 then increase sequentially after success
            (3) ORDER change: adjacent swaps (guards fixed)

    Notes:
      - max_iters is treated as an EVALUATION BUDGET.
      - Utility <= 0 or NaN/Inf is treated as bug => invalid => +inf, but still counted and logged.
      - Guards (invalid_value, missing_value, model) are NEVER deleted and guards are kept fixed at the front.
    """

    def __init__(self, dataset_name, pipeline_order, metric_type, pipeline_type):
        self.pipeline_order = list(pipeline_order)
        self.metric_type = metric_type
        self.dataset_name = dataset_name

        self.executor_pass = PipelineExecutor(
            pipeline_type=pipeline_type,
            dataset_name=self.dataset_name,
            metric_type=self.metric_type,
            pipeline_ord=self.pipeline_order,
            execution_type="fail",
        )

    # ==========================================================
    # Public API (keep same inputs)
    # ==========================================================
    def grid_search(
        self,
        f_goal: float,
        failing_order: List[str],
        failing_vec: List[int],
        new_components: Optional[Iterable[str]] = None,  # kept for compatibility (not used)
        max_iters: Optional[int] = None,                  # EVALUATION BUDGET
        rng: Optional[random.Random] = None,
    ) -> Tuple[int, float, List[str], List[int]]:
        """
        Returns:
          (evals_used, best_f, best_order, best_vec)

        evals_used counts ALL evaluations (valid/invalid/crash).
        """
        _rng = rng or random.SystemRandom()
        if max_iters is None:
            max_iters = 500

        logging.info("[GridSearch-StagedEval] Start | target=%.6f | eval_budget=%d", f_goal, max_iters)

        # Seed sanitize + guard positions
        cur_order, cur_vec = self._sanitize_and_impute_pipeline(failing_order, failing_vec)
        cur_order, cur_vec = self._force_front_guards(cur_order, cur_vec)

        evals = 0
        cur_f, evals = self._safe_eval_counted(cur_order, cur_vec, evals, tag="seed")
        best_order, best_vec, best_f = list(cur_order), list(cur_vec), float(cur_f)

        # ------------------------
        # Stage 1: subset deletions (k = 1,2,3,...)
        # ------------------------
        cur_order, cur_vec, cur_f, best_order, best_vec, best_f, evals = self._stage_subset_kdelete(
            f_goal=f_goal,
            eval_budget=max_iters,
            cur_order=cur_order,
            cur_vec=cur_vec,
            cur_f=cur_f,
            best_order=best_order,
            best_vec=best_vec,
            best_f=best_f,
            evals=evals,
            rng=_rng
        )
        if evals >= max_iters:
            return evals, best_f, best_order, best_vec
        if self._is_valid_utility(cur_f) and cur_f <= f_goal:
            return evals, cur_f, cur_order, cur_vec

        # ------------------------
        # Stage 2: strategy (k sequential param changes)
        # ------------------------
        cur_order, cur_vec, cur_f, best_order, best_vec, best_f, evals = self._stage_strategy_kseq(
            f_goal=f_goal,
            eval_budget=max_iters,
            cur_order=cur_order,
            cur_vec=cur_vec,
            cur_f=cur_f,
            best_order=best_order,
            best_vec=best_vec,
            best_f=best_f,
            evals=evals,
        )
        if evals >= max_iters:
            return evals, best_f, best_order, best_vec
        if self._is_valid_utility(cur_f) and cur_f <= f_goal:
            return evals, cur_f, cur_order, cur_vec

        # ------------------------
        # Stage 3: reorder (adjacent swap; guards fixed)
        # ------------------------
        cur_order, cur_vec, cur_f, best_order, best_vec, best_f, evals = self._stage_reorder(
            f_goal=f_goal,
            eval_budget=max_iters,
            cur_order=cur_order,
            cur_vec=cur_vec,
            cur_f=cur_f,
            best_order=best_order,
            best_vec=best_vec,
            best_f=best_f,
            evals=evals,
        )

        logging.info("[GridSearch-StagedEval] Finished | evals=%d | best=%.6f", evals, best_f)
        return evals, best_f, best_order, best_vec

    # ==========================================================
    # Stage 1: subset deletions (k = 1,2,3,...)
    # ==========================================================
    def _stage_subset_kdelete(
        self,
        f_goal: float,
        eval_budget: int,
        cur_order: List[str],
        cur_vec: List[int],
        cur_f: float,
        best_order: List[str],
        best_vec: List[int],
        best_f: float,
        evals: int,
        rng: random.Random,
    ):
        """
        More thorough deletion stage:
          - Try delete k components, starting k=1 then k=2 then ...
          - For each k, evaluate a set of candidates (combinations or sampled combinations)
          - If best candidate improves, APPLY it and restart from k=1 (because pipeline changed)
          - Stop when no k yields improvement or budget exhausted
        """
        logging.info("[Stage1] subset deletions (k=1,2,3,...)")

        protected = {"invalid_value", "missing_value", "model"}

        # Control explosion: cap how many combinations to evaluate per k
        # (still "thorough" but won't blow up)
        MAX_CANDS_PER_K = 200  # adjust if you want more/less thorough

        while evals < eval_budget:
            if self._is_valid_utility(cur_f) and cur_f <= f_goal:
                break

            deletable_indices = [i for i, c in enumerate(cur_order) if c not in protected]
            n_del = len(deletable_indices)
            if n_del == 0:
                break

            improved_any = False

            # k = 1..n_del
            for k in range(1, n_del + 1):
                if evals >= eval_budget:
                    break

                # Generate candidate deletion index-sets for this k
                comb_count = self._nCk(n_del, k)

                # Enumerate if small, else sample
                if comb_count <= MAX_CANDS_PER_K:
                    index_sets = itertools.combinations(deletable_indices, k)
                else:
                    # sample distinct k-sets
                    sampled = set()
                    tries = 0
                    target = min(MAX_CANDS_PER_K, comb_count, eval_budget - evals)
                    index_sets = []
                    while len(index_sets) < target and tries < target * 20:
                        tries += 1
                        pick = tuple(sorted(rng.sample(deletable_indices, k)))
                        if pick in sampled:
                            continue
                        sampled.add(pick)
                        index_sets.append(pick)

                cand_best = None  # (nf, norder, nvec, deleted_components)

                for del_set in index_sets:
                    if evals >= eval_budget:
                        break

                    del_set = sorted(del_set)
                    deleted_components = [cur_order[i] for i in del_set]

                    # delete from end to start so indices remain valid
                    norder = list(cur_order)
                    nvec = list(cur_vec)
                    for idx in sorted(del_set, reverse=True):
                        norder.pop(idx)
                        nvec.pop(idx)

                    norder, nvec = self._sanitize_and_impute_pipeline(norder, nvec)
                    norder, nvec = self._force_front_guards(norder, nvec)

                    nf, evals = self._safe_eval_counted(
                        norder, nvec, evals,
                        tag=f"stage1_delete_k={k}:{'+'.join(deleted_components)}"
                    )

                    if nf < best_f:
                        best_f, best_order, best_vec = float(nf), list(norder), list(nvec)

                    if cand_best is None or nf < cand_best[0]:
                        cand_best = (nf, norder, nvec, deleted_components)

                # If no candidate evaluated for this k, continue
                if cand_best is None:
                    continue

                nf, norder, nvec, deleted_components = cand_best

                # Apply if improves current
                if nf < cur_f:
                    cur_f, cur_order, cur_vec = nf, norder, nvec
                    improved_any = True
                    logging.debug(
                        "[Stage1] APPLY delete_k=%d (%s) | evals=%d utility=%.6f",
                        k, ",".join(deleted_components), evals, cur_f
                    )
                    # restart from k=1 after an apply
                    break

            if not improved_any:
                break

        return cur_order, cur_vec, cur_f, best_order, best_vec, best_f, evals

    def _nCk(self, n: int, k: int) -> int:
        try:
            if k < 0 or k > n:
                return 0
            k = min(k, n - k)
            num = 1
            den = 1
            for i in range(1, k + 1):
                num *= (n - k + i)
                den *= i
            return num // den
        except Exception:
            return 10**18  # "very large"

    # ==========================================================
    # Stage 2: strategy (k sequential param changes)
    # ==========================================================
    def _stage_strategy_kseq(
        self,
        f_goal: float,
        eval_budget: int,
        cur_order: List[str],
        cur_vec: List[int],
        cur_f: float,
        best_order: List[str],
        best_vec: List[int],
        best_f: float,
        evals: int,
    ):
        """
        k-sequential strategy stage:
          - Start k_changes = 1
          - If we APPLY an improvement, k_changes += 1
          - Candidate is built by applying k single-param changes sequentially (deterministic)
          - Final pipeline is evaluated once (counts 1 eval)
        """
        logging.info("[Stage2] strategy (k-sequential param changes)")

        k_changes = 1

        while evals < eval_budget:
            if self._is_valid_utility(cur_f) and cur_f <= f_goal:
                break

            seq_order, seq_vec, seq_desc = self._build_kseq_candidate(
                base_order=cur_order,
                base_vec=cur_vec,
                k_changes=k_changes,
            )
            if seq_order is None:
                break

            nf, evals = self._safe_eval_counted(seq_order, seq_vec, evals, tag=f"stage2_kseq:k={k_changes}:{seq_desc}")

            if nf < best_f:
                best_f, best_order, best_vec = float(nf), list(seq_order), list(seq_vec)

            if nf < cur_f:
                cur_order, cur_vec, cur_f = seq_order, seq_vec, nf
                logging.debug("[Stage2] APPLY kseq(k=%d) | evals=%d utility=%.6f | %s", k_changes, evals, cur_f, seq_desc)
                k_changes += 1
            else:
                break

        return cur_order, cur_vec, cur_f, best_order, best_vec, best_f, evals

    def _build_kseq_candidate(
        self,
        base_order: List[str],
        base_vec: List[int],
        k_changes: int,
    ) -> Tuple[Optional[List[str]], Optional[List[int]], str]:
        """
        Deterministically apply k single-param changes sequentially.
        To keep it stable + not blow evaluation budget, we don't evaluate intermediate steps.
        """
        order, vec = self._sanitize_and_impute_pipeline(base_order, base_vec)
        order, vec = self._force_front_guards(order, vec)

        changed_positions = set()
        desc_parts = []

        for _ in range(k_changes):
            # choose next mutable position (skip model; prefer non-guards)
            candidates = [i for i, c in enumerate(order) if c != "model" and i not in changed_positions]
            if not candidates:
                break

            non_guard = [i for i in candidates if i not in (0, 1)]
            if non_guard:
                candidates = non_guard

            i = candidates[0]  # deterministic
            comp = order[i]

            dom = self._strategies_for(comp)
            if len(dom) <= 1:
                changed_positions.add(i)
                continue

            cur_s = int(vec[i])
            alts = [int(s) for s in dom if int(s) != cur_s]
            if not alts:
                changed_positions.add(i)
                continue

            new_s = alts[0]  # deterministic
            vec[i] = int(new_s)
            changed_positions.add(i)
            desc_parts.append(f"{comp}:{cur_s}->{new_s}")

        if not desc_parts:
            return None, None, ""

        order, vec = self._sanitize_and_impute_pipeline(order, vec)
        order, vec = self._force_front_guards(order, vec)
        return order, vec, "|".join(desc_parts)

    # ==========================================================
    # Stage 3: reorder (adjacent swap greedy; guards fixed)
    # ==========================================================
    def _stage_reorder(
        self,
        f_goal: float,
        eval_budget: int,
        cur_order: List[str],
        cur_vec: List[int],
        cur_f: float,
        best_order: List[str],
        best_vec: List[int],
        best_f: float,
        evals: int,
    ):
        logging.info("[Stage3] reorder (adjacent swap greedy)")

        while evals < eval_budget:
            if self._is_valid_utility(cur_f) and cur_f <= f_goal:
                break

            # guards fixed: idx0 invalid_value, idx1 missing_value, last model
            if len(cur_order) <= 4:
                break

            cand_best = None  # (nf, norder, nvec, i, left, right)

            last_non_model_idx = len(cur_order) - 2
            for i in range(2, last_non_model_idx):
                if evals >= eval_budget:
                    break
                if i + 1 > last_non_model_idx:
                    continue

                left = cur_order[i]
                right = cur_order[i + 1]

                norder = list(cur_order)
                nvec = list(cur_vec)
                norder[i], norder[i + 1] = norder[i + 1], norder[i]
                nvec[i], nvec[i + 1] = nvec[i + 1], nvec[i]

                norder, nvec = self._sanitize_and_impute_pipeline(norder, nvec)
                norder, nvec = self._force_front_guards(norder, nvec)

                nf, evals = self._safe_eval_counted(
                    norder, nvec, evals, tag=f"stage3_swap:i={i}:{left}<->{right}"
                )

                if nf < best_f:
                    best_f, best_order, best_vec = float(nf), list(norder), list(nvec)

                if cand_best is None or nf < cand_best[0]:
                    cand_best = (nf, norder, nvec, i, left, right)

            if cand_best is None:
                break

            nf, norder, nvec, i, left, right = cand_best
            if nf < cur_f:
                cur_f, cur_order, cur_vec = nf, norder, nvec
                logging.debug("[Stage3] APPLY swap(i=%d %s<->%s) | evals=%d utility=%.6f", i, left, right, evals, cur_f)
            else:
                break

        return cur_order, cur_vec, cur_f, best_order, best_vec, best_f, evals

    # ==========================================================
    # EVAL: always counts as 1; logs every evaluation
    # ==========================================================
    def _safe_eval_counted(
        self,
        order: List[str],
        vec: List[int],
        evals: int,
        tag: str = "",
    ) -> Tuple[float, int]:
        """
        Always increments evals by 1 (valid/invalid/crash all count).
        Logs order+vec on every evaluation.
        """
        evals += 1

        order, vec = self._sanitize_and_impute_pipeline(order, vec)
        order, vec = self._force_front_guards(order, vec)

        try:
            val = float(self.executor_pass.current_par_lookup(order, vec))
            valid = self._is_valid_utility(val)

            if not valid:
                logging.info(
                    "[EVAL %d] tag=%s | utility=%.6f | VALID=0 | order=%s | vec=%s",
                    evals, tag, val, order, vec
                )
                return float("inf"), evals

            logging.info(
                "[EVAL %d] tag=%s | utility=%.6f | VALID=1 | order=%s | vec=%s",
                evals, tag, val, order, vec
            )
            return val, evals

        except Exception as e:
            logging.info(
                "[EVAL %d] tag=%s | utility=INF | VALID=0 | CRASH=%s | order=%s | vec=%s",
                evals, tag, repr(e), order, vec
            )
            return float("inf"), evals

    def _is_valid_utility(self, u: float) -> bool:
        """Treat <=0 and NaN/Inf as invalid (bug)."""
        try:
            if u is None:
                return False
            if u != u:
                return False
            if u == float("inf") or u == float("-inf"):
                return False
            if u <= 0:
                return False
            return True
        except Exception:
            return False

    # ==========================================================
    # Structure + domains + imputing
    # ==========================================================
    def _strategies_for(self, component: str) -> List[int]:
        counts: Optional[Dict[str, int]] = getattr(self.executor_pass, "strategy_counts", None)
        if not counts or component not in counts:
            return [1]
        return list(range(1, int(counts[component]) + 1))

    def _is_missing(self, x) -> bool:
        if x is None:
            return True
        try:
            if isinstance(x, float) and x != x:
                return True
        except Exception:
            pass
        if isinstance(x, str) and x.strip() == "":
            return True
        return False

    def _coerce_int_or_none(self, x) -> Optional[int]:
        if self._is_missing(x):
            return None
        try:
            return int(float(x))
        except Exception:
            return None

    def _sanitize_and_impute_pipeline(self, order: List[str], vec: List[int]) -> Tuple[List[str], List[int]]:
        """
        - remove duplicates
        - enforce model last
        - align vec length
        - impute missing strategy indices to default
        - clamp invalid indices to default
        """
        if order is None:
            order = []
        order = list(order)

        # dedup
        seen = set()
        dedup = []
        for c in order:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        order = dedup

        # enforce model last
        if "model" in order:
            order = [c for c in order if c != "model"] + ["model"]
        else:
            order = order + ["model"]

        vec = list(vec or [])

        if len(vec) < len(order):
            vec = vec + [None] * (len(order) - len(vec))
        elif len(vec) > len(order):
            vec = vec[: len(order)]

        fixed_vec: List[int] = []
        for i, c in enumerate(order):
            dom = self._strategies_for(c)
            default = int(dom[0]) if dom else 1

            xi = self._coerce_int_or_none(vec[i])
            if xi is None:
                fixed_vec.append(default)
            elif dom and int(xi) not in dom:
                fixed_vec.append(default)
            else:
                fixed_vec.append(int(xi))

        return order, fixed_vec

    def _pick_real_strategy(self, component: str) -> int:
        """
        Prefer first strategy > 1 to avoid NONE/no-op for guards.
        """
        dom = self._strategies_for(component)
        for s in dom:
            if int(s) > 1:
                return int(s)
        return int(dom[0]) if dom else 1

    def _force_front_guards(self, order: List[str], vec: List[int]) -> Tuple[List[str], List[int]]:
        """
        HARD POSITION FIX:
          invalid_value MUST be position 0
          missing_value MUST be position 1
          model MUST be last
        Also force both guard strategies to be REAL (prefer >1).
        """
        order = list(order)
        vec = list(vec)

        # enforce model last
        if "model" in order:
            mi = order.index("model")
            if mi != len(order) - 1:
                mv = vec[mi]
                order.pop(mi); vec.pop(mi)
                order.append("model"); vec.append(int(mv))
        else:
            order.append("model"); vec.append(1)

        def pop_comp(comp: str):
            if comp in order:
                i = order.index(comp)
                sv = vec[i]
                order.pop(i); vec.pop(i)
                return sv
            return None

        inv_sv = pop_comp("invalid_value")
        mis_sv = pop_comp("missing_value")

        inv_real = self._pick_real_strategy("invalid_value")
        mis_real = self._pick_real_strategy("missing_value")

        try:
            if inv_sv is None or int(inv_sv) <= 1:
                inv_sv = inv_real
        except Exception:
            inv_sv = inv_real

        try:
            if mis_sv is None or int(mis_sv) <= 1:
                mis_sv = mis_real
        except Exception:
            mis_sv = mis_real

        order.insert(0, "invalid_value")
        vec.insert(0, int(inv_sv))

        order.insert(1, "missing_value")
        vec.insert(1, int(mis_sv))

        # dedup again (keep first)
        seen = set()
        new_order, new_vec = [], []
        for c, v in zip(order, vec):
            if c in seen:
                continue
            seen.add(c)
            new_order.append(c)
            new_vec.append(int(v))

        # enforce model last again
        if "model" in new_order:
            mi = new_order.index("model")
            if mi != len(new_order) - 1:
                mv = new_vec[mi]
                new_order.pop(mi); new_vec.pop(mi)
                new_order.append("model"); new_vec.append(int(mv))

        return new_order, new_vec
