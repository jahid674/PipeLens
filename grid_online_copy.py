import random
import logging
from typing import Iterable, List, Optional, Tuple, Dict
from pipeline_execution import PipelineExecutor


class GridSearch:
    """
    One-change-per-iteration search starting from a failing pipeline configuration.

    IMPORTANT UPDATE (per your request):
      ✅ Every evaluation counts as 1 — even if the candidate is invalid (crash, NaN, <=0, etc.)

    What we count:
      - seed evaluation counts as 1
      - every neighbor evaluated counts as 1
      - we stop when (best_valid_utility <= f_goal) OR eval_budget exhausted

    Guarantees:
      1) 'model' included and last.
      2) Strategy vector missing values are imputed.
      3) Every candidate evaluation forces missing_value to run FIRST
         AND uses a REAL imputation strategy (prefers >1 to avoid NONE).
      4) If candidate crashes or returns invalid utility (<=0/NaN/Inf), it is treated as +inf
         BUT still counts toward eval_count.
    """

    def __init__(self, dataset_name, pipeline_order, metric_type, pipeline_type):
        self.pipeline_order = list(pipeline_order)
        self.metric_type = metric_type
        self.dataset_name = dataset_name

        self.gs_idistr: List[int] = []
        self.gs_fdistr: List[float] = []

        self.executor_pass = PipelineExecutor(
            pipeline_type=pipeline_type,
            dataset_name=self.dataset_name,
            metric_type=self.metric_type,
            pipeline_ord=self.pipeline_order,
            execution_type='fail'
        )

    # ---------------------------
    # Public API
    # ---------------------------
    def grid_search(
        self,
        f_goal: float,
        failing_order: List[str],
        failing_vec: List[int],
        new_components: Optional[Iterable[str]] = None,
        max_evals: Optional[int] = None,
        neighbors_per_iter: int = 25,
        rng: Optional[random.Random] = None,
        allow_no_improve_steps: bool = False,
    ) -> Tuple[int, float, List[str], List[int]]:
        """
        Returns:
          (evals_used, best_valid_f, best_order, best_vec)

        NOTE:
          - evals_used counts ALL evaluations (including invalid)
          - best_valid_f is +inf if no valid candidate was found
        """
        logging.info("[GridSearch] Start. Target utility = %.4f", f_goal)

        _rng = rng or random.SystemRandom()
        if max_evals is None:
            max_evals = 500  # safe default

        uniq_new = list(dict.fromkeys(list(new_components or [])))
        uniq_new = [c for c in uniq_new if c != "model"]

        # ---- sanitize seed ----
        cur_order, cur_vec = self._sanitize_and_impute_pipeline(failing_order, failing_vec)
        cur_order, cur_vec = self._force_impute_first(cur_order, cur_vec)

        evals = 0

        # Evaluate seed (counts as 1 no matter what)
        cur_f = self._safe_eval_counted(cur_order, cur_vec)
        evals += 1

        best_order, best_vec = list(cur_order), list(cur_vec)
        best_f = float(cur_f)  # may be inf if invalid

        logging.info("[GridSearch] Seed eval=%d utility=%.6f | order=%s | vec=%s",
                     evals, cur_f, cur_order, cur_vec)

        self.gs_idistr.clear()
        self.gs_fdistr.clear()

        # We do hill-climb steps; each step evaluates up to neighbors_per_iter candidates
        # but STOP based on eval budget.
        while evals < max_evals:
            # If current is valid and already meets goal, stop.
            if self._is_valid_utility(cur_f) and cur_f <= f_goal:
                logging.info("[GridSearch] 🎯 Goal achieved | evals=%d | utility=%.6f", evals, cur_f)
                self.gs_idistr.append(evals)
                self.gs_fdistr.append(cur_f)
                return evals, cur_f, cur_order, cur_vec

            cand_best = None  # (f, order, vec, move_type)
            seen_local = set()

            # Evaluate up to neighbors_per_iter neighbors, respecting eval budget
            for _ in range(max(1, neighbors_per_iter)):
                if evals >= max_evals:
                    break

                n_order, n_vec, move_type = self._sample_one_change_neighbor(
                    cur_order, cur_vec, uniq_new, _rng
                )
                if n_order is None:
                    # even generating failed isn't an evaluation
                    continue

                n_order, n_vec = self._sanitize_and_impute_pipeline(n_order, n_vec)
                n_order, n_vec = self._force_impute_first(n_order, n_vec)

                key = (tuple(n_order), tuple(n_vec))
                if key in seen_local:
                    continue
                seen_local.add(key)

                # Evaluate candidate (COUNTS as 1 even if invalid)
                n_f = self._safe_eval_counted(n_order, n_vec)
                evals += 1

                if (cand_best is None) or (n_f < cand_best[0]):
                    cand_best = (n_f, n_order, n_vec, move_type)

                # Track global best valid
                if n_f < best_f:
                    best_f, best_order, best_vec = float(n_f), list(n_order), list(n_vec)

            # If we couldn't evaluate any neighbor, stop.
            if cand_best is None:
                logging.info("[GridSearch] No neighbors evaluated. Stop. evals=%d", evals)
                break

            n_f, n_order, n_vec, move_type = cand_best

            # Apply the best-improving neighbor (one change) only if it improves current
            if n_f < cur_f:
                cur_f, cur_order, cur_vec = n_f, n_order, n_vec
                logging.debug("[GridSearch] APPLY move=%s | evals=%d | utility=%.6f | order=%s | vec=%s",
                              move_type, evals, cur_f, cur_order, cur_vec)
            else:
                logging.debug("[GridSearch] NO-IMPROVE | evals=%d | best_neighbor=%.6f >= current=%.6f",
                              evals, n_f, cur_f)
                if not allow_no_improve_steps:
                    break

        logging.info("[GridSearch] Finished. evals=%d | best=%.6f | order=%s | vec=%s",
                     evals, best_f, best_order, best_vec)
        return evals, best_f, best_order, best_vec

    # ---------------------------
    # Utility validity + safe eval (counted externally)
    # ---------------------------
    def _is_valid_utility(self, u: float) -> bool:
        """Reject NaN/Inf and utility <= 0 (treat as bug)."""
        try:
            if u is None:
                return False
            if u != u:  # NaN
                return False
            if u == float("inf") or u == float("-inf"):
                return False
            if u <= 0:
                return False
            return True
        except Exception:
            return False

    def _safe_eval_counted(self, order: List[str], vec: List[int]) -> float:
        """
        Evaluate pipeline safely.
        - If crashes -> +inf
        - If returns invalid utility (<=0/NaN/Inf) -> +inf
        NOTE: counting is done by caller (so you can count every eval exactly once).
        """
        try:
            val = float(self.executor_pass.current_par_lookup(order, vec))
            if not self._is_valid_utility(val):
                logging.debug("[GridSearch] invalid utility=%.6f -> inf | order=%s vec=%s",
                              val, order, vec)
                return float("inf")
            return val
        except Exception as e:
            logging.debug("[GridSearch] crash -> inf | err=%s | order=%s vec=%s",
                          repr(e), order, vec)
            return float("inf")

    # ---------------------------
    # Strategy domains + vector imputing
    # ---------------------------
    def _strategies_for(self, component: str) -> List[int]:
        counts: Optional[Dict[str, int]] = getattr(self.executor_pass, "strategy_counts", None)
        if not counts or component not in counts:
            return [1]
        return list(range(1, int(counts[component]) + 1))

    def _is_missing(self, x) -> bool:
        if x is None:
            return True
        try:
            if isinstance(x, float) and x != x:  # NaN
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
        if order is None:
            order = []
        order = list(order)

        # de-duplicate
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

        # align lengths
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
            elif dom and xi not in dom:
                fixed_vec.append(default)
            else:
                fixed_vec.append(int(xi))

        return order, fixed_vec

    # ---------------------------
    # Force imputation first (and REAL strategy)
    # ---------------------------
    def _force_impute_first(self, order: List[str], vec: List[int]) -> Tuple[List[str], List[int]]:
        """
        Ensure 'missing_value' is at position 0 AND uses a REAL imputation strategy.
        We prefer the first strategy > 1 to avoid NONE=1.
        """
        order = list(order)
        vec = list(vec)

        imputer_name = "missing_value"

        # enforce model last
        if "model" in order:
            m_idx = order.index("model")
            if m_idx != len(order) - 1:
                m_val = vec[m_idx]
                order.pop(m_idx); vec.pop(m_idx)
                order.append("model"); vec.append(m_val)
        else:
            order.append("model")
            vec.append(1)

        # choose a real imputation strategy
        dom = self._strategies_for(imputer_name)
        preferred = None
        for s in dom:
            if int(s) > 1:
                preferred = int(s)
                break
        if preferred is None:
            preferred = int(dom[0]) if dom else 1

        # move/insert missing_value to front
        if imputer_name in order:
            idx = order.index(imputer_name)
            imp_val = vec[idx]
            if idx != 0:
                order.pop(idx); vec.pop(idx)
                order.insert(0, imputer_name)
                vec.insert(0, imp_val)

            try:
                if vec[0] is None or int(vec[0]) <= 1:
                    vec[0] = preferred
            except Exception:
                vec[0] = preferred
        else:
            order.insert(0, imputer_name)
            vec.insert(0, preferred)

        # de-dup (keep first)
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
            m_idx = new_order.index("model")
            if m_idx != len(new_order) - 1:
                m_val = new_vec[m_idx]
                new_order.pop(m_idx); new_vec.pop(m_idx)
                new_order.append("model"); new_vec.append(m_val)

        return new_order, new_vec

    # ---------------------------
    # One-change neighbor sampler
    # ---------------------------
    def _sample_one_change_neighbor(
        self,
        cur_order: List[str],
        cur_vec: List[int],
        new_pool: List[str],
        rng: random.Random,
    ) -> Tuple[Optional[List[str]], Optional[List[int]], Optional[str]]:
        order, vec = self._sanitize_and_impute_pipeline(cur_order, cur_vec)

        non_model = [c for c in order if c != "model"]
        move_types = ["param_change"]

        if len(non_model) >= 1:
            move_types.append("delete_one")
        if len(non_model) >= 2:
            move_types.append("swap_adjacent")

        insertables = [c for c in new_pool if c not in order and c != "model"]
        if insertables:
            move_types.append("insert_one")

        if not move_types:
            return None, None, None

        move = rng.choice(move_types)

        if move == "param_change":
            idxs = list(range(len(order)))
            rng.shuffle(idxs)
            for i in idxs:
                c = order[i]
                dom = self._strategies_for(c)
                if len(dom) <= 1:
                    continue
                choices = [s for s in dom if int(s) != int(vec[i])]
                if not choices:
                    continue
                nvec = list(vec)
                nvec[i] = int(rng.choice(choices))
                return list(order), nvec, "param_change"
            return None, None, None

        if move == "delete_one":
            del_candidates = [i for i, c in enumerate(order) if c != "model"]
            if not del_candidates:
                return None, None, None
            i = rng.choice(del_candidates)
            norder = order[:i] + order[i + 1:]
            nvec = vec[:i] + vec[i + 1:]
            return norder, nvec, "delete_one"

        if move == "insert_one":
            if not insertables:
                return None, None, None
            comp = rng.choice(insertables)

            model_idx = order.index("model")
            pos = rng.randrange(0, model_idx + 1)

            norder = order[:pos] + [comp] + order[pos:]
            dom = self._strategies_for(comp)
            strat = int(rng.choice(dom)) if dom else 1
            nvec = vec[:pos] + [strat] + vec[pos:]
            return norder, nvec, "insert_one"

        if move == "swap_adjacent":
            nm_indices = [i for i, c in enumerate(order) if c != "model"]
            if len(nm_indices) < 2:
                return None, None, None
            j = rng.randrange(0, len(nm_indices) - 1)
            i1, i2 = nm_indices[j], nm_indices[j + 1]

            norder = list(order)
            nvec = list(vec)
            norder[i1], norder[i2] = norder[i2], norder[i1]
            nvec[i1], nvec[i2] = nvec[i2], nvec[i1]
            return norder, nvec, "swap_adjacent"

        return None, None, None
