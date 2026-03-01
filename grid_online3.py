import itertools
import logging
import random
from typing import Iterable, List, Optional, Tuple, Dict, Set
from pipeline_execution import PipelineExecutor
import numpy as np
np.random.seed(42)


class GridSearch:
    """
    Randomized staged grid search (crash-tolerant, eval-budgeted).

    UPDATE (per your request):
      ✅ Every evaluation MUST start from the ORIGINAL failing pipeline (seed),
         i.e., we DO NOT walk/accept into a new current pipeline.
         Each candidate is generated as: candidate = seed + one random intervention.

    Everything else stays the same:
      - guard positions fixed on every candidate
      - every evaluation counts
      - log every eval
      - invalid/crash/<=0 utility -> treated as +inf
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
    # Public API
    # ==========================================================
    def grid_search(
        self,
        f_goal: float,
        failing_order: List[str],
        failing_vec: List[int],
        new_components: Optional[Iterable[str]] = None,  # kept for compatibility (not used here)
        max_iters: Optional[int] = None,                  # EVALUATION BUDGET
        rng: Optional[random.Random] = None,
        # knobs:
        p_delete: float = 0.40,
        p_param: float = 0.40,
        p_swap: float = 0.20,
        max_delete_k: int = 4,
        max_param_k: int = 4,
        dedup_seen: bool = True,
    ) -> Tuple[int, float, List[str], List[int]]:

        _rng = rng or random.SystemRandom()
        if max_iters is None:
            max_iters = 1000

        # normalize probabilities
        psum = p_delete + p_param + p_swap
        if psum <= 0:
            p_delete, p_param, p_swap = 0.4, 0.4, 0.2
            psum = 1.0
        p_delete /= psum
        p_param /= psum
        p_swap /= psum

        logging.info(
            "[GridSearch-Random] Start | target=%.6f | eval_budget=%d | p(del)=%.2f p(param)=%.2f p(swap)=%.2f",
            f_goal, max_iters, p_delete, p_param, p_swap
        )

        # -----------------------------
        # FIXED SEED (never changes)
        # -----------------------------
        seed_order, seed_vec = self._sanitize_and_impute_pipeline(failing_order, failing_vec)
        seed_order, seed_vec = self._force_front_guards(seed_order, seed_vec)

        evals = 0
        seed_f, evals = self._safe_eval_counted(seed_order, seed_vec, evals, tag="seed")

        best_order, best_vec, best_f = list(seed_order), list(seed_vec), float(seed_f)

        seen: Set[Tuple[Tuple[str, ...], Tuple[int, ...]]] = set()
        if dedup_seen:
            seen.add((tuple(seed_order), tuple(seed_vec)))

        # IMPORTANT CHANGE:
        #   We DO NOT update "current pipeline".
        #   Every candidate is created from (seed_order, seed_vec).
        while evals < max_iters:
            # early stop if we ever found a valid <= goal
            if self._is_valid_utility(best_f) and best_f <= f_goal:
                logging.info("[GridSearch-Random] 🎯 Achieved | evals=%d | best_utility=%.6f", evals, best_f)
                return evals, best_f, best_order, best_vec

            # pick random operator
            r = _rng.random()
            if r < p_delete:
                op = "delete"
            elif r < p_delete + p_param:
                op = "param"
            else:
                op = "swap"

            # generate ONE random neighbor from SEED (not from last accepted)
            if op == "delete":
                norder, nvec, desc = self._random_delete_k(seed_order, seed_vec, _rng, max_delete_k)
            elif op == "param":
                norder, nvec, desc = self._random_param_k(seed_order, seed_vec, _rng, max_param_k)
            else:
                norder, nvec, desc = self._random_swap(seed_order, seed_vec, _rng)

            if norder is None:
                continue

            norder, nvec = self._sanitize_and_impute_pipeline(norder, nvec)
            norder, nvec = self._force_front_guards(norder, nvec)

            if dedup_seen:
                key = (tuple(norder), tuple(nvec))
                if key in seen:
                    continue
                seen.add(key)

            nf, evals = self._safe_eval_counted(norder, nvec, evals, tag=f"rand_{op}:{desc}")

            if nf < best_f:
                best_f, best_order, best_vec = float(nf), list(norder), list(nvec)

        logging.info("[GridSearch-Random] Finished | evals=%d | best=%.6f", evals, best_f)
        return evals, best_f, best_order, best_vec

    # ==========================================================
    # Random neighbor generators (operate on provided base)
    # ==========================================================
    def _random_delete_k(
        self,
        order: List[str],
        vec: List[int],
        rng: random.Random,
        max_k: int,
    ) -> Tuple[Optional[List[str]], Optional[List[int]], str]:
        protected = {"invalid_value", "missing_value", "model"}
        deletable = [i for i, c in enumerate(order) if c not in protected]
        if not deletable:
            return None, None, ""

        k = rng.randint(1, min(max_k, len(deletable)))
        del_set = sorted(rng.sample(deletable, k))
        deleted = [order[i] for i in del_set]

        norder = list(order)
        nvec = list(vec)
        for idx in sorted(del_set, reverse=True):
            norder.pop(idx)
            nvec.pop(idx)

        return norder, nvec, f"delete_k={k}:{'+'.join(deleted)}"

    def _random_param_k(
        self,
        order: List[str],
        vec: List[int],
        rng: random.Random,
        max_k: int,
    ) -> Tuple[Optional[List[str]], Optional[List[int]], str]:
        norder = list(order)
        nvec = list(vec)

        positions = [i for i, c in enumerate(norder) if c != "model"]
        if not positions:
            return None, None, ""

        k = rng.randint(1, min(max_k, len(positions)))
        rng.shuffle(positions)
        positions = positions[:k]

        desc_parts = []
        changed_any = False

        for i in positions:
            comp = norder[i]
            dom = self._strategies_for(comp)
            if len(dom) <= 1:
                continue
            cur_s = int(nvec[i])
            choices = [int(s) for s in dom if int(s) != cur_s]
            if not choices:
                continue
            new_s = rng.choice(choices)
            nvec[i] = int(new_s)
            desc_parts.append(f"{comp}:{cur_s}->{new_s}")
            changed_any = True

        if not changed_any:
            return None, None, ""

        return norder, nvec, f"param_k={k}:" + "|".join(desc_parts)

    def _random_swap(
        self,
        order: List[str],
        vec: List[int],
        rng: random.Random,
    ) -> Tuple[Optional[List[str]], Optional[List[int]], str]:
        if len(order) <= 4:
            return None, None, ""

        last_non_model_idx = len(order) - 2
        if last_non_model_idx <= 2:
            return None, None, ""

        i = rng.randint(2, last_non_model_idx - 1)

        norder = list(order)
        nvec = list(vec)
        left, right = norder[i], norder[i + 1]

        norder[i], norder[i + 1] = norder[i + 1], norder[i]
        nvec[i], nvec[i + 1] = nvec[i + 1], nvec[i]

        return norder, nvec, f"swap_i={i}:{left}<->{right}"

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
        if order is None:
            order = []
        order = list(order)

        seen = set()
        dedup = []
        for c in order:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        order = dedup

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
        dom = self._strategies_for(component)
        for s in dom:
            if int(s) > 1:
                return int(s)
        return int(dom[0]) if dom else 1

    def _force_front_guards(self, order: List[str], vec: List[int]) -> Tuple[List[str], List[int]]:
        order = list(order)
        vec = list(vec)

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

        seen = set()
        new_order, new_vec = [], []
        for c, v in zip(order, vec):
            if c in seen:
                continue
            seen.add(c)
            new_order.append(c)
            new_vec.append(int(v))

        if "model" in new_order:
            mi = new_order.index("model")
            if mi != len(new_order) - 1:
                mv = new_vec[mi]
                new_order.pop(mi); new_vec.pop(mi)
                new_order.append("model"); new_vec.append(int(mv))

        return new_order, new_vec
