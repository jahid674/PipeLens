import itertools
import random
import logging
from typing import Iterable, List, Optional, Tuple
from pipeline_execution import PipelineExecutor


class GridSearch:
    """
    Two-phase grid search with caching.

    Phase A (Base only):
      • Build the full candidate set using ONLY the base pipeline components.
      • Power set over base components (must include 'model'), permute non-model, model last.
      • For each order, enumerate Cartesian product of strategy indices.
      • Shuffle and evaluate until threshold met.

    Phase B (Union base ∪ new):
      • If Phase A didn't meet f_goal, explore the full candidate set for the union.
      • Same enumeration rules (power set, permutations, strategies, model last).
      • Shuffle and evaluate, skipping duplicates already seen in Phase A.

    Notes:
      • 'model' MUST be present and MUST be last in any pipeline.
      • No component may appear more than once in a pipeline.
      • Lower utility is better; early-stop when utility ≤ f_goal.
      • Caches both Phase A and Phase B candidate lists for reuse across runs.
      • Public API kept consistent with your original GridSearch usage.
    """

    # -------- constructor: unchanged inputs --------
    def __init__(self, dataset_name, historical_data, pipeline_order, metric_type, pipeline_type):
        self.historical_data_pd = historical_data
        self.historical_data = getattr(historical_data, "values", [[]]).tolist() if hasattr(historical_data, "values") else []
        self.pipeline_order = list(pipeline_order)
        self.metric_type = metric_type
        self.dataset_name = dataset_name

        self.gs_idistr: List[int] = []
        self.gs_fdistr: List[float] = []

        self.executor_pass = PipelineExecutor(
            pipeline_type=pipeline_type,
            dataset_name=self.dataset_name,
            metric_type=self.metric_type,
            pipeline_ord=self.pipeline_order
        )

        # Caches (keys + candidate lists)
        self._base_key: Optional[Tuple[str, ...]] = None
        self._union_key: Optional[Tuple[str, ...]] = None
        self._candidates_base: Optional[List[Tuple[List[str], List[int]]]] = None
        self._candidates_union: Optional[List[Tuple[List[str], List[int]]]] = None

    # -------- public API: same signature --------
    def grid_search(
        self,
        f_goal: float,
        new_components: Optional[Iterable[str]] = None,
        max_configs: Optional[int] = 1000,
        randomize: bool = True,
        reuse_cached: bool = True,
        rng: Optional[random.Random] = None
    ) -> Tuple[int, float]:
        """
        Returns (gs_iter, f_value)
          - gs_iter: number of evaluations performed across both phases
          - f_value: achieved utility (≤ f_goal if early-stopped; else best observed)
        """
        logging.info("[GS2] Start. Target utility = %.6f", f_goal)

        uniq_new = tuple(dict.fromkeys(new_components or []))
        base_key = tuple(dict.fromkeys(self.pipeline_order))
        union_key = tuple(dict.fromkeys(self.pipeline_order + list(uniq_new)))

        # --- PHASE A: BASE ONLY (build once, then reuse) ---
        if (not reuse_cached) or self._candidates_base is None or self._base_key != base_key:
            logging.info("[GS2] Building candidates from %s", base_key)
            self._candidates_base = self._enumerate_full_space_for_components(list(base_key))
            self._base_key = base_key
            logging.info("[GS2] candidates: %d", len(self._candidates_base))
        else:
            logging.info("[GS2] Reusing cache: %d candidates", len(self._candidates_base))

        # --- PHASE B: UNION (build once, then reuse) ---
        if (not reuse_cached) or self._candidates_union is None or self._union_key != union_key:
            logging.info("[GS2] Building andidates from %s", union_key)
            self._candidates_union = self._enumerate_full_space_for_components(list(union_key))
            self._union_key = union_key
            logging.info("[GS2] candidates: %d", len(self._candidates_union))
        else:
            logging.info("[GS2] Reusing cache: %d candidates", len(self._candidates_union))

        # Shuffle helpers
        rng_obj = rng or random.SystemRandom()
        if randomize and self._candidates_base:
            rng_obj.shuffle(self._candidates_base)
        if randomize and self._candidates_union:
            rng_obj.shuffle(self._candidates_union)

        self.gs_idistr.clear()
        self.gs_fdistr.clear()

        gs_iter = 0
        best_f = float("inf")
        seen = set()  # (order, vec) pairs to avoid double work across phases

        # ---- Evaluate Phase A (BASE) ----
        for order, vec in (self._candidates_base or []):
            if max_configs is not None and gs_iter >= max_configs:
                break
            key = (tuple(order), tuple(int(v) for v in vec))
            if key in seen:
                continue
            seen.add(key)

            cur_f = float(self.executor_pass.current_par_lookup(order, [int(v) for v in vec]))
            gs_iter += 1
            if gs_iter == 1:
                logging.info("[GS2] Initial utility = %.6f", cur_f)

            if cur_f < best_f:
                best_f = cur_f

            logging.debug("[GS2] A | iter=%d | order=%s | vec=%s | utility=%.6f",
                          gs_iter, order, vec, cur_f)

            if cur_f <= f_goal:
                logging.info("[GS2] 🎯 Achieved at iter=%d, utility=%.6f", gs_iter, cur_f)
                self.gs_fdistr.append(cur_f)
                self.gs_idistr.append(gs_iter)
                return gs_iter, cur_f

        # ---- Evaluate Phase B (UNION) ----
        for order, vec in (self._candidates_union or []):
            if max_configs is not None and gs_iter >= max_configs:
                break
            key = (tuple(order), tuple(int(v) for v in vec))
            if key in seen:
                continue
            seen.add(key)

            cur_f = float(self.executor_pass.current_par_lookup(order, [int(v) for v in vec]))
            gs_iter += 1

            if cur_f < best_f:
                best_f = cur_f

            logging.debug("[GS2] B | iter=%d | order=%s | vec=%s | utility=%.6f",
                          gs_iter, order, vec, cur_f)

            if cur_f <= f_goal:
                logging.info("[GS2] 🎯 Achieved at iter=%d, utility=%.6f", gs_iter, cur_f)
                self.gs_fdistr.append(cur_f)
                self.gs_idistr.append(gs_iter)
                return gs_iter, cur_f

        logging.info("[GS2] Finished. iters=%d, best=%.6f", gs_iter, best_f)
        return gs_iter, best_f if best_f < float("inf") else float("inf")

    # -------- internals --------
    def _strategies_for(self, component: str) -> List[int]:
        counts = getattr(self.executor_pass, "strategy_counts", None)
        if counts is None or component not in counts:
            return [1]
        return list(range(1, int(counts[component]) + 1))

    def _enumerate_full_space_for_components(
        self, components: List[str]
    ) -> List[Tuple[List[str], List[int]]]:
        """
        Build candidate list for a given component set:
          • Power set (must include 'model')
          • Permutations of non-model
          • Model last
          • Cartesian product of strategies for the chosen order
        """
        candidates: List[Tuple[List[str], List[int]]] = []
        dedup = set()

        # Unique order; ensure model exists
        universe = list(dict.fromkeys(components))
        if "model" not in universe:
            logging.warning("[GS2] 'model' not present in: %s", universe)
            return []

        strat_domain = {c: self._strategies_for(c) for c in universe}

        for r in range(1, len(universe) + 1):
            for subset in itertools.combinations(universe, r):
                if "model" not in subset:
                    continue
                non_model = [c for c in subset if c != "model"]
                for perm in itertools.permutations(non_model, len(non_model)):
                    order = list(perm) + ["model"]
                    domains = [strat_domain[c] for c in order]
                    for vec in itertools.product(*domains):
                        key = (tuple(order), tuple(int(v) for v in vec))
                        if key not in dedup:
                            dedup.add(key)
                            candidates.append((order, [int(v) for v in vec]))
        return candidates
