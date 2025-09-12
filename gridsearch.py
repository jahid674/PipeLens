import itertools
import random
import logging
from typing import Iterable, List, Optional, Tuple

from score_lookup import ScoreLookup
from pipeline_execution import PipelineExecutor


class GridSearch:
    """
    One-shot exhaustive grid search over the FULL pipeline space:

      • Parameter-change interventions:
          - All strategy assignments for the components already in `pipeline_order`.

      • New-component insertions:
          - All subsets (any size) of `new_components`, preserving their input order,
          - Inserted at every possible position,
          - With every valid strategy for each inserted component.

    Algorithm (single phase):
      1) Enumerate the FULL search space as concrete (order, vec) pairs.
      2) Shuffle the space uniformly at random (no seed).
      3) Evaluate in that random order until target met or space exhausted.

    Notes:
      - No `max_new`: we consider all subset sizes of `new_components`.
      - `randomize=True` shuffles with system entropy; `False` keeps deterministic build order.
    """

    def __init__(self, dataset_name, historical_data, pipeline_order, metric_type, pipeline_type):
        self.historical_data_pd = historical_data
        self.historical_data = getattr(historical_data, "values", [[]]).tolist() if hasattr(historical_data, "values") else []
        self.pipeline_order = list(pipeline_order)
        self.metric_type = metric_type
        self.dataset_name = dataset_name

        self.gs_idistr: List[int] = []
        self.gs_fdistr: List[float] = []

        # Executors
        self.score_lookup = ScoreLookup(self.pipeline_order, metric_type)
        self.executor_pass = PipelineExecutor(
            pipeline_type=pipeline_type,
            dataset_name=self.dataset_name,
            metric_type=self.metric_type,
            pipeline_ord=self.pipeline_order
        )

    # -------------------- Public API --------------------

    def grid_search(
        self,
        f_goal: float,
        new_components: Optional[Iterable[str]] = None,
        max_configs: Optional[int] = None,
        randomize: bool = True,
    ) -> Tuple[int, float]:
        """
        Returns (gs_iter, f_value). Single-phase: build full space, shuffle, evaluate.
        """
        logging.info("[GridSearch] Starting grid search (single-phase)...")
        logging.info(f"[GridSearch] Target utility = {f_goal:.6f}")

        uniq_new = list(dict.fromkeys(new_components or []))
        all_candidates = self._enumerate_full_space(uniq_new)

        logging.info(f"[GridSearch] Full search space size = {len(all_candidates)}")

        if not all_candidates:
            logging.info("[GridSearch] Empty search space.")
            return 0, float("inf")

        # Shuffle uniformly at random (no seed) unless randomize=False
        if randomize:
            random.SystemRandom().shuffle(all_candidates)

        self.gs_idistr.clear()
        self.gs_fdistr.clear()

        gs_iter = 0
        best_f = float("inf")
        seen = set()  # dedup safety

        for order, vec in all_candidates:
            if max_configs is not None and gs_iter >= max_configs:
                break

            key = (tuple(order), tuple(int(v) for v in vec))
            if key in seen:
                continue
            seen.add(key)

            logging.debug(f"[GridSearch] Evaluating pipeline: order={order}, vec={vec}")
            cur_f = float(self.executor_pass.current_par_lookup(order, [int(v) for v in vec]))
            gs_iter += 1

            if gs_iter == 1:
                logging.info(f"[GridSearch] Initial utility = {cur_f:.6f}")
            logging.debug(f"[GridSearch] Result: Utility={cur_f:.6f}")

            if cur_f < best_f:
                best_f = cur_f

            if cur_f <= f_goal:
                logging.info(f"[GridSearch] 🎯 Target achieved at iter={gs_iter}, utility={cur_f:.6f}")
                self.gs_fdistr.append(cur_f)
                self.gs_idistr.append(gs_iter)
                return gs_iter, cur_f

        logging.info(f"[GridSearch] Finished. Total iters={gs_iter}, best utility={best_f:.6f}")
        return gs_iter, best_f if best_f < float("inf") else float("inf")

    # -------------------- Internals --------------------

    def _strategies_for(self, component: str) -> List[int]:
        """
        Returns the list of valid strategy indices for a component.
        Falls back to [1] if unknown.
        """
        counts = getattr(self.executor_pass, "strategy_counts", None)
        if counts is None or component not in counts:
            return [1]
        return list(range(1, int(counts[component]) + 1))

    def _enumerate_full_space(self, new_components: List[str]) -> List[Tuple[List[str], List[int]]]:
        """
        Build the entire candidate set as concrete (order, vec) pairs.
        This includes:
          - All parameter-assignments for the existing pipeline (base).
          - For every base assignment, all insertions of every subset of `new_components`
            with all strategy assignments, inserted at every position.
        """
        candidates: List[Tuple[List[str], List[int]]] = []
        dedup = set()

        # 1) Enumerate all strategy assignments for existing pipeline
        existing_domains = [self._strategies_for(c) for c in self.pipeline_order]
        for existing_vec in itertools.product(*existing_domains):
            base_order = list(self.pipeline_order)
            base_vec = list(existing_vec)

            # Base itself (parameter-change only)
            self._add_candidate(candidates, dedup, base_order, base_vec)

            # 2) For all non-empty subsets of new components
            for k in range(1, len(new_components) + 1):
                for subset in itertools.combinations(new_components, k):
                    # enumerate all strategy assignments for the chosen subset
                    strat_domains = [self._strategies_for(c) for c in subset]
                    for subset_strats in itertools.product(*strat_domains):
                        comps = list(subset)
                        strats = list(subset_strats)
                        # insert comps (in given order) at all possible positions
                        self._enumerate_insertions_and_add(
                            candidates, dedup, base_order, base_vec, comps, strats
                        )

        return candidates

    def _enumerate_insertions_and_add(
        self,
        candidates: List[Tuple[List[str], List[int]]],
        dedup: set,
        base_order: List[str],
        base_vec: List[int],
        comps: List[str],
        strats: List[int],
    ):
        """
        Recursively insert a sequence of components (comps) with fixed strategies (strats)
        into every possible position of base_order/base_vec.
        """
        def rec(order: List[str], vec: List[int], idx: int):
            if idx == len(comps):
                self._add_candidate(candidates, dedup, order, vec)
                return
            comp = comps[idx]
            s = int(strats[idx])
            for pos in range(len(order) + 1):
                new_order = order[:pos] + [comp] + order[pos:]
                new_vec = vec[:pos] + [s] + vec[pos:]
                rec(new_order, new_vec, idx + 1)

        rec(list(base_order), list(base_vec), 0)

    def _add_candidate(
        self,
        candidates: List[Tuple[List[str], List[int]]],
        dedup: set,
        order: List[str],
        vec: List[int],
    ):
        key = (tuple(order), tuple(int(v) for v in vec))
        if key not in dedup:
            dedup.add(key)
            candidates.append((list(order), [int(v) for v in vec]))
