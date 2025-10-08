import itertools
import random
import logging
from typing import Iterable, List, Optional, Tuple
from pipeline_execution import PipelineExecutor


class GridSearch:
    """
    Grid search over POWER-SET of {base ∪ new} components with caching.

    Rules:
      • 'model' MUST be included and MUST be LAST
      • No component appears more than once
      • Order matters (permute non-model, then append 'model')
      • For each order, enumerate Cartesian product of per-component strategies

    Caching:
      • The full (order, vec) candidate list is built ONCE and cached in `self._candidates`
      • Subsequent `grid_search()` calls only SHUFFLE and EVALUATE that cached list
      • Set `reuse_cached=False` to rebuild explicitly
    """

    # ---------- ctor unchanged ----------
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
            pipeline_ord=self.pipeline_order,
            execution_type='fail'
        )

        # --- NEW: cache for full search space ---
        self._cached_new_components: Optional[Tuple[str, ...]] = None
        self._candidates: Optional[List[Tuple[List[str], List[int]]]] = None

    # ---------- public API unchanged signature; adds reuse_cached flag ----------
    def grid_search(
        self,
        f_goal: float,
        new_components: Optional[Iterable[str]] = None,
        max_configs: Optional[int] = None,
        randomize: bool = True,
        reuse_cached: bool = True,   # <--- NEW (default True)
        rng: Optional[random.Random] = None  # optional RNG for reproducibility
    ) -> Tuple[int, float]:
        """
        Returns (gs_iter, f_value).
        If reuse_cached=True and the universe (base ∪ new) hasn't changed,
        we only shuffle and evaluate the cached candidates (no rebuild).
        """
        logging.info("[GridSearch] Start. Target utility = %.6f", f_goal)

        uniq_new = tuple(dict.fromkeys(new_components or []))
        # (Re)build cache only if needed
        if (not reuse_cached) or self._candidates is None or self._cached_new_components != uniq_new:
            logging.info("[GridSearch] Building full search space (power set over base+new)...")
            self._candidates = self._enumerate_full_space(list(uniq_new))
            self._cached_new_components = uniq_new
            logging.info("[GridSearch] Built %d candidates.", len(self._candidates))
        else:
            logging.info("[GridSearch] Reusing cached %d candidates.", len(self._candidates))

        if not self._candidates:
            logging.info("[GridSearch] Empty search space.")
            return 0, float("inf")

        # Shuffle in-place (no rebuild)
        if randomize:
            (rng or random.SystemRandom()).shuffle(self._candidates)

        self.gs_idistr.clear()
        self.gs_fdistr.clear()

        gs_iter = 0
        best_f = float("inf")
        seen = set()

        for order, vec in self._candidates:
            if max_configs is not None and gs_iter >= max_configs:
                break

            key = (tuple(order), tuple(int(v) for v in vec))
            if key in seen:
                continue
            seen.add(key)

            cur_f = float(self.executor_pass.current_par_lookup(order, [int(v) for v in vec]))
            gs_iter += 1
            if gs_iter == 1:
                logging.info("[GridSearch] Initial utility = %.6f", cur_f)

            if cur_f < best_f:
                best_f = cur_f
            logging.debug(f"[GridSearch] iter={gs_iter} | order={order} | vec={vec} | utility={cur_f:.6f}")
            # Early stop on threshold
            if cur_f <= f_goal:
                logging.info("[GridSearch] 🎯 Achieved at iter=%d, utility=%.6f", gs_iter, cur_f)
                self.gs_fdistr.append(cur_f)
                self.gs_idistr.append(gs_iter)
                return gs_iter, cur_f

        logging.info("[GridSearch] Finished. iters=%d, best=%.6f", gs_iter, best_f)
        return gs_iter, best_f if best_f < float("inf") else float("inf")

    # ---------- internals (power-set enumeration; same as spec) ----------
    def _strategies_for(self, component: str) -> List[int]:
        counts = getattr(self.executor_pass, "strategy_counts", None)
        if counts is None or component not in counts:
            return [1]
        return list(range(1, int(counts[component]) + 1))

    def _enumerate_full_space(self, new_components: List[str]) -> List[Tuple[List[str], List[int]]]:
        """
        Universe U = unique(base pipeline_order ∪ new_components)
        Require 'model' ∈ subset and LAST in order.
        For each subset S containing 'model':
          - permute A = S \ {model}
          - order = perm + ['model']
          - vec = Cartesian product of strategies for components in 'order'
        """
        candidates: List[Tuple[List[str], List[int]]] = []
        dedup = set()

        universe = list(dict.fromkeys(list(self.pipeline_order) + list(new_components)))
        if "model" not in universe:
            logging.warning("[GridSearch] 'model' not in universe; empty space.")
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
                            candidates.append((list(order), [int(v) for v in vec]))
        return candidates
