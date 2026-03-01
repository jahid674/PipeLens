import itertools
import random
import logging
from typing import Iterable, List, Optional, Tuple
from pipeline_execution import PipelineExecutor


class GridSearch:
    """
    Random search over the space of subsets/orders/strategies from {base ∪ new}.

    Rules:
      • 'model' MUST be included and MUST be LAST
      • No component appears more than once
      • Order matters (permute non-model, then append 'model')
      • For each order, sample one strategy per component

    Behavior:
      • No precomputation or shuffling of full space
      • Repeatedly (uniformly) samples a random valid (order, vec) and evaluates
      • Stops early if utility ≤ f_goal, or after max_configs evaluations
      • If max_configs is None, defaults to 10_000 random trials to avoid infinite loops
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

        # Retained for compatibility (no longer used in random search)
        self._cached_new_components: Optional[Tuple[str, ...]] = None
        self._candidates: Optional[List[Tuple[List[str], List[int]]]] = None


    def grid_search(
        self,
        f_goal: float,
        new_components: Optional[Iterable[str]] = None,
        max_configs: Optional[int] = None,
        randomize: bool = True,        
        reuse_cached: bool = True,
        rng: Optional[random.Random] = None
    ) -> Tuple[int, float]:
        """
        Returns (gs_iter, f_value) for the best seen.
        Randomly samples valid (order, vec) and evaluates until threshold or budget.
        """
        logging.info("[GridSearch] Start (random). Target utility = %.2f", f_goal)

        # RNG: use provided rng or a system RNG
        _rng = rng or random.SystemRandom()

        # Build universe = unique(base ∪ new_components)
        uniq_new = tuple(dict.fromkeys(new_components or []))
        universe = list(dict.fromkeys(list(self.pipeline_order) + list(uniq_new)))

        if "model" not in universe:
            logging.warning("[GridSearch] 'model' not in universe; empty space.")
            return 0, float("inf")

        # Strategy domains per component
        strat_domain = {c: self._strategies_for(c) for c in universe}

        # Budget safeguard to avoid unbounded loops
        if max_configs is None:
            max_configs = 10_000  # safe default

        self.gs_idistr.clear()
        self.gs_fdistr.clear()

        gs_iter = 0
        best_f = float("inf")
        seen = set()

        # Precompute convenience lists
        non_model_all = [c for c in universe if c != "model"]

        while gs_iter < max_configs:
            # --- Sample a random subset that includes 'model' ---
            # Randomly choose subset size in [1, len(universe)], ensuring 'model' present
            # We choose k_non_model in [0, len(non_model_all)] then add 'model'
            k_non_model = _rng.randrange(0, len(non_model_all) + 1)
            subset_non_model = _rng.sample(non_model_all, k_non_model)
            order_non_model = subset_non_model[:]  # copy

            # Random permutation of chosen non-model components
            _rng.shuffle(order_non_model)

            # Enforce 'model' last
            order = order_non_model + ["model"]

            # --- Sample a random strategy vector for this order ---
            vec = [ _rng.choice(strat_domain[c]) for c in order ]

            key = (tuple(order), tuple(int(v) for v in vec))
            if key in seen:
                # Avoid duplicate work; try again without counting toward budget
                continue
            seen.add(key)

            # Evaluate
            cur_f = float(self.executor_pass.current_par_lookup(order, [int(v) for v in vec]))
            gs_iter += 1
            if gs_iter == 1:
                logging.info("[GridSearch] Initial utility = %:.2f", cur_f)

            if cur_f < best_f:
                best_f = cur_f

            logging.debug(f"[GridSearch] iter={gs_iter} | order={order} | vec={vec} | utility={cur_f:.2f}")

            # Early stopping on threshold
            if cur_f <= f_goal:
                logging.info("[GridSearch] 🎯 Achieved at iter=%d, utility=:.2f", gs_iter, cur_f)
                self.gs_fdistr.append(cur_f)
                self.gs_idistr.append(gs_iter)
                return gs_iter, cur_f

        logging.info("[GridSearch] Finished (random). iters=%d, best=%.2f", gs_iter, best_f)
        return gs_iter, best_f if best_f < float("inf") else float("inf")

    # ---------- internals ----------
    def _strategies_for(self, component: str) -> List[int]:
        counts = getattr(self.executor_pass, "strategy_counts", None)
        if counts is None or component not in counts:
            return [1]
        return list(range(1, int(counts[component]) + 1))

    # Keeping this for compatibility even though random search doesn't use it
    def _enumerate_full_space(self, new_components: List[str]) -> List[Tuple[List[str], List[int]]]:
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
