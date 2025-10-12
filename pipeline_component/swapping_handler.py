# handlers/structure/swap_handler.py
from typing import List, Dict, Any
from modules.swapping.swapper import Swapper
from typing import List, Dict, Any, Tuple

class SwapHandler:
    def __init__(self, strategy: int, config: Dict[str, Any]):
        self.strategy = int(strategy)
        self.constraints = (config or {}).get("swap_constraints", None)
        self.verbose = bool((config or {}).get("verbose", False))
        self.drop_labels = set((config or {}).get("drop_labels", []))

    def apply(self, pipeline: List[Any]) -> List[Any]:
        swapper = Swapper(
            pipeline=pipeline,
            index=self.strategy,
            verbose=self.verbose,
            constraints=self.constraints
        )
        return swapper.transform()

    def apply_with_params(
        self, pipeline: List[Any], params: List[int]
    ) -> Tuple[List[Any], List[int]]:
        i = self.strategy
        n = len(pipeline)

        new_pipeline = self.apply(pipeline)
        new_params   = list(params)

        # If swap was invalid or rejected, pipeline unchanged -> keep params
        if new_pipeline == pipeline:
            pass
        else:
            # valid adjacent swap -> mirror the same swap in params
            if 0 <= i < n - 1 and len(params) == n:
                new_params[i], new_params[i+1] = new_params[i+1], new_params[i]

        # Optional: drop meta-ops (e.g., "swapping") and keep params aligned
        if self.drop_labels:
            keep_idx = [k for k, m in enumerate(new_pipeline) if m not in self.drop_labels]
            new_pipeline = [new_pipeline[k] for k in keep_idx]
            new_params   = [new_params[k]   for k in keep_idx]

        return new_pipeline, new_params

