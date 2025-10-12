# modules/structure/adjacent_swapper.py
import copy
from typing import List, Dict, Any, Optional

class Swapper:
    """
    Atomic module that swaps a single adjacent pair (i, i+1) in a pipeline.

    Parameters
    ----------
    pipeline : List[Any]
        The current pipeline as an ordered list of component labels (e.g., ["missing_value","normalization","fselection","model"]).
    index : int
        Left index of the adjacent pair to swap; must satisfy 0 <= index < len(pipeline) - 1.
    verbose : bool
        If True, prints details.
    constraints : Optional[Dict[str, Any]]
        Optional soft precedence constraints:
          - "must_precede": Dict[str, List[str]]  (e.g., {"missing_value": ["model"]})
        If provided, swaps that violate constraints are rejected (returning the original pipeline).
    """

    def __init__(self,
                 pipeline: List[Any],
                 index: int,
                 verbose: bool = False,
                 constraints: Optional[Dict[str, Any]] = None):
        self.pipeline = list(pipeline)
        self.index = int(index)
        self.verbose = verbose
        self.constraints = constraints or {}

    def _violates_precedence(self, pipe: List[Any]) -> bool:
        """
        Minimal precedence checker. If constraints['must_precede'] exists,
        ensure every a in keys appears before all b in list.
        """
        must_precede: Dict[str, List[str]] = self.constraints.get("must_precede", {}) or {}
        pos = {name: i for i, name in enumerate(pipe)}
        for a, bs in must_precede.items():
            if a not in pos:
                continue
            for b in bs or []:
                if b not in pos:
                    continue
                if pos[a] > pos[b]:
                    return True
        return False

    def transform(self) -> List[Any]:
        n = len(self.pipeline)
        if n < 2:
            if self.verbose:
                print("[AdjacentSwapper] Pipeline too short for swapping.")
            return self.pipeline

        if not (0 <= self.index < n - 1):
            if self.verbose:
                print(f"[AdjacentSwapper] Invalid index {self.index} for pipeline of length {n}.")
            return self.pipeline

        new_pipe = copy.copy(self.pipeline)
        i = self.index
        new_pipe[i], new_pipe[i + 1] = new_pipe[i + 1], new_pipe[i]

        if self._violates_precedence(new_pipe):
            if self.verbose:
                print("[AdjacentSwapper] Swap rejected due to precedence constraints.")
            return self.pipeline

        if self.verbose:
            print(f"[AdjacentSwapper] Swapped positions {i} and {i+1}:")
            print("  before:", self.pipeline)
            print("  after :", new_pipe)

        return new_pipe
