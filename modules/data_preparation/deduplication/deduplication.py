import pandas as pd
import numpy as np
np.random.seed(42)

class Deduplicator:
    """
    Deduplication that keeps y/sensitive aligned safely.

    Key idea:
      - Always reset indices first (so X,y,sensitive share 0..n-1).
      - Use numpy boolean masks + iloc (positional), never loc with mask Series.
      - For duplicates, compute keep_mask on X and apply to X,y,sensitive.
    """

    def __init__(self, dataset, strategy='dd', subset=None, verbose=False):
        self.dataset = dataset.copy()
        self.subset = subset  # columns for drop_duplicates, optional
        self.verbose = verbose
        self.strategy = str(strategy).lower()

        if self.strategy not in ["dd", "none"]:
            raise ValueError("strategy must be one of ['dd', 'none']")

    def transform(self, y=None, sensitive=None):
        # -------------------------
        # 0) force consistent indexing
        # -------------------------
        X = self.dataset.reset_index(drop=True)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.reset_index(drop=True)
        elif y is not None:
            y = pd.Series(y).reset_index(drop=True)

        if isinstance(sensitive, (pd.Series, pd.DataFrame)):
            sensitive = sensitive.reset_index(drop=True)
        elif sensitive is not None:
            sensitive = pd.Series(sensitive).reset_index(drop=True)

        before_all = len(X)

        # -------------------------
        # 1) drop rows with missing values in X (and align y/sensitive)
        # -------------------------
        na_mask = X.notna().all(axis=1).to_numpy(dtype=bool)

        X = X.iloc[na_mask].reset_index(drop=True)
        if y is not None:
            y = y.iloc[na_mask].reset_index(drop=True)
        if sensitive is not None:
            sensitive = sensitive.iloc[na_mask].reset_index(drop=True)

        if self.verbose:
            print(f"Dropped {before_all - len(X)} rows with missing values.")

        # -------------------------
        # 2) deduplication
        # -------------------------
        if self.strategy == "none":
            if self.verbose:
                print("No deduplication applied (strategy='none').")
            self.dataset = X
            return X, y, sensitive

        before = len(X)

        # Compute keep mask for duplicates (True = keep)
        # IMPORTANT: keep='first' like pandas drop_duplicates default
        dup_mask = X.duplicated(subset=self.subset, keep="first").to_numpy(dtype=bool)
        keep_mask = ~dup_mask

        X = X.iloc[keep_mask].reset_index(drop=True)
        if y is not None:
            y = y.iloc[keep_mask].reset_index(drop=True)
        if sensitive is not None:
            sensitive = sensitive.iloc[keep_mask].reset_index(drop=True)

        after = len(X)

        if self.verbose:
            print(f"Removed {before - after} duplicate rows.")

        self.dataset = X
        return X, y, sensitive