# modules/sampling/stratification_split.py

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split


class StratificationSplitter:
    """
    Creates a train/test split with either:
      - "random"      : random split (no stratification)
      - "stratified"  : stratified split by y

    This is designed as a PIPELINE module that *changes the dataset container*:
      Input:
        - X as a DataFrame (full dataset)
        - y as a Series/1D array
        - sensitive as a Series/1D array (optional)
      Output:
        - X_dict = {"train": X_train_df, "test": X_test_df}
        - y_train, y_test
        - s_train, s_test

    If your PipelineExecutor expects (X, y, sensitive) for downstream steps, you have two choices:
      1) Keep y/sensitive as dicts too (recommended), OR
      2) Keep them separate and store test labels elsewhere.
    Here we return dicts for consistency with X.
    """

    def __init__(
        self,
        dataset,
        strategy="random",
        test_size=0.2,
        shuffle=True,
        random_state=42,
        verbose=False,
        exclude=None,
    ):
        self.dataset = dataset.copy()  # should be a DataFrame for this module
        self.strategy = str(strategy).lower().strip()
        self.test_size = float(test_size)
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

    def transform(self, y_train=None, sensitive_attr_train=None):
        """
        Returns:
          X_split: {"train": X_train_df, "test": X_test_df}
          y_split: {"train": y_train, "test": y_test}
          s_split: {"train": s_train, "test": s_test}  (or None)
        """
        start_time = time.time()
        if self.verbose:
            print("----- Starting Stratification Split -----")

        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("StratificationSplitter expects dataset as a single pandas DataFrame (full data).")

        if y_train is None:
            raise ValueError("StratificationSplitter.transform requires y_train (labels for splitting).")

        X = self.dataset.copy()

        # optional: drop excluded columns before split (rare). Usually exclude=None.
        excluded_cols = X[self.exclude].copy() if self.exclude else pd.DataFrame(index=X.index)
        X_core = X.drop(columns=self.exclude, errors="ignore")

        y = pd.Series(y_train).reset_index(drop=True)
        if len(y) != len(X_core):
            raise ValueError("X and y length mismatch in StratificationSplitter.")

        s = None
        if sensitive_attr_train is not None:
            s = pd.Series(sensitive_attr_train).reset_index(drop=True)
            if len(s) != len(X_core):
                raise ValueError("X and sensitive length mismatch in StratificationSplitter.")

        stratify = None
        if self.strategy in ("stratified", "stratify"):
            stratify = y
        elif self.strategy in ("random", "none"):
            stratify = None
        else:
            raise ValueError("Invalid strategy. Choose from {'random','stratified'}.")

        # Split indices
        idx = np.arange(len(X_core))
        idx_train, idx_test = train_test_split(
            idx,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
            stratify=stratify if self.shuffle else None,
        )

        X_train = X_core.iloc[idx_train].reset_index(drop=True)
        X_test = X_core.iloc[idx_test].reset_index(drop=True)

        y_tr = y.iloc[idx_train].reset_index(drop=True)
        y_te = y.iloc[idx_test].reset_index(drop=True)

        s_tr, s_te = None, None
        if s is not None:
            s_tr = s.iloc[idx_train].reset_index(drop=True)
            s_te = s.iloc[idx_test].reset_index(drop=True)

        # Reattach excluded columns after split
        if not excluded_cols.empty:
            excl_train = excluded_cols.iloc[idx_train].reset_index(drop=True)
            excl_test = excluded_cols.iloc[idx_test].reset_index(drop=True)

            X_train = pd.concat([X_train, excl_train], axis=1)
            X_test = pd.concat([X_test, excl_test], axis=1)

            # restore original column order
            X_train = X_train[X.columns]
            X_test = X_test[X.columns]

        X_split = {"train": X_train, "test": X_test}
        y_split = {"train": y_tr, "test": y_te}
        s_split = None if s is None else {"train": s_tr, "test": s_te}

        if self.verbose:
            print(f"Split completed in {time.time() - start_time:.2f} seconds. "
                  f"Train={len(X_train)}, Test={len(X_test)}")

        return X_split, y_split, s_split
