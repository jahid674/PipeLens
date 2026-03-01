#!/usr/bin/env python3
# coding: utf-8
# Learn2Clean-compatible FeatureSelector (rewritten)

import warnings
import time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class FeatureSelector():
    """
    Learn2Clean-compatible Feature Selector.

    Strategies
    ----------
    - "NONE_fs"      : no-op
    - "VARIANCE"     : VarianceThreshold on numeric columns only
    - "MUTUAL_INFO"  : mutual_info_classif on numeric columns only (requires y_train)

    Behavior
    ----------
    - Accepts dataset as dict {'train','test'} (DataFrames). 'test' optional.
    - Fits selection on TRAIN only (variance / MI), then applies same columns to TEST.
    - Keeps all non-numeric columns untouched (optionally excludes a column).
    - Does NOT change row count (IMPORTANT for Learn2Clean alignment).
    - Replaces inf with NaN for safety; if NaNs remain in numeric inputs, raises
      (because L2C expects imputer/missing module earlier).
    """

    def __init__(self, dataset, strategy='NONE_fs', threshold=0.01,
                 top_k=None, verbose=False, exclude=None):

        self.dataset = dataset
        self.strategy = str(strategy).upper().strip()

        self.threshold = float(threshold)
        self.top_k = None if top_k is None else int(top_k)
        self.verbose = bool(verbose)

        # exclude can be str or list
        self.exclude = exclude if isinstance(exclude, list) else ([exclude] if exclude else [])

        # learned artifacts
        self.selected_numeric_features_ = None
        self.selected_features_ = None   # final columns to keep (incl non-numeric + excluded if present)

        if self.strategy not in ("NONE_FS", "VARIANCE", "MUTUAL_INFO"):
            raise ValueError("Invalid strategy. Choose 'NONE_fs', 'VARIANCE', or 'MUTUAL_INFO'.")

    # ---------------- Learn2Clean API ----------------

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'threshold': self.threshold,
            'top_k': self.top_k,
            'verbose': self.verbose,
            'exclude': self.exclude
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for FeatureSelector. "
                              "Parameter(s) IGNORED. "
                              "Check with `feature_selector.get_params().keys()`")
            else:
                setattr(self, k, v)
        self.strategy = str(self.strategy).upper().strip()

    # ---------------- helpers ----------------

    def _split_numeric_non_numeric(self, df: pd.DataFrame):
        """
        Returns:
          - X_num: numeric-only DataFrame (excluding exclude cols if they are numeric)
          - X_other: all remaining columns (non-numeric + excluded)
        """
        df_work = df.copy()

        # pull out excluded columns (keep them always)
        excluded_cols = df_work[self.exclude].copy() if self.exclude else pd.DataFrame(index=df_work.index)
        df_work = df_work.drop(columns=self.exclude, errors="ignore")

        num_cols = df_work.select_dtypes(include=["number"]).columns.tolist()
        other_cols = [c for c in df_work.columns if c not in num_cols]

        X_num = df_work[num_cols].copy()
        X_other = df_work[other_cols].copy()

        return X_num, X_other, excluded_cols, num_cols

    def _sanitize_numeric(self, X_num: pd.DataFrame):
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        if X_num.isnull().any().any():
            raise ValueError(
                "FeatureSelector: NaNs present in numeric columns. "
                "Run missing-value handling (imputer) before feature selection."
            )
        return X_num

    def _fit(self, X_train: pd.DataFrame, y_train=None):
        X_num, X_other, excluded_cols, num_cols = self._split_numeric_non_numeric(X_train)

        # strategy NONE: keep everything
        if self.strategy == "NONE_FS" or len(num_cols) == 0:
            self.selected_numeric_features_ = num_cols
            # final: keep original order
            self.selected_features_ = list(X_train.columns)
            return

        X_num = self._sanitize_numeric(X_num)

        if self.strategy == "VARIANCE":
            selector = VarianceThreshold(threshold=self.threshold)
            selector.fit(X_num.values)
            keep_idx = selector.get_support(indices=True)
            keep_cols = [num_cols[i] for i in keep_idx]

        elif self.strategy == "MUTUAL_INFO":
            if y_train is None:
                raise ValueError("FeatureSelector (MUTUAL_INFO) requires y_train.")
            y_arr = np.asarray(y_train)

            scores = mutual_info_classif(X_num.values, y_arr, discrete_features="auto")
            n_features = len(num_cols)

            # Your note said 80%, but your code used 100%. We'll implement 80% default as described.
            if self.top_k is None:
                top_k = int(np.ceil(0.8 * n_features))
            else:
                top_k = int(self.top_k)

            top_k = max(1, min(top_k, n_features))
            keep_idx = np.argsort(scores)[-top_k:]
            keep_cols = [num_cols[i] for i in keep_idx]

        else:
            raise RuntimeError("Unexpected strategy in _fit().")

        # save learned numeric keepers
        self.selected_numeric_features_ = keep_cols

        # build final keep list in original order:
        # keep (a) non-numeric columns, (b) selected numeric, (c) excluded columns
        final_keep = []
        for c in X_train.columns:
            if c in self.exclude:
                final_keep.append(c)
            elif c in X_other.columns:
                final_keep.append(c)
            elif c in keep_cols:
                final_keep.append(c)

        self.selected_features_ = final_keep

    def _transform_one(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            # safe fallback: identity
            return X.copy()
        return X.loc[:, self.selected_features_].copy()

    # ---------------- driver ----------------

    def transform(self, y_train=None, sensitive_attr_train=None):
        """
        Learn2Clean-style transform:
          - expects dataset dict with 'train' (and optional 'test')
          - fits on train only
          - returns dict with same keys
        """
        start_time = time.time()
        outd = self.dataset

        print(">>Feature Selection ")

        if isinstance(self.dataset, dict):

            if "train" not in self.dataset or isinstance(self.dataset["train"], dict):
                print("No train dataset, no feature selection")
                return outd

            Xtr = self.dataset["train"].copy()
            if self.verbose:
                print("* For train dataset")

            self._fit(Xtr, y_train=y_train)

            outd["train"] = self._transform_one(Xtr)

            if "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                outd["test"] = self._transform_one(self.dataset["test"].copy())

            if self.verbose:
                print("Selected features:", self.selected_features_)
                print("Selected numeric features:", self.selected_numeric_features_)

        elif isinstance(self.dataset, pd.DataFrame):
            # fallback support if someone passes a DF directly
            X = self.dataset.copy()
            self._fit(X, y_train=y_train)
            outd = self._transform_one(X)

        else:
            raise TypeError("dataset must be a pandas DataFrame or dict with keys {'train','test'}.")

        print("Feature selection done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return outd
