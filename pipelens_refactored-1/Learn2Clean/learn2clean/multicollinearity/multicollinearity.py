#!/usr/bin/env python3
# coding: utf-8
# Author: (adapted to Learn2Clean-compatible format)

import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class VIFMulticollinearityCleaner():
    """
    Multicollinearity Detection & Repair using VIF (Learn2Clean compatible).

    Strategies:
      - "NONE"           : do nothing
      - "DROP_HIGH_VIF"  : iteratively drop numeric features with VIF > vif_threshold

    Notes:
      - Requires numeric features with no NaNs/inf (run imputer + inf handling first)
      - Keeps all non-numeric columns unchanged
      - Drops ONLY from numeric feature set (excluding `exclude` if provided)
      - Preserves original column order among kept columns
    """

    def __init__(self, dataset, strategy="DROP_HIGH_VIF",
                 vif_threshold=10.0, max_iter=50, min_features=2,
                 standardize=True, verbose=False, exclude=None, threshold=None):

        self.dataset = dataset
        self.strategy = str(strategy).upper().strip()

        self.vif_threshold = float(vif_threshold)
        self.max_iter = int(max_iter)
        self.min_features = int(min_features)
        self.standardize = bool(standardize)

        self.verbose = bool(verbose)
        self.exclude = exclude  # can be str or list; handled internally
        self.threshold = threshold  # unused; kept for API compatibility

        if self.strategy not in ("NONE", "DROP_HIGH_VIF"):
            raise ValueError("Strategy invalid. Please choose between "
                             "'NONE' or 'DROP_HIGH_VIF'.")

        # learned
        self.selected_features_ = None  # list of numeric features kept
        self.vif_table_ = None          # last VIF table on train

    # ------------------- Learn2Clean API -------------------

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'vif_threshold': self.vif_threshold,
            'max_iter': self.max_iter,
            'min_features': self.min_features,
            'standardize': self.standardize,
            'verbose': self.verbose,
            'exclude': self.exclude,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for VIFMulticollinearityCleaner. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`vif_cleaner.get_params().keys()`")
            else:
                setattr(self, k, v)

        # normalize strategy if changed
        self.strategy = str(self.strategy).upper().strip()

    # ------------------- helpers -------------------

    def _exclude_list(self):
        if self.exclude is None:
            return []
        if isinstance(self.exclude, list):
            return [c for c in self.exclude if c is not None]
        return [self.exclude]

    def _get_numeric_cols(self, df: pd.DataFrame):
        cols = df.select_dtypes(include=["number"]).columns.tolist()
        excl = set(self._exclude_list())
        return [c for c in cols if c not in excl]

    def _standardize_mat(self, mat: np.ndarray) -> np.ndarray:
        mu = np.mean(mat, axis=0)
        sd = np.std(mat, axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        return (mat - mu) / sd

    def _compute_vif(self, mat: np.ndarray, cols):
        """
        Compute VIF for each column in `mat` via OLS with intercept.
        mat: shape (n, p)
        """
        vifs = []
        n, p = mat.shape

        for j in range(p):
            y = mat[:, j]
            X_others = np.delete(mat, j, axis=1)  # shape (n, p-1)
            X_design = np.column_stack([np.ones(n), X_others])  # intercept

            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            y_hat = X_design @ beta

            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
            r2 = float(min(max(r2, 0.0), 0.999999999))  # avoid div by zero
            vifs.append(1.0 / (1.0 - r2))

        return (pd.DataFrame({"feature": list(cols), "vif": vifs})
                .sort_values("vif", ascending=False)
                .reset_index(drop=True))

    def _fit_on_train(self, df_train: pd.DataFrame):
        num_cols = self._get_numeric_cols(df_train)

        # reset learned state
        self.vif_table_ = None
        self.selected_features_ = list(num_cols)

        if self.strategy == "NONE" or len(num_cols) < self.min_features:
            return

        X = df_train[num_cols].replace([np.inf, -np.inf], np.nan)
        if X.isnull().any().any():
            raise ValueError("VIFMulticollinearityCleaner: VIF requires no NaNs/inf "
                             "in numeric features. Run imputation first.")

        keep = list(num_cols)

        for _ in range(self.max_iter):
            if len(keep) < self.min_features:
                break

            mat = X[keep].to_numpy(dtype=float)
            if self.standardize:
                mat = self._standardize_mat(mat)

            vif_df = self._compute_vif(mat, keep)
            self.vif_table_ = vif_df

            # stop if highest VIF is within threshold
            if float(vif_df.loc[0, "vif"]) <= self.vif_threshold:
                break

            # drop the feature with the highest VIF
            drop_feat = str(vif_df.loc[0, "feature"])
            if drop_feat in keep:
                keep.remove(drop_feat)
            else:
                break

        self.selected_features_ = keep

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # no-op
        if self.strategy == "NONE":
            return df.copy()

        num_cols = self._get_numeric_cols(df)
        selected = self.selected_features_ if self.selected_features_ is not None else num_cols

        # Keep all non-numeric columns + selected numeric columns, preserving original order
        keep_set = set(selected)
        keep_cols = []
        for c in df.columns:
            if c in num_cols:
                if c in keep_set:
                    keep_cols.append(c)
            else:
                keep_cols.append(c)
        return df.loc[:, keep_cols].copy()

    # ------------------- driver -------------------

    def transform(self):

        start_time = time.time()
        outd = self.dataset

        print(">>VIF multicollinearity cleaning ")

        for key in ['train']:

            if (isinstance(self.dataset, dict)
                    and key in self.dataset
                    and (not isinstance(self.dataset[key], dict))):

                d = self.dataset[key].copy()
                print("* For", key, "dataset")

                # fit on train once
                self._fit_on_train(d)

                # transform train
                outd[key] = self._transform_df(d)

                # transform test if present
                if "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                    outd["test"] = self._transform_df(self.dataset["test"].copy())

                if self.verbose and self.vif_table_ is not None:
                    print("Selected numeric features:", self.selected_features_)
                    print("Top VIF rows:\n", self.vif_table_.head(10))

            else:
                print("No", key, "dataset, no VIF cleaning")

        print("VIF multicollinearity cleaning done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return outd
