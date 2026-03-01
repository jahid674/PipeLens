#!/usr/bin/env python3
# coding: utf-8
# Author: (adapted to Learn2Clean-compatible format)

import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class DistributionShapeCorrector:
    """
    Distribution Shape Correction for numeric data (Learn2Clean compatible).

    Strategies:
      - "NONE"        : no-op
      - "LOG1P"       : apply log1p(x + shift) per numeric column (shift learned on train if needed)
      - "SQRT"        : apply sqrt(x + shift) per numeric column (shift learned on train if needed)
      - "BOXCOX"      : PowerTransformer(method="box-cox") on numeric columns (requires > 0; shift learned)
      - "YEOJOHNSON"  : PowerTransformer(method="yeo-johnson") on numeric columns (supports negatives)

    Notes:
      - Auto-detects numeric columns (excluding `exclude` if provided)
      - Fits shifts / PowerTransformer on TRAIN only (dataset['train'])
      - Transforms train (and optionally test) consistently
      - Does NOT change row count; preserves non-numeric columns & column order
      - Requires NO NaNs / infs in numeric columns (run imputer first)
    """

    def __init__(self, dataset, strategy="NONE", standardize=False, epsilon=1e-6,
                 verbose=False, exclude=None, threshold=None):
        self.dataset = dataset
        self.strategy = str(strategy).upper().strip()
        self.standardize = bool(standardize)
        self.epsilon = float(epsilon)
        self.verbose = bool(verbose)
        self.exclude = exclude  # can be str or list; handled internally
        self.threshold = threshold  # unused, kept for API compatibility

        # learned artifacts
        self._num_cols_ = None
        self._shift_map_ = {}   # per-column shifts for LOG1P/SQRT/BOXCOX
        self._pt_ = None        # PowerTransformer for BOXCOX/YEOJOHNSON

        if self.strategy not in ("NONE", "LOG1P", "SQRT", "BOXCOX", "YEOJOHNSON"):
            raise ValueError("Strategy invalid. Please choose between "
                             "'NONE', 'LOG1P', 'SQRT', 'BOXCOX', or 'YEOJOHNSON'.")

    # ------------------- Learn2Clean API -------------------

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'standardize': self.standardize,
            'epsilon': self.epsilon,
            'verbose': self.verbose,
            'exclude': self.exclude,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for DistributionShapeCorrector. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`distribution_shape_corrector.get_params().keys()`")
            else:
                setattr(self, k, v)

        # normalize strategy if user changed it
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
        cols = [c for c in cols if c not in excl]
        return cols

    def _sanitize_numeric(self, X_num: pd.DataFrame):
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        if X_num.isnull().any().any():
            raise ValueError(
                "DistributionShapeCorrector: NaNs/inf present in numeric columns. "
                "Run missing-value handling (and remove infs) first."
            )
        return X_num

    def _learn_shift_for_nonneg(self, series: pd.Series):
        """Smallest shift so (x + shift) >= 0 + epsilon."""
        mn = series.min()
        if mn >= 0:
            return 0.0
        return float(-mn + self.epsilon)

    def _learn_shift_for_positive(self, series: pd.Series):
        """Smallest shift so (x + shift) > 0 + epsilon (needed for Box-Cox)."""
        mn = series.min()
        if mn > 0:
            return 0.0
        return float(-mn + self.epsilon)

    def _fit_on_train(self, df_train: pd.DataFrame):
        self._num_cols_ = self._get_numeric_cols(df_train)

        # reset artifacts
        self._shift_map_ = {}
        self._pt_ = None

        if len(self._num_cols_) == 0:
            return

        X_num = self._sanitize_numeric(df_train[self._num_cols_].copy())

        if self.strategy == "NONE":
            return

        if self.strategy in ("LOG1P", "SQRT"):
            self._shift_map_ = {c: self._learn_shift_for_nonneg(X_num[c]) for c in self._num_cols_}

        elif self.strategy == "BOXCOX":
            self._shift_map_ = {c: self._learn_shift_for_positive(X_num[c]) for c in self._num_cols_}
            X_shifted = X_num.copy()
            for c in self._num_cols_:
                X_shifted[c] = X_shifted[c] + self._shift_map_[c]

            self._pt_ = PowerTransformer(method="box-cox", standardize=self.standardize)
            self._pt_.fit(X_shifted.values)

        elif self.strategy == "YEOJOHNSON":
            self._pt_ = PowerTransformer(method="yeo-johnson", standardize=self.standardize)
            self._pt_.fit(X_num.values)

        else:
            raise RuntimeError("Unexpected strategy in _fit_on_train().")

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        num_cols = self._num_cols_ if self._num_cols_ is not None else self._get_numeric_cols(out)
        if len(num_cols) == 0 or self.strategy == "NONE":
            return out

        X_num = self._sanitize_numeric(out[num_cols].copy())

        if self.strategy == "LOG1P":
            for c in num_cols:
                shift = self._shift_map_.get(c, 0.0)
                out[c] = np.log1p(X_num[c] + shift)

        elif self.strategy == "SQRT":
            for c in num_cols:
                shift = self._shift_map_.get(c, 0.0)
                out[c] = np.sqrt(X_num[c] + shift)

        elif self.strategy == "BOXCOX":
            if self._pt_ is None:
                return out
            X_shifted = X_num.copy()
            for c in num_cols:
                X_shifted[c] = X_shifted[c] + self._shift_map_.get(c, 0.0)
            arr = self._pt_.transform(X_shifted.values)
            out[num_cols] = arr

        elif self.strategy == "YEOJOHNSON":
            if self._pt_ is None:
                return out
            arr = self._pt_.transform(X_num.values)
            out[num_cols] = arr

        else:
            raise RuntimeError("Unexpected strategy in _transform_df().")

        # safety: avoid inf
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    # ------------------- driver -------------------

    def transform(self):

        start_time = time.time()
        dscd = self.dataset

        print(">>Distribution shape correction ")

        for key in ['train']:

            if (isinstance(self.dataset, dict)
                    and key in self.dataset
                    and (not isinstance(self.dataset[key], dict))):

                d = self.dataset[key].copy()
                print("* For", key, "dataset")

                # Fit artifacts on train split only (once)
                self._fit_on_train(d)

                # Apply transform to train
                dn = self._transform_df(d)
                dscd[key] = dn

                # Apply to test if present
                if isinstance(self.dataset, dict) and "test" in self.dataset and self.dataset["test"] is not None:
                    if not isinstance(self.dataset["test"], dict):
                        d_test = self.dataset["test"].copy()
                        dscd["test"] = self._transform_df(d_test)

            else:
                # If dataset isn't dict-like with train, keep unchanged
                print("No", key, "dataset, no distribution shape correction")

        print("Distribution shape correction done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return dscd
