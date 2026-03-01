# modules/numerical/distribution_shape_correction.py

import pandas as pd
import numpy as np
np.random.seed(42)
import time
from sklearn.preprocessing import PowerTransformer


class DistributionShapeCorrector:
    """
    Distribution Shape Correction for numeric data.

    Strategies:
      - "log1p"       : apply log1p(x + shift) per numeric column (shift learned on train if needed)
      - "sqrt"        : apply sqrt(x + shift) per numeric column (shift learned on train if needed)
      - "boxcox"      : sklearn PowerTransformer(method="box-cox") on numeric columns (requires > 0; shift learned)
      - "yeojohnson"  : sklearn PowerTransformer(method="yeo-johnson") on numeric columns (supports negatives)

    Behavior:
      - Auto-detects numeric columns
      - Fits shifts / PowerTransformer on TRAIN only (if dataset is dict)
      - Transforms train/test consistently
      - Does NOT change row count; y and sensitive unchanged
    """

    def __init__(
        self,
        dataset,
        strategy,
        standardize=False,     # for PowerTransformer only
        epsilon=1e-6,          # for positivity in boxcox/log/sqrt shifts
        verbose=False,
        exclude=None,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.strategy = str(strategy).lower().strip()
        self.standardize = bool(standardize)
        self.epsilon = float(epsilon)
        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

        # learned artifacts
        self._num_cols_ = None
        self._shift_map_ = {}          # per-column shifts for log/sqrt/boxcox
        self._pt_ = None               # PowerTransformer for boxcox/yeojohnson

        if self.strategy not in ("none","log1p", "sqrt", "boxcox", "yeojohnson"):
            raise ValueError("Invalid strategy. Choose from {'log1p','sqrt','boxcox','yeojohnson'}.")

    def _get_numeric_cols(self, df: pd.DataFrame):
        cols = df.select_dtypes(include=["number"]).columns.tolist()
        cols = [c for c in cols if c not in self.exclude]
        return cols

    def _sanitize_numeric(self, X_num: pd.DataFrame):
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        if X_num.isnull().any().any():
            raise ValueError(
                "DistributionShapeCorrector: NaNs present in numeric columns. "
                "Run missing-value handling first."
            )
        return X_num

    def _learn_shift_for_nonneg(self, series: pd.Series):
        """
        Learn smallest shift so that (x + shift) >= 0 + epsilon.
        """
        mn = series.min()
        if mn >= 0:
            return 0.0
        return float(-mn + self.epsilon)

    def _learn_shift_for_positive(self, series: pd.Series):
        """
        Learn smallest shift so that (x + shift) > 0 + epsilon.
        Needed for boxcox.
        """
        mn = series.min()
        if mn > 0:
            return 0.0
        # ensure strictly positive
        return float(-mn + self.epsilon)

    def _fit(self, X_train: pd.DataFrame):
        # NEW: if none, don't fit anything
        if self.strategy == "none":
            self._num_cols_ = self._get_numeric_cols(X_train)
            self._shift_map_ = {}
            self._pt_ = None
            return

        df = X_train.copy()
        num_cols = self._get_numeric_cols(df)
        self._num_cols_ = num_cols

        if len(num_cols) == 0:
            self._pt_ = None
            self._shift_map_ = {}
            return

        X_num = self._sanitize_numeric(df[num_cols])

        if self.strategy in ("log1p", "sqrt"):
            # per-column nonnegative shift
            self._shift_map_ = {c: self._learn_shift_for_nonneg(X_num[c]) for c in num_cols}
            self._pt_ = None

        elif self.strategy == "boxcox":
            # shift to strictly positive, then fit PT
            self._shift_map_ = {c: self._learn_shift_for_positive(X_num[c]) for c in num_cols}
            X_shifted = X_num.copy()
            for c in num_cols:
                X_shifted[c] = X_shifted[c] + self._shift_map_[c]
            # fit transformer
            self._pt_ = PowerTransformer(method="box-cox", standardize=self.standardize)
            self._pt_.fit(X_shifted.values)

        elif self.strategy == "yeojohnson":
            self._shift_map_ = {}  # not needed
            self._pt_ = PowerTransformer(method="yeo-johnson", standardize=self.standardize)
            self._pt_.fit(X_num.values)

        else:
            raise RuntimeError("Unexpected strategy in _fit.")

    def _transform_one(self, df: pd.DataFrame) -> pd.DataFrame:
        # NEW: do nothing immediately
        if self.strategy == "none":
            return df.copy()

        out = df.copy()
        num_cols = self._num_cols_ if self._num_cols_ is not None else self._get_numeric_cols(out)
        if len(num_cols) == 0:
            return out

        X_num = self._sanitize_numeric(out[num_cols])

        if self.strategy == "log1p":
            for c in num_cols:
                shift = self._shift_map_.get(c, 0.0)
                out[c] = np.log1p(X_num[c] + shift)

        elif self.strategy == "sqrt":
            for c in num_cols:
                shift = self._shift_map_.get(c, 0.0)
                out[c] = np.sqrt(X_num[c] + shift)

        elif self.strategy == "boxcox":
            if self._pt_ is None:
                return out
            X_shifted = X_num.copy()
            for c in num_cols:
                X_shifted[c] = X_shifted[c] + self._shift_map_.get(c, 0.0)
            arr = self._pt_.transform(X_shifted.values)
            out[num_cols] = arr

        elif self.strategy == "yeojohnson":
            if self._pt_ is None:
                return out
            arr = self._pt_.transform(X_num.values)
            out[num_cols] = arr

        elif self.strategy == "none":
            return df.copy()

        else:
            raise RuntimeError("Unexpected strategy in _transform_one.")

        out = out.replace([np.inf, -np.inf], np.nan)

        return out

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Distribution Shape Correction -----")

        if isinstance(self.dataset, dict):
            self._fit(self.dataset["train"].copy())
            out = {"train": self._transform_one(self.dataset["train"].copy())}
            if "test" in self.dataset and self.dataset["test"] is not None:
                out["test"] = self._transform_one(self.dataset["test"].copy())

        elif isinstance(self.dataset, pd.DataFrame):
            self._fit(self.dataset.copy())
            out = self._transform_one(self.dataset.copy())

        else:
            raise TypeError("dataset must be a pandas DataFrame or a dict with keys {'train','test'}.")

        if self.verbose:
            print(f"Distribution shape correction done in {time.time() - start_time:.2f}s")

        return out
