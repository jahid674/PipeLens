# modules/feature_engineering/nonlinear_transform.py

import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import QuantileTransformer, PowerTransformer


class NonLinearTransformer:
    """
    Non-linear numeric feature transformation.

    Strategies:
      - "none"     : no-op
      - "quantile" : QuantileTransformer (fixed safe settings)
      - "power"    : PowerTransformer using Yeo-Johnson (fixed safe settings)

    Notes:
      * Fits on train and transforms both train/test if dataset is a dict {"train","test"}.
      * Applies ONLY to numeric columns; keeps non-numeric columns if keep_non_numeric=True.
      * Excludes columns listed in `exclude`.
      * Expects missing values already handled (raises if NaNs remain after sanitation).
    """

    def __init__(
        self,
        dataset,
        strategy="none",

        # quantile (fixed defaults; still configurable if you want later)
        n_quantiles=1000,
        output_distribution="normal",
        subsample=1_000_000,
        ignore_implicit_zeros=False,
        random_state=42,

        # power (fixed safe variant)
        power_method="yeo-johnson",
        standardize=True,

        # general
        verbose=False,
        exclude=None,
        keep_non_numeric=True,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.strategy = str(strategy).lower().strip()

        self.n_quantiles = int(n_quantiles)
        self.output_distribution = str(output_distribution)
        self.subsample = int(subsample)
        self.ignore_implicit_zeros = bool(ignore_implicit_zeros)
        self.random_state = int(random_state)

        self.power_method = str(power_method).lower().strip()
        self.standardize = bool(standardize)

        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else ([exclude] if exclude else [])
        self.keep_non_numeric = bool(keep_non_numeric)

        self._model = None
        self._numeric_cols_ = None

    def _split_cols(self, df: pd.DataFrame):
        df_work = df.copy()

        excluded_cols = df_work[self.exclude].copy() if self.exclude else pd.DataFrame(index=df_work.index)
        df_work = df_work.drop(columns=self.exclude, errors="ignore")

        numeric_cols = df_work.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = [c for c in df_work.columns if c not in numeric_cols]

        numeric_df = df_work[numeric_cols].copy()
        non_numeric_df = df_work[non_numeric_cols].copy()
        return numeric_df, non_numeric_df, excluded_cols, numeric_cols

    def _sanity(self, numeric_df: pd.DataFrame):
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        if numeric_df.isnull().any().any():
            raise ValueError(
                "NonLinearTransformer: numeric inputs contain NaN/inf after sanitation. "
                "Run missing-value handling before this module."
            )
        return numeric_df

    def _build_model(self):
        if self.strategy in ("none", "no", "noop"):
            return None

        if self.strategy == "quantile":
            return QuantileTransformer(
                n_quantiles=self.n_quantiles,
                output_distribution=self.output_distribution,
                subsample=self.subsample,
                ignore_implicit_zeros=self.ignore_implicit_zeros,
                random_state=self.random_state,
                copy=True,
            )

        if self.strategy == "power":
            # Use Yeo-Johnson by default (works with zeros/negatives too)
            return PowerTransformer(
                method=self.power_method,      # "yeo-johnson"
                standardize=self.standardize,
                copy=True,
            )

        raise ValueError("Invalid strategy. Use one of {'none','quantile','power'}.")

    def _fit(self, X_train: pd.DataFrame):
        numeric_train, _, _, num_cols = self._split_cols(X_train)
        self._numeric_cols_ = num_cols
        numeric_train = self._sanity(numeric_train)

        self._model = self._build_model()
        if self._model is None or len(self._numeric_cols_) == 0:
            return

        self._model.fit(numeric_train.values)

    def _transform_one(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_df, non_numeric_df, excluded_cols, num_cols = self._split_cols(X)
        numeric_df = self._sanity(numeric_df)

        if self._model is None or len(num_cols) == 0:
            out_numeric = numeric_df.copy()
        else:
            arr = self._model.transform(numeric_df.values)
            out_numeric = pd.DataFrame(arr, columns=num_cols, index=X.index)

        out_df = out_numeric
        if self.keep_non_numeric and non_numeric_df.shape[1] > 0:
            out_df = pd.concat([out_df, non_numeric_df], axis=1)

        if excluded_cols is not None and excluded_cols.shape[1] > 0:
            out_df = pd.concat([out_df, excluded_cols], axis=1)

        return out_df

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Non-linear Feature Transformation -----")

        if isinstance(self.dataset, dict):
            X_train = self.dataset["train"].copy()
            self._fit(X_train)

            out = {"train": self._transform_one(self.dataset["train"].copy())}
            if "test" in self.dataset and self.dataset["test"] is not None:
                out["test"] = self._transform_one(self.dataset["test"].copy())
        elif isinstance(self.dataset, pd.DataFrame):
            X_train = self.dataset.copy()
            self._fit(X_train)
            out = self._transform_one(X_train)
        else:
            raise TypeError("dataset must be a pandas DataFrame or a dict with keys {'train','test'}.")

        if self.verbose:
            print(f"Non-linear transformation completed in {time.time() - start_time:.2f}s.")

        return out
