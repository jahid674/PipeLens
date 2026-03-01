#!/usr/bin/env python3
# coding: utf-8
# Author: (adapted to Learn2Clean-compatible format)

import warnings
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA, KernelPCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class PolyPCATransformer():
    """
    Polynomial feature expansion (numeric cols) + optional dimensionality reduction (Learn2Clean compatible).

    Reducer strategies (parameter name: reducer):
      - "NONE"               : SKIP THIS MODULE (identity; return original data unchanged)
      - "PCA"                : PolynomialFeatures -> PCA
      - "SPARSEPCA"          : PolynomialFeatures -> SparsePCA
      - "MINIBATCHSPARSEPCA" : PolynomialFeatures -> MiniBatchSparsePCA
      - "KERNELPCA"          : PolynomialFeatures -> KernelPCA

    Notes:
      - For reducer != "NONE": numeric-only transformation; can keep non-numeric columns.
      - Requires missing values handled upstream (raises if NaNs/inf remain in numeric data).
      - Fits on TRAIN only; transforms train/test consistently.
      - `exclude` columns are preserved and appended back.
    """

    def __init__(self, dataset,
                 degree=2, include_bias=False, interaction_only=False,
                 reducer="NONE", n_components=None,
                 kernel="rbf", gamma=None, coef0=1.0, kpca_fit_inverse_transform=False,
                 alpha=1.0, ridge_alpha=0.01, batch_size=256, max_iter=1000, tol=1e-3,
                 random_state=42, verbose=False, exclude=None, keep_non_numeric=True,
                 threshold=None):

        self.dataset = dataset

        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)

        self.reducer = str(reducer).upper().strip() if reducer is not None else "NONE"
        self.n_components = None if n_components is None else int(n_components)

        self.kernel = str(kernel)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.kpca_fit_inverse_transform = bool(kpca_fit_inverse_transform)

        self.alpha = float(alpha)
        self.ridge_alpha = float(ridge_alpha)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.exclude = exclude  # str or list; handled internally
        self.keep_non_numeric = bool(keep_non_numeric)
        self.threshold = threshold  # unused; API compatibility

        if self.reducer not in ("NONE", "PCA", "SPARSEPCA", "MINIBATCHSPARSEPCA", "KERNELPCA"):
            raise ValueError("Invalid reducer. Choose from "
                             "{'NONE','PCA','SPARSEPCA','MINIBATCHSPARSEPCA','KERNELPCA'}.")

        # fitted artifacts (only used if reducer != "NONE")
        self._poly = None
        self._reducer_model = None
        self._numeric_cols_ = None
        self._poly_feature_names_ = None

    # ------------------- Learn2Clean API -------------------

    def get_params(self, deep=True):
        return {
            'degree': self.degree,
            'include_bias': self.include_bias,
            'interaction_only': self.interaction_only,
            'reducer': self.reducer,
            'n_components': self.n_components,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'kpca_fit_inverse_transform': self.kpca_fit_inverse_transform,
            'alpha': self.alpha,
            'ridge_alpha': self.ridge_alpha,
            'batch_size': self.batch_size,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'exclude': self.exclude,
            'keep_non_numeric': self.keep_non_numeric,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for PolyPCATransformer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`poly_pca_transformer.get_params().keys()`")
            else:
                setattr(self, k, v)

        self.reducer = str(self.reducer).upper().strip() if self.reducer is not None else "NONE"

    # ------------------- helpers -------------------

    def _exclude_list(self):
        if self.exclude is None:
            return []
        if isinstance(self.exclude, list):
            return [c for c in self.exclude if c is not None]
        return [self.exclude]

    def _split_cols(self, df: pd.DataFrame):
        """
        Return:
          numeric_df, non_numeric_df, excluded_df, numeric_cols, original_columns
        """
        original_columns = list(df.columns)
        df_work = df.copy()

        excl = self._exclude_list()
        excluded_df = df_work[excl].copy() if len(excl) > 0 and all(c in df_work.columns for c in excl) else pd.DataFrame(index=df_work.index)
        df_work = df_work.drop(columns=excl, errors="ignore")

        numeric_cols = df_work.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = [c for c in df_work.columns if c not in numeric_cols]

        numeric_df = df_work[numeric_cols].copy()
        non_numeric_df = df_work[non_numeric_cols].copy()
        return numeric_df, non_numeric_df, excluded_df, numeric_cols, original_columns

    def _sanitize_numeric(self, numeric_df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        if numeric_df.isnull().any().any():
            raise ValueError(
                "PolyPCATransformer: numeric inputs contain NaN/inf after sanitation. "
                "Run missing-value handling before this module."
            )
        return numeric_df

    def _build_reducer(self):
        # reducer != "NONE" is enforced by callers
        if self.reducer == "PCA":
            return PCA(n_components=self.n_components, random_state=self.random_state)

        if self.reducer == "SPARSEPCA":
            return SparsePCA(
                n_components=self.n_components,
                alpha=self.alpha,
                ridge_alpha=self.ridge_alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )

        if self.reducer == "MINIBATCHSPARSEPCA":
            return MiniBatchSparsePCA(
                n_components=self.n_components,
                alpha=self.alpha,
                ridge_alpha=self.ridge_alpha,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )

        if self.reducer == "KERNELPCA":
            # sklearn compatibility: KernelPCA may not accept random_state in some versions
            try:
                return KernelPCA(
                    n_components=self.n_components,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    coef0=self.coef0,
                    fit_inverse_transform=self.kpca_fit_inverse_transform,
                    random_state=self.random_state,
                )
            except TypeError:
                return KernelPCA(
                    n_components=self.n_components,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    coef0=self.coef0,
                    fit_inverse_transform=self.kpca_fit_inverse_transform,
                )

        raise RuntimeError("Unexpected reducer in _build_reducer().")

    def _fit_on_train(self, X_train: pd.DataFrame):
        numeric_train, _, _, num_cols, _ = self._split_cols(X_train)
        self._numeric_cols_ = num_cols

        if len(num_cols) == 0:
            self._poly = None
            self._reducer_model = None
            self._poly_feature_names_ = None
            return

        numeric_train = self._sanitize_numeric(numeric_train)

        self._poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only,
        )

        Z_train = self._poly.fit_transform(numeric_train.values.astype(float))

        try:
            self._poly_feature_names_ = self._poly.get_feature_names_out(self._numeric_cols_)
        except Exception:
            self._poly_feature_names_ = np.array([f"poly_{i}" for i in range(Z_train.shape[1])])

        self._reducer_model = self._build_reducer()
        self._reducer_model.fit(Z_train)

    def _transform_one(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_df, non_numeric_df, excluded_df, num_cols, _ = self._split_cols(X)

        if len(num_cols) == 0 or self._poly is None or self._reducer_model is None:
            # no numeric features -> return X (or X with excluded columns already in place)
            return X.copy()

        numeric_df = self._sanitize_numeric(numeric_df)

        Z = self._poly.transform(numeric_df.values.astype(float))
        Z_out = self._reducer_model.transform(Z)

        if self.reducer == "PCA":
            prefix = "pca"
        elif self.reducer == "SPARSEPCA":
            prefix = "spca"
        elif self.reducer == "MINIBATCHSPARSEPCA":
            prefix = "mbspca"
        else:
            prefix = f"kpca_{self.kernel}"

        out_cols = [f"{prefix}_{i}" for i in range(Z_out.shape[1])]
        out_df = pd.DataFrame(Z_out, columns=out_cols, index=X.index)

        if self.keep_non_numeric and non_numeric_df.shape[1] > 0:
            out_df = pd.concat([out_df, non_numeric_df], axis=1)

        if excluded_df is not None and excluded_df.shape[1] > 0:
            out_df = pd.concat([out_df, excluded_df], axis=1)

        return out_df

    # ------------------- driver -------------------

    def transform(self):

        start_time = time.time()
        outd = self.dataset

        print(">>Polynomial + reducer feature engineering ")

        for key in ['train']:

            if (isinstance(self.dataset, dict)
                    and key in self.dataset
                    and (not isinstance(self.dataset[key], dict))):

                d = self.dataset[key].copy()
                print("* For", key, "dataset")

                # identity skip
                if self.reducer == "NONE":
                    if self.verbose:
                        print("Reducer='NONE' => skipping PolyPCATransformer (identity).")
                    outd[key] = d

                    if "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                        outd["test"] = self.dataset["test"].copy()

                    continue

                # fit on train once, then transform
                self._fit_on_train(d)
                outd[key] = self._transform_one(d)

                if "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                    outd["test"] = self._transform_one(self.dataset["test"].copy())

            else:
                print("No", key, "dataset, no PolyPCA transformation")

        print("PolyPCA feature engineering done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return outd
