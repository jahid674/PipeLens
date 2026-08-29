# modules/feature_engineering/poly_pca.py

import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA, KernelPCA


class PolyPCATransformer:
    """
    Polynomial feature expansion (numeric cols) + optional dimensionality reduction.

    reducer strategies:
      - "none"               : SKIP THIS MODULE (identity; return original data unchanged)
      - "pca"                : PolynomialFeatures -> PCA
      - "sparsepca"          : PolynomialFeatures -> SparsePCA
      - "minibatchsparsepca" : PolynomialFeatures -> MiniBatchSparsePCA
      - "kernelpca"          : PolynomialFeatures -> KernelPCA

    Input:
      - DataFrame OR dict {"train": df_train, "test": df_test}
    Output:
      - DataFrame OR dict {"train","test"} with transformed data

    Notes:
      - For reducers != "none": numeric-only transformation; can keep non-numeric columns.
      - Requires missing values handled upstream (raises if NaNs remain in numeric data).
    """

    def __init__(
        self,
        dataset,
        degree=2,
        include_bias=False,
        interaction_only=False,

        reducer="none",          # "none" => identity skip
        n_components=None,

        # KernelPCA options
        kernel="rbf",
        gamma=None,
        coef0=1.0,
        kpca_fit_inverse_transform=False,

        # SparsePCA / MiniBatchSparsePCA options
        alpha=1.0,
        ridge_alpha=0.01,
        batch_size=256,
        max_iter=1000,
        tol=1e-3,

        random_state=42,
        verbose=False,
        exclude=None,
        keep_non_numeric=True,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()

        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)

        self.reducer = str(reducer).lower().strip() if reducer is not None else "none"
        self.n_components = n_components if n_components is None else int(n_components)

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
        self.exclude = exclude if isinstance(exclude, list) else ([exclude] if exclude else [])
        self.keep_non_numeric = bool(keep_non_numeric)

        if self.reducer not in ("none", "pca", "sparsepca", "minibatchsparsepca", "kernelpca"):
            raise ValueError(
                "Invalid reducer. Choose from "
                "{'none','pca','sparsepca','minibatchsparsepca','kernelpca'}."
            )

        # fitted artifacts (only used if reducer != "none")
        self._poly = None
        self._reducer_model = None
        self._numeric_cols_ = None
        self._poly_feature_names_ = None

    # ---------------- helpers ----------------
    def _split_cols(self, df: pd.DataFrame):
        df_work = df.copy()

        excluded_cols = df_work[self.exclude].copy() if self.exclude else pd.DataFrame(index=df_work.index)
        df_work = df_work.drop(columns=self.exclude, errors="ignore")

        numeric_cols = df_work.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = [c for c in df_work.columns if c not in numeric_cols]

        numeric_df = df_work[numeric_cols].copy()
        non_numeric_df = df_work[non_numeric_cols].copy()
        return numeric_df, non_numeric_df, excluded_cols, numeric_cols

    def _sanitize_numeric(self, numeric_df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        if numeric_df.isnull().any().any():
            raise ValueError(
                "PolyPCATransformer: numeric inputs contain NaN/inf after sanitation. "
                "Run missing-value handling before this module."
            )
        return numeric_df

    def _build_reducer(self):
        # reducer != "none" is enforced by callers
        if self.reducer == "pca":
            return PCA(n_components=self.n_components, random_state=self.random_state)

        if self.reducer == "sparsepca":
            return SparsePCA(
                n_components=self.n_components,
                alpha=self.alpha,
                ridge_alpha=self.ridge_alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )

        if self.reducer == "minibatchsparsepca":
            return MiniBatchSparsePCA(
                n_components=self.n_components,
                alpha=self.alpha,
                ridge_alpha=self.ridge_alpha,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )

        if self.reducer == "kernelpca":
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

    def _fit(self, X_train: pd.DataFrame):
        numeric_train, _, _, num_cols = self._split_cols(X_train)
        self._numeric_cols_ = num_cols

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
        numeric_df, non_numeric_df, excluded_cols, _ = self._split_cols(X)
        numeric_df = self._sanitize_numeric(numeric_df)

        Z = self._poly.transform(numeric_df.values.astype(float))
        Z_out = self._reducer_model.transform(Z)

        if self.reducer == "pca":
            prefix = "pca"
        elif self.reducer == "sparsepca":
            prefix = "spca"
        elif self.reducer == "minibatchsparsepca":
            prefix = "mbspca"
        else:
            prefix = f"kpca_{self.kernel}"

        out_cols = [f"{prefix}_{i}" for i in range(Z_out.shape[1])]
        out_df = pd.DataFrame(Z_out, columns=out_cols, index=X.index)

        if self.keep_non_numeric and non_numeric_df.shape[1] > 0:
            out_df = pd.concat([out_df, non_numeric_df], axis=1)

        if excluded_cols is not None and excluded_cols.shape[1] > 0:
            out_df = pd.concat([out_df, excluded_cols], axis=1)

        return out_df

    # ---------------- main API ----------------
    def transform(self, y_train=None, sensitive_attr_train=None):
        """
        If reducer == "none": identity transform (returns original dataset unchanged).
        Otherwise: fit on train then transform.
        """
        start_time = time.time()
        if self.verbose:
            print("----- Starting Polynomial + Reducer Feature Engineering -----")

        # ✅ identity mode
        if self.reducer == "none":
            if self.verbose:
                print("PolyPCATransformer reducer='none' => skipping module (identity).")
            if isinstance(self.dataset, dict):
                out = {"train": self.dataset["train"].copy()}
                if "test" in self.dataset and self.dataset["test"] is not None:
                    out["test"] = self.dataset["test"].copy()
                return out
            elif isinstance(self.dataset, pd.DataFrame):
                return self.dataset.copy()
            else:
                raise TypeError("dataset must be a pandas DataFrame or a dict with keys {'train','test'}.")

        # ✅ normal mode
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
            print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds.")

        return out
