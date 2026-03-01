import pandas as pd
import numpy as np
np.random.seed(42)
import time


class VIFMulticollinearityCleaner:
    """
    Multicollinearity Detection & Repair using VIF.

    Strategies:
      - "none"          : do nothing
      - "drop_high_vif" : iteratively drop features with VIF > threshold

    Notes:
      - Requires numeric features with no NaNs (run imputer first)
      - Keeps all non-numeric columns
    """

    def __init__(
        self,
        dataset,
        strategy="drop_high_vif",
        vif_threshold=10.0,
        max_iter=50,
        min_features=2,
        standardize=True,
        verbose=False,
        exclude=None,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.strategy = str(strategy).lower().strip()
        self.vif_threshold = float(vif_threshold)
        self.max_iter = int(max_iter)
        self.min_features = int(min_features)
        self.standardize = bool(standardize)
        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else ([exclude] if exclude else [])

        if self.strategy not in ("none", "drop_high_vif"):
            raise ValueError("Invalid strategy. Choose from {'none','drop_high_vif'}.")

        self.selected_features_ = None
        self.vif_table_ = None

    def _get_numeric_cols(self, df):
        cols = df.select_dtypes(include=["number"]).columns.tolist()
        return [c for c in cols if c not in self.exclude]

    def _standardize(self, X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        return (X - mu) / sd

    def _compute_vif(self, X, cols):
        vifs = []
        n = X.shape[0]

        for j, col in enumerate(cols):
            y = X[:, j]
            X_others = np.delete(X, j, axis=1)
            X_design = np.column_stack([np.ones(n), X_others])
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            y_hat = X_design @ beta

            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)

            r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
            r2 = min(max(r2, 0.0), 0.999999999)
            vifs.append(1.0 / (1.0 - r2))

        return pd.DataFrame({"feature": cols, "vif": vifs}).sort_values("vif", ascending=False)

    def _fit(self, df):
        num_cols = self._get_numeric_cols(df)

        if self.strategy == "none" or len(num_cols) < self.min_features:
            self.selected_features_ = num_cols
            self.vif_table_ = None
            return

        X = df[num_cols].replace([np.inf, -np.inf], np.nan)
        if X.isnull().any().any():
            raise ValueError("VIF requires no NaNs. Run imputation first.")

        keep = list(num_cols)
        for _ in range(self.max_iter):
            if len(keep) < self.min_features:
                break

            mat = X[keep].values.astype(float)
            if self.standardize:
                mat = self._standardize(mat)

            vif_df = self._compute_vif(mat, keep)
            self.vif_table_ = vif_df

            if vif_df.iloc[0]["vif"] <= self.vif_threshold:
                break

            keep.remove(vif_df.iloc[0]["feature"])

        self.selected_features_ = keep

    def _transform_one(self, df):
        if self.strategy == "none":
            return df.copy()

        num_cols = self._get_numeric_cols(df)
        keep_cols = [c for c in df.columns if c not in num_cols] + self.selected_features_
        return df[keep_cols]

    def transform(self, y_train=None, sensitive_attr_train=None):
        start = time.time()
        

        if isinstance(self.dataset, dict):
            self._fit(self.dataset["train"])
            out = {"train": self._transform_one(self.dataset["train"])}
            if "test" in self.dataset:
                out["test"] = self._transform_one(self.dataset["test"])
        else:
            self._fit(self.dataset)
            out = self._transform_one(self.dataset)

        if self.verbose:
            print(f"[VIF] done in {time.time() - start:.2f}s")

        return out
