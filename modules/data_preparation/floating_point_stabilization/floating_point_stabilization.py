import pandas as pd
import numpy as np
np.random.seed(42)
import time


class FloatingPointStabilizer:
    def __init__(self, dataset, strategy="both", decimals=6, tol=1e-8, verbose=False, exclude=None):
        """
        Parameters:
        - dataset: DataFrame OR dict containing {'train','test'}
        - strategy: 'none', 'snap', 'round', 'both'
        - decimals: used for 'round' and 'both'
        - tol: used for 'snap' and 'both'
        - verbose: prints diagnostics
        - exclude: list of columns to exclude
        """
        self.dataset = dataset.copy()
        self.strategy = str(strategy).lower().strip()
        self.decimals = int(decimals)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

        if self.strategy not in ("none", "snap", "round", "both"):
            raise ValueError(f"Invalid strategy: {self.strategy}. Choose from 'none','snap','round','both'.")

    def _apply_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # true no-op
        if self.strategy == "none":
            return df.copy()

        df_updated = df.copy()

        numeric_cols = df_updated.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in self.exclude]

        if len(numeric_cols) == 0:
            return df_updated

        # Replace inf with NaN for safety (do not fill NaNs here; MV module should handle)
        X = df_updated[numeric_cols].replace([np.inf, -np.inf], np.nan)

        if self.strategy in ("snap", "both"):
            X = X.mask(X.abs() < self.tol, 0.0)

        if self.strategy in ("round", "both"):
            X = X.round(self.decimals)

        df_updated[numeric_cols] = X
        return df_updated

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Floating Point Stabilization -----")

        # Accept either DataFrame or dict({'train','test'})
        if isinstance(self.dataset, pd.DataFrame):
            out = self._apply_to_df(self.dataset.copy())

        elif isinstance(self.dataset, dict):
            out = {"train": self._apply_to_df(self.dataset["train"].copy())}
            if "test" in self.dataset and self.dataset["test"] is not None:
                out["test"] = self._apply_to_df(self.dataset["test"].copy())
        else:
            raise TypeError("dataset must be a pandas DataFrame or dict with keys {'train','test'}.")

        if self.verbose:
            print(f"Floating point stabilization completed in {time.time() - start_time:.2f} seconds.")

        return out
