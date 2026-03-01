import time
import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

class Normalizer:
    def __init__(self, dataset, strategy='standard', verbose=False, exclude=None):
        """
        Parameters:
        - dataset: input DataFrame (e.g., X_train)
        - strategy: one of 'none', 'ss', 'rb', 'ma', 'mm' (standard scaler, robust scaler, max absolute scaler, minmax scaler)
        - verbose: if True, prints logs
        - exclude: list of columns to exclude from normalization
        """
        self.dataset = dataset.copy()
        self.strategy = strategy.lower()
        self.verbose = verbose
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

    def _select_scaler(self):
        if self.strategy == 'ss':
            return StandardScaler()
        elif self.strategy == 'rs':
            return RobustScaler()
        elif self.strategy == 'ma':
            return MaxAbsScaler()
        elif self.strategy == 'mm':
            return MinMaxScaler()
        elif self.strategy == 'none':
            return None
        else:
            raise ValueError("Invalid strategy. Choose from 'none', 'standard', 'robust', 'maxabs', 'minmax'.")

    def transform(self):
        start_time = time.time()

        if self.verbose:
            print("---------->> Started Normalization <<-----------")

        df = self.dataset.copy()

        # --- NEW: drop rows with any missing values first (all columns) ---

        excluded_cols = df[self.exclude] if self.exclude else pd.DataFrame()
        df = df.drop(columns=self.exclude, errors='ignore')

        scaler = self._select_scaler()

        if scaler is not None:
            df_scaled = scaler.fit(df).transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)
            if self.verbose:
                print(f"Normalization applied with strategy: {self.strategy}.")
        else:
            df = df.copy()
            if self.verbose:
                print("No normalization applied (strategy='none').")

        if not excluded_cols.empty:
            df = pd.concat([df.reset_index(drop=True), excluded_cols.reset_index(drop=True)], axis=1)
            df = df[self.dataset.columns]

        if self.verbose:
            print(f"Normalization done -- time: {time.time() - start_time:.2f} seconds\n")

        return df
