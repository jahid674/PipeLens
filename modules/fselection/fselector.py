from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import pandas as pd
import numpy as np
import time

class FeatureSelector:
    """
    - dataset: input DataFrame (X)
    - strategy: 'none', 'variance', 'mutual_info'
    - threshold: threshold for variance (if using 'variance')
    - top_k: number of top features to select (only used if explicitly provided; otherwise 80% will be used)
    - verbose: whether to print selection details
    - exclude: list of columns to exclude from feature selection
    """

    def __init__(self, dataset, strategy='none', threshold=0.01, top_k=None, verbose=False, exclude=None):
        self.dataset = dataset.copy()
        self.strategy = strategy.lower()
        self.threshold = threshold
        self.top_k = top_k  # can be None, will be set dynamically in mutual_info
        self.verbose = verbose
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []
        self.selected_features = None

        if self.strategy not in ['none', 'variance', 'mutual_info']:
            raise ValueError("Strategy must be one of 'none', 'variance', or 'mutual_info'.")

    def transform(self, y_train=None):
        start_time = time.time()
        if self.verbose:
            print("---------->> Starting Feature Selection <<-----------")

        # --- NEW: drop rows with any missing values first (all columns) ---
        before_na = len(self.dataset)
        na_mask = self.dataset.notna().all(axis=1)
        df_all = self.dataset.loc[na_mask].reset_index(drop=True)
        if y_train is not None:
            y_train = y_train.loc[na_mask].reset_index(drop=True)
        if self.verbose:
            print(f"Dropped {before_na - len(df_all)} rows with missing values.")

        # Store excluded columns separately (from NA-filtered df)
        excluded_cols = df_all[self.exclude] if self.exclude else pd.DataFrame()
        df = df_all.drop(columns=self.exclude, errors='ignore')

        if self.strategy == 'none':
            if self.verbose:
                print("No feature selection applied.")
            selected_df = df
            self.selected_features = df.columns.tolist()

        elif self.strategy == 'variance':
            if self.verbose:
                print(f"Applying Variance Threshold with threshold={self.threshold}")
            selector = VarianceThreshold(threshold=self.threshold)
            selector.fit(df)
            selected_df = df[df.columns[selector.get_support(indices=True)]]
            self.selected_features = selected_df.columns.tolist()

        elif self.strategy == 'mutual_info':
            if y_train is None:
                raise ValueError("y_train must be provided for mutual_info strategy.")
            if self.verbose:
                print("Selecting top features using Mutual Information")

            # Dynamically determine top_k as 80% of total features
            n_features = df.shape[1]
            top_k = int(np.ceil(0.8 * n_features)) if self.top_k is None else self.top_k

            scores = mutual_info_classif(df, y_train, discrete_features='auto')
            selected_indices = np.argsort(scores)[-top_k:]
            selected_df = df.iloc[:, selected_indices]
            self.selected_features = selected_df.columns.tolist()

        # Add excluded columns back
        if not excluded_cols.empty:
            selected_df = pd.concat([selected_df, excluded_cols], axis=1)
            selected_df = selected_df[self.dataset.columns.intersection(selected_df.columns)]

        if self.verbose:
            print(f"Selected features: {self.selected_features}")
            print(f"Feature selection completed in {time.time() - start_time:.2f} seconds.\n")

        return selected_df
