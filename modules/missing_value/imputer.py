import pandas as pd
import numpy as np
import time
from sklearn.impute import SimpleImputer, KNNImputer

class Imputer:
    def __init__(self, dataset, strategy='mean', k=None, verbose=False, exclude=None):
        """
        Parameters:
        - dataset: dictionary containing 'train' and 'test' datasets
        - strategy: 'drop', 'mean', 'median', 'most_frequent', 'knn'
        - k: for 'knn' strategy, k must be in {1, 5, 10, 15, 20}
        - verbose: if True, print detailed information
        - exclude: list of columns to exclude from imputation (not applicable in our case)
        """
        self.dataset = dataset.copy()
        self.strategy = strategy.lower()
        self.k = k
        self.verbose = verbose
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

    def drop_missing(self, df, y_train=None, sensitive_attr_train=None):
        df_updated = df.copy()
        missing_idx = df_updated[df_updated.isnull().any(axis=1)].index.tolist()

        df_updated = df_updated.drop(missing_idx)

        if y_train is not None:
            y_train_updated = y_train.copy()
            for idx in sorted(missing_idx, reverse=True):
                    del y_train_updated[idx]

        if sensitive_attr_train is not None:
            updated_sensitive_attr_train = sensitive_attr_train.drop(missing_idx)
            updated_sensitive_attr_train.reset_index(drop=True, inplace=True)


        return df_updated, y_train_updated, updated_sensitive_attr_train

    def mean_imputer(self, df):
        df_updated = df.copy()
        numeric_cols = df_updated.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            mean_val = df_updated[col].mean()
            df[col] = df_updated[col].fillna(mean_val)
        return df_updated
    
    def median_imputer(self, df):
        df_updated = df.copy()
        numeric_cols = df_updated.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            median_val = df_updated[col].median()
            df_updated[col] = df_updated[col].fillna(median_val)
        return df_updated

    def most_frequent_imputer(self, df):
        df_updated = df.copy()
        for col in df_updated.columns:
            if df_updated[col].isnull().sum() > 0:
                mode_val = df_updated[col].mode()
                if not mode_val.empty:
                    df_updated[col] = df_updated[col].fillna(mode_val[0])
        return df_updated

    def knn_imputer(self, df):
        df_updated = df.copy()
        numeric_data = df_updated.select_dtypes(include=['number'])

        if numeric_data.isnull().sum().sum() > 0:
            imputer = KNNImputer(n_neighbors=self.k)
            imputed_array = imputer.fit_transform(numeric_data)
            numeric_imputed = pd.DataFrame(imputed_array, columns=numeric_data.columns, index=numeric_data.index)

            non_numeric = df_updated.select_dtypes(exclude=['number'])
            df_updated = pd.concat([numeric_imputed, non_numeric], axis=1)
            df_updated = df_updated[df_updated.columns]

        return df_updated

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Missing Value Imputation -----")

        df = self.dataset.copy()

        excluded_cols = df[self.exclude] if self.exclude else pd.DataFrame()
        df = df.drop(columns=self.exclude, errors='ignore')

        if self.verbose:
            print("Missing values before imputation:", df.isnull().sum().sum())

        if self.strategy == 'drop':
            df, y_train, sensitive_attr_train = self.drop_missing(df, y_train, sensitive_attr_train)
        elif self.strategy == 'mean':
            df = self.mean_imputer(df)
        elif self.strategy == 'median':
            df = self.median_imputer(df)
        elif self.strategy == 'most_frequent':
            df = self.most_frequent_imputer(df)
        elif self.strategy == 'knn':
            df = self.knn_imputer(df)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Choose from 'drop', 'mean', 'median', 'most_frequent', 'knn'.")

        if not excluded_cols.empty:
            df = pd.concat([df.reset_index(drop=True), excluded_cols.reset_index(drop=True)], axis=1)
            df = df[self.dataset.columns]

        if self.verbose:
            print("Missing values after imputation:", df.isnull().sum().sum())
            print(f"\nImputation completed in {time.time() - start_time:.2f} seconds.\n")

        if self.strategy == 'drop':
            return df, y_train, sensitive_attr_train
        else:
            return df
