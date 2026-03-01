#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille

import warnings
import time
import numpy as np
import pandas as pd
from fancyimpute import KNN

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class Imputer():
    """
    Replace or remove the missing values using a particular strategy
    """

    def __init__(self, dataset, strategy='DROP', verbose=False,
                 exclude=None, threshold=None):

        self.dataset = dataset
        self.strategy = strategy
        self.verbose = verbose
        self.threshold = threshold
        self.exclude = exclude  # to implement

    def get_params(self, deep=True):
        return {'strategy': self.strategy,
                'verbose': self.verbose,
                'exclude': self.exclude,
                'threshold': self.threshold}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`imputer.get_params().keys()`")
            else:
                setattr(self, k, v)

    # ---------- helpers: numeric vs other (preserve all non-numeric) ----------

    def _split_numeric(self, df: pd.DataFrame):
        """Return numeric X (optionally drop exclude if numeric) and O = all other cols."""
        X = df.select_dtypes(include=['number']).copy()
        if self.exclude in X.columns:
            X = X.drop(columns=[self.exclude])
        other_cols = df.columns.difference(X.columns)
        O = df[other_cols].copy()
        return X, O

    def _recombine(self, X_new: pd.DataFrame, O: pd.DataFrame, original: pd.DataFrame):
        """
        Recombine imputed numeric X_new with untouched O.
        Ensure original column order is preserved.
        """
        df = pd.concat([X_new, O], axis=1)
        # Reorder to original columns
        df = df.reindex(columns=original.columns)
        return df

    # ---------------------------- strategies ---------------------------- #

    def mean_imputation(self, dataset):
        # numeric only → fill with column mean
        df = dataset.copy()
        X, O = self._split_numeric(df)

        if X.isnull().sum().sum() > 0:
            for c in X.columns:
                mean_val = X[c].mean()
                X[c] = X[c].fillna(mean_val)
            df = self._recombine(X, O, dataset)
        return df

    def median_imputation(self, dataset):
        # numeric only → fill with column median
        df = dataset.copy()
        X, O = self._split_numeric(df)

        if X.isnull().sum().sum() > 0:
            for c in X.columns:
                med_val = X[c].median()
                X[c] = X[c].fillna(med_val)
            df = self._recombine(X, O, dataset)
        return df

    def NaN_drop(self, dataset):
        # drop any row with any NaN in any column
        print("Dataset size reduced from", len(dataset), "to", len(dataset.dropna()))
        return dataset.dropna()

    def MF_most_frequent_imputation(self, dataset):
        # for both categorical and numerical data
        # replace missing values by the most frequent value per column
        df = dataset.copy()
        for col in df.columns:
            s = df[col]
            if s.isnull().any():
                # handle all-NaN safely
                if s.dropna().empty:
                    continue
                mfv = s.dropna().value_counts().idxmax()
                df[col] = s.fillna(mfv)
                if self.verbose:
                    print("Most frequent value for ", col, "is:", mfv)
        return df

    def NaN_random_replace(self, dataset):
        # WARNING: original code overwrote all values with random numbers via update().
        # Keeping behavior but only for NaN cells (safer): fill NaNs with random draws.
        df = dataset.copy()
        nan_mask = df.isna()
        if nan_mask.values.any():
            rand = pd.DataFrame(
                np.random.randn(*df.shape), columns=df.columns, index=df.index
            )
            df = df.where(~nan_mask, rand)  # replace only NaNs
        return df

    def KNN_imputation(self, dataset, k):
        # numeric only → KNN impute, then recombine with untouched non-numeric
        df = dataset.copy()
        X, O = self._split_numeric(df)

        if X.isnull().sum().sum() > 0:
            # fancyimpute expects numpy array; keep index/columns to rebuild DataFrame
            X_imputed = KNN(k=k, verbose=False).fit_transform(X.to_numpy())
            X_new = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
            df = self._recombine(X_new, O, dataset)
        return df

    def MICE_imputation(self, dataset):
        # numeric only → MICE, then recombine with untouched non-numeric
        import impyute as imp
        df = dataset.copy()
        X, O = self._split_numeric(df)

        if X.isnull().sum().sum() > 0:
            X_imputed = imp.mice(X.to_numpy())
            X_new = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
            df = self._recombine(X_new, O, dataset)
        return df

    def EM_imputation(self, dataset):
        # numeric only → EM, then recombine with untouched non-numeric
        import impyute as imp
        df = dataset.copy()
        X, O = self._split_numeric(df)

        if X.isnull().sum().sum() > 0:
            X_imputed = imp.em(X.to_numpy())
            X_new = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
            df = self._recombine(X_new, O, dataset)
        return df

    # ------------------------------ driver ------------------------------ #

    def transform(self):

        start_time = time.time()
        print(">>Imputation ")

        impd = self.dataset

        for key in ['train']:
            if (not isinstance(self.dataset[key], dict)):

                d = self.dataset[key].copy()
                print("* For", key, "dataset")

                total_missing_before = d.isnull().sum().sum()
                Num_missing_before = d.select_dtypes(
                    include=['number']).isnull().sum().sum()
                NNum_missing_before = d.select_dtypes(
                    exclude=['number']).isnull().sum().sum()

                print("Before imputation:")
                if total_missing_before == 0:
                    print("No missing values in the given data")
                    impd[key] = d
                    continue
                else:
                    print("Total", total_missing_before, "missing values in",
                          d.columns[d.isnull().any()].tolist())
                    if Num_missing_before > 0:
                        print("-", Num_missing_before, "numerical missing values in",
                              d.select_dtypes(['number']).columns[
                                  d.select_dtypes(['number']).isnull().any()
                              ].tolist())
                    if NNum_missing_before > 0:
                        print("-", NNum_missing_before, "non-numerical missing values in",
                              d.select_dtypes(exclude=['number']).columns[
                                  d.select_dtypes(exclude=['number']).isnull().any()
                              ].tolist())

                    if (self.strategy == "EM"):
                        dn = self.EM_imputation(d)
                    elif (self.strategy == "MICE"):
                        dn = self.MICE_imputation(d)
                    elif (self.strategy == "KNN_1"):
                        dn = self.KNN_imputation(d, k=1)
                    elif (self.strategy == "KNN_5"):
                        dn = self.KNN_imputation(d, k=5)
                    elif (self.strategy == "KNN_10"):
                        dn = self.KNN_imputation(d, k=10)
                    elif (self.strategy == "KNN_20"):
                        dn = self.KNN_imputation(d, k=20)
                    elif (self.strategy == "KNN_30"):
                        dn = self.KNN_imputation(d, k=30)
                    elif (self.strategy == "RAND"):
                        dn = self.NaN_random_replace(d)
                    elif (self.strategy == "MF"):
                        dn = self.MF_most_frequent_imputation(d)
                    elif (self.strategy == "MEAN"):
                        dn = self.mean_imputation(d)
                    elif (self.strategy == "MEDIAN"):
                        dn = self.median_imputation(d)
                    elif (self.strategy == "DROP"):
                        dn = self.NaN_drop(d)
                    else:
                        raise ValueError("Strategy invalid. Please choose between "
                                         "'EM', 'MICE', 'KNN', 'RAND', 'MF', "
                                         "'MEAN', 'MEDIAN', or 'DROP'")

                    impd[key] = dn

                    print("After imputation:")
                    print("Total", impd[key].isnull().sum().sum(), "missing values")
                    print("-", impd[key].select_dtypes(include=['number']
                                                       ).isnull().sum().sum(),
                          "numerical missing values")
                    print("-", impd[key].select_dtypes(exclude=['number']
                                                       ).isnull().sum().sum(),
                          "non-numerical missing values")
            else:
                print("No", key, "dataset, no imputation")

        print("Imputation done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return impd
