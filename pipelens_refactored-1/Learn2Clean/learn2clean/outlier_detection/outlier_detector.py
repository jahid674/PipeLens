#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille

import warnings
import time
import numpy as np
import pandas as pd


class Outlier_detector():
    """
    Identify and remove outliers using a particular strategy.
    Outlier decisions are computed on numeric columns only, but
    rows are kept/removed on the FULL dataset so all dtypes are preserved.
    """

    def __init__(self, dataset, strategy='ZSB', threshold=0.2,
                 verbose=False, exclude=None):

        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold
        self.verbose = verbose
        self.exclude = exclude  # reserved; not used yet

    def get_params(self, deep=True):
        return {'strategy': self.strategy,
                'threshold': self.threshold,
                'verbose': self.verbose,
                'exclude': self.exclude}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`outlier_detector.get_params().keys()`")
            else:
                setattr(self, k, v)

    # ---------- helpers ----------

    def _numeric_view(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return numeric-only view for detection (optionally excluding a target)."""
        X = df.select_dtypes(include=['number']).copy()
        if self.exclude in X.columns:
            X = X.drop(columns=[self.exclude])
        return X

    def _apply_row_mask(self, df: pd.DataFrame, keep_mask: pd.Series) -> pd.DataFrame:
        """Keep rows in original df according to keep_mask (aligned by index)."""
        out = df.loc[keep_mask.index[keep_mask]].copy()
        return out

    # ---------- ZSB (robust z-score) ----------

    def ZSB_outlier_detection(self, dataset, threshold):
        X = self._numeric_view(dataset)

        if X.shape[1] < 1:
            print("Error: Need at least one numeric variable for ZSB "
                  "outlier detection\nDataset unchanged")
            return dataset

        median = X.median(axis=0)
        mad = 1.4296 * (X.sub(median)).abs().median(axis=0)

        # Avoid div-by-zero: where MAD==0, mark z as 0
        denom = mad.replace(0, np.nan)
        z = (X.sub(median)).div(denom)
        z = z.fillna(0.0)

        outlier_cells = (z.abs() > 1.6)
        if threshold == -1:
            # remove any row with at least one outlier cell
            to_drop_idx = outlier_cells.any(axis=1)
        else:
            # remove rows where fraction of outlier cells > threshold
            frac = outlier_cells.sum(axis=1) / (outlier_cells.shape[1] if outlier_cells.shape[1] else 1)
            to_drop_idx = frac > float(threshold)

        kept_mask = ~to_drop_idx
        n_removed = int(to_drop_idx.sum())
        print(n_removed, "outlying rows have been removed:")

        if n_removed > 0 and self.verbose:
            print("with indexes:", list(dataset.index[to_drop_idx]))
            print("\nOutliers:\n", dataset.loc[to_drop_idx], "\n")

        return self._apply_row_mask(dataset, kept_mask)

    # ---------- IQR ----------

    def IQR_outlier_detection(self, dataset, threshold):
        X = self._numeric_view(dataset)

        if X.shape[1] < 1:
            print("Error: Need at least one numeric variable for IQR "
                  "outlier detection\nDataset unchanged")
            return dataset

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        outlier_cells = (X.lt(Q1 - 1.5 * IQR)) | (X.gt(Q3 + 1.5 * IQR))
        if threshold == -1:
            to_drop_idx = outlier_cells.any(axis=1)
        else:
            frac = outlier_cells.sum(axis=1) / (outlier_cells.shape[1] if outlier_cells.shape[1] else 1)
            to_drop_idx = frac > float(threshold)

        kept_mask = ~to_drop_idx
        n_removed = int(to_drop_idx.sum())
        print(n_removed, "outlying rows have been removed")

        if n_removed > 0 and self.verbose:
            print("with indexes:", list(dataset.index[to_drop_idx]))
            print("\nOutliers:\n", dataset.loc[to_drop_idx], "\n")

        return self._apply_row_mask(dataset, kept_mask)

    # ---------- LOF ----------

    def LOF_outlier_detection(self, dataset, threshold, n_neighbors):
        """
        LOF requires no missing numeric values. We:
          1) build numeric view X;
          2) drop rows with NaNs in X (announce);
          3) run LOF on remaining rows;
          4) keep only inliers on those rows;
          5) return the subset of the ORIGINAL dataset rows.
        """
        from sklearn.neighbors import LocalOutlierFactor

        X = self._numeric_view(dataset)
        if X.shape[1] < 1 or X.shape[0] < 1:
            print("Error: Need at least one numeric variable for LOF "
                  "outlier detection\nDataset unchanged")
            return dataset

        # Step 2: drop rows with NaNs in numeric view only
        row_has_nan_in_X = X.isnull().any(axis=1)
        if row_has_nan_in_X.any():
            print("LOF requires no missing values in numeric features; "
                  "rows with NaNs in numeric columns have been removed for LOF fit.")
        X_clean = X.loc[~row_has_nan_in_X]
        if X_clean.empty:
            print("All rows had NaNs in numeric features; LOF skipped. Dataset unchanged.")
            return dataset

        # Step 3: LOF
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.2)
        y_pred = clf.fit_predict(X_clean)  # 1=inlier, -1=outlier

        # Step 4: build keep mask for the full dataset
        keep_mask = pd.Series(False, index=dataset.index)
        keep_mask.loc[X_clean.index] = (y_pred != -1)

        n_removed = int((~keep_mask).sum())
        # Note: n_removed here counts both numeric-NaN rows plus LOF outliers
        print(n_removed, "outlying rows have been removed")

        if n_removed > 0 and self.verbose:
            print("with indexes:", list(dataset.index[~keep_mask]))
            print("\nOutliers:\n", dataset.loc[~keep_mask], "\n")

        return self._apply_row_mask(dataset, keep_mask)

    # ---------- Isolation Forest ----------

    def IF_outlier_detection(self, dataset, threshold):
        """
        Isolation Forest on numeric features (requires no NaNs).
        Same approach as LOF for NaN handling and row keep/drop.
        """
        from sklearn.ensemble import IsolationForest

        X = self._numeric_view(dataset)
        if X.shape[1] < 1 or X.shape[0] < 1:
            print("Error: Need at least one numeric variable for IF "
                  "outlier detection\nDataset unchanged")
            return dataset

        row_has_nan_in_X = X.isnull().any(axis=1)
        if row_has_nan_in_X.any():
            print("IF requires no missing values in numeric features; "
                  "rows with NaNs in numeric columns have been removed for IF fit.")

        X_clean = X.loc[~row_has_nan_in_X]
        if X_clean.empty:
            print("All rows had NaNs in numeric features; IF skipped. Dataset unchanged.")
            return dataset

        clf = IsolationForest(n_estimators=50, contamination=0.2, random_state=0)
        y_pred = clf.fit_predict(X_clean)  # 1=inlier, -1=outlier

        keep_mask = pd.Series(False, index=dataset.index)
        keep_mask.loc[X_clean.index] = (y_pred == 1)

        n_removed = int((~keep_mask).sum())
        print(n_removed, "outlying rows have been removed")

        if n_removed > 0 and self.verbose:
            print("with indexes:", list(dataset.index[~keep_mask]))
            print("\nOutliers:\n", dataset.loc[~keep_mask], "\n")

        return self._apply_row_mask(dataset, keep_mask)

    # ---------- NONE (no-op) ----------

    def None_outlier_detection(self, dataset, threshold):
        if self.verbose:
            print("No outlier detection... ")
        # return dataset unchanged, preserving all columns and dtypes
        return dataset

    # ---------- driver ----------

    def transform(self):

        start_time = time.time()
        osd = self.dataset

        print()
        print(">>Outlier detection and removal:")

        for key in ['train']:

            if (not isinstance(self.dataset[key], dict)):

                if not self.dataset[key].empty:

                    print("* For", key, "dataset")
                    d = self.dataset[key]

                    if (self.strategy == "ZSB"):
                        dn = self.ZSB_outlier_detection(d, self.threshold)

                    elif (self.strategy == 'IQR'):
                        dn = self.IQR_outlier_detection(d, self.threshold)

                    elif (self.strategy == "LOF_1"):
                        dn = self.LOF_outlier_detection(d, self.threshold, n_neighbors=1)

                    elif (self.strategy == "LOF_5"):
                        dn = self.LOF_outlier_detection(d, self.threshold, n_neighbors=5)

                    elif (self.strategy == "LOF_10"):
                        dn = self.LOF_outlier_detection(d, self.threshold, n_neighbors=10)

                    elif (self.strategy == "LOF_20"):
                        dn = self.LOF_outlier_detection(d, self.threshold, n_neighbors=20)

                    elif (self.strategy == "LOF_30"):
                        dn = self.LOF_outlier_detection(d, self.threshold, n_neighbors=30)

                    elif (self.strategy == 'IF'):
                        dn = self.IF_outlier_detection(d, self.threshold)

                    elif (self.strategy == "NONE_outlier"):
                        dn = self.None_outlier_detection(d, self.threshold)

                    else:
                        raise ValueError("Threshold invalid. "
                                         "Please choose between "
                                         "'-1' for any outlying value in "
                                         "a row or a value in [0,1] for "
                                         "multivariate outlying row. For "
                                         "example, with threshold=0.5 "
                                         "if a row has outlying values in "
                                         "half of the attribute set and more, "
                                         "it is considered as an outlier and "
                                         "removed")

                    osd[key] = dn

                else:
                    print("No outlier detection for", key, "dataset")

            else:
                print("No outlier detection for", key, "dataset")

        print("Outlier detection and removal done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return osd
