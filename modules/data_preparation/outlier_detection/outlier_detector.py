import numpy as np
np.random.seed(42)
import pandas as pd
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetector:
    """
    - dataset: input DataFrame (df = scaled_X_train)
    - strategy: 'none', 'if', 'iqr', or 'lof'
    - k: if using 'lof', number of neighbors
    - contamination: contamination rate for Isolation Forest and LOF
    - verbose: whether to print information
    - exclude: list of columns to exclude from outlier detection
    """

    def __init__(self, dataset, strategy='none', k=None, contamination=0.2, verbose=False, exclude=None):
        self.df1 = dataset.copy()
        self.df = dataset.copy()
        self.strategy = str(strategy).lower()
        self.contamination = contamination
        self.verbose = verbose
        self.exclude = exclude if isinstance(exclude, list) else ([exclude] if exclude else [])
        self.frac = None

        if self.strategy not in ['none', 'if', 'iqr', 'lof']:
            raise ValueError("Strategy must be one of 'none', 'if', 'iqr', or 'lof'.")

        if self.strategy == 'lof':
            if k is None:
                raise ValueError("For strategy='lof', you must provide k (n_neighbors).")
            self.k = int(k)

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("---------->> Starting Outlier Detection <<-----------")

        # ---------------------------------------------------------
        # 0) FORCE CONSISTENT ROW INDEXING FIRST (critical fix)
        # ---------------------------------------------------------
        self.df1 = self.df1.reset_index(drop=True)
        self.df = self.df.reset_index(drop=True)

        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.reset_index(drop=True)
        if isinstance(sensitive_attr_train, (pd.Series, pd.DataFrame)):
            sensitive_attr_train = sensitive_attr_train.reset_index(drop=True)

        # ---------------------------------------------------------
        # 1) DROP ROWS WITH MISSING VALUES IN X (safe masking)
        # ---------------------------------------------------------
        na_mask = self.df1.notna().all(axis=1)          # pandas Series aligned with df1 index
        mask_np = na_mask.to_numpy(dtype=bool)          # numpy mask for positional filtering

        if self.verbose:
            removed = len(self.df1) - int(mask_np.sum())
            print(f"Dropping rows with missing values: removed {removed} rows.")

        self.df1 = self.df1.iloc[mask_np].reset_index(drop=True)
        self.df = self.df.iloc[mask_np].reset_index(drop=True)

        if y_train is not None:
            # IMPORTANT: iloc with numpy mask avoids index alignment issues
            y_train = y_train.iloc[mask_np].reset_index(drop=True)
        if sensitive_attr_train is not None:
            sensitive_attr_train = sensitive_attr_train.iloc[mask_np].reset_index(drop=True)

        # ---------------------------------------------------------
        # 2) HANDLE EXCLUDED COLUMNS
        # ---------------------------------------------------------
        excluded_cols = self.df[self.exclude].copy() if self.exclude else pd.DataFrame(index=self.df.index)
        self.df = self.df.drop(columns=self.exclude, errors='ignore')

        # ---------------------------------------------------------
        # 3) OUTLIER DETECTION
        # ---------------------------------------------------------
        if self.strategy == 'none':
            if self.verbose:
                print("No outlier detection applied.")
            outlier_y_pred = np.ones(len(self.df), dtype=int)

        elif self.strategy == 'if':
            if self.verbose:
                print(f"Applying Isolation Forest with contamination={self.contamination}")
            clf = IsolationForest(n_estimators=50, contamination=self.contamination, random_state=0)
            outlier_y_pred = clf.fit_predict(self.df)

        elif self.strategy == 'lof':
            if self.verbose:
                print(f"Applying Local Outlier Factor with k={self.k}, contamination={self.contamination}")
            lof = LocalOutlierFactor(n_neighbors=self.k, contamination=self.contamination)
            outlier_y_pred = lof.fit_predict(self.df)

        elif self.strategy == 'iqr':
            if self.verbose:
                print("Applying IQR-based outlier detection.")
            numeric_df = self.df.select_dtypes(include=[np.number])

            # if no numeric columns, do nothing
            if numeric_df.shape[1] == 0:
                outlier_y_pred = np.ones(len(self.df), dtype=int)
            else:
                keep_mask = np.ones(len(numeric_df), dtype=bool)
                for col in numeric_df.columns:
                    Q1 = numeric_df[col].quantile(0.25)
                    Q3 = numeric_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    keep_mask &= numeric_df[col].between(lower_bound, upper_bound).to_numpy()
                outlier_y_pred = np.where(keep_mask, 1, -1)

        else:
            raise ValueError("Invalid strategy selected.")

        # ---------------------------------------------------------
        # 4) FILTER BY OUTLIER MASK (ALSO POSITIONAL)
        # ---------------------------------------------------------
        keep = (outlier_y_pred != -1)
        self.frac = round((1 - keep.sum() / len(outlier_y_pred)) * 100, 4) if len(outlier_y_pred) else 0.0

        outlier_X_train = self.df1.copy()
        outlier_y_train = y_train.copy() if y_train is not None else None
        outlier_sensitive_train = sensitive_attr_train.copy() if sensitive_attr_train is not None else None

        if self.verbose:
            print(f"Total samples: {len(self.df)}, Remaining after outlier removal: {keep.sum()}")

        if (keep.sum() > 0) and (keep.sum() < len(outlier_y_pred)):
            keep_np = np.asarray(keep, dtype=bool)

            outlier_X_train = self.df1.iloc[keep_np].reset_index(drop=True)

            if outlier_y_train is not None:
                outlier_y_train = y_train.iloc[keep_np].reset_index(drop=True)

            if outlier_sensitive_train is not None:
                outlier_sensitive_train = sensitive_attr_train.iloc[keep_np].reset_index(drop=True)

            if not excluded_cols.empty:
                excluded_cols = excluded_cols.iloc[keep_np].reset_index(drop=True)

        # ---------------------------------------------------------
        # 5) ADD EXCLUDED COLS BACK (preserve original column order)
        # ---------------------------------------------------------
        if not excluded_cols.empty:
            outlier_X_train = pd.concat([outlier_X_train, excluded_cols], axis=1)

            # Reorder to match original dataset column order
            # self.df1.columns == original columns after NA-drop, before outlier filtering
            outlier_X_train = outlier_X_train.loc[:, self.df1.columns]

        if self.verbose:
            print(f"Outlier detection completed in {time.time() - start_time:.2f} seconds.\n")

        return outlier_X_train, outlier_y_train, outlier_sensitive_train

    def get_frac(self):
        if self.frac is None:
            raise ValueError("Fraction of outliers not calculated. Please run transform() first.")
        return self.frac