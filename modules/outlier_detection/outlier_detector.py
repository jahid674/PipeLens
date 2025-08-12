import numpy as np
import pandas as pd
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class OutlierDetector:
    """
    - dataset: input DataFrame (df = scaled_X_train)
    - strategy: 'none', 'if', or 'lof'
    - k: if using 'lof', number of neighbors (must be one of {1,5,10,15,20})
    - contamination: contamination rate for Isolation Forest and LOF
    - verbose: whether to print information
    - exclude: list of columns to exclude from outlier detection
    """

    def __init__(self, dataset, strategy='none', k=None, contamination=0.2, verbose=False, exclude=None):
        self.df1 = dataset.copy()
        self.df = dataset.copy()
        #self.df = dataset[['Education_Num']]
        self.strategy = strategy.lower()
        self.contamination = contamination
        self.verbose = verbose
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []
        self.frac = None

        if self.strategy not in ['none', 'if', 'iqr', 'lof']:
            raise ValueError("Strategy must be one of 'none', 'if', or 'lof'.")

        if self.strategy == 'lof':
            self.k = k

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("---------->> Starting Outlier Detection <<-----------")

        excluded_cols = self.df[self.exclude] if self.exclude else pd.DataFrame()
        self.df = self.df.drop(columns=self.exclude, errors='ignore')

        if self.strategy == 'none':
            if self.verbose:
                print("No outlier detection applied.")
            outlier_y_pred = np.ones(len(self.df))

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
                print(f"Applying IQR-based outlier detection.")
            
            numeric_df = self.df.select_dtypes(include=[np.number])
            outlier_mask = np.ones(len(numeric_df), dtype=bool)

            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask &= numeric_df[col].between(lower_bound, upper_bound)

            # Re-map mask to full df size
            outlier_y_pred = np.where(outlier_mask, 1, -1)
        else:
            raise ValueError("Invalid strategy selected.")

        mask = outlier_y_pred != -1
        self.frac = round((1 - sum(mask)/len(outlier_y_pred)) * 100, 4)
        outlier_X_train = self.df1.copy()
        outlier_y_train = y_train.copy() if y_train is not None else None
        outlier_sensitive_train = sensitive_attr_train.copy() if sensitive_attr_train is not None else None

        if self.verbose:
            print(f"Total samples: {len(self.df)}, Remaining after outlier removal: {sum(mask)}")

        if (sum(mask) > 0) and (sum(mask) < len(outlier_y_pred)):
            outlier_X_train = self.df1[mask]
            outlier_X_train.reset_index(drop=True, inplace=True)
            if outlier_y_train is not None:
                outlier_y_train = y_train[mask]
                outlier_y_train.reset_index(drop=True, inplace=True)
            if outlier_sensitive_train is not None:
                outlier_sensitive_train = sensitive_attr_train[mask]
                outlier_sensitive_train.reset_index(drop=True, inplace=True)

        

        if not excluded_cols.empty:
            excluded_cols = excluded_cols.iloc[mask].reset_index(drop=True)
            outlier_X_train = pd.concat([outlier_X_train, excluded_cols], axis=1)
            outlier_X_train = outlier_X_train[self.dataset.columns]

        if self.verbose:
            print(f"Outlier detection completed in {time.time() - start_time:.2f} seconds.\n")

        return outlier_X_train, outlier_y_train, outlier_sensitive_train
    
    def get_frac(self):
        if self.frac is None:
            raise ValueError("Fraction of outliers not calculated. Please run transform() first.")
        return self.frac


'''X_train = pd.DataFrame({
    'age': [25, 35, 45, 55, 1000],
    'income': [50000, 60000, 70000, 80000, 250000],
    'gender': ['M', 'F', 'F', 'M', 'F']
})

y_train = pd.Series([1, 0, 1, 0, 1])
sensitive_attr_train = pd.Series([1, 0, 0, 1, 1])
detector = OutlierDetector(dataset=X_train, strategy='lof', k=2, contamination=0.1, verbose=True, exclude=['gender'])
outlier_X, outlier_y, outlier_sensitive = detector.transform(y_train, sensitive_attr_train)

print(outlier_X)
print(outlier_y)
print(outlier_sensitive)'''
