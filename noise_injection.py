import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # (unused here but kept)
np.random.seed(42) 

class NoiseInjector:
    def __init__(self, pipeline_type, dataset_name, target_variable_name=None, seed: int = 42):
        self.pipeline_type = pipeline_type
        self.dataset_name = dataset_name
        self.target_variable_name = target_variable_name
        self.seed = seed
        # Use a per-instance RNG for reproducibility
        self.rng = np.random.default_rng(seed)
        self.outlier_indices = None  # make sure this exists

    def inject_outliers(self, X, frac=0.3, multiplier=5.0):
        X_modified = X.copy()
        idx_train = np.arange(len(X_modified))
        # Use deterministic sampling via random_state
        self.outlier_indices = (
            pd.DataFrame(idx_train)
            .sample(frac=frac, replace=False, random_state=self.seed)
            .index
        )

        if self.pipeline_type == 'ml':
            if self.dataset_name == 'hmda':
                col = 'lien_status'
                #col = X.select_dtypes(include=['int', 'float']).columns[0]
            elif self.dataset_name == 'adult':
                col = X.select_dtypes(include=['int', 'float']).columns[0]
                sens_col = 'Sex'

                # Filter rows where Sex == 1 (as in your code)
                target_mask = (X_modified[sens_col] == 1)
                eligible_indices = X_modified.index[target_mask]

                n_inject = max(1, int(frac * len(eligible_indices)))
                if n_inject > 0 and len(eligible_indices) > 0:
                    # Use the local RNG for deterministic choice
                    self.outlier_indices = self.rng.choice(
                        eligible_indices, size=min(n_inject, len(eligible_indices)), replace=False
                    )
                else:
                    self.outlier_indices = np.array([], dtype=int)

                if not pd.api.types.is_numeric_dtype(X_modified[col]):
                    print(f"[WARNING] Column {col} is not numeric. Outlier injection skipped.")
                    return X_modified

            elif self.dataset_name == 'housing':
                col='OverallQual'
                #col = X.select_dtypes(include=['int', 'float']).columns[0]
            else:
                return X_modified

            if not pd.api.types.is_numeric_dtype(X_modified[col]):
                print(f"[WARNING] Column {col} is not numeric. Outlier injection skipped.")
                return X_modified

            Q1 = X_modified[col].quantile(0.25)
            Q3 = X_modified[col].quantile(0.75)
            IQR = Q3 - Q1
            high_outlier_value = Q1 - multiplier * IQR

            # Ensure outlier_indices is index-like
            X_modified.loc[self.outlier_indices, col] = high_outlier_value

        return X_modified

    def inject_class_imbalance(self, X, y):
        """
        Flips labels at indices selected during outlier injection (self.outlier_indices).
        Always returns (X_modified, y_modified) for consistency.
        """
        if self.outlier_indices is None or len(self.outlier_indices) == 0:
            # Nothing to flip; return inputs in a consistent tuple form
            return X, y

        # Normalize y -> pandas Series with an index
        original_type = "series"
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y DataFrame must have exactly one column.")
            y_series = y.iloc[:, 0].copy()
            original_type = "dataframe"
        elif isinstance(y, pd.Series):
            y_series = y.copy()
            original_type = "series"
        elif isinstance(y, np.ndarray):
            y_series = pd.Series(y)
            original_type = "ndarray"
        else:
            raise TypeError("y must be a Series, 1-col DataFrame, or ndarray.")

        # Ensure binary
        uniq = pd.unique(y_series.dropna())
        if len(uniq) != 2:
            raise ValueError(f"Labels must be binary; got {len(uniq)} unique values: {uniq}")

        # Build flip map (swap the two classes)
        a, b = uniq[0], uniq[1]
        flip_map = {a: b, b: a}

        # Intersect indices to avoid KeyError
        flip_idx = y_series.index.intersection(self.outlier_indices)

        # Flip
        y_series.loc[flip_idx] = y_series.loc[flip_idx].map(flip_map)

        # Return in original type
        if original_type == "dataframe":
            y_modified = pd.DataFrame(y_series, columns=y.columns)
        elif original_type == "ndarray":
            y_modified = y_series.to_numpy()
        else:
            y_modified = y_series

        return X, y_modified

    def inject_missing_values(self, X, frac=0.1):
        X_modified = X.copy()
        idx_train = np.arange(len(X_modified))
        mv_train = (
            pd.DataFrame(idx_train)
            .sample(frac=frac, replace=False, random_state=self.seed)
            .index
        )

        if self.pipeline_type == 'ml':
            if self.dataset_name == 'hmda':
                X_modified.loc[mv_train, 'lien_status'] = np.nan
            elif self.dataset_name == 'adult':
                # Note: check your column spelling ('Marital_Status' vs 'Martial_Status')
                X_modified.loc[mv_train, 'Martial_Status'] = np.nan
            elif self.dataset_name == 'housing':
                X_modified.loc[mv_train, 'OverallQual'] = np.nan

        return X_modified

    def inject_noise(self, X, y=None, noise_type='outlier', frac=0.1):
        """
        Injects a specified type of noise into the dataset.

        Returns:
            - For 'class_imbalance': (X_modified, y_modified)
            - Otherwise: X_modified
        """
        if noise_type == 'outlier':
            return self.inject_outliers(X, frac)
        elif noise_type == 'missing':
            return self.inject_missing_values(X, frac)
        elif noise_type == 'class_imbalance':
            if y is None:
                raise ValueError("Target variable y is required for class imbalance injection.")
            return self.inject_class_imbalance(X, y)
        else:
            raise ValueError(f"Unsupported noise_type '{noise_type}'. Choose from 'outlier', 'missing', or 'imbalance'.")
