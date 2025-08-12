import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#np.random.seed(42)


class NoiseInjector:
    def __init__(self, pipeline_type, dataset_name, target_variable_name=None):
        self.pipeline_type = pipeline_type
        self.dataset_name = dataset_name
        self.target_variable_name = target_variable_name

    def inject_outliers(self, X, frac=0.3, multiplier=5.0):
        X_modified = X.copy()
        idx_train = np.arange(len(X_modified))
        self.outlier_indices = pd.DataFrame(idx_train).sample(frac=frac, replace=False, random_state=42).index

        if self.pipeline_type == 'ml':
            if self.dataset_name == 'hmda':
                col = 'lien_status'
            elif self.dataset_name == 'adult':
                #col = 'Education_Num'
                col = X.select_dtypes(include=['int', 'float']).columns[0]

                #col = 'Education_Num'
                sens_col = 'Sex'

                # Filter rows where Sex == 0 and income == 1
                target_mask = (X_modified[sens_col] == 1)
                eligible_indices = X_modified.index[target_mask]

                # Sample fraction of these rows
                n_inject = max(1, int(frac * len(eligible_indices)))
                if n_inject > 0:
                    self.outlier_indices = np.random.choice(eligible_indices, size=n_inject, replace=False)

                    if not pd.api.types.is_numeric_dtype(X_modified[col]):
                        print(f"[WARNING] Column {col} is not numeric. Outlier injection skipped.")
                        return X_modified
            elif self.dataset_name == 'housing':
                #col = 'OverallQual'
                col = X.select_dtypes(include=['int', 'float']).columns[0]
            else:
                return X_modified

            if not pd.api.types.is_numeric_dtype(X_modified[col]):
                print(f"[WARNING] Column {col} is not numeric. Outlier injection skipped.")
                return X_modified

            Q1 = X_modified[col].quantile(0.25)
            Q3 = X_modified[col].quantile(0.75)
            IQR = Q3 - Q1
            high_outlier_value = Q1 - multiplier * IQR
            X_modified.loc[self.outlier_indices, col] = high_outlier_value

        return X_modified
    
    def inject_class_imbalance(self, X, y):
        if self.outlier_indices is None or len(self.outlier_indices) == 0:
            return y

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

        # Align indices (if y has no index from X, outlier_indices will still work
        # as position-based if y_series has a default RangeIndex matching X)
        # Intersect to avoid KeyErrors
        flip_idx = y_series.index.intersection(self.outlier_indices)

        # Flip
        y_series.loc[flip_idx] = y_series.loc[flip_idx].map(flip_map)

        # Return in original type
        if original_type == "dataframe":
            return pd.DataFrame(y_series, columns=y.columns)
        if original_type == "ndarray":
            return y_series.to_numpy()
        return X, y_series



    '''def inject_class_imbalance(self, X, y, minority_ratio=0.2):
        if not 0 < minority_ratio < 0.5:
            raise ValueError("minority_ratio must be between 0 and 0.5 (exclusive)")

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Target DataFrame must have exactly one column.")
            y_series = y.iloc[:, 0].rename(self.target_variable_name)
        else:
            y_series = pd.Series(y, name=self.target_variable_name)

        df = X.copy()
        df[self.target_variable_name] = y_series

        class_counts = df[self.target_variable_name].value_counts()
        if len(class_counts) != 2:
            raise ValueError("Target must be binary.")

        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        total_desired = len(df)
        n_minority = int(total_desired * minority_ratio)
        n_majority = total_desired - n_minority

        minority_df = df[df[self.target_variable_name] == minority_class]
        majority_df = df[df[self.target_variable_name] == majority_class]

        minority_sampled = minority_df.sample(n=n_minority, replace=(n_minority > len(minority_df)), random_state=42)
        majority_sampled = majority_df.sample(n=n_majority, replace=(n_majority > len(majority_df)), random_state=42)

        resampled_df = pd.concat([minority_sampled, majority_sampled])
        resampled_df = resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

        X_modified = resampled_df.drop(columns=[self.target_variable_name])
        y_modified = resampled_df[self.target_variable_name]

        if isinstance(y, pd.DataFrame):
            y_modified = pd.DataFrame(y_modified)

        return X_modified, y_modified'''

    def inject_missing_values(self, X, frac=0.1):
        X_modified = X.copy()
        idx_train = np.arange(len(X_modified))
        mv_train = pd.DataFrame(idx_train).sample(frac=frac, replace=False, random_state=1).index

        if self.pipeline_type == 'ml':
            if self.dataset_name == 'hmda':
                X_modified.loc[mv_train, 'lien_status'] = np.nan
            elif self.dataset_name == 'adult':
                X_modified.loc[mv_train, 'Martial_Status'] = np.nan
            elif self.dataset_name == 'housing':
                X_modified.loc[mv_train, 'OverallQual'] = np.nan

        return X_modified

    def inject_noise(self, X, y=None, noise_type='outlier', frac=0.1):
        """
        Injects a specified type of noise into the dataset.

        Parameters:
            X (DataFrame): Feature matrix.
            y (Series/DataFrame, optional): Target variable.
            noise_type (str): One of 'outlier', 'missing', or 'imbalance'.
            **kwargs: Additional parameters for each injection method.

        Returns:
            - For 'imbalance': (X_modified, y_modified)
            - Otherwise: X_modified
        """
        if noise_type == 'outlier':
            return self.inject_outliers(X, frac)
        elif noise_type == 'missing':
            return self.inject_missing_values(X, frac)
        elif noise_type == 'class_imbalance':
            if y is None:
                raise ValueError("Target variable y is required for class imbalance injection.")
            #return self.inject_class_imbalance(X, y, minority_ratio=frac)
            return self.inject_class_imbalance(X, y)
        else:
            raise ValueError(f"Unsupported noise_type '{noise_type}'. Choose from 'outlier', 'missing', or 'imbalance'.")
