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

        # Store excluded columns separately
        excluded_cols = self.dataset[self.exclude] if self.exclude else pd.DataFrame()
        df = self.dataset.drop(columns=self.exclude, errors='ignore')

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


'''import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=100, n_features=10, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["user_id"] = np.random.randint(1000, 9999, size=len(df))
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[feature_names] = scaler.fit_transform(df[feature_names])
print(" Initial features:", df_scaled.columns.tolist())

selector = FeatureSelector(
    dataset=df_scaled,
    strategy='mutual_info',
    top_k=None,                
    verbose=True,
    exclude=['user_id']        
)

X_selected = selector.transform(y_train=y)

print("\nFinal selected columns:")
print(X_selected.columns.tolist())
print("\nShape of selected X:", X_selected.shape)'''
