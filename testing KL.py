import numpy as np
import pandas as pd
from scipy.stats import entropy
from LoadDataset import LoadDataset


import numpy as np
import pandas as pd
from scipy.stats import entropy

dataset='adult'
loader = LoadDataset(dataset)
dataset, X_train, y_train, X_test, y_test = loader.load()
print(f"Training data shape: {X_train.shape[0]/X_test.shape[0]}")

#file_train='historical_data/partial_pipeline/sim_historical_data_train_profile_lr_sp_adult.csv'
#ile_test='historical_data/noise/class_sim_historical_data_test_profile_lr_sp_adult.csv'

#data1= pd.read_csv(file_train)
#data2= pd.read_csv(file_test)

#print(f"Training data shape: {data1.shape}")
#print(f"Test data shape: {data2.shape}")

import numpy as np
from scipy.stats import entropy

def kl_divergence(df_p, df_q, bins=50, epsilon=1e-10):
    total_kl = 0.0

    # Same split logic as before
    continuous_cols = df_p.select_dtypes(include=[np.number]).columns
    categorical_cols = df_p.select_dtypes(exclude=[np.number]).columns

    # ---- Continuous ----
    for col in continuous_cols:
        if col not in df_q.columns:
            continue
        p = df_p[col].dropna().to_numpy()
        q = df_q[col].dropna().to_numpy()
        if p.size == 0 or q.size == 0:
            continue  # keep total behavior; just skip empty

        # Keep your original binning approach (edges from P)
        p_hist, bin_edges = np.histogram(p, bins=bins, density=False)
        q_hist, _         = np.histogram(q, bins=bin_edges, density=False)

        # Smooth + renormalize (same as your code)
        p_hist = p_hist.astype(float) + epsilon
        q_hist = q_hist.astype(float) + epsilon
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()

        total_kl += entropy(p_hist, q_hist)

    # ---- Categorical ----
    for col in categorical_cols:
        if col not in df_q.columns:
            continue
        # Keep your original approach (proportions) and re-normalize after epsilon
        p_counts = df_p[col].value_counts(normalize=True, dropna=False)
        q_counts = df_q[col].value_counts(normalize=True, dropna=False)

        all_categories = p_counts.index.union(q_counts.index)
        p_probs = p_counts.reindex(all_categories, fill_value=0.0).astype(float) + epsilon
        q_probs = q_counts.reindex(all_categories, fill_value=0.0).astype(float) + epsilon

        p_probs /= p_probs.sum()
        q_probs /= q_probs.sum()

        total_kl += entropy(p_probs, q_probs)

    return total_kl


def inject_outliers(X, frac, multiplier=5.0):
        X_modified = X.copy()
        
        idx_train = np.arange(len(X_modified))
        outlier_indices = pd.DataFrame(idx_train).sample(frac=frac, replace=False, random_state=42).index
        col = X.select_dtypes(include=['int', 'float']).columns[0]
        #col = 'Education_Num' 

        if not pd.api.types.is_numeric_dtype(X_modified[col]):
            print(f"[WARNING] Column {col} is not numeric. Outlier injection skipped.")
            return X_modified

        Q1 = X_modified[col].quantile(0.25)
        Q3 = X_modified[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
               
        high_outlier_value = Q3 +  multiplier* IQR
        X_modified.loc[outlier_indices, col] = high_outlier_value

        return X_modified

def inject_class_imbalance( X, y, target_col='target', minority_ratio=0.2):
    
    if not 0 < minority_ratio < 0.5:
        raise ValueError("minority_ratio must be between 0 and 0.5 (exclusive)")

    # Ensure y is a Series and assign the correct column name
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Target DataFrame must have exactly one column.")
        y_series = y.iloc[:, 0].rename(target_col)
    else:
        y_series = pd.Series(y, name=target_col)

    df = X.copy()
    df[target_col] = y_series

    class_counts = df[target_col].value_counts()
    if len(class_counts) != 2:
        raise ValueError("Target must be binary.")

    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    total_desired = len(df)
    n_minority = int(total_desired * minority_ratio)
    n_majority = total_desired - n_minority

    minority_df = df[df[target_col] == minority_class]
    majority_df = df[df[target_col] == majority_class]

    minority_sampled = minority_df.sample(n=n_minority, replace=(n_minority > len(minority_df)), random_state=42)
    majority_sampled = majority_df.sample(n=n_majority, replace=(n_majority > len(majority_df)), random_state=42)

    resampled_df = pd.concat([minority_sampled, majority_sampled])
    resampled_df = resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_modified = resampled_df.drop(columns=[target_col])
    y_modified = resampled_df[target_col]
    
    if isinstance(y, pd.DataFrame):
        y_modified = pd.DataFrame(y_modified)

    return X_modified, y_modified


#X_modified = inject_class_imbalance(X_test, y_test, target_col='income', minority_ratio=0.49)
#X_test.head()

kl_div= kl_divergence(X_train, X_test, bins=10, epsilon=1e-10)
print(f"KL Divergence between training and test datasets: {kl_div}")
