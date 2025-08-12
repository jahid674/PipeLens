import numpy as np
import pandas as pd
from scipy.stats import entropy
from LoadDataset import LoadDataset

dataset='adult'
loader = LoadDataset(dataset)
dataset, X_train, y_train, X_test, y_test = loader.load()

import numpy as np
import pandas as pd
from scipy.stats import entropy

def kl_divergence(df_p, df_q, bins=50, epsilon=1e-10):
    kl_results = {}
    total_kl = 0.0

    continuous_cols = df_p.select_dtypes(include=[np.number]).columns
    categorical_cols = df_p.select_dtypes(exclude=[np.number]).columns

    for col in continuous_cols:
        if col not in df_q.columns:
            continue
        p_hist, bin_edges = np.histogram(df_p[col].dropna(), bins=bins, density=True)
        q_hist, _ = np.histogram(df_q[col].dropna(), bins=bin_edges, density=True)

        p_hist += epsilon
        q_hist += epsilon
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()

        kl_val = entropy(p_hist, q_hist)
        kl_results[col] = kl_val
        total_kl += kl_val

    for col in categorical_cols:
        if col not in df_q.columns:
            continue
        p_counts = df_p[col].value_counts(normalize=True)
        q_counts = df_q[col].value_counts(normalize=True)

        all_categories = p_counts.index.union(q_counts.index)
        p_probs = p_counts.reindex(all_categories, fill_value=epsilon)
        q_probs = q_counts.reindex(all_categories, fill_value=epsilon)

        p_probs /= p_probs.sum()
        q_probs /= q_probs.sum()

        kl_val = entropy(p_probs, q_probs)
        kl_results[col] = kl_val
        total_kl += kl_val

    kl_results['total_kl'] = total_kl
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

    return resampled_df


X_modified = inject_class_imbalance(X_test, y_test, target_col='income', minority_ratio=0.49)
X_test.head()

#kl_div= kl_divergence(X_train, X_modified)
#print(f"KL Divergence between training and test datasets: {kl_div}")
