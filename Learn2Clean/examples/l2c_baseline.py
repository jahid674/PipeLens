#!/usr/bin/env python3
# coding: utf-8

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import time
import random
import numpy as np
import pandas as pd

# Learn2Clean modules
import learn2clean.loading.reader as rd
import learn2clean.qlearning.qlearner as ql  # your extended Qlearner

from sklearn.preprocessing import LabelEncoder

# --------------------------- Config --------------------------- #
data = 'hmda'            # 'adult' | 'hmda' | 'housing'
metric_type = 'accuracy_score'  # for adult this means 1 - |SP|; for others, accuracy
tau = 0.1                 # missingness fraction to inject (if used below)

# ---------------------- Dataset loading ----------------------- #
if data == 'hmda':
    dataset_paths = ["data/noisy/noise_injected_hmda.csv",
                     "data/noisy/noise_injected_hmda.csv"]
    hr = rd.Reader(sep=',', verbose=False, encoding=False)
    dataset = hr.train_test_split(dataset_paths, 'action_taken')

    idx_train = np.arange(0, len(dataset['train']), 1)
    mv_train = pd.DataFrame(idx_train).sample(frac=tau, replace=False, random_state=1).index
    if 'Martial_Status' in dataset['train'].columns:
        dataset['train'].loc[mv_train, 'Martial_Status'] = np.NaN

    idx_test = np.arange(0, len(dataset['test']), 1)
    mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
    if 'lien_status' in dataset['test'].columns:
        dataset['test'].loc[mv_test, 'lien_status'] = np.NaN

    metric_path = f"metric/metric_l2c_NN_{metric_type}_{data}.csv"
elif data == 'adult':
    # Use train twice to keep original split behavior (reader will split internally)
    dataset_paths = ["data/noisy/noise_injected_adult.csv", 
                     "data/noisy/noise_injected_adult.csv"]
    hr = rd.Reader(sep=',', verbose=False, encoding=False)
    dataset = hr.train_test_split(dataset_paths, 'income')
    # Optional: inject some NaNs to exercise imputation
    categorical_columns = dataset['train'].select_dtypes(include=['object']).columns
    '''for column in categorical_columns:
        le = LabelEncoder()
        column_unique = pd.unique(list(dataset['train'][column]))
        #print(f"Encoding column: {column} with unique values: {column_unique}")
        le.fit(column_unique)
        dataset['train'][column] = le.transform(dataset['train'][column])
        dataset['train'][column] = dataset['train'][column].astype('category')
        dataset['test'][column] = le.transform(dataset['test'][column])
        dataset['test'][column] = dataset['test'][column].astype('category')
    #dataset['target'] = dataset['target'].apply(lambda x: 1 if str(x).strip() in ['>50K', '>50K.'] else 0)'''

    idx_train = np.arange(0, len(dataset['train']), 1)
    mv_train = pd.DataFrame(idx_train).sample(frac=tau, replace=False, random_state=1).index
    if 'Martial_Status' in dataset['train'].columns:
        dataset['train'].loc[mv_train, 'Martial_Status'] = np.NaN

    idx_test = np.arange(0, len(dataset['test']), 1)
    mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
    if 'Martial_Status' in dataset['test'].columns:
        dataset['test'].loc[mv_test, 'Martial_Status'] = np.NaN

    metric_path = f"metric/metric_l2c_lr_{metric_type}_{data}.csv"
elif data == 'housing':
    d2 = pd.read_csv('data/house/housing_test.csv')
    d2 = pd.DataFrame(d2)

    dataset = {
        'train': d2.copy(),
        'test': d2.copy(),
        'target': d2['SalePrice']
    }

    metric_path = f"metric/metric_l2c_reg_{metric_type}_{data}.csv"
    selected_features = [
        'OverallQual', 'GarageFinish', 'GarageArea', 'YearBuilt', 'TotalBsmtSF',
        '1stFlrSF', 'YearRemodAdd', 'GrLivArea', 'GarageCars', 'FullBath',
        'Fireplaces', 'BsmtQual', 'KitchenQual', 'ExterQual', 'TotRmsAbvGrd'
    ]
    print("These features have been selected as the KBest for passing data in our case.")
    missing_colmn_categorical = [
        'Electrical', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'
    ]
    missing_colmn_numerical = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']

    for column in missing_colmn_categorical:
        if column in dataset['train'].columns:
            most_frequent = dataset['train'][column].mode(dropna=True)
            if len(most_frequent) > 0:
                dataset['train'][column].fillna(most_frequent.iloc[0], inplace=True)

    for column in missing_colmn_numerical:
        if column in dataset['train'].columns:
            median_value = dataset['train'][column].median(skipna=True)
            dataset['train'][column].fillna(median_value, inplace=True)

    selected_features_plus = selected_features + ['SalePrice']
    dataset['train'] = dataset['train'][selected_features_plus]

    categorical_columns = dataset['train'].select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        dataset['train'][column] = dataset['train'][column].astype(str)
        dataset['train'][column] = le.fit_transform(dataset['train'][column])
        dataset['train'][column] = dataset['train'][column].astype('category')

    if 'OverallQual' in dataset['train'].columns:
        idx = np.arange(0, len(dataset['train']), 1)
        mv_idx = pd.DataFrame(idx).sample(frac=tau, replace=False, random_state=1).index
        dataset['train'].loc[mv_idx, 'OverallQual'] = np.NaN
else:
    raise ValueError("Unsupported `data` choice. Use 'hmda', 'adult', or 'housing'.")

# ---------------------- Reporting helper ---------------------- #
def write_quartiles(csv_writer, algorithm, metric, quartiles, f_goal, goals_list, dataset_tag):
    """
    Mirrors your original reporting, with a safer handling of numpy percentile API.
    For hmda/adult: write f_goal directly.
    For housing: write the normalized progress value: 1 - (f_goal - min(goals))/min(goals)
    """
    if dataset_tag in ['adult', 'hmda']:
        prefix = round(f_goal, 2)
    else:
        prefix = round(1 - (f_goal - min(goals_list)) / min(goals_list), 2)

    csv_writer.writerow([prefix, algorithm, f"{metric} q1", round(quartiles[0], 5)])
    csv_writer.writerow([prefix, algorithm, f"{metric} q2", round(quartiles[1], 5)])
    csv_writer.writerow([prefix, algorithm, f"{metric} q3", round(quartiles[2], 5)])
    csv_writer.writerow([prefix, algorithm, f"{metric} q4", round(quartiles[3], 5)])


if data == 'hmda':
    goals = [0.93]
    goal = 'NN'
    target_goal = 'action_taken'
    target_prepare = 'action_taken'
elif data == 'housing':
    goals = [155, 160, 170, 180]
    goal = 'MARS'
    target_goal = 'SalePrice'
    target_prepare = 'SalePrice'
elif data == 'adult':
    goals = [0.9, 0.88, 0.86, 0.84]
    goal = 'NN'
    target_goal = 'income'
    target_prepare = 'income'

# --------------------- Run experiments ------------------------ #
random_seeds = random.sample(range(0, 1000000), 1)
os.makedirs(os.path.dirname(metric_path), exist_ok=True)
# seed 42 for effieciency
# NEW: start timer
t_start = time.perf_counter()

with open(metric_path, 'w', newline='') as f:
    csv_writer = csv.writer(f)

    for g in goals:
        iterations = []
        for seed in random_seeds:
            # Pass dataset_name to toggle fairness-vs-accuracy in Classifier
            l2c = ql.Qlearner(
                dataset=dataset,
                goal=goal,
                target_goal=target_goal,
                target_prepare=target_prepare,
                verbose=False,
                f_goal=g,
                dataset_name=data  # <-- 'adult' yields quality_metric = 1 - |SP|
            )
            result = l2c.learn2clean(r_state=seed)
            if result[0]:
                iterations.append(result[1])
            else:
                iterations.append(-1)

        rank_quartiles = np.percentile(iterations, [25, 50, 75, 100], method='midpoint')
        write_quartiles(csv_writer, "l2c", "iterations", rank_quartiles, g, goals, data)
        csv_writer.writerow([])

# NEW: end timer and pretty print duration
elapsed_sec = time.perf_counter() - t_start
hours = int(elapsed_sec // 3600)
mins  = int((elapsed_sec % 3600) // 60)
secs  = elapsed_sec % 60

print(f"Finished. Wrote results to: {metric_path}")
print("Goals :", goals)
print("Iterations (last goal) :", iterations)
print(dataset['train'].dtypes)
print(f"Total execution time: {hours:02d}:{mins:02d}:{secs:06.3f} (hh:mm:ss)")

