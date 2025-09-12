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
import learn2clean.qlearning.qlearner_order as ql  # your extended Qlearner

# Sklearn utilities (for the housing prep)
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Config (edit as needed)
# -----------------------

data = 'housing'         # 'housing' or 'hmda'
metric_type = 'rmse'     # 'rmse' or 'accuracy' (used only for naming the metric file)
tau = 0.1                # fraction of entries to set as missing (simulated)

# -----------------------
# Data loading & shaping
# -----------------------
if data == 'hmda':
    # Train/test from files via Reader (target is 'action_taken')
    # NOTE: both paths are the same in your snippet; keep as-is unless you have a separate test path
    dataset_paths = ["data/hmda/hmda_Orleans_X_test_1.csv",
                     "data/hmda/hmda_Orleans_X_test_1.csv"]
    hr = rd.Reader(sep=',', verbose=False, encoding=False)
    dataset = hr.train_test_split(dataset_paths, 'action_taken')

    # Inject missing values in a chosen column (example: 'lien_status')
    # Make sure the column exists in your CSVs; otherwise pick a different column name present in both sets.
    idx_train = np.arange(0, len(dataset['train']), 1)
    mv_train = pd.DataFrame(idx_train).sample(frac=tau, replace=False, random_state=1).index
    if 'Martial_Status' in dataset['train'].columns:
        dataset['train'].loc[mv_train, 'Martial_Status'] = np.NaN

    idx_test = np.arange(0, len(dataset['test']), 1)
    mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
    if 'lien_status' in dataset['test'].columns:
        dataset['test'].loc[mv_test, 'lien_status'] = np.NaN

    metric_path = f"metric/metric_l2c_lr_{metric_type}_{data}.csv"

elif data == 'adult':
    dataset_paths  = ["data/adult/adult_test.csv",
                    "data/adult/adult_train.csv"]
    hr = rd.Reader(sep=',', verbose=False, encoding=False)
    dataset = hr.train_test_split(dataset_paths, 'income')
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
    # Single CSV used for both train and test (as in your snippet)
    d2 = pd.read_csv('data/house/housing_test.csv')
    d2 = pd.DataFrame(d2)

    dataset = {
        'train': d2.copy(),
        'test': d2.copy(),
        'target': d2['SalePrice']
    }

    metric_path = f"metric/metric_l2c_reg_{metric_type}_{data}.csv"

    # K-best style feature subset (as per your snippet)
    selected_features = [
        'OverallQual', 'GarageFinish', 'GarageArea', 'YearBuilt', 'TotalBsmtSF',
        '1stFlrSF', 'YearRemodAdd', 'GrLivArea', 'GarageCars', 'FullBath',
        'Fireplaces', 'BsmtQual', 'KitchenQual', 'ExterQual', 'TotRmsAbvGrd'
    ]
    print("These features have been selected as the KBest for passing data in our case.")

    # Fill missing values (categorical -> mode, numerical -> median)
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

    # Restrict to selected features + target
    selected_features_plus = selected_features + ['SalePrice']
    dataset['train'] = dataset['train'][selected_features_plus]

    # Encode categoricals
    categorical_columns = dataset['train'].select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        # Handle potential NaNs before encoding
        dataset['train'][column] = dataset['train'][column].astype(str)
        dataset['train'][column] = le.fit_transform(dataset['train'][column])
        dataset['train'][column] = dataset['train'][column].astype('category')

    # Inject missing values in a numeric column (simulate missingness)
    if 'OverallQual' in dataset['train'].columns:
        idx = np.arange(0, len(dataset['train']), 1)
        mv_idx = pd.DataFrame(idx).sample(frac=tau, replace=False, random_state=1).index
        dataset['train'].loc[mv_idx, 'OverallQual'] = np.NaN

else:
    raise ValueError("Unsupported `data` choice. Use 'hmda' or 'housing'.")

# -----------------------
# Helpers for CSV output
# -----------------------
def write_quartiles(csv_writer, algorithm, metric, quartiles, f_goal, goals_list, dataset_tag):
    """
    Mirrors your original reporting, with a safer handling of numpy percentile API.
    For hmda/adult: write f_goal directly.
    For housing: write the normalized progress value: 1 - (f_goal - min(goals))/min(goals)
    """
    if dataset_tag in ['adult', 'hmda']:
        prefix = round(f_goal, 2)
    else:
        # Using your original normalization formula
        prefix = round(1 - (f_goal - min(goals_list)) / min(goals_list), 2)

    csv_writer.writerow([prefix, algorithm, f"{metric} q1", round(quartiles[0], 5)])
    csv_writer.writerow([prefix, algorithm, f"{metric} q2", round(quartiles[1], 5)])
    csv_writer.writerow([prefix, algorithm, f"{metric} q3", round(quartiles[2], 5)])
    csv_writer.writerow([prefix, algorithm, f"{metric} q4", round(quartiles[3], 5)])

if data == 'hmda':
    goals = [0.91, 0.92, 0.93, 0.94]   # accuracy targets
    goal = 'LR'
    target_goal = 'action_taken'
    target_prepare = 'action_taken'
elif data == 'housing':
    goals = [162, 170, 180, 185]       # RMSE-like targets (lower is better)
    goal = 'MARS'
    target_goal = 'SalePrice'
    target_prepare = 'SalePrice'
else:  # adult
    goals = [0.05, 0.06, 0.13, 0.15]  # accuracy targets
    goal = 'LR'
    target_goal = 'income'
    target_prepare = 'income'

# 315 random seeds to reproduce your setup
random_seeds = random.sample(range(0, 1_000_000), 315)

# Ensure metric folder exists
os.makedirs(os.path.dirname(metric_path), exist_ok=True)

# -----------------------
# Run experiments
# -----------------------
with open(metric_path, 'w', newline='') as f:
    csv_writer = csv.writer(f)

    for g in goals:
        iterations = []
        for seed in random_seeds:
            l2c = ql.Qlearner(dataset=dataset,
                              goal=goal,
                              target_goal=target_goal,
                              target_prepare=target_prepare,
                              verbose=False,
                              f_goal=g)
            result = l2c.learn2clean(r_state=seed)
            # result = (achieved_bool, iterations_count)
            if result[0]:
                iterations.append(result[1])
            else:
                iterations.append(-1)  # not achieved

        # Quartiles of iteration counts (including -1 for not achieved)
        # np.percentile: use 'method' instead of deprecated 'interpolation'
        rank_quartiles = np.percentile(iterations, [25, 50, 75, 100], method='midpoint')
        write_quartiles(csv_writer, "l2c", "iterations", rank_quartiles, g, goals, data)
        csv_writer.writerow([])

print(f"Finished. Wrote results to: {metric_path}")
print("Goals :", goals)
# `iterations` keeps the last goal's iteration distribution
print("Iterations (last goal) :", iterations)
