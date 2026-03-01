# Custom pipeline

import itertools
# from modules.outlier_detection.outlier_detector import OutlierDetector
from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
from modules.normalization.normalizer import DataNormalizer
from modules.metric.metric import metric
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from regression import Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import time
import csv
import os
import pandas as pd
import numpy as np
import math
from itertools import product
import random
from itertools import cycle
import sys
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
# from modules.outlier_detection.outlier_detector import OutlierDetector
from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
from modules.normalization.normalizer import DataNormalizer
from sklearn.naive_bayes import GaussianNB
from regression import Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import KNNImputer, SimpleImputer
from random import sample 
import time
import os
import pandas as pd
import numpy as np
from itertools import product
import random
import copy
import csv 
from sklearn.metrics import f1_score,accuracy_score
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
import logging
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

import statistics 
from sklearn.preprocessing import LabelEncoder
# import prose.datainsights as di
from scipy.stats import chisquare,chi2_contingency
from scipy import stats

def getIdxSensitive(df, dataset):
        if dataset == 'hmda':
                priv_idx = df.index[df['race']==1]
                unpriv_idx = df.index[df['race']==0]
                sensitive_attr = df['race']
        elif dataset == 'adult':
                priv_idx = df.index[df['Sex']==1]
                unpriv_idx = df.index[df['Sex']==0]
                sensitive_attr = df['Sex']
        return priv_idx, unpriv_idx, sensitive_attr

def computeStatisticalParity(p_priv, p_unpriv):
        # p_priv = pd.DataFrame(p_priv)[1]
        # p_unpriv = pd.DataFrame(p_unpriv)[1]
        diff = p_priv.mean() - p_unpriv.mean()
        return diff

dataset = 'adult'
adult_train  = "data/adult/adult_test.csv"
adult_test = "data/adult/adult_train.csv"
train,test = Reader(adult_train,adult_test).load_data()

categorical_columns = train.select_dtypes(include=['object']).columns
numerical_columns = train.select_dtypes(include=['int', 'float']).columns

label_encoder = LabelEncoder()
for column in categorical_columns:
    le = LabelEncoder()
    column_unique = list(train[column]) + list(test[column])
    column_unique = pd.unique(column_unique)
    le.fit(column_unique)
    train[column] = le.transform(train[column])
    test[column] = le.transform(test[column])
    train[column] = train[column].astype('category')
    test[column] = test[column].astype('category')

y_train = train['income']
X_train = train.drop('income', axis=1)
y_test = test['income']
X_test = test.drop('income', axis=1)

tau_train = 0.1 # fraction of missing values
tau_test = 0.1
contamination_train = 0.2
contamination_test = 0.2
contamination_train_lof = 'auto'
contamination_test_lof = 'auto'

run_train = False
if not run_train:
      X_train = X_test.copy()
      y_train = y_test.copy()
priv_idx_train, unpriv_idx_train, sensitive_attr_train = getIdxSensitive(X_train, dataset)
sens_attr_name = sensitive_attr_train.name
target_variable_name = y_train.name

idx_train = np.arange(0, len(X_train), 1)
mv_train = pd.DataFrame(idx_train).sample(frac=tau_train, replace=False, random_state=1).index
# inject missing values in the most important column
X_train['Martial_Status'][mv_train] = np.NaN
print("Length of df: ", len(X_train))
print("Privileged: ", sum(X_train["Sex"]))
print("Ratio privileged:", sum(X_train["Sex"])/len(X_train))
print("Base rate privileged: ", sum(y_train.iloc[priv_idx_train])/len(priv_idx_train))
print("Base rate unprivileged: ", sum(y_train.iloc[unpriv_idx_train])/len(unpriv_idx_train))
# mv
drop = False
if drop:
    updated_y_train = y_train.copy()
    for idx in sorted(mv_train, reverse=True):
        del updated_y_train[idx]
    updated_y_train.reset_index(drop=True, inplace=True)
    imputed_X_train = X_train.drop(mv_train)
    imputed_X_train.reset_index(drop=True, inplace=True)
    updated_sensitive_attr_train = sensitive_attr_train.drop(mv_train)
    updated_sensitive_attr_train.reset_index(drop=True, inplace=True)
else:
    k = 30
    imputed_X_train = KNNImputer(n_neighbors=k).fit_transform(X_train)
    updated_y_train = y_train.copy()
    updated_sensitive_attr_train = sensitive_attr_train.copy()

imputed_df = pd.DataFrame(imputed_X_train, columns = X_train.columns)
imputed_df['target'] = updated_y_train
br_priv = ((imputed_df['Sex'] == 1) & (imputed_df['target'] == 1)).sum()/(imputed_df['Sex'] == 1).sum()
br_unpriv = ((imputed_df['Sex'] == 0) & (imputed_df['target'] == 1)).sum()/(imputed_df['Sex'] == 0).sum()
print(br_priv)
print(br_unpriv)


# print(imputed_X_train_df)
# print("Length of imputed df: ", len(imputed_X_train_df))
# print("Privileged:", sum(imputed_X_train_df["Sex"]))
# print("Ratio privileged:", sum(imputed_X_train_df["Sex"])/len(imputed_X_train_df))
# print("Base rate privileged: ", sum(updated_y_train.iloc[priv_idx_train])/len(priv_idx_train))
# print("Base rate unprivileged: ", sum(updated_y_train.iloc[unpriv_idx_train])/len(unpriv_idx_train))


# norm
scaled_X_train = imputed_X_train.copy()

# od
k=30
outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=contamination_train_lof).fit_predict(scaled_X_train)
mask = outlier_y_pred != -1

outlier_X_train = scaled_X_train.copy()
outlier_y_train = updated_y_train.copy()

outlier_sensitive_train = updated_sensitive_attr_train.copy()
if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
    outlier_X_train, outlier_y_train, outlier_sensitive_train = scaled_X_train[mask], updated_y_train[mask], updated_sensitive_attr_train[mask]
    
    # outlier_y_train.reset_index(drop=True, inplace=True)
    # outlier_X_train.reset_index(drop=True, inplace=True)
    # outlier_sensitive_train.reset_index(drop=True, inplace=True)          
priv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 1]
unpriv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 0]

final_df = pd.DataFrame(outlier_X_train, columns = X_train.columns)
final_df['target'] = outlier_y_train

br_priv = ((final_df['Sex'] == 1) & (final_df['target'] == 1)).sum()/(final_df['Sex'] == 1).sum()
br_unpriv = ((final_df['Sex'] == 0) & (final_df['target'] == 1)).sum()/(final_df['Sex'] == 0).sum()
print(br_priv)
print(br_unpriv)

# model
updated_model = LogisticRegression(random_state=0).fit(outlier_X_train, outlier_y_train)

y_pred = updated_model.predict(outlier_X_train)
outc = accuracy_score(outlier_y_train, y_pred)
outc = computeStatisticalParity(y_pred[priv_idx_train],y_pred[unpriv_idx_train])

print(1 - outc)