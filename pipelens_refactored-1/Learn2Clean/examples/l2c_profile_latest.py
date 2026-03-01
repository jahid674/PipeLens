import itertools


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import time
import csv
import os
import pandas as pd
import numpy as np
from itertools import product,permutations
import random
from itertools import cycle
import sys
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression



from sklearn.naive_bayes import GaussianNB

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

from learn2clean.normalization.normalizer import Normalizer
from learn2clean.duplicate_detection.duplicate_detector import Duplicate_detector
from learn2clean.outlier_detection.outlier_detector import Outlier_detector
from learn2clean.consistency_checking.consistency_checker import Consistency_checker
from learn2clean.imputation.imputer import Imputer
from learn2clean.feature_selection.feature_selector import Feature_selector
from learn2clean.regression.regressor import Regressor
from learn2clean.clustering.clusterer import Clusterer
from learn2clean.classification.classifier import Classifier

import pandas as pd

class Reader():

    def __init__(self, train_path,test_path):
        self.train_path = train_path
        self.test_path = test_path


    def load_data(self):
        train = pd.DataFrame(pd.read_csv(self.train_path))
        test = pd.DataFrame(pd.read_csv(self.test_path))
        # import pdb;pdb.set_trace()
        return train,test

class Regression:
    def __init__(self):
        print("initializing regression")
    def generate_regression(self, xvals, yvals):
        model = LinearRegression().fit(xvals, yvals)
        return model
    def test(self):
        token_stats = pd.read_csv('ERmetrics/token_blocking.csv')
        X = token_stats[['blocking threshold', 'match threshold']]
        y = token_stats['f-score']
        model = self.generate_regression(X, y)
        print(model.coef_)
        print(model.intercept_)

tau_train = 0.05 # fraction of missing values
tau_test = 0.05
contamination_train = 0.2
contamination_test = 0.2

contamination_train_lof = 0.2
contamination_test_lof = 0.2

# contamination_train = 0.1
# contamination_test = 0.4

# tau_train = 0.05 # fraction of missing values
# tau_test = 0.4

# contamination_train_lof = 0.1
# contamination_test_lof = 0.4

# filtered_duplicate_list = [['IQR', 'None', 'None'], ['MICE', 'LOF', 'AD'], ['IQR', 'MICE', 'None'], ['None', 'IQR', 'KNN'], ['IQR', 'MICE', 'ED'], ['None', 'EM', 'None'], ['LOF', 'KNN', 'AD'], ['LOF', 'MF', 'ED'], ['EM', 'ZSB', 'ED'], ['None', 'EM', 'ED'], ['IQR', 'None', 'KNN'], ['LOF', 'None', 'AD'], ['MICE', 'None', 'AD'], ['IQR', 'None', 'EM'], ['MF', 'LOF', 'AD'], ['None', 'AD', 'MICE'], ['MF', 'None', 'ED'], ['IQR', 'EM', 'AD'], ['LOF', 'KNN', 'ED'], ['AD', 'IQR', 'MICE'], ['IQR', 'KNN', 'ED'], ['MICE', 'None', 'None'], ['LOF', 'None', 'ED'], ['LOF', 'MICE', 'ED'], ['IQR', 'None', 'ED'], ['MICE', 'ZSB', 'AD'], ['None', 'LOF', 'AD'], ['None', 'AD', 'None'], ['IQR', 'EM', 'None'], ['IQR', 'EM', 'ED'], ['None', 'None', 'MICE'], ['AD', 'None', 'MICE'], ['None', 'MICE', 'AD'], ['MICE', 'LOF', 'ED'], ['LOF', 'AD', 'MICE'], ['IQR', 'AD', 'MICE'], ['None', 'AD', 'KNN'], ['LOF', 'EM', 'AD'], ['EM', 'LOF', 'AD'], ['None', 'AD', 'EM'], ['None', 'MF', 'AD'], ['AD', 'IQR', 'KNN'], ['None', 'MICE', 'None'], ['None', 'IQR', 'MICE'], ['AD', 'IQR', 'EM'], ['None', 'AD', 'MF'], ['MICE', 'None', 'ED'], ['MF', 'LOF', 'ED'], ['EM', 'None', 'AD'], ['KNN', 'LOF', 'AD'], ['None', 'MF', 'None'], ['AD', 'IQR', 'MF'], ['LOF', 'EM', 'ED'], ['EM', 'LOF', 'ED'], ['None', 'KNN', 'AD'], ['None', 'MF', 'ED'], ['IQR', 'None', 'MICE'], ['AD', 'IQR', 'None'], ['None', 'None', 'EM'], ['None', 'None', 'AD'], ['MF', 'ZSB', 'AD'], ['AD', 'None', 'EM'], ['EM', 'None', 'None'], ['MICE', 'ZSB', 'ED'], ['LOF', 'MF', 'AD'], ['None', 'LOF', 'ED'], ['KNN', 'LOF', 'ED'], ['EM', 'ZSB', 'AD'], ['KNN', 'None', 'AD'], ['None', 'KNN', 'None'], ['IQR', 'AD', 'EM'], ['IQR', 'MF', 'AD'], ['None', 'None', 'MF'], ['None', 'KNN', 'ED'], ['AD', 'None', 'MF'], ['LOF', 'AD', 'MF'], ['AD', 'None', 'None'], ['IQR', 'AD', 'MF'], ['None', 'None', 'ED'], ['MF', 'ZSB', 'ED'], ['None', 'IQR', 'EM'], ['None', 'ZSB', 'AD'], ['MF', 'None', 'AD'], ['None', 'MICE', 'ED'], ['LOF', 'AD', 'None'], ['KNN', 'None', 'None'], ['IQR', 'MF', 'None'], ['IQR', 'AD', 'None'], ['KNN', 'None', 'ED'], ['IQR', 'KNN', 'AD'], ['KNN', 'ZSB', 'AD'], ['IQR', 'MF', 'ED'], ['None', 'IQR', 'MF'], ['LOF', 'MICE', 'AD'], ['None', 'None', 'KNN'], ['IQR', 'None', 'AD'], ['IQR', 'MICE', 'AD'], ['None', 'IQR', 'None'], ['AD', 'None', 'KNN'], ['MF', 'None', 'None'], ['None', 'ZSB', 'ED'], ['LOF', 'AD', 'KNN'], ['IQR', 'KNN', 'None'], ['IQR', 'AD', 'KNN'], ['EM', 'None', 'ED'], ['LOF', 'AD', 'EM'], ['IQR', 'None', 'MF'], ['None', 'EM', 'AD'], ['KNN', 'ZSB', 'ED']]
filtered_duplicate_list =[['LOF', 'KNN', 'AD'], ['AD', 'IQR', 'None'], ['LOF', 'MICE', 'AD'], ['AD', 'None', 'KNN'], ['IQR', 'AD', 'MICE'], ['IQR', 'MF', 'AD'], ['None', 'KNN', 'ED'], ['LOF', 'AD', 'MICE'], ['LOF', 'MF', 'AD'], ['IQR', 'AD', 'EM'], ['None', 'AD', 'MF'], ['EM', 'None', 'ED'], ['None', 'None', 'MF'], ['LOF', 'AD', 'EM'], ['None', 'EM', 'AD'], ['IQR', 'None', 'None'], ['IQR', 'MICE', 'None'], ['MF', 'ZSB', 'ED'], ['EM', 'ZSB', 'AD'], ['None', 'MF', 'AD'], ['MICE', 'None', 'None'], ['None', 'EM', 'ED'], ['IQR', 'EM', 'None'], ['None', 'ZSB', 'AD'], ['None', 'IQR', 'KNN'], ['MICE', 'None', 'AD'], ['None', 'MF', 'None'], ['MF', 'LOF', 'ED'], ['AD', 'IQR', 'KNN'], ['EM', 'LOF', 'AD'], ['AD', 'IQR', 'EM'], ['None', 'IQR', 'MF'], ['None', 'LOF', 'AD'], ['IQR', 'AD', 'KNN'], ['LOF', 'AD', 'KNN'], ['IQR', 'None', 'MICE'], ['IQR', 'None', 'AD'], ['IQR', 'MICE', 'AD'], ['KNN', 'ZSB', 'AD'], ['IQR', 'KNN', 'ED'], ['LOF', 'None', 'AD'], ['IQR', 'None', 'EM'], ['LOF', 'KNN', 'ED'], ['AD', 'None', 'MF'], ['IQR', 'AD', 'MF'], ['LOF', 'MICE', 'ED'], ['MF', 'None', 'None'], ['None', 'LOF', 'ED'], ['IQR', 'MF', 'ED'], ['MICE', 'ZSB', 'AD'], ['LOF', 'AD', 'MF'], ['LOF', 'MF', 'ED'], ['IQR', 'EM', 'AD'], ['IQR', 'None', 'ED'], ['LOF', 'EM', 'AD'], ['KNN', 'LOF', 'AD'], ['LOF', 'None', 'ED'], ['None', 'KNN', 'None'], ['None', 'None', 'MICE'], ['None', 'None', 'AD'], ['None', 'MICE', 'AD'], ['MICE', 'LOF', 'AD'], ['None', 'AD', 'None'], ['EM', 'ZSB', 'ED'], ['IQR', 'EM', 'ED'], ['None', 'None', 'None'], ['None', 'MICE', 'None'], ['LOF', 'EM', 'ED'], ['None', 'MF', 'ED'], ['AD', 'None', 'None'], ['None', 'ZSB', 'ED'], ['MICE', 'None', 'ED'], ['AD', 'IQR', 'MF'], ['IQR', 'None', 'KNN'], ['MF', 'None', 'AD'], ['EM', 'LOF', 'ED'], ['KNN', 'None', 'AD'], ['KNN', 'None', 'None'], ['IQR', 'None', 'MF'], ['None', 'KNN', 'AD'], ['IQR', 'MICE', 'ED'], ['MF', 'None', 'ED'], ['KNN', 'ZSB', 'ED'], ['None', 'AD', 'MICE'], ['None', 'IQR', 'None'], ['None', 'AD', 'EM'], ['MICE', 'ZSB', 'ED'], ['EM', 'None', 'None'], ['AD', 'None', 'MICE'], ['None', 'None', 'EM'], ['IQR', 'KNN', 'None'], ['MF', 'ZSB', 'AD'], ['AD', 'None', 'EM'], ['KNN', 'LOF', 'ED'], ['IQR', 'MF', 'None'], ['IQR', 'AD', 'None'], ['None', 'None', 'ED'], ['None', 'MICE', 'ED'], ['LOF', 'AD', 'None'], ['MICE', 'LOF', 'ED'], ['MF', 'LOF', 'AD'], ['None', 'EM', 'None'], ['None', 'IQR', 'MICE'], ['KNN', 'None', 'ED'], ['EM', 'None', 'AD'], ['AD', 'IQR', 'MICE'], ['None', 'IQR', 'EM'], ['None', 'AD', 'KNN'], ['IQR', 'KNN', 'AD'], ['None', 'None', 'KNN']]
# knn_k = 1 # knn number of neighbors
# lof_k = 50 # number of neighbors for local outlier factor
knn_k_lst = [4]
lof_k_lst = [4]
len_knn = len(knn_k_lst)
len_lof = len(lof_k_lst)
# norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm'] # standard scaler, robust scaler, max absolute scaler, minmax scaler
norm_strategy = ['none', 'ss', 'ma', 'log10', 'mm'] 
mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']


# od_strategy = ['zs', 'iqr', 'if', 'lof'] # local outlier factor, z-score, interquartile range, isolation forest
od_strategy = ['none', 'if', 'lof'] # local outlier factor, isolation forest
base_strategy = mv_strategy+od_strategy
dataset = 'housing'
modelType = 'lr' #'lr' # 'nb' Logistic Regression or Gaussian Naive Bayes
metric_type = 'rmse'
algo_type = '2step'
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(filename='logs/l2c_'+algo_type+"_"+dataset+'_'+modelType+'_'+'metric_type'+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

valid_moves = []
column_names = ['Age','Workclass','fnlwgt','Education','Education_Num','Martial_Status','Occupation','Relationship','Race','Sex','Capital_Gain','Capital_Loss','Hours_per_week','Country','income']
categorical_cols = ['Age','Workclass', 'Education', 'Martial_Status', 'Relationship', 'Race', 'Sex','Hours_per_week','income']
from scipy.stats import pearsonr
class Profile:
  profile_lst=[]
  def __init__(self):
    pass
  def outlier(self,lst):
    mean=statistics.mean(lst)
    std=statistics.stdev(lst)
    count=0
    for v in lst:
      if v>mean+2*std or v<mean-2*std:
        count+=1
    return count*1.0/len(lst)
  def missing(self,lst):
    count = 0
    for v in lst:
      try:
        if np.isnan(v):
          count += 1
      except:
        if len(v) == 0:
          count += 1
    return count*1.0/len(lst)
  

  def correlation(self,lst1,lst2):
    try:
        (r,p)= pearsonr(lst1,lst2)
        if True:
                return r
        else:
                return 0
    except:
           print("efe")


    
  def categorical_correlation(self,lst1,lst2):
    cross_tab=pd.crosstab(lst1,lst2)
    chi2, p, dof, ex=chi2_contingency(cross_tab)
    return chi2


  def categorical_numerical_correlation(self,lst1,lst2):
    (chi2,p)=stats.f_oneway(lst1,lst2)
    return chi2
  def get_fraction_of_outlier(self,data):
        svm_model = OneClassSVM(kernel='rbf')  # You can adjust the parameters as needed
        svm_model.fit(data)

        # Step 2: Predict the labels of your data points
        predicted_labels = svm_model.predict(data)

        # Step 3: Count the number of predicted outliers
        n_outliers = (predicted_labels == -1).sum()

        # Step 4: Calculate the fraction of outliers
        fraction_outliers = n_outliers / len(data)
        return fraction_outliers
  

  def populate_profiles(self,data_final,outlier):
    scaling_factor = 1
    
    profile_map={}
    # import pdb;pdb.set_trace()


    categorical_values={}
    #Partition each column as categorical, numerical and textual
    #Each profilehas four parameters where 3rd is conditional attribute 4th is value
#     le = LabelEncoder()
    
#     for column in categorical_columns:
      
#       data_final[column] = le.fit_transform(data_final[column]) 
      
    profile = {}
    i = 0
    # import pdb;pdb.set_trace()
    
    if(dataset == 'hmda'):
       target = 'action_taken'
    elif dataset =='adult':
       target = 'income'
    elif dataset =='housing':
        target = 'SalePrice'

       
    
    for column in data_final.columns:

        if(column==target):
          
          continue
        if(metric_type=='rmse' or metric_type=='mae'):
                if column in numerical_columns :
                        #pearson -  regression 
                        corr = self.correlation(data_final[column],data_final[target])
                else:
                #categorical_numerical_correlation - regression
                        corr = self.categorical_numerical_correlation(data_final[column],data_final[target])
        else:
                if column in numerical_columns :
                       corr = self.categorical_numerical_correlation(data_final[column],data_final[target])
                else:
                        #categorical_numerical_correlation - regression
                        corr = self.categorical_correlation(data_final[column],data_final[target])
               
        # missing_value = self.missing(self.df[categorical_columns[i]])
        #outlier  = self.outlier(self.df[categorical_columns[i]])
        
        name = column
        tuple = ('corr_' + name)
        profile[tuple]= [column,round(corr*scaling_factor,5)]
        i+=1
    dd = []
    keys = []
    for val in profile:
        # import pdb;pdb.set_trace()
        if(profile[val][1] is None):
               import pdb;pdb.set_trace()
        dd.append(profile[val][1])
        keys.append(val)
    return dd,keys


numerical_columns = []
if dataset == 'hmda':
    hmda_train  = "data/hmda/hmda_Orleans_X_train_1.csv"
    hmda_test = "data/hmda/hmda_Calcasieu_X_test_1.csv"

    
    # hmda_train = "data/hmda/hmda_Calcasieu_X_test_1.csv"
    train,test = Reader(hmda_train,hmda_test).load_data()
    y_train = train['action_taken']
    X_train = train.drop('action_taken', axis=1)

    y_test = test['action_taken']
    X_test = test.drop('action_taken', axis=1)
    tau_train = 0.05 # fraction of missing values
    tau_test = 0.1
    contamination_train = 0.1
    contamination_test = 0.1
    contamination_train_lof = 0.1
    contamination_test_lof = 0.1
elif dataset == 'adult':
    adult_train  = "data/adult/adult_test.csv"
    adult_test = "data/adult/adult_train.csv"
    train,test = Reader(adult_train,adult_test).load_data()
    #train,test =  train[categorical_cols],test[categorical_cols]
    categorical_columns = train.select_dtypes(include=['object']).columns
    numerical_columns = train.select_dtypes(include=['int', 'float']).columns
    label_encoders = {}
    label_encoder = LabelEncoder()
    
    for column in categorical_columns:
        le = LabelEncoder()
        train[column] = label_encoder.fit_transform(train[column]) 
        test[column] = label_encoder.fit_transform(test[column]) 
        label_encoders[column] = le

    
    y_train = train['income']
    X_train = train.drop('income', axis=1)
    
    y_test = test['income']
    X_test = test.drop('income', axis=1)
    unique_labels = np.unique(y_test)
    print("Unique labels in y_test:", unique_labels)
    tau_train = 0.1 # fraction of missing values
    tau_test = 0.1
    contamination_train = 0.2
    contamination_test = 0.2
    contamination_train_lof = 'auto'
    contamination_test_lof = 'auto'
elif dataset == 'housing':
    house_train  = "data/house/house_train.csv"
    house_test = "data/house/test.csv"
    train,test = Reader(house_train,house_test).load_data()

    missing_percentage = (train.isnull().sum() / len(train)) * 100
    columns_to_drop = missing_percentage[missing_percentage > 40].index.tolist()
    train.drop(columns=columns_to_drop, inplace=True)

    missing_values_count = train.isnull().sum()
    total_cells = np.product(train.shape)
    total_missing = missing_values_count.sum()
    missing_percentage = (total_missing / total_cells) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values_count,
                        'Percentage': (missing_values_count / train.shape[0]) * 100})
    print("Total missing values:", total_missing)
    print("Percentage of missing values:", missing_percentage)
    print("\nMissing value count and percentage per column:")
#     logging.warning(missing_data)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(missing_data)
    missing_colmn_categorical = ['Electrical','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','GarageType','GarageFinish','GarageQual','GarageCond']
    missing_colmn_numerical  = ['LotFrontage','GarageYrBlt','MasVnrArea']
#     for column in missing_colmn_categorical:
#         most_frequent = train[column].mode()[0]
#         train[column].fillna(most_frequent, inplace=True)
#     for column in missing_colmn_numerical:
#         median_value = train[column].median()
#         train[column].fillna(median_value, inplace=True)
    missing_values_count = train.isnull().sum()
    total_cells = np.product(train.shape)
    total_missing = missing_values_count.sum()
    missing_percentage = (total_missing / total_cells) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values_count,
                        'Percentage': (missing_values_count / train.shape[0]) * 100})
    print("Total missing values:", total_missing)
    print("Percentage of missing values:", missing_percentage)
    print("\nMissing value count and percentage per column:")
#     logging.warning(missing_data)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(missing_data)
#     y_train = X_train['SalePrice']
#     y_test = X_test['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(train, train, test_size=0.6, random_state=10)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    categorical_columns = train.select_dtypes(include=['object']).columns
    numerical_columns = train.select_dtypes(include=['int','float']).columns
    
#     label_encoder = LabelEncoder()
#     for column in categorical_columns:
#         le = LabelEncoder()
#         X_train[column] = label_encoder.fit_transform(y_train[column]) 
#         X_test[column] = label_encoder.fit_transform(y_test[column]) 
    y_train = X_train['SalePrice']
    y_test = X_test['SalePrice']

    X_train = X_train.drop('SalePrice', axis=1)
    X_test = X_test.drop('SalePrice', axis=1)
#     selector = SelectKBest(score_func=mutual_info_regression, k=15)
#     selector.fit(X_train, y_train)


#     selected_indices = selector.get_support(indices=True)
#     selected_features = X_train.columns[selected_indices]
#     feature_scores = selector.scores_
#     selected_features_scores = pd.DataFrame({'Feature': X_train.columns, 'Score': feature_scores})
#     selected_features_scores.sort_values(by='Score', ascending=False, inplace=True)
#     print(selected_features_scores)
#     selected_features_set = set(selected_features)
#     # Filter the DataFrame to include only the scores of the selected features
#     selected_features_scores_filtered = selected_features_scores[selected_features_scores['Feature'].isin(selected_features_set)]
#     print(selected_features_scores_filtered)

#     label_encoder = LabelEncoder()
#     label_encoders = {}
#     X_train = X_train[selected_features]
#     X_test = X_test [selected_features]
# #     import pdb;pdb.set_trace()
    
    
    
#     categorical_columns = X_train.select_dtypes(include=['object']).columns
#     for column in categorical_columns:
#         le = LabelEncoder()
#         X_train[column] = label_encoder.fit_transform(X_train[column]) 
#         X_test[column] = label_encoder.fit_transform(X_test[column]) 
#         label_encoders[column] = le
    X_test_concat = pd.concat([X_test,y_test],axis=1)
    X_test_concat.to_csv('Learn2Clean/datasets/house/concatenated_data_test.csv', index=False)

# if(metric_type=='sp' or metric_type=='accuracy_score' or metric_type=='mae'  or metric_type=='rmse'):
#         if modelType == 'lr':
#                 model = LogisticRegression(random_state=0).fit(X_train, y_train)
#         elif modelType == 'nb':
#                 model = GaussianNB().fit(X_train, y_train)
#         elif modelType == 'rf':
#                 model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
#         elif modelType == 'reg':
#                 model = Regression().generate_regression(X_train, y_train)
#         print("Training accuracy : " + str(round(model.score(X_train, y_train), 4)))
#         print("Test accuracy : " + str(round(model.score(X_test, y_test), 4)))

#         y_pred_train = model.predict(X_train)
#         y_pred_test = model.predict(X_test)

#         # p_train = model.predict_proba(X_train)
#         # p_test = model.predict_proba(X_test)
#         if metric_type=='mae' or metric_type=='rmse':
#                 outc_train_mae = mean_absolute_error( y_train,y_pred_train)
#                 outc_test_mae = mean_absolute_error(y_test, y_pred_test)
#                 print(f'MAE training {outc_train_mae} ,test {outc_test_mae}')
#                 outc_train_mse = np.sqrt(root_mean_squared_error(y_train, y_pred_train)) 
#                 outc_test_mse = np.sqrt(root_mean_squared_error(y_test,y_pred_test)) 
#                 print(f'MSE training {outc_train_mse} ,test {outc_test_mse}')
#         print('Done')

class base:
        def __init__(self):
                self.f = 'tdst';
                self.ranking = None
                self.ranges = {}
                self.imputer_strategies = ['drop', 'mean', 'median', 'most_frequent', 'knn']
                self.mv_name_mapping = {'drop': 'mv_drop', 'mean': 'mv_mean', 'median': 'mv_median', 'most_frequent': 'mv_mode', 'knn': 'mv_knn'}
                self.fail = 0
                self.pass_ = 0
                self.fail_grid_search = 0
                #KNN 
                self.outlier_strategies = ['none', 'if', 'lof']
                self.ot_name_mapping = {'none': 'od_none', 'if': 'od_if', 'lof': 'od_lof'}

                #1.15 paramater 
                
                self.no_name_mapping = {'none': 'norm_none', 'ss': 'norm_ss', 'rs': 'norm_rs', 'ma': 'norm_ma', 'mm': 'norm_mm'}

                #normalization 
                #
 
                self.imputation = ["MICE", "EM", "KNN", "MF",'None']
                self.normalizer_strategies = ['DS','MM','ZS','None']
                self.feature_selection = ['MR','WR','LC','Tree','None']
                self.outlier_detection =  ["ZSB", "LOF", "IQR",'None']
                self.consist_checker =  ["CC", "PC"]
                self.duplicate_detector =  ["ED", "AD",'None']


                self.L2C_class = [Imputer, Imputer, Imputer, Imputer,
                                Normalizer, Normalizer, Normalizer,
                                Feature_selector, Feature_selector, Feature_selector,
                                Feature_selector,
                                Outlier_detector, Outlier_detector, Outlier_detector,
                                Consistency_checker, Consistency_checker,
                                Duplicate_detector, Duplicate_detector,
                                Regressor]
                
                self.ranges['module_1'] = [0,1,2]
                self.ranges['module_2'] = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14]
                self.ranges['module_3'] = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14]
                self.ranges['module_4'] = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14]

                self.base_strategies  =['module_0','module_1', 'module_2', 'module_3','module_4']
                

                self.class_map = {
                        "MICE": 0, "EM": 1, "KNN": 2, "MF": 3,
                        "ZSB": 4, "LOF": 5, "IQR": 6,
                        "CC": 7, "PC": 8,
                        "ED": 9, "AD": 10,'None':11,
                        'Reg':12,
                        "DS":0,"MM":1, "ZS":2,
                        "MR":0, "WR":1, "LC":2, "Tree":3
                        
                        }

                self.historical_data = []   
                self.historical_data_pd = []
                self.gs_idistr = []
                self.gs_fdistr = []
                self.k = {}
                self.k[0] = 2
                self.k[1] = 8
                self.k[2] = 4
                self.k[3] = 8
                self.k[4] = 6
                self.k[5] = 8
                self.k[6] = 8
                self.k[7] = 8
                self.sublist_1  = []
                self.sublist_2 = []
                self.sublist_3 = [] 
                self.column_name = ['normal','imputer','outlier_strategy' ]
                self.profile_dist = {}
                self.r = np.array([
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],

                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0,  0,  0,  0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],

                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],

                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],
                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],
                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],

                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, 0, 0, -1, -1,
                 0, 0, 100],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, 0, 0, -1, -1,
                 0, 0, 100],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1]]).astype("float32")
        def Log10_normalization(self, dataset):

                d = dataset



                if (type(dataset) != pd.core.series.Series):
                        
                        X = dataset.select_dtypes(['number'])

                        Y = dataset.select_dtypes(['object'])

                        Z = dataset.select_dtypes(['datetime64'])

                for column in X.columns:

                        X[column] = np.around(np.log10(X[column].max())) + 1

                df = X.join(Y)

                df = df.join(Z)

                X = dataset

                X -= X.mean()

                X /= X.std()

                df = X


                return df

        def MA_normalization(self, dataset):
                d = dataset
                if (type(dataset) != pd.core.series.Series):

                        X = dataset.select_dtypes(['number'])

                        Y = dataset.select_dtypes(['object'])

                        Z = dataset.select_dtypes(['datetime64'])

                for column in X.columns:

                        X[column] /= X[column].max()

                df = X.join(Y)

                df = df.join(Z)



                X = dataset

                X /= X.max()

                df = X


                return df.sort_index()
        def getIdxSensitive(self, df, dataset):
                if dataset == 'hmda':
                        priv_idx = df.index[df['race']==1]
                        unpriv_idx = df.index[df['race']==0]
                        sensitive_attr = df['race']
                elif dataset == 'adult':
                        priv_idx = df.index[df['Sex']==1]
                        unpriv_idx = df.index[df['Sex']==0]
                        sensitive_attr = df['Sex']
                                   

                        

                return priv_idx, unpriv_idx, sensitive_attr
        def computeStatisticalParity(self, p_priv, p_unpriv):
                # p_priv = pd.DataFrame(p_priv)[1]
                # p_unpriv = pd.DataFrame(p_unpriv)[1]
                diff = p_priv.mean() - p_unpriv.mean()
                return diff
        def inject_null_values(self):
                ## Injected missing values with 5% of data 
                np.random.seed(42)

                for col in ['lien_status']:
                        mask_train = np.random.rand(len(self.dataset['train'])) < 0.05
                        mask_test = np.random.rand(len(self.dataset['test'])) < 0.05
                
                
                self.dataset['train'].loc[mask_train, col] = np.nan
                
                self.dataset['test'].loc[mask_test, col] = np.nan

                #Imputed summary 
                print(self.dataset['train'].isnull().sum())
                print(self.dataset['test'].isnull().sum())
                ##end 

                
        def load_data(self):
                hmda_train  = "data/hmda/hmda_Orleans_X_train_1.csv"
                hmda_test = "data/hmda/hmda_Calcasieu_X_test_1.csv"
                train,test = Reader(hmda_train,hmda_test).load_data()

                self. dataset = {
                        'train': train,
                        'test': test,
                }

 
 
        
        def get_profile(self,profiles_df,elem,profile_index):
                # profile_index = profile_index +len(self.param_columns)
                column_names =['missing_value', 'normalization', 'outlier']
                # return elem[-1]
                # import pdb;pdb.set_trace()
                try:
                        return profiles_df.loc[(profiles_df[column_names[0]] == elem[0]) & (profiles_df[column_names[1]] == elem[1] ) 
                                               & (profiles_df[column_names[2]] == elem[2])
                                               
                                               ].iloc[0][profile_index]
                except Exception as e :
                        print(e)
                        # import pdb;pdb.set_trace()





        def remove_duplicates_and_check(self,lst):
                seen = set()
                result = []
                has_duplicates = False

                for item in lst:
                        # Convert item to a rounded value if it is a number
                        if isinstance(item, (int, float)):
                                rounded_item = round(item, 10)
                        else:
                                rounded_item = item
                        
                        if rounded_item in seen:
                                has_duplicates = True
                        else:
                                seen.add(rounded_item)
                        result.append(item)
                
                return  result,has_duplicates
        def optimize(self, init_params, f_goal):
                # logging.info(f'2step Starting Grasp search for fairness score:{f_goal}')
                #======Intitalize variable start here =====
                self.profile_dist = {}
                self.profile_dist['profile_outlier']= 0
                self.profile_dist['diff_sensitive_attr'] = 0
                self.profile_dist['ratio_sensitive_attr'] = 0
                # self.profile_dist['cov'] = 0
                # self.profile_dist['cov'] = 0
                self.rank_iter = 0  #Iteration count by ranking algorithm
                self.rank_f = 0    # Iteration count whn we found the fairness score less than the calculated fairness from seed value 
                iter_size = 0   # Total iteration allowed in ranking algorithm , falllback is after 1 to grid search 
                # cur_params = init_params.copy()
                logging.info(f'Initial parameter {init_params}')
                #========end=========
                # self.fail = 0
                # self.pass_ = 0
                
                #  From seed value calculate the new fairness score
                #=========Code========
                #                 norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm'] # standard scaler, robust scaler, max absolute scaler, minmax scaler
                # mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
                # # od_strategy = ['zs', 'iqr', 'if', 'lof'] # local outlier factor, z-score, interquartile range, isolation forest
                # od_strategy = ['none', 'if', 'lof'] # local outlier factor, isolation forest
                cur_params_opt = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params[:len(self.base_strategies)])}
                positive_index = [index for index, value in enumerate(cur_params_opt.values()) if value > 0]

                self.ranges['module_0'] = list(np.unique(self.historical_data_pd['module_0']))
                self.ranges['module_1'] = list(np.unique(self.historical_data_pd['module_1']))
                self.ranges['module_2'] = list(np.unique(self.historical_data_pd['module_2']))
                self.ranges['module_3'] = list(np.unique(self.historical_data_pd['module_3']))
                self.ranges['module_4'] = list(np.unique(self.historical_data_pd['module_4']))
                opt_f  = self.f_score_look_up2(self.historical_data_pd,init_params)

                # print(f'Fairness score from seed value : {opt_f}')
                #======Code end ============

                seen = set()
                
                seen.add(tuple(cur_params_opt.items()))
                
                                
                iter_size  = 0
                if(opt_f<f_goal):
                        self.rank_iter = 1  #Iteration count by ranking algorithm
                        self.rank_f = opt_f
                        # logging.info("Target achieved")
                        return 
                sorted_params = []

                       
                while(True):
                        threshold = 1e-3
                        for idx,profile_index in enumerate(self.profile_ranking):
                                if iter_size == 0 or idx!=0:
                                        # print(f'optimal {iter_size}')
                                        
                                        param_name  = self.profiles[profile_index]
                                        coef_rank = self.ranking_param[param_name]
                                        # import pdb;pdb.set_trace()
                                        logging.info(f'Current Profile name {param_name } : corresponding ranking parameter: {coef_rank}')
                                        for val in coef_rank:
                                                cur_strategy = self.base_strategies[val]
                                                # logging.info(f'Strategy selected  : {cur_strategy}')    
                                                if(self.profile_coefs[profile_index]>0):

                                                        if(self.param_coeff[param_name][val]>0 ):
                                                                current_paramter_value =  self.ranges[cur_strategy][0]
                                                        else:
                                                                current_paramter_value =  self.ranges[cur_strategy][-1]
                                                else:
                                                        if(self.param_coeff[param_name][val]<0 ):
                                                                current_paramter_value =  self.ranges[cur_strategy][0]
                                                        else:
                                                                current_paramter_value =  self.ranges[cur_strategy][-1]
                                                
                                                # logging.info(f'order of parameter  value: {current_paramter_value}')                
                                                # logging.warn(f'current Iteration before parameter selection {self.rank_iter}')
                                                cur_params = cur_params_opt.copy()
                                                cur_params[cur_strategy] = current_paramter_value
                                                # if(cur_strategy!='module_1'):
                                                #         for key, value in cur_params.items():
                                                #                 if key != cur_strategy and value == current_paramter_value and current_paramter_value not in [5,8,14]:
                                                #                         cur_params[cur_strategy] = self.class_map[self.get_none(current_paramter_value)]
                                                #                         break
                                                
                                                

                                                
                                                # logging.warn(f'Current iteration after parameter selection {self.rank_iter}')
                                                #profile_opt = self.get_profile(self.historical_data_pd,list(cur_params.values()),profile_index)
                                                
                                                #profile_cur = self.get_profile(self.historical_data_pd,list(cur_params.values()),profile_index)
                                                
                                                if(tuple(cur_params.items())) in seen  :
                                                        continue
                                                seen.add(tuple(cur_params.items()))
                                                
                                                # logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                                                # try:
                                                cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))
                                                logging.info(f'parameter {cur_params} ,Fairness :{cur_f}')
                                                # except:
                                                #        continue
                                                # self.profile_dist[param_name] +=1
                                                if(cur_f is None):
                                                       continue
                                                #if(profile_cur<profile_opt):
                                                self.rank_iter += 1

                                                # logging.info(f'updated fairness after parameter selection {cur_f}')
                                                if metric_type=='sp' or metric_type=='mae' or metric_type=='rmse' or metric_type=='accuracy_score':                                                                           
                                                        if cur_f <= f_goal:
                                                                self.rank_f = cur_f
                                                                self.pass_ = self.pass_  + 1
                                                                # logging.error("Target achieved")
                                                                return cur_params_opt.values() # early exit when f_goal obtained
                                                        elif cur_f < opt_f:
                                                                opt_f = cur_f
                                                                cur_params_opt = cur_params.copy()
                                                                logging.info(f'New parameter {cur_params_opt}')
                                                                
                                                                   
                                else :
                                        # if(iter_size==1 and self.rank_iter==13):
                                        #        print(seen)
                                        #        print(self.rank_iter)
                                        import operator
                                        # print(self.rank_iter)
                                        # if(self.rank_iter>6):
                                        #            print(f'fall back {self.rank_iter}')
                                        profile = self.profiles[profile_index] #name of profile
                                        coeff = self.profile_coefs[profile_index]<0
                                        
                                        #Co-ef comparison should  not be on true value 
                                        #missing_value,normalization,outlier,fairness
                                        map = {}
                                        for param in self.historical_data:
                                                map[param[0]*self.param_coeff[profile][0]
                                                +param[1]*self.param_coeff[profile][1]
                                                +param[2]*self.param_coeff[profile][2] 
                                                +param[3]*self.param_coeff[profile][3]]= param

                                        sorted_params = sorted(map.items(), key=operator.itemgetter(0))
                                        
                                        if(coeff):
                                                sorted_params.reverse()
                                        # print(sorted_params[iter_size-1])
                                        cur_params = cur_params_opt.copy()
                                        for id,val in enumerate(self.base_strategies):
                                        #        print(iter_size)
                                               cur_params[val] = sorted_params[iter_size-1][1][id]
                                               
                                        if(tuple(cur_params.items())) in seen:
                                               continue
                                        seen.add(tuple(cur_params.items()))
                                        cur_f = sorted_params[iter_size-1][1][-1]
                                        self.rank_iter += 1

                                        # logging.info(f'updated fairness after parameter selection {cur_f}')
                                                
                                        if cur_f <= f_goal:
                                                self.rank_f = cur_f
                                                self.pass_ = self.pass_  + 1
                                                # logging.error("Target achieved")
                                                return  # early exit when f_goal obtained
                                        elif cur_f < opt_f:
                                                opt_f = cur_f
                                                cur_params_opt = cur_params

                                        self.fail = self.fail  + 1
                        iter_size +=1
                        if iter_size>len(self.historical_data):
                           print('Not able to find ')
                           break


        def f_score_look_up2(self,profiles_df,elem):
                column_names =['module_0','module_1', 'module_2', 'module_3', 'module_4','fairness']
                # return elem[-1]
                # import pdb;pdb.set_trace()
                try:
                        return profiles_df.loc[(profiles_df[column_names[0]] == elem[0]) & (profiles_df[column_names[1]] == elem[1] ) 
                                               & (profiles_df[column_names[2]] == elem[2]) 
                                             
                                               
                                               ].iloc[0]['fairness']
                except Exception as e :
                        # print(e)
                        pass
                        # import pdb;pdb.set_trace()
                return None


        def grid_search(self, f_goal, iterations,seen):
                self.gs_idistr = []
                self.gs_fdistr = []
                gs_iter = 0     
                gs_f = 0
                for i in range(iterations):
                        gs_iter = 0
                        gs_f = 0
                        cur_order = self.historical_data
                        random.shuffle(cur_order)
                        for elem in cur_order:
                                cur_params = {strategy: selection for strategy, selection in zip(self.base_strategies, elem[:len(self.base_strategies)])}


                                if(tuple(cur_params.items())) in seen:
                                        continue
                                
                                seen.add(tuple(cur_params.items()))
                                cur_f = self.f_score_look_up2(self.historical_data_pd,elem)
                                gs_iter += 1
                                if metric_type=='sp' or metric_type=='mae' or metric_type=='rmse' or metric_type=='accuracy_score':
                                        if cur_f < f_goal:
                                                gs_f = cur_f
                                                self.gs_fdistr.append(gs_f)
                                                self.gs_idistr.append(gs_iter)
                                                return gs_iter,gs_f
                                elif metric_type=='f-1':
                                        if cur_f >= f_goal:
                                                self.gs_fdistr.append(gs_f)
                                                self.gs_idistr.append(gs_iter)
                                                return gs_iter,gs_f


        def get_none(self,value):
                if value in [0,1,2,3,4] :
                        return 'None_imp'
                elif value in [5,6,7,8]:
                        return 'None_od'
                elif value in [12,13,14]:
                        return 'None_duplicate'
                else:
                        import pdb;pdb.set_trace()
        def get_valid_moves(self,traverse_tuple):
                class_map = {
                        "MICE": 0, "EM": 1, "KNN": 2, "MF": 3,'None_imp':4,
                        "ZSB": 5, "LOF": 6, "IQR": 7,'None_od':8,
                        "CC": 9, "PC": 10,"None_cons":11,
                        "ED": 12, "AD": 13,'None_duplicate':14,
                        'Reg':15,
                        "DS":0,"MM":1, "ZS":2,
                        "MR":3, "WR":4, "LC":5, "Tree":6,
                        }
                

                       
                list_of_mov = []
                for i in range(19):
                        current_mov = []

                        a = self.r[i,self.class_map[traverse_tuple[0]]]
                        b = self.r[i,self.class_map[traverse_tuple[1]]]
                        c = self.r[i,self.class_map[traverse_tuple[2]]]

                        if(a>-1):
                                current_mov.append(traverse_tuple[0])
                        else :
                                current_mov.append(self.get_none(class_map[traverse_tuple[0]]))
                        

                        if(b>-1):
                                current_mov.append(traverse_tuple[1])
                        else:
                                current_mov.append(self.get_none(class_map[traverse_tuple[0]]))
                        
                        if(c>-1):
                                current_mov.append(traverse_tuple[2])
                        else:
                               current_mov.append(self.get_none(class_map[traverse_tuple[0]]))
                        list_of_mov.append(current_mov)
                return list_of_mov  # All moves were valid


        def remove_duplicates(self,arrays):
        # Convert each list to a tuple so they can be added to a set
                unique_tuples = set(tuple(array) for array in arrays)
                
                # Convert tuples back to lists
                unique_lists = [list(tup) for tup in unique_tuples]
                
                return unique_lists


        def add_missing_feature_with_zero(self,data):
                import pdb;pdb.set_trace()
                for feature in data.columns : 
                       #["module_1","module_2","module_3","module_4"]:
                       if feature not in data:
                              data[feature] =0


                return data

        def create_historic_data(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
                key_profile = []
                p = Profile()
                from collections import OrderedDict
                profile_names = {}
                class_map = self.class_map
        
                L2C_class_map = {  "MICE": 0, "EM": 1, "KNN": 2, "MF": 3,
                        "DS":4,"MM":5, "ZS":6,
                        "MR":7, "WR":8, "LC":9, "Tree":10,
                        "ZSB": 11, "LOF": 12, "IQR": 13,
                        "CC": 14, "PC": 15,
                        "ED": 16, "AD": 17,
                        'Reg':18}
                self.verbose = False
                # if True:

                        # Generate the Cartesian product of the lists
                cartesian_product = list(product( self.outlier_detection, self.duplicate_detector, self.imputation))

                # Generate all permutations of each combination in the Cartesian product
                # all_combinations_with_permutations = []
                # for combination in cartesian_product:
                #         all_combinations_with_permutations.extend(permutations(combination))
                
                
                # for val in all_combinations_with_permutations:
                #         for filter_tuple in self.get_valid_moves(val):
                #                 valid_moves.append(filter_tuple)
                        
                # filtered_duplicate_list =self.remove_duplicates(valid_moves)

                # cartesian_product_feature_selection_and_normalization = self.normalizer_strategies +self.feature_selection


                if not(os.path.exists(file_name)):
                        idx_train = np.arange(0, len(X_train), 1)
                        mv_train = pd.DataFrame(idx_train).sample(frac=tau_train, replace=False, random_state=1).index
                        X_train['OverallQual'][mv_train] = np.NaN

                        # idx_test = np.arange(0, len(X_test), 1)
                        # mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
                        # X_test['lien_status'][mv_test] = np.NaN
                        # # X_test['income_brackets'][mv_test] = np.NaN

                        params_metrics = []
                        print("Running pipeline combinations ...")
                        

                        mv_param = ''
                        norm_param = ''
                        od_param = ''

                        param_lst = []
                        sens_attr_name = ''
                        target_variable_name = ''
                        
                       
                        
                        



                        for norm_fs in self.normalizer_strategies:
                                X_train_transform = X_train.copy()
                                y_train_transform = y_train.copy()
                                dataset_norm = {'test':X_train_transform,
                                   'target':y_train_transform,
                                   'SalePrice':y_train_transform,
                                   }
                                if(norm_fs.find('None')<0):
                                        n =self.L2C_class[L2C_class_map[norm_fs]](dataset=dataset_norm, strategy=norm_fs,
                                                        exclude='SalePrice',
                                                        verbose=self.verbose).transform()
                                for feature_selection in self.feature_selection:
                                        X_train_transform = dataset_norm['test'].copy()
                                        y_train_transform = dataset_norm['SalePrice'].copy()
                                        dataset_feature = {'test':X_train_transform,
                                        'target':y_train_transform,
                                        'SalePrice':y_train_transform,
                                        }
                                        if(feature_selection.find('None')<0):
                                                n =self.L2C_class[L2C_class_map[feature_selection]](dataset=dataset_feature, strategy=feature_selection,
                                                                exclude='SalePrice',
                                                                verbose=self.verbose).transform()

                                

                                        for transformers in  filtered_duplicate_list:   
                                                # tranformer_comb = ('knn','lof')
                                                dataset_fs_norm = dataset_feature.copy()
                                                for transformer in transformers:
                                                        if(transformer.find('None')<0):
                                                                # continue
                                                                print(transformer)
                                                                try:
                                                                        n = self.L2C_class[L2C_class_map[transformer]](dataset=dataset_fs_norm, strategy=transformer,
                                                                        exclude='SalePrice',
                                                                        
                                                                        verbose=self.verbose).transform()
                                                                except:
                                                                        n = self.L2C_class[L2C_class_map[transformer]](dataset=dataset_fs_norm, strategy=transformer,
                                                                                exclude='SalePrice',
                                                                                file_name='house_discovered',
                                                                                verbose=self.verbose).transform()
                                                
                                                outc = None
                                                try:
                                                        dataset_fs_norm['test'] = dataset_fs_norm['test'].drop('New_ID', axis=1)
                                                except:
                                                        pass
                                                if dataset_fs_norm['test'].shape[0]!=0:
                                                        outc = self.L2C_class[18](dataset=dataset_fs_norm,strategy='MARS',target='SalePrice',verbose=self.verbose).transform()['quality_metric']
                                                else:
                                                        continue
                                                f = [outc]
                                                # param_lst.append(mv_param + norm_param + od_param + f)
                                                #dataset_fs_norm['test'] = self.add_missing_feature_with_zero(dataset_fs_norm['test'])
                                                conc_list = pd.concat([dataset_fs_norm['test'], dataset_fs_norm['target']], axis=1)
                                                ordered_dict = {}
                                                profile_gen,key_profile = p.populate_profiles(conc_list,0.2)
                                                profile_median = dataset_fs_norm['target'].median()
                                                
                                                module_0 = [self.class_map[norm_fs]]
                                                module_1 = [self.class_map[feature_selection]]
                                                module_2 = [self.class_map[transformers[0]]]
                                                module_3 = [self.class_map[transformers[1]]]
                                                module_4 = [self.class_map[transformers[2]]]
                                                
                                                ordered_dict['module_0'] = module_0[0]
                                                ordered_dict['module_1'] = module_1[0]
                                                ordered_dict['module_2'] = module_2[0]
                                                ordered_dict['module_3'] = module_3[0]
                                                ordered_dict['module_4'] = module_4[0]
                                                ordered_dict['fairness'] = outc

                                                ordered_dict['profile_median'] = profile_median


                                                for it_r in range(len(key_profile)):
                                                        ordered_dict[key_profile[it_r]] = profile_gen[it_r]
                                                        profile_names[key_profile[it_r]] = 1
                                        
                                                # profile_names['module_1'] = module_1
                                                # profile_names['module_2'] = module_2
                                                # profile_names['module_3'] = module_3
                                                # profile_names['module_4'] = module_4
                                                # profile_names['f'] = f
                                                
                                                # param_lst.append(module_1 + module_2 + module_3  +module_4+profile_median+profile_gen+f)
                                                param_lst.append(ordered_dict)
                                                
                                                print(str(module_0+module_1 + module_2 + module_3 +module_4 +f))

                        

                        # param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        param_column = ["module_0","module_1","module_2","module_3","module_4"]
                        profiles  = param_column+ list(profile_names.keys())+['profile_median','fairness']
                        param_lst_new = []
                        for param in param_lst:
                               temp = []
                               for profile_itr  in  profiles:
                                      if profile_itr in param:
                                        #      if profile_itr =='corr_YrSold':
                                                # temp.append(param[profile_itr][0])
                                        #      else:
                                                temp.append(param[profile_itr])
                                      else:
                                        temp.append(0)
                                
                               param_lst_new.append(temp)
                        param_lst = param_lst_new
                        
                               
                               
                        if(metric_type=='sp'):
                                param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['out_before_out_strat','out_before_norm_strat',"diff_sensitive_attr","ratio_sensitive_attr","cov","class_imbalance_ratio"]+key_profile+["fairness"])
                        else:

                               param_lst_df = pd.DataFrame(param_lst, columns= profiles)
                               

                        param_lst_df.to_csv(file_name, index=False)
                        
                else :
                        param_lst_df = pd.read_csv(file_name)
                #//ocuupation,education,maritial,education,sex
                self.param_columns = ["module_0","module_1","module_2","module_3","module_4"]
                
                #key_profile = 
                # ['profile_median','corr_Neighborhood','corr_GrLivArea','corr_TotalBsmtSF','corr_GarageArea','corr_GarageYrBlt','corr_1stFlrSF','corr_YearBuilt','corr_MSSubClass','corr_OverallQual','corr_ExterQual','corr_BsmtQual','corr_2ndFlrSF','corr_FullBath','corr_KitchenQual','corr_GarageCars']
                #key_profile = ['corr_MSSubClass', 'corr_Neighborhood',  'corr_OverallQual', 'corr_YearBuilt', 'corr_ExterQual', 'corr_BsmtQual',  'corr_TotalBsmtSF',  'corr_1stFlrSF', 'corr_GrLivArea',  'corr_FullBath','corr_KitchenQual',  'corr_GarageYrBlt','corr_GarageCars', 'corr_GarageArea' ]
                self.profiles = param_lst_df.copy().drop(param_lst_df.columns[:4], axis=1).columns#+[ 'out_before_out_strat','out_before_norm_strat']
                
                #rank profile first
                # if(dataset=='housing'):
                #         t = StandardScaler().fit(param_lst_df).transform(param_lst_df)
                #         param_lst_df = pd.DataFrame(data=t,columns=param_lst_df.columns)
                param_lst_df = param_lst_df.fillna(0)
                y = param_lst_df['fairness']
                X = param_lst_df.copy()[self.profiles]
                X = StandardScaler().fit(X).transform(X)

                reg = Regression()
                try:
                        model = reg.generate_regression(X, y)
                except:
                        import pdb;pdb.set_trace()
                coefs = model.coef_
                print(coefs)
                
                print(model.intercept_)
                
                self.profile_ranking = np.argsort(np.abs(coefs))[::-1]
                self.profile_coefs = coefs

                #ranking parameter 
                self.ranking_param ={}
                self.param_coeff  = {}
                for index, elem in enumerate(self.profiles):
                        y = param_lst_df[elem]
                        
                        X = param_lst_df.copy()[self.param_columns]
                        # X = StandardScaler().fit(X).transform(X)
                        reg = Regression()
                        model = reg.generate_regression(X, y)
                        coefs = model.coef_
                        # print(model.intercept_)
                        self.ranking_param[elem] =  np.argsort(np.abs(coefs))[::-1]
                        print(self.ranking_param[elem])
                        
                        self.param_coeff[elem] =  coefs
                        print(f'name : {elem} {self.param_coeff[elem]}')

                for idx,profile_index in enumerate(self.profile_ranking):
                        print(self.profiles[profile_index])
                print('33')
                
                        

                                      

#GPU 
                
        

        def create_historic_data_test(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
                key_profile = []
                p = Profile()
           

        
                L2C_class_map = {  "MICE": 0, "EM": 1, "KNN": 2, "MF": 3,
                        "DS":4,"MM":5, "ZS":6,
                        "MR":7, "WR":8, "LC":9, "Tree":10,
                        "ZSB": 11, "LOF": 12, "IQR": 13,
                        "CC": 14, "PC": 15,
                        "ED": 16, "AD": 17,
                        'Reg':18}
                self.verbose = False
                
                if not(os.path.exists(file_name)):
                        idx_test = np.arange(0, len(X_test), 1)
                        mv_test = pd.DataFrame(idx_test).sample(frac=tau_test, replace=False, random_state=1).index
                        X_test['OverallQual'][mv_test] = np.NaN

                        # idx_test = np.arange(0, len(X_test), 1)
                        # mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
                        # X_test['lien_status'][mv_test] = np.NaN
                        # # X_test['income_brackets'][mv_test] = np.NaN

                        params_metrics = []
                        print("Running pipeline combinations ...")
                        

                        param_lst = []

                        # Generate the Cartesian product of the lists
                        # cartesian_product = list(product( self.outlier_detection, self.duplicate_detector, self.imputation))

                        # # Generate all permutations of each combination in the Cartesian product
                        # all_combinations_with_permutations = []
                        # for combination in cartesian_product:
                        #         all_combinations_with_permutations.extend(permutations(combination))
                        
                        # valid_moves = []
                        # for val in all_combinations_with_permutations:
                        #         for filter_tuple in self.get_valid_moves(val):
                        #                 valid_moves.append(filter_tuple)
                                
                        # filtered_duplicate_list =self.remove_duplicates(valid_moves)

                        # cartesian_product_feature_selection_and_normalization = self.normalizer_strategies +self.feature_selection


                        for norm_fs in self.normalizer_strategies:
                                X_test_transform = X_test.copy()
                                y_test_transform = y_test.copy()
                                dataset_norm = {'test':X_test_transform,
                                   'target':y_test_transform,
                                   'SalePrice':y_test_transform,
                                   }
                                if(norm_fs.find('None')<0):
                                        n =self.L2C_class[L2C_class_map[norm_fs]](dataset=dataset_norm, strategy=norm_fs,
                                                        exclude='SalePrice',
                                                        verbose=self.verbose).transform()
                                for feature_selection in self.feature_selection:
                                        X_test_transform = dataset_norm['test'].copy()
                                        y_test_transform = dataset_norm['SalePrice'].copy()
                                        dataset_feature = {'test':X_test_transform,
                                        'target':y_test_transform,
                                        'SalePrice':y_test_transform,
                                        }
                                        if(feature_selection.find('None')<0):
                                                n =self.L2C_class[L2C_class_map[feature_selection]](dataset=dataset_feature, strategy=feature_selection,
                                                                exclude='SalePrice',
                                                                verbose=self.verbose).transform()

                                

                                        for transformers in  filtered_duplicate_list:   
                                                # tranformer_comb = ('knn','lof')
                                                dataset_fs_norm = dataset_feature.copy()
                                                for transformer in transformers:
                                                        if(transformer.find('None')>-1):
                                                                continue
                                                        try:
                                                                n = self.L2C_class[L2C_class_map[transformer]](dataset=dataset_fs_norm, strategy=transformer,
                                                                exclude='SalePrice',
                                                                
                                                                verbose=self.verbose).transform()
                                                        except:
                                                                n = self.L2C_class[L2C_class_map[transformer]](dataset=dataset_fs_norm, strategy=transformer,
                                                                        exclude='SalePrice',
                                                                        file_name='house_discovered',
                                                                        verbose=self.verbose).transform()
                                                        
                                                outc = None
                                                try:
                                                        dataset_fs_norm['test'] = dataset_fs_norm['test'].drop('New_ID', axis=1)
                                                except:
                                                        pass
                                                if dataset_fs_norm['test'].shape[0]!=0:
                                                        outc = self.L2C_class[18](dataset=dataset_fs_norm,strategy='MARS',target='SalePrice',verbose=self.verbose).transform()['quality_metric']
                                                else:
                                                        continue
                                                f = [outc]
                                                # param_lst.append(mv_param + norm_param + od_param + f)
                                                
                                                conc_list = pd.concat([dataset_fs_norm['test'], dataset_fs_norm['target']], axis=1)
                                                profile_gen,key_profile = p.populate_profiles(conc_list,0.2)
                                                # if len(key_profile)>15:
                                                #        import pdb;pdb.set_trace()
                                                
                                                module_0 = [self.class_map[norm_fs]]
                                                module_1 = [self.class_map[feature_selection]]
                                                module_2 = [self.class_map[transformers[0]]]
                                                module_3 = [self.class_map[transformers[1]]]
                                                module_4 = [self.class_map[transformers[2]]]
                                                
                                                
                                                
                                                param_lst.append(module_0+module_1 + module_2 + module_3  +module_4+f)
                                                
                                                print(str(module_1 + module_2 + module_3  +f))

                        

                        # param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        param_column = ["module_0","module_1","module_2","module_3","module_4"]
                        
                        param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ["fairness"])
                               

                        param_lst_df.to_csv(file_name, index=False)


        def write_quartiles(self,csv_writer, algorithm, metric, quartiles):
                csv_writer.writerow([f_goal, algorithm, f"{metric} q1", round(quartiles[0], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q2", round(quartiles[1], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q3", round(quartiles[2], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q4", round(quartiles[3], 5)])
p = base()





filename_test = '/Users/apple/Documents/code/OptimizingPipeline/historical_data/l2c_historical_data_test_profile_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
filename_train = '/Users/apple/Documents/code/OptimizingPipeline/historical_data/l2c_historical_data_train_profile_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
# filename_train = 'historical_data/historical_data_train_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
p.create_historic_data(filename_train)

p.create_historic_data_test(filename_test)

f_goals = []
if(dataset =='adult'):
#        f_goals = [0.025,0.03,0.04,0.05,0.06,0.07,0.08,0.09,.16]
       f_goals = [0.045, 0.055, 0.07, 0.14]
#        f_goals = [0.055]
elif(dataset=='hmda'):
        # f_goals  = [.135,0.14,0.15,0.16,0.17]
       f_goals  = [0.17,0.18,0.19]
elif(dataset=='housing'):
        # f_goals = [25000,26000,27000,28000,29000,30000,31000,32000,33000,34000,35000]
       f_goals = [162,165,172,175]
else:
        print('Please profile goals ')

# #Read from historical data gererated on training data 
historical_data = pd.read_csv(filename_test)
p.historical_data_pd = historical_data;
# #Convert to list of list containing all the combination of transformers in a fixed pipeline
p.historical_data = historical_data.values.tolist();


f = sys.stdout
metric_path  = 'metric/metric_profile_l2c'+algo_type+"_"+modelType+'_'+metric_type+'_'+dataset+'.csv'
f = open(metric_path, 'w')

gg  = historical_data.values.tolist()
csv_writer = csv.writer(f)

for f_goal in f_goals:
        # logging.info(f'Fairness goal {f_goal}')
        rank_idistr = []
        rank_fdistr = []
        gs_idistr =   []
        gs_fdistr =   []
        profile_itr = {}
        profile_itr['profile_outlier']= 0
        profile_itr['diff_sensitive_attr'] = 0
        profile_itr['ratio_sensitive_attr'] = 0
        # profile_itr['cov'] = 0
        print(f_goal)
        # random.shuffle(gg)
        failures = 0
        # gg = [[1.0, 2.0, 1.0, 0.1765017004225869]]
        for seed_ in gg:
                seen = set()
                p.grid_search(f_goal, 1,seen)
                # print(p.gs_idistr)
                # print(p.gs_idistr)
                gs_idistr.append(p.gs_idistr[0])
                gs_fdistr.append(p.gs_fdistr[0])
 #               if(algo_type=='projection'):
                p.optimize(seed_, f_goal)

                # if(p.gs_idistr[0]>5):
                #        logging.error(f'bad seed {seed_}')
                if(p.rank_iter > 4):
                #        print(p.rank_iter)
                       logging.error(f'bad seed {seed_}')
                if p.rank_iter != -1:
                        rank_idistr.append(p.rank_iter)
                        rank_fdistr.append(p.rank_f)
                        profile_itr['profile_outlier'] += p.profile_dist['profile_outlier']
                        profile_itr['diff_sensitive_attr'] += p.profile_dist['diff_sensitive_attr'] 
                        profile_itr['ratio_sensitive_attr'] += p.profile_dist['ratio_sensitive_attr'] 
                        # profile_itr['cov'] += p.profile_dist['cov'] 
                else:
                        failures += 1
        # print('rank appended',rank_idistr)
        print(p.fail)
        print(p.pass_)
        print(p.fail_grid_search)
        csv_writer.writerow([f" failed {p.fail}" , f"passed{p.pass_}",f"Grid search fall back {p.fail_grid_search}"])
        p.pass_   = 0 
        p.fail  = 0 
        p.fail_grid_search = 0
        # import pdb;pdb.set_trace()
        rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75,100], interpolation='midpoint')
        rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75,100], interpolation='midpoint')
        g_iquartiles = np.percentile(gs_idistr, [25, 50, 75,100], interpolation='midpoint')
        g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75,100], interpolation='midpoint')

        # profile_itr_1 = np.percentile(profile_itr['profile_outlier'], [25, 50, 75,100], interpolation='midpoint')
        # profile_itr_2 = np.percentile(profile_itr['diff_sensitive_attr'], [25, 50, 75,100], interpolation='midpoint')
        # profile_itr_3 = np.percentile(profile_itr['ratio_sensitive_attr'], [25, 50, 75,100], interpolation='midpoint')


        print("Fairness goal stats: " + str(f_goal))



# Write header
        # csv_writer.writerow(["Fairness Goal", "Grid search", "Iteration", "Value"])

        # Write data for ranking algorithm
        p.write_quartiles(csv_writer, "ranking", "iterations", rank_iquartiles)
        p.write_quartiles(csv_writer, "ranking", "Fairness", rank_fquartiles)
        csv_writer.writerow([])

        # Write data for grid search algorithm
        p.write_quartiles(csv_writer, "grid search", "iterations", g_iquartiles)
        p.write_quartiles(csv_writer, "grid search", "Fairness", g_fquartiles)
        csv_writer.writerow([])

        # csv_writer.writerow(["Fairness Goal", "Profile param ", "Iteration", "Value"])
        # p.write_quartiles(csv_writer, "profile outlier", "Itration", profile_itr_1)
        # csv_writer.writerow([])
        # p.write_quartiles(csv_writer, "diff sensitive attr", "Itration", profile_itr_2)
        # csv_writer.writerow([])
        # p.write_quartiles(csv_writer, "ratio sensitive attr", "Itration", profile_itr_3)
        


        # csv_writer.writerow([])
        #import pdb;pdb.set_trace()
f.close()



