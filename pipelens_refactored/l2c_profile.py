import itertools

from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
from modules.normalization.normalizer import DataNormalizer
from modules.metric.metric import metric
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from regression import Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import time
import csv
import os
import pandas as pd
import numpy as np
from itertools import product
import random
from itertools import cycle
import sys
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

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

logging.basicConfig(filename='logs/profile_'+algo_type+"_"+dataset+'_'+modelType+'_'+'metric_type'+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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
        tuple = ('corr_' + name,  'ot_' + name)
        profile[tuple]= [column,round(corr*scaling_factor,5),round(outlier*scaling_factor,5)]
        i+=1
    dd = []
    keys = []
    for val in profile:
        # import pdb;pdb.set_trace()
        dd.append(profile[val][1])
        dd.append(profile[val][2])
        keys.append(val[0])
        keys.append(val[1])
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
    logging.warn(train.isnull().sum())
    logging.warn(test.isnull().sum())
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
    for column in missing_colmn_categorical:
        most_frequent = train[column].mode()[0]
        train[column].fillna(most_frequent, inplace=True)
    for column in missing_colmn_numerical:
        median_value = train[column].median()
        train[column].fillna(median_value, inplace=True)
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
    
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        le = LabelEncoder()
        X_train[column] = label_encoder.fit_transform(y_train[column]) 
        X_test[column] = label_encoder.fit_transform(y_test[column]) 
    y_train = X_train['SalePrice']
    y_test = X_test['SalePrice']

    X_train = X_train.drop('SalePrice', axis=1)
    X_test = X_test.drop('SalePrice', axis=1)
    selector = SelectKBest(score_func=mutual_info_regression, k=15)
    selector.fit(X_train, y_train)


    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices]
    feature_scores = selector.scores_
    selected_features_scores = pd.DataFrame({'Feature': X_train.columns, 'Score': feature_scores})
    selected_features_scores.sort_values(by='Score', ascending=False, inplace=True)
    print(selected_features_scores)
    selected_features_set = set(selected_features)
    # Filter the DataFrame to include only the scores of the selected features
    selected_features_scores_filtered = selected_features_scores[selected_features_scores['Feature'].isin(selected_features_set)]
    print(selected_features_scores_filtered)

    label_encoder = LabelEncoder()
    label_encoders = {}
    X_train = X_train[selected_features]
    X_test = X_test [selected_features]
#     import pdb;pdb.set_trace()
    
    
    
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        X_train[column] = label_encoder.fit_transform(X_train[column]) 
        X_test[column] = label_encoder.fit_transform(X_test[column]) 
        label_encoders[column] = le
    X_test_concat = pd.concat([X_test,y_test],axis=1)
    X_test_concat.to_csv('Learn2Clean/datasets/house/concatenated_data_test.csv', index=False)
permutations = list(itertools.product(mv_strategy, od_strategy))
permutations += list(itertools.product(od_strategy,mv_strategy))

for val in permutations:
       print(val)
if(metric_type=='sp' or metric_type=='accuracy_score' or metric_type=='mae'  or metric_type=='rmse'):
        if modelType == 'lr':
                model = LogisticRegression(random_state=0).fit(X_train, y_train)
        elif modelType == 'nb':
                model = GaussianNB().fit(X_train, y_train)
        elif modelType == 'rf':
                model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        elif modelType == 'reg':
                model = Regression().generate_regression(X_train, y_train)
        print("Training accuracy : " + str(round(model.score(X_train, y_train), 4)))
        print("Test accuracy : " + str(round(model.score(X_test, y_test), 4)))

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # p_train = model.predict_proba(X_train)
        # p_test = model.predict_proba(X_test)
        if metric_type=='mae' or metric_type=='rmse':
                outc_train_mae = mean_absolute_error( y_train,y_pred_train)
                outc_test_mae = mean_absolute_error(y_test, y_pred_test)
                print(f'MAE training {outc_train_mae} ,test {outc_test_mae}')
                outc_train_mse = np.sqrt(root_mean_squared_error(y_train, y_pred_train)) 
                outc_test_mse = np.sqrt(root_mean_squared_error(y_test,y_pred_test)) 
                print(f'MSE training {outc_train_mse} ,test {outc_test_mse}')
        print('Done')

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
                self.normalizer_strategies = ['none', 'ss', 'rs', 'ma', 'mm']
                self.no_name_mapping = {'none': 'norm_none', 'ss': 'norm_ss', 'rs': 'norm_rs', 'ma': 'norm_ma', 'mm': 'norm_mm'}

                #normalization 
                #

                self.ranges['module_1'] = [0,1,2,3]
                self.ranges['module_2'] = [0, 1, 2, 3, 4,5,6,7]
                self.ranges['module_3'] = [0, 1, 2, 3, 4,5,6,7]

                self.base_strategies  =['module_1', 'module_2', 'module_3']
                



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






        def optimize(self, init_params, f_goal):
                logging.info(f'2step Starting Grasp search for fairness score:{f_goal}')
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
                logging.info(f'initial parameter {init_params}')
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

                self.ranges['missing_value'] = list(np.unique(self.historical_data_pd['module_1']))
                self.ranges['normalization'] = list(np.unique(self.historical_data_pd['module_2']))
                self.ranges['outlier'] = list(np.unique(self.historical_data_pd['module_3']))
                opt_f  = self.f_score_look_up2(self.historical_data_pd,init_params)

                # print(f'Fairness score from seed value : {opt_f}')
                #======Code end ============

                seen = set()
                
                seen.add(tuple(cur_params_opt.items()))
                
                                
                iter_size  = 0
                if(opt_f<f_goal):
                        self.rank_iter = 1  #Iteration count by ranking algorithm
                        self.rank_f = opt_f
                        logging.info("Target achieved")
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
                                                logging.info(f'Strategy selected  : {cur_strategy}')    
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
                                                
                                                logging.info(f'order of parameter  value: {current_paramter_value}')                
                                                logging.warn(f'current Iteration before parameter selection {self.rank_iter}')
                                                cur_params = cur_params_opt.copy()
                                                cur_params[cur_strategy] = current_paramter_value
                                                if(cur_strategy=='module_3' and cur_params[cur_strategy]==cur_params['module_2']):
                                                       cur_params['module_2'] = 5
                                                if(cur_strategy=='module_2' and cur_params[cur_strategy]==cur_params['module_3']):
                                                       cur_params['module_3'] = 5

                                                
                                                logging.warn(f'Current iteration after parameter selection {self.rank_iter}')
                                                #profile_opt = self.get_profile(self.historical_data_pd,list(cur_params.values()),profile_index)
                                                
                                                #profile_cur = self.get_profile(self.historical_data_pd,list(cur_params.values()),profile_index)
                                                logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                                                if(tuple(cur_params.items())) in seen:
                                                        continue
                                                seen.add(tuple(cur_params.items()))
                                                
                                                # logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                                                cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))

                                                # self.profile_dist[param_name] +=1
                                                
                                                #if(profile_cur<profile_opt):
                                                self.rank_iter += 1

                                                logging.info(f'updated fairness after parameter selection {cur_f}')
                                                if metric_type=='sp' or metric_type=='mae' or metric_type=='rmse' or metric_type=='accuracy_score':                                                                           
                                                        if cur_f <= f_goal:
                                                                self.rank_f = cur_f
                                                                self.pass_ = self.pass_  + 1
                                                                logging.error("Target achieved")
                                                                return cur_params_opt.values() # early exit when f_goal obtained
                                                        elif cur_f < opt_f:
                                                                opt_f = cur_f
                                                                cur_params_opt = cur_params.copy()
                                                                
                                                                   
                                else :
                                        if(iter_size==1 and self.rank_iter==13):
                                               print(seen)
                                               print(self.rank_iter)
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
                                                +param[2]*self.param_coeff[profile][2]] = param

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

                                        logging.info(f'updated fairness after parameter selection {cur_f}')
                                                
                                        if cur_f <= f_goal:
                                                self.rank_f = cur_f
                                                self.pass_ = self.pass_  + 1
                                                logging.error("Target achieved")
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
                column_names =['module_1', 'module_2', 'module_3', 'fairness']
                # return elem[-1]
                # import pdb;pdb.set_trace()
                try:
                        return profiles_df.loc[(profiles_df[column_names[0]] == elem[0]) & (profiles_df[column_names[1]] == elem[1] ) 
                                               & (profiles_df[column_names[2]] == elem[2]) 
                                             
                                               
                                               ].iloc[0]['fairness']
                except Exception as e :
                        print(e)
                        import pdb;pdb.set_trace()


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


        def create_historic_data(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
                key_profile = []
                p = Profile()
                
                # if True:
                if not(os.path.exists(file_name)):
                        idx_train = np.arange(0, len(X_train), 1)
                        mv_train = pd.DataFrame(idx_train).sample(frac=tau_train, replace=False, random_state=1).index
                        if (dataset == 'hmda'):
                                X_train['lien_status'][mv_train] = np.NaN
                        elif (dataset == 'adult'):
                                # 
                                X_train['Martial_Status'][mv_train] = np.NaN
                        elif(dataset=='housing'):
                                X_train['OverallQual'][mv_train] = np.NaN

                        # idx_test = np.arange(0, len(X_test), 1)
                        # mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
                        # X_test['lien_status'][mv_test] = np.NaN
                        # # X_test['income_brackets'][mv_test] = np.NaN

                        params_metrics = []
                        print("Running pipeline combinations ...")
                        
                        
                        
                        mv_params = len(mv_strategy)
                        od_params = len(od_strategy)
                        norm_params = len(norm_strategy)
                        if 'knn' in mv_strategy:
                                mv_params += len(knn_k_lst) - 1
                        if 'lof' in od_strategy:
                                od_params += len(lof_k_lst) - 1

                        mv_param = ''
                        norm_param = ''
                        od_param = ''

                        param_lst = []
                        sens_attr_name = ''
                        target_variable_name = ''
                        if(metric_type=='sp' or metric_type=='accuracy_score'):
                                priv_idx_train, unpriv_idx_train, sensitive_attr_train = self.getIdxSensitive(X_train, dataset)
                                sens_attr_name = sensitive_attr_train.name
                                target_variable_name = y_train.name
                                train_eqOdds = self.computeStatisticalParity(y_pred_train[priv_idx_train], y_pred_train[unpriv_idx_train])
                                print("Training fairness : " + str(round(train_eqOdds, 4)))

                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)
                                test_eqOdds = self.computeStatisticalParity(y_pred_test[priv_idx_test], y_pred_test[unpriv_idx_test])
                                print("Test fairness : " + str(round(test_eqOdds, 4)))
                        permutations = list(itertools.product(mv_strategy, od_strategy))
                        permutations += list(itertools.product(od_strategy,mv_strategy))
                        permutations+=list(itertools.product(od_strategy, repeat=2))

                        count = 0
                        for norm in norm_strategy:
                                X_train_transform_norm = X_train.copy()
                                y_train_transform_norm = y_train.copy()
                                mv_param_index = -1;
                                od_param_index = -1;
                                if norm == 'none':
                                        X_train_transform_norm = X_train_transform_norm.copy()
                                        norm_param = [1]
                                elif norm == 'ss':
                                        X_train_transform_norm = StandardScaler().fit(X_train_transform_norm).transform(X_train_transform_norm) 
                                elif norm  == 'log10':
                                        X_train_transform_norm = self.Log10_normalization(X_train_transform_norm) 
                                        norm_param = [3]
                                elif norm == 'ma':
                                        X_train_transform_norm = self.MA_normalization(X_train_transform_norm) 
                                elif norm == 'mm':
                                        X_train_transform_norm = MinMaxScaler().fit(X_train_transform_norm).transform(X_train_transform_norm)
                                
                                for tranformer_comb in  permutations:   
                                        # tranformer_comb = ('knn','lof')
                                        X_train_transform = X_train_transform_norm.copy()
                                        y_train_transform = y_train_transform_norm.copy()
                                        if isinstance(X_train_transform_norm, np.ndarray):
                                                X_train_transform = pd.DataFrame(data=X_train_transform_norm,columns=X_train.columns)
                                                
                                        for tranformer in tranformer_comb:
                                                mv = False
                                                if isinstance(X_train_transform, np.ndarray):
                                                        X_train_transform = pd.DataFrame(data=X_train_transform,columns=X_train.columns)
                                                
                                                if tranformer == 'drop':
                                                        mv = True
                                                        mv_idx = []
                                                        if dataset == 'hmda':
                                                                mv_idx = X_train_transform[X_train_transform['lien_status'].isna()].index.tolist()
                                                        elif dataset == 'adult':
                                                                mv_idx = X_train_transform[X_train_transform['Martial_Status'].isna()].index.tolist()
                                                        elif dataset == 'housing':
                                                                try:
                                                                        mv_idx = X_train_transform[X_train_transform['OverallQual'].isna()].index.tolist()
                                                                except:
                                                                        import pdb;pdb.set_trace()
                                                        
                                                        
                                                        X_train_transform = X_train_transform.drop(mv_idx)
                                                        X_train_transform.reset_index(drop=True, inplace=True)
                                                        y_train_transform = y_train_transform.copy()
                                                        for idx in sorted(mv_idx, reverse=True):
                                                                del y_train_transform[idx]
                                                        y_train_transform.reset_index(drop=True, inplace=True)
                                                elif tranformer in ['mean', 'median', 'most_frequent']:
                                                        mv = True
                                                        X_train_transform = SimpleImputer(missing_values=np.nan, strategy=tranformer).fit(X_train_transform).transform(X_train_transform)
                                                        y_train_transform = y_train_transform.copy()
                                                elif tranformer == 'knn':
                                                        mv = True
                                                        k = 4 # start accessing number of neighbors in knn
                                                        X_train_transform = KNNImputer(n_neighbors=k).fit_transform(X_train_transform)

                                                if isinstance(X_train_transform, np.ndarray):
                                                        X_train_transform = pd.DataFrame(data=X_train_transform,columns=X_train.columns)
                                
                                                # out_before_norm_strat = p.get_fraction_of_outlier(X_train_transform)

                                                if isinstance(X_train_transform, np.ndarray):
                                                        X_train_transform = pd.DataFrame(data=X_train_transform,columns=X_train.columns)
                                                
                                                od = False
                                                outlier_x_pred = None
                                                print(len(X_train_transform['OverallQual'].isna()))
                                                if tranformer in od_strategy and len(X_train_transform['OverallQual'].isna())>0:
                                                        
                                                        mv_idx = []
                                                        if dataset == 'hmda':
                                                                mv_idx = X_train_transform[X_train_transform['lien_status'].isna()].index.tolist()
                                                        elif dataset == 'adult':
                                                                mv_idx = X_train_transform[X_train_transform['Martial_Status'].isna()].index.tolist()
                                                        elif dataset == 'housing':
                                                                mv_idx = X_train_transform[X_train_transform['OverallQual'].isna()].index.tolist()
                                                                
                                                        # import pdb;pdb.set_trace()
                                                        
                                                        
                                                        X_train_transform = X_train_transform.drop(mv_idx)
                                                        X_train_transform.reset_index(drop=True, inplace=True)
                                                        y_train_transform = y_train_transform.copy()
                                                        for idx in sorted(mv_idx, reverse=True):
                                                                del y_train_transform[idx]
                                                        y_train_transform.reset_index(drop=True, inplace=True)
                                                # import pdb;pdb.set_trace()
                                                if tranformer == 'none':
                                                        outlier_x_pred = np.ones(len(X_train_transform))
                                                        od = True
                                                if tranformer == 'if':
                                                        
                                                        outlier_x_pred = IsolationForest(contamination=contamination_train,random_state=0).fit_predict(X_train_transform)
                                                        od = True
                                                if tranformer=='lof':
                                                        k =4 # start accessing number of neighbors in lof
                                                        outlier_x_pred = LocalOutlierFactor(n_neighbors=k, contamination=contamination_train_lof).fit_predict(X_train_transform)
                                                        od = True
                                                if od:
                                                        mask = outlier_x_pred != -1

                                                        # fraction_out = round((1 - sum(mask)/len(outlier_x_pred) * 100, 4))
                                                        priv_idx_train = []
                                                        unpriv_idx_train = []

                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_x_pred)): # at least one outlier
                                                                X_train_transform, y_train_transform = X_train_transform[mask], y_train_transform[mask]
                                                                y_train_transform.reset_index(drop=True, inplace=True)
                                                                X_train_transform.reset_index(drop=True, inplace=True)
                                                if mv:
                                                       mv_param_index  = base_strategy.index(tranformer)
                                                       
                                                if od:
                                                      od_param_index  = base_strategy.index(tranformer) 
                                                if isinstance(X_train_transform, np.ndarray):
                                                        X_train_transform = pd.DataFrame(data=X_train_transform,columns=X_train.columns)
                                        count+=1       
                                  
                                        updated_model = None
                                        
                                        if modelType == 'lr':
                                                updated_model = LogisticRegression(random_state=0).fit(X_train_transform, y_train_transform)
                                        elif modelType == 'nb':
                                                updated_model = GaussianNB().fit(X_train_transform, y_train_transform)
                                        elif modelType == 'rf':
                                                updated_model = RandomForestClassifier(random_state=0).fit(X_train_transform, y_train_transform)
                                        elif modelType=='reg':
                                                updated_model = Regression().generate_regression(X_train_transform,y_train_transform)
                                        
                                        y_pred = updated_model.predict(X_train_transform)
                                        
                                        sens_data =[None,None,None]
                                        
                                        outc = None
                                        if metric_type== 'sp':
                                                outc = self.computeStatisticalParity(y_pred[priv_idx_train],y_pred[unpriv_idx_train])
                                        elif metric_type=='f-1':
                                                outc = f1_score(y_train_transform,y_pred)
                                        elif metric_type=='mae':
                                                outc = mean_absolute_error(y_train_transform, y_pred)
                                        elif metric_type=='rmse':
                                                outc = np.sqrt(root_mean_squared_error(y_train_transform, y_pred)) 
                                        elif metric_type=='accuracy_score':
                                                outc = 1-accuracy_score(y_train_transform, y_pred)
                                                

                                                
                                        f = [outc]
                                        # param_lst.append(mv_param + norm_param + od_param + f)
                                        
                                        conc_list = pd.concat([X_train_transform, y_train_transform], axis=1)
                                        profile_gen,key_profile = p.populate_profiles(conc_list,0.2)

                                        module_2 = [base_strategy.index(tranformer_comb[0])]
                                        module_1 = [norm_strategy.index(norm)]
                                        module_3 = [base_strategy.index(tranformer_comb[1])]
                                        profile_median = [y_train_transform.median()]
                                        param_lst.append(module_1 + module_2 + module_3  +profile_median+ profile_gen+f)
                                        # if eqOdds_train == 0.0:
                                        
                                        print(str(module_1 + module_2 + module_3  +f))
                                        # print(len(unpriv_idx_train))
                                        # import pdb;pdb.set_trace()
                        

                        # param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        param_column = ["module_1","module_2","module_3"]
                        
                        if(metric_type=='sp'):
                                param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['out_before_out_strat','out_before_norm_strat',"diff_sensitive_attr","ratio_sensitive_attr","cov","class_imbalance_ratio"]+key_profile+["fairness"])
                        else:
                               param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['profile_median']+key_profile+["fairness"])
                               

                        param_lst_df.to_csv(file_name, index=False)
                        # import pdb;pdb.set_trace()
                else :
                        param_lst_df = pd.read_csv(file_name)
                #//ocuupation,education,maritial,education,sex
                self.param_columns = ["module_1","module_2","module_3"]
                if(dataset=='hmda'):
                        key_profile = ['corr_race', 'corr_gender',  'corr_loan_type',  'corr_applicant_age',  'corr_lien_status',  'corr_LV', 'corr_DI', 'corr_income_brackets']
                elif dataset=='housing':
                       key_profile = ['profile_median','corr_MSSubClass','corr_Neighborhood','corr_OverallQual','corr_YearBuilt','corr_ExterQual','corr_BsmtQual','corr_TotalBsmtSF','corr_1stFlrSF','corr_2ndFlrSF','corr_GrLivArea','corr_KitchenQual','corr_GarageYrBlt','corr_GarageCars','corr_GarageArea']
                       #key_profile = ['corr_MSSubClass', 'corr_Neighborhood',  'corr_OverallQual', 'corr_YearBuilt', 'corr_ExterQual', 'corr_BsmtQual',  'corr_TotalBsmtSF',  'corr_1stFlrSF', 'corr_GrLivArea',  'corr_FullBath','corr_KitchenQual',  'corr_GarageYrBlt','corr_GarageCars', 'corr_GarageArea' ]
                elif dataset=='adult':
                #        key_profile = ["diff_sensitive_attr","ratio_sensitive_attr"]
                       key_profile = ["diff_sensitive_attr","ratio_sensitive_attr","class_imbalance_ratio","corr_Age","corr_Workclass","corr_Education","corr_Martial_Status","corr_Occupation","corr_Relationship","corr_Race","corr_Sex","corr_Hours_per_week"]
                self.profiles = key_profile#+[ 'out_before_out_strat','out_before_norm_strat']
                
                #rank profile first
                if(dataset=='housing'):
                        t = StandardScaler().fit(param_lst_df).transform(param_lst_df)
                        param_lst_df = pd.DataFrame(data=t,columns=param_lst_df.columns)
                
                        y = param_lst_df['fairness']
                        X = param_lst_df.copy()[self.profiles]
                else :
                       y = param_lst_df['fairness']
                       X = param_lst_df.copy()[self.profiles]
                #        X = StandardScaler().fit(X).transform(X)


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
                param_lst = []
                if not(os.path.exists(file_name)):
                        idx_test = np.arange(0, len(X_test), 1)
                        mv_test = pd.DataFrame(idx_test).sample(frac=tau_test, replace=False, random_state=1).index
                        if (dataset == 'hmda'):
                                X_test['lien_status'][mv_test] = np.NaN
                        elif (dataset == 'adult'):
                                # 
                                X_test['Martial_Status'][mv_test] = np.NaN
                        elif(dataset=='housing'):
                                X_test['OverallQual'][mv_test] = np.NaN



                        mv_param = ''
                        norm_param = ''
                        od_param = ''
                        permutations = list(itertools.product(mv_strategy, od_strategy))
                        permutations += list(itertools.product(od_strategy,mv_strategy))
                        permutations += list(itertools.product(od_strategy, repeat=2))

                        count = 0
                        for norm in norm_strategy:
                                X_test_transform_norm = X_test.copy()
                                y_test_transform_norm = y_test.copy()
                                mv_param_index = -1;
                                od_param_index = -1;
                                if norm == 'none':
                                        X_test_transform_norm = X_test_transform_norm.copy()
                                        norm_param = [1]
                                elif norm == 'ss':
                                        X_test_transform_norm = StandardScaler().fit(X_test_transform_norm).transform(X_test_transform_norm) 
                                elif norm  == 'log10':
                                        X_test_transform_norm = self.Log10_normalization(X_test_transform_norm) 
                                        norm_param = [3]
                                elif norm == 'ma':
                                        X_test_transform_norm = self.MA_normalization(X_test_transform_norm) 
                                elif norm == 'mm':
                                        X_test_transform_norm = MinMaxScaler().fit(X_test_transform_norm).transform(X_test_transform_norm)
                                
                                for tranformer_comb in  permutations:   
                                        # tranformer_comb = ('knn','lof')
                                        X_test_transform = X_test_transform_norm.copy()
                                        y_test_transform = y_test_transform_norm.copy()
                                        if isinstance(X_test_transform_norm, np.ndarray):
                                                X_test_transform = pd.DataFrame(data=X_test_transform_norm,columns=X_test.columns)
                                                
                                        for tranformer in tranformer_comb:
                                                mv = False
                                                if isinstance(X_test_transform, np.ndarray):
                                                        X_test_transform = pd.DataFrame(data=X_test_transform,columns=X_test.columns)
                                                
                                                if tranformer == 'drop':
                                                        mv = True
                                                        mv_idx = []
                                                        if dataset == 'hmda':
                                                                mv_idx = X_test_transform[X_test_transform['lien_status'].isna()].index.tolist()
                                                        elif dataset == 'adult':
                                                                mv_idx = X_test_transform[X_test_transform['Martial_Status'].isna()].index.tolist()
                                                        elif dataset == 'housing':
                                                                try:
                                                                        mv_idx = X_test_transform[X_test_transform['OverallQual'].isna()].index.tolist()
                                                                except:
                                                                        import pdb;pdb.set_trace()
                                                        
                                                        
                                                        X_test_transform = X_test_transform.drop(mv_idx)
                                                        X_test_transform.reset_index(drop=True, inplace=True)
                                                        y_test_transform = y_test_transform.copy()
                                                        for idx in sorted(mv_idx, reverse=True):
                                                                del y_test_transform[idx]
                                                        y_test_transform.reset_index(drop=True, inplace=True)
                                                elif tranformer in ['mean', 'median', 'most_frequent']:
                                                        mv = True
                                                        X_test_transform = SimpleImputer(missing_values=np.nan, strategy=tranformer).fit(X_test_transform).transform(X_test_transform)
                                                        y_test_transform = y_test_transform.copy()
                                                elif tranformer == 'knn':
                                                        mv = True
                                                        k = 4 # start accessing number of neighbors in knn
                                                        X_test_transform = KNNImputer(n_neighbors=k).fit_transform(X_test_transform)

                                                if isinstance(X_test_transform, np.ndarray):
                                                        X_test_transform = pd.DataFrame(data=X_test_transform,columns=X_test.columns)
                                
                                                # out_before_norm_strat = p.get_fraction_of_outlier(X_test_transform)

                                                if isinstance(X_test_transform, np.ndarray):
                                                        X_test_transform = pd.DataFrame(data=X_test_transform,columns=X_test.columns)
                                                
                                                od = False
                                                outlier_x_pred = None
                                                print(len(X_test_transform['OverallQual'].isna()))
                                                if tranformer in od_strategy and len(X_test_transform['OverallQual'].isna())>0:
                                                        
                                                        mv_idx = []
                                                        if dataset == 'hmda':
                                                                mv_idx = X_test_transform[X_test_transform['lien_status'].isna()].index.tolist()
                                                        elif dataset == 'adult':
                                                                mv_idx = X_test_transform[X_test_transform['Martial_Status'].isna()].index.tolist()
                                                        elif dataset == 'housing':
                                                                mv_idx = X_test_transform[X_test_transform['OverallQual'].isna()].index.tolist()
                                                                
                                                        # import pdb;pdb.set_trace()
                                                        
                                                        
                                                        X_test_transform = X_test_transform.drop(mv_idx)
                                                        X_test_transform.reset_index(drop=True, inplace=True)
                                                        y_test_transform = y_test_transform.copy()
                                                        for idx in sorted(mv_idx, reverse=True):
                                                                del y_test_transform[idx]
                                                        y_test_transform.reset_index(drop=True, inplace=True)
                                                # import pdb;pdb.set_trace()
                                                if tranformer == 'none':
                                                        outlier_x_pred = np.ones(len(X_test_transform))
                                                        od = True
                                                if tranformer == 'if':
                                                        
                                                        outlier_x_pred = IsolationForest(contamination=contamination_test,random_state=0).fit_predict(X_test_transform)
                                                        od = True
                                                if tranformer=='lof':
                                                        k =4 # start accessing number of neighbors in lof
                                                        outlier_x_pred = LocalOutlierFactor(n_neighbors=k, contamination=contamination_test_lof).fit_predict(X_test_transform)
                                                        od = True
                                                if od:
                                                        mask = outlier_x_pred != -1

                                                        # fraction_out = round((1 - sum(mask)/len(outlier_x_pred) * 100, 4))
                                                        priv_idx_test = []
                                                        unpriv_idx_test = []

                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_x_pred)): # at least one outlier
                                                                X_test_transform, y_test_transform = X_test_transform[mask], y_test_transform[mask]
                                                                y_test_transform.reset_index(drop=True, inplace=True)
                                                                X_test_transform.reset_index(drop=True, inplace=True)
                                                if mv:
                                                       mv_param_index  = base_strategy.index(tranformer)
                                                       
                                                if od:
                                                      od_param_index  = base_strategy.index(tranformer) 
                                                if isinstance(X_test_transform, np.ndarray):
                                                        X_test_transform = pd.DataFrame(data=X_test_transform,columns=X_test.columns)
                                        count+=1       
                                  
                                        updated_model = None
                                        
                                        if modelType == 'lr':
                                                updated_model = LogisticRegression(random_state=0).fit(X_test_transform, y_test_transform)
                                        elif modelType == 'nb':
                                                updated_model = GaussianNB().fit(X_test_transform, y_test_transform)
                                        elif modelType == 'rf':
                                                updated_model = RandomForestClassifier(random_state=0).fit(X_test_transform, y_test_transform)
                                        elif modelType=='reg':
                                                updated_model = Regression().generate_regression(X_test_transform,y_test_transform)
                                        
                                        y_pred = updated_model.predict(X_test_transform)
                                        
                                        sens_data =[None,None,None]
                                        
                                        outc = None
                                        if metric_type== 'sp':
                                                outc = self.computeStatisticalParity(y_pred[priv_idx_test],y_pred[unpriv_idx_test])
                                        elif metric_type=='f-1':
                                                outc = f1_score(y_test_transform,y_pred)
                                        elif metric_type=='mae':
                                                outc = mean_absolute_error(y_test_transform, y_pred)
                                        elif metric_type=='rmse':
                                                outc = np.sqrt(root_mean_squared_error(y_test_transform, y_pred)) 
                                        elif metric_type=='accuracy_score':
                                                outc = 1-accuracy_score(y_test_transform, y_pred)
                                                

                                                
                                        f = [outc]
                                        # param_lst.append(mv_param + norm_param + od_param + f)
                                        
                                        conc_list = pd.concat([X_test_transform, y_test_transform], axis=1)
                                        # profile_gen,key_profile = p.populate_profiles(conc_list,0.2)

                                        module_2 = [base_strategy.index(tranformer_comb[0])]
                                        module_1 = [norm_strategy.index(norm)]
                                        module_3 = [base_strategy.index(tranformer_comb[1])]
                                        param_lst.append(module_1 + module_2 + module_3  +f)
                                        # if eqOdds_test == 0.0:
                                        
                                        print(str(module_1 + module_2 + module_3  +f))
                                        # print(len(unpriv_idx_test))
                                        # import pdb;pdb.set_trace()
                        

                        # param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        param_column = ["module_1","module_2","module_3"]

                        param_lst_df = pd.DataFrame(param_lst, columns= param_column   +["fairness"])
                               

                        param_lst_df.to_csv(file_name, index=False)


        def write_quartiles(self,csv_writer, algorithm, metric, quartiles):
                csv_writer.writerow([f_goal, algorithm, f"{metric} q1", round(quartiles[0], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q2", round(quartiles[1], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q3", round(quartiles[2], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q4", round(quartiles[3], 5)])
p = base()




p = base()



filename_test = 'historical_data/historical_data_test_profile_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
filename_train = 'historical_data/historical_data_train_profile_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
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
       f_goals = [160,165,172,175]
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
        logging.info(f'Fairness goal {f_goal}')
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



