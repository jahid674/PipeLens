import itertools
# from modules.outlier_detection.outlier_detector import OutlierDetector
# from modules.missing_value.imputer import DataImputer
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
import operator
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
# from modules.outlier_detection.outlier_detector import OutlierDetector
# from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
from modules.normalization.normalizer import DataNormalizer
from sklearn.naive_bayes import GaussianNB
from regression import Regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np

tau_train = 0.1 # fraction of missing values
tau_test = 0.1
contamination_train = 0.2
contamination_test = 0.2
contamination_train_lof = 'auto'
contamination_test_lof = 'auto'

knn_k_lst = [1, 5, 10, 20, 30]
lof_k_lst = [1, 5, 10, 20, 30]
len_knn = len(knn_k_lst)
len_lof = len(lof_k_lst)
norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm'] # standard scaler, robust scaler, max absolute scaler, minmax scaler
mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
od_strategy = ['none', 'if', 'lof'] # local outlier factor, isolation forest
dataset = 'adult'
modelType = 'lr' # rf, 'lr' # 'nb' Logistic Regression or Gaussian Naive Bayes,reg - regression
metric_type = 'sp' #sp, accuracy_score, mae, rmse, f-1
logging.basicConfig(filename='logs/'+dataset+'_modelType_'+'metric_type'+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

h_sample_bool = False
h_sample = 0.005 # sample h_sample fraction of historical data

scalability_bool = False
if scalability_bool:
        knn_k_lst = [1, 5, 10, 20, 30]
        lof_k_lst = [1, 5, 10, 20, 30]
        len_knn = len(knn_k_lst)
        len_lof = len(lof_k_lst)
        norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm']
        mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
        od_strategy = ['none', 'if', 'lof']


column_names = ['Age','Workclass','fnlwgt','Education','Education_Num','Martial_Status','Occupation','Relationship','Race','Sex','Capital_Gain','Capital_Loss','Hours_per_week','Country','income']
categorical_cols = ['Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex','Country','income']
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
    contamination_test = 0.2
    contamination_train_lof = 0.1
    contamination_test_lof = 0.2
    
elif dataset == 'adult':
    adult_train  = "data/adult/adult_test.csv"
    adult_test = "data/adult/adult_train.csv"
    train,test = Reader(adult_train,adult_test).load_data()
    #train,test =  train[categorical_cols],test[categorical_cols]
    
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
elif dataset == 'housing':
    house_train  = "data/house/housing_train.csv"
    house_test = "data/house/housing_test.csv"
    train, test = Reader(house_train, house_test).load_data()
    
    missing_percentage = (train.isnull().sum() / len(train)) * 100
    columns_to_drop = missing_percentage[missing_percentage > 40].index.tolist()
    train.drop(columns=columns_to_drop, inplace=True)

    missing_values_count = train.isnull().sum()
    total_cells = np.product(train.shape)
    total_missing = missing_values_count.sum()
    missing_percentage = (total_missing / total_cells) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values_count,
                        'Percentage': (missing_values_count / train.shape[0]) * 100})
#     print("Total missing values:", total_missing)
#     print("Percentage of missing values:", missing_percentage)
#     print("\nMissing value count and percentage per column:")
#     logging.warning(missing_data)

#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#             print(missing_data)
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
#     print("Total missing values:", total_missing)
#     print("Percentage of missing values:", missing_percentage)
#     print("\nMissing value count and percentage per column:")
#     logging.warning(missing_data)

#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(missing_data)
#     y_train = X_train['SalePrice']
#     y_test = X_test['SalePrice']
#     X_train, X_test, y_train, y_test = train_test_split(train, train, test_size=0.6, random_state=10)

#     X_train = train
#     y_train = train['SalePrice'] 
#     X_train.reset_index(drop=True, inplace=True)
#     y_train.reset_index(drop=True, inplace=True)
    categorical_columns = train.select_dtypes(include=['object']).columns
    numerical_columns = train.select_dtypes(include=['int','float']).columns
    
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
    
    X_train = train.copy()
    y_train = X_train['SalePrice'].loc[X_train.index]
    X_train = X_train.drop(['SalePrice'], axis=1)

    X_test = test.copy()
    y_test = X_test['SalePrice'].loc[X_test.index]
    X_test = X_test.drop(['SalePrice'], axis=1)
    
#     X_train = RobustScaler().fit(X_train).transform(X_train) 
# #     y_train = RobustScaler().fit(y_train.to_frame()).transform(y_train.to_frame()) 
#     model = LinearRegression()
#     y_pred = model.fit(X_train, y_train).predict(X_train)
#     print(y_train)
#     print(y_pred)
#     rmse = np.sqrt(root_mean_squared_error(y_train, y_pred)) 
#     print("---RMSE----", rmse)

    from sklearn.feature_selection import f_regression
    selector = SelectKBest(score_func=f_regression, k=15)
#     selector = SelectKBest(score_func=mutual_info_regression, k=15)
    selector.fit(X_train, y_train)
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices]
    feature_scores = selector.scores_
    selected_features_scores = pd.DataFrame({'Feature': X_train.columns, 'Score': feature_scores})
    selected_features_scores.sort_values(by='Score', ascending=False, inplace=True)
#     print(selected_features_scores)
    selected_features_set = set(selected_features)
    # Filter the DataFrame to include only the scores of the selected features
    selected_features_scores_filtered = selected_features_scores[selected_features_scores['Feature'].isin(selected_features_set)]
#     print(selected_features_scores_filtered)
#     logging.warn(X_train.isnull().sum())
#     logging.warn(X_test.isnull().sum())
    print("Selected features: ", selected_features_set)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    tau_train = 0.2 # fraction of missing values
    tau_test = 0.1
    contamination_train = 0.3
    contamination_test = 0.2
    contamination_train_lof = 0.3
    contamination_test_lof = 0.2



if(metric_type=='sp' or metric_type=='accuracy_score' or metric_type=='mae'  or metric_type=='rmse'):
        if modelType == 'lr':
                model = LogisticRegression(random_state=0).fit(X_train, y_train)
        elif modelType == 'nb':
                model = GaussianNB().fit(X_train, y_train)
        elif modelType == 'rf':
                model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        elif modelType == 'reg':
                model = Regression().generate_regression(X_train, y_train)
        # print("Training accuracy : " + str(round(model.score(X_train, y_train), 4)))
        # print("Test accuracy : " + str(round(model.score(X_test, y_test), 4)))

        # y_pred_train = model.predict(X_train)
        # y_pred_test = model.predict(X_test)

        # # p_train = model.predict_proba(X_train)
        # # p_test = model.predict_proba(X_test)
        # if metric_type=='mae' or metric_type=='rmse':
        #         outc_train_mae = mean_absolute_error( y_train,y_pred_train)
        #         outc_test_mae = mean_absolute_error(y_test, y_pred_test)
        #         print(f'MAE training {outc_train_mae} ,test {outc_test_mae}')
        #         outc_train_mse = np.sqrt(root_mean_squared_error(y_train, y_pred_train)) 
        #         outc_test_mse = np.sqrt(root_mean_squared_error(y_test,y_pred_test)) 
        #         print(f'MSE training {outc_train_mse} ,test {outc_test_mse}')
        # print('Done')

class base:
        def __init__(self):
                self.f = 'tdst';
                self.ranking = None
                self.ranges = {}
                self.imputer_strategies = ['drop', 'mean', 'median', 'most_frequent', 'knn']
                self.mv_name_mapping = {'drop': 'mv_drop', 'mean': 'mv_mean', 'median': 'mv_median', 'most_frequent': 'mv_mode', 'knn': 'mv_knn'}
                self.fail = 0
                self.pass_ = 0
                self.fail_with_fallback  = 0
                #KNN 
                self.outlier_strategies = ['none', 'if', 'lof']
                self.ot_name_mapping = {'none': 'od_none', 'if': 'od_if', 'lof': 'od_lof'}

                #1.15 paramater 
                self.normalizer_strategies = ['none', 'ss', 'rs', 'ma', 'mm']
                self.no_name_mapping = {'none': 'norm_none', 'ss': 'norm_ss', 'rs': 'norm_rs', 'ma': 'norm_ma', 'mm': 'norm_mm'}

                #normalization 
                #

                self.ranges['missing_value'] = [1,2,3,4]
                self.ranges['normalization'] = [1, 5, 10, 20, 30]
                self.ranges['outlier'] = [1, 5, 10, 20, 30]

                self.base_strategies  =['missing_value', 'normalization', 'outlier']
                



                self.historical_data = []   
                self.historical_data_pd = []
                self.gs_idistr = []
                self.gs_fdistr = []


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

        def optimize(self, init_params, f_goal):
                
                self.rank_iter = 0  #Iteration count by ranking algorithm
                self.rank_f = 0    # Iteration count whn we found the fairness score less than the calculated fairness from seed value 
                iter_size = 0   # Total iteration allowed in ranking algorithm , falllback is after 1 to grid search 
                cur_params = init_params.copy()
                logging.info(f'Initial seed {init_params}')
                cur_params_opt= {strategy: selection for strategy, selection in zip(self.base_strategies, init_params[:len(self.base_strategies)])}

                opt_f  = self.f_score_look_up2(self.historical_data_pd,init_params)
                
                self.ranges['missing_value'] = list(np.unique(self.historical_data_pd['missing_value']))
                self.ranges['normalization'] = list(np.unique(self.historical_data_pd['normalization']))
                self.ranges['outlier'] = list(np.unique(self.historical_data_pd['outlier']))
                seen = set()
                
                logging.info(f'current param {cur_params}')

                iter_size  = 0
                if(opt_f<f_goal):
                        self.rank_iter = 1  #Iteration count by ranking algorithm
                        self.rank_f = opt_f
                        return 
                seen.add(tuple(cur_params_opt.items()))
                while(iter_size<len(self.coef_rank)):
                
                        if iter_size == 0:
                                
                                for val in self.coef_rank:        
                                        cur_strategy = self.base_strategies[val]
                                        
                                        if(self.coefs[val]<0):
                                                current_paramter_value =  self.ranges[cur_strategy][-1]
                                        else:
                                                current_paramter_value =  self.ranges[cur_strategy][0]
                
                                        cur_params = cur_params_opt.copy()
                                        cur_params[cur_strategy] = current_paramter_value

                                        logging.info(f'Next param {cur_params}')
                                        if(tuple(cur_params.items())) in seen:
                                                continue
                                        cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))
                                        seen.add(tuple(cur_params.items()))
                                        self.rank_iter += 1
                                        
                                        logging.info(f'fairness found : {cur_f} ,optimal fairness : {opt_f}')
                                        opt_f,cur_params_opt,found = self.f_lookup(cur_f,f_goal,cur_params_opt,cur_params,opt_f)
                                        logging.info(f'Optimal paramater {cur_params_opt} ')
                                        
                                        if(found):
                                                return 

                        else:
                                logging.info('Fall back')
                                self.fail_with_fallback +=1
                                comb_size = iter_size
                                i=0
                                comb_lst=[]
                                if(comb_size>2):
                                        print(comb_size)

                                for comb in (itertools.combinations(self.coef_rank, comb_size)):
                                        comb_lst.append(comb)
                                
                                for comb in comb_lst:
                                        i=0
                                        coef_lst=[]
                                        score = {}
                                        while i<comb_size:
                                                coef_lst.append(self.coefs[comb[i]])
                                                i+=1
                                        
                                        for param in self.historical_data:
                                                result = [param[i] for i in list(comb)]
                                                score[tuple(param)] = sum([x*y for x,y in zip( coef_lst,result)])
                                        sorted_params = sorted(score.items(), key=operator.itemgetter(1))
                                        
                                        # cur_params_opt = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params[:len(self.base_strategies)])}
                                        # opt_f  = self.f_score_look_up2(self.historical_data_pd,init_params)
                                        for (elem,score) in sorted_params:#sorted_params:#(elem,score)
                                                cur_params = cur_params_opt.copy()
                                                for j in range(comb_size):
                                                        cur_strategy = self.base_strategies[comb[j]]
                                                        cur_params[cur_strategy] = round(elem[comb[j]], 5)
                                                
                                                # logging.info(f'Next param {cur_params}')
                                                if(tuple(cur_params.items())) in seen:
                                                        continue
                                                seen.add(tuple(cur_params.items()))
                                                self.rank_iter += 1  #Iteration count by ranking algorithm
                                                self.rank_f = opt_f
                                                
                                                logging.info(f'Next param {cur_params}')
                                                cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))
                                                opt_f,cur_params_opt,found = self.f_lookup(cur_f,f_goal,cur_params_opt,cur_params,opt_f)
                                                logging.info(f'Optimal paramater {cur_params_opt}, optimal fairness {opt_f} ')

                                                if(found):
                                                        return cur_params
                        iter_size = iter_size + 1
                        if iter_size == len(self.coef_rank)+1:
                                print('failed to find the optimal pipeline')
                                break

        def f_lookup(self,cur_f,f_goal,cur_params_opt,cur_params,opt_f):
                found = False
                if metric_type=='sp' or metric_type=='mae' or metric_type=='rmse' or metric_type=='accuracy_score':                                                                           
                        if cur_f <= f_goal:
                                self.rank_f = cur_f
                                self.pass_ = self.pass_  + 1
                                logging.info(f'Found the lowest fairness {cur_f}')
                                found  = True
                                # return cur_params.values() # early exit when f_goal obtained

                        elif cur_f < opt_f:
                                opt_f = cur_f
                                cur_params_opt = cur_params
                        
                        
                elif metric_type=='f-1':
                        
                        if(cur_f==0):
                                import pdb;pdb.set_trace()
                        if cur_f >= f_goal:
                                self.rank_f = cur_f
                                self.pass_ = self.pass_  + 1
                                found  = True
                                # return cur_params.values() # early exit when f_goal obtained

                        elif cur_f > opt_f:
                                # cur_params = cur_params.values()
                                opt_f = cur_f
                                cur_params_opt = cur_params
                return opt_f,cur_params_opt,found

        def f_score_look_up2(self,profiles_df,elem):
                column_names =['missing_value', 'normalization', 'outlier', 'fairness']
                try:
                        return round(profiles_df.loc[(profiles_df[column_names[0]] == elem[0]) & (profiles_df[column_names[1]] == elem[1] ) 
                                               & (profiles_df[column_names[2]] == elem[2])].iloc[0]['fairness'],5)
                except Exception as e :
                        print(e)
                        import pdb;pdb.set_trace()

        def grid_search(self, f_goal,seen):
                self.gs_idistr = []
                self.gs_fdistr = []
                gs_iter = 0     
                gs_f = 0
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
                                if cur_f <= f_goal:
                                        gs_f = cur_f
                                        cur_params = []
                                        cur_params.append(elem[0]) 
                                        cur_params.append(elem[1])
                                        cur_params.append(elem[2])
                                        self.gs_fdistr.append(gs_f)
                                        self.gs_idistr.append(gs_iter)
                                        return gs_iter,gs_f
                        elif metric_type=='f-1':
                                if cur_f >= f_goal:
                                        gs_f = cur_f
                                        cur_params = []
                                        cur_params.append(elem[0]) 
                                        cur_params.append(elem[1])
                                        cur_params.append(elem[2])
                                        self.gs_fdistr.append(gs_f)
                                        self.gs_idistr.append(gs_iter)
                                        return gs_iter,gs_f

        def create_historic_data(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
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
                        if(metric_type=='sp' or metric_type=='accuracy_score'):
                                priv_idx_train, unpriv_idx_train, sensitive_attr_train = self.getIdxSensitive(X_train, dataset)
                                # train_eqOdds = self.computeStatisticalParity(y_pred_train[priv_idx_train], y_pred_train[unpriv_idx_train])
                                # print("Training fairness : " + str(round(train_eqOdds, 4)))


                        for param1 in range(mv_params):
                                for param2 in range(norm_params):
                                        for param3 in range(od_params):
                                                # import pdb;pdb.set_trace()
                                                if param1 < len(mv_strategy) - 1:
                                                        
                                                        if mv_strategy[param1] == 'drop':
                                        # print("Dropping missing values ...")
                                                                mv_idx = []
                                                                if dataset == 'hmda':
                                                                        mv_idx = X_train[X_train['lien_status'].isna()].index.tolist()
                                                                elif dataset == 'adult':
                                                                        mv_idx = X_train[X_train['Martial_Status'].isna()].index.tolist()
                                                                elif dataset == 'housing':
                                                                        mv_idx = X_train[X_train['OverallQual'].isna()].index.tolist()
                                                                        
                                                                # import pdb;pdb.set_trace()
                                                                
                                                                
                                                                imputed_X_train = X_train.drop(mv_idx)
                                                                imputed_X_train.reset_index(drop=True, inplace=True)
                                                                updated_y_train = y_train.copy()
                                                                for idx in sorted(mv_idx, reverse=True):
                                                                        del updated_y_train[idx]
                                                                updated_y_train.reset_index(drop=True, inplace=True)
                                                                if metric_type=='sp' or metric_type=='accuracy_score':
                                                                        
                                                                        updated_sensitive_attr_train = sensitive_attr_train.drop(mv_idx)
                                                                        updated_sensitive_attr_train.reset_index(drop=True, inplace=True)
                                                                mv_param = [1]
                                                        elif mv_strategy[param1] in ['mean', 'median', 'most_frequent']:
                                                                imputed_X_train = SimpleImputer(missing_values=np.nan, strategy=mv_strategy[param1]).fit(X_train).transform(X_train)
                                                                updated_y_train = y_train.copy()
                                                                if metric_type=='sp' or metric_type=='accuracy_score':
                                                                        updated_sensitive_attr_train = sensitive_attr_train.copy()
                                                                if mv_strategy[param1] == 'mean':
                                                                        mv_param = [2]
                                                                if mv_strategy[param1] == 'median':
                                                                        mv_param = [3]
                                                                if mv_strategy[param1] == 'most_frequent':
                                                                        mv_param = [4]
                                                else:
                                                        k = knn_k_lst[param1-4] # start accessing number of neighbors in knn
                                                        imputed_X_train = KNNImputer(n_neighbors=k).fit_transform(X_train)
                                                        mv_param = [param1+1]
                                
                                
                                                if norm_strategy[param2] == 'none':
                                                        scaled_X_train = imputed_X_train.copy()
                                                        norm_param = [1]
                                                elif norm_strategy[param2] == 'ss':
                                                        scaled_X_train = StandardScaler().fit(imputed_X_train).transform(imputed_X_train) 
                                                        norm_param = [2]
                                                elif norm_strategy[param2] == 'rs':
                                                        scaled_X_train = RobustScaler().fit(imputed_X_train).transform(imputed_X_train) 
                                                        norm_param = [3]
                                                elif norm_strategy[param2] == 'ma':
                                                        scaled_X_train = MaxAbsScaler().fit(imputed_X_train).transform(imputed_X_train) 
                                                        norm_param = [4]
                                                elif norm_strategy[param2] == 'mm':
                                                        scaled_X_train = MinMaxScaler().fit(imputed_X_train).transform(imputed_X_train) 
                                                        norm_param = [5]

                                                if isinstance(scaled_X_train, np.ndarray):
                                                       scaled_X_train = pd.DataFrame(data=scaled_X_train,columns=X_train.columns)
                                                if param3 < len(od_strategy) - 1:
                                                        if od_strategy[param3] == 'none':
                                                                outlier_y_pred = np.ones(len(scaled_X_train))
                                                                od_param = [1]
                                                        if od_strategy[param3] == 'if':
                                                                outlier_y_pred = IsolationForest(n_estimators=50,contamination=contamination_train,random_state=0).fit_predict(scaled_X_train)
                                                                od_param = [2]
                                                                # print("Fraction of outliers: ", round((1 - sum(mask)/len(outlier_y_pred)) * 100, 4))
                                                else:
                                                        k = lof_k_lst[param3 - 2] # start accessing number of neighbors in lof
                                                        outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=contamination_train_lof).fit_predict(scaled_X_train)
                                                        od_param =  [param3+1]
                                                mask = outlier_y_pred != -1
 
                                                outlier_X_train = scaled_X_train.copy()
                                                outlier_y_train = updated_y_train.copy()
                                                if metric_type=='sp'  :
                                                        outlier_sensitive_train = updated_sensitive_attr_train.copy()
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                outlier_X_train, outlier_y_train, outlier_sensitive_train = scaled_X_train[mask], updated_y_train[mask], updated_sensitive_attr_train[mask]
                                                                outlier_y_train.reset_index(drop=True, inplace=True)
                                                                outlier_X_train.reset_index(drop=True, inplace=True)
                                                                outlier_sensitive_train.reset_index(drop=True, inplace=True)          
                                                        priv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 1]
                                                        unpriv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 0]
                                                elif(modelType == 'reg' or metric_type=='accuracy_score'):
                                                        
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                outlier_X_train, outlier_y_train = scaled_X_train[mask], updated_y_train[mask]
                                                                outlier_y_train.reset_index(drop=True, inplace=True)
                                                                outlier_X_train.reset_index(drop=True, inplace=True)
                                                        
                                                
                                                updated_model = None
                                                # import pdb;pdb.set_trace()
                                                if modelType == 'lr':
                                                        updated_model = LogisticRegression(random_state=0).fit(outlier_X_train, outlier_y_train)
                                                elif modelType == 'nb':
                                                        updated_model = GaussianNB().fit(outlier_X_train, outlier_y_train)
                                                elif modelType == 'rf':
                                                        updated_model = RandomForestClassifier(random_state=0).fit(outlier_X_train, outlier_y_train)
                                                elif modelType=='reg':
                                                        updated_model = Regression().generate_regression(outlier_X_train,outlier_y_train)



                                                y_pred = updated_model.predict(outlier_X_train)
                                                # import pdb;pdb.set_trace()
                                                outc = None
                                                if metric_type== 'sp':
                                                        outc = self.computeStatisticalParity(y_pred[priv_idx_train],y_pred[unpriv_idx_train])
                                                elif metric_type=='f-1':
                                                        outc = f1_score(outlier_y_train,y_pred)
                                                elif metric_type=='mae':
                                                        outc = mean_absolute_error(outlier_y_train, y_pred)
                                                elif metric_type=='rmse':
                                                        outc = np.sqrt(root_mean_squared_error(outlier_y_train, y_pred)) 
                                                elif metric_type=='accuracy_score':
                                                        outc = 1-accuracy_score(outlier_y_train,y_pred)

                                                if(outc==0):
                                                        print()

                                                f = [outc]
                                                param_lst.append(mv_param + norm_param + od_param + f)
                                                # if eqOdds_train == 0.0:
                                                print(str(mv_param + norm_param + od_param + f))
                                        # print(len(unpriv_idx_train))

                        param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        
                        param_lst_df.to_csv(file_name, index=False)
                        # import pdb;pdb.set_trace()
                else :
                        param_lst_df = pd.read_csv(file_name)[["missing_value","normalization","outlier","fairness"]]

                y = param_lst_df['fairness']
                X = param_lst_df.copy().drop("fairness", axis=1)
                # X = StandardScaler().fit(X).transform(X)

                if (h_sample_bool):
                        import math
                        print(" ------ Sampling --------", h_sample)
                        random.seed(42)
                        sample_idx = random.sample(list(range(len(X))), math.ceil(h_sample * len(X)))
                        X = X.iloc[sample_idx]
                        y = y.iloc[sample_idx]

                reg = Regression()
                model = reg.generate_regression(X, y)
                coefs = model.coef_
                print(coefs)
                print(model.intercept_)
                self.coefs = coefs
                self.coef_rank = np.argsort(np.abs(self.coefs)).tolist()[::-1]
                print(self.coef_rank)
                
                logging.info(f'coef {self.coefs}')

        def create_historic_data_test(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
                if not(os.path.exists(file_name)):
                        idx_test = np.arange(0, len(X_test), 1)
                        mv_test = pd.DataFrame(idx_test).sample(frac=tau_test, replace=False, random_state=1).index
                        if (dataset == 'hmda'):
                                X_test['lien_status'][mv_test] = np.NaN
                        elif (dataset == 'adult'):
                                X_test['Martial_Status'][mv_test] = np.NaN
                        elif(dataset=='housing'):
                                X_test['OverallQual'][mv_test] = np.NaN

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
                        if metric_type == 'sp' or metric_type=='accuracy_score' :
                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)


                        for param1 in range(mv_params):
                                for param2 in range(norm_params):
                                        for param3 in range(od_params):
                                                if param1 < len(mv_strategy) - 1:
                                                        if mv_strategy[param1] == 'drop':
                                        # print("Dropping missing values ...")
                                                                if dataset == 'hmda':
                                                                        mv_idx = X_test[X_test['lien_status'].isna()].index.tolist()
                                                                elif dataset == 'adult':
                                                                        mv_idx = X_test[X_test['Martial_Status'].isna()].index.tolist()
                                                                elif dataset == 'housing':
                                                                        mv_idx = X_test[X_test.isna().any(axis=1)].index.tolist()
                                                                imputed_X_test = X_test.drop(mv_idx)
                                                                imputed_X_test.reset_index(drop=True, inplace=True)
                                                                updated_y_test = y_test.copy()
                                                                for idx in sorted(mv_idx, reverse=True):
                                                                        del updated_y_test[idx]
                                                                updated_y_test.reset_index(drop=True, inplace=True)
                                                                if metric_type=='sp' or metric_type=='accuracy_score':
                                                                        updated_sensitive_attr_test = sensitive_attr_test.drop(mv_idx)
                                                                        updated_sensitive_attr_test.reset_index(drop=True, inplace=True)
                                                                mv_param = [1]
                                                        elif mv_strategy[param1] in ['mean', 'median', 'most_frequent']:
                                                                imputed_X_test = SimpleImputer(missing_values=np.nan, strategy=mv_strategy[param1]).fit(X_test).transform(X_test)
                                                                updated_y_test = y_test.copy()
                                                                if metric_type=='sp' or metric_type=='accuracy_score' :
                                                                        updated_sensitive_attr_test = sensitive_attr_test.copy()
                                                                if mv_strategy[param1] == 'mean':
                                                                        mv_param = [2]
                                                                if mv_strategy[param1] == 'median':
                                                                        mv_param = [3]
                                                                if mv_strategy[param1] == 'most_frequent':
                                                                        mv_param = [4]
                                                                        # import pdb;pdb.set_trace()
                                                else:
                                                        k = knn_k_lst[param1-4] # start accessing number of neighbors in knn
                                                        imputed_X_test = KNNImputer(n_neighbors=k).fit_transform(X_test)
                                                        mv_param = [param1+1]
                                
                                
                                                if norm_strategy[param2] == 'none':
                                                        scaled_X_test = imputed_X_test.copy()
                                                        norm_param = [1]
                                                elif norm_strategy[param2] == 'ss':
                                                        scaled_X_test = StandardScaler().fit(imputed_X_test).transform(imputed_X_test) 
                                                        norm_param = [2]
                                                elif norm_strategy[param2] == 'rs':
                                                        scaled_X_test = RobustScaler().fit(imputed_X_test).transform(imputed_X_test) 
                                                        norm_param = [3]
                                                elif norm_strategy[param2] == 'ma':
                                                        scaled_X_test = MaxAbsScaler().fit(imputed_X_test).transform(imputed_X_test) 
                                                        norm_param = [4]
                                                elif norm_strategy[param2] == 'mm':
                                                        scaled_X_test = MinMaxScaler().fit(imputed_X_test).transform(imputed_X_test) 
                                                        norm_param = [5]
                                                if isinstance(scaled_X_test, np.ndarray):
                                                       
                                                       scaled_X_test = pd.DataFrame(data=scaled_X_test,columns=X_test.columns)
                                                if param3 < len(od_strategy) - 1:
                                                        if od_strategy[param3] == 'none':
                                                                outlier_y_pred = np.ones(len(scaled_X_test))
                                                                od_param = [1]
                                                        if od_strategy[param3] == 'if':
                                                                outlier_y_pred = IsolationForest(n_estimators=50,contamination=contamination_test,random_state=0).fit_predict(scaled_X_test)
                                                                od_param = [2]
                                                else:
                                                        k = lof_k_lst[param3 - 2] # start accessing number of neighbors in lof
                                                        outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=contamination_test_lof).fit_predict(scaled_X_test)
                                                        od_param = [param3+1]
                                                mask = outlier_y_pred != -1

                                                outlier_X_test = scaled_X_test.copy()
                                                outlier_y_test = updated_y_test.copy()
                                                priv_idx_test = None
                                                unpriv_idx_test = None
                                                if(metric_type=='sp' ):
                                                        outlier_sensitive_test = updated_sensitive_attr_test.copy()
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                # import pdb;pdb.set_trace()
                                                                outlier_X_test, outlier_y_test, outlier_sensitive_test = scaled_X_test[mask], updated_y_test[mask], updated_sensitive_attr_test[mask]
                                                                outlier_y_test.reset_index(drop=True, inplace=True)
                                                                outlier_X_test.reset_index(drop=True, inplace=True)
                                                                outlier_sensitive_test.reset_index(drop=True, inplace=True)   

                                                        priv_idx_test = [i for i, val in enumerate(outlier_sensitive_test) if val == 1]
                                                        unpriv_idx_test = [i for i, val in enumerate(outlier_sensitive_test) if val == 0]
                                                elif dataset=='housing' or metric_type=='accuracy_score':
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                # import pdb;pdb.set_trace()
                                                                outlier_X_test, outlier_y_test = scaled_X_test[mask], updated_y_test[mask]
                                                                outlier_y_test.reset_index(drop=True, inplace=True)
                                                                outlier_X_test.reset_index(drop=True, inplace=True)


                                                updated_model = None
                                                #import pdb;pdb.set_trace()
                                                if modelType == 'lr':
                                                        updated_model = LogisticRegression(random_state=0).fit(outlier_X_test, outlier_y_test)
                                                elif modelType == 'nb':
                                                        updated_model = GaussianNB().fit(outlier_X_test, outlier_y_test)
                                                elif modelType == 'rf':
                                                        updated_model = RandomForestClassifier(random_state=0).fit(outlier_X_test, outlier_y_test)
                                                elif modelType=='reg':
                                                        updated_model = Regression().generate_regression(outlier_X_test,outlier_y_test)

                                                y_pred = updated_model.predict(outlier_X_test)
                                                # eqOdds_test = self.computeEqualizedOdds(y_pred, outlier_y_test, priv_idx_test, unpriv_idx_test)
                                                outc = None
                                                # import pdb;pdb.set_trace()
                                                if metric_type== 'sp':
                                                        outc = self.computeStatisticalParity(y_pred[priv_idx_test],y_pred[unpriv_idx_test])
                                                elif metric_type=='f-1':
                                                        outc = f1_score(outlier_y_test,y_pred)
                                                elif metric_type=='accuracy_score':
                                                        outc = 1-accuracy_score(outlier_y_test,y_pred)
                                                elif metric_type=='mae':
                                                        outc = mean_absolute_error(outlier_y_test, y_pred)
                                                elif metric_type=='rmse':
                                                        outc = np.sqrt(root_mean_squared_error(outlier_y_test, y_pred)) 
                                                f = [outc]
                                                param_lst.append(mv_param + norm_param + od_param + f)
                                                
                                                print(str(mv_param + norm_param + od_param + f))
                                        


                        param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        
                        param_lst_df.to_csv(file_name,index=False)

        def write_quartiles(self,csv_writer, algorithm, metric, quartiles):
                if dataset in ['adult', 'hmda']:
                        csv_writer.writerow([round(1 - f_goal, 2), algorithm, f"{metric} q1", round(quartiles[0], 5)])
                        csv_writer.writerow([round(1 - f_goal, 2), algorithm, f"{metric} q2", round(quartiles[1], 5)])
                        csv_writer.writerow([round(1 - f_goal, 2), algorithm, f"{metric} q3", round(quartiles[2], 5)])
                        csv_writer.writerow([round(1 - f_goal, 2), algorithm, f"{metric} q4", round(quartiles[3], 5)])
                else:
                        csv_writer.writerow([round(1 - (f_goal - min(f_goals))/min(f_goals), 2), algorithm, f"{metric} q1", round(quartiles[0], 5)])
                        csv_writer.writerow([round(1 - (f_goal - min(f_goals))/min(f_goals), 2), algorithm, f"{metric} q2", round(quartiles[1], 5)])
                        csv_writer.writerow([round(1 - (f_goal - min(f_goals))/min(f_goals), 2), algorithm, f"{metric} q3", round(quartiles[2], 5)])
                        csv_writer.writerow([round(1 - (f_goal - min(f_goals))/min(f_goals), 2), algorithm, f"{metric} q4", round(quartiles[3], 5)])
p = base()



filename_test = 'historical_data/historical_data_test_profile_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
filename_train = 'historical_data/historical_data_train_profile_'+modelType+'_'+metric_type+'_'+dataset+'.csv'

if scalability_bool:
        filename_test = 'metric/scalability/historical_data_test_profile_'+modelType+'_'+metric_type+'_'+dataset+'_'+str(len(knn_k_lst))+'.csv'
        filename_train = 'metric/scalability/historical_data_train_profile_'+modelType+'_'+metric_type+'_'+dataset+'_'+str(len(knn_k_lst))+'.csv'

p.create_historic_data(filename_train)
p.create_historic_data_test(filename_test)

if(dataset =='adult'):
        f_goals = [0.1, 0.15, 0.2, 0.32]
        f_goals = [0.05, 0.06, 0.13, 0.16]
elif(dataset=='hmda'):
        f_goals  = [.06, .07, 0.08, 0.09] #original data
        # f_goals  = [.06] #scalability
elif(dataset=='housing'):
        f_goals = [162, 170, 180, 185]
else:
        print('Please profile goals ')

# #Read from historical data gererated on training data 
historical_data = pd.read_csv(filename_test)
p.historical_data_pd = historical_data;
# #Convert to list of list containing all the combination of transformers in a fixed pipeline
p.historical_data = historical_data.values.tolist();

if (h_sample_bool):
        metric_path  = 'metric/ablation/historical_1step_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
elif (scalability_bool):
        # metric_path  = 'metric/scalability/historical_1step_'+modelType+'_'+metric_type+'_'+dataset+'_'+str(len(knn_k_lst))+'.csv'
        metric_path  = 'metric/scalability/historical_1step_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
else:
      metric_path  = 'metric/metric_'+modelType+'_'+metric_type+'_'+dataset+'.csv'

f = sys.stdout
f_mode = 'w'
if h_sample_bool or scalability_bool:
       f_mode = 'a'
f = open(metric_path, f_mode)


csv_writer = csv.writer(f)

for f_goal in f_goals:
        logging.info(f'Fairness goal {f_goal}')
        rank_idistr = []
        rank_fdistr = []
        gs_idistr = []
        gs_fdistr = []

        failures = 0
        for seed_ in historical_data.values.tolist():
                seen = set()
                # p.grid_search(f_goal,seen)
                # gs_idistr.append(p.gs_idistr[0])
                # gs_fdistr.append(p.gs_fdistr[0])
                p.optimize(seed_, f_goal)

                if (h_sample_bool):
                        csv_writer.writerow([seed_[0], seed_[1], seed_[2], h_sample, 'param', p.rank_iter])
                elif (scalability_bool):
                        csv_writer.writerow([seed_[0], seed_[1], seed_[2], len(historical_data.values.tolist()), 'param', p.rank_iter])

                if p.rank_iter != -1:
                        rank_idistr.append(p.rank_iter)
                        rank_fdistr.append(p.rank_f)
                else:
                        failures += 1

        print(p.fail)
        print(p.pass_)
        print(p.fail_with_fallback)
        # csv_writer.writerow([f" failed {p.fail}" , f" passed { p.pass_}"])
        p.pass_   = 0 
        p.fail  = 0 
        p.fail_with_fallback = 0
        rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75,100], interpolation='midpoint')
        rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75,100], interpolation='midpoint')
        # g_iquartiles = np.percentile(gs_idistr, [25, 50, 75,100], interpolation='midpoint')
        # g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75,100], interpolation='midpoint')
        print("Fairness goal stats: " + str(f_goal))

        if not (h_sample_bool or scalability_bool):
                csv_writer.writerow(["Fairness Goal", "Grid search", "Iteration", "Value"])

                # Write data for ranking algorithm
                p.write_quartiles(csv_writer, "ranking", "iterations", rank_iquartiles)
                p.write_quartiles(csv_writer, "ranking", "Fairness", rank_fquartiles)
                csv_writer.writerow([])

                # Write data for grid search algorithm
                # p.write_quartiles(csv_writer, "grid search", "iterations", g_iquartiles)
                # p.write_quartiles(csv_writer, "grid search", "Fairness", g_fquartiles)
                # csv_writer.writerow([])
f.close()


