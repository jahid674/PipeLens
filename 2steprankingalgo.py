import itertools
from modules.outlier_detection.outlier_detector import OutlierDetector
from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
from modules.normalization.normalizer import DataNormalizer
from modules.metric.metric import metric
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from regression import Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
from modules.outlier_detection.outlier_detector import OutlierDetector
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



import statistics 
from sklearn.preprocessing import LabelEncoder
# import prose.datainsights as di
from scipy.stats import chisquare,chi2_contingency
from scipy import stats


tau = 0.1 # fraction of missing values
# knn_k = 1 # knn number of neighbors
# lof_k = 50 # number of neighbors for local outlier factor
lof_contamination = 'auto' 
knn_k_lst = [1, 5, 10, 20, 30]
lof_k_lst = [1, 5, 10, 20, 30]
len_knn = len(knn_k_lst)
len_lof = len(lof_k_lst)
norm_strategy = [ 'none', 'ss', 'rs', 'ma', 'mm'] # standard scaler, robust scaler, max absolute scaler, minmax scaler
mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
# od_strategy = ['zs', 'iqr', 'if', 'lof'] # local outlier factor, z-score, interquartile range, isolation forest
od_strategy = ['none', 'if', 'lof'] # local outlier factor, isolation forest
dataset = 'hmda'
modelType = 'lr' #'lr' # 'nb' Logistic Regression or Gaussian Naive Bayes
metric_type = 'sp'
algo_type = 'projection'

logging.basicConfig(filename='logs/profile_'+algo_type+"_"+dataset+'_'+modelType+'_'+'metric_type'+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



from scipy.stats import pearsonr
class Profile:
  profile_lst=[]
  def __init__(self,df):
    self.df=df
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
    else:
       target = 'income'
       
    
    for column in data_final.columns:

        if(column==target):
          
          continue
        
        if column in numerical_columns and column != 'lien_status':
                corr = self.correlation(data_final[column],data_final[target])
        else:
               corr = self.categorical_correlation(data_final[column],data_final[target])
               
        # missing_value = self.missing(self.df[categorical_columns[i]])
        #outlier  = self.outlier(self.df[categorical_columns[i]])
        
        name = column
        tuple = ('corr_' + name)
        profile[tuple]= column,round(corr,5)
        i+=1
    dd = []
    keys = []
    for val in profile:
        # import pdb;pdb.set_trace()
        dd.append(profile[val][1])
        
        keys.append(val)
    dd.append(outlier)
    keys.append('Outlier')
    return dd,keys


numerical_columns = []
if dataset == 'hmda':
    hmda_train  = "data/hmda/hmda_Orleans_X_train_1.csv"
    hmda_test = "data/hmda/hmda_Calcasieu_X_test_1.csv"
    # hmda_train = "data/hmda/hmda_Calcasieu_X_test_1.csv"
    train,test = Reader(hmda_train,hmda_test).load_data()
    categorical_columns = train.select_dtypes(include=['object']).columns
    numerical_columns = train.select_dtypes(include=['int', 'float']).columns
    y_train = train['action_taken']
    X_train = train.drop('action_taken', axis=1)

    y_test = test['action_taken']
    X_test = test.drop('action_taken', axis=1)
elif dataset == 'adult':
    adult_train  = "data/adult/adult_train.csv"
    adult_test = "data/adult/adult_test.csv"
    train,test = Reader(adult_train,adult_test).load_data()
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
elif dataset == 'housing':
    house_train  = "data/house/house_train.csv"
    house_test = "data/house/test.csv"
    train,test = Reader(house_train,house_test).load_data()
#     import pdb;pdb.set_trace()
    y = train['SalePrice']
    categorical_columns = ['SaleType',
    'MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','TotRmsAbvGrd','Functional','FireplaceQu','GarageType','GarageFinish',
    'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleCondition']
    X = train.drop('SalePrice', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#     y_train = train['SalePrice']
    label_encoder = LabelEncoder()
    label_encoders = {}
    
    for column in categorical_columns:
        le = LabelEncoder()
        X_train[column] = label_encoder.fit_transform(X_train[column]) 
        X_test[column] = label_encoder.fit_transform(X_test[column]) 
        label_encoders[column] = le

#     y_test = test['SalePrice']
#     X_test = test.drop('SalePrice', axis=1)

if(metric_type=='sp'):
        if modelType == 'lr':
                model = LogisticRegression(random_state=0).fit(X_train, y_train)
        elif modelType == 'nb':
                model = GaussianNB().fit(X_train, y_train)
        print("Training accuracy : " + str(round(model.score(X_train, y_train), 4)))
        print("Test accuracy : " + str(round(model.score(X_test, y_test), 4)))

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        p_train = model.predict_proba(X_train)
        p_test = model.predict_proba(X_test)

class base:
        def __init__(self):
                self.f = 'tdst';
                self.ranking = None
                self.ranges = {}
                self.imputer_strategies = ['drop', 'mean', 'median', 'most_frequent', 'knn']
                self.mv_name_mapping = {'drop': 'mv_drop', 'mean': 'mv_mean', 'median': 'mv_median', 'most_frequent': 'mv_mode', 'knn': 'mv_knn'}
                self.fail = 0
                self.pass_ = 0
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

        def generate_combination_of_all_transformer(self):
                self.all_binary_vectors = []
                final_strategies = self.normalizer_strategies + self.imputer_strategies + self.outlier_strategies
                for nor in self.normalizer_strategies:
                        for impt in self.imputer_strategies:
                                for neighbors in self.outlier_strategies:
                                        # Select one strategy for each category
                                        selected_imputer_strategy = impt
                                        selected_outlier_strategy = neighbors
                                        selected_normalizer_strategy = nor 

                                        # Create binary vectors for each category
                                        imputer_vector = [1 if strategy == selected_imputer_strategy else 0 for strategy in self.imputer_strategies]
                                        outlier_vector = [1 if strategy == selected_outlier_strategy else 0 for strategy in self.outlier_strategies]
                                        normalizer_vector = [1 if strategy == selected_normalizer_strategy else 0 for strategy in self.normalizer_strategies]
                                        binary_vector = normalizer_vector + outlier_vector + imputer_vector  
                                        self.all_binary_vectors.append(binary_vector)

       
                # Print all generated binary vectors
                for binary_vector in self.all_binary_vectors:
                        print(binary_vector)


 
        
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


        def optimize_1step(self, init_params, f_goal):
             
                # print(f'1 step Starting Grasp search for fairness score:{f_goal}')
                #======Intitalize variable start here =====
                # test = [0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 30.0, 0.2296946068875893]
                test = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.06203095684803]
                self.rank_iter = 0  #Iteration count by ranking algorithm
                self.rank_f = 0    # Iteration count whn we found the fairness score less than the calculated fairness from seed value 
                iter_size = 0   # Total iteration allowed in ranking algorithm , falllback is after 1 to grid search 
                cur_params = init_params.copy()
                #========end=========
                # self.fail = 0
                # self.pass_ = 0
                
                #  From seed value calculate the new fairness score
                #=========Code Start========
                logging.info(f'Initita seed {init_params}')
                cur_params_opt = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params[:len(self.base_strategies)])}
                
                # positive_index = [index for index, value in enumerate(cur_params_init.values()) if value > 0]
                # try:
                #         i_imputer = positive_index[0]
                #         i_normal  = positive_index[1]-2
                # i_outlier = positive_index[2]-2
                # import pdb;pdb.set_trace()
                opt_f  = self.f_score_look_up2(self.historical_data_pd,init_params)
                
                self.ranges['missing_value'] = list(np.unique(self.historical_data_pd['missing_value']))
                self.ranges['normalization'] = list(np.unique(self.historical_data_pd['normalization']))
                self.ranges['outlier'] = list(np.unique(self.historical_data_pd['outlier']))
                #print(f'Fairness score from seed value : {opt_f}')
                #======Code end ============
                # print(init_params)
                # if(init_params==test):
                #         import pdb;pdb.set_trace()
                seen = set()
                # import pdb;pdb.set_trace()
                logging.info(f'current param {cur_params}')
         
                # while ( iter_size < 9 ):
                #         iter_size += 1      
                #         index = 0
                iter_size  = 1
                # prev_strategies_imp = self.base_strategies[positive_index[0]]
                # prev_strategies_norm = self.base_strategies[positive_index[1]]
                # prev_strategies_outlier =  self.base_strategies[positive_index[2]]
                if(opt_f<f_goal):
                        self.rank_iter = 1  #Iteration count by ranking algorithm
                        self.rank_f = opt_f
                        return 
                seen.add(tuple(cur_params_opt.items()))
                if iter_size == 1:
                        goal_pipline = False
                        
                        for val in self.coef_rank:
                                cur_strategy = self.base_strategies[val]
                                
                                current_paramter_selected = cur_params_opt[cur_strategy]
                                                
                                indx = 0
                                
                                indx = self.ranges[cur_strategy].index(current_paramter_selected)
                                
                                if(self.coefs[val]<0):
                                        current_paramter =  self.ranges[cur_strategy][indx:len(self.ranges[cur_strategy])]
                                else:
                                        current_paramter =  self.ranges[cur_strategy][0:indx][::-1]
                                
                                for param_val  in current_paramter:
                                        cur_params = cur_params_opt
                                        
                                        cur_params[cur_strategy] = param_val
                                                
                                        cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))
                                        seen.add(tuple(cur_params.items()))
                                        self.rank_iter += 1
                                        logging.info(f'Next param {cur_params}')
                                        logging.info(f'fairness found : {cur_f} ,optimal fairness : {opt_f}')
                                        logging.info(f'Optimal paramater {cur_params_opt} ')
                                        if metric_type=='sp' or metric_type=='mae' or metric_type=='rmse':                                                                           
                                                if cur_f <= f_goal:
                                                        self.rank_f = cur_f
                                                        self.pass_ = self.pass_  + 1
                                                        logging.info(f'Found the lowest fairness {cur_f}')
                                                        return cur_params.values() # early exit when f_goal obtained
                                                elif cur_f < opt_f:
                                                        # cur_params = cur_params.values()
                                                        opt_f = cur_f
                                                        cur_params_opt = cur_params
                                                
                                                
                                        elif metric_type=='f-1':
                                                
                                                if(cur_f==0):
                                                        import pdb;pdb.set_trace()
                                                if cur_f >= f_goal:
                                                        # cur_params = test_params.copy()
                                                # print(f'Found Fairness score found for tupple {tuple(cur_params.items())}: {cur_f}')
                                                        self.rank_f = cur_f
                                                        f_goal_found = True
                                                        #print('Found')
                                                        self.pass_ = self.pass_  + 1
                                                        return cur_params.values() # early exit when f_goal obtained
                        
                                                elif cur_f > opt_f:
                                                        # cur_params = cur_params.values()
                                                        opt_f = cur_f
                                                        cur_params_opt = cur_params
                                        



                        iter_size = iter_size + 1

                if(iter_size>1):
                        #print("Grasp algo not found ,Starting grid search")
                        # import pdb;pdb.set_trace()
                        # print('yes')
                        self.fail = self.fail  + 1
                        # print(init_params)
                        self.rank_iter,self.rank_f   = self.grid_search(f_goal,1,seen)
                return cur_params



        


        def f_score_look_up2(self,profiles_df,elem):
                column_names =['missing_value', 'normalization', 'outlier', 'fairness']
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
                iter_lst = []
                for i in range(iterations):
                        gs_iter = 0
                        gs_f = 0
                        cur_order = self.historical_data
                        
                        #Added randomness for grid search 
                        random.shuffle(cur_order)
                        
                        for elem in cur_order:
                                
                                
                                # cur_f = self.get_fairness_score_from_test_data(dict_strategy,normalizer_strat = self.base_strategies[positive_positions[0]],imputer_strat = self.base_strategies[positive_positions[1]],outlier_strat = self.base_strategies[positive_positions[2]])
                                # import pdb;pdb.set_trace()
                                cur_f = self.f_score_look_up2(self.historical_data_pd,elem)
                                cur_params = {strategy: selection for strategy, selection in zip(self.base_strategies, elem[:len(self.base_strategies)])}
                                if(tuple(cur_params.items())) in seen:
                                        continue
                                        # seen.add(tuple(cur_params.items()))
                                
                                seen.add(tuple(cur_params.items()))
                                gs_iter += 1
                                if metric_type=='sp' or metric_type=='mae' or metric_type=='rmse':
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
                                        # print('i am in f1 score')
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
                if(metric_type=='sp'):
                        key_profile = ['profile_outlier','diff_sensitive_attr','ratio_sensitive_attr']
                else:
                       key_profile = ['corr_race', 'corr_gender', 'corr_loan_type', 'corr_applicant_age', 'corr_lien_status', 'corr_LV', 'corr_DI', 'corr_income_brackets', 'Outlier']
                       
                
                if not(os.path.exists(file_name)):
                        idx_train = np.arange(0, len(X_train), 1)
                        mv_train = pd.DataFrame(idx_train).sample(frac=tau, replace=False, random_state=1).index
                        if (dataset == 'hmda'):
                                X_train['lien_status'][mv_train] = np.NaN
                        elif (dataset == 'adult'):
                                # 
                                X_train['Martial_Status'][mv_train] = np.NaN

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
                        
                        if(metric_type=='sp'):
                                priv_idx_train, unpriv_idx_train, sensitive_attr_train = self.getIdxSensitive(X_train, dataset)
                                sens_attr_name = sensitive_attr_train.name
                                target_variable_name = y_train.name
                                train_eqOdds = self.computeStatisticalParity(y_pred_train[priv_idx_train], y_pred_train[unpriv_idx_train])
                                print("Training fairness : " + str(round(train_eqOdds, 4)))

                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)
                                test_eqOdds = self.computeStatisticalParity(y_pred_test[priv_idx_test], y_pred_test[unpriv_idx_test])
                                print("Test fairness : " + str(round(test_eqOdds, 4)))

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
                                                                else:
                                                                        mv_idx = X_train[X_train.isna().any(axis=1)].index.tolist()
                                                                        
                                                                # import pdb;pdb.set_trace()
                                                                
                                                                
                                                                imputed_X_train = X_train.drop(mv_idx)
                                                                imputed_X_train.reset_index(drop=True, inplace=True)
                                                                updated_y_train = y_train.copy()
                                                                for idx in sorted(mv_idx, reverse=True):
                                                                        del updated_y_train[idx]
                                                                updated_y_train.reset_index(drop=True, inplace=True)
                                                                if metric_type=='sp':
                                                                        
                                                                        updated_sensitive_attr_train = sensitive_attr_train.drop(mv_idx)
                                                                        updated_sensitive_attr_train.reset_index(drop=True, inplace=True)
                                                                mv_param = [1]
                                                        elif mv_strategy[param1] in ['mean', 'median', 'most_frequent']:
                                                                imputed_X_train = SimpleImputer(missing_values=np.nan, strategy=mv_strategy[param1]).fit(X_train).transform(X_train)
                                                                updated_y_train = y_train.copy()
                                                                if metric_type=='sp':
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
                                                                outlier_y_pred = IsolationForest(n_estimators=50,contamination=0.2,random_state=0).fit_predict(scaled_X_train)
                                                                od_param = [2]
                                                                # print("Fraction of outliers: ", round((1 - sum(mask)/len(outlier_y_pred)) * 100, 4))
                                                else:
                                                        k = lof_k_lst[param3 - 2] # start accessing number of neighbors in lof
                                                        outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=lof_contamination).fit_predict(scaled_X_train)
                                                        od_param =  [param3+1]
                                                mask = outlier_y_pred != -1
 
                                                outlier_X_train = scaled_X_train.copy()
                                                outlier_y_train = updated_y_train.copy()
                                                fraction_out = round((1 - sum(mask)/len(outlier_y_pred)) * 100, 4)
                                                priv_idx_train = []
                                                unpriv_idx_train = []

                                                if metric_type=='sp':
                                                        outlier_sensitive_train = updated_sensitive_attr_train.copy()
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                outlier_X_train, outlier_y_train, outlier_sensitive_train = scaled_X_train[mask], updated_y_train[mask], updated_sensitive_attr_train[mask]
                                                                outlier_y_train.reset_index(drop=True, inplace=True)
                                                                outlier_X_train.reset_index(drop=True, inplace=True)
                                                                outlier_sensitive_train.reset_index(drop=True, inplace=True)          
                                                        priv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 1]
                                                        unpriv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 0]
                                                
                                                updated_model = None
                                                # import pdb;pdb.set_trace()
                                                sens_data = []
                                                if modelType == 'lr':
                                                        updated_model = LogisticRegression(random_state=0).fit(outlier_X_train, outlier_y_train)
                                                elif modelType == 'nb':
                                                        updated_model = GaussianNB().fit(outlier_X_train, outlier_y_train)
                                                
                                                y_pred = updated_model.predict(outlier_X_train)
                                                concat_X_y =None
                                                if metric_type== 'sp':
                                                        concat_X_y  =  pd.concat([outlier_sensitive_train, outlier_y_train], axis=1)
                                                        y_pred_priv       = len(concat_X_y[(concat_X_y[sens_attr_name] == 1) & (concat_X_y[target_variable_name] == 1)])/ len(concat_X_y[(concat_X_y[sens_attr_name] == 1)])
                                                        y_pred_unpriv = len(concat_X_y[(concat_X_y[sens_attr_name] == 0) & (concat_X_y[target_variable_name] == 1)])/ len(concat_X_y[(concat_X_y[sens_attr_name] == 0)])
                                                        diff_sensitive_attr = round(y_pred_priv - y_pred_unpriv,5)
                                                        ratio_sensitive_attr =  round(len(concat_X_y[(concat_X_y[sens_attr_name] == 1)] )/len(concat_X_y[(concat_X_y[sens_attr_name] == 0)]),5)
                                                        profile_gen =[fraction_out,diff_sensitive_attr,ratio_sensitive_attr]
                                                        
                                                        
                                                # import pdb;pdb.set_trace()
                                                outc = None
                                                if metric_type== 'sp':
                                                        outc = self.computeStatisticalParity(y_pred[priv_idx_train],y_pred[unpriv_idx_train])
                                                        
 
                                                elif metric_type=='f-1':
                                                        outc = f1_score(outlier_y_train,y_pred)
                                                elif metric_type=='mae':
                                                        outc = mean_absolute_error(outlier_y_train, y_pred)
                                                elif metric_type=='rmse':
                                                        outc = np.sqrt(mean_squared_error(outlier_y_train, y_pred)) 
                                                elif metric_type=='accuracy_score':
                                                        outc = 1-accuracy_score(outlier_y_train,y_pred)
                                                        p = Profile(concat_X_y)
                                                        profile_gen,_ = p.populate_profiles(pd.concat([outlier_X_train, outlier_y_train], axis=1),fraction_out)
                                                
                                                        

                                                       
                                                f = [outc]
                                                # param_lst.append(mv_param + norm_param + od_param + f)
                                                
                                             
                                                param_lst.append(mv_param + norm_param + od_param + profile_gen + f)
                                                # if eqOdds_train == 0.0:
                                                print(str(mv_param + norm_param + od_param + f))
                                        # print(len(unpriv_idx_train))
                        

                        # param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        param_column = ["missing_value","normalization","outlier"]
                        param_lst_df = pd.DataFrame(param_lst, columns= param_column   + key_profile +['fairness'])
                        param_lst_df.to_csv(file_name, index=False)
                else :
                        self.profiles = key_profile
                        param_lst_df = pd.read_csv(file_name)

                self.param_columns = ["missing_value","normalization","outlier"]
                self.profiles = key_profile
                
                y = param_lst_df['fairness']
                X = param_lst_df.copy()[self.profiles]
                X = StandardScaler().fit(X).transform(X)
                reg = Regression()
                model = reg.generate_regression(X, y)
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
                        X = StandardScaler().fit(X).transform(X)
                        reg = Regression()
                        model = reg.generate_regression(X, y)
                        coefs = model.coef_
                        # print(model.intercept_)
                        self.ranking_param[elem] =  np.argsort(np.abs(coefs))[::-1]
                        # print(self.ranking_param[elem])
                        self.param_coeff[elem] =  coefs
                
                if(algo_type == 'projection'):
                        parameter_coefs = [0]*3
                        for id_prof,profile_coef in enumerate(self.profile_coefs):
                               for parameter_id,parameter_coef in enumerate(self.param_coeff[self.profiles[id_prof]]):
                                      parameter_coefs[parameter_id] += profile_coef * parameter_coef
                        print(parameter_coefs)
                        self.coefs = parameter_coefs
                        self.coef_rank = np.argsort(np.abs(parameter_coefs))[::-1]
                        print(self.coef_rank)
                        

                                      

#GPU 
                
        

        def create_historic_data_test(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
                if not(os.path.exists(file_name)):
                        idx_test = np.arange(0, len(X_test), 1)
                        mv_test = pd.DataFrame(idx_test).sample(frac=0.4, replace=False, random_state=1).index
                        if (dataset == 'hmda'):
                                X_test['lien_status'][mv_test] = np.NaN
                        elif (dataset == 'adult'):
                                X_test['Martial_Status'][mv_test] = np.NaN

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
                        sens_attr_name = ''
                        target_variable_name = ''
                        param_lst = []
                        if metric_type == 'sp':
                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)
                                # test_eqOdds = self.computeEqualizedOdds(y_pred_test, y_test, priv_idx_test, unpriv_idx_test)
                                # print("testing fairness : " + str(round(test_eqOdds, 4)))
                                sens_attr_name = sensitive_attr_test.name
                                target_variable_name = y_test.name
                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)
                                # test_eqOdds = self.computeEqualizedOdds(y_pred_test, y_test, priv_idx_test, unpriv_idx_test)
                                # print("Test fairness : " + str(round(test_eqOdds, 4)))

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
                                                                if metric_type=='sp':
                                                                        updated_sensitive_attr_test = sensitive_attr_test.drop(mv_idx)
                                                                        updated_sensitive_attr_test.reset_index(drop=True, inplace=True)
                                                                mv_param = [1]
                                                        elif mv_strategy[param1] in ['mean', 'median', 'most_frequent']:
                                                                imputed_X_test = SimpleImputer(missing_values=np.nan, strategy=mv_strategy[param1]).fit(X_test).transform(X_test)
                                                                updated_y_test = y_test.copy()
                                                                if metric_type=='sp':
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
                                                                outlier_y_pred = IsolationForest(n_estimators=50,contamination=0.2,random_state=0).fit_predict(scaled_X_test)
                                                                od_param = [2]
                                                else:
                                                        k = lof_k_lst[param3 - 2] # start accessing number of neighbors in lof
                                                        outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=lof_contamination).fit_predict(scaled_X_test)
                                                        od_param = [param3+1]
                                                mask = outlier_y_pred != -1

                                                outlier_X_test = scaled_X_test.copy()
                                                outlier_y_test = updated_y_test.copy()
                                                priv_idx_test = None
                                                unpriv_idx_test = None
                                                fraction_out = round((1 - sum(mask)/len(outlier_y_pred)) * 100, 4)
                                                if(metric_type=='sp'):
                                                        outlier_sensitive_test = updated_sensitive_attr_test.copy()
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                # import pdb;pdb.set_trace()
                                                                outlier_X_test, outlier_y_test, outlier_sensitive_test = scaled_X_test[mask], updated_y_test[mask], updated_sensitive_attr_test[mask]
                                                                outlier_y_test.reset_index(drop=True, inplace=True)
                                                                outlier_X_test.reset_index(drop=True, inplace=True)
                                                                outlier_sensitive_test.reset_index(drop=True, inplace=True)   

                                                        priv_idx_test = [i for i, val in enumerate(outlier_sensitive_test) if val == 1]
                                                        unpriv_idx_test = [i for i, val in enumerate(outlier_sensitive_test) if val == 0]

                                                updated_model = None
                                                #import pdb;pdb.set_trace()
                                                if modelType == 'lr':
                                                        updated_model = LogisticRegression(random_state=0).fit(outlier_X_test, outlier_y_test)
                                                elif modelType == 'nb':
                                                        updated_model = GaussianNB().fit(outlier_X_test, outlier_y_test)

                                                y_pred = updated_model.predict(outlier_X_test)
                                                # eqOdds_test = self.computeEqualizedOdds(y_pred, outlier_y_test, priv_idx_test, unpriv_idx_test)
                                                outc = None
                                                # import pdb;pdb.set_trace()
                                                if metric_type== 'sp':
                                                        outc = self.computeStatisticalParity(y_pred[priv_idx_test],y_pred[unpriv_idx_test])
                                                elif metric_type=='f-1':
                                                        outc = f1_score(outlier_y_test,y_pred)
                                                elif metric_type=='mae':
                                                        outc = mean_absolute_error(outlier_y_test, y_pred)
                                                elif metric_type=='rmse':
                                                        outc = np.sqrt(mean_squared_error(outlier_y_test, y_pred)) 
                                                elif metric_type=='accuracy_score':
                                                        outc = 1-accuracy_score(outlier_y_test,y_pred)
                                                f = [outc]
                                                                                               
                                                # param_lst.append(mv_param + norm_param + od_param + profile_gen+ f)
                                                # except:
                                                  
                                                gen_profiles = []
                                                p = Profile(pd.concat([outlier_X_test, outlier_y_test], axis=1))
                                                profile_gen,key_profile = p.populate_profiles(pd.concat([outlier_X_test, outlier_y_test], axis=1),fraction_out)
                                                        
                                                if  metric_type== 'sp':
                                                       
                                                        length_parameter = len(mv_param + norm_param + od_param + profile_gen+ f)
                                                        # print(f'len {length_parameter}')
                                                        y_pred = updated_model.predict(outlier_X_test)
                                                        concat_X_y  =  pd.concat([outlier_sensitive_test, outlier_y_test], axis=1)
                                                        y_pred = updated_model.predict(outlier_X_test)
                                                        y_pred_priv       = len(concat_X_y[(concat_X_y[sens_attr_name] == 1) & (concat_X_y[target_variable_name] == 1)])/ len(concat_X_y[(concat_X_y[sens_attr_name] == 1)])
                                                        y_pred_unpriv = len(concat_X_y[(concat_X_y[sens_attr_name] == 0) & (concat_X_y[target_variable_name] == 1)])/ len(concat_X_y[(concat_X_y[sens_attr_name] == 0)])
                                                        diff_sensitive_attr = round(y_pred_priv - y_pred_unpriv,5)
                                                        ratio_sensitive_attr =  round(len(concat_X_y[(concat_X_y[sens_attr_name] == 1)] )/len(concat_X_y[(concat_X_y[sens_attr_name] == 0)]),5)
                                                        gen_profiles =[fraction_out,diff_sensitive_attr,ratio_sensitive_attr]
                                                elif  metric_type=='accuracy_score':
                                                       pass

                                                param_lst.append(mv_param + norm_param + od_param + f)

                                        


                        # param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        
                        # profile_headers = ['corr_race', 'ot_race', 'corr_gender', 'ot_gender', 'corr_loan_type', 'ot_loan_type', 'corr_applicant_age', 'ot_applicant_age', 'corr_lien_status', 'ot_lien_status', 'corr_LV', 'ot_LV', 'corr_DI', 'ot_DI', 'corr_income_brackets', 'ot_income_brackets']
                        param_column = ["missing_value","normalization","outlier"]

                        param_lst_df = pd.DataFrame(param_lst, columns= param_column  + ["fairness"])

                        param_lst_df.to_csv(file_name,index=False)


        def write_quartiles(self,csv_writer, algorithm, metric, quartiles):
                csv_writer.writerow([f_goal, algorithm, f"{metric} q1", round(quartiles[0], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q2", round(quartiles[1], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q3", round(quartiles[2], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q4", round(quartiles[3], 5)])
p = base()




p = base()



filename_test = 'historical_data/historical_data_test_profile_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
filename_train = 'historical_data/historical_data_train_profile_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
p.create_historic_data(filename_train)

p.create_historic_data_test(filename_test)
if(dataset =='adult'):
       
        f_goals = [0.025,0.03,0.04,0.05,0.06,0.07,0.08,0.09,.16]
elif(dataset=='hmda'):
        f_goals = [.001,0.002,0.01,0.02,0.03]
else:
        print('Please profile goals ')

# #Read from historical data gererated on training data 
historical_data = pd.read_csv(filename_test)
p.historical_data_pd = historical_data;
# #Convert to list of list containing all the combination of transformers in a fixed pipeline
p.historical_data = historical_data.values.tolist();


f = sys.stdout
metric_path  = 'metric/metric_profile_'+algo_type+"_"+modelType+'_'+metric_type+'_'+dataset+'.csv'
f = open(metric_path, 'w')

gg  = historical_data.values.tolist()
csv_writer = csv.writer(f)

for f_goal in f_goals:
        logging.info(f'Fairness goal {f_goal}')
        rank_idistr = []
        rank_fdistr = []
        gs_idistr =   []
        gs_fdistr =   []
        failures = 0
        for seed_ in gg:
                seen = set()
                p.grid_search(f_goal,1, seen)
                gs_idistr.append(p.gs_idistr[0])
                gs_fdistr.append(p.gs_fdistr[0])
                p.optimize_1step(seed_, f_goal)

                if p.rank_iter != -1:
                        rank_idistr.append(p.rank_iter)
                        rank_fdistr.append(p.rank_f)
                else:
                        failures += 1

        print(p.fail)
        print(p.pass_)
        csv_writer.writerow([f" failed {p.fail}" , f"passed{p.pass_}"])
        p.pass_   = 0 
        p.fail  = 0 
        rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75,100], interpolation='midpoint')
        rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75,100], interpolation='midpoint')
        g_iquartiles = np.percentile(gs_idistr, [25, 50, 75,100], interpolation='midpoint')
        g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75,100], interpolation='midpoint')
        print("Fairness goal stats: " + str(f_goal))




        csv_writer.writerow(["Fairness Goal", "Grid search", "Iteration", "Value"])

        # Write data for ranking algorithm
        p.write_quartiles(csv_writer, "ranking", "iterations", rank_iquartiles)
        p.write_quartiles(csv_writer, "ranking", "Fairness", rank_fquartiles)
        csv_writer.writerow([])

        # Write data for grid search algorithm
        p.write_quartiles(csv_writer, "grid search", "iterations", g_iquartiles)
        p.write_quartiles(csv_writer, "grid search", "Fairness", g_fquartiles)
        csv_writer.writerow([])
        #import pdb;pdb.set_trace()
f.close()


