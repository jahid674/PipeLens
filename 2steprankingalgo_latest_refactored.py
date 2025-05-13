import itertools
# from modules.outlier_detection.outlier_detector import OutlierDetector
# from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
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
from modules.normalization.normalizer import Normalizer
from modules.missing_value.imputer import Imputer
from modules.models.model import ModelTrainer
from pipeline_execution import PipelineExecutor
from modules.models.metric import MetricEvaluator
from modules.outlier_detection.outlier_detector import OutlierDetector
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
from LoadDataset import LoadDataset
from modules.profiling.profile import Profile


knn_k_lst = [1, 5, 10, 20, 30]
lof_k_lst = [1, 5, 10, 20, 30]
len_knn = len(knn_k_lst)
len_lof = len(lof_k_lst)
norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm'] # standard scaler, robust scaler, max absolute scaler, minmax scaler
mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
od_strategy = ['none', 'if', 'lof'] # local outlier factor, isolation forest
model_selection = ['lr']#, 'rf' #, 'nb', 'reg']
dataset_name = 'adult' # 'hmda', 'housing'
modelType = 'lr' #'lr' 'reg'
metric_type = 'accuracy_score' # rmse, accuracy_score, sp
algo_type = '2step'
from sklearn.ensemble import RandomForestClassifier
logging.basicConfig(filename='logs/profile_'+algo_type+"_"+dataset_name+'_'+modelType+'_'+'metric_type'+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        model_selection = ['lr', 'rf', 'nb', 'reg']

#column_names = ['Age','Workclass','fnlwgt','Education','Education_Num','Martial_Status','Occupation','Relationship','Race','Sex','Capital_Gain','Capital_Loss','Hours_per_week','Country','income']
#categorical_cols = ['Age','Workclass', 'Education', 'Martial_Status', 'Relationship', 'Race', 'Sex','Hours_per_week','income']
#from scipy.stats import pearsonr

loader = LoadDataset(dataset_name)
dataset, X_train, y_train, X_test, y_test = loader.load()

if dataset_name == 'adult':
        tau_train = 0.1 # fraction of missing values
        tau_test = 0.1
        contamination_train = 0.2
        contamination_test = 0.2
        contamination_train_lof = 'auto'
        contamination_test_lof = 'auto'
        target = 'income'
        sensitive_variable='Sex'
        numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns
        categorical_columns = X_train.select_dtypes(include=['object']).columns
elif dataset_name == 'hmda':
        tau_train = 0.2 # fraction of missing values
        tau_test = 0.1
        contamination_train = 0.3
        contamination_test = 0.2
        contamination_train_lof = 0.3
        contamination_test_lof = 0.2
        target = 'action_taken'
        sensitive_variable='race'
        numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns
        categorical_columns = X_train.select_dtypes(include=['object']).columns
elif dataset_name == 'housing':
        tau_train = 0.2 # fraction of missing values
        tau_test = 0.1
        contamination_train = 0.3
        contamination_test = 0.2
        contamination_train_lof = 0.3
        contamination_test_lof = 0.2
        target = 'SalePrice'
        numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns
        categorical_columns = X_train.select_dtypes(include=['object']).columns

profiler = Profile()
#outlier_fraction = profiler.get_fraction_of_outlier(dataset[numerical_columns])
'''results, keys = profiler.populate_profiles(
    data_final=dataset,
    numerical_columns=numerical_columns,
    target_column=target,
    outlier=outlier_fraction,
    metric_type='classification'
)'''


        
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

                self.model_selection = ['lr', 'rf', 'nb', 'reg']
                self.model_name_mapping = {'lr': 'model_lr', 'rf': 'model_rf', 'nb': 'model_nb', 'reg': 'model_reg'}
                #normalization 
                #

                self.ranges['missing_value'] = [1,2,3,4]
                self.ranges['normalization'] = [1, 5, 10, 20, 30]
                self.ranges['outlier'] = [1, 5, 10, 20, 30]

                self.base_strategies  =['missing_value', 'normalization', 'outlier', 'model']
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
                logging.info(f'2step Starting Grasp search for utility score:{f_goal}')
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

                self.ranges['missing_value'] = list(np.unique(self.historical_data_pd['missing_value']))
                self.ranges['normalization'] = list(np.unique(self.historical_data_pd['normalization']))
                self.ranges['outlier'] = list(np.unique(self.historical_data_pd['outlier']))
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
                                        
                                        logging.info(f'first loop iter_size = {iter_size}')
                                        logging.info(f'first loop profile = {self.profiles[profile_index]}')

                                        param_name  = self.profiles[profile_index]
                                        coef_rank = self.ranking_param[param_name]
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
                                                logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                                                
                                                #profile_opt = self.get_profile(self.historical_data_pd,list(cur_params.values()),profile_index)
                                                
                                                #profile_cur = self.get_profile(self.historical_data_pd,list(cur_params.values()),profile_index)
                                                
                                                if(tuple(cur_params.items())) in seen:
                                                        continue
                                                seen.add(tuple(cur_params.items()))
                                                
                                                # logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                                                cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))

                                                # self.profile_dist[param_name] +=1
                                                
                                                #if(profile_cur<profile_opt):
                                                self.rank_iter += 1
                                                logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                                                logging.warn(f'Current iteration after parameter selection {self.rank_iter}')
                                                logging.info(f'updated utility after parameter selection {cur_f}')
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
                                        
                                        logging.info(f'second loop iter_size = {iter_size}')
                                        logging.info(f'profile = {profile}')
                                        
                                        #Co-ef comparison should  not be on true value 
                                        #missing_value,normalization,outlier,fairness
                                        map = {}
                                        lst = []
                                        for param in self.historical_data:
                                                map[param[0]*self.param_coeff[profile][0]
                                                +param[1]*self.param_coeff[profile][1]
                                                +param[2]*self.param_coeff[profile][2]] = param
                                                
                                                
                                                lst.append((param[0]*self.param_coeff[profile][0]
                                                +param[1]*self.param_coeff[profile][1]
                                                +param[2]*self.param_coeff[profile][2],param))

                                        sorted_params = sorted(map.items(), key=operator.itemgetter(0))
                                        sorted_params_lst = sorted(lst, key=lambda x: x[0])
                                        
                                        if(coeff):
                                                sorted_params.reverse()
                                                sorted_params_lst.reverse()
                                        # print(sorted_params[iter_size-1])
                                        cur_params = cur_params_opt.copy()
                                        for id,val in enumerate(self.base_strategies):
                                        #        print(iter_size)
                                                #try:
                                                cur_params[val] = sorted_params_lst[iter_size-1][1][id]

                                               
                                        if(tuple(cur_params.items())) in seen:
                                               continue
                                        seen.add(tuple(cur_params.items()))
                                        cur_f = sorted_params_lst[iter_size-1][1][-1]
                                        self.rank_iter += 1
                                        logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                                        logging.warn(f'Current iteration after parameter selection {self.rank_iter}')
                                        logging.info(f'updated utility after parameter selection {cur_f}')
                                                        
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
                param_lst_df = None
                key_profile = []
                p = Profile()
                
                # if True
                if not(os.path.exists(file_name)):
                        idx_train = np.arange(0, len(X_train), 1)
                        mv_train = pd.DataFrame(idx_train).sample(frac=tau_train, replace=False, random_state=1).index
                        # inject missing values in the most important column
                        if (dataset == 'hmda'):
                                X_train['lien_status'][mv_train] = np.NaN
                        elif (dataset == 'adult'):
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
                        sens_attr_name = ''
                        target_variable_name = ''
                        if(metric_type=='sp' or metric_type=='accuracy_score'):
                                priv_idx_train, unpriv_idx_train, sensitive_attr_train = self.getIdxSensitive(X_train, dataset)
                                sens_attr_name = sensitive_attr_train.name
                                target_variable_name = y_train.name
                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)

                        for param1 in range(mv_params):
                                for param2 in range(norm_params):
                                        for param3 in range(od_params):
                                                #out_before_imp_strat = p.get_fraction_of_outlier(X_train)
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
                                                        k = knn_k_lst[param1 - len(mv_strategy) + 1] # start accessing number of neighbors in knn
                                                        imputed_X_train = KNNImputer(n_neighbors=k).fit_transform(X_train)
                                                        updated_y_train = y_train.copy()
                                                        if dataset in ['adult']:
                                                                updated_sensitive_attr_train = sensitive_attr_train.copy()
                                                        mv_param = [param1 + 1]
                                
                                                out_before_norm_strat = p.get_fraction_of_outlier(imputed_X_train)
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
                                                out_before_out_strat = p.get_fraction_of_outlier(scaled_X_train)
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
                                                        k = lof_k_lst[param3 - len(od_strategy) + 1] # start accessing number of neighbors in lof
                                                        outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=contamination_train_lof).fit_predict(scaled_X_train)
                                                        od_param =  [param3 + 1]
                                                mask = outlier_y_pred != -1
 
                                                outlier_X_train = scaled_X_train.copy()
                                                outlier_y_train = updated_y_train.copy()
                                                fraction_out = round((1 - sum(mask)/len(outlier_y_pred)) * 100, 4)
                                                priv_idx_train = []
                                                unpriv_idx_train = []

                                                if metric_type=='sp' or metric_type=='accuracy_score':
                                                        outlier_sensitive_train = updated_sensitive_attr_train.copy()
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                # print(len(scaled_X_train), len(updated_y_train), len(updated_sensitive_attr_train))
                                                                outlier_X_train, outlier_y_train, outlier_sensitive_train = scaled_X_train[mask], updated_y_train[mask], updated_sensitive_attr_train[mask]
                                                                outlier_y_train.reset_index(drop=True, inplace=True)
                                                                outlier_X_train.reset_index(drop=True, inplace=True)
                                                                outlier_sensitive_train.reset_index(drop=True, inplace=True)          
                                                        priv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 1]
                                                        unpriv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 0]
                                                else:
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                outlier_X_train, outlier_y_train = scaled_X_train[mask], updated_y_train[mask]
                                                                outlier_y_train.reset_index(drop=True, inplace=True)
                                                                outlier_X_train.reset_index(drop=True, inplace=True)
                                                                
                                                       
                                                updated_model = None
                                                
                                                if modelType == 'lr':
                                                        updated_model = LogisticRegression(random_state=0).fit(outlier_X_train, outlier_y_train)
                                                elif modelType == 'nb':
                                                        updated_model = GaussianNB().fit(outlier_X_train, outlier_y_train)
                                                elif modelType == 'rf':
                                                       updated_model = RandomForestClassifier(random_state=0).fit(outlier_X_train, outlier_y_train)
                                                elif modelType=='reg':
                                                        updated_model = Regression().generate_regression(outlier_X_train,outlier_y_train)
                                                
                                                
                                                y_pred = updated_model.predict(outlier_X_train)
                                                class_imbalance_ratio =  None
                                                if(metric_type=='sp' or metric_type=='accuracy_score'):
                                                        concat_X_y  =  pd.concat([outlier_sensitive_train, outlier_y_train], axis=1)
                                                        
                                                        y_pred_priv = len(concat_X_y[(concat_X_y[sens_attr_name] == 1) & (concat_X_y[target_variable_name] == 1)])/ len(concat_X_y[(concat_X_y[sens_attr_name] == 1)])
                                                        y_pred_unpriv = len(concat_X_y[(concat_X_y[sens_attr_name] == 0) & (concat_X_y[target_variable_name] == 1)])/ len(concat_X_y[(concat_X_y[sens_attr_name] == 0)])
                                                        diff_sensitive_attr = round(y_pred_priv - y_pred_unpriv,5)
                                                        ratio_sensitive_attr =  round(len(concat_X_y[(concat_X_y[sens_attr_name] == 1)] )/len(concat_X_y[(concat_X_y[sens_attr_name] == 0)]),5)
                                                        cov = concat_X_y[sens_attr_name].cov(concat_X_y[outlier_y_train.name])
                                                        class_imbalance_ratio = round((outlier_y_train == 1).sum()/len(y_train),5)
                                                        sens_data =[out_before_out_strat,out_before_norm_strat,diff_sensitive_attr,ratio_sensitive_attr,cov,class_imbalance_ratio]
                                                        if(metric_type=='accuracy_score'):
                                                               sens_data =[out_before_out_strat,out_before_norm_strat,class_imbalance_ratio]
                                                else:
                                                        profile_median = outlier_y_train.median()
                                                        sens_data =[out_before_out_strat,out_before_norm_strat,profile_median]
                                                       

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
                                                       outc = 1-accuracy_score(outlier_y_train, y_pred)
                                                
                                                f = [outc]
                                                profile_gen,key_profile = p.populate_profiles(pd.concat([outlier_X_train, outlier_y_train], axis=1),fraction_out)
                                                        
                                                param_lst.append(mv_param + norm_param + od_param + sens_data + profile_gen + f)
                                                print(str(mv_param + norm_param + od_param + f))

                        param_column = ["missing_value","normalization","outlier"]
                        if(metric_type=='sp'):
                                param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['out_before_out_strat','out_before_norm_strat',"diff_sensitive_attr","ratio_sensitive_attr","cov","class_imbalance_ratio"]+key_profile+["fairness"])
                        elif(metric_type=='accuracy_score'):
                               param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['out_before_out_strat','out_before_norm_strat','class_imbalance_ratio']+key_profile+["fairness"])
                        else:
                               param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['out_before_out_strat','out_before_norm_strat','profile_median'] + key_profile + ["fairness"])
                               
                        param_lst_df.to_csv(file_name, index=False)
                else :
                        param_lst_df = pd.read_csv(file_name)
                        
                self.param_columns = ["missing_value","normalization","outlier"]
                if(dataset=='hmda'):
                        key_profile = ['class_imbalance_ratio']
                elif dataset=='housing':
                       key_profile = ['profile_median']
                elif dataset=='adult':
                       key_profile = ["diff_sensitive_attr","ratio_sensitive_attr","class_imbalance_ratio"]
                for column in X_train.columns:
                        key_profile.append('corr_' + column)
                self.profiles = key_profile + ['out_before_out_strat','out_before_norm_strat']
                # self.profiles = key_profile
                if h_sample_bool and dataset == 'adult':
                        self.profiles = ['diff_sensitive_attr', 'class_imbalance_ratio','ratio_sensitive_attr','corr_Country','corr_Workclass']
                
                #rank profile first
                if(dataset=='housing'):
                        y = param_lst_df['fairness']
                        
                        t = StandardScaler().fit(param_lst_df).transform(param_lst_df)
                        # t = param_lst_df
                        X = pd.DataFrame(data = t, columns = param_lst_df.columns)[self.profiles]
                else :
                       y = param_lst_df['fairness']
                       X = param_lst_df.copy()[self.profiles]       
                
                if (h_sample_bool):
                        print(" ------ Sampling --------", h_sample)
                        random.seed(1000)
                        sample_idx = random.sample(list(range(len(X))), math.ceil(h_sample * len(X)))
                        X = X.iloc[sample_idx]
                        y = y.iloc[sample_idx]
                
                reg = Regression()
                model = reg.generate_regression(X, y)
                coefs = model.coef_
                print(coefs)
                print(model.intercept_)
                
                self.profile_ranking = np.argsort(np.abs(coefs))[::-1]
                self.profile_coefs = coefs
                print(self.profile_coefs)
                
                #ranking parameter 
                self.ranking_param ={}
                self.param_coeff  = {}
                for index, elem in enumerate(self.profiles):
                        y = param_lst_df[elem]
                        X = param_lst_df.copy()[self.param_columns]
                        if h_sample_bool:
                                X = X.iloc[sample_idx]
                                y = y.iloc[sample_idx]
                        
                        # X = StandardScaler().fit(X).transform(X)
                        reg = Regression()
                        model = reg.generate_regression(X, y)
                        coefs_par = model.coef_
                        # print(model.intercept_)
                        self.ranking_param[elem] =  np.argsort(np.abs(coefs_par))[::-1]
                        print(self.ranking_param[elem])
                        
                        self.param_coeff[elem] =  coefs_par
                        print(f'name : {elem} {self.param_coeff[elem]}')

                for idx,profile_index in enumerate(self.profile_ranking):
                       print(self.profiles[profile_index])
                print('33')
                
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
                        if metric_type == 'sp' or metric_type=='accuracy_score':
                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)
                                sens_attr_name = sensitive_attr_test.name
                                target_variable_name = y_test.name
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
                                                                if metric_type=='sp' or  metric_type=='accuracy_score':
                                                                        updated_sensitive_attr_test = sensitive_attr_test.copy()
                                                                if mv_strategy[param1] == 'mean':
                                                                        mv_param = [2]
                                                                if mv_strategy[param1] == 'median':
                                                                        mv_param = [3]
                                                                if mv_strategy[param1] == 'most_frequent':
                                                                        mv_param = [4]
                                                                        # import pdb;pdb.set_trace()
                                                else:
                                                        k = knn_k_lst[param1 - len(mv_strategy) + 1] # start accessing number of neighbors in knn
                                                        imputed_X_test = KNNImputer(n_neighbors=k).fit_transform(X_test)
                                                        updated_y_test = y_test.copy()
                                                        if dataset in ['adult']:
                                                                updated_sensitive_attr_test = sensitive_attr_test.copy()
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
                                                        k = lof_k_lst[param3 - len(od_strategy) + 1] # start accessing number of neighbors in lof
                                                        outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=contamination_test_lof).fit_predict(scaled_X_test)
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
                                                else:
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
                                                elif metric_type=='mae':
                                                        outc = mean_absolute_error(outlier_y_test, y_pred)
                                                elif metric_type=='rmse':
                                                        outc = np.sqrt(root_mean_squared_error(outlier_y_test, y_pred)) 
                                                elif metric_type=='accuracy_score':
                                                       outc = 1-accuracy_score(outlier_y_test, y_pred)
                                                f = [outc]
                                                                                               
                                                # param_lst.append(mv_param + norm_param + od_param + profile_gen+ f)
                                                # except:
                                                
                                                length_parameter = len(mv_param + norm_param + od_param + f)
                                                # print(f'len {length_parameter}')
                                                y_pred = updated_model.predict(outlier_X_test)
                                                
                                                param_lst.append(mv_param + norm_param + od_param + f)
                                                
                                                # print(str(mv_param + norm_param + od_param + f))
                                        


                        # param_lst_df = pd.DataFrame(param_lst, columns=["missing_value","normalization","outlier","fairness"])
                        
                        # profile_headers = ['corr_race', 'ot_race', 'corr_gender', 'ot_gender', 'corr_loan_type', 'ot_loan_type', 'corr_applicant_age', 'ot_applicant_age', 'corr_lien_status', 'ot_lien_status', 'corr_LV', 'ot_LV', 'corr_DI', 'ot_DI', 'corr_income_brackets', 'ot_income_brackets']
                        param_column = ["missing_value","normalization","outlier"]

                        param_lst_df = pd.DataFrame(param_lst, columns= param_column  + ["fairness"])

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

filename_test = 'historical_data/historical_data_test_profile_'+modelType+'_'+metric_type+'_'+dataset_name+'.csv'
filename_train = 'historical_data/historical_data_train_profile_'+modelType+'_'+metric_type+'_'+dataset_name+'.csv'

if scalability_bool:
        filename_test = 'metric/scalability/historical_data_test_profile_'+modelType+'_'+metric_type+'_'+dataset_name+'_'+str(len(knn_k_lst))+'.csv'
        filename_train = 'metric/scalability/historical_data_train_profile_'+modelType+'_'+metric_type+'_'+dataset_name+'_'+str(len(knn_k_lst))+'.csv'

executor = PipelineExecutor(
    dataset_name=dataset_name,
    metric_type=metric_type,
    mv_strategy=mv_strategy,
    norm_strategy=norm_strategy,
    od_strategy=od_strategy,
    model_selection=model_selection,
    knn_k_lst=knn_k_lst,
    lof_k_lst=lof_k_lst,
    tau_train=tau_train,
    contamination_train=contamination_train,
    contamination_train_lof=contamination_train_lof
)
_, _, sensitive_attr_train = executor.getIdxSensitive(X_train, sensitive_variable)
coefs, _, coefs_par, _ = executor.run_pipeline_algo2(filename_train, X_train, y_train)
_,_,_,_ = executor.run_pipeline_algo2(filename_train, X_train, y_train, pipeline_order=['missing_value', 'normalization', 'outlier', 'model'])
historical_data = pd.read_csv(filename_train)

f_goals = []
if(dataset =='adult'):
       f_goals = [0.1, 0.15, 0.2, 0.32]
       f_goals = [0.05, 0.06, 0.13, 0.16]
       f_goals = [0.045]
elif(dataset=='hmda'):
       f_goals  = [.06, .07, .08, .09]
#        f_goals  = [.06] #scalability
elif(dataset=='housing'):
#        f_goals = [165,170,175,180,185]
       f_goals = [162, 170, 180, 185]
#        f_goals = [152]
else:
        print('Please profile goals ')

# #Read from historical data gererated on training data 
p.historical_data_pd = historical_data;
# #Convert to list of list containing all the combination of transformers in a fixed pipeline
p.historical_data = historical_data.values.tolist();

if (h_sample_bool):
#        metric_path  = 'metric/ablation/historical_'+algo_type+"_"+modelType+'_'+metric_type+'_'+dataset+'_'+str(h_sample)+'.csv'
       metric_path  = 'metric/ablation/historical_'+algo_type+"_"+modelType+'_'+metric_type+'_'+dataset+'.csv'
elif (scalability_bool):
        # metric_path  = 'metric/scalability/historical_profile_'+modelType+'_'+metric_type+'_'+dataset+'_'+str(len(knn_k_lst))+'.csv'
        metric_path  = 'metric/scalability/historical_'+algo_type+"_"+modelType+'_'+metric_type+'_'+dataset+'.csv'
else:
       metric_path  = 'metric/metric_profile_'+algo_type+"_"+modelType+'_'+metric_type+'_'+dataset+'.csv'

gg  = historical_data.values.tolist()
f = sys.stdout
f_mode = 'w'
if h_sample_bool or scalability_bool:
       f_mode = 'a'
f = open(metric_path, f_mode)

csv_writer = csv.writer(f)

for f_goal in f_goals:
        logging.info(f'Utility goal {f_goal}')
        rank_idistr = []
        rank_fdistr = []
        gs_idistr =   []
        gs_fdistr =   []
        profile_itr = {}
        profile_itr['profile_outlier']= 0
        profile_itr['diff_sensitive_attr'] = 0
        profile_itr['ratio_sensitive_attr'] = 0
        
        failures = 0
        # gg = [[1.0, 1.0, 7.0, 0.022510349]]
        for seed_ in gg:
                seen = set()
                p.grid_search(f_goal, 1,seen)
                # print(p.gs_idistr)
                # print(p.gs_idistr)
                gs_idistr.append(p.gs_idistr[0])
                gs_fdistr.append(p.gs_fdistr[0])
 #               if(algo_type=='projection'):
                p.optimize(seed_, f_goal)

                if (h_sample_bool):
                        csv_writer.writerow([seed_[0], seed_[1], seed_[2], h_sample, 'profile', p.rank_iter])
                elif (scalability_bool):
                        csv_writer.writerow([seed_[0], seed_[1], seed_[2], len(gg), 'profile', p.rank_iter])
                
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
        # csv_writer.writerow([f" failed {p.fail}" , f"passed{p.pass_}",f"Grid search fall back {p.fail_grid_search}"])
        p.pass_   = 0 
        p.fail  = 0 
        p.fail_grid_search = 0
        # import pdb;pdb.set_trace()
        rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75,100], interpolation='midpoint')
        rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75,100], interpolation='midpoint')
        g_iquartiles = np.percentile(gs_idistr, [25, 50, 75,100], interpolation='midpoint')
        g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75,100], interpolation='midpoint')

        print("Utility goal stats: " + str(round(1 - f_goal, 2)))

        if not (h_sample_bool or scalability_bool):
                # Write header
                # csv_writer.writerow(["Fairness Goal", "Grid search", "Iteration", "Value"])

                # Write data for ranking algorithm
                p.write_quartiles(csv_writer, "ranking", "iterations", rank_iquartiles)
                p.write_quartiles(csv_writer, "ranking", "utility", rank_fquartiles)
                csv_writer.writerow([])

                # Write data for grid search algorithm
                p.write_quartiles(csv_writer, "grid search", "iterations", g_iquartiles)
                p.write_quartiles(csv_writer, "grid search", "utility", g_fquartiles)
                csv_writer.writerow([])
f.close()