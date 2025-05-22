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
                self.profile=[]


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
pipeline_order = ['missing_value', 'normalization', 'outlier', 'model']
executor = PipelineExecutor(
                pipeline_type='ml',
                dataset_name=dataset_name,
                metric_type=metric_type,
                pipeline_ord=pipeline_order
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