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


knn_k_lst = [1, 5, 10, 20, 30]
lof_k_lst = [1, 5, 10, 20, 30]
len_knn = len(knn_k_lst)
len_lof = len(lof_k_lst)
norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm'] # standard scaler, robust scaler, max absolute scaler, minmax scaler
mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
od_strategy = ['none', 'if', 'lof'] # local outlier factor, isolation forest
dataset_name = 'adult' # 'hmda', 'housing'
modelType = 'lr' # rf, 'lr' # 'nb' Logistic Regression or Gaussian Naive Bayes,reg - regression
metric_type = 'sp' #sp, accuracy_score, mae, rmse, f-1
logging.basicConfig(filename='logs/'+dataset_name+'_modelType_'+'metric_type'+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
model_selection = ['lr']#, 'rf' #, 'nb', 'reg']

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


loader = LoadDataset(dataset_name)
dataset, X_train, y_train, X_test, y_test = loader.load()

if dataset_name == 'adult':
        tau_train = 0.1 # fraction of missing values
        tau_test = 0.1
        contamination_train = 0.2
        contamination_test = 0.2
        contamination_train_lof = 'auto'
        contamination_test_lof = 'auto'
elif dataset_name == 'hmda':
        tau_train = 0.05 # fraction of missing values
        tau_test = 0.1
        contamination_train = 0.1
        contamination_test = 0.2
        contamination_train_lof = 0.1
        contamination_test_lof = 0.2
elif dataset_name == 'housing':
        tau_train = 0.2 # fraction of missing values
        tau_test = 0.1
        contamination_train = 0.3
        contamination_test = 0.2
        contamination_train_lof = 0.3
        contamination_test_lof = 0.2

if dataset_name == 'adult':
        sensitive_variable='Sex'
elif dataset_name == 'hmda':
        sensitive_variable='race'


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

                self.outlier_strategies = ['none', 'if', 'lof']
                self.ot_name_mapping = {'none': 'od_none', 'if': 'od_if', 'lof': 'od_lof'}

                self.normalizer_strategies = ['none', 'ss', 'rs', 'ma', 'mm']
                self.no_name_mapping = {'none': 'norm_none', 'ss': 'norm_ss', 'rs': 'norm_rs', 'ma': 'norm_ma', 'mm': 'norm_mm'}

                self.model_selection = ['lr', 'rf', 'nb', 'reg']
                self.model_name_mapping = {'lr': 'model_lr', 'rf': 'model_rf', 'nb': 'model_nb', 'reg': 'model_reg'}

                self.ranges['missing_value'] = [1,2,3,4]
                self.ranges['normalization'] = [1, 5, 10, 20, 30]
                self.ranges['outlier'] = [1, 5, 10, 20, 30]
                self.ranges['model'] = [1, 2, 3, 4]

                self.base_strategies  =['missing_value', 'normalization', 'outlier', 'model']
                self.historical_data = []   
                self.historical_data_pd = []
                self.gs_idistr = []
                self.gs_fdistr = []

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
                self.ranges['model'] = list(np.unique(self.historical_data_pd['model']))
                seen = set()
                
                logging.info(f'current param {cur_params}')

                iter_size  = 0
                if(opt_f<f_goal):
                        self.rank_iter = 1  #Iteration count by ranking algorithm
                        self.rank_f = opt_f
                        return 
                seen.add(tuple(cur_params_opt.items()))
                while(iter_size<len(coef_rank)):
                
                        if iter_size == 0:
                                
                                for val in coef_rank:        
                                        cur_strategy = self.base_strategies[val]
                                        
                                        if(coefs[val]<0):
                                                current_paramter_value =  self.ranges[cur_strategy][-1]
                                        else:
                                                current_paramter_value =  self.ranges[cur_strategy][0]

                                        cur_params = cur_params_opt.copy()
                                        cur_params[cur_strategy] = current_paramter_value

                                        logging.info(f'Next param {cur_params}')
                                        if(tuple(cur_params.items())) in seen:
                                                continue
                                        
                                        cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))
                                        #cur_f = executor.current_par_lookup(X_test, y_test,
                                                                #pipeline_order=['missing_value', 'normalization', 'outlier', 'model'],
                                                                #cur_par=[cur_params['missing_value'], cur_params['normalization'], cur_params['outlier'], cur_params['model']])
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

                                for comb in (itertools.combinations(coef_rank, comb_size)):
                                        comb_lst.append(comb)
                                
                                for comb in comb_lst:
                                        i=0
                                        coef_lst=[]
                                        score = {}
                                        while i<comb_size:
                                                coef_lst.append(coefs[comb[i]])
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
                                                #cur_f = executor.current_par_lookup(X_test, y_test,
                                                 #               pipeline_order=['missing_value', 'normalization', 'outlier', 'model'],
                                                 #               cur_par=[cur_params['missing_value'], cur_params['normalization'], cur_params['outlier'], cur_params['model']])

                                                opt_f,cur_params_opt,found = self.f_lookup(cur_f,f_goal,cur_params_opt,cur_params,opt_f)
                                                logging.info(f'Optimal paramater {cur_params_opt}, optimal fairness {opt_f} ')

                                                if(found):
                                                        return cur_params
                        iter_size = iter_size + 1
                        if iter_size == len(coef_rank)+1:
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
                column_names =['missing_value', 'normalization', 'outlier', 'model', 'fairness']
                try:
                        return round(profiles_df.loc[(profiles_df[column_names[0]] == elem[0]) & (profiles_df[column_names[1]] == elem[1] ) 
                                               & (profiles_df[column_names[2]] == elem[2]) & (profiles_df[column_names[3]] == elem[3])].iloc[0]['fairness'],5)
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



        '''def pipeline_execution_test(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
                if not(os.path.exists(file_name)):
                        idx_test = np.arange(0, len(X_test), 1)
                        mv_test = pd.DataFrame(idx_test).sample(frac=tau_test, replace=False, random_state=1).index
                        if (dataset_name == 'hmda'):
                                X_test['lien_status'][mv_test] = np.NaN
                        elif (dataset_name == 'adult'):
                                X_test['Martial_Status'][mv_test] = np.NaN
                        elif(dataset_name=='housing'):
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

                        param_lst = []
                        if metric_type == 'sp' or metric_type=='accuracy_score' :
                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, sensitive_variable)


                        for param1 in range(mv_params):
                                for param2 in range(norm_params):
                                        for param3 in range(od_params):
                                                if param1 < len(mv_strategy) - 1:
                                                        if mv_strategy[param1] == 'drop':
                                                                if dataset_name == 'hmda':
                                                                        mv_idx = X_test[X_test['lien_status'].isna()].index.tolist()
                                                                elif dataset_name == 'adult':
                                                                        mv_idx = X_test[X_test['Martial_Status'].isna()].index.tolist()
                                                                elif dataset_name == 'housing':
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
                                                elif dataset_name=='housing' or metric_type=='accuracy_score':
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
                        
                        param_lst_df.to_csv(file_name,index=False)'''

        def write_quartiles(self,csv_writer, algorithm, metric, quartiles):
                if dataset_name in ['adult', 'hmda']:
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


filename_test = 'historical_data/historical_data_test_'+modelType+'_'+metric_type+'_'+dataset_name+'.csv'
filename_train = 'historical_data/historical_data_train_'+modelType+'_'+metric_type+'_'+dataset_name+'.csv'

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
pipeline_df, coefs, coef_rank = executor.run_pipeline(filename_train, X_train, y_train, pipeline_order=['missing_value', 'normalization', 'outlier', 'model'])
historical_test,_,_ = executor.run_pipeline(filename_test, X_test, y_test, pipeline_order=['missing_value', 'normalization', 'outlier', 'model'])
historical_data = pd.read_csv(filename_test)

if scalability_bool:
        filename_test = 'metric/scalability/historical_data_test_'+modelType+'_'+metric_type+'_'+dataset_name+'_'+str(len(knn_k_lst))+'.csv'
        filename_train = 'metric/scalability/historical_data_train_'+modelType+'_'+metric_type+'_'+dataset_name+'_'+str(len(knn_k_lst))+'.csv'

#p.pipeline_execution(filename_train)
#p.pipeline_execution_test(filename_test)

if(dataset_name =='adult'):
        f_goals = [0.1, 0.15, 0.2, 0.32]
        f_goals = [0.05, 0.06, 0.13, 0.16]
elif(dataset_name=='hmda'):
        f_goals  = [.06, .07, 0.08, 0.09] #original data
        # f_goals  = [.06] #scalability
elif(dataset_name=='housing'):
        f_goals = [162, 170, 180, 185]
else:
        print('Please profile goals ')


p.historical_data_pd = historical_data;
# #Convert to list of list containing all the combination of transformers in a fixed pipeline
p.historical_data = historical_data.values.tolist();

if (h_sample_bool):
        metric_path  = 'metric/ablation/historical_1step_'+modelType+'_'+metric_type+'_'+dataset_name+'.csv'
elif (scalability_bool):
        # metric_path  = 'metric/scalability/historical_1step_'+modelType+'_'+metric_type+'_'+dataset+'_'+str(len(knn_k_lst))+'.csv'
        metric_path  = 'metric/scalability/historical_1step_'+modelType+'_'+metric_type+'_'+dataset_name+'.csv'
else:
      metric_path  = 'metric/metric_'+modelType+'_'+metric_type+'_'+dataset_name+'.csv'

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
                #print(seed_)
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


