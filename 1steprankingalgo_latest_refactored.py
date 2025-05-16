import csv
import pandas as pd
import numpy as np
import sys
from pipeline_execution import PipelineExecutor
import pandas as pd
import numpy as np
import csv 
import logging
import numpy as np
from LoadDataset import LoadDataset
from score_lookup import ScoreLookup
from gridsearch import GridSearch

dataset_name = 'adult' # 'hmda', 'housing'
modelType = 'lr' # rf, 'lr' # 'nb' Logistic Regression or Gaussian Naive Bayes,reg - regression
metric_type = 'sp' #sp, accuracy_score, mae, rmse, f-1
pipeline_type= 'ml' # ml or em
logging.basicConfig(filename='logs/'+dataset_name+'_modelType_'+'metric_type'+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

'''loader = LoadDataset(dataset_name)
dataset, X_train, y_train, X_test, y_test = loader.load()
sensitive_variable = loader.get_sensitive_variable()'''

filename_train = f'historical_data/historical_data_train_{modelType}_{metric_type}_{dataset_name}.csv'
filename_test = f'historical_data/historical_data_test_{modelType}_{metric_type}_{dataset_name}.csv'
pipeline_order= ['missing_value', 'normalization', 'outlier', 'model']


class base:
        def __init__(self):
                self.f = 'tdst';
                self.ranking = None
                self.ranges = {}
                self.fail = 0
                self.pass_ = 0
                self.fail_with_fallback  = 0

                self.base_strategies=pipeline_order
                self.historical_data_pd = pd.read_csv(filename_test)
                self.historical_data = self.historical_data_pd.values.tolist(); 
                self.gs_idistr = []
                self.gs_fdistr = []

                executor_pass=PipelineExecutor(pipeline_type='ml', 
                          dataset_name=dataset_name, 
                          metric_type=metric_type, 
                          pipeline_ord=pipeline_order)
                pasing_hist_data = pd.read_csv(filename_train)
                self.coefs, self.coef_rank = executor_pass.score_parameter(pasing_hist_data)

                self.score_lookup = ScoreLookup(pipeline_order, metric_type)



        def optimize(self, init_params, f_goal):
                
                self.rank_iter = 0  #Iteration count by ranking algorithm
                self.rank_f = 0    # Iteration count whn we found the fairness score less than the calculated fairness from seed value 
                iter_size = 0   # Total iteration allowed in ranking algorithm , falllback is after 1 to grid search 
                cur_params = init_params.copy()
                logging.info(f'Initial seed {init_params}')
                cur_params_opt= {strategy: selection for strategy, selection in zip(self.base_strategies, init_params[:len(self.base_strategies)])}

                opt_f  = self.score_lookup.f_score_look_up2(self.historical_data_pd,init_params)

                if pipeline_type == 'ml':
                        self.ranges['missing_value'] = list(np.unique(self.historical_data_pd['missing_value']))
                        self.ranges['normalization'] = list(np.unique(self.historical_data_pd['normalization']))
                        self.ranges['outlier'] = list(np.unique(self.historical_data_pd['outlier']))
                        self.ranges['model'] = list(np.unique(self.historical_data_pd['model']))
                elif pipeline_type == 'em':
                        print('EM pipeline')
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
                                        
                                        cur_f  = self.score_lookup.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))
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

                                comb_lst= self.score_lookup.identify_param(self.coef_rank, comb_size)
                                
                                for comb in comb_lst:
                                        sorted_params=self.score_lookup.score_values(self.historical_data, self.coefs, comb_size, comb)
                                        for (elem,score) in sorted_params:#sorted_params:#(elem,score)
                                                cur_params = cur_params_opt.copy()
                                                for j in range(comb_size):
                                                        cur_strategy = self.base_strategies[comb[j]]
                                                        cur_params[cur_strategy] = round(elem[comb[j]], 5)
                                                
                                                if(tuple(cur_params.items())) in seen:
                                                        continue
                                                seen.add(tuple(cur_params.items()))
                                                self.rank_iter += 1  #Iteration count by ranking algorithm
                                                self.rank_f = opt_f
                                                
                                                logging.info(f'Next param {cur_params}')
                                                cur_f  = self.score_lookup.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))
                                                #cur_f = executor.current_par_lookup(X_test, y_test,
                                                 #               pipeline_order=['missing_value', 'normalization', 'outlier', 'model'],
                                                 #               cur_par=[cur_params['missing_value'], cur_params['normalization'], cur_params['outlier'], cur_params['model']])

                                                opt_f,cur_params_opt,found = self.f_lookup(cur_f,f_goal,cur_params_opt,cur_params,opt_f)
                                                logging.info(f'Optimal paramater {cur_params_opt}, optimal fairness {opt_f} ')

                                                if(found):
                                                        return cur_params
                        iter_size = iter_size + 1
                        if iter_size == len(self.coef_rank)+1:
                                print('failed to find the optimal pipeline')
                                break
                return sorted_params

        def f_lookup(self,cur_f,f_goal,cur_params_opt,cur_params,opt_f):
                found = False
                if pipeline_type == 'ml':
                        if metric_type!='f-1':                                                                           
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

historical_data = pd.read_csv(filename_test)
h_sample_bool=False
h_sample = 0.05
scalability_bool=False
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


#p.historical_data_pd = historical_data;
# #Convert to list of list containing all the combination of transformers in a fixed pipeline
#p.historical_data = historical_data.values.tolist();

grid=GridSearch(historical_data, pipeline_order, metric_type)

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
                gs_iter, gs_f = grid.grid_search(f_goal, seen)
                gs_idistr.append(gs_iter)
                gs_fdistr.append(gs_f)

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
        g_iquartiles = np.percentile(gs_idistr, [25, 50, 75,100], interpolation='midpoint')
        g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75,100], interpolation='midpoint')
        print("Fairness goal stats: " + str(f_goal))

        if not (h_sample_bool or scalability_bool):
                csv_writer.writerow(["Fairness Goal", "Grid search", "Iteration", "Value"])

                # Write data for ranking algorithm
                p.write_quartiles(csv_writer, "ranking", "iterations", rank_iquartiles)
                p.write_quartiles(csv_writer, "ranking", "Fairness", rank_fquartiles)
                csv_writer.writerow([])

                # Write data for grid search algorithm
                p.write_quartiles(csv_writer, "grid search", "iterations", g_iquartiles)
                p.write_quartiles(csv_writer, "grid search", "Fairness", g_fquartiles)
                csv_writer.writerow([])
f.close()