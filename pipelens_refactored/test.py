import itertools
from modules.outlier_detection.outlier_detector import OutlierDetector
from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
from modules.normalization.normalizer import DataNormalizer
from modules.metric.metric import metric
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from regression import Regression
import time
import csv
import os
import pandas as pd
import numpy as np
from itertools import product
import random
#'MEAN', 'ZSB', 'MM'
# Define parameter grids for each component
# imputer_strategies = ['MEAN','MEDIAN','KNN']
# outlier_strategies = ['IQR','ZSB','LOF']
# normalizer_strategies = [ 'MM','ZS','DS']

imputer_strategies = ['KNN']
outlier_strategies = ['LOF']
normalizer_strategies = [ 'DS']
import sys

hdma_train  = "data/hmda/hmda_Orleans_X_train_1.csv"
hdma_test = "data/hmda/hmda_Calcasieu_X_test_1.csv"
train,test = Reader(hdma_train,hdma_test).load_data()

df_ty = train['action_taken'] 
df_race= train['race'] 
mask = np.random.rand(*train.shape) < 0.001

train[mask] = np.nan
train['action_taken'] = df_ty

# import pdb;pdb.set_trace()
print(train.isnull().sum())
dataset = {
        'train': train,
        'test': test,
}
# Iterate through all possible combinations
ranking = None

class base:
        def __init__(self):
                self.f = 'tdst';
                self.ranking = None
                self.ranges = {}
                self.ranges[0] = [1, 10, 100, 1000]
                self.ranges[1] = [ 1,4,5, 6,10,12]
                self.ranges[2] = [8,10,12,16,20]
                # self.ranges[0] = [0,1,2]
                # self.ranges[1] = [0,1,2]
                # self.ranges[2] = [0,1,2]
                # self.ranges[2] = [0.1,0.15,0.2,0.25,0.3]
                

                self.k = {}
                self.k[0] = 2
                self.k[1] = 8
                self.k[2] = 4
                self.k[3] = 8
                self.column_name = ['normal','imputer','outlier_strategy' ]
        def run(self,key):
                start_time = time.time()
                print("---------->> Started Normalization <<-----------")
                best_metric_acc = 0
                best_metric_eqq_odd = 99999999
                best_combination_acc = None
                best_combination_eqq_odd = None
                zer = 0;
                for imputer_strategy, outlier_strategy, normalizer_strategy in itertools.product(imputer_strategies, outlier_strategies, normalizer_strategies):
                # Create the components using the selected strategies
                        X = dataset.copy()
                        normalizer = DataNormalizer(X.copy(), strategy=normalizer_strategy).transform()
                        imputer = DataImputer(normalizer.copy(), strategy=imputer_strategy).transform()
                        outlier_detector = OutlierDetector(imputer.copy(), strategy=outlier_strategy).transform()
                        

                        # Build your pipeline using these components
                        df = outlier_detector[key]
                        
                        y = df['action_taken']
                        X = df.drop("action_taken", axis=1)
                        
                        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)
                        #     import pdb;pdb.set_trace()
                        print('Strategy-->',imputer_strategy, outlier_strategy, normalizer_strategy)
                        model = GaussianNB()
                        try:
                                model.fit(X, y)
                        except :
                                import pdb;pdb.set_trace()
                        # Perform cross-validation and evaluate the pipeline using your chosen metric
                        met = metric.equalized_odds(model,X,y)

                        # Keep track of the best combination
                        if(met==0):
                                zer = zer +1 
                        # print(met)
                        if  met < best_metric_eqq_odd:
                                # import pdb;pdb.set_trace()
                                best_metric_eqq_odd = met
                                best_combination_eqq_odd = (imputer_strategy, outlier_strategy, normalizer_strategy)
                        met = metric.calculate_accuracy(model,X,y)
                        if met > best_metric_acc:
                                best_metric_acc = met
                                best_combination_acc = (imputer_strategy, outlier_strategy, normalizer_strategy)

                        # Print the best combination
                # import pdb;pdb.set_trace()
                print("Best Combination "+key +" : ", best_combination_acc,'Accuracy Score: ',best_metric_acc)
                print("Best Combination  "+key +" : ", best_combination_eqq_odd,'Fairness Score(Eqq odd): ',best_metric_eqq_odd)
                print(zer)
                print("done -- CPU time:", time.time() - start_time, "seconds")

        def optimize(self, init_params, f_goal):
                self.rank_iter = 0
                seen = set()
                self.rank_f = 0
                base_parma = None
                profiles_df = pd.read_csv('historical_data.csv')
                opt_f = self.f_score_look_up(profiles_df,init_params[0],init_params[1],init_params[2])
                search_size = 1
                for key in self.ranges.keys():
                        search_size *= len(self.ranges[key])
                iter_size = 0   


                cur_params = init_params.copy()
                while (iter_size < len(self.ranking)):
                        iter_size += 1
                        # self.rank_f = 0
                        # cur_params = init_params.copy()
                        index = 0
                        #for each iter_size
                        if iter_size == 1:
                                for elem  in self.ranking:
                                        if self.coefs[elem]>0:
                                                for parameter in self.ranges[elem][:self.k[elem]]:
                                                        test_params = cur_params.copy()
                                                        test_params[elem] = round(parameter,5)
                                                        if tuple(test_params) not in seen:
                                                                self.rank_iter += 1
                                                                seen.add(tuple(test_params))

                                                        cur_f = self.f_score_look_up(profiles_df,test_params[0],test_params[1],test_params[2])
                                                        if cur_f <= f_goal:
                                                                cur_params = test_params.copy()
                                                                self.rank_f = cur_f
                                                                return cur_params # early exit when f_goal obtained
                                                        # update opt f for current iteration to improve upon in next iter
                                                        elif cur_f > opt_f:
                                                                cur_params = test_params.copy()
                                                                opt_f = cur_f
                                        else:
                                                for parameter in self.ranges[elem][-self.k[elem]:]:
                                                        test_params = cur_params.copy()
                                                        test_params[elem] = round(parameter,5)
                                                        if tuple(test_params) not in seen:
                                                                self.rank_iter += 1
                                                                seen.add(tuple(test_params))
                                                        cur_f = self.f_score_look_up(profiles_df,test_params[0],test_params[1],test_params[2])
                                                        if cur_f <= f_goal:
                                                                cur_params = test_params.copy()
                                                                self.rank_f = cur_f
                                                                return cur_params # early exit when f_goal obtained
                                                        # update opt f for current iteration to improve upon in next iter
                                                        elif cur_f < opt_f:
                                                                cur_params = test_params.copy()
                                                                opt_f = cur_f


                        else:
                        #for each iter_size
                                # print("backkkk")
                                # import pdb;pdb.set_trace()
                                while index < len(self.ranking):
                                        cur_iterlst = []
                                        for i in range(iter_size):
                                                if (index >= len(self.ranking)):
                                                        break
                                                cur_iterlst.append(self.ranges[self.ranking[index]])
                                                index += 1
                                        for elem in product(*cur_iterlst):
                                                test_params = cur_params.copy()
                                                for j in range(len(cur_iterlst)):
                                                        test_params[self.ranking[index - len(cur_iterlst) + j]] = round(elem[j], 5)
                                        #account for hashing
                                        if tuple(test_params) not in seen:
                                                self.rank_iter += 1
                                                seen.add(tuple(test_params))
                                        #  print(index)
                                        #  print(cur_iterlst)
                                        #  print(test_params)
                                        cur_f = self.f_score_look_up(profiles_df,test_params[0],test_params[1],test_params[2])
                                        if cur_f <= f_goal:
                                                for j in range(len(cur_iterlst)):
                                                        cur_params[self.ranking[index - len(cur_iterlst) + j]] = round(elem[j], 5)
                                                self.rank_f = cur_f
                                                return cur_params # early exit when f_goal obtained
                                        # update opt f for current iteration to improve upon in next iter
                                        elif cur_f < opt_f:
                                                for j in range(len(cur_iterlst)):
                                                        cur_params[self.ranking[index - len(cur_iterlst) + j]] = round(elem[j], 5)
                                                opt_f = cur_f
                if self.rank_iter == search_size:
                        print("failure")
                return cur_params

        def f_score_look_up(self,profiles_df,nor,imp,neighbors):
                column_names = self.column_name
                try:
                        return profiles_df.loc[(profiles_df[column_names[0]] == nor) & (profiles_df[column_names[1]] == imp ) & (profiles_df[column_names[2]] == neighbors) ].iloc[0]['f-score']
                except:
                        import pdb;pdb.set_trace()
        
        def grid_search2(self, f_goal):
                self.gs_iter = 0
                self.gs_f = 0
                optimize_param = None
                column_names = self.column_name
                profiles_df = pd.read_csv('historical_data.csv')
                count  = 0
                for nor in self.ranges[0]:
                        for impt in self.ranges[1] :
                                for neighbors in self.ranges[2]:
                                        # for thres in self.ranges[3]:
                                                # import pdb;pdb.set_trace()
                                                cur_f = profiles_df.loc[(profiles_df[column_names[0]] == nor) & 
                                                        (profiles_df[column_names[1]] == impt ) &
                                                        (profiles_df[column_names[2]] == neighbors) ].iloc[0]['f-score']
                                                self.gs_iter += 1
                                                if cur_f <= f_goal:
                                                        self.gs_f = cur_f
                                                        cur_params = []
                                                        cur_params.append(nor) 
                                                        cur_params.append(impt)
                                                        cur_params.append(neighbors)
                                                        
                                                        # print(self.gs_f)
                                                        # print('Iteratiaon',)
                                                        return cur_params
                return optimize_param
        def grid_search(self, f_goal, iterations):
                self.gs_idistr = []
                self.gs_fdistr = []
                profiles_df = pd.read_csv('historical_data.csv')
                iter_lst = []
                for i in range(3):
                        iter_lst.append(self.ranges[i])
                
                for i in range(iterations):
                        gs_iter = 0
                        gs_f = 0
                        cur_order = list(product(*iter_lst))
                        random.shuffle(cur_order)
                        for elem in cur_order:
                                cur_f = self.f_score_look_up(profiles_df,elem[0],elem[1],elem[2])
                                gs_iter += 1
                                if cur_f <= f_goal:
                                        gs_f = cur_f
                                        cur_params = []
                                        cur_params.append(elem[0]) 
                                        cur_params.append(elem[1])
                                        cur_params.append(elem[2])
                                        # print(gs_iter)
                                        self.gs_fdistr.append(gs_f)
                                        self.gs_idistr.append(gs_iter)
                                        break
                        #return cur_params
        def generate_historic_data(self):
                best_metric_eqq_odd = 9999999
                historical_data = []
                import random
                e = 0;
                dict = {}
                best_params = None
                if not(os.path.exists('historical_data.csv')):
                        for nor in self.ranges[0]:
                                for impt in self.ranges[1] :
                                        for neighbors in  self.ranges[2]:
                                                # for thres in self.out_thres:
                                                if True:
                                                        X = dataset.copy()
                                                        y = X['train']['action_taken']
                                                        
                                                        # dict[random_sample] = 1
                                                        X_train, y_train = X['train'], y

                                                        d_copy = {
                                                                'train': X_train,
                                                                'test': X_train,
                                                        }
                                                        normalizer_strategy = normalizer_strategies[0]
                                                        imputer_strategy = imputer_strategies[0]
                                                        outlier_strategy = outlier_strategies[0]
                                                        normalizer = DataNormalizer(d_copy.copy(), strategy=normalizer_strategy,threshold=.3,decimal=nor).transform()
                                                        imputer = DataImputer(normalizer.copy(), strategy=imputer_strategy,threshold=.3,neighbour=impt).transform()
                                                        outlier_detector = OutlierDetector(imputer.copy(), strategy=outlier_strategy,threshold=.3,n_neighbors = neighbors).transform()
                                                        df = outlier_detector['train']
                                                
                                                        y_t = df['action_taken']
                                                        X_t = df.drop("action_taken", axis=1)
                                                        
                                                        model = GaussianNB()
                                                        try:
                                                                model.fit(X_t, y_t)
                                                        except Exception as e:
                                                                import pdb;pdb.set_trace()
                                                                print(e)
                                                # Perform cross-validation and evaluate the pipeline using your chosen metric
                                                        met = metric.equalized_odds(model,X_t,y_t)
                                                        param_run = {
                                                                'f-score':met,
                                                                'normal':nor,
                                                                'imputer':impt,
                                                                'outlier_strategy':neighbors,
                                                                
                                                        }
                                                        if(met<best_metric_eqq_odd):
                                                                best_params = param_run;
                                                        historical_data.append(param_run)
                                
                        field_names = historical_data[0].keys()
                        with open('historical_data.csv', 'w', newline='') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                                writer.writeheader()
                                for val in (historical_data):
                                        writer.writerow(val)
                print(best_params)
                profiles_df = pd.read_csv('historical_data.csv')
                print("beginning regression generation")

                y = profiles_df['f-score']
                X = profiles_df.copy().drop("f-score", axis=1)
                reg = Regression()
                model = reg.generate_regression(X, y)
                # import pdb;pdb.set_trace()
                coefs = model.coef_
                print(coefs)
                print(model.intercept_)

                self.ranking = np.argsort(np.abs(coefs))[::-1]
                self.coefs = coefs
                print(self.ranking)

# {'metric': 0.03519426047730656, 'normal_DS': 10000, 'imputer_KNN': 8, 'outlier_strategy_LOF_neighbour ': 23, 'outlier_LOF_thres': 0.3}
# generate_historic_data()

# naiveparameterselection.grid_xsearch()

obj = base()
obj.generate_historic_data()
# print(obj.grid_search(0.03))
# print(, 0.999)
# print(obj.rank_iter)
# print(obj.rank_f)
f_goals = [0.2, 0.15, 0.1, 0.05, 0.025, 0.01]
profile_data = []
# # f = sys.stdout
# # if not(os.path.exists('ERmetrics/1drankingstats2.txt')):
# #         f = open('ERMetrics/1drankingstats2.txt', 'w')
# # f.write("naive parameter ranking algorithm statistics")
# # f.write('\n')
# itr = 0
# result =   []
# metric_data = []
# for f_goal in f_goals:
#         # print(f'---->>')
#         param = obj.grid_search(f_goal)
#         dict_csv ={
#         'grid_search_score':obj.gs_f,
#         'Grasp_score':'',
#         'param_grid_search':param,
#         'param_grasp_search':'',
#         'f_goal':f_goal
#         }       
#         itr = 0
#         fail = 0
#         passed = 0
#         print(itr)
#         print(f'best param in grid search for f goal {f_goal} -  f-score found {obj.gs_f} iteration {obj.gs_iter}')
#         # for param1 in obj.ranges[0]:
#         #   for param2 in obj.ranges[1]:
#         #      for param3 in obj.ranges[2]:
#         param_found,iteration_grasp = obj.optimize([10,8,10], f_goal)
#         print(f'best param in optimize search for f goal {f_goal} - {param_found},Iteration {iteration_grasp} f-score found {obj.rank_f}')
        
#         dict_csv ={
#         'grid_search_score':obj.gs_f,
#         'Grasp_score':obj.rank_f,
#         'param_grid_search':param,
#         'param_grasp_search':param_found,
#         'f_goal':f_goal
#         }      
#         if(obj.rank_f!=0 and obj.gs_f>obj.rank_f):
#                 passed +=1
#         else:
#                 fail +=1
#         result.append(dict_csv)
#         itr +=1

#         ff = {
#                 'grasp_failed':fail,
#                 'grasp_passed':passed,
#                 'total_iteration_with different seed':itr,
#                 'target_f_score':f_goal
#         }
#         metric_data.append(ff)
#         # print(f'<<<----')
# print(metric_data)

p = base()
p.generate_historic_data()
f = sys.stdout
if not(os.path.exists('metric.txt')):
        f = open('metric.txt', 'w')
f.write("Ranking  and Grid search statistic ")
f.write('\n')
for f_goal in f_goals:
        rank_idistr = []
        rank_fdistr = []
        p.grid_search(f_goal, 100)
        gs_idistr = p.gs_idistr
        gs_fdistr = p.gs_fdistr
        # import pdb;pdb.set_trace()
        try:
                g_iquartiles = np.percentile(gs_idistr, [25, 50, 75], interpolation='midpoint')
                g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75], interpolation='midpoint')
        except:
                import pdb;pdb.set_trace()
        failures = 0
        for param1 in p.ranges[0]:
                for param2 in p.ranges[1]:
                        for param3 in p.ranges[2]:
                                p.optimize([param1, param2, param3], f_goal)
                                if p.rank_iter != -1:
                                        rank_idistr.append(p.rank_iter)
                                        rank_fdistr.append(p.rank_f)
                                else:
                                        failures += 1
                                profile_data.append((param1, param2,param3, f_goal, p.rank_iter, p.rank_f,round(g_iquartiles[1],5), round(g_fquartiles[1],5)))
        rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75], interpolation='midpoint')
        rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75], interpolation='midpoint')
        print("f-score goal stats: " + str(f_goal))
        f.write("stats for f-score goal: " + str(f_goal))
        f.write('\n')
        print(rank_iquartiles)
        f.write("ranking algorithm iterations q1: " + str(round(rank_iquartiles[0], 5)))
        f.write('\n')
        f.write("ranking algorithm iterations q2: " + str(round(rank_iquartiles[1], 5)))
        f.write('\n')
        f.write("ranking algorithm iterations q3: " + str(round(rank_iquartiles[2], 5)))
        f.write('\n')
        print(rank_fquartiles)
        f.write("ranking algorithm f-score q1: " + str(round(rank_fquartiles[0], 5)))
        f.write('\n')
        f.write("ranking algorithm f-score q2: " + str(round(rank_fquartiles[1], 5)))
        f.write('\n')
        f.write("ranking algorithm f-score q3: " + str(round(rank_fquartiles[2], 5)))
        f.write('\n')
        print(failures)
        f.write("ranking algorithm failures: " + str(failures))
        f.write('\n')
        print(g_iquartiles)
        f.write("grid search iterations q1: " + str(round(g_iquartiles[0], 5)))
        f.write('\n')
        f.write("grid search iterations q2: " + str(round(g_iquartiles[1], 5)))
        f.write('\n')
        f.write("grid search iterations q3: " + str(round(g_iquartiles[2], 5)))
        f.write('\n')
        print(g_fquartiles)
        f.write("grid search f-score q1: " + str(round(g_fquartiles[0], 5)))
        f.write('\n')
        f.write("grid search f-score q2: " + str(round(g_fquartiles[1], 5)))
        f.write('\n')
        f.write("grid search f-score q3: " + str(round(g_fquartiles[2], 5)))
        f.write('\n')
      
f.close()

