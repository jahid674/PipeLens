import pandas as pd
import numpy as np
from workingerpipeline import BlockBuilding, BlockCleaning, ComparisonCleaning, Matching
from regression import Regression
import os
import sys
from itertools import product
import random
import matplotlib.pyplot as plt
import operator
import itertools

class Param:
    def __init__(self):
        print("Parameter finding")
        #datasets to apply record linkage
        self.t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
        self.t2=pd.read_csv('DBLP-ACM/ACM.csv')

        #ground truth
        gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
        self.gt_list = gt.values.tolist()

        self.bstep_size = 8
        self.mstep_size = 8

        self.ranges = {}
        self.ranges[0] = [0,1]
        self.ranges[1] = np.linspace(0.05, 0.95,self.bstep_size).tolist()
        self.ranges[2] = list(range(4))
        self.ranges[3] = np.linspace(0.05, 0.95,self.mstep_size).tolist()

        self.k = {}
        self.k[0] = 2
        self.k[1] = self.bstep_size
        self.k[2] = 4
        self.k[3] = self.mstep_size

        self.coefs = []
        self.rank_f = -1
        self.rank_iter = -999

        self.gs_fdistr = []
        self.gs_idistr = []

        q_size = 0
        BuBl = BlockBuilding(q_size)
        self.blocks = BuBl.create_blocks_from_dataframe([self.t1,self.t2],['title'])

    def init_searchspace(self):
       if not(os.path.exists('ERmetrics/parameter_ranking3.csv')):
        profile_data = []
        for i in range(2):
            for j in self.ranges[1]:
                block_clean_thres = round(j,5)
                for k in range(4):
                    for l in self.ranges[3]:
                        matching_thres = round(l,5)
                        params = [i, block_clean_thres, k, matching_thres]
                        cur_f = self.pipeline(params)
                        profile_data.append((i, block_clean_thres, k, matching_thres, cur_f))
        #print(os.path.exists('ERmetrics/parameter_ranking.csv'))
        profiles_df = pd.DataFrame.from_records(profile_data, columns=['pf','block clean threshold', 'weighting schema', 'matching threshold', 'f-score'])
        profiles_df.to_csv('ERmetrics/parameter_ranking3.csv')
        #print(os.path.exists('ERmetrics/parameter_ranking.csv'))

    def ranking(self, fr = 1.0):
      if not(os.path.exists('ERmetrics/ranking_dataset1.csv')):
        profile_data = []
        for i in range(2):
            for j in np.linspace(0.05, 0.95, 3).tolist():
                block_clean_thres = round(j,5)
                for k in range(4):
                    for l in np.linspace(0.05, 0.95, 3).tolist():
                        matching_thres = round(l,5)
                        params = [i, block_clean_thres, k, matching_thres]
                        cur_f = self.pipeline(params)
                        profile_data.append((i, block_clean_thres, k, matching_thres, cur_f))
        #print(os.path.exists('ERmetrics/parameter_ranking.csv'))
        profiles_df = pd.DataFrame.from_records(profile_data, columns=['pf','block clean threshold', 'weighting schema', 'matching threshold', 'f-score'])
        profiles_df.to_csv('ERmetrics/ranking_dataset1.csv')
        #print(os.path.exists('ERmetrics/parameter_ranking.csv'))

      historical_df = pd.read_csv('ERmetrics/ranking_dataset1.csv').sample(frac = fr, random_state = 1)
      print(historical_df.shape)
      print("beginning regression generation")
      X = historical_df[['pf', 'block clean threshold', 'weighting schema', 'matching threshold']]
      y = historical_df['f-score']
      reg = Regression()
      model = reg.generate_regression(X, y)
      coefs = model.coef_
      print(coefs)
      print(model.intercept_)
      self.rank = np.argsort(np.abs(coefs))[::-1]
      self.coefs = coefs
      print(self.rank)
    
    # if f_goal is unachievable, failure is printed
    def optimize(self, init_params, f_goal):
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       profiles_df = pd.read_csv('ERmetrics/parameter_ranking2.csv') if self.bstep_size == 8 else pd.read_csv('ERmetrics/parameter_ranking3.csv')
       opt_f = profiles_df.loc[(profiles_df['pf'] == init_params[0]) & 
                                (profiles_df['block clean threshold'] == init_params[1]) &
                                (profiles_df['weighting schema'] == init_params[2]) &
                                (profiles_df['matching threshold'] == init_params[3])].iloc[0]['f-score']
       #print(opt_f)
       cur_params = init_params.copy()

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])

       iter_size = 0
       
       while (iter_size < len(self.rank)):
          iter_size += 1
          index = 0
          #print (iter_size,"********************")
          #if iter_size==2:
          #  iter_size=4
         #  self.rank_f = 0
         #  cur_params = init_params.copy()
          
          if iter_size == 1:
             for elem in self.rank:
                if self.coefs[elem] > 0:
                  # first self.k[elem] elems
                  #MAJOR CHANGE
                  #for parameter in self.ranges[elem][-self.k[elem]:]:
                  varlst=self.ranges[elem]
                  varlst.reverse()
                  for parameter in varlst:#self.ranges[elem]:#[:self.k[elem]]:
                     test_params = cur_params.copy()
                     test_params[elem] = round(parameter,5)
                     if tuple(test_params) not in seen:
                        self.rank_iter += 1
                        seen.add(tuple(test_params))
                     cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                     if cur_f >= f_goal:
                        cur_params = test_params.copy()
                        self.rank_f = cur_f
                        return cur_params # early exit when f_goal obtained
                     # update opt f for current iteration to improve upon in next iter
                     elif cur_f > opt_f:
                        cur_params = test_params.copy()
                        opt_f = cur_f
                     #MAJOR CHANGE
                     break
                else:
                  # last self.k[elem] elems
                  #MAJOR CHANGE
                  #for parameter in self.ranges[elem][:self.k[elem]]:
                  varlst=self.ranges[elem]
                  #varlst.reverse()
                  for parameter in varlst:#self.ranges[elem]:#[-self.k[elem]:]:
                     test_params = cur_params.copy()
                     test_params[elem] = round(parameter,5)
                     if tuple(test_params) not in seen:
                        self.rank_iter += 1
                        seen.add(tuple(test_params))
                     cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                     if cur_f >= f_goal:
                        cur_params = test_params.copy()
                        self.rank_f = cur_f
                        return cur_params # early exit when f_goal obtained
                     # update opt f for current iteration to improve upon in next iter
                     elif cur_f > opt_f:
                        cur_params = test_params.copy()
                        opt_f = cur_f
                     #MAJOR CHANGE
                     break
          else:
            #for each iter_size
            #Major change
            #iter_size=len(self.rank)
            #print (index)
            comb_size = iter_size-1
            #print (iter_size,self.rank)
            i=0
            comb_lst=[]
            for comb in (itertools.combinations(self.rank, comb_size)):
               comb_lst.append(comb)
            
            for comb in comb_lst:
               i=0
               cur_iterlst = []
               coef_lst=[]
               while i<comb_size:
                  cur_iterlst.append(self.ranges[comb[i]])
                  coef_lst.append(self.coefs[comb[i]])
                  i+=1
               score = {}
               for elem in product(*cur_iterlst):
                  score[elem] = sum([x*y for x,y in zip( coef_lst,elem)])
               sorted_params = sorted(score.items(), key=operator.itemgetter(1))
               sorted_params.reverse()
               #print (sorted_params,coef_lst)
               #print (sorted_params,comb)
               iter=0
               for (elem,score) in sorted_params:#sorted_params:#(elem,score)
                  iter+=1
                  test_params = cur_params.copy()
                  for j in range(comb_size):
                     test_params[comb[j]] = round(elem[j], 5)
                  #account for hashing
                  #print (test_params,cur_params)
                  if tuple(test_params) not in seen:
                     self.rank_iter += 1
                     seen.add(tuple(test_params))
                  #  print(index)
                  #  print(cur_iterlst)
                  #  print(test_params)
                  #print (set(list(profiles_df['pf'])))
                  cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                  if cur_f >= f_goal:
                     for j in range(comb_size):
                        cur_params[comb[j]] = test_params[comb[j]]#round(elem[j], 5)
                     self.rank_f = cur_f
                     return cur_params # early exit when f_goal obtained
                  # update opt f for current iteration to improve upon in next iter
                  elif cur_f > opt_f:
                     for j in range(comb_size):
                        cur_params[comb[j]] = test_params[comb[j]]#cur_params[j] = round(elem[j], 5)
                     opt_f = cur_f
            iter_size+=1

       if self.rank_iter == search_size:
          print("failure")
       return cur_params
    
    def optimize_fixed(self, init_params, f_goal):
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       nm_rank = np.copy(self.rank)
       nm_rank = np.delete(nm_rank, np.where(nm_rank == 3))
       profiles_df = pd.read_csv('ERmetrics/parameter_ranking2.csv')
       opt_f = profiles_df.loc[(profiles_df['pf'] == init_params[0]) & 
                                (profiles_df['block clean threshold'] == init_params[1]) &
                                (profiles_df['weighting schema'] == init_params[2]) &
                                (profiles_df['matching threshold'] == init_params[3])].iloc[0]['f-score']
       #print(opt_f)
       cur_params = init_params.copy()

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])

       iter_size = 0

       while (iter_size < len(nm_rank)):
          iter_size += 1
         #  self.rank_f = 0
         #  cur_params = init_params.copy()
          index = 0
          if iter_size == 1:
             for elem in nm_rank:
                if self.coefs[elem] > 0:
                  # first self.k[elem] elems
                  for parameter in self.ranges[elem][:self.k[elem]]:
                     test_params = cur_params.copy()
                     test_params[elem] = round(parameter,5)
                     if tuple(test_params) not in seen:
                        self.rank_iter += 1
                        seen.add(tuple(test_params))
                     cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                     if cur_f >= f_goal:
                        cur_params = test_params.copy()
                        self.rank_f = cur_f
                        return cur_params # early exit when f_goal obtained
                     # update opt f for current iteration to improve upon in next iter
                     elif cur_f > opt_f:
                        cur_params = test_params.copy()
                        opt_f = cur_f
                else:
                  # last self.k[elem] elems
                  for parameter in self.ranges[elem][-self.k[elem]:]:
                     test_params = cur_params.copy()
                     test_params[elem] = round(parameter,5)
                     if tuple(test_params) not in seen:
                        self.rank_iter += 1
                        seen.add(tuple(test_params))
                     cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                     if cur_f >= f_goal:
                        cur_params = test_params.copy()
                        self.rank_f = cur_f
                        return cur_params # early exit when f_goal obtained
                     # update opt f for current iteration to improve upon in next iter
                     elif cur_f > opt_f:
                        cur_params = test_params.copy()
                        opt_f = cur_f
          else:
            #for each iter_size
            while index < len(nm_rank):
               cur_iterlst = []
               for i in range(iter_size):
                  if (index >= len(nm_rank)):
                     break
                  cur_iterlst.append(self.ranges[nm_rank[index]])
                  index += 1
               for elem in product(*cur_iterlst):
                  test_params = cur_params.copy()
                  for j in range(len(cur_iterlst)):
                     test_params[nm_rank[index - len(cur_iterlst) + j]] = round(elem[j], 5)
                  #account for hashing
                  if tuple(test_params) not in seen:
                     self.rank_iter += 1
                     seen.add(tuple(test_params))
                  #  print(index)
                  #  print(cur_iterlst)
                  #  print(test_params)
                  cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                  if cur_f >= f_goal:
                     for j in range(len(cur_iterlst)):
                        cur_params[nm_rank[index - len(cur_iterlst) + j]] = round(elem[j], 5)
                     self.rank_f = cur_f
                     return cur_params # early exit when f_goal obtained
                  # update opt f for current iteration to improve upon in next iter
                  elif cur_f > opt_f:
                     for j in range(len(cur_iterlst)):
                        cur_params[nm_rank[index - len(cur_iterlst) + j]] = round(elem[j], 5)
                     opt_f = cur_f
       if self.rank_iter == search_size:
          print("failure")
       return cur_params
    

    def grid_search(self, f_goal, iterations):
       random.seed(0)
       self.gs_idistr = []
       self.gs_fdistr = []
       profiles_df = pd.read_csv('ERmetrics/parameter_ranking2.csv') if self.bstep_size == 8 else pd.read_csv('ERmetrics/parameter_ranking3.csv')
       iter_lst = []
       for i in range(4):
          iter_lst.append(self.ranges[i])
       
       for i in range(iterations):
         gs_iter = 0
         gs_f = 0
         cur_order = list(product(*iter_lst))
         random.shuffle(cur_order)
         for elem in cur_order:
            #print (np.linspace(0.05, 0.95,self.bstep_size).tolist())
            #print (np.linspace(0.05, 0.95,self.mstep_size).tolist())
            #print (elem)
            #print (profiles_df.loc[(profiles_df['pf'] == elem[0]) & 
            #                    (profiles_df['block clean threshold'] == round(elem[1],5)) &
            #                    (profiles_df['weighting schema'] == elem[2])] )
            cur_f = profiles_df.loc[(profiles_df['pf'] == elem[0]) & 
                                (profiles_df['block clean threshold'] == round(elem[1],5)) &
                                (profiles_df['weighting schema'] == elem[2]) &
                                (profiles_df['matching threshold'] == round(elem[3],5))].iloc[0]['f-score']
            gs_iter += 1
            if cur_f >= f_goal:
               gs_f = cur_f
               cur_params = []
               cur_params.append(elem[0]) 
               cur_params.append(round(elem[1],5))
               cur_params.append(elem[2])
               cur_params.append(round(elem[3],5))
               self.gs_fdistr.append(gs_f)
               self.gs_idistr.append(gs_iter)
               break
               #return cur_params

    def grid_search_fixed(self, f_goal, fixed_mt, iterations):
       self.gs_idistr = []
       self.gs_fdistr = []
       profiles_df = pd.read_csv('ERmetrics/parameter_ranking2.csv')
       iter_lst = []
       for i in range(3):
          iter_lst.append(self.ranges[i])
       
       for i in range(iterations):
         gs_iter = 0
         gs_f = 0
         cur_order = list(product(*iter_lst))
         random.shuffle(cur_order)
         for elem in cur_order:
            cur_f = profiles_df.loc[(profiles_df['pf'] == elem[0]) & 
                                (profiles_df['block clean threshold'] == round(elem[1],5)) &
                                (profiles_df['weighting schema'] == elem[2]) &
                                (profiles_df['matching threshold'] == fixed_mt)].iloc[0]['f-score']
            gs_iter += 1
            if cur_f >= f_goal:
               gs_f = cur_f
               cur_params = []
               cur_params.append(elem[0]) 
               cur_params.append(round(elem[1],5))
               cur_params.append(elem[2])
               cur_params.append(fixed_mt)
               self.gs_fdistr.append(gs_f)
               self.gs_idistr.append(gs_iter)
               break

    def pipeline(self, params):
        pf = False if params[0] == 0 else True
        BlCl = BlockCleaning(pf, params[1])
        CoCl = ComparisonCleaning(params[2])
        pairs = CoCl.generate_pairs(BlCl.clean_blocks(self.blocks))
        f_score = 0
        if len(pairs) > 0:
          Jm = Matching(params[3])
          (tp,fp,tn,fn) = Jm.pair_matching(pairs,[self.t1,self.t2],self.gt_list)
          if tp == 0:
             return 0
          fn = len(self.gt_list) - tp
          cur_p = round(tp / (tp + fp), 5)
          cur_r = round(tp / (tp + fn), 5)
          f_score = round((2 * cur_p * cur_r) / (cur_p + cur_r), 5)
        return f_score
    
if __name__ == "__main__":
    p = Param()
    p.init_searchspace()
    p.ranking()
   #  print(p.optimize([0,0.05,0,0.56429], 0.999))
   #  print(p.rank_iter)
   #  print(p.rank_f)

    data_df = pd.read_csv('ERmetrics/parameter_ranking3.csv')
    data = data_df[['f-score']].values
    #plt.hist(data, color='lightgreen', ec='black', bins=[x / 40.0 for x in range(41)])
    #plt.show()

    f_goals = [0.8, 0.9, 0.95, 0.96]
    profile_data = []
    f = sys.stdout
    if not(os.path.exists('ERmetrics/opt1drankingstats2.txt')):
      f = open('ERMetrics/opt1drankingstats2.txt', 'w')
    f.write("naive parameter ranking algorithm statistics")
    f.write('\n')
    for f_goal in f_goals:
      rank_idistr = []
      rank_fdistr = []
      p.grid_search(f_goal, 100)
      gs_idistr = p.gs_idistr
      gs_fdistr = p.gs_fdistr
      g_iquartiles = np.percentile(gs_idistr, [25, 50, 75, 100], interpolation='midpoint')
      g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75,100], interpolation='midpoint')
      failures = 0
      for param1 in [0,1]:
          for param2 in np.linspace(0.05, 0.95,p.bstep_size).tolist():
             for param3 in list(range(4)):
                for param4 in np.linspace(0.05, 0.95,p.mstep_size).tolist():
                   p.optimize([param1, round(param2, 5), param3, round(param4, 5)], f_goal)
                   if p.rank_iter != -1:
                    rank_idistr.append(p.rank_iter)
                    rank_fdistr.append(p.rank_f)
                   else:
                      failures += 1
                   profile_data.append((param1, round(param2, 5), param3, round(param4, 5), f_goal, p.rank_iter, p.rank_f,round(g_iquartiles[1],5), round(g_fquartiles[1],5)))
      rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75,100], interpolation='midpoint')
      rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75,100], interpolation='midpoint')
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
    profiles_df = pd.DataFrame.from_records(profile_data, columns=['init pf','init block clean threshold', 'init weighting schema', 'init matching threshold', 'f-score goal', 'rank iterations', 'rank f-score', 'grid search iterations', 'grid search f-score'])
    if not(os.path.exists('ERmetrics/optrankvsgs2.csv')):
      profiles_df.to_csv('ERmetrics/optrankvsgs2.csv')
    