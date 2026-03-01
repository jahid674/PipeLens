import pandas as pd
import numpy as np
import os
import sys
from itertools import product
import random
import operator
import matplotlib.pyplot as plt
import itertools

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from AmazonData.erpipeline import BlockBuilding, BlockCleaning, ComparisonCleaning, Matching
from regression import Regression

from AmazonData.profiling.profile import Profile

random.seed(0)

class OptimizingParam:
    def __init__(self):
        print("Parameter finding")

        self.passing_profile = Profile()
        #self.passing_profile.random_sample()
        self.failing_profile = Profile()
        #self.failing_profile.random_sample()

        self.p1=pd.read_csv('AmazonData/historical/original_sample.csv',encoding="latin-1")
        self.p2=pd.read_csv('AmazonData/historical/noisy_sample.csv',encoding="latin-1")

        #self.p1sample = pd.read_csv('DBLP-ACM/DBLP2_Sample.csv',encoding="latin-1")
        #self.p2sample = pd.read_csv('DBLP-ACM/ACM_Sample.csv')

        self.t1=pd.read_csv('AmazonData/failing/english_products.csv',encoding="latin-1")
        self.t2=pd.read_csv('AmazonData/failing/multilingual_products.csv',encoding="latin-1")

        #self.t1sample = pd.read_csv('DBLP-ACM/noisyDBLP2_Sample.csv',encoding="latin-1")
        #self.t2sample = pd.read_csv('DBLP-ACM/noisyACM_Sample.csv')

        # sim_dtypes = {
        #    'asin1': 'string',
        #    'asin2': 'string',
        #    'avg_sim': 'float64'
        # }
        # self.passing_sim = pd.read_csv('AmazonData/metrics/passing_similarities.csv',dtype=sim_dtypes)
        # self.failing_sim = pd.read_csv('AmazonData/metrics/failing_similarities.csv',dtype=sim_dtypes)

        self.passing_profile.generate_bbprofiles([self.p1, self.p2], ['title'])
        self.failing_profile.generate_bbprofiles([self.t1, self.t2], ['title'])

        #ground truth
        gt1 = pd.read_csv('AmazonData/historical/asin_mapping.csv')
        self.gt1_list = gt1.values.tolist()

        gt2 = pd.read_csv('AmazonData/failing/asin_mapping.csv')
        self.gt2_list = gt2.values.tolist()

        #should be 9
        self.bstep_size = 4
        self.mstep_size = 4

        self.ranges = {}
        self.ranges[0] = [0,1]
        self.ranges[1] = [0,4,5]
        self.ranges[2] = [0,1]
        self.ranges[3] = np.linspace(0.05, 0.95,self.bstep_size).tolist()
        self.ranges[4] = list(range(4))
        self.ranges[5] = np.linspace(0.05, 0.95,self.mstep_size).tolist()

        self.k = {}
        self.k[0] = 3
        self.k[1] = 2
        self.k[2] = 8
        self.k[3] = 4
        self.k[4] = 8

        q_lst = [0,4,5]
        self.blocks = {}
        BuBl1 = {}
        #self.sample_blocks = {}
        #BuBl1_Sample = {}
        for q in q_lst:
          BuBl1[q] = BlockBuilding(q)
          #BuBl1_Sample[q] = BlockBuilding(q)
          self.blocks[q] = BuBl1[q].create_blocks_from_dataframe([self.p1,self.p2],['title'])
          #self.sample_blocks[q] = BuBl1_Sample[q].create_blocks_from_dataframe([self.p1sample, self.p2sample], ['title'])

        self.failing_blocks = {}
        self.failing_translated_blocks = {}
        BuBl2 = {}
        BuBl_translate = {}
        #self.sample_failing_blocks = {}
        #BuBl2_Sample = {}
        for q in q_lst:
          BuBl2[q] = BlockBuilding(q)
          BuBl_translate[q] = BlockBuilding(q)
          #BuBl2_Sample[q] = BlockBuilding(q)
          self.failing_blocks[q] = BuBl2[q].create_blocks_from_dataframe([self.t1,self.t2],['title'])
          self.failing_translated_blocks[q] = BuBl_translate[q].create_blocks_from_dataframe([self.t1,self.t2],['title'],True)
          #self.sample_failing_blocks[q] = BuBl2_Sample[q].create_blocks_from_dataframe([self.t1sample, self.t2sample], ['title'])

        self.Jm_pass = {}
        self.Jm_fail = {}
        for m_thres in self.ranges[5]:
          cur_thres = round(m_thres, 5)
          self.Jm_pass[cur_thres] = Matching(cur_thres)
          self.Jm_fail[cur_thres] = Matching(cur_thres)

    def ranking(self):
      #generate historical data from passing dataset
      if not(os.path.exists('AmazonData/metrics/passing_historical.csv')):
        profile_data = []
        q_lst = [0,4,5]
        for q in q_lst:
          for i in range(2):
              for j in np.linspace(0.05, 0.95, 4).tolist():
                  block_clean_thres = round(j,5)
                  for k in range(4):
                      for l in np.linspace(0.05, 0.95, 4).tolist():
                          matching_thres = round(l,5)
                          params = [q, i, block_clean_thres, k, matching_thres]
                          (_, _, _, _, _, cur_f) = self.pipeline(params, True, False)
                          profile_data.append((q, i, block_clean_thres, k, matching_thres, cur_f))
        #print(os.path.exists('ERmetrics/parameter_ranking.csv'))
        profiles_df = pd.DataFrame.from_records(profile_data, columns=['q size', 'pf','block clean threshold', 'weighting schema', 'matching threshold', 'f-score'])
        profiles_df.to_csv('AmazonData/metrics/passing_historical.csv')
        #print(os.path.exists('ERmetrics/parameter_ranking.csv'))

      profiles_df = pd.read_csv('AmazonData/metrics/passing_historical.csv')
      print("beginning regression generation")
      X = profiles_df[['q size', 'pf', 'block clean threshold', 'weighting schema', 'matching threshold']]
      y = profiles_df['f-score']
      reg = Regression()
      model = reg.generate_regression(X, y)
      coefs = model.coef_
      # print(coefs)
      # print(model.intercept_)
      self.rank = np.argsort(np.abs(coefs))[::-1]
      self.coefs = coefs
      # print(self.ranking)
    
    def rank_profiles(self):
      if not(os.path.exists('AmazonData/metrics/passing_historical_profiles.csv')):
        profile_data = []
        q_lst = [0,4,5]
        for q in q_lst:
          for i in range(2):
            for j in np.linspace(0.05, 0.95, 4).tolist():
                block_clean_thres = round(j,5)
                for k in range(4):
                    for l in np.linspace(0.05, 0.95, 4).tolist():
                        matching_thres = round(l,5)
                        params = [q, i, block_clean_thres, k, matching_thres]
                        (stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, _) = self.pipeline(params, True, True)
                        (_, _, _, _, _, cur_f) = self.pipeline(params, True, False)
                        profile_data.append((q, i, block_clean_thres, k, matching_thres, stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, cur_f))
       
        profiles_df = pd.DataFrame.from_records(profile_data, columns=['q size', 'pf','block clean threshold', 'weighting schema', 'matching threshold', 'stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3', 'f-score'])
        profiles_df.to_csv('AmazonData/metrics/passing_historical_profiles.csv')
        
      profiles_df = pd.read_csv('AmazonData/metrics/passing_historical_profiles.csv')
      print("beginning profile regression generation")
      X = profiles_df[['stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3']]
      y = profiles_df['f-score']
      reg = Regression()
      model = reg.generate_regression(X, y)
      coefs = model.coef_
      # print(coefs)
      # print(model.intercept_)
      self.profile_ranking = np.argsort(np.abs(coefs))[::-1]
      self.profile_coefs = coefs
      # print(self.profile_ranking)

    def failing_rank_profiles(self):
      profiles_df = pd.read_csv('AmazonData/metrics/failing_profiles.csv')
      print("beginning failing profile regression generation")
      X = profiles_df[['stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3']]
      y = profiles_df['f-score']
      reg = Regression()
      model = reg.generate_regression(X, y)
      coefs = model.coef_
      # print(coefs)
      # print(model.intercept_)
      # print(np.argsort(np.abs(coefs))[::-1])

    def init_profile_searchspace(self):
       if not(os.path.exists('AmazonData/metrics/failing_profiles.csv')):
        profile_data = []
        for t in self.ranges[0]:
          for q in self.ranges[1]:
            for i in self.ranges[2]:
              for j in self.ranges[3]:
                  block_clean_thres = round(j,5)
                  for k in self.ranges[4]:
                      for l in self.ranges[5]:
                          matching_thres = round(l,5)
                          params = [q, i, block_clean_thres, k, matching_thres]
                          (stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, _) = self.pipeline(params, False, True, t == 1)
                          (_, _, _, _, _, cur_f) = self.pipeline(params, False, False, t == 1)
                          profile_data.append((t, q, i, block_clean_thres, k, matching_thres, stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, cur_f))
        profiles_df = pd.DataFrame.from_records(profile_data, columns=['translation', 'q size', 'pf','block clean threshold', 'weighting schema', 'matching threshold', 'stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3', 'f-score'])
        profiles_df.to_csv('AmazonData/metrics/failing_profiles.csv')
       print('profile search space inited')

    def rank_parameter(self):
      self.hparameter_rank = {}
      self.hparameter_coefs = {}
      profiles_df = pd.read_csv('AmazonData/metrics/passing_historical_profiles.csv')
      print("beginning parameter regression generation")
      self.profiles = ['stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3']
      for elem in self.profiles:
        X = profiles_df[['q size', 'pf', 'block clean threshold', 'weighting schema', 'matching threshold']]
        y = profiles_df[elem]
        reg = Regression()
        model = reg.generate_regression(X, y)
        coefs = model.coef_
      #   print(coefs)
      #   print(model.intercept_)
        self.hparameter_rank[elem] = np.argsort(np.abs(coefs))[::-1]
        self.hparameter_coefs[elem] = coefs
      #   print(elem + ':')
      #   print(self.ranking[elem])

    def failing_rank_parameters(self):
        self.parameter_ranking = {}
        self.parameter_coefs = {}
        profiles_df = pd.read_csv('AmazonData/metrics/failing_profiles.csv')
        print("beginning parameter regression generation")
        for elem in self.profiles:
          X = profiles_df[['q size', 'pf', 'block clean threshold', 'weighting schema', 'matching threshold']]
          y = profiles_df[elem]
          reg = Regression()
          model = reg.generate_regression(X, y)
          coefs = model.coef_
         #  print(coefs)
         #  print(model.intercept_)
          self.parameter_ranking[elem] = np.argsort(np.abs(coefs))[::-1]
          self.parameter_coefs[elem] = coefs
         #  print(elem + ':')
         #  print(ranking[elem])

    # if f_goal is unachievable, failure is printed
    # 1-d optimize
    def optimize(self, init_params, f_goal):
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       self.trace_data = []
       profiles_df = pd.read_csv('AmazonData/metrics/failing_profiles.csv')
       
       #(_, _, _, _, _, opt_f) = self.pipeline([init_params[0], init_params[1], init_params[2], init_params[3], init_params[4]], False, False)
       opt_f = profiles_df.loc[(profiles_df['q size'] == init_params[0]) & 
                                (profiles_df['pf'] == init_params[1]) & 
                                (profiles_df['block clean threshold'] == init_params[2]) &
                                (profiles_df['weighting schema'] == init_params[3]) &
                                (profiles_df['matching threshold'] == init_params[4])].iloc[0]['f-score']
       #print(opt_f)

       cur_params = init_params.copy()

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])

       iter_size = 0

       while (iter_size < len(self.rank)):
          iter_size += 1
         #  self.rank_f = 0
         #  cur_params = init_params.copy()
          index = 0
          if iter_size == 1:
             for elem in self.rank:
                if self.coefs[elem] > 0:
                  varlst=self.ranges[elem]
                  varlst.reverse
                  for parameter in varlst:
                     test_params = cur_params.copy()
                     test_params[elem] = round(parameter,5)
                     cur_f = profiles_df.loc[(profiles_df['q size'] == test_params[0]) &
                                 (profiles_df['pf'] == test_params[1]) & 
                                 (profiles_df['block clean threshold'] == test_params[2]) &
                                 (profiles_df['weighting schema'] == test_params[3]) &
                                 (profiles_df['matching threshold'] == test_params[4])].iloc[0]['f-score']
                     if tuple(test_params) not in seen:
                        self.rank_iter += 1
                        #(_, _, _, _, _, cur_f) = self.pipeline([test_params[0], test_params[1], test_params[2], test_params[3], test_params[4]], False, False)
                        # print(cur_f)
                        # print(opt_f)
                        seen.add(tuple(test_params))
                        self.trace_data.append((test_params[0], test_params[1], test_params[2], test_params[3], test_params[4], cur_f))
                     if cur_f >= f_goal:
                        cur_params = test_params.copy()
                        self.rank_f = cur_f
                        return cur_params # early exit when f_goal obtained
                     # update opt f for current iteration to improve upon in next iter
                     elif cur_f > opt_f:
                        cur_params = test_params.copy()
                        opt_f = cur_f
                     break
                else:
                  # last self.k[elem] elems
                  varlst=self.ranges[elem]
                  for parameter in varlst:
                     test_params = cur_params.copy()
                     test_params[elem] = round(parameter,5)
                     cur_f = profiles_df.loc[(profiles_df['q size'] == test_params[0]) &
                                 (profiles_df['pf'] == test_params[1]) & 
                                 (profiles_df['block clean threshold'] == test_params[2]) &
                                 (profiles_df['weighting schema'] == test_params[3]) &
                                 (profiles_df['matching threshold'] == test_params[4])].iloc[0]['f-score']
                     if tuple(test_params) not in seen:
                        self.rank_iter += 1
                        #(_, _, _, _, _, cur_f) = self.pipeline([test_params[0], test_params[1], test_params[2], test_params[3], test_params[4]], False, False)
                        # print(cur_f)
                        # print(opt_f)
                        seen.add(tuple(test_params))
                        self.trace_data.append((test_params[0], test_params[1], test_params[2], test_params[3], test_params[4], cur_f))
                     if cur_f >= f_goal:
                        cur_params = test_params.copy()
                        self.rank_f = cur_f
                        return cur_params # early exit when f_goal obtained
                     # update opt f for current iteration to improve upon in next iter
                     elif cur_f > opt_f:
                        cur_params = test_params.copy()
                        opt_f = cur_f
                     break
          else:
            #for each iter_size
            comb_size = iter_size - 1
            i = 0
            comb_lst = []
            for comb in (itertools.combinations(self.rank, comb_size)):
               comb_lst.append(comb)

            for comb in comb_lst:
               i=0
               cur_iterlst=[]
               coef_lst=[]
               while i < comb_size:
                  cur_iterlst.append(self.ranges[comb[i]])
                  coef_lst.append(self.coefs[comb[i]])
                  i+=1
               score = {}
               for elem in product(*cur_iterlst):
                  score[elem] = sum([x*y for x,y in zip( coef_lst, elem)])
               sorted_params = sorted(score.items(), key = operator.itemgetter(1))
               sorted_params.reverse()
               iter=0
               for (elem, score) in sorted_params:
                  iter+=1
                  test_params = cur_params.copy()
                  for j in range(comb_size):
                     test_params[comb[j]] = round(elem[j], 5)
                  cur_f = profiles_df.loc[(profiles_df['q size'] == test_params[0]) &
                                 (profiles_df['pf'] == test_params[1]) & 
                                 (profiles_df['block clean threshold'] == test_params[2]) &
                                 (profiles_df['weighting schema'] == test_params[3]) &
                                 (profiles_df['matching threshold'] == test_params[4])].iloc[0]['f-score']
                  if tuple(test_params) not in seen:
                     self.rank_iter +=1
                     seen.add(tuple(test_params))
                     self.trace_data.append((test_params[0], test_params[1], test_params[2], test_params[3], test_params[4], cur_f))
                  if cur_f >= f_goal:
                     for j in range(comb_size):
                        cur_params[comb[j]] = test_params[comb[j]]
                     self.rank_f = cur_f
                     return cur_params
                  elif cur_f > opt_f:
                     for j in range(comb_size):
                        cur_params[comb[j]] = test_params[comb[j]]
                     opt_f = cur_f
            iter_size+=1
       if self.rank_iter == search_size:
          print("failure")
       return cur_params

    # if f_goal unachievable, failure is printed
    def profile_pipeline_optimize(self, init_params, f_goal):
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       (_, _, _, _, _, opt_f) = self.pipeline([init_params[0], init_params[1], init_params[2], init_params[3], init_params[4]], False, False)
       #print(opt_f)
       opt_params = tuple(init_params)

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])
      
       opt_index = 0 #represents opt_index th best set of parameters to take from most important profile, no fallback when opt_index = 0
       sorted_params = [] #represents parameters sorted by most important profile
       
       while(opt_index < search_size):
        for idx, profile_index in enumerate(self.profile_ranking):
          opt_profile = self.pipeline([init_params[0], init_params[1], init_params[2], init_params[3], init_params[4]], False, True)[profile_index]
          if(opt_index == 0 or idx != 0):
           for elem in self.ranking[self.profiles[profile_index]]:
             cur_params = list(opt_params)
             test_params = cur_params.copy()
             #want maximal profile value
             if self.profile_coefs[profile_index] > 0:
               if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                  test_params[elem] = round(self.ranges[elem][-1],5)
               else:
                  test_params[elem] = round(self.ranges[elem][0],5)
               cur_profile = self.pipeline([test_params[0], test_params[1], test_params[2], test_params[3], test_params[4]], False, True)[profile_index]
               if cur_profile > opt_profile:
                    cur_params[elem] = test_params[elem]
                    opt_profile = cur_profile
             #want minimal profile value
             else:
               if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                  test_params[elem] = round(self.ranges[elem][0],5)
               else:
                  test_params[elem] = round(self.ranges[elem][-1],5)
               cur_profile = self.pipeline([test_params[0], test_params[1], test_params[2], test_params[3], test_params[4]], False, True)[profile_index]
               if cur_profile < opt_profile:
                    cur_params[elem] = test_params[elem]
                    opt_profile = cur_profile
             (_, _, _, _, _, cur_f) = self.pipeline([cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4]], False, False)
             if tuple(cur_params) not in seen:
                   self.rank_iter += 1
                   seen.add(tuple(cur_params))
             if cur_f >= opt_f:
                opt_f = cur_f
                opt_params = tuple(cur_params)
          #need to sort
          elif(opt_index == 1): 
            #print("case two")
            map = {}
            iter_list = []
            for key in range(5):
               iter_list.append(self.ranges[key])
            for param_lst in product(*iter_list):
               rparam_lst = (param_lst[0], param_lst[1],round(param_lst[2], 5), param_lst[3], round(param_lst[4], 5)) #tuple that fixes rounding errors in param_lst
               map[rparam_lst] = self.pipeline([rparam_lst[0], rparam_lst[1], rparam_lst[2], rparam_lst[3], rparam_lst[4]], False, True)[profile_index]
            #sort descending
            if self.profile_coefs[profile_index] > 0:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))
              sorted_params.reverse()
            #sort ascending
            else:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))

            #print(sorted_params)
            cur_params = sorted_params[opt_index][0]
            (_, _, _, _, _, cur_f) = self.pipeline([cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4]], False, False)
            if cur_params not in seen:
                   self.rank_iter += 1
                   seen.add(cur_params)
            if cur_f >= opt_f:
              opt_f = cur_f
              opt_params = cur_params
          #need to use sorted
          else:
            # print("case three")
            # print(idx)
            # print(opt_index)
            # print(len(sorted_params))
            cur_params = sorted_params[opt_index][0]
            (_, _, _, _, _, cur_f) = self.pipeline([cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4]], False, False)
            if cur_params not in seen:
                   self.rank_iter += 1
                   seen.add(cur_params)
            if cur_f >= opt_f:
              opt_f = cur_f
              opt_params = cur_params

          if opt_f >= f_goal:
             #done
             self.rank_f = opt_f
             return opt_params
        opt_index += 1 
       # indicates f_goal is not achievable
       print("failure")
       self.rank_f = opt_f
       return opt_params
    
    # 2-d optimize
    def profile_memoization_optimize(self, init_params, f_goal):
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       profiles_df = pd.read_csv('AmazonData/metrics/failing_profiles.csv')
       self.trace_data = []

       print(self.profile_ranking)
       print(self.profile_coefs)
       for profile in self.profiles:
          print(profile)
          print(self.parameter_ranking[profile])
          print(self.parameter_coefs[profile])
       
       #(_, _, _, _, _, _, opt_f) = self.pipeline([init_params[0], init_params[1], init_params[2], init_params[3], init_params[4]], False, False)
       opt_f = profiles_df.loc[ (profiles_df['translation'] == init_params[0]) &
                                (profiles_df['q size'] == init_params[1]) & 
                                (profiles_df['pf'] == init_params[2]) & 
                                (profiles_df['block clean threshold'] == init_params[3]) &
                                (profiles_df['weighting schema'] == init_params[4]) &
                                (profiles_df['matching threshold'] == init_params[5])].iloc[0]['f-score']
       #print(opt_f)
       opt_params = tuple(init_params)

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])
      
       opt_index = 0 #represents opt_index th best set of parameters to take from most important profile, no fallback when opt_index = 0
       sorted_params = [] #represents parameters sorted by most important profile
       
       while(opt_index < search_size):
        for idx, profile_index in enumerate(self.profile_ranking):
          print(self.profiles[profile_index])
          opt_profile = profiles_df.loc[(profiles_df['translation'] == opt_params[0]) &
                                (profiles_df['q size'] == opt_params[1]) & 
                                (profiles_df['pf'] == opt_params[2]) & 
                                (profiles_df['block clean threshold'] == opt_params[3]) &
                                (profiles_df['weighting schema'] == opt_params[4]) &
                                (profiles_df['matching threshold'] == opt_params[5])].iloc[0][self.profiles[profile_index]]

          if(opt_index == 0 or idx != 0):
           for elem in self.parameter_ranking[self.profiles[profile_index]]:
             cur_params = list(opt_params)
             test_params = cur_params.copy()
             #want maximal profile value
             if self.profile_coefs[profile_index] > 0:
               if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                  test_params[elem] = round(self.ranges[elem][-1],5)
               else:
                  test_params[elem] = round(self.ranges[elem][0],5)
               
               cur_profile = profiles_df.loc[(profiles_df['translation'] == test_params[0]) &
                              (profiles_df['q size'] == test_params[1]) & 
                              (profiles_df['pf'] == test_params[2]) & 
                              (profiles_df['block clean threshold'] == test_params[3]) &
                              (profiles_df['weighting schema'] == test_params[4]) &
                              (profiles_df['matching threshold'] == test_params[5])].iloc[0][self.profiles[profile_index]]
                  
               if cur_profile > opt_profile:
                    print(test_params)
                    cur_params[elem] = test_params[elem]
                    opt_profile = cur_profile
             #want minimal profile value
             else:
               if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                  test_params[elem] = round(self.ranges[elem][0],5)
               else:
                  test_params[elem] = round(self.ranges[elem][-1],5)
               
               cur_profile = profiles_df.loc[(profiles_df['translation'] == test_params[0]) &
                                (profiles_df['q size'] == test_params[1]) & 
                                (profiles_df['pf'] == test_params[2]) & 
                                (profiles_df['block clean threshold'] == test_params[3]) &
                                (profiles_df['weighting schema'] == test_params[4]) &
                                (profiles_df['matching threshold'] == test_params[5])].iloc[0][self.profiles[profile_index]]
          
               if cur_profile < opt_profile:
                    print(test_params)
                    cur_params[elem] = test_params[elem]
                    opt_profile = cur_profile
             
             cur_f = profiles_df.loc[(profiles_df['translation'] == cur_params[0]) &
                                (profiles_df['q size'] == cur_params[1]) & 
                                (profiles_df['pf'] == cur_params[2]) & 
                                (profiles_df['block clean threshold'] == cur_params[3]) &
                                (profiles_df['weighting schema'] == cur_params[4]) &
                                (profiles_df['matching threshold'] == cur_params[5])].iloc[0]['f-score']

             if tuple(cur_params) not in seen:
                   self.rank_iter += 1
                   seen.add(tuple(cur_params))
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4], cur_params[5], cur_f))
             if cur_f >= opt_f:
                opt_f = cur_f
                opt_params = tuple(cur_params)
             #break
          #need to sort
          elif(opt_index >= 1): 
            print("case two")
            map = {}
            iter_list = []
            for key in range(6):
               iter_list.append(self.ranges[key])
            for param_lst in product(*iter_list):
               rparam_lst = (param_lst[0], param_lst[1], param_lst[2],round(param_lst[3], 5), param_lst[4], round(param_lst[5], 5)) #tuple that fixes rounding errors in param_lst
               '''
               map[rparam_lst] = profiles_df.loc[(profiles_df['q size'] == rparam_lst[0]) & 
                                (profiles_df['pf'] == rparam_lst[1]) & 
                                (profiles_df['block clean threshold'] == rparam_lst[2]) &
                                (profiles_df['weighting schema'] == rparam_lst[3]) &
                                (profiles_df['matching threshold'] == rparam_lst[4])].iloc[0][self.profiles[profile_index]]
               '''
               map[rparam_lst] = sum([x*y for x,y in zip(self.parameter_coefs[self.profiles[profile_index]], rparam_lst)])
            #sort descending
            if self.profile_coefs[profile_index] > 0:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))
              sorted_params.reverse()
            #sort ascending
            else:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))

            #print(sorted_params)
            cur_params = sorted_params[opt_index][0]
            cur_f = profiles_df.loc[(profiles_df['translation'] == cur_params[0]) &
                                (profiles_df['q size'] == cur_params[1]) &
                                (profiles_df['pf'] == cur_params[2]) & 
                                (profiles_df['block clean threshold'] == cur_params[3]) &
                                (profiles_df['weighting schema'] == cur_params[4]) &
                                (profiles_df['matching threshold'] == cur_params[5])].iloc[0]['f-score']
            if tuple(cur_params) not in seen:
                   self.rank_iter += 1
                   seen.add(cur_params)
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4], cur_params[5], cur_f))
            if cur_f >= opt_f:
              opt_f = cur_f
              opt_params = cur_params
          #need to use sorted
          else:
            # print("case three")
            # print(idx)
            # print(opt_index)
            # print(len(sorted_params))
            cur_params = sorted_params[opt_index][0]
            cur_f = profiles_df.loc[(profiles_df['translation'] == cur_params[0]) &
                                (profiles_df['q size'] == cur_params[1]) & 
                                (profiles_df['pf'] == cur_params[2]) & 
                                (profiles_df['block clean threshold'] == cur_params[3]) &
                                (profiles_df['weighting schema'] == cur_params[4]) &
                                (profiles_df['matching threshold'] == cur_params[5])].iloc[0]['f-score']
            if tuple(cur_params) not in seen:
                   self.rank_iter += 1
                   seen.add(cur_params)
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4], cur_params[5], cur_f))
            if cur_f >= opt_f:
              opt_f = cur_f
              opt_params = cur_params

          if opt_f >= f_goal:
             #done
             self.rank_f = opt_f
             return opt_params
        opt_index += 1 
       # indicates f_goal is not achievable
       print("failure")
       self.rank_f = opt_f
       return opt_params
    
    def profile_memoization_randomization(self, init_params, f_goal, count):
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       profiles_df = pd.read_csv('ERmetrics/failing_profiles1.csv')
       self.trace_data = []

       opt_f = []
       opt_params = []
       
       opt_f.append(profiles_df.loc[(profiles_df['q size'] == init_params[0]) & 
                                (profiles_df['pf'] == init_params[1]) & 
                                (profiles_df['block clean threshold'] == init_params[2]) &
                                (profiles_df['weighting schema'] == init_params[3]) &
                                (profiles_df['matching threshold'] == init_params[4])].iloc[0]['f-score'])
       opt_params.append(tuple(init_params))

       iter_lst = []
       for i in range(len(self.ranges)):
          iter_lst.append(self.ranges[i])

       for i in range(count):
          cur_params = tuple(random.choice(list(product(*iter_lst))))
          rcur_params = (cur_params[0], cur_params[1],round(cur_params[2], 5), cur_params[3], round(cur_params[4], 5))
          opt_params.append(rcur_params)
          opt_f.append(profiles_df.loc[(profiles_df['q size'] == rcur_params[0]) & 
                                (profiles_df['pf'] == rcur_params[1]) & 
                                (profiles_df['block clean threshold'] == rcur_params[2]) &
                                (profiles_df['weighting schema'] == rcur_params[3]) &
                                (profiles_df['matching threshold'] == rcur_params[4])].iloc[0]['f-score'])

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])
      
       opt_index = 0 #represents opt_index th best set of parameters to take from most important profile, no fallback when opt_index = 0
       sorted_params = [] #represents parameters sorted by most important profile
       
       while(opt_index < search_size):
        for idx, profile_index in enumerate(self.profile_ranking):
          opt_profile = [None] * (count + 1)
          for i in range(count + 1):
            opt_profile[i] = profiles_df.loc[(profiles_df['q size'] == opt_params[i][0]) & 
                                 (profiles_df['pf'] == opt_params[i][1]) & 
                                 (profiles_df['block clean threshold'] == opt_params[i][2]) &
                                 (profiles_df['weighting schema'] == opt_params[i][3]) &
                                 (profiles_df['matching threshold'] == opt_params[i][4])].iloc[0][self.profiles[profile_index]]

          if(opt_index == 0 or idx != 0):
           for elem in self.ranking[self.profiles[profile_index]]:
             for i in range(count + 1):
               cur_params = list(opt_params[i])
               test_params = cur_params.copy()
               #want maximal profile value
               if self.profile_coefs[profile_index] > 0:
                  if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                     test_params[elem] = round(self.ranges[elem][-1],5)
                  else:
                     test_params[elem] = round(self.ranges[elem][0],5)
                  
                  cur_profile = profiles_df.loc[(profiles_df['q size'] == test_params[0]) & 
                                 (profiles_df['pf'] == test_params[1]) & 
                                 (profiles_df['block clean threshold'] == test_params[2]) &
                                 (profiles_df['weighting schema'] == test_params[3]) &
                                 (profiles_df['matching threshold'] == test_params[4])].iloc[0][self.profiles[profile_index]]
                     
                  if cur_profile > opt_profile[i]:
                     cur_params[elem] = test_params[elem]
                     opt_profile[i] = cur_profile
               #want minimal profile value
               else:
                  if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                     test_params[elem] = round(self.ranges[elem][0],5)
                  else:
                     test_params[elem] = round(self.ranges[elem][-1],5)
                  
                  cur_profile = profiles_df.loc[(profiles_df['q size'] == test_params[0]) & 
                                 (profiles_df['pf'] == test_params[1]) & 
                                 (profiles_df['block clean threshold'] == test_params[2]) &
                                 (profiles_df['weighting schema'] == test_params[3]) &
                                 (profiles_df['matching threshold'] == test_params[4])].iloc[0][self.profiles[profile_index]]
            
                  if cur_profile < opt_profile[i]:
                     cur_params[elem] = test_params[elem]
                     opt_profile[i] = cur_profile
               
               cur_f = profiles_df.loc[(profiles_df['q size'] == cur_params[0]) & 
                                 (profiles_df['pf'] == cur_params[1]) & 
                                 (profiles_df['block clean threshold'] == cur_params[2]) &
                                 (profiles_df['weighting schema'] == cur_params[3]) &
                                 (profiles_df['matching threshold'] == cur_params[4])].iloc[0]['f-score']

               if tuple(cur_params) not in seen:
                     self.rank_iter += 1
                     seen.add(tuple(cur_params))
                     self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4], cur_f))
               if cur_f >= f_goal:
                self.rank_f = cur_f
                return tuple(cur_params) # early exit
               elif cur_f >= opt_f[i]:
                  opt_f[i] = cur_f
                  opt_params[i] = tuple(cur_params)
          #need to sort
          elif(opt_index == 1): 
            #print("case two")
            map = {}
            iter_list = []
            for key in range(5):
               iter_list.append(self.ranges[key])
            for param_lst in product(*iter_list):
               rparam_lst = (param_lst[0], param_lst[1],round(param_lst[2], 5), param_lst[3], round(param_lst[4], 5)) #tuple that fixes rounding errors in param_lst
               map[rparam_lst] = profiles_df.loc[(profiles_df['q size'] == rparam_lst[0]) & 
                                (profiles_df['pf'] == rparam_lst[1]) & 
                                (profiles_df['block clean threshold'] == rparam_lst[2]) &
                                (profiles_df['weighting schema'] == rparam_lst[3]) &
                                (profiles_df['matching threshold'] == rparam_lst[4])].iloc[0][self.profiles[profile_index]]
            #sort descending
            if self.profile_coefs[profile_index] > 0:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))
              sorted_params.reverse()
            #sort ascending
            else:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))

            #print(sorted_params)
            cur_params = sorted_params[opt_index][0]
            cur_f = profiles_df.loc[(profiles_df['q size'] == cur_params[0]) &
                                (profiles_df['pf'] == cur_params[1]) & 
                                (profiles_df['block clean threshold'] == cur_params[2]) &
                                (profiles_df['weighting schema'] == cur_params[3]) &
                                (profiles_df['matching threshold'] == cur_params[4])].iloc[0]['f-score']
            if cur_params not in seen:
                   self.rank_iter += 1
                   seen.add(cur_params)
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4], cur_f))
            
            if cur_f >= max(opt_f):
              opt_f[0] = cur_f
              opt_params[0] = cur_params
          #need to use sorted
          else:
            # print("case three")
            # print(idx)
            # print(opt_index)
            # print(len(sorted_params))
            cur_params = sorted_params[opt_index][0]
            cur_f = profiles_df.loc[(profiles_df['q size'] == cur_params[0]) & 
                                (profiles_df['pf'] == cur_params[1]) & 
                                (profiles_df['block clean threshold'] == cur_params[2]) &
                                (profiles_df['weighting schema'] == cur_params[3]) &
                                (profiles_df['matching threshold'] == cur_params[4])].iloc[0]['f-score']
            if cur_params not in seen:
                   self.rank_iter += 1
                   seen.add(cur_params)
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_params[4], cur_f))
            if cur_f >= max(opt_f):
              opt_f[0] = cur_f
              opt_params[0] = cur_params

          if max(opt_f) >= f_goal:
             #done
             self.rank_f = max(opt_f)
             for i in range(count + 1):
                if round(opt_f[i],5) == round(max(opt_f), 5):
                   return opt_params[i]
        opt_index += 1 
       # indicates f_goal is not achievable
       print("failure")
       self.rank_f = max(opt_f)
       for i in range(count + 1):
                if round(opt_f[i],5) == round(max(opt_f), 5):
                   return opt_params[i]
    
    def grid_search(self, f_goal, iterations, trace_value):
       self.gs_idistr = []
       self.gs_fdistr = []
       profiles_df = pd.read_csv('ERmetrics/failing_profiles1.csv')
       iter_lst = []
       for i in range(5):
          iter_lst.append(self.ranges[i])
       
       for i in range(iterations):
         gs_iter = 0
         gs_f = 0
         cur_order = list(product(*iter_lst))
         random.shuffle(cur_order)
         temp_trace = []
         for elem in cur_order:
            cur_f = profiles_df.loc[(profiles_df['q size'] == elem[0]) &
                                (profiles_df['pf'] == elem[1]) & 
                                (profiles_df['block clean threshold'] == round(elem[2],5)) &
                                (profiles_df['weighting schema'] == elem[3]) &
                                (profiles_df['matching threshold'] == round(elem[4],5))].iloc[0]['f-score']
            temp_trace.append((elem[0], elem[1], round(elem[2], 5), elem[3], round(elem[4], 5), cur_f))
            gs_iter += 1
            if cur_f >= f_goal:
               gs_f = cur_f
               cur_params = []
               cur_params.append(elem[0]) 
               cur_params.append(round(elem[1],5))
               cur_params.append(elem[2])
               cur_params.append(round(elem[3],5))
               if(len(temp_trace) == trace_value):
                  self.trace_data = temp_trace
               self.gs_fdistr.append(gs_f)
               self.gs_idistr.append(gs_iter)
               break
               #return cur_params

    def pipeline(self, params, passing, profile_flag, translation = False):
        pf = False if params[1] == 0 else True
        BlCl = BlockCleaning(pf, params[2])
        CoCl = ComparisonCleaning(params[3])
        if translation:
         assert passing == False
         pairs = CoCl.generate_pairs(BlCl.clean_blocks(self.failing_translated_blocks[params[0]]))
        else:
         pairs = CoCl.generate_pairs(BlCl.clean_blocks(self.blocks[params[0]])) if passing else CoCl.generate_pairs(BlCl.clean_blocks(self.failing_blocks[params[0]]))
        stop_cnt = -1
        av_sim = -1
        sim_q1 = -1
        sim_q2 = -1
        sim_q3 = -1
        #sample_f = -1
        f_score = -1
        if len(pairs) > 0:
          if profile_flag:
            ab_profiles = self.passing_profile.generate_abprofiles(pairs, params[4]) if passing else self.failing_profile.generate_abprofiles(pairs, params[4])
            stop_cnt = round(ab_profiles['stopcnt_title'], 5)
            av_sim = round(ab_profiles['avsim_title'], 5)
            sim_q1 = round(ab_profiles['sim_title_q1'], 5)
            sim_q2 = round(ab_profiles['sim_title_q2'], 5)
            sim_q3 = round(ab_profiles['sim_title_q3'], 5)
            #sample_f = round(ab_profiles['f-score'], 5)
          else:
            matching_thres = params[4]
            (tp,fp,tn,fn) = self.Jm_pass[matching_thres].pair_matching(pairs,[self.p1,self.p2],self.gt1_list) if passing else self.Jm_fail[matching_thres].pair_matching(pairs,[self.t1,self.t2],self.gt2_list)
            #print((tp, fp, tn, fn))
            if tp == 0:
              return (-1,-1,-1,-1,-1,0)
            fn = len(self.gt1_list) - tp if passing else len(self.gt2_list) - tp
            cur_p = round(tp / (tp + fp), 5)
            cur_r = round(tp / (tp + fn), 5)
            f_score = round((2 * cur_p * cur_r) / (cur_p + cur_r), 5)
        else:
           return (0,0,0,0,0,0)
        #return (stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, sample_f, f_score)
        return (stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, f_score)
    
    def memoize_all_pairs(self):
      if not(os.path.exists('AmazonData/metrics/failing_similarities.csv')):
        similarity_data = []
        Jm = Matching(0)
        for _, row1 in self.t1.iterrows():
            for _, row2 in self.t2.iterrows():
              avg_sim = Jm.get_sim(row1['title'], row2['title']) + Jm.get_sim(row1['brand'], row2['brand']) + Jm.get_sim(row1['categories'], row2['categories'])
              similarity_data.append((row1['asin'], row2['asin'], avg_sim))
        similarity_df = pd.DataFrame.from_records(similarity_data, columns=['asin1', 'asin2', 'avg_sim'])
        similarity_df.to_csv('AmazonData/metrics/failing_similarities.csv')
      if not(os.path.exists('AmazonData/metrics/passing_similarities.csv')):
        similarity_data = []
        Jm = Matching(0)
        for _, row1 in self.p1.iterrows():
            for _, row2 in self.p2.iterrows():
              avg_sim = Jm.get_sim(row1['title'], row2['title']) + Jm.get_sim(row1['brand'], row2['brand']) + Jm.get_sim(row1['categories'], row2['categories'])
              similarity_data.append((row1['asin'], row2['asin'], avg_sim))
        similarity_df = pd.DataFrame.from_records(similarity_data, columns=['asin1', 'asin2', 'avg_sim'])
        similarity_df.to_csv('AmazonData/metrics/passing_similarities.csv')
             

if __name__ == "__main__":
    p = OptimizingParam()
    # p.memoize_all_pairs()
    # print(p.pipeline([0,0,0.35,2,0.65], True, True))
    # print(p.pipeline([0,0,0.35,2,0.65], False, True))
    # print("end testing")
    p.init_profile_searchspace()
    # p.ranking()
    # print(len(p.blocks[0][0]))
    # print(len(p.blocks[0][1]))
    # print(len(p.failing_blocks[0][0]))
    # print(len(p.failing_blocks[0][1]))
    # p.ranking()
    p.rank_profiles()
    p.rank_parameter()
    p.failing_rank_profiles()
    p.failing_rank_parameters()
    # print(p.pipeline([0,0,0.35,2,0.65], True)) #0.95105
    # print(p.pipeline([0,0,0.35,2,0.65], False)) #0.3637 0.81551
    #print(p.pipeline([5,0,0.35,2,0.65], True)) #0.94778
    #print(p.pipeline([5,0,0.35,2,0.65], False)) #0.3655 0.8183 0.92857
    print("optimizing")
    print("f-goal of 0.85")
    # opt_params = p.optimize([0,0,0.35,2,0.65], 0.90)
    # print(opt_params)
    # print(p.rank_iter)
    # print(p.rank_f)
    # [0, 5, 1, 0.05, 3, 0.65] with 0.97769 row 334
    opt_params = p.profile_memoization_optimize([0, 5, 1, 0.05, 3, 0.65], 0.85) #22 0.9311 [4,0,0.35,1,0.65]
    print(opt_params)
    print(p.rank_iter)
    print(p.rank_f)
    # p.grid_search(0.90, 100, 4)
    # gs_idistr = p.gs_idistr
    # gs_fdistr = p.gs_fdistr
    # g_iquartiles = np.percentile(gs_idistr, [25, 50, 75], interpolation='midpoint')
    # g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75], interpolation='midpoint')
    # print(g_iquartiles)
    # print(g_fquartiles)

    print("f-goal of 0.90")
    print("default")
    # opt_params = p.optimize([0,0,0.35,2,0.65], 0.90)
    # print(opt_params)
    # print(p.rank_iter)
    # print(p.rank_f)
    opt_params = p.profile_memoization_optimize([0, 5, 1, 0.05, 3, 0.65], 0.90) #22 0.9311 [4,0,0.35,1,0.65]
    print(opt_params)
    print(p.rank_iter)
    print(p.rank_f)
    # p.grid_search(0.93, 100, 6)
    # gs_idistr = p.gs_idistr
    # gs_fdistr = p.gs_fdistr
    # g_iquartiles = np.percentile(gs_idistr, [25, 50, 75], interpolation='midpoint')
    # g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75], interpolation='midpoint')
    # print(g_iquartiles)
    # print(g_fquartiles)

   #  data_df = pd.read_csv('ERmetrics/failing_profiles1.csv')
   #  data = data_df[['f-score']].values
   #  plt.hist(data, color='lightgreen', ec='black', bins=[x / 40.0 for x in range(41)])
   #  plt.show()
    
    profiles_df = pd.DataFrame.from_records(p.trace_data, columns=['translation', 'q size', 'pf','block clean threshold', 'weighting schema', 'matching threshold', 'cur_f'])
    if not(os.path.exists('AmazonData/metrics/profilefailingtrace.csv')):
         profiles_df.to_csv('AmazonData/metrics/profilefailingtrace.csv')

    # print("grid search")
    # p.grid_search(0.9, 10)
    # gs_idistr = p.gs_idistr
    # gs_fdistr = p.gs_fdistr
    # g_iquartiles = np.percentile(gs_idistr, [25, 50, 75], interpolation='midpoint')
    # g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75], interpolation='midpoint')
    # print(round(g_iquartiles[0], 5))
    # print(round(g_iquartiles[1], 5))
    # print(round(g_iquartiles[2], 5))
    # print(round(g_fquartiles[0], 5))
    # print(round(g_fquartiles[1], 5))
    # print(round(g_fquartiles[2], 5))

    # Grid search comparisons
    # 4.5
    # 6.0
    # 13.0
    # 0.91116
    # 0.91852
    # 0.92789