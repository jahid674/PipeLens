import pandas as pd
import numpy as np
from workingerpipeline import BlockBuilding, BlockCleaning, ComparisonCleaning, Matching
from regression import Regression
import os
import operator
import sys
import random
from modules.profiling.profile import Profile
from itertools import product

random.seed(0)

class ProfileParam:
    def __init__(self):
        print("Parameter finding")

        self.profile = Profile()
        self.profile.random_sample()
        #datasets to apply record linkage
        self.t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
        self.t2=pd.read_csv('DBLP-ACM/ACM.csv')

        self.t1sample = pd.read_csv('DBLP-ACM/DBLP2_Sample.csv',encoding="latin-1")
        self.t2sample = pd.read_csv('DBLP-ACM/ACM_Sample.csv')

        self.profile.generate_bbprofiles([self.t1sample, self.t2sample], ['title'])

        #ground truth
        gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
        self.gt_list = gt.values.tolist()

        self.bstep_size = 9
        self.mstep_size = 9

        self.ranges = {}
        self.ranges[0] = [0,1]
        self.ranges[1] = np.linspace(0.05, 0.95,self.bstep_size).tolist()
        self.ranges[2] = list(range(4))
        self.ranges[3] = np.linspace(0.05, 0.95,self.mstep_size).tolist()

        q_size = 0
        BuBl = BlockBuilding(q_size)
        BuBl_Sample = BlockBuilding(q_size)
        self.blocks = BuBl.create_blocks_from_dataframe([self.t1,self.t2],['title'])
        self.sample_blocks = BuBl_Sample.create_blocks_from_dataframe([self.t1sample, self.t2sample], ['title'])
    
    def init_searchspace(self):
      if not(os.path.exists('ERmetrics/profile_parameter_ranking3.csv')):
        profile_data = []
        for i in range(2):
            for j in self.ranges[1]:
                block_clean_thres = round(j,5)
                for k in range(4):
                    for l in self.ranges[3]:
                        matching_thres = round(l,5)
                        params = [i, block_clean_thres, k, matching_thres]
                        (stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, sample_f, _) = self.pipeline(params, True)
                        (_, _, _, _, _, _, cur_f) = self.pipeline(params, False)
                        profile_data.append((i, block_clean_thres, k, matching_thres, stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, sample_f, cur_f))

        profiles_df = pd.DataFrame.from_records(profile_data, columns=['pf','block clean threshold', 'weighting schema', 'matching threshold', 'stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3', 'sample f-score', 'f-score'])
        profiles_df.to_csv('ERmetrics/profile_parameter_ranking3.csv')

    def rank_profiles(self, fr = 1.0):
      if not(os.path.exists('ERmetrics/profile_ranking_dataset2.csv')):
        profile_data = []
        for i in range(2):
            for j in np.linspace(0.05, 0.95, 3).tolist():
                block_clean_thres = round(j,5)
                for k in range(4):
                    for l in np.linspace(0.05, 0.95, 3).tolist():
                        matching_thres = round(l,5)
                        params = [i, block_clean_thres, k, matching_thres]
                        (stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, sample_f,  _) = self.pipeline(params, True)
                        (_, _, _, _, _, _, cur_f) = self.pipeline(params, False)
                        profile_data.append((i, block_clean_thres, k, matching_thres, stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, sample_f, cur_f))
       
        profiles_df = pd.DataFrame.from_records(profile_data, columns=['pf','block clean threshold', 'weighting schema', 'matching threshold', 'stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3', 'sample f-score', 'f-score'])
        profiles_df.to_csv('ERmetrics/profile_ranking_dataset2.csv')
        
      self.historical_df = pd.read_csv('ERmetrics/profile_ranking_dataset2.csv').sample(frac = fr, random_state = 1)
      print("beginning profile regression generation")
      X = self.historical_df[['stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3', 'sample f-score']]
      y = self.historical_df['f-score']
      reg = Regression()
      model = reg.generate_regression(X, y)
      coefs = model.coef_
      print(coefs)
      print(model.intercept_)
      self.profile_ranking = np.argsort(np.abs(coefs))[::-1]
      self.profile_coefs = coefs
      print(self.profile_ranking)

    def rank_parameter(self):
      self.ranking = {}
      self.parameter_coefs = {}
      self.intercepts={}
      print("beginning parameter regression generation")
      self.profiles = ['stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3', 'sample f-score']
      for elem in self.profiles:
        X = self.historical_df[['pf', 'block clean threshold', 'weighting schema', 'matching threshold']]
        y = self.historical_df[elem]
        reg = Regression()
        model = reg.generate_regression(X, y)
        coefs = model.coef_
        print(coefs)
        print(model.intercept_)
        self.ranking[elem] = np.argsort(np.abs(coefs))[::-1]
        self.parameter_coefs[elem] = coefs
        self.intercepts[elem] = model.intercept_
        print(elem + ':')
        print(self.ranking[elem])

    # if f_goal unachievable, failure is printed
    def optimize(self, init_params, f_goal):
       self.rank_iter = 0
       seen = set()
       self.profile_iter = {}
       self.profile_set = {}
       for prof in self.profiles:
          self.profile_iter[prof] = 0
          self.profile_set[prof] = set()
       self.rank_f = 0
       profiles_df = pd.read_csv('ERmetrics/profile_parameter_ranking2.csv') if self.bstep_size == 8 else pd.read_csv('ERmetrics/profile_parameter_ranking3.csv')
       opt_f = profiles_df.loc[(profiles_df['pf'] == init_params[0]) & 
                                (profiles_df['block clean threshold'] == init_params[1]) &
                                (profiles_df['weighting schema'] == init_params[2]) &
                                (profiles_df['matching threshold'] == init_params[3])].iloc[0]['f-score']
       #print(opt_f)
       opt_params = tuple(init_params)

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])
      
       opt_index = 0 #represents opt_index th best set of parameters to take from most important profile, no fallback when opt_index = 0
       sorted_params = [] #represents parameters sorted by most important profile
       #print ("profile iter is ",self.profile_iter)

       while(opt_index < search_size):
        for idx, profile_index in enumerate(self.profile_ranking):
          opt_profile = profiles_df.loc[(profiles_df['pf'] == opt_params[0]) & 
                                (profiles_df['block clean threshold'] == opt_params[1]) &
                                (profiles_df['weighting schema'] == opt_params[2]) &
                                (profiles_df['matching threshold'] == opt_params[3])].iloc[0][self.profiles[profile_index]]
          if(opt_index == 0 or idx != 0):
           for elem in self.ranking[self.profiles[profile_index]]:
             cur_params = list(opt_params)
            #  test_params = cur_params.copy()
             #want maximal profile value
             #print ("profile iter is0 ",self.profile_iter)
             if self.profile_coefs[profile_index] > 0:
               if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                  cur_params[elem] = round(self.ranges[elem][-1],5)
               else:
                  cur_params[elem] = round(self.ranges[elem][0],5)

              #  #print ("profile iter is1 ",self.profile_iter)
              #  cur_profile = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
              #                   (profiles_df['block clean threshold'] == test_params[1]) &
              #                   (profiles_df['weighting schema'] == test_params[2]) &
              #                   (profiles_df['matching threshold'] == test_params[3])].iloc[0][self.profiles[profile_index]]
              #  #print ("profile iter is2 ",self.profile_iter)
              #  if tuple(cur_params) not in self.profile_set[self.profiles[profile_index]]:
              #      self.profile_iter[self.profiles[profile_index]] += 1
              #      self.profile_set[self.profiles[profile_index]].add(tuple(cur_params))
              #  #print ("profile iter is3 ",self.profile_iter)
              #  #print (profiles_df)
              #  if cur_profile > opt_profile:
              #       cur_params[elem] = test_params[elem]
              #       opt_profile = cur_profile
               
             #want minimal profile value
             else:
               if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                  cur_params[elem] = round(self.ranges[elem][0],5)
               else:
                  cur_params[elem] = round(self.ranges[elem][-1],5)
               
              #  cur_profile = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
              #                   (profiles_df['block clean threshold'] == test_params[1]) &
              #                   (profiles_df['weighting schema'] == test_params[2]) &
              #                   (profiles_df['matching threshold'] == test_params[3])].iloc[0][self.profiles[profile_index]]
              #  if cur_profile < opt_profile:
              #       cur_params[elem] = test_params[elem]
              #       opt_profile = cur_profile
               
             cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
             if tuple(cur_params) not in seen:
                   self.rank_iter += 1
                   seen.add(tuple(cur_params))
             if cur_f >= opt_f:
                opt_f = cur_f
                opt_params = tuple(cur_params)
             #break
            #Added by me
          #need to sort
          elif(opt_index >= 1): 
            #print("case two")
            map = {}
            iter_list = []
            for key in range(4):
               iter_list.append(self.ranges[key])
            for param_lst in product(*iter_list):
               rparam_lst = (param_lst[0],round(param_lst[1], 5), param_lst[2], round(param_lst[3], 5)) #tuple that fixes rounding errors in param_lst
               #print (self.profile_coefs,self.profiles,profile_index,self.parameter_coefs
               iter = 0
               sumval=0
               r = profiles_df.loc[(profiles_df['pf'] == rparam_lst[0]) & 
                                (profiles_df['block clean threshold'] == rparam_lst[1]) &
                                (profiles_df['weighting schema'] == rparam_lst[2]) &
                                (profiles_df['matching threshold'] == rparam_lst[3])].iloc[0]
               #while iter<len(self.profiles):
               #   sumval+=self.profile_coefs[iter]*r[self.profiles[iter]]
               #   iter+=1
               #print (sumval, self.profile_coefs)
               #MAJOR CHANGE
               map[rparam_lst] =  sum([x*y for x,y in zip( self.parameter_coefs[self.profiles[profile_index]],rparam_lst)])
               #print (profile_index,self.profile_coefs)
              
               #random.random()
               if rparam_lst not in self.profile_set[self.profiles[profile_index]]:
                   #self.profile_iter[self.profiles[profile_index]] += 1
                   self.profile_set[self.profiles[profile_index]].add(rparam_lst)
            #sort descending
            if self.profile_coefs[profile_index] > 0:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))
              sorted_params.reverse()
            #sort ascending
            else:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))

            #print(sorted_params)
            cur_params = sorted_params[opt_index][0]
            cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
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
            cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
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
    
    def optimize_randomization(self, init_params, f_goal, count):
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       profiles_df = pd.read_csv('ERmetrics/profile_parameter_ranking2.csv')

       opt_f = []
       opt_params = []

       opt_f.append(profiles_df.loc[(profiles_df['pf'] == init_params[0]) & 
                                (profiles_df['block clean threshold'] == init_params[1]) &
                                (profiles_df['weighting schema'] == init_params[2]) &
                                (profiles_df['matching threshold'] == init_params[3])].iloc[0]['f-score'])
       opt_params.append(tuple(init_params))

       iter_lst = []
       for i in range(len(self.ranges)):
          iter_lst.append(self.ranges[i])

       for i in range(count):
          cur_params = tuple(random.choice(list(product(*iter_lst))))
          rcur_params = (cur_params[0],round(cur_params[1], 5), cur_params[2], round(cur_params[3], 5))
          opt_params.append(rcur_params)
          opt_f.append(profiles_df.loc[(profiles_df['pf'] == rcur_params[0]) & 
                                (profiles_df['block clean threshold'] == rcur_params[1]) &
                                (profiles_df['weighting schema'] == rcur_params[2]) &
                                (profiles_df['matching threshold'] == rcur_params[3])].iloc[0]['f-score'])

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])
      
       opt_index = 0 #represents opt_index th best set of parameters to take from most important profile, no fallback when opt_index = 0
       sorted_params = [] #represents parameters sorted by most important profile
       
       while(opt_index < search_size):
        for idx, profile_index in enumerate(self.profile_ranking):
          opt_profile = [None] * (count + 1)
          for i in range(count + 1):
              opt_profile[i] = profiles_df.loc[(profiles_df['pf'] == opt_params[i][0]) & 
                                (profiles_df['block clean threshold'] == opt_params[i][1]) &
                                (profiles_df['weighting schema'] == opt_params[i][2]) &
                                (profiles_df['matching threshold'] == opt_params[i][3])].iloc[0][self.profiles[profile_index]]
          
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
                  cur_profile = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                    (profiles_df['block clean threshold'] == test_params[1]) &
                                    (profiles_df['weighting schema'] == test_params[2]) &
                                    (profiles_df['matching threshold'] == test_params[3])].iloc[0][self.profiles[profile_index]]
                  
                  if cur_profile > opt_profile[i]:
                        cur_params[elem] = test_params[elem]
                        opt_profile[i] = cur_profile
                #want minimal profile value
                else:
                  if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                      test_params[elem] = round(self.ranges[elem][0],5)
                  else:
                      test_params[elem] = round(self.ranges[elem][-1],5)
                  cur_profile = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                    (profiles_df['block clean threshold'] == test_params[1]) &
                                    (profiles_df['weighting schema'] == test_params[2]) &
                                    (profiles_df['matching threshold'] == test_params[3])].iloc[0][self.profiles[profile_index]]
                  if cur_profile < opt_profile[i]:
                        cur_params[elem] = test_params[elem]
                        opt_profile[i] = cur_profile
                
                cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                    (profiles_df['block clean threshold'] == cur_params[1]) &
                                    (profiles_df['weighting schema'] == cur_params[2]) &
                                    (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
                if tuple(cur_params) not in seen:
                      self.rank_iter += 1
                      seen.add(tuple(cur_params))
                      # print((tuple(cur_params)), cur_f)
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
            for key in range(4):
               iter_list.append(self.ranges[key])
            for param_lst in product(*iter_list):
               rparam_lst = (param_lst[0],round(param_lst[1], 5), param_lst[2], round(param_lst[3], 5)) #tuple that fixes rounding errors in param_lst
               map[rparam_lst] = random.random()
               '''profiles_df.loc[(profiles_df['pf'] == rparam_lst[0]) & 
                                (profiles_df['block clean threshold'] == rparam_lst[1]) &
                                (profiles_df['weighting schema'] == rparam_lst[2]) &
                                (profiles_df['matching threshold'] == rparam_lst[3])].iloc[0][self.profiles[profile_index]]
                                '''
            #sort descending
            if self.profile_coefs[profile_index] > 0:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))
              sorted_params.reverse()
            #sort ascending
            else:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))

            #print(sorted_params)
            cur_params = sorted_params[opt_index][0]
            cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
            if cur_params not in seen:
                   self.rank_iter += 1
                   seen.add(cur_params)
                   # print(cur_params)
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
            cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
            if cur_params not in seen:
                   self.rank_iter += 1
                   seen.add(cur_params)
                   # print(cur_params)
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
                
    def optimize_fixed(self, init_params, f_goal):
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       nm_ranking = {}
       for prof in self.profiles:
          nm_ranking[prof] = np.copy(self.ranking[prof])
          nm_ranking[prof] = np.delete(nm_ranking[prof], np.where(nm_ranking[prof] == 3))

       profiles_df = pd.read_csv('ERmetrics/profile_parameter_ranking2.csv')
       opt_f = profiles_df.loc[(profiles_df['pf'] == init_params[0]) & 
                                (profiles_df['block clean threshold'] == init_params[1]) &
                                (profiles_df['weighting schema'] == init_params[2]) &
                                (profiles_df['matching threshold'] == init_params[3])].iloc[0]['f-score']
       #print(opt_f)
       opt_params = tuple(init_params)

       search_size = 1
       for key in self.ranges.keys():
          search_size *= len(self.ranges[key])
      
       opt_index = 0 #represents opt_index th best set of parameters to take from most important profile, no fallback when opt_index = 0
       sorted_params = [] #represents parameters sorted by most important profile
       
       while(opt_index < search_size):
        for idx, profile_index in enumerate(self.profile_ranking):
          opt_profile = profiles_df.loc[(profiles_df['pf'] == opt_params[0]) & 
                                (profiles_df['block clean threshold'] == opt_params[1]) &
                                (profiles_df['weighting schema'] == opt_params[2]) &
                                (profiles_df['matching threshold'] == opt_params[3])].iloc[0][self.profiles[profile_index]]
          if(opt_index == 0 or idx != 0):
           for elem in nm_ranking[self.profiles[profile_index]]:
             cur_params = list(opt_params)
             test_params = cur_params.copy()
             #want maximal profile value
             if self.profile_coefs[profile_index] > 0:
               if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                  test_params[elem] = round(self.ranges[elem][-1],5)
               else:
                  test_params[elem] = round(self.ranges[elem][0],5)
               cur_profile = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                (profiles_df['block clean threshold'] == test_params[1]) &
                                (profiles_df['weighting schema'] == test_params[2]) &
                                (profiles_df['matching threshold'] == test_params[3])].iloc[0][self.profiles[profile_index]]
               if cur_profile > opt_profile:
                    cur_params[elem] = test_params[elem]
                    opt_profile = cur_profile
             #want minimal profile value
             else:
               if self.parameter_coefs[self.profiles[profile_index]][elem] > 0:
                  test_params[elem] = round(self.ranges[elem][0],5)
               else:
                  test_params[elem] = round(self.ranges[elem][-1],5)
               cur_profile = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                (profiles_df['block clean threshold'] == test_params[1]) &
                                (profiles_df['weighting schema'] == test_params[2]) &
                                (profiles_df['matching threshold'] == test_params[3])].iloc[0][self.profiles[profile_index]]
               if cur_profile < opt_profile:
                    cur_params[elem] = test_params[elem]
                    opt_profile = cur_profile
             
             cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
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
            for key in range(3):
               iter_list.append(self.ranges[key])
            for param_lst in product(*iter_list):
               rparam_lst = (param_lst[0],round(param_lst[1], 5), param_lst[2], init_params[3]) #tuple that fixes rounding errors in param_lst
               map[rparam_lst] = profiles_df.loc[(profiles_df['pf'] == rparam_lst[0]) & 
                                (profiles_df['block clean threshold'] == rparam_lst[1]) &
                                (profiles_df['weighting schema'] == rparam_lst[2]) &
                                (profiles_df['matching threshold'] == rparam_lst[3])].iloc[0][self.profiles[profile_index]]
            #sort descending
            if self.profile_coefs[profile_index] > 0:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))
              sorted_params.reverse()
            #sort ascending
            else:
              sorted_params = sorted(map.items(), key=operator.itemgetter(1))

            #print(sorted_params)
            cur_params = sorted_params[opt_index][0]
            cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
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
            cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
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
    
    def grid_search(self, f_goal, iterations):
       self.gs_idistr = []
       self.gs_fdistr = []
       profiles_df = pd.read_csv('ERmetrics/profile_parameter_ranking2.csv') if self.bstep_size == 8 else pd.read_csv('ERmetrics/profile_parameter_ranking3.csv')
       iter_lst = []
       for i in range(4):
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

    def pipeline(self, params, profile_flag):
        pf = False if params[0] == 0 else True
        if profile_flag:
          BlCl = BlockCleaning(pf, params[1])
          CoCl = ComparisonCleaning(params[2])
          pairs = CoCl.generate_pairs(BlCl.clean_blocks(self.sample_blocks))
        else:
          BlCl = BlockCleaning(pf, params[1])
          CoCl = ComparisonCleaning(params[2])
          pairs = CoCl.generate_pairs(BlCl.clean_blocks(self.blocks))
        stop_cnt = -1
        av_sim = -1
        sim_q1 = -1
        sim_q2 = -1
        sim_q3 = -1
        sample_f = -1
        f_score = -1
        if len(pairs) > 0:
          if profile_flag:
            ab_profiles = self.profile.generate_abprofiles(pairs, params[3])
            stop_cnt = round(ab_profiles['stopcnt_title'], 5)
            av_sim = round(ab_profiles['avsim_title'], 5)
            sim_q1 = round(ab_profiles['sim_title_q1'], 5)
            sim_q2 = round(ab_profiles['sim_title_q2'], 5)
            sim_q3 = round(ab_profiles['sim_title_q3'], 5)
            sample_f = round(ab_profiles['f-score'], 5)
          else:
            Jm = Matching(params[3])
            (tp,fp,tn,fn) = Jm.pair_matching(pairs,[self.t1,self.t2],self.gt_list)
            if tp == 0:
              return (-1,-1,-1,-1,-1,-1,0)
            fn = len(self.gt_list) - tp
            cur_p = round(tp / (tp + fp), 5)
            cur_r = round(tp / (tp + fn), 5)
            f_score = round((2 * cur_p * cur_r) / (cur_p + cur_r), 5)
        else:
           return (0,0,0,0,0,0,0)
        return (stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, sample_f, f_score)
    
if __name__ == "__main__":
    p = ProfileParam()
    p.init_searchspace()
    p.rank_profiles()
    p.rank_parameter()

    f_goals = [0.8, 0.9, 0.95, 0.96]#, 0.99]
    profile_data = []
    f = sys.stdout
    if not(os.path.exists('ERmetrics/2drankingstats3.txt')):
      f = open('ERMetrics/2drankingstats3.txt', 'w')
    f.write("profile ranking algorithm statistics")
    f.write('\n')
    print (p.ranges)
    for f_goal in f_goals:
      rank_idistr = []
      rank_fdistr = []
      rank_profiledistr = {}
      for prof in p.profiles:
         rank_profiledistr[prof] = []
      print ("grid search")
      p.grid_search(f_goal, 100)

      print ("**********************************")
      print ("**********************************")
      gs_idistr = p.gs_idistr
      gs_fdistr = p.gs_fdistr
      g_iquartiles = np.percentile(gs_idistr, [25, 50, 75, 100], method='midpoint')
      g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75, 100], method='midpoint')
      failures = 0
      for param1 in p.ranges[0]:
          for param2 in p.ranges[1]:
             for param3 in p.ranges[2]:
                for param4 in p.ranges[3]:
                   p.optimize([param1, round(param2, 5), param3, round(param4, 5)], f_goal)
                   '''if param1==0 and param2>=0.34 and param2<=0.36 and param3==2 and param4>=0.64 and param4<=0.66:
                     print ("THIS is the case")
                     print (p.rank_iter,p.rank_f)
                     fsjakljlk
                   '''
                   if p.rank_iter != -1:
                    rank_idistr.append(p.rank_iter)
                    rank_fdistr.append(p.rank_f)
                    for prof in p.profiles:
                       rank_profiledistr[prof].append(p.profile_iter[prof])
                   else:
                      failures += 1
                   profile_data.append((param1, round(param2, 5), param3, round(param4, 5), f_goal, p.rank_iter, p.rank_f, round(g_iquartiles[1],5), round(g_fquartiles[1],5)))
                   #print (p.rank_iter,f_goal,"*********")
      rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75, 100], interpolation='midpoint')
      rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75, 100], interpolation='midpoint')
      rank_profilequartiles = {}
      for prof in p.profiles:
         rank_profilequartiles[prof] = np.percentile(rank_profiledistr[prof], [25, 50, 75, 100], interpolation='midpoint')
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
      f.write("ranking algorithm iterations q4: " + str(round(rank_iquartiles[3], 5)))
      f.write('\n')
      for prof in p.profiles:
         f.write(prof + " sample iterations q1: " + str(round(rank_profilequartiles[prof][0], 5)))
         f.write('\n')
         f.write(prof + " sample iterations q2: " + str(round(rank_profilequartiles[prof][1], 5)))
         f.write('\n')
         f.write(prof + " sample iterations q3: " + str(round(rank_profilequartiles[prof][2], 5)))
         f.write('\n')
         f.write(prof + " sample iterations q4: " + str(round(rank_profilequartiles[prof][3], 5)))
         f.write('\n')
      print(rank_fquartiles)
      f.write("ranking algorithm f-score q1: " + str(round(rank_fquartiles[0], 5)))
      f.write('\n')
      f.write("ranking algorithm f-score q2: " + str(round(rank_fquartiles[1], 5)))
      f.write('\n')
      f.write("ranking algorithm f-score q3: " + str(round(rank_fquartiles[2], 5)))
      f.write('\n')
      f.write("ranking algorithm f-score q4: " + str(round(rank_fquartiles[3], 5)))
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
      f.write("grid search iterations q4: " + str(round(g_iquartiles[3], 5)))
      print(g_fquartiles)
      f.write("grid search f-score q1: " + str(round(g_fquartiles[0], 5)))
      f.write('\n')
      f.write("grid search f-score q2: " + str(round(g_fquartiles[1], 5)))
      f.write('\n')
      f.write("grid search f-score q3: " + str(round(g_fquartiles[2], 5)))
      f.write('\n')
      
    

    '''
    print ("Testing for a single input")
    p.optimize([0,0,0.35,2,0.65], 0.93)
    print (p.rank_iter,p.rank_f)
    '''
    f.close()
    profiles_df = pd.DataFrame.from_records(profile_data, columns=['init pf','init block clean threshold', 'init weighting schema', 'init matching threshold', 'f-score goal', 'rank iterations', 'rank f-score', 'grid search iterations', 'grid search f-score'])
    if not(os.path.exists('ERmetrics/rankvsgs4.csv')):
      profiles_df.to_csv('ERmetrics/rankvsgs4.csv')

    # print(p.optimize([1,0.725,3,0.95], 0.9))
    # print(p.rank_f)
    # print(p.rank_iter)
   
