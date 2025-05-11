import pandas as pd
import numpy as np
from itertools import product
from naiveparameterselection import Param
from optimizedprofileparameterselection import ProfileParam
from rankingcomparison import Projection
import os
import operator
import random

random.seed(0)

class Trace:
    def __init__(self):
        print("Parameter finding")

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
    
    def set_ranking(self, ranking):
       self.ranking = ranking

    def set_coefs(self, coefs):
       self.coefs = coefs

 # if f_goal is unachievable, failure is printed
    def optimize(self, init_params, f_goal):
       self.trace_data = []
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       profiles_df = pd.read_csv('ERmetrics/parameter_ranking1.csv')
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

       while (iter_size < len(self.ranking)):
          iter_size += 1
          # self.rank_f = 0
          # cur_params = init_params.copy()
          if iter_size == 1:
             for elem in self.ranking:
                if self.coefs[elem] > 0:
                  # first self.k[elem] elems
                  for parameter in self.ranges[elem][:self.k[elem]]:
                     test_params = cur_params.copy()
                     test_params[elem] = round(parameter,5)
                     cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                     if tuple(test_params) not in seen:
                        self.rank_iter += 1
                        seen.add(tuple(test_params))
                        self.trace_data.append((test_params[0], test_params[1], test_params[2], test_params[3], cur_f))
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
                     cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                     if tuple(test_params) not in seen:
                        self.rank_iter += 1
                        seen.add(tuple(test_params))
                        self.trace_data.append((test_params[0], test_params[1], test_params[2], test_params[3], cur_f))
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
            index = 0
            self.trace_data.append((iter_size, iter_size, iter_size, iter_size, iter_size))
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
                  #  print(index)
                  #  print(cur_iterlst)
                  #  print(test_params)
                  cur_f = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                 (profiles_df['block clean threshold'] == test_params[1]) &
                                 (profiles_df['weighting schema'] == test_params[2]) &
                                 (profiles_df['matching threshold'] == test_params[3])].iloc[0]['f-score']
                  #account for hashing
                  if tuple(test_params) not in seen:
                     self.rank_iter += 1
                     seen.add(tuple(test_params))
                     self.trace_data.append((test_params[0], test_params[1], test_params[2], test_params[3], cur_f))
                  if cur_f >= f_goal:
                     for j in range(len(cur_iterlst)):
                        cur_params[self.ranking[index - len(cur_iterlst) + j]] = round(elem[j], 5)
                     self.rank_f = cur_f
                     return cur_params # early exit when f_goal obtained
                  # update opt f for current iteration to improve upon in next iter
                  elif cur_f > opt_f:
                     for j in range(len(cur_iterlst)):
                        cur_params[self.ranking[index - len(cur_iterlst) + j]] = round(elem[j], 5)
                     opt_f = cur_f
       if self.rank_iter == search_size:
          print("failure")
       return cur_params
    
class Trace2:
    def __init__(self):
        print("Parameter finding")

        self.bstep_size = 8
        self.mstep_size = 8

        self.ranges = {}
        self.ranges[0] = [0,1]
        self.ranges[1] = np.linspace(0.05, 0.95,self.bstep_size).tolist()
        self.ranges[2] = list(range(4))
        self.ranges[3] = np.linspace(0.05, 0.95,self.mstep_size).tolist()

    def set_profiles(self, profiles):
       self.profiles = profiles

    def set_ranking(self, ranking):
       self.ranking = ranking

    def set_profilecoefs(self, profile_coefs):
       self.profile_coefs = profile_coefs

    def set_profileranking(self, profile_ranking):
       self.profile_ranking = profile_ranking
    
    def set_parametercoefs(self, parameter_coefs):
       self.parameter_coefs = parameter_coefs

    # if f_goal unachievable, failure is printed
    def optimize(self, init_params, f_goal):
       self.trace_data = []
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       profiles_df = pd.read_csv('ERmetrics/profile_parameter_ranking.csv')
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
          opt_profile = profiles_df.loc[(profiles_df['pf'] == init_params[0]) & 
                                (profiles_df['block clean threshold'] == init_params[1]) &
                                (profiles_df['weighting schema'] == init_params[2]) &
                                (profiles_df['matching threshold'] == init_params[3])].iloc[0][self.profiles[profile_index]]
          if(opt_index == 0 or idx != 0):
           for elem in self.ranking[self.profiles[profile_index]]:
             cur_params = list(opt_params)
             #iterate through params to improve profile
             for param in self.ranges[elem]:
                test_params = cur_params.copy()
                test_params[elem] = round(param,5)
                cur_profile = profiles_df.loc[(profiles_df['pf'] == test_params[0]) & 
                                (profiles_df['block clean threshold'] == test_params[1]) &
                                (profiles_df['weighting schema'] == test_params[2]) &
                                (profiles_df['matching threshold'] == test_params[3])].iloc[0][self.profiles[profile_index]]
                #want minimal vs maximal profile value
                if self.profile_coefs[profile_index] > 0:
                  if cur_profile > opt_profile:
                    cur_params[elem] = round(param,5)
                    opt_profile = cur_profile
                else:
                  if cur_profile < opt_profile:
                    cur_params[elem] = round(param,5)
                    opt_profile = cur_profile
             cur_f = profiles_df.loc[(profiles_df['pf'] == cur_params[0]) & 
                                (profiles_df['block clean threshold'] == cur_params[1]) &
                                (profiles_df['weighting schema'] == cur_params[2]) &
                                (profiles_df['matching threshold'] == cur_params[3])].iloc[0]['f-score']
             if tuple(cur_params) not in seen:
                   self.rank_iter += 1
                   seen.add(tuple(cur_params))
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_f))
             if cur_f >= opt_f:
                opt_f = cur_f
                opt_params = tuple(cur_params)
          #need to sort
          elif(opt_index == 1): 
            #print("case two")
            map = {}
            iter_list = []
            for key in range(4):
               iter_list.append(self.ranges[key])
            for param_lst in product(*iter_list):
               rparam_lst = (param_lst[0],round(param_lst[1], 5), param_lst[2], round(param_lst[3], 5)) #tuple that fixes rounding errors in param_lst
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
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_f))
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
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_f))
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
       self.trace_data = []
       self.rank_iter = 0
       seen = set()
       self.rank_f = 0
       profiles_df = pd.read_csv('ERmetrics/profile_parameter_ranking.csv')

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
                      self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_f))
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
               #print(rparam_lst)
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
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_f))
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
                   self.trace_data.append((cur_params[0], cur_params[1], cur_params[2], cur_params[3], cur_f))
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

if __name__ == "__main__":
    p = Param()
    p.ranking()

    t = Trace()
    t.set_ranking(p.rank)
    t.set_coefs(p.coefs)

    #t.optimize([1,0.95,3,0.95], 0.985)
    t.optimize([0,0.69286,0,0.95], 0.99)
    print(t.rank_iter)

    pp = ProfileParam()
    pp.init_searchspace()
    pp.rank_profiles()
    pp.rank_parameter()

    t2 = Trace2()
    t2.set_profiles(pp.profiles)
    t2.set_ranking(pp.ranking)
    t2.set_profileranking(pp.profile_ranking)
    t2.set_profilecoefs(pp.profile_coefs)
    t2.set_parametercoefs(pp.parameter_coefs)

    t2.optimize_randomization([1,0.95,3,0.95], 0.95, 0)
    print(t2.rank_iter)

    proj = Projection()
    proj.ranking()
    t3 = Trace()
    t3.set_ranking(proj.rank)
    t3.set_coefs(proj.coefs)

    t3.optimize([1,0.95,3,0.95], 0.985)
    print(t3.rank_iter)

    profiles_df = pd.DataFrame.from_records(t.trace_data, columns=['pf','block clean threshold', 'weighting schema', 'matching threshold', 'cur_f'])
    if not(os.path.exists('ERmetrics/1dtracetemp.csv')):
      profiles_df.to_csv('ERmetrics/1dtracetemp.csv')

    profiles2_df = pd.DataFrame.from_records(t2.trace_data, columns=['pf','block clean threshold', 'weighting schema', 'matching threshold', 'cur_f'])
    if not(os.path.exists('ERmetrics/2dtrace.csv')):
      profiles2_df.to_csv('ERmetrics/2dtrace.csv')

    profiles3_df = pd.DataFrame.from_records(t3.trace_data, columns=['pf','block clean threshold', 'weighting schema', 'matching threshold', 'cur_f'])
    if not(os.path.exists('ERmetrics/projtrace.csv')):
      profiles3_df.to_csv('ERmetrics/projtrace.csv')