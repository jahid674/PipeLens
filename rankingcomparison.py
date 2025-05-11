from naiveparameterselection_final import Param
from optimizedprofileparameterselection_new import ProfileParam
import sys
import os
import numpy as np
import pandas as pd

class Projection:
    def __init__(self, fr = 1):
      self.p = Param()
      self.p.init_searchspace()
      self.p.ranking(fr)

      self.pr = ProfileParam()
      self.pr.init_searchspace()
      self.pr.rank_profiles(fr)
      self.pr.rank_parameter()
      print(self.pr.profile_coefs)

    def ranking(self):
      parameter_coefs = [0] * 4
      for pr_idx, pr_coef in enumerate(self.pr.profile_coefs):
          print(self.pr.parameter_coefs[self.pr.profiles[pr_idx]])
          for param_idx, param_coef in enumerate(self.pr.parameter_coefs[self.pr.profiles[pr_idx]]):
              parameter_coefs[param_idx] += pr_coef * param_coef
      print(parameter_coefs)
      print(np.argsort(np.abs(parameter_coefs))[::-1])
      print(self.p.coefs)
      print(self.p.rank)
      self.coefs = parameter_coefs
      self.rank = np.argsort(np.abs(parameter_coefs))[::-1]

if __name__ == "__main__":
  proj = Projection()
  proj.ranking()

  p = Param()
  p.coefs = proj.coefs
  p.rank = proj.rank
  
  f_goals = [0.8, 0.9, 0.95, 0.98, 0.99]
  profile_data = []
  f = sys.stdout
  if not(os.path.exists('ERmetrics/1dprojectionrankingstats.txt')):
    f = open('ERMetrics/1dprojectionrankingstats.txt', 'w')
  f.write("naive parameter projection ranking algorithm statistics")
  f.write('\n')
  for f_goal in f_goals:
    rank_idistr = []
    rank_fdistr = []
    p.grid_search(f_goal, 100)
    gs_idistr = p.gs_idistr
    gs_fdistr = p.gs_fdistr
    g_iquartiles = np.percentile(gs_idistr, [25, 50, 75], interpolation='midpoint')
    g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75], interpolation='midpoint')
    failures = 0
    for param1 in p.ranges[0]:
        for param2 in p.ranges[1]:
            for param3 in p.ranges[2]:
              for param4 in p.ranges[3]:
                  p.optimize([param1, round(param2, 5), param3, round(param4, 5)], f_goal)
                  if p.rank_iter != -1:
                    rank_idistr.append(p.rank_iter)
                    rank_fdistr.append(p.rank_f)
                  else:
                    failures += 1
                  profile_data.append((param1, round(param2, 5), param3, round(param4, 5), f_goal, p.rank_iter, p.rank_f,round(g_iquartiles[1],5), round(g_fquartiles[1],5)))
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
  profiles_df = pd.DataFrame.from_records(profile_data, columns=['init pf','init block clean threshold', 'init weighting schema', 'init matching threshold', 'f-score goal', 'rank iterations', 'rank f-score', 'grid search iterations', 'grid search f-score'])
  if not(os.path.exists('ERmetrics/rankvsgsprojection.csv')):
    profiles_df.to_csv('ERmetrics/rankvsgsprojection.csv')


