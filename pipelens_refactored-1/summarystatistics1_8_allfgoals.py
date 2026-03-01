from naiveparameterselection_final import Param
from optimizedprofileparameterselection_new import ProfileParam
from rankingcomparison import Projection
import random
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)

if __name__ == "__main__":
    size = (8,8)
    
    param = Param()
    param.init_searchspace()
    param.ranking()
    param.bstep_size = size[0]
    param.mstep_size = size[1]
    param.ranges[1] = np.linspace(0.05, 0.95,param.bstep_size).tolist()
    param.ranges[3] = np.linspace(0.05, 0.95,param.mstep_size).tolist()

    proj = Projection()
    proj.ranking()
    projparam = Param()
    projparam.coefs = proj.coefs
    projparam.rank = proj.rank

    projparam.bstep_size = size[0]
    projparam.mstep_size = size[1]
    projparam.ranges[1] = np.linspace(0.05, 0.95,projparam.bstep_size).tolist()
    projparam.ranges[3] = np.linspace(0.05, 0.95,projparam.mstep_size).tolist()

    profile = ProfileParam()
    profile.init_searchspace()
    profile.rank_profiles()
    profile.rank_parameter()

    profile.bstep_size = size[0]
    profile.mstep_size = size[1]
    profile.ranges[1] = np.linspace(0.05, 0.95,profile.bstep_size).tolist()
    profile.ranges[3] = np.linspace(0.05, 0.95,profile.mstep_size).tolist()

    f_goals = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96]

    f = sys.stdout
    if not(os.path.exists('ERmetrics/experiment1summary_8.txt')):
      f = open('ERMetrics/experiment1summary_8.txt', 'w')
    f.write("experiment 1 summary statistics")
    f.write('\n')

    profile_data = []
    for f_goal in f_goals:
      param_idistr = []
      param_fdistr = []
      proj_idistr = []
      proj_fdistr = []
      profile_idistr = []
      profile_fdistr = []
      rank_profiledistr = {}
      for prof in profile.profiles:
         rank_profiledistr[prof] = []
      param.grid_search(f_goal, 100)
      gs_idistr = param.gs_idistr
      gs_fdistr = param.gs_fdistr
      g_iquartiles = np.percentile(gs_idistr, [25, 50, 75, 100], interpolation='midpoint')
      g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75, 100], interpolation='midpoint')
      param_failures = 0
      proj_failures = 0
      profile_failures = 0

      for param1 in [0,1]:
          for param2 in np.linspace(0.05, 0.95, size[0]).tolist():
              for param3 in list(range(4)):
                for param4 in np.linspace(0.05, 0.95, size[1]).tolist():
                    param.optimize([param1, round(param2, 5), param3, round(param4, 5)], f_goal)
                    if param.rank_iter != -1:
                      param_idistr.append(param.rank_iter)
                      param_fdistr.append(param.rank_f)
                    else:
                      param_failures += 1

                    projparam.optimize([param1, round(param2, 5), param3, round(param4, 5)], f_goal)
                    if projparam.rank_iter != -1:
                      proj_idistr.append(projparam.rank_iter)
                      proj_fdistr.append(projparam.rank_f)
                    else:
                      proj_failures += 1
                    
                    profile.optimize([param1, round(param2, 5), param3, round(param4, 5)], f_goal)
                    if profile.rank_iter != -1:
                      profile_idistr.append(profile.rank_iter)
                      profile_fdistr.append(profile.rank_f)
                      for prof in profile.profiles:
                       rank_profiledistr[prof].append(profile.profile_iter[prof])
                    else:
                      profile_failures += 1

                    profile_data.append((param1, round(param2, 5), param3, round(param4, 5), f_goal, param.rank_iter, param.rank_f, projparam.rank_iter, projparam.rank_f, profile.rank_iter, profile.rank_f, round(g_iquartiles[1],5), round(g_fquartiles[1],5)))
      param_iquartiles = np.percentile(param_idistr, [25, 50, 75, 100], interpolation='midpoint')
      param_fquartiles = np.percentile(param_fdistr, [25, 50, 75, 100], interpolation='midpoint')
      proj_iquartiles = np.percentile(proj_idistr, [25, 50, 75, 100], interpolation='midpoint')
      proj_fquartiles = np.percentile(proj_fdistr, [25, 50, 75, 100], interpolation='midpoint')
      profile_iquartiles = np.percentile(profile_idistr, [25, 50, 75, 100], interpolation='midpoint')
      profile_fquartiles = np.percentile(profile_fdistr, [25, 50, 75, 100], interpolation='midpoint')
      rank_profilequartiles = {}
      for prof in profile.profiles:
         rank_profilequartiles[prof] = np.percentile(rank_profiledistr[prof], [25, 50, 75, 100], interpolation='midpoint')
      print("f-score goal stats: " + str(f_goal))
      f.write("stats for f-score goal: " + str(f_goal))
      f.write('\n')
      f.write("parameter ranking algorithm:")
      f.write('\n')
      print(param_iquartiles)
      f.write("parameter ranking algorithm iterations q1: " + str(round(param_iquartiles[0], 5)))
      f.write('\n')
      f.write("parameter ranking algorithm iterations q2: " + str(round(param_iquartiles[1], 5)))
      f.write('\n')
      f.write("parameter ranking algorithm iterations q3: " + str(round(param_iquartiles[2], 5)))
      f.write('\n')
      f.write("parameter ranking algorithm iterations q4: " + str(round(param_iquartiles[3], 5)))
      f.write('\n')
      print(param_fquartiles)
      f.write("parameter ranking algorithm f-score q1: " + str(round(param_fquartiles[0], 5)))
      f.write('\n')
      f.write("parameter ranking algorithm f-score q2: " + str(round(param_fquartiles[1], 5)))
      f.write('\n')
      f.write("parameter ranking algorithm f-score q3: " + str(round(param_fquartiles[2], 5)))
      f.write('\n')
      f.write("parameter ranking algorithm f-score q4: " + str(round(param_fquartiles[3], 5)))
      f.write('\n')
      print(param_failures)
      f.write("parameter ranking algorithm failures: " + str(param_failures))
      f.write('\n')

      f.write("projection parameter ranking algorithm:")
      f.write('\n')
      print(proj_iquartiles)
      f.write("projection parameter ranking algorithm iterations q1: " + str(round(proj_iquartiles[0], 5)))
      f.write('\n')
      f.write("projection parameter ranking algorithm iterations q2: " + str(round(proj_iquartiles[1], 5)))
      f.write('\n')
      f.write("projection parameter ranking algorithm iterations q3: " + str(round(proj_iquartiles[2], 5)))
      f.write('\n')
      f.write("projection parameter ranking algorithm iterations q4: " + str(round(proj_iquartiles[3], 5)))
      f.write('\n')
      print(proj_fquartiles)
      f.write("projection parameter ranking algorithm f-score q1: " + str(round(proj_fquartiles[0], 5)))
      f.write('\n')
      f.write("projection parameter ranking algorithm f-score q2: " + str(round(proj_fquartiles[1], 5)))
      f.write('\n')
      f.write("projection parameter ranking algorithm f-score q3: " + str(round(proj_fquartiles[2], 5)))
      f.write('\n')
      f.write("projection parameter ranking algorithm f-score q4: " + str(round(proj_fquartiles[3], 5)))
      f.write('\n')
      print(proj_failures)
      f.write("projection parameter ranking algorithm failures: " + str(proj_failures))
      f.write('\n')

      f.write("profile ranking algorithm:")
      f.write('\n')
      print(profile_iquartiles)
      f.write("profile ranking algorithm iterations q1: " + str(round(profile_iquartiles[0], 5)))
      f.write('\n')
      f.write("profile ranking algorithm iterations q2: " + str(round(profile_iquartiles[1], 5)))
      f.write('\n')
      f.write("profile ranking algorithm iterations q3: " + str(round(profile_iquartiles[2], 5)))
      f.write('\n')
      f.write("profile ranking algorithm iterations q4: " + str(round(profile_iquartiles[3], 5)))
      f.write('\n')
      for prof in profile.profiles:
         f.write(prof + " sample iterations q1: " + str(round(rank_profilequartiles[prof][0], 5)))
         f.write('\n')
         f.write(prof + " sample iterations q2: " + str(round(rank_profilequartiles[prof][1], 5)))
         f.write('\n')
         f.write(prof + " sample iterations q3: " + str(round(rank_profilequartiles[prof][2], 5)))
         f.write('\n')
         f.write(prof + " sample iterations q4: " + str(round(rank_profilequartiles[prof][3], 5)))
         f.write('\n')
      print(profile_fquartiles)
      f.write("profile ranking algorithm f-score q1: " + str(round(profile_fquartiles[0], 5)))
      f.write('\n')
      f.write("profile ranking algorithm f-score q2: " + str(round(profile_fquartiles[1], 5)))
      f.write('\n')
      f.write("profile ranking algorithm f-score q3: " + str(round(profile_fquartiles[2], 5)))
      f.write('\n')
      f.write("profile ranking algorithm f-score q4: " + str(round(profile_fquartiles[3], 5)))
      f.write('\n')
      print(profile_failures)
      f.write("profile ranking algorithm failures: " + str(profile_failures))
      f.write('\n')

      f.write("grid search algorithm:")
      f.write('\n')
      print(g_iquartiles)
      f.write("grid search iterations q1: " + str(round(g_iquartiles[0], 5)))
      f.write('\n')
      f.write("grid search iterations q2: " + str(round(g_iquartiles[1], 5)))
      f.write('\n')
      f.write("grid search iterations q3: " + str(round(g_iquartiles[2], 5)))
      f.write('\n')
      f.write("grid search iterations q4: " + str(round(g_iquartiles[3], 5)))
      f.write('\n')
      print(g_fquartiles)
      f.write("grid search f-score q1: " + str(round(g_fquartiles[0], 5)))
      f.write('\n')
      f.write("grid search f-score q2: " + str(round(g_fquartiles[1], 5)))
      f.write('\n')
      f.write("grid search f-score q3: " + str(round(g_fquartiles[2], 5)))
      f.write('\n')
      f.write("grid search f-score q4: " + str(round(g_fquartiles[3], 5)))
      f.write('\n')
      
    f.close()
    profiles_df = pd.DataFrame.from_records(profile_data, columns=['init pf','init block clean threshold', 'init weighting schema', 'init matching threshold', 'f-score goal', 'param iterations', 'param f-score', 'proj iterations', 'proj f-score', 'profile iterations', 'profile f-score', 'grid search iterations', 'grid search f-score'])
    if not(os.path.exists('ERmetrics/experiment1summary_8.csv')):
      profiles_df.to_csv('ERmetrics/experiment1summary_8.csv')

    summary_df = pd.read_csv('ERmetrics/experiment1summary_8.csv')
    summary_group = summary_df.groupby('f-score goal')
    #f_goals = [0.8, 0.9, 0.95, 0.96]
    param_x, param_y = ([] for i in range(2))
    proj_x, proj_y = ([] for i in range(2))
    profile_x, profile_y = ([] for i in range(2))
    grid_x, grid_y = ([] for i in range(2))
    for f_goal in f_goals:
        param_iterlst = summary_group.get_group(f_goal)['param iterations'].tolist()
        param_itermed = np.percentile(param_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
        param_x.append(f_goal)
        param_y.append(param_itermed)

        proj_iterlst = summary_group.get_group(f_goal)['proj iterations'].tolist()
        proj_itermed = np.percentile(proj_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
        proj_x.append(f_goal)
        proj_y.append(proj_itermed)

        profile_iterlst = summary_group.get_group(f_goal)['profile iterations'].tolist()
        profile_itermed = np.percentile(profile_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
        profile_x.append(f_goal)
        profile_y.append(profile_itermed)

        grid_iterlst = summary_group.get_group(f_goal)['grid search iterations'].tolist()
        grid_itermed = np.percentile(grid_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
        grid_x.append(f_goal)
        grid_y.append(grid_itermed)

    plt.yscale("log")
    plt.plot(param_x, param_y, proj_x, proj_y, profile_x, profile_y, grid_x, grid_y)
    plt.show()
