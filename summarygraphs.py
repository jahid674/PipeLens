import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parameter_df = pd.read_csv('ERmetrics/optrankvsgs2.csv')
profile_df = pd.read_csv('ERmetrics/rankvsgs4.csv')
proj_df = pd.read_csv('ERmetrics/rankvsgsprojection.csv')

parameter_group = parameter_df.groupby('f-score goal')
f_goals = [0.8, 0.9, 0.95, 0.98, 0.99]
parameter_x = []
parameter_y = []
for f_goal in f_goals:
    rank_iterlst = parameter_group.get_group(f_goal)['rank iterations'].tolist()
    itermed = np.percentile(rank_iterlst, [25, 50, 75], interpolation='midpoint')[1
                                                                                  ]
    parameter_x.append(f_goal)
    parameter_y.append(itermed)

gridsearch_group = parameter_df.groupby('f-score goal')
f_goals = [0.8, 0.9, 0.95, 0.98, 0.99]
gridsearch_x = []
gridsearch_y = []
for f_goal in f_goals:
    rank_iterlst = parameter_group.get_group(f_goal)['grid search iterations'].tolist()
    itermed = np.percentile(rank_iterlst, [25, 50, 75], interpolation='midpoint')[1
                                                                                  ]
    gridsearch_x.append(f_goal)
    gridsearch_y.append(itermed)

profile_group = profile_df.groupby('f-score goal')
f_modgoals = [0.8, 0.9, 0.95, 0.98, 0.99]
profile_x = []
profile_y = []
for f_goal in f_modgoals:
    rank_iterlst = profile_group.get_group(f_goal)['rank iterations'].tolist()
    itermed = np.percentile(rank_iterlst, [25, 50, 75], interpolation='midpoint')[1
                                                                                  ]
    profile_x.append(f_goal)
    profile_y.append(itermed)

proj_group = proj_df.groupby('f-score goal')
proj_x = []
proj_y = []
for f_goal in f_goals:
    rank_iterlst = proj_group.get_group(f_goal)['rank iterations'].tolist()
    itermed = np.percentile(rank_iterlst, [25, 50, 75], interpolation='midpoint')[1
                                                                                  ]
    proj_x.append(f_goal)
    proj_y.append(itermed)

plt.yscale("log")
plt.plot(parameter_x, parameter_y, profile_x, profile_y, proj_x, proj_y, gridsearch_x, gridsearch_y)
plt.show()
