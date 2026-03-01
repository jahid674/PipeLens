import numpy as np
from matplotlib import pyplot as plt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import pylab as plot


fsize=20
params = {'legend.fontsize': fsize,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : fsize}
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)


if __name__ == "__main__":
  #Q2 graph
  summary_df = pd.read_csv('ERmetrics/experiment1summary_8.csv')
  summary_group = summary_df.groupby('f-score goal')
  f_goals = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92,0.94, 0.96]
  q2param_x, q2param_y = ([] for i in range(2))
  q2proj_x, q2proj_y = ([] for i in range(2))
  q2profile_x, q2profile_y = ([] for i in range(2))
  q2grid_x, q2grid_y = ([] for i in range(2))
  for f_goal in f_goals:
      q2param_iterlst = summary_group.get_group(f_goal)['param iterations'].tolist()
      q2param_itermed = np.percentile(q2param_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
      q2param_x.append(f_goal)
      q2param_y.append(q2param_itermed)

      q2proj_iterlst = summary_group.get_group(f_goal)['proj iterations'].tolist()
      q2proj_itermed = np.percentile(q2proj_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
      q2proj_x.append(f_goal)
      q2proj_y.append(q2proj_itermed)

      q2profile_iterlst = summary_group.get_group(f_goal)['profile iterations'].tolist()
      q2profile_itermed = np.percentile(q2profile_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
      q2profile_x.append(f_goal)
      q2profile_y.append(q2profile_itermed)

      q2grid_iterlst = summary_group.get_group(f_goal)['grid search iterations'].tolist()
      q2grid_itermed = np.percentile(q2grid_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
      q2grid_x.append(f_goal)
      q2grid_y.append(q2grid_itermed)

  plt.figure(figsize=(6, 5)) # in inches!
  plt.xticks(fontsize= fsize/1.2)
  plt.yscale("linear")
  plt.plot(q2param_x, q2param_y, 'k-v', label='Parameter',color='salmon',markersize=18)
  plt.plot(q2proj_x, q2proj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
  plt.plot(q2profile_x, q2profile_y, 'k-s',label='Profile',color='black',markersize=18)
  plt.plot(q2grid_x, q2grid_y, 'k-o',label='Grid',color='navy',markersize=18)#, q2proj_x, q2proj_y, q2profile_x, q2profile_y, q2grid_x, q2grid_y)
  plt.legend()

  
  #plt.xticks([0, 1000,50000, 100000,200000], ['', '1K','50K', '100K','200K'])
  #plot.ylim([0.55,0.85])

  plt.xlabel('Goal Utility',labelpad=0, fontsize=fsize/1.2)
  plt.ylabel('#iterations',labelpad=0, fontsize=fsize/1.2)
  plt.savefig('ERfigures/q2.pdf')




  #Q4 graph
  plt.clf()
  summary_df = pd.read_csv('ERmetrics/experiment1summary_8.csv')
  summary_group = summary_df.groupby('f-score goal')
  f_goals = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92,0.94, 0.96]
  q4param_x, q4param_y = ([] for i in range(2))
  q4proj_x, q4proj_y = ([] for i in range(2))
  q4profile_x, q4profile_y = ([] for i in range(2))
  q4grid_x =  [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92,0.94, 0.96]
  q4grid_y = [17.0, 25, 35, 45, 45, 45.0, 45, 88, 268.0] #from experiment1summary_8.txt
  for f_goal in f_goals:
      q4param_iterlst = summary_group.get_group(f_goal)['param iterations'].tolist()
      param_itermax = np.percentile(q4param_iterlst, [25, 50, 75, 100], interpolation='midpoint')[3]                                                         
      q4param_x.append(f_goal)
      q4param_y.append(param_itermax)

      q4proj_iterlst = summary_group.get_group(f_goal)['proj iterations'].tolist()
      proj_itermax = np.percentile(q4proj_iterlst, [25, 50, 75, 100], interpolation='midpoint')[3]                                                         
      q4proj_x.append(f_goal)
      q4proj_y.append(proj_itermax)

      q4profile_iterlst = summary_group.get_group(f_goal)['profile iterations'].tolist()
      profile_itermax = np.percentile(q4profile_iterlst, [25, 50, 75, 100], interpolation='midpoint')[3]                                                         
      q4profile_x.append(f_goal)
      q4profile_y.append(profile_itermax)

  
  #plt.plot(q4param_x, q4param_y, q4proj_x, q4proj_y, q4profile_x, q4profile_y, q4grid_x, q4grid_y)

  plt.figure(figsize=(6, 5)) # in inches!
  plt.xticks(fontsize= fsize/1.2)
  plt.yscale("log")
  plt.plot(q4param_x, q4param_y, 'k-v', label='Parameter',color='salmon',markersize=18)
  plt.plot(q4proj_x, q4proj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
  plt.plot(q4profile_x, q4profile_y, 'k-s',label='Profile',color='black',markersize=18)
  plt.plot(q4grid_x, q4grid_y, 'k-o',label='Grid',color='navy',markersize=18)#, q2proj_x, q2proj_y, q2profile_x, q2profile_y, q2grid_x, q2grid_y)
  plt.legend()

  
  #plt.xticks([0, 1000,50000, 100000,200000], ['', '1K','50K', '100K','200K'])
  #plot.ylim([0.55,0.85])

  plt.xlabel('Goal Utility',labelpad=0, fontsize=fsize/1.2)
  plt.ylabel('#iterations',labelpad=-5, fontsize=fsize/1.2)
  plt.savefig('ERfigures/q4.pdf')





  #0.8 graph
  plt.clf()
  summary_df = pd.read_csv('ERmetrics/experiment1summary_8.csv')
  summary_group = summary_df.groupby('f-score goal')
  f_goal1 = 0.8
  minparam_x, minparam_y = ([] for i in range(2))
  minproj_x, minproj_y = ([] for i in range(2))
  minprofile_x, minprofile_y = ([] for i in range(2))
  mingrid_x, mingrid_y = ([] for i in range(2))
  
  minparam_iterlst = summary_group.get_group(f_goal1)['param iterations'].tolist()
  minparam_iterdistr = np.percentile(minparam_iterlst, [25, 50, 75, 100], interpolation='midpoint')
  for i in range(4):                                                        
    minparam_x.append(i + 1)
    minparam_y.append(minparam_iterdistr[i])

  minproj_iterlst = summary_group.get_group(f_goal1)['proj iterations'].tolist()
  minproj_iterdistr = np.percentile(minproj_iterlst, [25, 50, 75, 100], interpolation='midpoint')                                                         
  for i in range(4):                                                        
    minproj_x.append(i + 1)
    minproj_y.append(minproj_iterdistr[i])

  minprofile_iterlst = summary_group.get_group(f_goal1)['profile iterations'].tolist()
  minprofile_iterdistr = np.percentile(minprofile_iterlst, [25, 50, 75, 100], interpolation='midpoint')                                                         
  for i in range(4):                                                        
    minprofile_x.append(i + 1)
    minprofile_y.append(minprofile_iterdistr[i])

  mingrid_iterlst = summary_group.get_group(f_goal1)['grid search iterations'].tolist()
  mingrid_iterdistr = [2.0, 3.0, 7.0, 17.0] #from experiment1summary_8.txt                                           
  for i in range(4):                                                        
    mingrid_x.append(i + 1)
    mingrid_y.append(mingrid_iterdistr[i])

  plt.yscale("linear")
  #plt.plot(minparam_x, minparam_y, minproj_x, minproj_y, minprofile_x, minprofile_y, mingrid_x, mingrid_y)
  plt.savefig('ERfigures/80.png')

  plt.figure(figsize=(6, 5)) # in inches!
  plt.xticks(fontsize= fsize/1.2)
  plt.plot(minparam_x, minparam_y, 'k-v', label='Parameter',color='salmon',markersize=18)
  plt.plot(minproj_x, minproj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
  plt.plot(minprofile_x, minprofile_y, 'k-s',label='Profile',color='black',markersize=18)
  plt.plot(mingrid_x, mingrid_y, 'k-o',label='Grid',color='navy',markersize=18)#, q2proj_x, q2proj_y, q2profile_x, q2profile_y, q2grid_x, q2grid_y)
  plt.legend()

  
  plt.xticks([1,2,3,4], ['25', '50','75', '100'])
  #plot.ylim([0.55,0.85])

  plt.xlabel('Percentile',labelpad=0, fontsize=fsize/1.2)
  plt.ylabel('#iterations',labelpad=-5, fontsize=fsize/1.2)
  plt.savefig('ERfigures/80.pdf')






  #0.96 graph
  plt.clf()
  summary_df = pd.read_csv('ERmetrics/experiment1summary_8.csv')
  summary_group = summary_df.groupby('f-score goal')
  f_goal2 = 0.96
  maxparam_x, maxparam_y = ([] for i in range(2))
  maxproj_x, maxproj_y = ([] for i in range(2))
  maxprofile_x, maxprofile_y = ([] for i in range(2))
  maxgrid_x, maxgrid_y = ([] for i in range(2))
  
  maxparam_iterlst = summary_group.get_group(f_goal2)['param iterations'].tolist()
  maxparam_iterdistr = np.percentile(maxparam_iterlst, [25, 50, 75, 100], interpolation='midpoint')
  for i in range(4):                                                        
    maxparam_x.append(i + 1)
    maxparam_y.append(maxparam_iterdistr[i])

  maxproj_iterlst = summary_group.get_group(f_goal2)['proj iterations'].tolist()
  maxproj_iterdistr = np.percentile(maxproj_iterlst, [25, 50, 75, 100], interpolation='midpoint')                                                         
  for i in range(4):                                                        
    maxproj_x.append(i + 1)
    maxproj_y.append(maxproj_iterdistr[i])

  maxprofile_iterlst = summary_group.get_group(f_goal2)['profile iterations'].tolist()
  maxprofile_iterdistr = np.percentile(maxprofile_iterlst, [25, 50, 75, 100], interpolation='midpoint')                                                         
  for i in range(4):                                                        
    maxprofile_x.append(i + 1)
    maxprofile_y.append(maxprofile_iterdistr[i])

  maxgrid_iterlst = summary_group.get_group(f_goal2)['grid search iterations'].tolist()
  maxgrid_iterdistr = [19.0, 43.5, 88.5, 268.0]  #from experiment1summary_8.txt                                                       
  for i in range(4):                                                        
    maxgrid_x.append(i + 1)
    maxgrid_y.append(maxgrid_iterdistr[i])

  #plt.plot(maxparam_x, maxparam_y, maxproj_x, maxproj_y, maxprofile_x, maxprofile_y, maxgrid_x, maxgrid_y)
  #plt.savefig('ERfigures/96.png')
  
  #plt.plot(minparam_x, minparam_y, minproj_x, minproj_y, minprofile_x, minprofile_y, mingrid_x, mingrid_y)
  #plt.savefig('ERfigures/80.png')

  plt.figure(figsize=(6, 5)) # in inches!
  plt.xticks(fontsize= fsize/1.2)
  plt.yscale("log")

  plt.plot(maxparam_x, maxparam_y, 'k-v', label='Parameter',color='salmon',markersize=18)
  plt.plot(maxproj_x, maxproj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
  plt.plot(maxprofile_x, maxprofile_y, 'k-s',label='Profile',color='black',markersize=18)
  plt.plot(maxgrid_x, maxgrid_y, 'k-o',label='Grid',color='navy',markersize=18)#, q2proj_x, q2proj_y, q2profile_x, q2profile_y, q2grid_x, q2grid_y)
  plt.legend()

  
  plt.xticks([1,2,3,4], ['25', '50','75', '100'])
  #plot.ylim([0.55,0.85])

  plt.xlabel('Percentile',labelpad=0, fontsize=fsize/1.2)
  plt.ylabel('#iterations',labelpad=-12, fontsize=fsize)
  plt.savefig('ERfigures/96.pdf')

