import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
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
  summary_df = pd.read_csv('ERmetrics/experiment2summary.csv')
  summary_group = summary_df.groupby('search space size')
  s_sizes = [48, 96, 200, 648]
  param_x, param_y = ([] for i in range(2))
  proj_x, proj_y = ([] for i in range(2))
  profile_x, profile_y = ([] for i in range(2))
  grid_x, grid_y = ([] for i in range(2))
  for sz in s_sizes:
      param_iterlst = summary_group.get_group(sz)['param iterations'].tolist()
      param_itermed = np.percentile(param_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
      param_x.append(sz)
      param_y.append(param_itermed)

      proj_iterlst = summary_group.get_group(sz)['proj iterations'].tolist()
      proj_itermed = np.percentile(proj_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
      proj_x.append(sz)
      proj_y.append(proj_itermed)

      profile_iterlst = summary_group.get_group(sz)['profile iterations'].tolist()
      profile_itermed = np.percentile(profile_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
      profile_x.append(sz)
      profile_y.append(profile_itermed)

      grid_iterlst = summary_group.get_group(sz)['grid search iterations'].tolist()
      grid_itermed = np.percentile(grid_iterlst, [25, 50, 75], interpolation='midpoint')[1]                                                         
      grid_x.append(sz)
      grid_y.append(grid_itermed)

  plt.yscale("linear")
  #plt.plot(param_x, param_y, proj_x, proj_y, profile_x, profile_y, grid_x, grid_y)
  

  plt.figure(figsize=(6, 5)) # in inches!
  plt.xticks(fontsize= fsize/1.2)
  plt.yscale("linear")
  plt.plot(param_x, param_y, 'k-v', label='Parameter',color='salmon',markersize=18)
  plt.plot(proj_x, proj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
  plt.plot(profile_x, profile_y, 'k-s',label='Profile',color='black',markersize=18)
  plt.plot(grid_x, grid_y, 'k-o',label='Grid',color='navy',markersize=18)#, q2proj_x, q2proj_y, q2profile_x, q2profile_y, q2grid_x, q2grid_y)
  plt.legend()

  plt.xlabel('Search Space Size',labelpad=0, fontsize=fsize/1.2)
  plt.ylabel('#iterations',labelpad=0, fontsize=fsize/1.2)
  plt.savefig('ERfigures/injectq2.pdf')

  #Q4 graph
  plt.clf()
  summary_df = pd.read_csv('ERmetrics/experiment2summary.csv')
  summary_group = summary_df.groupby('search space size')
  s_sizes = [48, 96, 200, 648]
  q4param_x, q4param_y = ([] for i in range(2))
  q4proj_x, q4proj_y = ([] for i in range(2))
  q4profile_x, q4profile_y = ([] for i in range(2))
  # q4grid_x, q4grid_y = ([] for i in range(2))
  q4grid_x = [48, 96, 200, 648]
  q4grid_y = [26.0, 46.0, 59.0, 136.0]
  for sz in s_sizes:
      q4param_iterlst = summary_group.get_group(sz)['param iterations'].tolist()
      param_itermax = np.percentile(q4param_iterlst, [25, 50, 75,100], interpolation='midpoint')[3]                                                         
      q4param_x.append(sz)
      q4param_y.append(param_itermax)

      q4proj_iterlst = summary_group.get_group(sz)['proj iterations'].tolist()
      proj_itermax = np.percentile(q4proj_iterlst, [25, 50, 75, 100], interpolation='midpoint')[3]                                                         
      q4proj_x.append(sz)
      q4proj_y.append(proj_itermax)

      q4profile_iterlst = summary_group.get_group(sz)['profile iterations'].tolist()
      profile_itermax = np.percentile(q4profile_iterlst, [25, 50, 75, 100], interpolation='midpoint')[3]                                                         
      q4profile_x.append(sz)
      q4profile_y.append(profile_itermax)

      # q4grid_iterlst = summary_group.get_group(sz)['grid search iterations'].tolist()
      # grid_itermax = np.percentile(q4grid_iterlst, [25, 50, 75, 100], interpolation='midpoint')[3]                                                         
      # q4grid_x.append(sz)
      # q4grid_y.append(grid_itermax)

#   plt.yscale("linear")
#   plt.plot(q4param_x, q4param_y, q4proj_x, q4proj_y, q4profile_x, q4profile_y, q4grid_x, q4grid_y)
#   plt.savefig('ERfigures/injectq4_9.pdf')

  plt.figure(figsize=(6, 5)) # in inches!
  plt.xticks(fontsize= fsize/1.2)
  plt.yscale("linear")
  plt.plot(q4param_x, q4param_y, 'k-v', label='Parameter',color='salmon',markersize=18)
  plt.plot(q4proj_x, q4proj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
  plt.plot(q4profile_x, q4profile_y, 'k-s',label='Profile',color='black',markersize=18)
  plt.plot(q4grid_x, q4grid_y, 'k-o',label='Grid',color='navy',markersize=18)#, q2proj_x, q2proj_y, q2profile_x, q2profile_y, q2grid_x, q2grid_y)
  plt.legend()

  plt.xlabel('Search Space Size',labelpad=0, fontsize=fsize/1.2)
  plt.ylabel('#iterations',labelpad=0, fontsize=fsize/1.2)
  plt.savefig('ERfigures/injectq4.pdf')