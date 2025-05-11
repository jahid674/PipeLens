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
import csv
import os
def find_csv_files(directory):

    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(directory, filename))
    return csv_files



def generate_percentile_graph(profile,grid,dataset,scale,padding):
  maxprofile_x, maxprofile_y = ([] for i in range(2))
  maxgrid_x, maxgrid_y = ([] for i in range(2))
  profile_x, profile_y = ([] for i in range(2))
  grid_x, grid_y = ([] for i in range(2))

  for i in range(4):                                                        
    # maxparam_x.append(i + 1)
    # maxproj_x.append(i + 1)
    maxprofile_x.append(i + 1)
    maxgrid_x.append(i + 1)
    profile_x.append(i + 1)
    grid_x.append(i + 1)

  maxprofile_y.append(float(profile['q1_'+str(best)][0]))
  maxprofile_y.append(float(profile['q2_'+str(best)][0]))
  maxprofile_y.append(float(profile['q3_'+str(best)][0]))
  maxprofile_y.append(float(profile['q4_'+str(best)][0]))


  maxgrid_y.append(float(grid['q1_'+str(best)][0]))
  maxgrid_y.append(float(grid['q2_'+str(best)][0]))
  maxgrid_y.append(float(grid['q3_'+str(best)][0]))
  maxgrid_y.append(float(grid['q4_'+str(best)][0]))

  profile_y.append(float(profile['q1_'+str(worst)][0]))
  profile_y.append(float(profile['q2_'+str(worst)][0]))
  profile_y.append(float(profile['q3_'+str(worst)][0]))
  profile_y.append(float(profile['q4_'+str(worst)][0]))

  grid_y.append(float(grid['q1_'+str(worst)][0]))
  grid_y.append(float(grid['q2_'+str(worst)][0]))
  grid_y.append(float(grid['q3_'+str(worst)][0]))
  grid_y.append(float(grid['q4_'+str(worst)][0]))


  plt.figure(figsize=(6, 5)) # in inches!
  plt.xticks(fontsize= fsize/1.4)
  plt.yticks(fontsize= fsize/1.4)
  plt.yscale(scale)

  # plt.plot(maxparam_x, maxparam_y, 'k-v', label='Parameter',color='salmon',markersize=18)
  # plt.plot(maxproj_x, maxproj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
  plt.plot(maxgrid_x, maxgrid_y, 'k-o',label='Grid, strict goal',color='blue',markersize=15)
  plt.plot(maxprofile_x, maxprofile_y, 'k-s',label='Profile, strict goal',color='black',markersize=15)
  plt.plot(grid_x, grid_y, 'k--o',label='Grid, easy goal',color='cornflowerblue',markersize=15)
  plt.plot(profile_x, profile_y, 'k--s',label='Profile, easy goal',color='grey',markersize=15)
  
  plt.legend(fontsize=fsize/1.2)

  plt.xticks([1,2,3,4], ['25', '50','75', '100'])
  
  plt.xlabel('Percentile',labelpad=0, fontsize=fsize/1.2)
  plt.ylabel('#iterations',labelpad=padding, fontsize=fsize/1.2)
  plt.savefig('MLfigures/rq1_best_worst_'+dataset+'.pdf')

if __name__ == "__main__":
  
  for dataset in ['housing']:  
    onestep = {}
    twostep = {}
    projection = {}
    grid_search = {}
    files = find_csv_files('metric')
    for val in files:
      with open(val, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        for line_num, row in enumerate(csv_reader, start=1):
          # Print the current line number and the row
            print(f"Line {line_num}: {row}")
            if(len(row)==4):
              if(row[1]=='ranking' and row[2]=='iterations q1'):
                  if(val.find(dataset)>-1 and val.find('projection')>-1):
                      if('q1_'+row[0] not in projection):
                         projection['q1_'+row[0]] = []
                         
                      projection['q1_'+row[0]].append(row[3])
                  elif(val.find(dataset)>-1 and val.find('2step')>-1):
                      if('q1_'+row[0] not in twostep):
                         twostep['q1_'+row[0]] = []
                         
                      twostep['q1_'+row[0]].append(row[3])
                  elif val.find(dataset)>-1 :
                      if('q1_'+row[0] not in onestep):
                          onestep['q1_'+row[0]] = []
                          
                      onestep['q1_'+row[0]].append(row[3])
              
              elif (row[1]=='grid search' and row[2]=='iterations q1' and val.find(dataset)>-1):
                if('q1_'+row[0] not in grid_search):
                  grid_search['q1_'+row[0]] = []
                grid_search['q1_'+row[0]].append(row[3])
              elif(row[1]=='ranking' and row[2]=='iterations q2'):
                  if(val.find(dataset)>-1 and val.find('projection')>-1):
                      if('q2_'+row[0] not in projection):
                         projection['q2_'+row[0]] = []
                         
                      projection['q2_'+row[0]].append(row[3])
                  elif(val.find(dataset)>-1 and val.find('2step')>-1):
                      if('q2_'+row[0] not in twostep):
                         twostep['q2_'+row[0]] = []
                         
                      twostep['q2_'+row[0]].append(row[3])
                  elif val.find(dataset)>-1 :
                      if('q2_'+row[0] not in onestep):
                          onestep['q2_'+row[0]] = []
                          
                      onestep['q2_'+row[0]].append(row[3])
              
              elif (row[1]=='grid search' and row[2]=='iterations q2' and val.find(dataset)>-1):
                if('q2_'+row[0] not in grid_search):
                  grid_search['q2_'+row[0]] = []
                grid_search['q2_'+row[0]].append(row[3])
              elif(row[1]=='ranking' and row[2]=='iterations q3'):
                  if(val.find(dataset)>-1 and val.find('projection')>-1):
                      if('q3_'+row[0] not in projection):
                         projection['q3_'+row[0]] = []
                         
                      projection['q3_'+row[0]].append(row[3])
                  elif(val.find(dataset)>-1 and val.find('2step')>-1):
                      if('q3_'+row[0] not in twostep):
                         twostep['q3_'+row[0]] = []
                         
                      twostep['q3_'+row[0]].append(row[3])
                  elif val.find(dataset)>-1 :
                      if('q3_'+row[0] not in onestep):
                          onestep['q3_'+row[0]] = []
                          
                      onestep['q3_'+row[0]].append(row[3])
              
              elif (row[1]=='grid search' and row[2]=='iterations q3' and val.find(dataset)>-1):
                if('q3_'+row[0] not in grid_search):
                  grid_search['q3_'+row[0]] = []
                grid_search['q3_'+row[0]].append(row[3])

              elif (row[1]=='ranking' and row[2]=='iterations q4'):
                  if(val.find(dataset)>-1 and val.find('projection')>-1):
                      if('q4_'+row[0] not in projection):
                         projection['q4_'+row[0]] = []
                         
                      projection['q4_'+row[0]].append(row[3])
                  elif(val.find(dataset)>-1 and val.find('2step')>-1):
                      if('q4_'+row[0] not in twostep):
                         twostep['q4_'+row[0]] = []
                         
                      twostep['q4_'+row[0]].append(row[3])
                  elif val.find(dataset)>-1 :
                      if('q4_'+row[0] not in onestep):
                          onestep['q4_'+row[0]] = []
                          
                      onestep['q4_'+row[0]].append(row[3])
              
              elif (row[1]=='grid search' and row[2]=='iterations q4' and val.find(dataset)>-1):
                if('q4_'+row[0] not in grid_search):
                  grid_search['q4_'+row[0]] = []
                grid_search['q4_'+row[0]].append(row[3])
             
          
    if(dataset =='adult'):
      f_goals = [0.045, 0.055, 0.07, 0.14]
      # f_goals = [0.045, 0.07, 0.1, 0.14]
      f_goals = [round(1 - goal, 2) for goal in f_goals]
      best = max(f_goals)
      worst = min(f_goals)
      # best = .14
      # worst = 0.045
    elif(dataset=='hmda'):
      f_goals  = [.06, .07, .08, .09]
      f_goals = [round(1 - goal, 2) for goal in f_goals]
      best = max(f_goals)
      worst = min(f_goals)
      # best = .06
      # worst = 0.09
    elif(dataset=='housing'):
      f_goals = [162,165,172,175]
      best = 162
      worst = 175
    else:
            print('Please profile goals ')
    q2param_x, q2param_y = ([] for i in range(2))
    q2proj_x, q2proj_y = ([] for i in range(2))
    q2profile_x, q2profile_y = ([] for i in range(2))
    q2grid_x, q2grid_y = ([] for i in range(2))

    q4param_x, q4param_y = ([] for i in range(2))
    q4proj_x, q4proj_y = ([] for i in range(2))
    q4profile_x, q4profile_y = ([] for i in range(2))
    q4grid_x, q4grid_y = ([] for i in range(2))

    q1param_x, q1param_y = ([] for i in range(2))
    q1proj_x, q1proj_y = ([] for i in range(2))
    q1profile_x, q1profile_y = ([] for i in range(2))
    q1grid_x, q1grid_y = ([] for i in range(2))

    q3param_x, q3param_y = ([] for i in range(2))
    q3proj_x, q3proj_y = ([] for i in range(2))
    q3profile_x, q3profile_y = ([] for i in range(2))
    q3grid_x, q3grid_y = ([] for i in range(2))

  # #RQ1 graphs
  #   for f_goal in f_goals:
  #       # q2param_iterlst = summary_group.get_group(f_goal)['param iterations'].tolist()
  #       key_q2  = 'q2_'+str(f_goal)
  #       # q2param_itermed = float(onestep[key_q2][0])
  #       # q2param_x.append(f_goal)
  #       # q2param_y.append(q2param_itermed)

  #       # q2proj_iterlst = summary_group.get_group(f_goal)['proj iterations'].tolist()
  #       # q2proj_itermed = float(projection[key_q2][0])
  #       # q2proj_x.append(f_goal)
  #       # q2proj_y.append(q2proj_itermed)

  #       # q2profile_iterlst = summary_group.get_group(f_goal)['profile iterations'].tolist()
  #       q2profile_itermed = float(twostep[key_q2][0])
  #       q2profile_x.append(f_goal)
  #       q2profile_y.append(q2profile_itermed)

  #       # q2grid_iterlst = summary_group.get_group(f_goal)['grid search iterations'].tolist()
  #       q2grid_itermed = float(grid_search[key_q2][0])
  #       q2grid_x.append(f_goal)
  #       q2grid_y.append(q2grid_itermed)


  #       key_q4  = 'q4_'+str(f_goal)
  #       # q4param_itermed = float(onestep[key_q4][0])
  #       # q4param_x.append(f_goal)
  #       # q4param_y.append(q4param_itermed)

  #       # q4proj_iterlst = summary_group.get_group(f_goal)['proj iterations'].tolist()
  #       # q4proj_itermed = float(projection[key_q4][0])
  #       # q4proj_x.append(f_goal)
  #       # q4proj_y.append(q4proj_itermed)

  #       # q4profile_iterlst = summary_group.get_group(f_goal)['profile iterations'].tolist()
  #       q4profile_itermed = float(twostep[key_q4][0])
  #       q4profile_x.append(f_goal)
  #       q4profile_y.append(q4profile_itermed)
        
  #       q4grid_itermed = float(grid_search[key_q4][0])
  #       q4grid_x.append(f_goal)
  #       q4grid_y.append(q4grid_itermed)
        
  #   plt.figure(figsize=(6, 5)) # in inches!
  #   plt.xticks(fontsize= fsize/1.4)
  #   plt.yticks(fontsize= fsize/1.4)
  #   plt.yscale("log")

  #   # plt.plot(q2param_x, q2param_y, 'k-v', label='Parameter',color='salmon',markersize=18)
  #   # plt.plot(q2proj_x, q2proj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
  #   plt.plot(q2profile_x, q2profile_y, 'k--s',label='Profile, Q2',color='grey',markersize=14)
  #   plt.plot(q4profile_x, q4profile_y, 'k-s',label='Profile, Q4',color='black',markersize=14)
  #   plt.plot(q2grid_x, q2grid_y, 'k--o',label='Grid, Q2',color='cornflowerblue',markersize=14)#, q2proj_x, q2proj_y, q2profile_x, q2profile_y, q2grid_x, q2grid_y)
  #   plt.plot(q4grid_x, q4grid_y, 'k-o',label='Grid, Q4',color='blue',markersize=14)#, q2proj_x, q2proj_y, q2profile_x, q2profile_y, q2grid_x, q2grid_y)
  #   plt.legend()

  #   plt.xlabel('Goal Utility',labelpad=0, fontsize=fsize/1.2)
  #   plt.ylabel('#iterations',labelpad=0, fontsize=fsize/1.2)
  #   plt.savefig('MLfigures/rq1'+dataset+'.pdf')
  #   print(dataset)

    # generating Q1-Q4 for best and worst goals
    generate_percentile_graph(twostep,grid_search,dataset,'log',0)