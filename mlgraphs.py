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

def generate_percentile_graph(goal,parameter,projection,profile,grid,dataset,scale,padding):
  maxparam_x, maxparam_y = ([] for i in range(2))
  maxproj_x, maxproj_y = ([] for i in range(2))
  maxprofile_x, maxprofile_y = ([] for i in range(2))
  maxgrid_x, maxgrid_y = ([] for i in range(2))
  if dataset in l2c_datasets:
    maxl2c_x, maxl2c_y = ([] for i in range(2))
  maxbugdoc_x, maxbugdoc_y = ([] for i in range(2))
  
  for i in range(4):                                                        
    maxparam_x.append(i + 1)
    maxproj_x.append(i + 1)
    maxprofile_x.append(i + 1)
    maxgrid_x.append(i + 1)
    if dataset in l2c_datasets:
      maxl2c_x.append(i + 1)
    maxbugdoc_x.append(i + 1)
    
  maxparam_y.append(float(parameter['q1_'+str(goal)][0]))
  maxparam_y.append(float(parameter['q2_'+str(goal)][0]))
  maxparam_y.append(float(parameter['q3_'+str(goal)][0]))
  maxparam_y.append(float(parameter['q4_'+str(goal)][0]))


  maxproj_y.append(float(projection['q1_'+str(goal)][0]))
  maxproj_y.append(float(projection['q2_'+str(goal)][0]))
  maxproj_y.append(float(projection['q3_'+str(goal)][0]))
  maxproj_y.append(float(projection['q4_'+str(goal)][0]))


  maxprofile_y.append(float(profile['q1_'+str(goal)][0]))
  maxprofile_y.append(float(profile['q2_'+str(goal)][0]))
  maxprofile_y.append(float(profile['q3_'+str(goal)][0]))
  maxprofile_y.append(float(profile['q4_'+str(goal)][0]))


  maxgrid_y.append(float(grid['q1_'+str(goal)][0]))
  maxgrid_y.append(float(grid['q2_'+str(goal)][0]))
  maxgrid_y.append(float(grid['q3_'+str(goal)][0]))
  maxgrid_y.append(float(grid['q4_'+str(goal)][0]))

  if dataset in l2c_datasets:
    maxl2c_y.append(float(l2c['q1_'+str(goal)][0]))
    maxl2c_y.append(float(l2c['q2_'+str(goal)][0]))
    maxl2c_y.append(float(l2c['q3_'+str(goal)][0]))
    maxl2c_y.append(float(l2c['q4_'+str(goal)][0]))
  
  maxbugdoc_y.append(float(bugdoc['q1_'+str(goal)][0]))
  maxbugdoc_y.append(float(bugdoc['q2_'+str(goal)][0]))
  maxbugdoc_y.append(float(bugdoc['q3_'+str(goal)][0]))
  maxbugdoc_y.append(float(bugdoc['q4_'+str(goal)][0]))

  plt.figure(figsize=(6, 5)) # in inches!
  plt.xticks(fontsize= fsize/1.2)
  plt.yticks(fontsize= fsize/1.2)
  plt.yscale(scale)

  plt.plot(maxgrid_x, maxgrid_y, 'k-o',label='GridSearch-ES',color='navy',markersize=18)
  if dataset in l2c_datasets:
     plt.plot(maxl2c_x, maxl2c_y, 'k-*',label='Learn2Clean',color='red',markersize=15)
  plt.plot(maxbugdoc_x, maxbugdoc_y, 'k-d',label='BugDoc',color='blueviolet',markersize=15)
  # plt.plot(maxparam_x, maxparam_y, 'k-v', label='PipeLens_O',color='salmon',markersize=18)
  # plt.plot(maxproj_x, maxproj_y, 'k-x', label='PipeLens_P',color='forestgreen',markersize=18)
  plt.plot(maxprofile_x, maxprofile_y, 'k-s',label='PipeLens',color='black',markersize=15)
  plt.legend(fontsize=fsize/1.2, loc='best')

  # plt.tight_layout()
  plt.xticks([1,2,3,4], ['25', '50','75', '100'])
  plt.xlabel('Percentile',labelpad=0, fontsize=fsize/1.2)
  plt.ylabel('#iterations',labelpad=padding, fontsize=fsize/1.2)
  plt.savefig('MLfigures/percentile_new_sys_'+dataset+'_'+str(goal)+'.pdf')
  
  


if __name__ == "__main__":
  l2c_datasets = ['hmda', 'housing']
  current_dataset = ['adult']
  for dataset in current_dataset:
    onestep = {}
    twostep = {}
    projection = {}
    grid_search = {}
    l2c = {}
    bugdoc = {}
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

              elif(row[1]=='l2c' and row[2]=='iterations q1'):
                if val.find(dataset)>-1 :
                    if('q1_'+row[0] not in l2c):
                        l2c['q1_'+row[0]] = []
                          
                    l2c['q1_'+row[0]].append(row[3])
              
              elif(row[1]=='bugdoc' and row[2]=='iterations q1'):
                if val.find(dataset)>-1 :
                    if('q1_'+row[0] not in bugdoc):
                        bugdoc['q1_'+row[0]] = []
                          
                    bugdoc['q1_'+row[0]].append(row[3])
              
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
              
              elif(row[1]=='l2c' and row[2]=='iterations q2'):
                if val.find(dataset)>-1 :
                    if('q2_'+row[0] not in l2c):
                        l2c['q2_'+row[0]] = []
                          
                    l2c['q2_'+row[0]].append(row[3])
              
              elif(row[1]=='bugdoc' and row[2]=='iterations q2'):
                if val.find(dataset)>-1 :
                    if('q2_'+row[0] not in bugdoc):
                        bugdoc['q2_'+row[0]] = []
                          
                    bugdoc['q2_'+row[0]].append(row[3])
              
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
              
              elif(row[1]=='l2c' and row[2]=='iterations q3'):
                if val.find(dataset)>-1 :
                    if('q3_'+row[0] not in l2c):
                        l2c['q3_'+row[0]] = []
                          
                    l2c['q3_'+row[0]].append(row[3])
              
              elif(row[1]=='bugdoc' and row[2]=='iterations q3'):
                if val.find(dataset)>-1 :
                    if('q3_'+row[0] not in bugdoc):
                        bugdoc['q3_'+row[0]] = []
                          
                    bugdoc['q3_'+row[0]].append(row[3])

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
              
              elif(row[1]=='l2c' and row[2]=='iterations q4'):
                if val.find(dataset)>-1 :
                    if('q4_'+row[0] not in l2c):
                        l2c['q4_'+row[0]] = []
                          
                    l2c['q4_'+row[0]].append(row[3])
              
              elif(row[1]=='bugdoc' and row[2]=='iterations q4'):
                if val.find(dataset)>-1 :
                    if('q4_'+row[0] not in bugdoc):
                        bugdoc['q4_'+row[0]] = []
                          
                    bugdoc['q4_'+row[0]].append(row[3])

              elif (row[1]=='grid search' and row[2]=='iterations q4' and val.find(dataset)>-1):
                if('q4_'+row[0] not in grid_search):
                  grid_search['q4_'+row[0]] = []
                grid_search['q4_'+row[0]].append(row[3])

    if(dataset =='adult'):
      f_goals = [0.84, 0.87, 0.94, 0.95]
      best = .95
      worst = 0.84
    elif(dataset=='hmda'):
      f_goals  = [.94, .93, .92, .91]
      best = .94
      worst = .91
    elif(dataset=='housing'):
      f_goals = [0.86,0.89,0.95,1.0]
      best = 1.0
      worst = 0.86
    else:
      print('Please profile goals ')

    # print("---- printing bugdoc --- ", bugdoc)

    q2param_x, q2param_y = ([] for i in range(2))
    q2proj_x, q2proj_y = ([] for i in range(2))
    q2profile_x, q2profile_y = ([] for i in range(2))
    q2grid_x, q2grid_y = ([] for i in range(2))
    q2l2c_x, q2l2c_y = ([] for i in range(2))
    q2bugdoc_x, q2bugdoc_y = ([] for i in range(2))

    q4param_x, q4param_y = ([] for i in range(2))
    q4proj_x, q4proj_y = ([] for i in range(2))
    q4profile_x, q4profile_y = ([] for i in range(2))
    q4grid_x, q4grid_y = ([] for i in range(2))
    q4l2c_x, q4l2c_y = ([] for i in range(2))
    q4bugdoc_x, q4bugdoc_y = ([] for i in range(2))

    q1param_x, q1param_y = ([] for i in range(2))
    q1proj_x, q1proj_y = ([] for i in range(2))
    q1profile_x, q1profile_y = ([] for i in range(2))
    q1grid_x, q1grid_y = ([] for i in range(2))
    q1l2c_x, q1l2c_y = ([] for i in range(2))
    q1bugdoc_x, q1bugdoc_y = ([] for i in range(2))

    q3param_x, q3param_y = ([] for i in range(2))
    q3proj_x, q3proj_y = ([] for i in range(2))
    q3profile_x, q3profile_y = ([] for i in range(2))
    q3grid_x, q3grid_y = ([] for i in range(2))
    q3l2c_x, q3l2c_y = ([] for i in range(2))
    q3bugdoc_x, q3bugdoc_y = ([] for i in range(2))

    for f_goal in f_goals:
        # q2param_iterlst = summary_group.get_group(f_goal)['param iterations'].tolist()
        key_q2  = 'q2_'+str(f_goal)
        q2param_itermed = float(onestep[key_q2][0])
        q2param_x.append(f_goal)
        q2param_y.append(q2param_itermed)

        # q2proj_iterlst = summary_group.get_group(f_goal)['proj iterations'].tolist()
        q2proj_itermed = float(projection[key_q2][0])
        q2proj_x.append(f_goal)
        q2proj_y.append(q2proj_itermed)

        # q2profile_iterlst = summary_group.get_group(f_goal)['profile iterations'].tolist()
        q2profile_itermed = float(twostep[key_q2][0])
        q2profile_x.append(f_goal)
        q2profile_y.append(q2profile_itermed)

        # q2grid_iterlst = summary_group.get_group(f_goal)['grid search iterations'].tolist()
        q2grid_itermed = float(grid_search[key_q2][0])
        q2grid_x.append(f_goal)
        q2grid_y.append(q2grid_itermed)

        # q2l2c
        if dataset in l2c_datasets:
          q2l2c_itermed = float(l2c[key_q2][0])
          q2l2c_x.append(f_goal)
          q2l2c_y.append(q2l2c_itermed)

        q2bugdoc_itermed = float(bugdoc[key_q2][0])
        q2bugdoc_x.append(f_goal)
        q2bugdoc_y.append(q2bugdoc_itermed)

        key_q4  = 'q4_'+str(f_goal)
        q4param_itermed = float(onestep[key_q4][0])
        q4param_x.append(f_goal)
        q4param_y.append(q4param_itermed)

        # q4proj_iterlst = summary_group.get_group(f_goal)['proj iterations'].tolist()
        q4proj_itermed = float(projection[key_q4][0])
        q4proj_x.append(f_goal)
        q4proj_y.append(q4proj_itermed)

        # q4profile_iterlst = summary_group.get_group(f_goal)['profile iterations'].tolist()
        q4profile_itermed = float(twostep[key_q4][0])
        q4profile_x.append(f_goal)
        q4profile_y.append(q4profile_itermed)
        
        q4grid_itermed = float(grid_search[key_q4][0])
        q4grid_x.append(f_goal)
        q4grid_y.append(q4grid_itermed)

        # q4l2c
        if dataset in l2c_datasets:
          q4l2c_itermed = float(l2c[key_q4][0])
          q4l2c_x.append(f_goal)
          q4l2c_y.append(q4l2c_itermed)

        q4bugdoc_itermed = float(bugdoc[key_q4][0])
        q4bugdoc_x.append(f_goal)
        q4bugdoc_y.append(q4bugdoc_itermed)

  #Q2 graph        
    plt.figure(figsize=(6, 5)) # in inches!
    plt.xticks(fontsize= fsize/1.2)
    plt.yticks(fontsize= fsize/1.2)
    plt.yscale("linear")

    plt.plot(q2grid_x, q2grid_y, 'k-o',label='GridSearch-ES',color='navy',markersize=18)
    if dataset in l2c_datasets:
      plt.plot(q2l2c_x, q2l2c_y, 'k-*',label='Learn2Clean',color='red',markersize=15)
    plt.plot(q2bugdoc_x, q2bugdoc_y, 'k-d',label='BugDoc',color='blueviolet',markersize=15)
    # plt.plot(q2param_x, q2param_y, 'k-v', label='PipeLens_O',color='salmon',markersize=18)
    # plt.plot(q2proj_x, q2proj_y, 'k-x', label='PipeLens_P',color='forestgreen',markersize=18)
    plt.plot(q2profile_x, q2profile_y, 'k-s',label='PipeLens',color='black',markersize=15)
    plt.legend(fontsize=fsize/1.2, loc='best')

    plt.xlabel('Goal Utility',labelpad=0, fontsize=fsize/1.2)
    plt.ylabel('#iterations',labelpad=0, fontsize=fsize/1.2)
    plt.savefig('MLfigures/q2_new_sys_'+dataset+'.pdf')
    print(dataset)

    # q4grid_iterlst = summary_group.get_group(f_goal)['grid search iterations'].tolist()

  #Q4 graph
    plt.figure(figsize=(6, 5)) # in inches!
    plt.xticks(fontsize= fsize/1.2)
    plt.yticks(fontsize= fsize/1.2)
    if dataset not in ['housing']:
      plt.yscale("log")

    plt.plot(q4grid_x, q4grid_y, 'k-o',label='GridSearch-ES',color='navy',markersize=18)
    if dataset in l2c_datasets:
      plt.plot(q4l2c_x, q4l2c_y, 'k-*',label='Learn2Clean',color='red',markersize=15)
    plt.plot(q4bugdoc_x, q4bugdoc_y, 'k-d',label='BugDoc',color='blueviolet',markersize=15)
    # plt.plot(q4param_x, q4param_y, 'k-v', label='PipeLens_O',color='salmon',markersize=18)
    # plt.plot(q4proj_x, q4proj_y, 'k-x', label='PipeLens_P',color='forestgreen',markersize=18)
    plt.plot(q4profile_x, q4profile_y, 'k-s',label='PipeLens',color='black',markersize=15)
    plt.legend(fontsize=fsize/1.2, loc='best')

    plt.xlabel('Goal Utility',labelpad=0, fontsize=fsize/1.2)
    plt.ylabel('#iterations',labelpad=0, fontsize=fsize/1.2)
    plt.savefig('MLfigures/q4_new_sys_'+dataset+'.pdf')
    print(dataset)
    
    q4_scale = 'linear'
    if dataset not in ['housing']:
       q4_scale = 'log'
    generate_percentile_graph(worst,onestep,projection,twostep,grid_search,dataset,'linear',0)
    generate_percentile_graph(best,onestep,projection,twostep,grid_search,dataset,q4_scale,0)
