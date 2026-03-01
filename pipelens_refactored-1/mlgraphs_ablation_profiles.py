import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pylab as plot

fsize=20
params = {'legend.fontsize': fsize,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : fsize}
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

if __name__ == "__main__":
    summary_df = pd.read_csv('metric/ablation-profiles/ablation_profiles_results_summary.txt')
    summary_group = summary_df.groupby(['method','kbest'])
    s_sizes = [1,3,5,7,9]
    
    proj_x, q2proj_y, q4proj_y = ([] for i in range(3))
    profile_x, q2profile_y, q4profile_y = ([] for i in range(3))
    
    for sz in s_sizes:
        proj_iterlst = summary_group.get_group(('projection',sz))['iterations'].tolist()
        proj_itermed = np.percentile(proj_iterlst, [25, 50, 75, 100], interpolation='midpoint')[1]                                                         
        proj_x.append(sz)
        q2proj_y.append(np.percentile(proj_iterlst, [25, 50, 75, 100], interpolation='midpoint')[1])
        q4proj_y.append(np.percentile(proj_iterlst, [25, 50, 75, 100], interpolation='midpoint')[3])

        profile_iterlst = summary_group.get_group(('profile',sz))['iterations'].tolist()                                                         
        profile_x.append(sz)
        q2profile_y.append(np.percentile(profile_iterlst, [25, 50, 75, 100], interpolation='midpoint')[1])
        q4profile_y.append(np.percentile(profile_iterlst, [25, 50, 75, 100], interpolation='midpoint')[3])

    # Q2 graph
    plt.figure(figsize=(6, 5)) # in inches!
    plt.xticks(fontsize= fsize/1.2)
    plt.yscale("linear")
    plt.plot(proj_x, q2proj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
    plt.plot(profile_x, q2profile_y, 'k-s',label='Profile',color='black',markersize=18)
    plt.legend()

    plt.xlabel('#best profiles not considered',labelpad=0, fontsize=fsize/1.2)
    plt.ylabel('#iterations',labelpad=0, fontsize=fsize/1.2)
    plt.savefig('MLfigures/remove_profiles_q2.pdf')


    # Q4 graph
    plt.figure(figsize=(6, 5)) # in inches!
    plt.xticks(fontsize= fsize/1.2)
    plt.yscale("linear")
    plt.plot(proj_x, q4proj_y, 'k-x', label='Projection',color='forestgreen',markersize=18)
    plt.plot(profile_x, q4profile_y, 'k-s',label='Profile',color='black',markersize=18)
    plt.legend(loc='upper right')

    plt.xlabel('#best profiles not considered',labelpad=0, fontsize=fsize/1.2)
    plt.ylabel('#iterations',labelpad=0, fontsize=fsize/1.2)
    plt.savefig('MLfigures/remove_profiles_q4.pdf')