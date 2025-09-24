"""
Debugger script
===========================

This script defines the pipeline entry point, the parameter-space,
 and invokes one of BugDoc's algorithm to debug the pipeline.
"""


# %%
# Importing algorithm from BugDoc's API
# --------------------------------------
# We choose the Stacked Shortcut Algorithm to debug the pipeline.

from bugdoc.algos.shortcut import Shortcut
from bugdoc.algos.stacked_shortcut import StackedShortcut
from bugdoc.algos.debugging_decision_trees import DebuggingDecisionTrees

import shutil
from ml_api_example import f_score_look_up2
import pandas as pd
import json
import os
        
# %%
# Parameter space definition
# ---------------------------
# The parameter-values that BugDoc tries can be retrieved in two ways:
# From previous executions of the pipeline or specifying the all possible values each parameter can take.
# In the following, we provide the entry point of the pipeline and a dictionary with the parameter names as keys
# and a list of parameter-values as the corresponding value.


filename = 'ml_pipeline.json'
parameter_space = {
    'missing_value': [
        '1', #'drop',
        '2', #'mean',
        '3', #'median',
        '4', #'most_frequent',
        '5', #'knn_1',
        '6', #'knn_5',
        '7', #'knn_10',
        '8', #'knn_20',
        '9', #'knn_30'
    ],
    'normalization': [
        '1', #'none',
        '2', #'StandardScaler',
        '3', #'RobustScaler',
        '4', #'MaxAbScaler',
        '5', #'MinMaxScaler',
    ],
    'outlier': [
        '1', #'none', 
        '2', #'IsolationForest', 
        '3', #'LOF_1', 
        '4', #'LOF_5', 
        '5', #'LOF_10', 
        '6', #'LOF_20', 
        '7', #'LOF_30'
    ]
}

# %%
# Pipeline Debugging
# ------------------------
# We initialize the Stacked Shortcut Algorithm object and run it passing the pipeline entry point and
# the parameter space. The Algorithm will generate new pipeline instances and exchange messages with the *Worker*
# script to execute and evaluate the instances. This process will be blocked if no *Worker* is running.

iter=10
dataset = 'housing'
allrunsdata = 'historical_data_test_profile_lr_sp_adult.csv'
historical_data = pd.read_csv(allrunsdata)
threshold = 0.16

iter_dist = []
parameters = list(parameter_space.keys())

# for i in range(len(historical_data)):
for i in range(1,len(historical_data)):
    row = historical_data.iloc[i]
    result = row['fairness'] <= threshold
    if result:
        iter_dist.append(1) 
    else:
        run = {
            "missing_value": str(int(row['missing_value'])),
            "normalization": str(int(row['normalization'])),
            "outlier": str(int(row['outlier'])),
            "result": str(result)
        }
        # print(run)
        with open(filename, "w") as f:
            json.dump(run, f)
            f.write("\n")

        new_filename = 'ml_pipeline_tmp.json'
        found_iter = 0
        for i in range(1, iter):
            debugger = StackedShortcut(max_iter=i)
            shutil.copy(filename, new_filename)
            root, _iter, _ = debugger.run(new_filename, parameter_space, outputs=["results"])
            
            print(i, root, _iter)
            if len(root) > 0:
                # print("Root cause: ", root)
                # print("Found root cause in iterations: ", _iter)
                # if _iter == 28:
                    # print("Buggy seed: ", run)
                iter_dist.append(_iter)
                found_iter = _iter
                break
        
        if found_iter == 0:
            iter_dist.append(iter)
            # print("Found root cause in iterations: ", ix)
        # else:
            # print("Root cause not found")

print("iter_dist is: ", iter_dist)


# new_filename = 'ml_pipeline_tmp.json'
# ix = 0
# for i in range(1, iter):
#     debugger = StackedShortcut(max_iter=i)
#     shutil.copy(filename, new_filename)
#     root, iter_ret, _ = debugger.run(new_filename, parameter_space, outputs=["results"])
#     if len(root) > 0:
#         print("Root cause: ", root)
#         ix = iter_ret
#         break

# if ix > 0:
#     print("Found root cause in iterations: ", ix)
# else:
#     print("Root cause not found")

# # Original debugging script
# debugger = StackedShortcut(max_iter=iter)
# # debugger = Shortcut(max_iter=iter)
# # debugger = DebuggingDecisionTrees(max_iter=iter)

# result = debugger.run(filename, parameter_space, outputs=["results"])

# # %%
# # Revealing the root cause
# # -------------------------
# # When the algorithm finishes we can display the root cause of error.

# print("Result: ", result)
# # if StackedShortcut or DebuggingDecisionTrees:
# root, iter, _ = result

# print("Root: ", type(root))
# # if Shortcut:
# # root, _ = result

# parameters = list(parameter_space.keys())

# # Root cause for StackedShortcut
# print('Root Cause: \n%s' % (
#     ' OR '.join(
#      [
#         ' AND '.join(
#          [parameters[pair[0]]+' = '+pair[1] for pair in r]
#                     )
#         for r in root])))

# # Root cause for DebuggingDecisionTrees
# # print('Root Cause: \n%s' % (
# #     ' OR '.join(
# #      [
# #         ' AND '.join(
# #          [parameters[pair[0]]+' = '+pair[1] for pair in r[1]]
# #                     )
# #         for r in root])))