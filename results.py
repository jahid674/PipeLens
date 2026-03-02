import json
import pandas as pd
import numpy as np
import csv
import sys
import time
import logging
from opaque_optimizer import OpaqueOptimizer
#from RL_glassbox import GlassBoxOptimizer
from glassbox_optimizer import GlassBoxOptimizer
from grid_online_randomized import GridSearch
from pipeline_execution import PipelineExecutor
np.random.seed(42)
import random


with open('config.json', 'r') as f:
    config = json.load(f)

method=config["method"]
dataset_name = config["dataset_name"]
model_type = config["model_type"]
metric_type = config["metric_type"]
pipeline_type = config["pipeline_type"]
pipeline_order = config["pipeline_order"]
new_components = config["new_components"]


utility_goals = config["f_goals"][dataset_name]

filename_train = config["paths"]["train_data"].format(
    model_type=model_type, metric_type=metric_type, dataset_name=dataset_name)
filename_test = config["paths"]["test_data"].format(
    model_type=model_type, metric_type=metric_type, dataset_name=dataset_name)
metric_path = config["paths"]["metric_output"].format(method=method,
    model_type=model_type, metric_type=metric_type, dataset_name=dataset_name)

logging.basicConfig(filename='logs/revision/0.1'+'PiepLensFinal'+"_"+dataset_name+'_'+model_type+'_'+metric_type+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


#p = OpaqueOptimizer(dataset_name, model_type, metric_type, pipeline_type, pipeline_order,
#                    filename_train, filename_test)
p = GlassBoxOptimizer(dataset_name, model_type, metric_type, pipeline_type, pipeline_order, filename_train, filename_test, new_components)

historical_data = pd.read_csv(filename_test)

grid = GridSearch(
    dataset_name=dataset_name,
    pipeline_order=pipeline_order,
    metric_type=metric_type,
    pipeline_type=pipeline_type
)

def seed_to_pipeline(seed_row, pipeline_order_names):
    """
    Convert a row from the CSV into (failing_order, failing_vec)
    using config['pipeline_order'] as the component names.

    Assumption:
      - first len(pipeline_order_names) columns of seed_row are strategy indices
    """
    order = list(pipeline_order_names)
    k = len(order)

    vec = [int(x) for x in seed_row[:k]]

    # enforce 'model' last and present
    if "model" in order:
        m_idx = order.index("model")
        if m_idx != len(order) - 1:
            m_val = vec[m_idx]
            order.pop(m_idx); vec.pop(m_idx)
            order.append("model"); vec.append(m_val)
    else:
        order.append("model")
        vec.append(1)

    return order, vec

# n_samples = min(100, len(historical_data))
# sampled_seeds = historical_data.sample(
#     n=n_samples,
#     random_state=42   # reproducible
# ).values

# k = len(config["pipeline_order"])   # number of pipeline parameters

# # Ensure numeric comparison
# df_num = historical_data.iloc[:, :k-1].apply(pd.to_numeric, errors="coerce")

# # Keep rows where ALL first k entries > 1
# valid_mask = (df_num > 1).all(axis=1)

# filtered_data = historical_data.loc[valid_mask]

# # Sample
# n_samples = min(20, len(filtered_data))
# sampled_seeds = filtered_data.sample(
#     n=n_samples,
#     random_state=42
# ).values

# # print("Pipeline length:", k)
# # print("Original rows:", len(historical_data))
# # print("Valid rows:", len(filtered_data))

# # print("Sample rows (first 5):")
# print(len(sampled_seeds))




with open(metric_path, 'w') as f:
    csv_writer = csv.writer(f)
    for f_goal in utility_goals:
        rank_idistr, rank_fdistr, gs_idistr, gs_fdistr = [], [], [], []
        #p.utiliy_threshold = f_goal
        profile_itr = {}

        for seed_ in historical_data[:8].values:
            print('seed',seed_)
            #seen = set()
            p.optimize(seed_, f_goal)
            rank_idistr.append(p.rank_iter)
            rank_fdistr.append(p.rank_f)
            # logging.info(f"pipeline{seed_[:3]}")

            # failing_order, failing_vec = seed_to_pipeline(seed_, pipeline_order)
            # rng = random.Random(42)

            # #One-change-per-iteration search FROM THIS SEED
            # gs_iter, gs_f, best_order, best_vec = grid.grid_search(
            #         f_goal=f_goal,
            #         failing_order=failing_order,
            #         failing_vec=failing_vec,
            #         new_components=new_components,
            #         max_iters=1000
            #     )

            # gs_idistr.append(gs_iter)
            # gs_fdistr.append(gs_f)

        rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75, 100], interpolation='midpoint')
        rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75, 100], interpolation='midpoint')
        # g_iquartiles = np.percentile(gs_idistr, [25, 50, 75, 100], interpolation='midpoint')
        # g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75, 100], interpolation='midpoint')

        csv_writer.writerow(["Utility Goal", "Method", "Iteration", "Value"])
        p.write_quartiles(csv_writer, "PipeLens", "iterations", rank_iquartiles, f_goal, utility_goals)
        p.write_quartiles(csv_writer, "PipeLens", "Fairness", rank_fquartiles, f_goal, utility_goals)
        csv_writer.writerow([])
        # p.write_quartiles(csv_writer, "Grid", "iterations", g_iquartiles, f_goal, utility_goals)
        # p.write_quartiles(csv_writer, "Grid", "Fairness", g_fquartiles, f_goal, utility_goals)
        # csv_writer.writerow([])
