import json
import pandas as pd
import numpy as np
import csv
import sys
import logging
from opaque_optimizer import OpaqueOptimizer
from glassbox_optimizer import GlassBoxOptimizer
from gridsearch import GridSearch
from pipeline_execution import PipelineExecutor

with open('config.json', 'r') as f:
    config = json.load(f)


dataset_name = config["dataset_name"]
model_type = config["model_type"]
metric_type = config["metric_type"]
pipeline_type = config["pipeline_type"]
pipeline_order = config["pipeline_order"]

f_goals = config["f_goals"][dataset_name]

filename_train = config["paths"]["train_data"].format(
    model_type=model_type, metric_type=metric_type, dataset_name=dataset_name)
filename_test = config["paths"]["test_data"].format(
    model_type=model_type, metric_type=metric_type, dataset_name=dataset_name)
metric_path = config["paths"]["metric_output"].format(
    model_type=model_type, metric_type=metric_type, dataset_name=dataset_name)
log_path = config["paths"]["log_file"].format(
    model_type=model_type, metric_type=metric_type, dataset_name=dataset_name)

logging.basicConfig(filename=log_path, filemode='w', level=logging.DEBUG)


p = OpaqueOptimizer(dataset_name, model_type, metric_type, pipeline_type, pipeline_order,
                    filename_train, filename_test)
historical_data = pd.read_csv(filename_test)
grid = GridSearch(historical_data, pipeline_order, metric_type)

with open(metric_path, 'w') as f:
    csv_writer = csv.writer(f)
    for f_goal in f_goals:
        rank_idistr, rank_fdistr, gs_idistr, gs_fdistr = [], [], [], []
        for seed_ in historical_data.values.tolist():
            seen = set()
            gs_iter, gs_f = grid.grid_search(f_goal, seen)
            gs_idistr.append(gs_iter)
            gs_fdistr.append(gs_f)

            p.optimize(seed_, f_goal)
            rank_idistr.append(p.rank_iter)
            rank_fdistr.append(p.rank_f)

        rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75, 100], interpolation='midpoint')
        rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75, 100], interpolation='midpoint')
        g_iquartiles = np.percentile(gs_idistr, [25, 50, 75, 100], interpolation='midpoint')
        g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75, 100], interpolation='midpoint')

        csv_writer.writerow(["Fairness Goal", "Grid search", "Iteration", "Value"])
        p.write_quartiles(csv_writer, "ranking", "iterations", rank_iquartiles, f_goal, f_goals)
        p.write_quartiles(csv_writer, "ranking", "Fairness", rank_fquartiles, f_goal, f_goals)
        csv_writer.writerow([])
        p.write_quartiles(csv_writer, "grid search", "iterations", g_iquartiles, f_goal, f_goals)
        p.write_quartiles(csv_writer, "grid search", "Fairness", g_fquartiles, f_goal, f_goals)
        csv_writer.writerow([])
