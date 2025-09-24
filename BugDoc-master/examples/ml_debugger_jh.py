"""
Debugger script
===========================
Entry point to define parameter space & invoke BugDoc algorithms to debug the pipeline.
Now generalized to new/mutable pipeline configurations.
"""

from bugdoc.algos.stacked_shortcut import StackedShortcut
# Optional alternatives:
# from bugdoc.algos.shortcut import Shortcut
# from bugdoc.algos.debugging_decision_trees import DebuggingDecisionTrees

import os
import json
import shutil
import pandas as pd

# If you have any helper lookups (kept for compatibility)
# from ml_api_example import f_score_look_up2  # not strictly needed here

# -----------------------------
# Config (env-overridable)
# -----------------------------
FILENAME = os.getenv("BUGDOC_ENTRY_FILE", "ml_pipeline.json")
TMP_FILENAME = os.getenv("BUGDOC_ENTRY_FILE_TMP", "ml_pipeline_tmp.json")

DATASET = os.getenv("BUGDOC_DATASET", "housing")
HISTORY_FILE = os.getenv("BUGDOC_HISTORY_FILE", "bugdoc_test_sim_historical_data_test_profile_lr_rmse_housing.csv")

# Debugging iterations
MAX_OUTER_ITER = int(os.getenv("BUGDOC_MAX_OUTER_ITER", "100"))

# Metric decision rule
METRIC_COL = os.getenv("BUGDOC_METRIC_COL", "fairness")   # e.g., 'utility_rmse', 'fairness'
THRESHOLD = float(os.getenv("BUGDOC_THRESHOLD", "150"))
BETTER_IS_LOWER = os.getenv("BUGDOC_BETTER_IS_LOWER", "1") == "1"

# -----------------------------
# Load historical data
# -----------------------------
historical_data = pd.read_csv(HISTORY_FILE)

# -----------------------------
# Infer parameter space automatically
# - All columns except METRIC_COL are treated as parameters
# - Values are the unique observed values (cast to str for BugDoc)
# -----------------------------
def infer_parameter_space(df: pd.DataFrame, metric_col: str):
    param_cols = [c for c in df.columns if c != metric_col]
    space = {}
    for col in param_cols:
        # Use unique observed values, keep stable ordering
        uniq = pd.unique(df[col].dropna())
        # Normalize numeric-like values to ints if possible, then to strings (BugDoc expects strings)
        def _coerce(v):
            try:
                iv = int(v)
                if float(v) == float(iv):
                    return str(iv)
            except Exception:
                pass
            return str(v)
        values = sorted({_coerce(v) for v in uniq})
        space[col] = values
    return space, param_cols

parameter_space, parameters = infer_parameter_space(historical_data, METRIC_COL)

# -----------------------------
# Decision rule for a row
# -----------------------------
def row_passes(row) -> bool:
    m = row[METRIC_COL]
    if pd.isna(m):
        return False
    return (m <= THRESHOLD) if BETTER_IS_LOWER else (m >= THRESHOLD)

# -----------------------------
# Debug loop:
# For each historical row that FAILS the metric rule,
#  - write an entry file with its parameter settings
#  - run StackedShortcut with increasing max_iter until a root is found or limit reached
# Collect how many iterations were needed (iter_dist).
# -----------------------------
iter_dist = []

# Start at 1 to mimic your previous loop style (skip header row semantics if any)
for idx in [1, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,10000, 12000, 14000, 16000]:#len(historical_data)):
    row = historical_data.iloc[idx]

    if row_passes(row):
        iter_dist.append(1)  # trivial: already satisfies threshold
        continue

    # Build the run config with all parameters (strings) + result=False
    run = {p: str(int(row[p])) if pd.api.types.is_numeric_dtype(historical_data[p]) else str(row[p])
           for p in parameters}
    run["result"] = "False"

    with open(FILENAME, "w") as f:
        json.dump(run, f)
        f.write("\n")

    found_iter = 0
    # Try progressively larger max_iter (often faster than one huge max_iter)
    for i in range(1, MAX_OUTER_ITER):
        debugger = StackedShortcut(max_iter=i)
        shutil.copy(FILENAME, TMP_FILENAME)
        root, _iter, _ = debugger.run(TMP_FILENAME, parameter_space, outputs=["results"])

        print(i, root, _iter)
        if len(root) > 0:
            iter_dist.append(_iter)
            found_iter = _iter
            break

    if found_iter == 0:
        iter_dist.append(MAX_OUTER_ITER)

print("iter_dist is: ", iter_dist)

# ------------------------------------------------------------------
# If you want a one-shot run instead of progressive search, uncomment:
# debugger = StackedShortcut(max_iter=MAX_OUTER_ITER)
# result = debugger.run(FILENAME, parameter_space, outputs=["results"])
# root, iters, _ = result
# print("Root Cause:", root)
# ------------------------------------------------------------------
