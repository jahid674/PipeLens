#!/usr/bin/env python3
# coding: utf-8
"""
Debugger script
===========================

Defines the parameter space (aligned to the Learn2Clean-style blocks),
writes a failing "current pipeline" JSON, then runs a BugDoc algorithm
(StackedShortcut by default) to search for root causes.
"""

import json
import os
import shutil

import pandas as pd
from bugdoc.algos.stacked_shortcut import StackedShortcut
# You can also import:
# from bugdoc.algos.shortcut import Shortcut
# from bugdoc.algos.debugging_decision_trees import DebuggingDecisionTrees

# ------------------ CONFIG ------------------ #
FILENAME = "ml_pipeline.json"          # where we write the failing instance
TMP_FILENAME = "ml_pipeline_tmp.json"  # BugDoc copies/reads
CSV_PATH = "historical_data/noise/bugdoc_test_sim_historical_data_test_profile_reg_rmse_housing.csv"
DATASET_TAG = "housing"
THRESHOLD = 170.0
MAX_ITER_SCAN = 100       # progressively increase until root is found
ROW_START = 0             # you can change which rows of historical_data to probe
ROW_STOP = 1              # exclusive end (default: just the first row)
# ------------------------------------------- #

# ---------------------------------------------------------------------
# Parameter space aligned with your new Learn2Clean blocks
# (Codes are strings "1", "2", ... so they match integer-coded CSV columns)
# ---------------------------------------------------------------------
parameter_space = {
    # Imputation (9)
    "imputation": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],  # DROP, MEAN, MEDIAN, MF, KNN_1..KNN_30
    # Deduplication (2)
    "dedup": ["1", "2"],  # NONE, FIRST
    # Lowercasing (2)
    "lowercase": ["1", "2"],  # NONE, LC
    # Translation (2)
    "translate": ["1", "2"],  # NONE, GT
    # Punctuation (2)
    "punct": ["1", "2"],  # NONE, PR
    # Stopword (2)
    "stopword": ["1", "2"],  # NONE, SW
    # Spell check (2)
    "spell": ["1", "2"],  # NONE, SC
    # Tokenization (3)
    "tokenize": ["1", "2", "3"],  # NONE, WS, NLTK
    # Unit conversion (2)
    "unitconvert": ["1", "2"],  # NONE, UC
    # Normalization (5)
    "normalize": ["1", "2", "3", "4", "5"],  # NONE, SS, RS, MA, MM
    # Outliers (7)
    "outlier": ["1", "2", "3", "4", "5", "6", "7"],  # NONE, IF, LOF_1..LOF_30
}

# If your CSV uses different column names for the parameters above,
# rename either the keys here or the CSV columns so they match exactly.

# ---------------------------------------------------------------------
# Main Debugging Loop
# ---------------------------------------------------------------------
def main():
    historical_data = pd.read_csv(CSV_PATH)
    iters_found = []

    rows = historical_data.iloc[ROW_START:ROW_STOP]

    for _, row in rows.iterrows():
        # Evaluate whether row already meets the threshold.
        # We assume the CSV has:
        #   - one column per parameter in parameter_space (int-coded),
        #   - a metric column "fairness" (or change THRESHOLD/METRIC in worker/API).
        metric_val = float(row.get("fairness", float("inf")))
        meets = metric_val <= THRESHOLD

        if meets:
            iters_found.append(1)
            continue

        # Build a failing configuration dict using this row (stringify values)
        run = {}
        for p in parameter_space.keys():
            if p in row:
                v = int(row[p]) if pd.notna(row[p]) else 1
            else:
                # default to the first option if column absent
                v = 1
            run[p] = str(v)
        run["result"] = str(False)  # explicitly mark as a failing run input

        # Write the failing pipeline instance
        with open(FILENAME, "w") as f:
            json.dump(run, f)
            f.write("\n")

        # Progressively increase max_iter until a root cause is found or limit reached
        found_iter = 0
        for i in range(1, int(MAX_ITER_SCAN) + 1):
            debugger = StackedShortcut(max_iter=i)
            shutil.copy(FILENAME, TMP_FILENAME)
            root, _iter, _ = debugger.run(TMP_FILENAME, parameter_space, outputs=["results"])
            if len(root) > 0:
                iters_found.append(_iter)
                found_iter = _iter
                break

        if found_iter == 0:
            iters_found.append(int(MAX_ITER_SCAN))

    print("iter_dist is:", iters_found)


if __name__ == "__main__":
    main()
