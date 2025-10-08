"""
Worker script
===========================
Receives pipeline configurations from BugDoc, executes/evaluates them,
and returns the result back to BugDoc.
"""

import ast
import os
import sys
import traceback
import zmq
import pandas as pd
from bugdoc.utils.utils import record_pipeline_run

# Your pipeline API (keep signature compatible)
from ml_api_example import execute_pipeline

# -----------------------------
# Config (env-overridable)
# -----------------------------
HOST = os.getenv("BUGDOC_HOST", "localhost")
PORT_RECV = os.getenv("BUGDOC_PORT_RECV", "5557")
PORT_SEND = os.getenv("BUGDOC_PORT_SEND", "5558")

DATASET = os.getenv("BUGDOC_DATASET", "housing")
HISTORY_FILE = os.getenv("BUGDOC_HISTORY_FILE", "Bugdoc_test_lr_sp_adult.csv")

# Metric & decision rule
#   METRIC_COL: the numeric column used to decide pass/fail (e.g., 'utility_rmse', 'fairness', etc.)
#   THRESHOLD: numeric cutoff
#   BETTER_IS_LOWER: '1' if lower is better (RMSE), '0' if higher is better (e.g., F1)
METRIC_COL = os.getenv("BUGDOC_METRIC_COL", "fairness")
THRESHOLD = float(os.getenv("BUGDOC_THRESHOLD", "0.06"))
BETTER_IS_LOWER = os.getenv("BUGDOC_BETTER_IS_LOWER", "1") == "1"

context = zmq.Context()

receiver = context.socket(zmq.PULL)
receiver.connect(f"tcp://{HOST}:{PORT_RECV}")

sender = context.socket(zmq.PUSH)
sender.connect(f"tcp://{HOST}:{PORT_SEND}")

# -----------------------------
# Load historical data
# -----------------------------
historical_data = pd.read_csv(HISTORY_FILE)

# -----------------------------
# Helper: evaluate a result row against the rule
# (Kept here if your execute_pipeline uses historical lookups but not the rule)
# -----------------------------
def decide_pass(metric_value: float) -> bool:
    if pd.isna(metric_value):
        return False
    return (metric_value <= THRESHOLD) if BETTER_IS_LOWER else (metric_value >= THRESHOLD)

# -----------------------------
# Main loop
# -----------------------------
while True:
    data = receiver.recv_string()
    fields = data.split("|")
    filename = fields[0]
    values = ast.literal_eval(fields[1])      # list of parameter values (strings/ints)
    parameters = ast.literal_eval(fields[2])  # list of parameter names (strings)

    try:
        configuration = {parameters[i]: values[i] for i in range(len(parameters))}
        # Keep the same API call you already use; it can look into `historical_data`,
        # and you can update it later to also take (METRIC_COL, THRESHOLD, BETTER_IS_LOWER) if you want.
        result = execute_pipeline(configuration, historical_data, THRESHOLD)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        result = False

    # Record and send back
    record_pipeline_run(filename, values, parameters, result)
    values.append(result)
    sender.send_string(str(values))
