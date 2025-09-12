import ast
import sys
import traceback
import zmq
import pandas as pd
from bugdoc.utils.utils import record_pipeline_run

from ml_api_example import execute_pipeline

# ------------------ CONFIG ------------------ #
HOST = "localhost"
RECEIVE_PORT = "5557"
SEND_PORT = "5558"

DATASET = "housing"  # tag only
CSV_PATH = 'bugdoc_test_sim_historical_data_test_profile_reg_rmse_housing.csv'
THRESHOLD = 170.0     # target metric bound (fairness / rmse-like) — lower is better
METRIC_COL = "utility_rmse"  # column in CSV holding the metric being thresholded

# ------------------------------------------- #

context = zmq.Context()

receiver = context.socket(zmq.PULL)
receiver.connect(f"tcp://{HOST}:{RECEIVE_PORT}")

sender = context.socket(zmq.PUSH)
sender.connect(f"tcp://{HOST}:{SEND_PORT}")

historical_data = pd.read_csv(CSV_PATH)

# Process tasks forever
while True:
    data = receiver.recv_string()
    fields = data.split("|")
    filename = fields[0]
    values = ast.literal_eval(fields[1])
    parameters = ast.literal_eval(fields[2])

    try:
        configuration = {parameters[i]: values[i] for i in range(len(parameters))}
        result = execute_pipeline(configuration, historical_data, THRESHOLD, metric_col=METRIC_COL)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        result = False

    record_pipeline_run(filename, values, parameters, result)
    values.append(result)
    sender.send_string(str(values))
