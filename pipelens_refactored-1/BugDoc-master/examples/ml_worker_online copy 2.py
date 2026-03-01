# =========================
# bugdoc_worker_order_aware.py
# =========================
"""
Online BugDoc Worker (order-aware, no historical CSV)
=====================================================
- Receives (filename | values | parameters) over ZMQ from the debugger
- Parameters may include strategy keys and 'pos_<step>' keys.
- Incoming configs MAY include 'model' but we IGNORE it (model is fixed).
- Computes the evaluation order from positions, appends model='1', and evaluates.
- Logs per-eval timing into the same JSON line:
    eval_time_sec, other_time_sec, utility
  (BugDoc ignores extra keys safely.)
"""

import os
import ast
import sys
import json
import time
import traceback
import zmq
import numpy as np

# --- Path bootstrap to import pipeline_execution.py from repo root ---
from pathlib import Path
_here = Path(__file__).resolve()
import sys as _sys

root = None
p = _here
for _ in range(6):
    if (p / "pipeline_execution.py").exists():
        root = p
        break
    p = p.parent
if root is None:
    root = _here.parents[2]
_sys.path.insert(0, str(root))
# --- end bootstrap ---

from pipeline_execution import PipelineExecutor

# -----------------------------
# Config (env-overridable)
# -----------------------------
HOST        = os.getenv("BUGDOC_HOST", "localhost")
PORT_RECV   = os.getenv("BUGDOC_PORT_RECV", "5557")
PORT_SEND   = os.getenv("BUGDOC_PORT_SEND", "5558")

DATASET     = os.getenv("BUGDOC_DATASET", "adult")
METRIC_TYPE = os.getenv("BUGDOC_METRIC_TYPE", "sp")
EXEC_MODE   = os.getenv("BUGDOC_EXEC_MODE", "fail")

THRESHOLD       = float(os.getenv("BUGDOC_THRESHOLD", ".2"))
BETTER_IS_LOWER = os.getenv("BUGDOC_BETTER_IS_LOWER", "1") == "1"

DEFAULT_MODEL_STRAT = os.getenv("BUGDOC_DEFAULT_MODEL", "1")

# -----------------------------
# ZMQ sockets
# -----------------------------
context  = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.connect(f"tcp://{HOST}:{PORT_RECV}")

sender   = context.socket(zmq.PUSH)
sender.connect(f"tcp://{HOST}:{PORT_SEND}")

# -----------------------------
# Executor cache
# -----------------------------
_executor_cache = {}

def get_executor(pipeline_order_base):
    key = (tuple(pipeline_order_base), DATASET, METRIC_TYPE, EXEC_MODE)
    if key not in _executor_cache:
        _executor_cache[key] = PipelineExecutor(
            pipeline_type='ml',
            dataset_name=DATASET,
            metric_type=METRIC_TYPE,
            pipeline_ord=pipeline_order_base,
            execution_type=EXEC_MODE,
        )
    return _executor_cache[key]

def decide_pass(metric_value: float) -> bool:
    if metric_value is None:
        return False
    # Your current semantics:
    # lower is better -> pass if abs(metric) <= threshold
    return (np.abs(metric_value) <= THRESHOLD) if BETTER_IS_LOWER else (np.abs(metric_value) >= THRESHOLD)

def parse_message(data):
    """
    returns filename, values(list[str]), parameters(list[str])
    """
    fields     = data.split("|")
    filename   = fields[0]
    values     = ast.literal_eval(fields[1])
    parameters = ast.literal_eval(fields[2])
    return filename, values, parameters

def strip_suffix(name: str) -> str:
    return name.split("#", 1)[0] if "#" in name else name

def is_model_key(k: str) -> bool:
    kk = strip_suffix(str(k))
    return kk == "model"

def split_params(params, values):
    """
    Split into strategy pairs and position pairs.
    Returns: ([(step, strat_str), ...], [(base, pos_int), ...])
    """
    strat, pos = [], []
    for k, v in zip(params, values):
        k_str = str(k)

        # ignore model if BugDoc sends it (we always append it ourselves)
        if is_model_key(k_str):
            continue

        if k_str.startswith("pos_"):
            base = k_str[4:]  # remove 'pos_'
            base = strip_suffix(base)
            try:
                pos.append((base, int(v)))
            except Exception:
                pos.append((base, 10**6))
        else:
            strat.append((strip_suffix(k_str), str(v)))
    return strat, pos

def compute_order(steps, positions, baseline_order):
    pos_map = {b: p for (b, p) in positions}
    base_index = {b: i for i, b in enumerate(baseline_order)}
    steps = [s for s in steps if s in base_index]

    ordered = sorted(
        steps,
        key=lambda s: (pos_map.get(s, base_index[s] + 1), base_index[s], s)
    )
    return ordered

def record_pipeline_run_extended(filename, values, parameters, result_bool, extra_fields=None):
    """
    Writes one JSON line:
      {param: value, ..., "result": "True"/"False", ...extras }
    BugDoc load_runs expects result is a STRING because it does eval(result_value).
    """
    paramDict = { parameters[i]: values[i] for i in range(len(parameters)) }
    paramDict["result"] = str(bool(result_bool))  # IMPORTANT: string

    if extra_fields:
        for k, v in extra_fields.items():
            # store as strings to avoid json/typing surprises
            paramDict[k] = str(v)

    with open(filename, "a") as f:
        f.write(json.dumps(paramDict) + "\n")

TOTAL_EVALS = 0

while True:
    data = receiver.recv_string()
    filename, values, parameters = parse_message(data)

    t_total0 = time.perf_counter()
    eval_time = 0.0
    other_time = 0.0
    utility = None

    try:
        # 1) split into strategies and positions (ignore model if present)
        strat_pairs, pos_pairs = split_params(parameters, values)

        # 2) derive baseline order from the present strategy keys
        baseline_order = []
        for k in parameters:
            k_str = str(k)
            if k_str.startswith("pos_"):
                continue
            if is_model_key(k_str):
                continue
            baseline_order.append(strip_suffix(k_str))

        # 3) compute evaluation order using positions
        step_names = [s for (s, _) in strat_pairs]
        eval_order = compute_order(step_names, pos_pairs, baseline_order)

        # 4) align strategy values to eval_order
        strat_map = {s: v for (s, v) in strat_pairs}
        eval_values = [int(strat_map[s]) for s in eval_order]

        # 5) append model fixed
        eval_order.append("model")
        eval_values.append(int(DEFAULT_MODEL_STRAT))

        # 6) evaluate
        TOTAL_EVALS += 1

        t_eval0 = time.perf_counter()
        executor = get_executor(eval_order)
        utility  = executor.current_par_lookup(eval_order, eval_values)
        eval_time = time.perf_counter() - t_eval0

        # 7) pass/fail
        result = decide_pass(utility)

        t_total1 = time.perf_counter()
        other_time = (t_total1 - t_total0) - eval_time

    except Exception:
        traceback.print_exc(file=sys.stdout)
        result = False
        t_total1 = time.perf_counter()
        other_time = (t_total1 - t_total0) - eval_time

    # Log to file used by BugDoc
    record_pipeline_run_extended(
        filename,
        values,
        parameters,
        result_bool=result,
        extra_fields={
            "utility": utility if utility is not None else "None",
            "eval_time_sec": f"{eval_time:.6f}",
            "other_time_sec": f"{other_time:.6f}",
        }
    )

    # Send back: original values plus boolean result (BugDoc protocol expects this)
    out = list(values) + [result]
    sender.send_string(str(out))
