# =========================
# ml_worker_online.py
# =========================
"""
Online BugDoc Worker (order-aware, no historical CSV)
=====================================================
Run with:
  python -u ml_worker_online.py 2>&1 | tee worker_log.txt

Config via environment variables (all optional, have defaults):
  BUGDOC_HOST            ZMQ host (default: localhost)
  BUGDOC_PORT_RECV       ZMQ receive port (default: 5557)
  BUGDOC_PORT_SEND       ZMQ send port (default: 5558)
  BUGDOC_DATASET         Dataset name (default: adult)
  BUGDOC_METRIC_TYPE     Metric type (default: sp)
  BUGDOC_EXEC_MODE       Execution mode (default: fail)
  BUGDOC_THRESHOLD       Pass/fail threshold (default: 0.05)
  BUGDOC_BETTER_IS_LOWER 1=lower is better, 0=higher is better (default: 1)
  BUGDOC_DEFAULT_MODEL   Default model strategy index (default: 1)
"""

import os
import ast
import sys
import traceback
import zmq
import numpy as np
from bugdoc.utils.utils import record_pipeline_run

# ==============================================================
# UNBUFFERED OUTPUT — must happen before any prints
# ==============================================================
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass  # Python < 3.7

def log(*args):
    """Print to both stdout and stderr with immediate flush."""
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)

log("=" * 60)
log(f">>> WORKER STARTING")
log(f">>> FILE : {os.path.abspath(__file__)}")
log(f">>> PID  : {os.getpid()}")
log("=" * 60)

# --- Path bootstrap ---
from pathlib import Path
_here = Path(__file__).resolve()

root = None
p    = _here
for _ in range(6):
    if (p / "pipeline_execution.py").exists():
        root = p
        break
    p = p.parent
if root is None:
    root = _here.parents[2]
sys.path.insert(0, str(root))
log(f">>> BOOTSTRAP | root = {root}")

from pipeline_execution import PipelineExecutor
log(">>> IMPORT    | PipelineExecutor OK")

# -----------------------------
# Static config (not threshold-sensitive)
# -----------------------------
HOST        = os.getenv("BUGDOC_HOST",        "localhost")
PORT_RECV   = os.getenv("BUGDOC_PORT_RECV",   "5557")
PORT_SEND   = os.getenv("BUGDOC_PORT_SEND",   "5558")
DATASET     = os.getenv("BUGDOC_DATASET",     "adult")
METRIC_TYPE = os.getenv("BUGDOC_METRIC_TYPE", "sp")
EXEC_MODE   = os.getenv("BUGDOC_EXEC_MODE",   "fail")

log(f">>> CONFIG | HOST={HOST} PORT_RECV={PORT_RECV} PORT_SEND={PORT_SEND}")
log(f">>> CONFIG | DATASET={DATASET} METRIC_TYPE={METRIC_TYPE} EXEC_MODE={EXEC_MODE}")

# NOTE: THRESHOLD, BETTER_IS_LOWER, DEFAULT_MODEL_STRAT are read
# dynamically inside the loop so env changes take effect per-message
# without restarting the worker.

# -----------------------------
# ZMQ sockets
# -----------------------------
context  = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.connect(f"tcp://{HOST}:{PORT_RECV}")
sender   = context.socket(zmq.PUSH)
sender.connect(f"tcp://{HOST}:{PORT_SEND}")
log(f">>> ZMQ | PULL on tcp://{HOST}:{PORT_RECV}")
log(f">>> ZMQ | PUSH on tcp://{HOST}:{PORT_SEND}")
log(f">>> READY | Waiting for messages...")

# -----------------------------
# Executor cache
# -----------------------------
_executor_cache = {}
_eval_counter   = 0

def get_executor(pipeline_order_base):
    key = (tuple(pipeline_order_base), DATASET, METRIC_TYPE, EXEC_MODE)
    if key not in _executor_cache:
        log(f">>> EXECUTOR | New executor | order={pipeline_order_base}")
        _executor_cache[key] = PipelineExecutor(
            pipeline_type='ml',
            dataset_name=DATASET,
            metric_type=METRIC_TYPE,
            pipeline_ord=pipeline_order_base,
            execution_type=EXEC_MODE,
        )
    return _executor_cache[key]

# -----------------------------
# Helpers
# -----------------------------
def decide_pass(metric_value, threshold, better_is_lower):
    if metric_value is None:
        return False
    return (np.abs(metric_value) <= threshold) if better_is_lower else (np.abs(metric_value) >= threshold)

def parse_message(data):
    fields     = data.split("|")
    filename   = fields[0]
    values     = ast.literal_eval(fields[1])
    parameters = ast.literal_eval(fields[2])
    return filename, values, parameters

def strip_suffix(name):
    return name.split("#", 1)[0] if "#" in name else name

def split_params(params, values):
    """Split into (strategy pairs, position pairs). pos_* keys never go into strat."""
    strat, pos = [], []
    for k, v in zip(params, values):
        if str(k).startswith("pos_"):
            base = str(k)[4:]
            try:
                pos.append((base, int(v)))
            except Exception:
                pos.append((base, 10**6))
        else:
            strat.append((strip_suffix(k), str(v)))
    return strat, pos

def compute_order(steps, positions, baseline_order):
    """Sort real step names by assigned position. Never returns pos_* keys."""
    pos_map    = {b: p for (b, p) in positions}
    base_index = {b: i for i, b in enumerate(baseline_order)}
    steps      = [s for s in steps if s in base_index]
    return sorted(
        steps,
        key=lambda s: (pos_map.get(s, base_index[s] + 1), base_index[s], s)
    )

def build_eval_order_and_values(parameters, values, default_model_strat):
    """
    Build eval_order and eval_values from raw message.
    Always returns real step names only — pos_* keys never leak through.
    Raises ValueError if sanity check fails.
    """
    strat_pairs, pos_pairs = split_params(parameters, values)

    # Baseline: strategy keys only, no pos_*
    baseline_order = [
        strip_suffix(k) for k in parameters
        if not str(k).startswith("pos_")
    ]

    step_names  = [s for (s, _) in strat_pairs]
    eval_order  = compute_order(step_names, pos_pairs, baseline_order)

    strat_map   = {s: v for (s, v) in strat_pairs}
    eval_values = [int(strat_map[s]) for s in eval_order]

    eval_order.append('model')
    eval_values.append(int(default_model_strat))

    # Sanity check
    bad = [s for s in eval_order if str(s).startswith("pos_")]
    if bad:
        raise ValueError(f"pos_* keys leaked into eval_order: {bad}")

    return eval_order, eval_values

def print_pipeline_evaluation(counter, eval_order, eval_values, utility,
                               threshold, better_is_lower, result, error=None):
    abs_util  = np.abs(utility) if utility is not None else None
    direction = "<=" if better_is_lower else ">="
    verdict   = "PASS" if result else "FAIL"

    log("=" * 60)
    log(f"[EVAL #{counter}]")
    if error:
        log(f"  *** ERROR: {error} ***")
    log(f"  {'STEP':<25}  STRATEGY")
    log(f"  {'-'*40}")
    for step, val in zip(eval_order, eval_values):
        tag = "  <-- model" if step == "model" else ""
        log(f"  {step:<25}  {val}{tag}")
    log(f"  {'-'*40}")
    log(f"  Raw utility  : {utility}")
    log(f"  abs(utility) : {abs_util}")
    log(f"  Threshold    : {threshold}  (better_is_lower={better_is_lower})")
    log(f"  Decision     : {abs_util} {direction} {threshold}  =>  *** {verdict} ***")
    log("=" * 60)

# -----------------------------
# Main loop
# -----------------------------
while True:
    data = receiver.recv_string()
    _eval_counter += 1
    log(f">>> MSG #{_eval_counter} | Received (len={len(data)})")

    filename, values, parameters = parse_message(data)

    # Read dynamically per message so env changes take effect without restart
    threshold           = float(os.getenv("BUGDOC_THRESHOLD",      "0.1"))
    better_is_lower     = os.getenv("BUGDOC_BETTER_IS_LOWER", "1") == "1"
    default_model_strat = os.getenv("BUGDOC_DEFAULT_MODEL",        "1")

    log(f">>> MSG #{_eval_counter} | threshold={threshold} better_is_lower={better_is_lower}")

    # Safe fallback eval_order — real steps only, no pos_* ever
    error_msg   = None
    utility     = None
    result      = False
    eval_order  = [strip_suffix(k) for k in parameters if not str(k).startswith("pos_")]
    eval_values = []

    try:
        eval_order, eval_values = build_eval_order_and_values(
            parameters, values, default_model_strat
        )
        log(f">>> MSG #{_eval_counter} | eval_order={eval_order}")
        log(f">>> MSG #{_eval_counter} | eval_values={eval_values}")

        executor = get_executor(eval_order)
        utility  = executor.current_par_lookup(eval_order, eval_values)
        result   = decide_pass(utility, threshold, better_is_lower)

    except Exception as e:
        error_msg = str(e)
        log(f">>> MSG #{_eval_counter} | EXCEPTION: {error_msg}")
        traceback.print_exc(file=sys.stdout)
        traceback.print_exc(file=sys.stderr)
        utility = None
        result  = False

    print_pipeline_evaluation(
        counter=_eval_counter,
        eval_order=eval_order,
        eval_values=eval_values,
        utility=utility,
        threshold=threshold,
        better_is_lower=better_is_lower,
        result=result,
        error=error_msg,
    )

    record_pipeline_run(filename, values, parameters, result)

    out = list(values) + [result]
    sender.send_string(str(out))
    log(f">>> MSG #{_eval_counter} | Reply sent: result={result}")