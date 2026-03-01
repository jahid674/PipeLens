# =========================
# bugdoc_worker_order_aware.py
# =========================
"""
Online BugDoc Worker (ORDER-AWARE, THRESHOLD-BASED)
===================================================
- Receives (filename | values | parameters) over ZMQ from the debugger
- Parameters may include strategy keys and 'pos_<step>' keys (no 'model')
- Computes the evaluation order from positions, appends model='1', evaluates
- PASS/FAIL is based on observed utility vs a threshold (your requirement)
"""

import os
import ast
import sys
import traceback
import zmq
import numpy as np
from bugdoc.utils.utils import record_pipeline_run

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

# Threshold settings (THIS drives pass/fail)
THRESHOLD       = float(os.getenv("BUGDOC_THRESHOLD", "0.05"))
BETTER_IS_LOWER = os.getenv("BUGDOC_BETTER_IS_LOWER", "1") == "1"
USE_ABS         = os.getenv("BUGDOC_USE_ABS", "1") == "1"  # useful for signed SP-diff

DEFAULT_MODEL_STRAT = os.getenv("BUGDOC_DEFAULT_MODEL", "1")

# Fixed baseline order for tie-breaks + default positions (exclude model)
BASELINE_ORDER = [
    "sampling","invalid_value","missing_value","floating_point",
    "distribution_shape","multicollinearity","normalization","outlier",
    "deduplication","punctuation","stopword","lowercase","whitespace",
]

# -----------------------------
# ZMQ sockets
# -----------------------------
context  = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.connect(f"tcp://{HOST}:{PORT_RECV}")

sender   = context.socket(zmq.PUSH)
sender.connect(f"tcp://{HOST}:{PORT_SEND}")

# -----------------------------
# Executor cache (keyed by tuple(order), dataset, metric, mode)
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
    """
    PASS/FAIL based on observed utility vs threshold.
      - BETTER_IS_LOWER=1: pass if (abs(metric) if USE_ABS else metric) <= THRESHOLD
      - BETTER_IS_LOWER=0: pass if (abs(metric) if USE_ABS else metric) >= THRESHOLD
    """
    if metric_value is None:
        return False
    try:
        mv = float(metric_value)
    except Exception:
        return False
    if np.isnan(mv) or np.isinf(mv):
        return False

    mv_cmp = abs(mv) if USE_ABS else mv
    return (mv_cmp <= THRESHOLD) if BETTER_IS_LOWER else (mv_cmp >= THRESHOLD)

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

def split_params(params, values):
    """
    Split into strategy pairs and position pairs.
    - strategy: base step names (no 'model'), values are '1'..'n'
    - position: keys named 'pos_<step>', values are '1'..'K'
    Returns: ([(step, strat_str), ...], [(base, pos_int), ...])
    """
    strat, pos = [], []
    for k, v in zip(params, values):
        k = str(k)
        if k.startswith("pos_"):
            base = k[4:]
            try:
                pos.append((base, int(v)))
            except Exception:
                pos.append((base, 10**6))  # sentinel if malformed
        else:
            strat.append((strip_suffix(k), str(v)))
    return strat, pos

def compute_order(steps, positions, baseline_order):
    """
    steps: list of base step names observed as strategy keys (no 'model')
    positions: list of (base, pos_int)
    baseline_order: reference order to break ties + default fallback
    Returns ordered list of base step names (no 'model').
    """
    pos_map = {b: p for (b, p) in positions}
    base_index = {b: i for i, b in enumerate(baseline_order)}

    # only consider steps that are in baseline_order (safety)
    steps = [s for s in steps if s in base_index]

    ordered = sorted(
        steps,
        key=lambda s: (pos_map.get(s, base_index[s] + 1), base_index[s], s)
    )
    return ordered

while True:
    data = receiver.recv_string()
    filename, values, parameters = parse_message(data)

    try:
        # 1) split into strategies and positions
        strat_pairs, pos_pairs = split_params(parameters, values)

        # 2) compute evaluation order using positions (stable by fixed BASELINE_ORDER)
        step_names = [s for (s, _) in strat_pairs]
        eval_order = compute_order(step_names, pos_pairs, BASELINE_ORDER)

        # 3) align strategy values to eval_order
        strat_map = {s: v for (s, v) in strat_pairs}
        eval_values = [int(strat_map[s]) for s in eval_order]

        # 4) append model at the end for evaluation
        eval_order.append('model')
        eval_values.append(int(DEFAULT_MODEL_STRAT))

        # 5) evaluate pipeline -> returns observed utility
        executor = get_executor(eval_order)
        utility  = executor.current_par_lookup(eval_order, eval_values)

        # 6) pass/fail based on observed utility threshold
        result = decide_pass(utility)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        result = False

    # Record exactly what BugDoc sent (no positions reformatting, no model)
    record_pipeline_run(filename, values, parameters, result)

    # Send back: original values plus the boolean result
    out = list(values) + [result]
    sender.send_string(str(out))
