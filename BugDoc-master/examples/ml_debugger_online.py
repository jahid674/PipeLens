# ==============================
# bugdoc_orchestrator_order_aware.py
# ==============================
"""
Online BugDoc Orchestrator (order-aware, no historical CSV)
===========================================================
- FULL search space for strategies (excludes 'model')
- Adds position parameters: pos_<step> in [1..K] (K = #steps excl. model)
- Samples START configs:
    * selected components pinned to strategy '1' (ONLY for selection, not limiting the space)
    * all other strategies sampled uniformly from their full ranges
    * IMPORTANT FIX: positions are RANDOMIZED per start config (so eval_order changes)
- Runs StackedShortcut per start config
- Post-processes roots to label 'strategy' vs 'position'
- Saves per-pipeline iterations and labeled roots

NOTE on threshold:
  StackedShortcut does not accept a threshold parameter — pass/fail decisions
  are made entirely in the worker (bugdoc_worker_order_aware.py).
  Set BUGDOC_THRESHOLD in the environment before starting the worker.
  The worker now reads it dynamically per message, so changes take effect
  immediately without restarting either process.
"""

import os
import json
import shutil
import random
import itertools
import time
from collections import defaultdict

# --- Path bootstrap to import pipeline_execution.py from repo root ---
import sys
from pathlib import Path
_here = Path(__file__).resolve()
root = None
p = _here
for _ in range(6):
    if (p / "pipeline_execution.py").exists():
        root = p
        break
    p = p.parent
if root is None:
    root = _here.parents[2]
sys.path.insert(0, str(root))
# --- end bootstrap ---

from bugdoc.algos.stacked_shortcut import StackedShortcut
from pipeline_execution import PipelineExecutor

FILENAME       = os.getenv("BUGDOC_ENTRY_FILE", "ml_pipeline.json")
TMP_FILENAME   = os.getenv("BUGDOC_ENTRY_FILE_TMP", "ml_pipeline_tmp.json")
RESULTS_JSON   = os.getenv("BUGDOC_RESULTS_JSON", "root_cause_results_order_aware.json")

DATASET        = os.getenv("BUGDOC_DATASET", "adult")
METRIC_TYPE    = os.getenv("BUGDOC_METRIC_TYPE", "accuracy_score")
EXEC_MODE      = os.getenv("BUGDOC_EXEC_MODE", "fail")

# FIX: Removed the unused THRESHOLD and BETTER_IS_LOWER constants that were
# previously defined here but never passed anywhere. StackedShortcut has no
# threshold parameter — the threshold is enforced solely by the worker.
# The worker now reads BUGDOC_THRESHOLD dynamically, so this orchestrator
# does not need to touch it at all.

MAX_OUTER_ITER = int(os.getenv("BUGDOC_MAX_OUTER_ITER", "1000"))

# How many distinct start pipelines to run
SAMPLE_SIZE    = int(os.getenv("BUGDOC_SAMPLE_SIZE", "10"))
SAMPLE_SEED    = int(os.getenv("BUGDOC_SAMPLE_SEED", "42"))

# Baseline order including model (model excluded from BugDoc space)
RAW_PIPELINE_ORDER = [
    "sampling","invalid_value","missing_value","floating_point",
    "distribution_shape","multicollinearity","normalization","outlier",
    "deduplication","punctuation","stopword","lowercase","whitespace","model"
]
EXCLUDE_FROM_SPACE = {'model'}

# For selecting the START pipelines ONLY (not restricting search)
PINNED_TO_ONE_FOR_SELECTION = {'deduplication', "distribution_shape", "multicollinearity", "outlier"}

# -----------------------------
# Helpers
# -----------------------------
def uniquify_steps(pipeline_order):
    counts = defaultdict(int)
    unique, base_map = [], {}
    for step in pipeline_order:
        counts[step] += 1
        tag = f"{step}#{counts[step]}" if counts[step] > 1 else step
        unique.append(tag)
        base_map[tag] = step
    return unique, base_map

def build_parameter_space_from_executor(pipeline_order_unique, base_map):
    """
    Build the full STRATEGY space (exclude 'model') and
    add POSITION variables pos_<base> in [1..K] for every step except 'model'.
    """
    exec_probe = PipelineExecutor(
        pipeline_type='ml',
        dataset_name=DATASET,
        metric_type=METRIC_TYPE,
        pipeline_ord=[base_map[u] for u in pipeline_order_unique],
        execution_type=EXEC_MODE,
    )

    space = {}
    ordered_bases = [base_map[u] for u in pipeline_order_unique if base_map[u] not in EXCLUDE_FROM_SPACE]
    K = len(ordered_bases)  # number of steps excluding model

    # strategy domains
    for u_name in pipeline_order_unique:
        base = base_map[u_name]
        if base in EXCLUDE_FROM_SPACE:
            continue
        n = exec_probe.strategy_counts[base]
        space[u_name] = [str(i) for i in range(1, n + 1)]

    # position domains (strings)
    for base in ordered_bases:
        pos_key = f"pos_{base}"
        space[pos_key] = [str(i) for i in range(1, K + 1)]

    return space, ordered_bases

def random_positions(ordered_bases, rng):
    """
    IMPORTANT FIX:
    Randomize positions so the worker computes a different eval_order.
    We use a permutation of 1..K (one-to-one), so positions are meaningful.
    """
    K = len(ordered_bases)
    perm = list(range(1, K + 1))
    rng.shuffle(perm)
    return {f"pos_{b}": str(perm[i]) for i, b in enumerate(ordered_bases)}

def sample_start_configurations(parameter_space, base_map, ordered_bases, sample_size, seed=42):
    """
    Start configs:
      - pinned strategies -> '1' for chosen components (ONLY for selection)
      - all other strategies -> random across their full ranges
      - positions -> RANDOM PER START (FIX)
    We also ensure uniqueness of the entire config dict.
    """
    rng = random.Random(seed)

    pos_keys = [k for k in parameter_space if k.startswith("pos_")]
    strat_keys = [k for k in parameter_space if not k.startswith("pos_")]

    pinned_strat_keys = [u for u in strat_keys if base_map.get(u, u) in PINNED_TO_ONE_FOR_SELECTION]
    free_strat_keys   = [u for u in strat_keys if u not in pinned_strat_keys]

    # We will generate candidates and keep unique signatures
    seen = set()
    start_configs = []

    # Hard cap attempts to avoid infinite loops if the space is tiny
    max_attempts = max(10_000, 200 * sample_size)
    attempts = 0

    while len(start_configs) < sample_size and attempts < max_attempts:
        attempts += 1
        cfg = {}

        # pinned strategies
        for k in pinned_strat_keys:
            cfg[k] = "1"

        # free strategies: random draw independently per key
        for k in free_strat_keys:
            cfg[k] = rng.choice(parameter_space[k])

        # RANDOM positions (permutation)
        pos_assign = random_positions(ordered_bases, rng)
        for pk in pos_keys:
            cfg[pk] = pos_assign[pk]

        # ensure all strat keys exist
        for k in strat_keys:
            if k not in cfg:
                cfg[k] = parameter_space[k][0]

        sig = tuple(sorted(cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        start_configs.append(cfg)

    if len(start_configs) < sample_size:
        print(f"[WARN] Only generated {len(start_configs)} unique start configs (requested {sample_size}).")

    return start_configs

def label_root_items(root_items):
    """
    Label each root item as 'position' if it begins with 'pos_', else 'strategy'.
    """
    labeled = []
    if not root_items:
        return labeled

    for item in root_items:
        name, value = None, None
        if isinstance(item, str):
            if '=' in item:
                name, value = item.split('=', 1)
            else:
                name, value = item, None
        elif isinstance(item, dict):
            for k, v in item.items():
                name, value = k, v
                break
        elif isinstance(item, (tuple, list)) and len(item) >= 1:
            name = item[0]
            value = item[1] if len(item) > 1 else None
        else:
            name = repr(item)
            value = None

        rtype = 'position' if str(name).startswith('pos_') else 'strategy'
        labeled.append({'type': rtype, 'name': str(name), 'value': None if value is None else str(value)})

    return labeled

def _format_hms(seconds: float) -> str:
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f} (hh:mm:ss)"

def main():
    t0 = time.perf_counter()

    # 1) unique names + base map
    parameters_unique, base_map = uniquify_steps(RAW_PIPELINE_ORDER)

    # 2) build full space (strategies + positions), get ordered base steps (excl. model)
    parameter_space, ordered_bases = build_parameter_space_from_executor(parameters_unique, base_map)

    # 3) pick SAMPLE_SIZE start configs (NOW WITH RANDOM POSITIONS)
    start_configs = sample_start_configurations(
        parameter_space=parameter_space,
        base_map=base_map,
        ordered_bases=ordered_bases,
        sample_size=SAMPLE_SIZE,
        seed=SAMPLE_SEED
    )

    results = []
    for idx, cfg in enumerate(start_configs, start=1):
        # run file for this start (no 'model' appears here)
        run = {**cfg, "result": "False"}
        with open(FILENAME, "w") as f:
            json.dump(run, f)
            f.write("\n")

        # StackedShortcut progressive search
        # NOTE: StackedShortcut has no threshold parameter — threshold is
        # handled entirely by the worker reading BUGDOC_THRESHOLD from env.
        root_found = []
        used_iter_final = None
        for i in range(1, MAX_OUTER_ITER + 1):
            debugger = StackedShortcut(max_iter=i)
            shutil.copy(FILENAME, TMP_FILENAME)
            root, used_iter, _ = debugger.run(
                TMP_FILENAME,
                parameter_space,
                outputs=["results"],
            )
            print(f"[OrderAware #{idx}] max_iter={i} | root={root} | iters={used_iter}")
            if len(root) > 0:
                root_found = root
                used_iter_final = used_iter
                print(f"[OrderAware #{idx}] Root cause found.")
                break

        labeled_root = label_root_items(root_found)

        results.append({
            "index": idx,
            "start_config": cfg,           # strategies + positions, no model
            "root_cause_raw": root_found,
            "root_cause_labeled": labeled_root,
            "iters": used_iter_final
        })

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    iter_list = [r["iters"] for r in results if r["iters"] is not None]
    print("\n[Summary] iteration counts:", iter_list)

    # Percentiles
    try:
        import numpy as np
        p25, p50, p75, p100 = np.percentile(iter_list, [25, 50, 75, 100])
    except Exception:
        from statistics import quantiles
        qs = quantiles(iter_list, n=4, method="inclusive")
        p25, p50, p75 = qs[0], qs[1], qs[2]
        p100 = max(iter_list) if iter_list else None

    print(f"[Summary] iteration percentiles: p25={p25}, p50={p50}, p75={p75}, p100={p100}")

    elapsed = time.perf_counter() - t0
    print(f"[Summary] Total execution time: {_format_hms(elapsed)}")

if __name__ == "__main__":
    main()