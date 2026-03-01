# ==============================
# bugdoc_orchestrator_order_aware.py
# ==============================
"""
Online BugDoc Orchestrator (ORDER-AWARE + FAILING-SEEDS-CSV)
===========================================================
- FULL search space for strategies (excludes 'model')
- Adds position parameters: pos_<step> in [1..K] (K = #steps excl. model)
- Loads START configs from a CSV (your "failing instances" pool)
- Runs StackedShortcut per start config
- Post-processes roots to label 'strategy' vs 'position'
- Saves per-seed results to JSON

NOTE:
- With threshold-based failure, "failing seeds" just mean configs you want to start from.
  BugDoc will still generate other configs and use worker True/False responses.
"""

import os
import json
import shutil
import random
import time
import csv
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
METRIC_TYPE    = os.getenv("BUGDOC_METRIC_TYPE", "sp")
EXEC_MODE      = os.getenv("BUGDOC_EXEC_MODE", "fail")

MAX_OUTER_ITER = int(os.getenv("BUGDOC_MAX_OUTER_ITER", "1000"))

# How many distinct start pipelines to run (from CSV)
SAMPLE_SIZE    = int(os.getenv("BUGDOC_SAMPLE_SIZE", "15"))
SAMPLE_SEED    = int(os.getenv("BUGDOC_SAMPLE_SEED", "42"))

# CSV path containing start configs (your failing instances pool)
FAILING_CSV    = os.getenv("BUGDOC_FAILING_CSV", "BugDoc-master/examples/0.1_filtered_adult_sp.csv")

# Baseline order including model (model excluded from BugDoc space)
RAW_PIPELINE_ORDER = [
    "sampling","invalid_value","missing_value","floating_point",
    "distribution_shape","multicollinearity","normalization","outlier",
    "deduplication","punctuation","stopword","lowercase","whitespace","model"
]
EXCLUDE_FROM_SPACE = {'model'}

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
    Build full STRATEGY space (exclude 'model') and
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
    K = len(ordered_bases)

    # strategy domains
    for u_name in pipeline_order_unique:
        base = base_map[u_name]
        if base in EXCLUDE_FROM_SPACE:
            continue
        n = exec_probe.strategy_counts[base]
        space[u_name] = [str(i) for i in range(1, n + 1)]

    # position domains
    for base in ordered_bases:
        pos_key = f"pos_{base}"
        space[pos_key] = [str(i) for i in range(1, K + 1)]

    return space, ordered_bases

def _read_csv_dicts(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rr = {str(k).strip(): (None if v is None else str(v).strip()) for k, v in r.items()}
            rows.append(rr)
        return rows

def load_failing_instances_csv(csv_path, parameter_space, sample_size, seed=42):
    """
    Loads configs from CSV and converts each row into a config dict
    matching BugDoc parameter_space keys.

    Rules:
      - keep only keys that exist in parameter_space
      - fill missing keys with default first value from parameter_space
      - ignore any 'result' column in CSV if present
    """
    rng = random.Random(seed)
    rows = _read_csv_dicts(csv_path)

    space_keys = set(parameter_space.keys())
    configs = []
    for row in rows:
        cfg = {}
        for k in space_keys:
            if k in row and row[k] not in (None, "", "nan", "NaN"):
                cfg[k] = str(row[k])
        for k in parameter_space:
            if k not in cfg:
                cfg[k] = parameter_space[k][0]
        configs.append(cfg)

    uniq, seen = [], set()
    for cfg in configs:
        sig = tuple(sorted(cfg.items()))
        if sig not in seen:
            seen.add(sig)
            uniq.append(cfg)

    if len(uniq) == 0:
        raise ValueError(f"No usable configs found in {csv_path}. Check CSV headers vs parameter_space keys.")

    if len(uniq) > sample_size:
        rng.shuffle(uniq)
        uniq = uniq[:sample_size]

    if len(uniq) < sample_size:
        print(f"[WARN] Only loaded {len(uniq)} unique configs (requested {sample_size}).")

    return uniq

def label_root_items(root_items):
    """Label each root item as 'position' if it begins with 'pos_', else 'strategy'."""
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

    # 2) build full space (strategies + positions)
    parameter_space, _ordered_bases = build_parameter_space_from_executor(parameters_unique, base_map)

    # 3) load start configs from CSV
    start_configs = load_failing_instances_csv(
        csv_path=FAILING_CSV,
        parameter_space=parameter_space,
        sample_size=SAMPLE_SIZE,
        seed=SAMPLE_SEED
    )
    print(f"[INFO] Loaded {len(start_configs)} start configs from {FAILING_CSV}")

    results = []
    for idx, cfg in enumerate(start_configs, start=1):
        # IMPORTANT:
        # BugDoc's utils.py historically expects "result" as a STRING it can eval ("True"/"False").
        # Keeping it a string avoids eval() type issues unless you've patched BugDoc utils.
        run = {**cfg, "result": "False"}
        with open(FILENAME, "w", encoding="utf-8") as f:
            json.dump(run, f)
            f.write("\n")

        root_found = []
        used_iter_final = None

        for i in range(1, MAX_OUTER_ITER + 1):
            debugger = StackedShortcut(max_iter=i)
            shutil.copy(FILENAME, TMP_FILENAME)

            root, used_iter, _created = debugger.run(
                TMP_FILENAME,
                parameter_space,
                outputs=["results"],
            )
            print(f"[OrderAware #{idx}] max_iter={i} | root={root} | iters={used_iter}")

            if isinstance(root, list) and len(root) > 0:
                root_found = root
                used_iter_final = used_iter
                print(f"[OrderAware #{idx}] Root cause found.")
                break

        labeled_root = label_root_items(root_found)

        results.append({
            "index": idx,
            "start_config": cfg,
            "root_cause_raw": root_found,
            "root_cause_labeled": labeled_root,
            "iters": used_iter_final
        })

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    iter_list = [r["iters"] for r in results if r["iters"] is not None]
    print("\n[Summary] iteration counts:", iter_list)

    try:
        import numpy as np
        if iter_list:
            p25, p50, p75, p100 = np.percentile(iter_list, [25, 50, 75, 100])
        else:
            p25 = p50 = p75 = p100 = None
    except Exception:
        p25 = p50 = p75 = p100 = None

    print(f"[Summary] iteration percentiles: p25={p25}, p50={p50}, p75={p75}, p100={p100}")
    elapsed = time.perf_counter() - t0
    print(f"[Summary] Total execution time: {_format_hms(elapsed)}")

if __name__ == "__main__":
    main()
