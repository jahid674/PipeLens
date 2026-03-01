# ==============================
# bugdoc_orchestrator_order_aware.py
# ==============================
"""
Online BugDoc Orchestrator (order-aware, no historical CSV)
===========================================================
- Uses PROVIDED failing pipeline configs (instead of random sampling).
- Failing configs may include 'model' (ignored for BugDoc search space because fixed).
- Builds parameter space:
    * strategies for all steps EXCEPT model
    * position params pos_<step> in [1..K] (K = #steps excl. model)
- For each provided failing config:
    * remove model keys from the start config
    * if pos_* not provided -> add identity positions (preserve base order)
- Runs StackedShortcut per provided start config
- Labels roots as 'strategy' vs 'position'
- Saves results
"""

import os
import json
import shutil
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
METRIC_TYPE    = os.getenv("BUGDOC_METRIC_TYPE", "sp")
EXEC_MODE      = os.getenv("BUGDOC_EXEC_MODE", "fail")

THRESHOLD      = float(os.getenv("BUGDOC_THRESHOLD", ".02"))
BETTER_IS_LOWER= os.getenv("BUGDOC_BETTER_IS_LOWER", "1") == "1"

# ================= USER CONTROL =================
EVALUATE_FIRST_N_PIPELINES = 10   # set to 0 to evaluate all
# ================================================

MAX_OUTER_ITER = int(os.getenv("BUGDOC_MAX_OUTER_ITER", "1000"))

# ---- NEW: provided failing pipelines ----
FAILING_FILE = os.getenv("BUGDOC_FAILING_PIPELINES_FILE", "").strip()
if not FAILING_FILE:
    raise ValueError("Set BUGDOC_FAILING_PIPELINES_FILE to your failing CSV path.")

MAX_PIPELINES = EVALUATE_FIRST_N_PIPELINES


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

def is_model_key(k: str) -> bool:
    # supports "model" and "model#2" etc.
    return str(k).split("#", 1)[0] == "model"

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

    # strategy domains (exclude model)
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

def identity_positions(ordered_bases):
    """
    Deterministic positions (preserve baseline order).
    """
    return {f"pos_{b}": str(i + 1) for i, b in enumerate(ordered_bases)}

def load_failing_configs(path: str):
    """
    Reads failing pipelines from a CSV like:
    sampling,invalid_value,...,model
    3.0,1.0,...,1.0
    2.0,4.0,...,1.0

    Returns: list[dict] where values are strings of ints ("3","1",...)
    """
    if not path:
        raise ValueError(
            "BUGDOC_FAILING_PIPELINES_FILE is empty. "
            "Set it to the path of your failing pipeline CSV."
        )

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        raise ValueError("Failing pipelines CSV must have a header + at least one row.")

    header = [h.strip() for h in lines[0].split(",")]
    out = []

    for ridx, line in enumerate(lines[1:], start=2):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header):
            print(f"[WARN] Skipping row {ridx}: expected {len(header)} cols, got {len(parts)}.")
            continue

        cfg = {}
        for k, v in zip(header, parts):
            # cast "3.0" -> "3"
            try:
                fv = float(v)
                iv = int(round(fv))
                cfg[str(k)] = str(iv)
            except Exception:
                cfg[str(k)] = str(v)

        out.append(cfg)

    return out


def normalize_start_config(cfg, parameter_space, parameters_unique, base_map, ordered_bases):
    """
    Convert a user-provided failing cfg into a BugDoc start config:
      - remove model keys
      - map base names -> unique names where needed
      - fill missing strategy keys with default (first option)
      - ensure pos_* exist: use provided, else identity positions
    """
    # Build mapping base -> unique key (1st occurrence)
    base_to_unique = {}
    for u in parameters_unique:
        b = base_map[u]
        if b not in base_to_unique:
            base_to_unique[b] = u

    out = {}

    # 1) copy over provided strategy keys (base or unique), excluding model
    for k, v in cfg.items():
        k_str = str(k)

        # ignore model (constant)
        if is_model_key(k_str):
            continue

        # keep positions as-is if they are pos_*
        if k_str.startswith("pos_"):
            out[k_str] = str(v)
            continue

        # strategy key: can be base ("missing_value") or unique ("missing_value#2")
        base = k_str.split("#", 1)[0]
        if base in EXCLUDE_FROM_SPACE:
            continue

        # map base -> unique if user gave base name
        if base == k_str and base in base_to_unique:
            ukey = base_to_unique[base]
        else:
            ukey = k_str  # assume already unique

        if ukey in parameter_space:
            out[ukey] = str(v)

    # 2) ensure all strategy keys exist
    for k in parameter_space:
        if k.startswith("pos_"):
            continue
        if k not in out:
            out[k] = parameter_space[k][0]

    # 3) ensure all pos_* exist
    needed_pos = [k for k in parameter_space if k.startswith("pos_")]
    if not any(k in out for k in needed_pos):
        # no positions provided -> identity
        pos_assign = identity_positions(ordered_bases)
        out.update(pos_assign)
    else:
        # fill any missing pos_* with identity defaults
        pos_defaults = identity_positions(ordered_bases)
        for pk in needed_pos:
            if pk not in out:
                out[pk] = pos_defaults[pk]

    return out

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

    # 3) load provided failing pipelines
    provided = load_failing_configs(FAILING_FILE)
    if MAX_PIPELINES > 0:
        provided = provided[:MAX_PIPELINES]

    # 4) normalize into BugDoc-ready start configs
    start_configs = []
    for cfg in provided:
        start_configs.append(
            normalize_start_config(
                cfg=cfg,
                parameter_space=parameter_space,
                parameters_unique=parameters_unique,
                base_map=base_map,
                ordered_bases=ordered_bases
            )
        )

    if not start_configs:
        raise RuntimeError("No valid failing pipeline configs found to run.")

    results = []
    for idx, cfg in enumerate(start_configs, start=1):
        # run file for this start (no explicit model key needed)
        run = {**cfg, "result": "False"}
        with open(FILENAME, "w") as f:
            json.dump(run, f)
            f.write("\n")

        # StackedShortcut progressive search
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
            "start_config": cfg,           # strategies + positions, model excluded
            "root_cause_raw": root_found,
            "root_cause_labeled": labeled_root,
            "iters": used_iter_final
        })

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    iter_list = [r["iters"] for r in results if r["iters"] is not None]
    print("\n[Summary] iteration counts:", iter_list)

    # Percentiles
    if iter_list:
        import numpy as np
        p25, p50, p75, p100 = np.percentile(iter_list, [25, 50, 75, 100])
        print(f"[Summary] iteration percentiles: p25={p25}, p50={p50}, p75={p75}, p100={p100}")

    elapsed = time.perf_counter() - t0
    print(f"[Summary] Total execution time: {_format_hms(elapsed)}")

if __name__ == "__main__":
    main()
