# ==============================
# bugdoc_orchestrator_order_aware.py
# ==============================
"""
Online BugDoc Orchestrator (order-aware, no historical CSV)
===========================================================
- Uses PROVIDED failing pipeline configs (CSV).
- Builds parameter space:
    * strategies for all steps EXCEPT model
    * position params pos_<step> in [1..K] (K = #steps excl. model)
- For each failing config:
    * remove model keys
    * if pos_* not provided -> add identity positions
- Runs StackedShortcut progressively increasing max_iter.
- DOES NOT reset tmp file per round: we keep one tmp log per pipeline index,
  so timing + experiment history accumulates properly.
- Reports:
    * cumulative executed pipelines (unique)
    * cumulative "new executions" across rounds (sum of increments)
    * final wall time for debugging that pipeline
    * final breakdown: sum_eval_time_sec vs sum_other_time_sec from worker logs
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

THRESHOLD      = float(os.getenv("BUGDOC_THRESHOLD", "156"))
BETTER_IS_LOWER= os.getenv("BUGDOC_BETTER_IS_LOWER", "1") == "1"

# ================= USER CONTROL =================
EVALUATE_FIRST_N_PIPELINES = 10   # set to 0 to evaluate all
# ================================================

MAX_OUTER_ITER = int(os.getenv("BUGDOC_MAX_OUTER_ITER", "1000"))

# ---- failing pipelines file (CSV) ----
FAILING_FILE = os.getenv("BUGDOC_FAILING_PIPELINES_FILE", "").strip()
if not FAILING_FILE:
    raise ValueError("Set BUGDOC_FAILING_PIPELINES_FILE to your failing CSV path.")

MAX_PIPELINES = EVALUATE_FIRST_N_PIPELINES

RAW_PIPELINE_ORDER = [
    "sampling","invalid_value","missing_value","floating_point",
    "distribution_shape","multicollinearity","normalization","outlier",
    "deduplication","punctuation","stopword","lowercase","whitespace","model"
]
EXCLUDE_FROM_SPACE = {"model"}

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
    return str(k).split("#", 1)[0] == "model"

def build_parameter_space_from_executor(pipeline_order_unique, base_map):
    exec_probe = PipelineExecutor(
        pipeline_type="ml",
        dataset_name=DATASET,
        metric_type=METRIC_TYPE,
        pipeline_ord=[base_map[u] for u in pipeline_order_unique],
        execution_type=EXEC_MODE,
    )

    space = {}
    ordered_bases = [base_map[u] for u in pipeline_order_unique if base_map[u] not in EXCLUDE_FROM_SPACE]
    K = len(ordered_bases)

    for u_name in pipeline_order_unique:
        base = base_map[u_name]
        if base in EXCLUDE_FROM_SPACE:
            continue
        n = exec_probe.strategy_counts[base]
        space[u_name] = [str(i) for i in range(1, n + 1)]

    for base in ordered_bases:
        space[f"pos_{base}"] = [str(i) for i in range(1, K + 1)]

    return space, ordered_bases

def identity_positions(ordered_bases):
    return {f"pos_{b}": str(i + 1) for i, b in enumerate(ordered_bases)}

def load_failing_configs(path: str):
    if not path:
        raise ValueError("BUGDOC_FAILING_PIPELINES_FILE is empty.")

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
            try:
                fv = float(v)
                iv = int(round(fv))
                cfg[str(k)] = str(iv)
            except Exception:
                cfg[str(k)] = str(v)
        out.append(cfg)

    return out

def normalize_start_config(cfg, parameter_space, parameters_unique, base_map, ordered_bases):
    base_to_unique = {}
    for u in parameters_unique:
        b = base_map[u]
        if b not in base_to_unique:
            base_to_unique[b] = u

    out = {}

    for k, v in cfg.items():
        k_str = str(k)

        if is_model_key(k_str):
            continue

        if k_str.startswith("pos_"):
            out[k_str] = str(v)
            continue

        base = k_str.split("#", 1)[0]
        if base in EXCLUDE_FROM_SPACE:
            continue

        ukey = base_to_unique.get(base, k_str) if (base == k_str) else k_str
        if ukey in parameter_space:
            out[ukey] = str(v)

    # fill missing strategies
    for k in parameter_space:
        if k.startswith("pos_"):
            continue
        if k not in out:
            out[k] = parameter_space[k][0]

    # fill positions
    needed_pos = [k for k in parameter_space if k.startswith("pos_")]
    if not any(k in out for k in needed_pos):
        out.update(identity_positions(ordered_bases))
    else:
        defaults = identity_positions(ordered_bases)
        for pk in needed_pos:
            if pk not in out:
                out[pk] = defaults[pk]

    return out

def label_root_items(root_items):
    labeled = []
    if not root_items:
        return labeled

    for item in root_items:
        name, value = None, None
        if isinstance(item, str):
            if "=" in item:
                name, value = item.split("=", 1)
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

        rtype = "position" if str(name).startswith("pos_") else "strategy"
        labeled.append({"type": rtype, "name": str(name), "value": None if value is None else str(value)})

    return labeled

def _format_hms(seconds: float) -> str:
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f} (hh:mm:ss)"

def sum_worker_times(logfile: str):
    """
    Parse tmp json-lines file and sum eval_time_sec/other_time_sec if present.
    """
    eval_sum = 0.0
    other_sum = 0.0
    count = 0

    if not os.path.isfile(logfile):
        return eval_sum, other_sum, count

    with open(logfile, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            if "eval_time_sec" in d:
                try:
                    eval_sum += float(d["eval_time_sec"])
                except Exception:
                    pass
            if "other_time_sec" in d:
                try:
                    other_sum += float(d["other_time_sec"])
                except Exception:
                    pass
            if "result" in d:
                # count only actual evaluated experiments (exclude the initial seed row if it has no timing)
                if "eval_time_sec" in d or "other_time_sec" in d:
                    count += 1

    return eval_sum, other_sum, count

def main():
    global_t0 = time.perf_counter()

    parameters_unique, base_map = uniquify_steps(RAW_PIPELINE_ORDER)
    parameter_space, ordered_bases = build_parameter_space_from_executor(parameters_unique, base_map)

    provided = load_failing_configs(FAILING_FILE)
    if MAX_PIPELINES > 0:
        provided = provided[:MAX_PIPELINES]

    start_configs = [
        normalize_start_config(cfg, parameter_space, parameters_unique, base_map, ordered_bases)
        for cfg in provided
    ]

    if not start_configs:
        raise RuntimeError("No valid failing pipeline configs found to run.")

    results = []

    for idx, cfg in enumerate(start_configs, start=1):
        pipeline_t0 = time.perf_counter()

        # create a per-pipeline tmp file so it accumulates across progressive runs
        tmp_for_idx = TMP_FILENAME.replace(".json", f"_{idx}.json")

        # seed: write initial config once into tmp_for_idx (start state)
        run = {**cfg, "result": "False"}  # MUST be string for BugDoc eval(...)
        with open(tmp_for_idx, "w") as f:
            json.dump(run, f)
            f.write("\n")

        # also keep a copy in FILENAME (optional, for inspection)
        with open(FILENAME, "w") as f:
            json.dump(run, f)
            f.write("\n")

        root_found = []
        used_iter_final = None

        prev_used_iter = 0
        cumulative_new_exec = 0  # sum of newly executed pipelines across rounds

        for i in range(1, MAX_OUTER_ITER + 1):
            debugger = StackedShortcut(max_iter=i)

            root, used_iter, created_now = debugger.run(
                tmp_for_idx,
                parameter_space,
                outputs=["result"],
            )

            # used_iter is total experiments in the tmp file seen by BugDoc
            new_exec = max(0, used_iter - prev_used_iter)
            prev_used_iter = used_iter
            cumulative_new_exec += new_exec

            print(f"[OrderAware #{idx}] max_iter={i} | root={root} | new_exec={new_exec} | total_exec={used_iter}")

            if len(root) > 0:
                root_found = root
                used_iter_final = used_iter
                print(f"[OrderAware #{idx}] Root cause found.")
                break

        labeled_root = label_root_items(root_found)

        # final timing + breakdown (ONLY ONCE)
        wall = time.perf_counter() - pipeline_t0
        eval_sum, other_sum, counted = sum_worker_times(tmp_for_idx)

        print(f"[OrderAware #{idx}] FINAL total_exec={used_iter_final} | sum_new_exec={cumulative_new_exec}")
        print(f"[OrderAware #{idx}] FINAL wall_time={_format_hms(wall)}")
        print(f"[OrderAware #{idx}] FINAL sum_eval_time_sec={eval_sum:.3f}s | sum_other_time_sec={other_sum:.3f}s | timed_evals_count={counted}")

        results.append({
            "index": idx,
            "start_config": cfg,
            "root_cause_raw": root_found,
            "root_cause_labeled": labeled_root,

            # counts:
            "total_exec_unique": used_iter_final,           # unique experiments in file
            "sum_new_exec_across_rounds": cumulative_new_exec,  # sum of increments per max_iter step

            # time:
            "debug_wall_time_sec": wall,
            "sum_eval_time_sec": eval_sum,
            "sum_other_time_sec": other_sum,
            "timed_eval_count": counted,

            # log file path (helpful for audit)
            "tmp_log_file": tmp_for_idx
        })

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.perf_counter() - global_t0
    print(f"\n[Summary] Total execution time (all pipelines): {_format_hms(elapsed)}")

if __name__ == "__main__":
    main()
