#!/usr/bin/env python3
# coding: utf-8

import os
import logging
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd

# Import your class from the separate file you created earlier
from pipeline_execution import PipelineExecutor



logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

_EXECUTOR_CACHE: Optional[PipelineExecutor] = None
_EXECUTOR_CFG: Optional[Tuple[str, str, str, Tuple[str, ...]]] = None


def _infer_pipeline_order_from_history(historical_data: pd.DataFrame, metric_col: str) -> List[str]:
    if historical_data is None or historical_data.empty:
        raise ValueError("[ml_api_example] historical_data is required to infer pipeline order.")

    cols = [c for c in historical_data.columns if c != metric_col]
    # If 'model' exists among the columns, force it to be the last component
    if "model" in cols:
        cols = [c for c in cols if c != "model"] + ["model"]
    else:
        # Ensure model exists; PipelineExecutor expects it at the end
        cols = cols + ["model"]

    return cols


def _metric_type_from_metric_col(metric_col: str) -> str:
    """
    Your PipelineExecutor names the utility column as 'utility_<metric_type>'.
    If worker sets METRIC_COL='utility_rmse', metric_type -> 'rmse'.
    If it's just 'fairness', we'll use 'fairness' directly.
    """
    if metric_col.startswith("utility_"):
        return metric_col[len("utility_"):]
    return metric_col


def _get_or_make_executor(pipeline_order: List[str], metric_type: str) -> PipelineExecutor:
    """
    Create (or reuse) a single PipelineExecutor instance based on env:
      BUGDOC_DATASET (default 'housing')
      BUGDOC_EXEC_TYPE ('pass' or 'fail', default 'pass')
    """
    global _EXECUTOR_CACHE, _EXECUTOR_CFG

    dataset = os.getenv("BUGDOC_DATASET", "housing")
    exec_type = os.getenv("BUGDOC_EXEC_TYPE", "pass")  # mirror your earlier usage

    cfg = (dataset, metric_type, exec_type, tuple(pipeline_order))
    if _EXECUTOR_CACHE is not None and _EXECUTOR_CFG == cfg:
        return _EXECUTOR_CACHE

    logging.info(
        "[ml_api_example] Initializing PipelineExecutor(dataset=%s, metric=%s, exec_type=%s, order=%s)",
        dataset, metric_type, exec_type, pipeline_order
    )
    _EXECUTOR_CACHE = PipelineExecutor(
        pipeline_type="ml",
        dataset_name=dataset,
        metric_type=metric_type,
        pipeline_ord=pipeline_order,
        execution_type=exec_type,
    )
    _EXECUTOR_CFG = cfg
    return _EXECUTOR_CACHE


# ---------------------------------------------------------------------
# (Kept for compatibility; not used by execute_pipeline anymore)
# ---------------------------------------------------------------------
def f_score_look_up2(
    profiles_df: pd.DataFrame,
    configuration: Dict[str, Any],
    threshold: float,
    metric_col: str = "fairness",
) -> bool:
    if profiles_df is None or profiles_df.empty:
        logging.warning("[f_score_look_up2] Empty profiles DF.")
        return False

    mask = pd.Series(True, index=profiles_df.index)
    used_cols = []
    for k, v in configuration.items():
        if k in profiles_df.columns:
            used_cols.append(k)
            try:
                iv = int(v) if (isinstance(v, str) and v.isdigit()) else v
                mask &= (profiles_df[k] == iv)
            except Exception:
                mask &= (profiles_df[k].astype(str) == str(v))

    if not used_cols:
        logging.error("[f_score_look_up2] None of the configuration keys matched DF columns.")
        return False

    filtered = profiles_df.loc[mask]
    if filtered.empty:
        logging.info("[f_score_look_up2] No matching row for configuration: %s", configuration)
        return False

    if metric_col not in filtered.columns:
        logging.error("[f_score_look_up2] Metric column '%s' not found in DF.", metric_col)
        return False

    metric_value = float(filtered.iloc[0][metric_col])
    return metric_value < float(threshold)


# ---------------------------------------------------------------------
# NEW: Online execution backed by PipelineExecutor.current_par_lookup
# ---------------------------------------------------------------------
def execute_pipeline(
    configuration: Dict[str, Any],
    historical_data: pd.DataFrame,
    threshold: float,
    metric_col: str = "fairness",
) -> bool:
    """
    ONLINE evaluation version:
      - Infers pipeline order from 'historical_data' columns (except 'metric_col'), with 'model' last.
      - Builds the 1-based 'cur_par' vector from 'configuration' following that order.
      - Calls PipelineExecutor.current_par_lookup(order, cur_par) to compute the metric in real time.
      - Applies the same threshold rule as before. By default lower-is-better; override with env:

          BUGDOC_BETTER_IS_LOWER = "1" (default)  -> pass if metric <= threshold
                                     "0"          -> pass if metric >= threshold
    """
    try:
        # 1) Pipeline order (model forced last)
        pipeline_order = _infer_pipeline_order_from_history(historical_data, metric_col)

        # 2) Metric type for PipelineExecutor
        metric_type = _metric_type_from_metric_col(metric_col)

        # 3) Executor (cached)
        ex = _get_or_make_executor(pipeline_order, metric_type)

        # 4) Build cur_par in the inferred order (1-based integers). Model is last by construction.
        cur_par: List[int] = []
        for comp in pipeline_order:
            if comp not in configuration:
                raise KeyError(f"[ml_api_example] Parameter '{comp}' missing from configuration {configuration}")
            val = configuration[comp]
            try:
                iv = int(val)
            except Exception:
                raise ValueError(f"[ml_api_example] Parameter '{comp}' must be an integer-like value; got {val!r}")
            cur_par.append(iv)

        # 5) Compute metric online
        metric_value = float(ex.current_par_lookup(pipeline_order, cur_par=cur_par))

        # 6) Decision rule (env-controllable)
        better_is_lower = os.getenv("BUGDOC_BETTER_IS_LOWER", "1") == "1"
        passed = (metric_value <= float(threshold)) if better_is_lower else (metric_value >= float(threshold))

        logging.info(
            "[ml_api_example] order=%s cur_par=%s -> metric=%.6f | threshold=%.6f | rule=%s | pass=%s",
            pipeline_order, cur_par, metric_value, threshold,
            "lower<=thr" if better_is_lower else "higher>=thr",
            passed
        )
        return bool(passed)

    except Exception as e:
        logging.exception("[execute_pipeline] Error in online evaluation: %s", e)
        return False
