#!/usr/bin/env python3
# coding: utf-8

import logging
from typing import Dict, Any

import pandas as pd

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# ---------------------------------------------------------------------
# Utility lookup: check if a configuration meets the threshold
# - profiles_df: CSV loaded DataFrame with one integer-coded column per parameter
#                plus a metric column named "fairness" (or your chosen metric)
# - configuration: dict like {"imputation": "3", "normalize": "5", ...}
# - threshold: numeric threshold; we return True if fairness < threshold
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

    # Build a boolean mask across ALL parameters present in the configuration
    # (we ignore keys absent from the DataFrame)
    mask = pd.Series(True, index=profiles_df.index)
    used_cols = []
    for k, v in configuration.items():
        if k in profiles_df.columns:
            used_cols.append(k)
            try:
                # Stored CSV values are typically ints; config values are strings
                # Normalize to int for comparison when possible
                iv = int(v) if isinstance(v, str) and v.isdigit() else v
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
# API that the worker calls
# ---------------------------------------------------------------------
def execute_pipeline(
    configuration: Dict[str, Any],
    historical_data: pd.DataFrame,
    threshold: float,
    metric_col: str = "utility_rmse",
) -> bool:
    """
    Given a config dict like:
      {
        "missing_value": "9",
        "normalization": "5",
        "punctuation": "2",
        "outlier": "7",
        "stopword": "2",
        "whitespace": "2",
        "tokenizer": "3",
        "model": "1"
      }
    look up the row in `historical_data` and return whether it meets the threshold.
    """
    try:
        ok = f_score_look_up2(historical_data, configuration, threshold, metric_col=metric_col)
        return bool(ok)
    except Exception as e:
        logging.exception("[execute_pipeline] Error evaluating configuration %s: %s", configuration, e)
        return False
