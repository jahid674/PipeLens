# Regenerate per_pipeline_actions.csv with missing_value options 1..9
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration (updated MV range to 1..9) ---
DATASETS = ["adult", "housing", "hmda"]
MV_OPTS = range(1, 9+1)      # missing_value 1..9
NORM_OPTS = range(1, 5+1)    # normalization 1..5
MODEL_OPTS = [1]             # single model option

# Thresholds and util distributions (per your last spec)
THRESHOLDS = {"adult": 0.05, "hmda": 0.08, "housing": 170.0}
UTIL_PARAMS = {
    "adult":   {"mean": 0.08,  "sd": 0.02},
    "hmda":    {"mean": 0.09,  "sd": 0.02},
    "housing": {"mean": 165.0, "sd": 8.0},
}
ATTEMPT_MAX = {"adult": 12, "hmda": 4, "housing": 2}

ACTIONS = ["Similarity", "Prediction"]

rng = np.random.RandomState(12345)
pipelines = list(itertools.product(MV_OPTS, NORM_OPTS, MODEL_OPTS))  # 9*5=45 per dataset

# Desired action composition per dataset for 45 pipelines
# adult: MOST Similarity -> 36 Similarity, 9 Prediction (80/20)
# hmda:  MOST Prediction -> 36 Prediction, 9 Similarity
# housing: HALF & HALF -> 22 Similarity, 23 Prediction
def planned_actions_for(ds: str):
    n = len(pipelines)  # 45
    if ds == "adult":
        return ["Similarity"] * 36 + ["Prediction"] * 9
    if ds == "hmda":
        return ["Prediction"] * 36 + ["Similarity"] * 9
    if ds == "housing":
        return ["Similarity"] * 22 + ["Prediction"] * 23
    return ["Prediction"] * n

def sample_passing_util(ds: str, thr: float) -> float:
    mu, sd = UTIL_PARAMS[ds]["mean"], UTIL_PARAMS[ds]["sd"]
    for _ in range(10):
        u = rng.normal(mu, sd)
        if u <= thr:
            return float(u)
    # Push below threshold in a plausible way
    if thr > 1.0:
        margin = abs(rng.normal(loc=3.0, scale=2.0))
        return float(thr - max(0.5, margin))
    else:
        frac = rng.uniform(0.50, 0.98)
        return float(thr * frac)

rows = []
for ds in DATASETS:
    thr = THRESHOLDS[ds]
    planned = planned_actions_for(ds)
    assert len(planned) == len(pipelines)
    for (mv, nm, mdl), action in zip(pipelines, planned):
        attempts = int(rng.randint(1, ATTEMPT_MAX[ds]))  # strict "<" bound
        util_if_pass = sample_passing_util(ds, thr)
        rows.append({
            "dataset_name": ds,
            "missing_value": int(mv),
            "normalization": int(nm),
            "model": int(mdl),
            "best_action": action,
            "attempts_to_pass": attempts,
            "passed": True,
            "util_if_pass": round(util_if_pass, 6),
            "threshold": float(thr),
        })

df_best = pd.DataFrame(rows, columns=[
    "dataset_name","missing_value","normalization","model",
    "best_action","attempts_to_pass","passed","util_if_pass","threshold"
])

# Sanity checks
assert all(df_best["passed"])
assert df_best.query("dataset_name=='adult' and attempts_to_pass >= 12").empty
assert df_best.query("dataset_name=='hmda' and attempts_to_pass >= 4").empty
assert df_best.query("dataset_name=='housing' and attempts_to_pass >= 2").empty
assert (df_best["util_if_pass"] <= df_best["threshold"]).all()

# Composition checks
adult_counts   = df_best.query("dataset_name=='adult'")["best_action"].value_counts().to_dict()
hmda_counts    = df_best.query("dataset_name=='hmda'")["best_action"].value_counts().to_dict()
housing_counts = df_best.query("dataset_name=='housing'")["best_action"].value_counts().to_dict()

out_path = "per_pipeline_actions.csv"
df_best.to_csv(out_path, index=False)

adult_counts, hmda_counts, housing_counts, str(out_path)
out_path
