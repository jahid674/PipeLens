#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import itertools
import pandas as pd

# ---------------- CONFIG ----------------
CSV_NAME    = "historical_data_test_profile_reg_rmse_housing.csv"
THRESHOLD   = 170.0            # fairness <= threshold => GOOD (True)
MAX_BUDGET  = 100              # like your iter
K_GOODS     = 4                # BugDoc's default k=4

PARAMS      = ["missing_value", "normalization", "outlier"]

def all_permutations():
    return list(itertools.product(
        [str(i) for i in range(1, 10)],  # mv
        [str(i) for i in range(1, 6)],   # normalization
        [str(i) for i in range(1, 8)]    # outlier
    ))

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for p in PARAMS:
        df[p] = df[p].astype(int).astype(str)
    return df

def eval_config(df: pd.DataFrame, cfg: tuple[str,str,str]) -> bool:
    """Return True if fairness <= THRESHOLD, else False (GOOD vs BAD)."""
    mv, nm, ol = cfg
    row = df[(df["missing_value"] == mv) &
             (df["normalization"] == nm) &
             (df["outlier"] == ol)]
    if row.empty:
        return False
    return float(row.iloc[0]["fairness"]) <= THRESHOLD

# -------------- Core logic --------------
def differ_on_all(a: tuple[str,str,str], b: tuple[str,str,str]) -> bool:
    return (a[0] != b[0]) and (a[1] != b[1]) and (a[2] != b[2])

def pick_cf_and_cgs(evaluated: dict[tuple[str,str,str], bool]):
    bads  = [cfg for cfg, res in evaluated.items() if res is False]
    goods = [cfg for cfg, res in evaluated.items() if res is True]
    best_cf = None
    best_cgs = []
    for cf in bads:
        cgs = [g for g in goods if differ_on_all(g, cf)]
        if len(cgs) > len(best_cgs):
            best_cgs = cgs
            best_cf  = cf
    return best_cf, best_cgs

def stacked_shortcut_step(df, evaluated, budget_remaining):
    """
    One Stacked Shortcut phase:
      - Ensure both classes exist.
      - Pick (cf, cgs). If cgs < K_GOODS, actively search goods that differ on all params from cf.
      - For each cg and p in 0..2, evaluate cf with p replaced by cg[p] (if unseen & budget allows).
      - If replacement is BAD, advance cf to that cf_aux; otherwise just record the GOOD.
      - Compute believed decisive params = positions where cf == cf_orig after the substitution loop.
      - Invalidate believed if any GOOD matches all believed pairs.

    Returns:
      root (list of one clause [[(idx,val), ...]] or []), consumed_count (int)
    """
    consumed = 0
    if not any(v is True for v in evaluated.values()) or not any(v is False for v in evaluated.values()):
        return [], consumed

    cf, cgs = pick_cf_and_cgs(evaluated)
    if cf is None:
        # no BAD available
        return [], consumed

    if len(cgs) < K_GOODS and budget_remaining > 0:
        for cfg in all_permutations():
            if consumed >= budget_remaining:
                break
            if cfg in evaluated:
                continue
            if not differ_on_all(cfg, cf):
                continue
            res = eval_config(df, cfg)
            evaluated[cfg] = res
            consumed += 1
            if res is True:
                cgs.append(cfg)
            if len(cgs) >= K_GOODS:
                break

    if len(cgs) == 0:
        return [], consumed

    cf_orig = cf
    for cg in cgs:
        for p in range(3):
            if consumed >= budget_remaining:
                break
            cf_aux = list(cf)
            cf_aux[p] = cg[p]
            cf_aux = tuple(cf_aux)
            if cf_aux not in evaluated:
                res = eval_config(df, cf_aux)
                evaluated[cf_aux] = res
                consumed += 1
            else:
                res = evaluated[cf_aux]
            if res is False:   # BAD -> move counterfactual along the "bad path"
                cf = cf_aux
        if consumed >= budget_remaining:
            break
    believed = []
    for i in range(3):
        if cf[i] == cf_orig[i]:
            believed.append((i, cf_orig[i]))

    if believed:
        for cfg, res in evaluated.items():
            if res is True:
                if all(cfg[idx] == val for (idx, val) in believed):
                    believed = []
                    break

    if believed:
        return [believed], consumed
    return [], consumed

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, CSV_NAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_df(csv_path)

    seed_idx = min(45, len(df) - 1)
    cf0 = (df.iloc[seed_idx]["missing_value"], df.iloc[seed_idx]["normalization"], df.iloc[seed_idx]["outlier"])
    if eval_config(df, cf0) is True:
        # find any BAD
        bad_idx = df.index[(df["fairness"] > THRESHOLD)].tolist()
        if not bad_idx:
            print("All configurations are GOOD under the threshold; nothing to debug.")
            return
        j = bad_idx[0]
        cf0 = (df.iloc[j]["missing_value"], df.iloc[j]["normalization"], df.iloc[j]["outlier"])

 
    evaluated = {cf0: False}
    created_total = 1

    if not any(v is True for v in evaluated.values()):
        for cfg in all_permutations():
            if cfg in evaluated:
                continue
            res = eval_config(df, cfg)
            evaluated[cfg] = res
            created_total += 1
            if res is True:
                break

    found_root = None

    for i in range(1, MAX_BUDGET + 1):
        budget_remaining = max(0, i - (created_total - 0))
        if budget_remaining > 0:
            if not (any(v is True for v in evaluated.values()) and any(v is False for v in evaluated.values())):
                for cfg in all_permutations():
                    if budget_remaining <= 0:
                        break
                    if cfg in evaluated:
                        continue
                    res = eval_config(df, cfg)
                    evaluated[cfg] = res
                    created_total += 1
                    budget_remaining -= 1
        root, consumed = stacked_shortcut_step(df, evaluated, budget_remaining)
        created_total += consumed

        total_seen = len(evaluated)
        print(i, root, total_seen)

        if root:
            found_root = root
            print("Found root cause at iteration:", i)
            names = PARAMS
            pretty = ' OR '.join(
                [' AND '.join([f"{names[p]} = {v}" for (p, v) in clause]) for clause in root]
            )
            print("Root Cause: \n" + pretty)
            break

    if not found_root:
        print("No root cause found up to max_iter =", MAX_BUDGET)

if __name__ == "__main__":
    main()
