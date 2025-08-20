# Save all K=3 demo outputs as tables (CSVs + Excel) and a chart (PNG).
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import os
from caas_jupyter_tools import display_dataframe_to_user

outdir = "logs/k3_exhaustive_demo_outputs"
os.makedirs(outdir, exist_ok=True)

# ---------- 1) Baseline pipeline ----------
baseline_order = ['missing_value', 'normalization', 'model']
baseline_params = {'missing_value': 2, 'normalization': 1, 'model': 1}
baseline_vec = [baseline_params[s] for s in baseline_order]
baseline_utility = 0.200  # illustrative

df_baseline = pd.DataFrame({
    'step': baseline_order,
    'strategy (1-indexed)': baseline_vec
})

# Save baseline
baseline_csv = os.path.join(outdir, "baseline.csv")
df_baseline.to_csv(baseline_csv, index=False)
display_dataframe_to_user("K=3 — Baseline Pipeline (saved: baseline.csv)", df_baseline)

# ---------- 2) Ranked single-intervention pool ----------
raw_actions = [
    ('change', 'missing_value', 4, None, 0.92),
    ('change', 'missing_value', 5, None, 0.88),
    ('change', 'normalization', 3, None, 0.76),
    ('change', 'normalization', 4, None, 0.70),
    ('insert', 'outlier',      7, 1,    0.81),
    ('insert', 'outlier',      5, 0,    0.72),
    ('insert', 'stopword',     1, 2,    0.65),
    ('insert', 'punctuation',  1, 0,    0.60),
    ('insert', 'whitespace',   1, 2,    0.58),
]
df_actions = pd.DataFrame(raw_actions, columns=['kind','component','strategy','position','similarity'])
df_actions.loc[df_actions['kind'] == 'insert', 'position'] = df_actions.loc[df_actions['kind'] == 'insert', 'position'].astype(int)
df_actions.loc[df_actions['kind'] == 'change', 'position'] = -1
df_actions = df_actions.sort_values('similarity', ascending=False).reset_index(drop=True)

# Save actions
actions_csv = os.path.join(outdir, "ranked_actions.csv")
df_actions.to_csv(actions_csv, index=False)
display_dataframe_to_user("K=3 — Ranked Single-Intervention Action Pool (saved: ranked_actions.csv)", df_actions)

# Typed actions for combos
actions_typed = []
for row in df_actions.itertuples(index=False):
    kind = row.kind
    comp = row.component
    strat = int(row.strategy)
    sim = float(row.similarity)
    pos = int(row.position) if kind == 'insert' else None
    actions_typed.append((kind, comp, strat, pos, sim))

# ---------- 3) Build k-way combos, conflict check, apply concurrently ----------
k = 3
baseline_set = set(baseline_order)

def conflict_free(combo, baseline_set):
    seen_change = set()
    seen_insert = set()
    for kind, comp, strat, pos, sim in combo:
        if kind == 'change':
            if comp in seen_change or comp not in baseline_set:
                return False
            seen_change.add(comp)
        else:
            if comp in seen_insert or comp in baseline_set or pos is None:
                return False
            seen_insert.add(comp)
    return True

def apply_combo_concurrently(base_order, base_params_dict, combo):
    changes = [(comp, strat) for (kind, comp, strat, pos, sim) in combo if kind == 'change']
    inserts = [(comp, strat, int(pos), sim) for (kind, comp, strat, pos, sim) in combo if kind == 'insert']
    inserts_sorted = sorted(inserts, key=lambda t: (t[2], -t[3]))

    order = base_order[:]
    params = base_params_dict.copy()
    shift = 0
    for comp, strat, pos, sim in inserts_sorted:
        ins_at = max(0, min(pos + shift, len(order)))
        order = order[:ins_at] + [comp] + order[ins_at:]
        left = list(params.items())[:ins_at]
        right = list(params.items())[ins_at:]
        params = dict(left + [(comp, int(strat))] + right)
        shift += 1

    for comp, strat in changes:
        if comp in order:
            params[comp] = int(strat)

    vec = [int(params[s]) for s in order]
    return order, vec, params

# Enumerate & conflict table
from itertools import combinations
all_combos = list(combinations(actions_typed, k))

def fmt_action(a):
    kind, comp, strat, pos, sim = a
    return f"{'change' if kind=='change' else 'insert'}:{comp}{'' if pos is None else '@'+str(pos)}->{strat}"

combo_rows = []
for combo in all_combos:
    combo_desc = " + ".join([fmt_action(a) for a in combo])
    ok = conflict_free(combo, baseline_set)
    combo_rows.append({'k': k, 'combo': combo_desc, 'conflict_free': ok})

df_combos = pd.DataFrame(combo_rows)

# Save combos
combos_csv = os.path.join(outdir, "k3_all_combos_conflicts.csv")
df_combos.to_csv(combos_csv, index=False)
display_dataframe_to_user("K=3 — All Combos and Conflict Check (saved: k3_all_combos_conflicts.csv)", df_combos)

# Apply valid combos and compute illustrative utility
valid_results = []
for combo in all_combos:
    if not conflict_free(combo, baseline_set):
        continue
    final_order, final_vec, final_params = apply_combo_concurrently(baseline_order, baseline_params, combo)
    sims = [a[4] for a in combo]
    n_inserts = sum(1 for a in combo if a[0]=='insert')
    predicted_utility = max(0.0, baseline_utility - 0.06 * float(np.mean(sims)) - 0.01 * n_inserts)
    valid_results.append({
        'combo': " + ".join([fmt_action(a) for a in combo]),
        'mean_similarity': round(float(np.mean(sims)), 4),
        '#inserts': int(n_inserts),
        'final_order': " → ".join(final_order),
        'final_params': final_vec,
        'predicted_utility (illustrative)': round(float(predicted_utility), 4)
    })

df_valid = pd.DataFrame(valid_results).sort_values('predicted_utility (illustrative)').reset_index(drop=True)

# Save valid combos
valid_csv = os.path.join(outdir, "k3_valid_combos_concurrent.csv")
df_valid.to_csv(valid_csv, index=False)
display_dataframe_to_user("K=3 — Valid Combos (Concurrent) with Final Pipelines (saved: k3_valid_combos_concurrent.csv)", df_valid)

# Save an Excel workbook with all tables
xlsx_path = os.path.join(outdir, "k3_outputs.xlsx")
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
    df_baseline.to_excel(writer, sheet_name="baseline", index=False)
    df_actions.to_excel(writer, sheet_name="ranked_actions", index=False)
    df_combos.to_excel(writer, sheet_name="all_combos_conflicts", index=False)
    df_valid.to_excel(writer, sheet_name="valid_combos", index=False)

# Chart: bar of illustrative utilities
plt.figure(figsize=(12, 5))
x = np.arange(len(df_valid))
y = df_valid['predicted_utility (illustrative)'].values
plt.bar(x, y)
plt.xticks(x, [f"C{i+1}" for i in range(len(df_valid))])
plt.ylabel("Illustrative Predicted Utility (lower is better)")
plt.title("K=3 Concurrent Combos (Illustrative Ranking)")
plt.tight_layout()

png_path = os.path.join(outdir, "k3_ranking.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()

# Return paths so the assistant can surface download links
(baseline_csv, actions_csv, combos_csv, valid_csv, xlsx_path, png_path)
