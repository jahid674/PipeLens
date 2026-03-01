import logging
import itertools
import random
from typing import List, Dict, Any, Tuple

import pandas as pd

# === You already have this in your repo ===
from pipeline_execution import PipelineExecutor
def generate_pipelines_from_executor(
    executor,
    pipeline_order=None,
    exhaustive=True,
    max_pipelines=None,
    random_seed=42,
):
    """
    Build pipeline vectors from the executor's strategy space.

    Returns a list of 1-based vectors aligned to pipeline_order.
    """
    if pipeline_order is None:
        pipeline_order = executor.pipeline_order

    # 1-based option ranges for each component (to match your CSV & handlers)
    ranges = []
    for comp in pipeline_order:
        n_opts = executor.strategy_counts.get(comp)
        if n_opts is None or n_opts <= 0:
            raise ValueError(f"Unknown or empty strategy space for step '{comp}'")
        ranges.append(list(range(1, n_opts + 1)))

    # Cartesian product
    all_combos = list(itertools.product(*ranges))
    if exhaustive:
        return [list(t) for t in all_combos]

    # Random sample without replacement
    if max_pipelines is None or max_pipelines >= len(all_combos):
        return [list(t) for t in all_combos]

    random.seed(random_seed)
    sample_idxs = random.sample(range(len(all_combos)), max_pipelines)
    return [list(all_combos[i]) for i in sample_idxs]


# ------------------------------------------------------------
# Helpers that mirror your optimizer’s internals (no-cap pilot)
# ------------------------------------------------------------
def _action_to_weights(name: str) -> Tuple[float, float]:
    if name.lower() == "similarity":
        return 1.0, 0.0
    # prediction
    return 0.0, 1.0


def _build_ranked_candidates(
    executor,
    cur_params_opt: Dict[str, int],
    filename_train: str,
    new_components: List[str],
    wS: float,
    wU: float,
    fuse_method: str = "arith",
):
    """
    Use your existing PipelineExecutor ranker. It returns tuples:
      (component, strategy, similarity, y_pred, pos, fused_score)
    """
    return executor.evaluate_interventions_pred_and_similarity(
        [int(v) for v in cur_params_opt.values()],
        filename_train,
        new_components=list(new_components),
        wS=wS,
        wU=wU,
    )


def _apply_candidate(
    pipeline_order: List[str],
    component: str,
    strategy: int,
    pos: int,
    cur_params_dict: Dict[str, int],
) -> Tuple[List[str], List[int], Tuple]:
    """
    Build (order, vec) for a candidate; also return a unique key for deduping.
    """
    original_order = list(pipeline_order)
    if component in original_order:
        idx = original_order.index(component)
        new_params = cur_params_dict.copy()
        new_params[original_order[idx]] = int(strategy)
        intervened_order = original_order
    else:
        if pos is None:
            return None, None, None
        pos = int(pos)
        intervened_order = original_order[:pos] + [component] + original_order[pos:]
        items = list(cur_params_dict.items())
        items = items[:pos] + [(component, int(strategy))] + items[pos:]
        new_params = dict(items)

    vec = [int(new_params[s]) for s in intervened_order]
    key = tuple((name, new_params[name]) for name in intervened_order)
    return intervened_order, vec, key


def _eval_once(executor, order: List[str], vec: List[int]) -> float:
    """
    Run one pipeline evaluation and return the utility (lower is better).
    """
    try:
        util = executor.current_par_lookup(order, vec)
    except Exception as e:
        logging.warning(f"[EVAL] evaluation error: {e}")
        return None
    return util


def pilot_k_for_action(
    executor,
    filename_train: str,
    new_components: List[str],
    pipeline_order: List[str],
    init_params_vec: List[int],
    f_goal: float,
    action_name: str,
) -> Tuple[int, bool, float]:
    """
    No-cap pilot for one action on one dataset/pipeline.
    Walk the action’s ranked candidates and evaluate until pass.

    Returns:
      - k (#evaluations until pass; if never passes, attempts+1),
      - passed (bool),
      - util_pass (utility if passed else None)
    """
    # Map vector to dict keyed by component name
    cur_params_opt = {name: int(sel) for name, sel in zip(pipeline_order, init_params_vec)}

    wS, wU = _action_to_weights(action_name)
    ranked = _build_ranked_candidates(
        executor=executor,
        cur_params_opt=cur_params_opt,
        filename_train=filename_train,
        new_components=new_components,
        wS=wS,
        wU=wU,
    )

    attempts = 0
    tried_keys = set()
    for component, strategy, similarity, y_pred, pos, _ in ranked:
        order, vec, key = _apply_candidate(pipeline_order, component, int(strategy), pos, cur_params_opt)
        if key is None or key in tried_keys:
            continue
        tried_keys.add(key)

        attempts += 1
        util = _eval_once(executor, order, vec)
        if util is None:
            continue

        if util <= f_goal:
            return attempts, True, util

    # No pass within ranked list: penalty attempts+1
    return attempts + 1, False, None


# ------------------------------------------------------------
# Full sweep driver: datasets × pipelines × actions → DataFrame
# ------------------------------------------------------------
def evaluate_actions_over_datasets(
    datasets: List[Dict[str, Any]],
    pipelines: List[List[int]],
    pipeline_order: List[str],
    model_type: str = "lr",
    metric_type: str = "rmse",
    pipeline_type: str = "ml",
    new_components: List[str] = None,
    log_level: int = logging.INFO,
) -> pd.DataFrame:
    
    logging.basicConfig(level=log_level)
    if new_components is None:
        new_components = ["outlier", "whitespace", "punctuation", "stopword", "deduplication"]

    rows = []

    for ds in datasets:
        dataset_name = ds["dataset_name"]
        filename_train = ds["filename_train"]
        filename_test = ds.get("filename_test", "")
        f_goal = float(ds["f_goal"])

        # Fresh executor per dataset; use execution_type='fail' to treat test as unseen
        executor = PipelineExecutor(
            pipeline_type=pipeline_type,
            dataset_name=dataset_name,
            metric_type=metric_type,
            pipeline_ord=pipeline_order,
            execution_type="fail",
        )

        logging.info(f"[DATASET] {dataset_name} | f_goal={f_goal}")

        for vec in pipelines:
            if len(vec) != len(pipeline_order):
                raise ValueError(
                    f"Pipeline vector length {len(vec)} != pipeline_order length {len(pipeline_order)}"
                )

            # Try both actions independently
            for action_name in ["Similarity", "Prediction"]:
                k, passed, util_pass = pilot_k_for_action(
                    executor=executor,
                    filename_train=filename_train,
                    new_components=new_components,
                    pipeline_order=pipeline_order,
                    init_params_vec=vec,
                    f_goal=f_goal,
                    action_name=action_name,
                )

                rows.append(
                    {
                        "pipeline_vec": str(vec),
                        "dataset_name": dataset_name,
                        "action": action_name,
                        "attempts_to_pass": int(k),
                        "passed": bool(passed),
                        "util_if_pass": float(util_pass) if util_pass is not None else None,
                    }
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "pipeline_vec",
            "dataset_name",
            "action",
            "attempts_to_pass",
            "passed",
            "util_if_pass",
        ],
    )
    return df


# ------------------------------------------------------------
# Example main: generate pipelines (Cartesian), then sweep
# ------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Define your global pipeline order (must match your handlers)
    pipeline_order = [
        "missing_value",
        "normalization",
        "model"
    ]

    space_probe_exec = PipelineExecutor(
        pipeline_type="ml",
        dataset_name="adult",
        metric_type="sp",
        pipeline_ord=pipeline_order,
        execution_type="pass",
    )

    pipelines = generate_pipelines_from_executor(
        space_probe_exec, exhaustive=False, max_pipelines=200, random_seed=42
    )

    datasets = [
            {"dataset_name": "adult",
            "filename_train": "historical_data/partial_pipeline/sim_historical_data_train_profile_lr_sp_adult.csv",
            "filename_test": "historical_data/noise/sim_historical_data_test_profile_lr_sp_adult.csv",
            "f_goal": 0.06}
    ]

    new_components = ["outlier", "whitespace"]

    df = evaluate_actions_over_datasets(
        datasets=datasets,
        pipelines=pipelines,
        pipeline_order=pipeline_order,
        model_type="lr",
        metric_type="sp",
        pipeline_type="ml",
        new_components=new_components,
        log_level=logging.INFO,
    )

    print("\n=== Summary (dataset × pipeline × action) ===")
    print(df.head())
    # Save if you want:
    df.to_csv("action_results_adult.csv", index=False)
