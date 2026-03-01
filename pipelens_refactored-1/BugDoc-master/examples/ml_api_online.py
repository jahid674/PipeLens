"""
Thin helpers for online BugDoc mode.
You can import these in tests or future extensions.
"""

import os
from pipeline_execution import PipelineExecutor

DATASET     = os.getenv("BUGDOC_DATASET", "adult")
METRIC_TYPE = os.getenv("BUGDOC_METRIC_TYPE", "sp")
EXEC_MODE   = os.getenv("BUGDOC_EXEC_MODE", "fail")

_executor_cache = {}

def get_executor(pipeline_order):
    key = (tuple(pipeline_order), DATASET, METRIC_TYPE, EXEC_MODE)
    if key not in _executor_cache:
        _executor_cache[key] = PipelineExecutor(
            pipeline_type='ml',
            dataset_name=DATASET,
            metric_type=METRIC_TYPE,
            pipeline_ord=pipeline_order,
            execution_type=EXEC_MODE,
        )
    return _executor_cache[key]

def execute_ordered(pipeline_order, values):
    """
    Executes the exact pipeline order with given 1-based strategy indices,
    returning the numeric utility.
    """
    ex = get_executor(pipeline_order)
    vals = [int(v) for v in values]
    return ex.current_par_lookup(pipeline_order, vals)

def parameter_space_from_executor(pipeline_order):
    """
    Returns a dict {step: ['1',...'n']} for each step, using executor.strategy_counts.
    """
    ex = get_executor(pipeline_order)
    space = {}
    for s in pipeline_order:
        n = ex.strategy_counts[s]
        space[s] = [str(i) for i in range(1, n + 1)]
    return space
