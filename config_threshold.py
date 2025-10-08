# config_thresholds.py
"""
Dataset-specific thresholds for tau, contamination, etc.
Only edit here if thresholds change.
"""

def get_thresholds(dataset_name, execution_type='pass'):
    thresholds = {}

    if dataset_name == 'adult':
        thresholds['tau'] = 0.1
        thresholds['contamination'] = 0.2
        thresholds['contamination_lof'] = 'auto'

    elif dataset_name == 'hmda':
        if execution_type == 'pass':
            thresholds['tau'] = 0.05
            thresholds['contamination'] = 0.1
            thresholds['contamination_lof'] = 0.1
        else:
            thresholds['tau'] = 0.05
            thresholds['contamination'] = 0.2
            thresholds['contamination_lof'] = 0.2

    elif dataset_name == 'housing':
        if execution_type == 'pass':
            thresholds['tau'] = 0.2
            thresholds['contamination'] = 0.3
            thresholds['contamination_lof'] = 0.3
        else:
            thresholds['tau'] = 0.1
            thresholds['contamination'] = 0.2
            thresholds['contamination_lof'] = 0.2

    else:
        raise ValueError("Invalid dataset name. Supported: 'adult', 'hmda', 'housing'.")

    return thresholds
