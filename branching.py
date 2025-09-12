import pandas as pd
import numpy as np
from modules.outlier_detection.outlier_detector import OutlierDetector

class BranchingHelper:
    """
    Encapsulates branching utilities:
      • get_data_branch: build a boolean mask from dict / callable / query string
      • apply_step_branch: run a handler ONLY on masked rows, then merge results back
    """

    @staticmethod
    def get_data_branch(X: pd.DataFrame, y: pd.Series, conditions=None) -> pd.Series:
        """
        Build a boolean mask indicating which rows belong to the branch.

        conditions can be:
          - dict: {"col": value_or_list, ...}  (== or .isin semantics)
          - callable: lambda df -> boolean Series mask
          - str: a pandas query expression (e.g., "SEX == 'Male' and AGE > 30")
          - None: meaning "no branching" (all rows True)
        """
        if conditions is None:
            return pd.Series(True, index=X.index)

        if callable(conditions):
            mask = conditions(X)
            if not isinstance(mask, pd.Series):
                raise ValueError("conditions callable must return a pandas Series[bool].")
            return mask.astype(bool)

        if isinstance(conditions, dict):
            mask = pd.Series(True, index=X.index)
            for col, val in conditions.items():
                if isinstance(val, (list, tuple, set)):
                    mask &= X[col].isin(list(val))
                else:
                    mask &= (X[col] == val)
            return mask.astype(bool)

        if isinstance(conditions, str):
            try:
                mask_idx = X.query(conditions).index
                return X.index.isin(mask_idx)
            except Exception as e:
                raise ValueError(f"Invalid query string for branch conditions: {e}")

        raise ValueError("Unsupported type for 'conditions'. Use dict, callable, str, or None.")

    def apply_step_branch(handler, X, y, sens, branch_mask):
        """
        Apply a handler ONLY to rows where branch_mask is True.
        - Runs handler.apply on (X_branch, y_branch, sens_branch)
        - Merges outputs back into (X, y, sens) at the same row indices
        - Returns (X_merged, y_merged, sens_merged, utility, fraction_outlier, frac_header, frac_value)

        If the handler returns a scalar (utility), the transform is treated as
        no-op for merging; final utility is expected from the terminal step.
        """
        Xb = X.loc[branch_mask].copy()
        yb = y.loc[branch_mask].copy()
        sensb = (sens.loc[branch_mask].copy() if sens is not None else None)
        result = handler.apply(Xb, yb, sensb)
        frac_header, frac_value = None, None
        method_name = f'get_outlier_bef_{handler.__class__.__name__.replace("Handler","").lower()}_strat'
        if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
            frac_value = getattr(handler, method_name)()
            step_name = handler.__class__.__name__.replace("Handler", "")
            frac_header = f'outlier_bef_{step_name.lower()}_strat'

        method_name2 = f'get_{handler.__class__.__name__.replace("Handler","").lower()}'
        if hasattr(handler, method_name2) and callable(getattr(handler, method_name2)):
            fraction_outlier = getattr(handler, method_name2)()
        else:
            detector = OutlierDetector(Xb)
            _, _, _ = detector.transform(yb, sensitive_attr_train=None)
            fraction_outlier = detector.get_frac()
        if isinstance(result, (float, int)):
            return X, y, sens, None, fraction_outlier, frac_header, frac_value

        Xb2, yb2, sensb2 = result
        overlapping_cols = [c for c in X.columns if c in Xb2.columns]
        X_merged = X.copy()
        X_merged.loc[branch_mask, overlapping_cols] = Xb2[overlapping_cols].values

        y_merged = y.copy()
        y_merged.loc[branch_mask] = yb2.values

        sens_merged = sens.copy() if sens is not None else None
        if (sens is not None) and (sensb2 is not None):
            sens_merged.loc[branch_mask] = sensb2.values

        return X_merged, y_merged, sens_merged, None, fraction_outlier, frac_header, frac_value
