#!/usr/bin/env python3
# coding: utf-8
# Author: Laure Berti-Equille <laure.berti@ird.fr>
# Edited: Only scale numeric columns; preserve all others

import warnings
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import quantile_transform

pd.options.mode.chained_assignment = None


class Normalizer:
    """
    Normalize only numeric columns; preserve all other dtypes/columns unchanged.

    Parameters
    ----------
    strategy : {'ZS','MM','DS','Log10','MA','NONE_normalization','SS','RS'}
        ZS: z-score; MM: MinMax; DS: decile-based via quantile_transform;
        Log10: log10 on positive values; MA: max-absolute (x / max);
        SS: StandardScaler; RS: RobustScaler;
        NONE_normalization: no-op
    exclude : str or None
        Column name to exclude from normalization (e.g., the target).
    verbose : bool
    threshold : float | None (unused, kept for API compatibility)
    """

    def __init__(self, dataset, strategy='ZS', exclude=None,
                 verbose=False, threshold=None):
        self.dataset = dataset
        self.strategy = strategy
        self.exclude = exclude
        self.verbose = verbose
        self.threshold = threshold

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'exclude': self.exclude,
            'verbose': self.verbose,
            'threshold': self.threshold,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn(
                    "Invalid parameter(s) for normalizer. Parameter(s) IGNORED. "
                    "Check with `normalizer.get_params().keys()`"
                )
            else:
                setattr(self, k, v)

    # ---------- helpers (apply per numeric column, preserve NaNs) ---------- #

    def _numeric_columns(self, df: pd.DataFrame):
        cols = df.select_dtypes(include=['number']).columns.tolist()
        if self.exclude in cols:
            cols.remove(self.exclude)
        return cols

    def _fit_transform_column_sklearn(self, s: pd.Series, transformer):
        """Fit `transformer` on non-null values of s, transform those, write back."""
        non_null_mask = s.notna()
        if non_null_mask.sum() == 0:
            return s  # all NaN
        vals = s.loc[non_null_mask].to_numpy(dtype=float).reshape(-1, 1)
        transformed = transformer.fit_transform(vals).reshape(-1)
        out = s.copy()
        out.loc[non_null_mask] = transformed
        return out

    def _zs(self, s: pd.Series):
        non_null = s.dropna()
        mean = non_null.mean()
        std = non_null.std()
        if std == 0 or np.isnan(std):
            return s  # avoid div-by-zero; leave as-is
        out = s.copy()
        mask = s.notna()
        out.loc[mask] = (s.loc[mask] - mean) / std
        return out

    def _mm(self, s: pd.Series):
        non_null = s.dropna()
        if non_null.empty:
            return s
        vmin = non_null.min()
        vmax = non_null.max()
        out = s.copy()
        mask = s.notna()
        if vmax == vmin:
            out.loc[mask] = 0.0
        else:
            out.loc[mask] = (s.loc[mask] - vmin) / (vmax - vmin)
        return out

    def _ma(self, s: pd.Series):
        non_null = s.dropna()
        if non_null.empty:
            return s
        vmax = non_null.max()
        out = s.copy()
        mask = s.notna()
        if vmax == 0 or np.isnan(vmax):
            return s
        out.loc[mask] = s.loc[mask] / vmax
        return out

    def _log10(self, s: pd.Series):
        """Log10 on positive values only; leave non-positive and NaNs unchanged."""
        out = s.copy()
        mask = s.notna() & (s > 0)
        if mask.any():
            out.loc[mask] = np.log10(s.loc[mask])
        return out

    def _ds(self, s: pd.Series, n_quantiles=10):
        """
        'Decimal scaling' here implemented as rank/quantile-based scaling using
        sklearn's quantile_transform to 10 quantiles by default.
        """
        out = s.copy()
        mask = s.notna()
        if mask.sum() == 0:
            return out
        vals = s.loc[mask].to_numpy(dtype=float).reshape(-1, 1)
        n_q = max(2, min(n_quantiles, len(vals)))  # guard for small columns
        transformed = quantile_transform(vals, n_quantiles=n_q, random_state=0, copy=True)
        out.loc[mask] = transformed.reshape(-1)
        return out

    # ---------- strategy dispatch (column-wise) ---------- #

    def _apply_strategy_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with only numeric (minus exclude) columns scaled."""
        if self.verbose:
            print(f"{self.strategy} normalizing... ")

        out = df.copy()
        num_cols = self._numeric_columns(out)
        if not num_cols:
            return out

        if self.strategy == 'ZS':
            for c in num_cols:
                out[c] = self._zs(out[c])
        elif self.strategy == 'MM':
            for c in num_cols:
                out[c] = self._mm(out[c])
        elif self.strategy == 'DS':
            for c in num_cols:
                out[c] = self._ds(out[c], n_quantiles=10)
        elif self.strategy == 'Log10':
            for c in num_cols:
                out[c] = self._log10(out[c])
        elif self.strategy == 'MA':
            for c in num_cols:
                out[c] = self._ma(out[c])
        elif self.strategy == 'SS':
            for c in num_cols:
                out[c] = self._fit_transform_column_sklearn(out[c], StandardScaler())
        elif self.strategy == 'RS':
            for c in num_cols:
                out[c] = self._fit_transform_column_sklearn(out[c], RobustScaler())
        elif self.strategy == 'NONE_normalization':
            # no-op
            return out
        else:
            raise ValueError("The normalization function should be one of "
                             "ZS, MM, DS, Log10, MA, SS, RS, or NONE_normalization")

        # ensure excluded column (if any) is preserved exactly
        if self.exclude is not None and self.exclude in df.columns:
            out[self.exclude] = df[self.exclude]

        return out

    # ---------- public API ---------- #

    def transform(self):
        normd = self.dataset  # dict with keys 'train', 'test', 'target'...

        start_time = time.time()
        print(">>Normalization ")

        # normalize for both splits if present
        for key in ['train', 'test']:
            if key in self.dataset and not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                print(f"* For {key} dataset")

                dn = self._apply_strategy_to_df(d)

                # preserve excluded column, again (safety)
                if self.exclude in d.columns:
                    dn[self.exclude] = d[self.exclude]

                normd[key] = dn
                print('...', key, 'dataset')
            else:
                if key in self.dataset:
                    normd[key] = self.dataset[key]
                    print('No', key, 'dataset, no normalization')

        print("Normalization done -- CPU time: %s seconds" % (time.time() - start_time))
        print()
        return normd
