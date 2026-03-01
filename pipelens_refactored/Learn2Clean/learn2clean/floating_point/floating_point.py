#!/usr/bin/env python3
# coding: utf-8
# Author: (adapted to Learn2Clean-compatible format)

import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class FloatingPointStabilizer():
    """
    Floating Point Stabilization (Learn2Clean compatible).

    Goal:
      Reduce floating-point jitter by:
        - snapping tiny magnitudes to exactly 0
        - rounding to a fixed number of decimals

    Strategies:
      - "NONE"   : no-op
      - "SNAP"   : set values with |x| < tol to 0.0
      - "ROUND"  : round numeric values to `decimals`
      - "BOTH"   : SNAP then ROUND

    Behavior:
      - Applies to numeric columns only (excluding `exclude` if provided)
      - Replaces inf with NaN for safety (does NOT impute NaNs)
      - Does not change row count; preserves all non-numeric columns
      - Fits nothing; purely deterministic transform applied to train/test
    """

    def __init__(self, dataset, strategy="BOTH", decimals=6, tol=1e-8,
                 verbose=False, exclude=None, threshold=None):

        self.dataset = dataset
        self.strategy = str(strategy).upper().strip()
        self.decimals = int(decimals)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.exclude = exclude  # can be str or list; handled internally
        self.threshold = threshold  # unused, kept for API compatibility

        if self.strategy not in ("NONE", "SNAP", "ROUND", "BOTH"):
            raise ValueError("Strategy invalid. Please choose between "
                             "'NONE', 'SNAP', 'ROUND', or 'BOTH'.")

    # ------------------- Learn2Clean API -------------------

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'decimals': self.decimals,
            'tol': self.tol,
            'verbose': self.verbose,
            'exclude': self.exclude,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for FloatingPointStabilizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`floating_point_stabilizer.get_params().keys()`")
            else:
                setattr(self, k, v)

        # normalize strategy if user changed it
        self.strategy = str(self.strategy).upper().strip()

    # ------------------- helpers -------------------

    def _exclude_list(self):
        if self.exclude is None:
            return []
        if isinstance(self.exclude, list):
            return [c for c in self.exclude if c is not None]
        return [self.exclude]

    def _apply_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # true no-op
        if self.strategy == "NONE":
            return df.copy()

        out = df.copy()

        numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
        excl = set(self._exclude_list())
        numeric_cols = [c for c in numeric_cols if c not in excl]

        if len(numeric_cols) == 0:
            return out

        # Replace inf with NaN for safety (do not fill NaNs here; MV module should handle)
        X = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

        if self.strategy in ("SNAP", "BOTH"):
            X = X.mask(X.abs() < self.tol, 0.0)

        if self.strategy in ("ROUND", "BOTH"):
            X = X.round(self.decimals)

        out[numeric_cols] = X
        return out

    # ------------------- driver -------------------

    def transform(self):

        start_time = time.time()
        outd = self.dataset

        print(">>Floating point stabilization ")

        for key in ['train']:

            if (isinstance(self.dataset, dict)
                    and key in self.dataset
                    and (not isinstance(self.dataset[key], dict))):

                d = self.dataset[key].copy()
                print("* For", key, "dataset")

                if self.verbose:
                    # diagnostics (optional)
                    n_inf = int(np.isinf(d.select_dtypes(include=["number"]).to_numpy()).sum()) \
                        if d.select_dtypes(include=["number"]).shape[1] > 0 else 0
                    print("Before: inf count (numeric) =", n_inf)

                dn = self._apply_to_df(d)
                outd[key] = dn

                # apply to test if present
                if "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                    outd["test"] = self._apply_to_df(self.dataset["test"].copy())

                if self.verbose:
                    dnum = outd[key].select_dtypes(include=["number"])
                    n_inf_after = int(np.isinf(dnum.to_numpy()).sum()) if dnum.shape[1] > 0 else 0
                    print("After:  inf count (numeric) =", n_inf_after)

            else:
                print("No", key, "dataset, no floating point stabilization")

        print("Floating point stabilization done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return outd
