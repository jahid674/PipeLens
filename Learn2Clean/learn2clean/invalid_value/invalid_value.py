#!/usr/bin/env python3
# coding: utf-8
# Author: (adapted to Learn2Clean-compatible format)

import warnings
import time
import re
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class InvalidValueRepair():
    """
    Invalid Value Repair (Learn2Clean compatible).

    Goal:
      Convert placeholder/sentinel/invalid tokens into NaN so that the missing-value
      module can handle them consistently.

    Strategies:
      - "NONE"      : no-op
      - "SENTINEL"  : replace numeric sentinel values (e.g., -999, 9999) with NaN
      - "REGEX"     : replace string tokens/patterns (e.g., "N/A", "na", "unknown", "") with NaN
      - "BOTH"      : apply both sentinel and regex

    Behavior:
      - Does not change row count
      - Applies to all columns unless excluded
      - For numeric columns: sentinel replacement
      - For object/string columns: regex replacement (case-insensitive by default)
    """

    def __init__(self, dataset, strategy="BOTH",
                 numeric_sentinels=None, string_patterns=None,
                 case_insensitive=True, strip_whitespace=True,
                 verbose=False, exclude=None, threshold=None):

        self.dataset = dataset
        self.strategy = str(strategy).upper().strip()

        self.numeric_sentinels = (numeric_sentinels if numeric_sentinels is not None
                                  else [-999, -9999, 999, 9999, 99999])

        self.string_patterns = (string_patterns if string_patterns is not None else [
            r"^\s*$",           # empty
            r"^na$", r"^n/a$", r"^none$", r"^null$", r"^nan$",
            r"^unknown$", r"^missing$",
            r"^\?$"
        ])

        self.case_insensitive = bool(case_insensitive)
        self.strip_whitespace = bool(strip_whitespace)
        self.verbose = bool(verbose)
        self.exclude = exclude  # can be str or list; handled internally
        self.threshold = threshold  # unused, kept for API compatibility

        if self.strategy not in ("NONE", "SENTINEL", "REGEX", "BOTH"):
            raise ValueError("Strategy invalid. Please choose between "
                             "'NONE', 'SENTINEL', 'REGEX', or 'BOTH'.")

        # compile patterns once
        flags = re.IGNORECASE if self.case_insensitive else 0
        self._compiled_patterns = [re.compile(p, flags=flags) for p in self.string_patterns]

    # ------------------- Learn2Clean API -------------------

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'numeric_sentinels': self.numeric_sentinels,
            'string_patterns': self.string_patterns,
            'case_insensitive': self.case_insensitive,
            'strip_whitespace': self.strip_whitespace,
            'verbose': self.verbose,
            'exclude': self.exclude,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for InvalidValueRepair. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`invalid_value_repair.get_params().keys()`")
            else:
                setattr(self, k, v)

        # normalize strategy if user changed it
        self.strategy = str(self.strategy).upper().strip()

        # recompile patterns if user changed them or case-insensitivity
        flags = re.IGNORECASE if bool(self.case_insensitive) else 0
        self._compiled_patterns = [re.compile(p, flags=flags) for p in self.string_patterns]

    # ------------------- helpers -------------------

    def _exclude_list(self):
        if self.exclude is None:
            return []
        if isinstance(self.exclude, list):
            return [c for c in self.exclude if c is not None]
        return [self.exclude]

    def _to_nan_if_match(self, x: str):
        for pat in self._compiled_patterns:
            if pat.match(x):
                return np.nan
        return x

    def _repair_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # true no-op
        if self.strategy == "NONE":
            return df.copy()

        out = df.copy()
        excl = set(self._exclude_list())
        cols = [c for c in out.columns if c not in excl]
        if not cols:
            return out

        # numeric sentinel replacement
        if self.strategy in ("SENTINEL", "BOTH"):
            num_cols = out[cols].select_dtypes(include=["number"]).columns.tolist()
            if len(num_cols) > 0:
                out[num_cols] = out[num_cols].replace(self.numeric_sentinels, np.nan)

        # string token/pattern replacement
        if self.strategy in ("REGEX", "BOTH"):
            str_cols = out[cols].select_dtypes(include=["object", "string"]).columns.tolist()
            for c in str_cols:
                s = out[c]
                # keep NaNs as NaNs; do not force them into 'nan' strings
                if self.strip_whitespace:
                    s2 = s.astype("string").str.strip()
                else:
                    s2 = s.astype("string")
                # apply regex match only to non-missing values
                out[c] = s2.where(s2.isna(), s2.apply(lambda v: self._to_nan_if_match(str(v))))

        return out

    # ------------------- driver -------------------

    def transform(self):

        start_time = time.time()
        ivrd = self.dataset

        print(">>Invalid value repair ")

        for key in ['train']:

            if (isinstance(self.dataset, dict)
                    and key in self.dataset
                    and (not isinstance(self.dataset[key], dict))):

                d = self.dataset[key].copy()
                print("* For", key, "dataset")

                if self.verbose:
                    # quick counts before
                    before_nan = int(d.isna().sum().sum())
                    print("Before repair: Total", before_nan, "NaN cells")

                dn = self._repair_df(d)
                ivrd[key] = dn

                # also repair test if present
                if "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                    ivrd["test"] = self._repair_df(self.dataset["test"].copy())

                if self.verbose:
                    after_nan = int(ivrd[key].isna().sum().sum())
                    print("After repair:  Total", after_nan, "NaN cells")

            else:
                print("No", key, "dataset, no invalid value repair")

        print("Invalid value repair done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return ivrd
