import pandas as pd
import numpy as np
np.random.seed(42)
import time
import re


class InvalidValueRepair:
    """
    Invalid Value Repair module.

    Goal:
      Convert placeholder/sentinel/invalid tokens into NaN so that the missing-value
      module can handle them consistently.

    Strategies:
      - "none"     : no-op
      - "sentinel" : replace numeric sentinel values (e.g., -999, 9999) with NaN
      - "regex"    : replace string tokens/patterns (e.g., "N/A", "na", "unknown", "") with NaN
      - "both"     : apply both sentinel and regex

    Behavior:
      - Does not change row count; y and sensitive unchanged
      - Applies to all columns unless excluded
      - For numeric columns: sentinel replacement
      - For object/string columns: regex replacement (case-insensitive by default)
    """

    def __init__(
        self,
        dataset,
        strategy="both",
        numeric_sentinels=None,
        string_patterns=None,
        case_insensitive=True,
        strip_whitespace=True,
        verbose=False,
        exclude=None,
    ):
        self.dataset = dataset.copy()
        self.strategy = str(strategy).lower().strip()

        self.numeric_sentinels = numeric_sentinels if numeric_sentinels is not None else [-999, -9999, 999, 9999, 99999]
        self.string_patterns = string_patterns if string_patterns is not None else [
            r"^\s*$",           # empty
            r"^na$", r"^n/a$", r"^none$", r"^null$", r"^nan$",
            r"^unknown$", r"^missing$",
            r"^\?$"
        ]
        self.case_insensitive = bool(case_insensitive)
        self.strip_whitespace = bool(strip_whitespace)
        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

        if self.strategy not in ("none", "sentinel", "regex", "both"):
            raise ValueError("Invalid strategy. Choose from {'none','sentinel','regex','both'}.")

        flags = re.IGNORECASE if self.case_insensitive else 0
        self._compiled_patterns = [re.compile(p, flags=flags) for p in self.string_patterns]

    def _repair_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # true no-op
        if self.strategy == "none":
            return df.copy()

        out = df.copy()
        cols = [c for c in out.columns if c not in self.exclude]
        if not cols:
            return out

        # numeric sentinel replacement
        if self.strategy in ("sentinel", "both"):
            num_cols = out[cols].select_dtypes(include=["number"]).columns.tolist()
            if num_cols:
                out[num_cols] = out[num_cols].replace(self.numeric_sentinels, np.nan)

        # string token/pattern replacement
        if self.strategy in ("regex", "both"):
            str_cols = out[cols].select_dtypes(include=["object", "string"]).columns.tolist()

            def _to_nan_if_match(x: str):
                for pat in self._compiled_patterns:
                    if pat.match(x):
                        return np.nan
                return x

            for c in str_cols:
                s = out[c]
                if self.strip_whitespace:
                    s = s.astype(str).str.strip()
                else:
                    s = s.astype(str)

                out[c] = s.apply(_to_nan_if_match)

        return out

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Invalid Value Repair -----")

        # consistent with your pipeline: dataset is a DataFrame (X)
        df = self.dataset.copy()

        df = self._repair_df(df)

        if self.verbose:
            print(f"Invalid value repair completed in {time.time() - start_time:.2f} seconds.")

        return df
