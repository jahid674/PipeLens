#!/usr/bin/env python3
# coding: utf-8
# Learn2Clean-compatible WhitespaceCleaner

import warnings
import time
import pandas as pd
pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class WhitespaceCleaner():
    """
    Remove redundant whitespace from text columns.

    Strategies:
      - "WC"   : collapse multiple spaces and trim leading/trailing whitespace
      - "NONE" : no operation

    Behavior:
      - Automatically detects object/string columns
      - Does not change row count
      - Preserves non-text columns
      - Applies to train and test consistently
    """

    def __init__(self, dataset, strategy='WC',
                 text_column=None, exclude=None,
                 verbose=False, threshold=None):

        self.dataset = dataset
        self.strategy = str(strategy).upper().strip()
        self.text_column = text_column  # optional manual column selection
        self.exclude = exclude
        self.verbose = bool(verbose)
        self.threshold = threshold  # unused, for compatibility

    # ---------------- Learn2Clean API ----------------

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'text_column': self.text_column,
            'exclude': self.exclude,
            'verbose': self.verbose,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for WhitespaceCleaner. "
                              "Parameter(s) IGNORED. "
                              "Check with `whitespace_cleaner.get_params().keys()`")
            else:
                setattr(self, k, v)
        self.strategy = str(self.strategy).upper().strip()

    # ---------------- helpers ----------------

    def _detect_text_columns(self, df: pd.DataFrame):
        """Auto-detect text columns if not specified."""
        if self.text_column is None:
            cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        else:
            if isinstance(self.text_column, str):
                cols = [self.text_column]
            else:
                cols = list(self.text_column)

        if self.exclude is not None and self.exclude in cols:
            cols.remove(self.exclude)

        return cols

    def _wc(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        text_cols = self._detect_text_columns(out)

        if len(text_cols) == 0:
            warnings.warn("No text column found. Skipping whitespace cleaning.")
            return out

        if self.verbose:
            print(f"Stripping extra whitespace from column(s): {text_cols}")

        for col in text_cols:
            if col in out.columns:
                out[col] = out[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

        return out.sort_index()

    def _none(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print("No whitespace cleaning applied (strategy='NONE').")
        return df.sort_index()

    # ---------------- driver ----------------

    def transform(self):

        start_time = time.time()
        outd = self.dataset

        print(">>Whitespace Cleaning ")

        for key in ['train']:

            if isinstance(self.dataset, dict) and key in self.dataset and (not isinstance(self.dataset[key], dict)):

                d = self.dataset[key]
                print("* For", key, "dataset")

                if self.strategy == "NONE":
                    dn = self._none(d.copy())
                elif self.strategy == "WC":
                    dn = self._wc(d.copy())
                else:
                    raise ValueError("Unknown strategy. Choose 'WC' or 'NONE'.")

                outd[key] = dn
                print('...', key, 'dataset')

                # apply same to test
                if "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                    if self.strategy == "NONE":
                        outd["test"] = self._none(self.dataset["test"].copy())
                    else:
                        outd["test"] = self._wc(self.dataset["test"].copy())

            else:
                print('No', key, 'dataset, no whitespace cleaning')

        print("Whitespace cleaning done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return outd
