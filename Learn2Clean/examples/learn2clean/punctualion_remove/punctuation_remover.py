#!/usr/bin/env python3
# coding: utf-8

import re
import warnings
import time
import pandas as pd
pd.options.mode.chained_assignment = None


class PunctuationRemover:
    """
    Remove punctuation from text columns.

    Parameters
    ----------
    dataset : dict-like with keys 'train','test','target' (or a DataFrame for fallback use)
    strategy : {'NONE','PR','NONE_punct'}  (case-insensitive)
        - 'NONE' / 'NONE_punct' : no-op
        - 'PR'                  : remove punctuation from detected text columns
    text_column : str | list[str] | None
        Columns to process. If None, auto-detect object/string columns from train set.
    exclude : str | None
        Column to safeguard from processing (restored afterward if present).
    verbose : bool
    threshold : unused (API compatibility)
    """

    def __init__(self, dataset, strategy='NONE_punct', text_column=None,
                 exclude=None, verbose=False, threshold=None):
        self.dataset = dataset
        self.strategy = strategy.upper()
        self.text_column = text_column
        self.exclude = exclude
        self.verbose = verbose
        self.threshold = threshold

        # Normalize text_column to list[str] if provided
        if isinstance(self.text_column, str):
            self.text_column = [self.text_column]

        # Auto-detect text columns if not provided
        try:
            # Prefer train split if dataset is a dict
            df = dataset['train'] if isinstance(dataset, dict) else dataset
            if self.text_column is None and isinstance(df, pd.DataFrame):
                self.text_column = df.select_dtypes(include=['object', 'string']).columns.tolist()
        except Exception:
            # If anything goes wrong, leave text_column as-is (can be handled later)
            pass

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
                warnings.warn("Invalid parameter(s) for PunctuationRemover. "
                              "Parameter(s) IGNORED. "
                              "Check with `punctuation_remover.get_params().keys()`")
            else:
                setattr(self, k, v)

    # ---- strategies ----
    def _none(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print("No punctuation removal applied (strategy='NONE').")
        return df.sort_index()

    def _pr(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.text_column or len(self.text_column) == 0:
            warnings.warn("No text columns found. Skipping punctuation removal.")
            return df

        if self.verbose:
            print(f"Removing punctuation from columns {self.text_column}...")

        for col in self.text_column:
            if col in df.columns:
                # Ensure string dtype for safe .str operations
                if not pd.api.types.is_string_dtype(df[col]):
                    try:
                        df[col] = df[col].astype('string')
                    except Exception:
                        df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)

        # Safeguard exclude column if requested
        if self.exclude in df.columns:
            df[self.exclude] = df[self.exclude]

        return df.sort_index()

    def transform(self):
        outd = self.dataset
        start_time = time.time()
        print(">>Punctuation Removal ")

        # unify strategy labels
        strat = self.strategy
        if strat == 'NONE_PUNCT':
            strat = 'NONE'

        for key in ['train']:
            if isinstance(self.dataset, dict) and key in self.dataset and not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                if not isinstance(d, pd.DataFrame):
                    warnings.warn(f"Expected DataFrame at dataset['{key}']; skipping punctuation removal.")
                    outd[key] = self.dataset[key]
                    continue

                print("* For", key, "dataset")

                if strat == "NONE":
                    dn = self._none(d.copy())
                elif strat == "PR":
                    dn = self._pr(d.copy())
                else:
                    raise ValueError("Unknown strategy. Choose 'NONE' (or 'NONE_punct') or 'PR'.")

                # Restore exclude if needed
                if self.exclude in d.columns:
                    dn[self.exclude] = d[self.exclude]

                outd[key] = dn
                print('...', key, 'dataset')
            else:
                # If dataset is not in the expected dict form, just return it unchanged
                outd = self.dataset
                print('No', key, 'dataset, no punctuation removal')

        print("Punctuation removal done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()
        return outd
