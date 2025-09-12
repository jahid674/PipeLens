#!/usr/bin/env python3
# coding: utf-8

import warnings
import time
import pandas as pd
pd.options.mode.chained_assignment = None


class Lowercaser():
    """
    Lowercase text in a dataset column.

    Parameters
    ----------
    * strategy : str, default = 'NONE'
      Available strategies:
        - 'NONE' : do nothing (no lowercasing)
        - 'LC'   : lowercase text

    * text_column : str or None, default=None
      Column to apply lowercasing on.

    * exclude : str or None
      Column to safeguard from processing (restored afterward if present).

    * verbose : bool, default=False
      Print information about processing.

    * threshold : float or None
      Unused; kept for compatibility with Learn2Clean components.
    """

    def __init__(self, dataset, strategy='NONE_lowercase', text_column=None,
                 exclude=None, verbose=False, threshold=None):

        self.dataset = dataset
        self.strategy = strategy.upper()
        self.text_column = text_column
        self.exclude = exclude
        self.verbose = verbose
        self.threshold = threshold

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
                warnings.warn("Invalid parameter(s) for Lowercaser. "
                              "Parameter(s) IGNORED. "
                              "Check with `lowercaser.get_params().keys()`")
            else:
                setattr(self, k, v)

    # === Strategy: do nothing ===
    def NONE(self, dataset):
        if self.verbose:
            print("No lowercasing applied (strategy='NONE').")
        return dataset.sort_index()

    # === Strategy: lowercase ===
    def LC(self, dataset):
        d = dataset

        if not pd.api.types.is_string_dtype(d[self.text_column]):
            warnings.warn(f"Column '{self.text_column}' is not string type. Lowercasing skipped.")
            return d

        if self.verbose:
            print(f"Lowercasing column '{self.text_column}'...")

        d[self.text_column] = d[self.text_column].str.lower()

        if (self.exclude in list(d.columns.values)):
            d[self.exclude] = dataset[self.exclude]

        return d.sort_index()

    def transform(self):
        outd = self.dataset
        start_time = time.time()
        print(">>Lowercasing ")

        for key in ['train']:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                print("* For", key, "dataset")

                if self.strategy == "NONE_LOWERCASE":
                    dn = self.NONE(d)
                elif self.strategy == "LC":
                    dn = self.LC(d)
                else:
                    raise ValueError("Unknown strategy. Choose 'NONE_lowercase' or 'LC'.")

                if (self.exclude in list(pd.DataFrame(d).columns.values)):
                    dn[self.exclude] = d[self.exclude]

                outd[key] = dn
                print('...', key, 'dataset')
            else:
                outd[key] = self.dataset[key]
                print('No', key, 'dataset, no lowercasing')

        print("Lowercasing done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()
        return outd
