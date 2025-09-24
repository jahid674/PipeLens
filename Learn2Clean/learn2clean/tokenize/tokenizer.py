#!/usr/bin/env python3
# coding: utf-8

import warnings
import time
import pandas as pd
pd.options.mode.chained_assignment = None
import nltk


class Tokenizer():
    """
    Tokenize text in dataset columns.

    Parameters
    ----------
    * strategy : str, default = 'NONE'
      Available strategies:
        - 'NONE' : do nothing (no tokenization)
        - 'WS'   : whitespace-based tokenization
        - 'NLTK' : NLTK word tokenizer

    * text_column : str or list[str] or None
      Columns to apply tokenization on. If None, auto-detects object/string columns.

    * exclude : str or None
      Column to safeguard from processing (restored afterward if present).

    * verbose : bool, default=False
      Print information about processing.

    * threshold : float or None
      Unused; for compatibility with Learn2Clean components.
    """

    def __init__(self, dataset, strategy='NONE_tokenize', text_column=None,
                 exclude=None, verbose=False, threshold=None):

        self.dataset = dataset
        self.strategy = strategy.upper()
        self.text_column = text_column
        self.exclude = exclude
        self.verbose = verbose
        self.threshold = threshold

        # auto-detect text columns if not provided
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
                warnings.warn("Invalid parameter(s) for Tokenizer. "
                              "Parameter(s) IGNORED. "
                              "Check with `tokenizer.get_params().keys()`")
            else:
                setattr(self, k, v)

    # === Strategy: do nothing ===
    def NONE(self, dataset):
        if self.verbose:
            print("No tokenization applied (strategy='NONE').")
        return dataset.sort_index()

    # === Strategy: whitespace tokenization ===
    def WS(self, dataset):
        d = dataset
        if not self.text_column or len(self.text_column) == 0:
            warnings.warn("No text columns found. Skipping tokenization.")
            return d

        if self.verbose:
            print(f"Tokenizing (whitespace) columns {self.text_column}...")

        for col in self.text_column:
            if pd.api.types.is_string_dtype(d[col]):
                d[col] = d[col].astype(str).apply(lambda x: x.split())

        if (self.exclude in list(d.columns.values)):
            d[self.exclude] = dataset[self.exclude]

        return d.sort_index()

    # === Strategy: nltk word_tokenize ===
    def NLTK(self, dataset):
        d = dataset
        if not self.text_column or len(self.text_column) == 0:
            warnings.warn("No text columns found. Skipping tokenization.")
            return d

        try:
            from nltk.tokenize import word_tokenize
        except ImportError:
            warnings.warn("NLTK not available. Falling back to whitespace tokenization.")
            return self.WS(d)

        if self.verbose:
            print(f"Tokenizing (NLTK) columns {self.text_column}...")

        for col in self.text_column:
            if pd.api.types.is_string_dtype(d[col]):
                d[col] = d[col].astype(str).apply(lambda x: word_tokenize(x))

        if (self.exclude in list(d.columns.values)):
            d[self.exclude] = dataset[self.exclude]

        return d.sort_index()

    def transform(self):
        outd = self.dataset
        start_time = time.time()
        print(">>Tokenization ")

        for key in ['train']:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                print("* For", key, "dataset")

                if self.strategy == "NONE_TOKENIZE":
                    dn = self.NONE(d)
                elif self.strategy == "WS":
                    dn = self.WS(d)
                elif self.strategy == "NLTK":
                    dn = self.NLTK(d)
                else:
                    raise ValueError("Unknown strategy. Choose 'NONE_tokenize', 'WS', or 'NLTK'.")

                if (self.exclude in list(pd.DataFrame(d).columns.values)):
                    dn[self.exclude] = d[self.exclude]

                outd[key] = dn
                print('...', key, 'dataset')
            else:
                outd[key] = self.dataset[key]
                print('No', key, 'dataset, no tokenization')

        print("Tokenization done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()
        return outd
