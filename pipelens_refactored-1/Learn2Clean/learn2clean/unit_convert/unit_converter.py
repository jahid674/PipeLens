#!/usr/bin/env python3
# coding: utf-8

import warnings
import time
import pandas as pd
pd.options.mode.chained_assignment = None


class UnitConverter():
    """
    Convert numeric values in a column using multiplier and offset.

    Parameters
    ----------
    * strategy : str, default = 'NONE'
      Available strategies:
        - 'NONE' : do nothing (no unit conversion)
        - 'UC'   : apply conversion with multiplier and offset

    * column : str or None
      Column to apply conversion on.

    * multiplier : float, default=1.0
      Multiplication factor.

    * offset : float, default=0.0
      Value added after multiplication.

    * exclude : str or None
      Column to safeguard from processing (restored afterward if present).

    * verbose : bool, default=False
      Print information about conversion.

    * threshold : float or None
      Unused; kept for compatibility with Learn2Clean components.
    """

    def __init__(self, dataset, strategy='NONE_convert', column=None,
                 multiplier=1.0, offset=0.0,
                 exclude=None, verbose=False, threshold=None):

        self.dataset = dataset
        self.strategy = strategy.upper()
        self.column = column
        self.multiplier = multiplier
        self.offset = offset
        self.exclude = exclude
        self.verbose = verbose
        self.threshold = threshold

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'column': self.column,
            'multiplier': self.multiplier,
            'offset': self.offset,
            'exclude': self.exclude,
            'verbose': self.verbose,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for UnitConverter. "
                              "Parameter(s) IGNORED. "
                              "Check with `unitconverter.get_params().keys()`")
            else:
                setattr(self, k, v)

    # === Strategy: do nothing ===
    def NONE(self, dataset):
        if self.verbose:
            print("No unit conversion applied (strategy='NONE').")
        return dataset.sort_index()

    # === Strategy: unit conversion ===
    def UC(self, dataset):
        d = dataset

        if self.column not in d.columns:
            warnings.warn(f"Column '{self.column}' not found. Skipping unit conversion.")
            return d

        if not pd.api.types.is_numeric_dtype(d[self.column]):
            warnings.warn(f"Column '{self.column}' is not numeric. Skipping unit conversion.")
            return d

        if self.verbose:
            print(f"Converting column '{self.column}' with multiplier={self.multiplier}, offset={self.offset}")

        d[self.column] = d[self.column] * self.multiplier + self.offset

        if (self.exclude in list(d.columns.values)):
            d[self.exclude] = dataset[self.exclude]

        return d.sort_index()

    def transform(self):
        outd = self.dataset
        start_time = time.time()
        print(">>Unit Conversion ")

        for key in ['train']:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                print("* For", key, "dataset")

                if self.strategy == "NONE_CONVERT":
                    dn = self.NONE(d)
                elif self.strategy == "UC":
                    dn = self.UC(d)
                else:
                    raise ValueError("Unknown strategy. Choose 'NONE_convert' or 'UC'.")

                if (self.exclude in list(pd.DataFrame(d).columns.values)):
                    dn[self.exclude] = d[self.exclude]

                outd[key] = dn
                print('...', key, 'dataset')
            else:
                outd[key] = self.dataset[key]
                print('No', key, 'dataset, no unit conversion')

        print("Unit conversion done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()
        return outd
