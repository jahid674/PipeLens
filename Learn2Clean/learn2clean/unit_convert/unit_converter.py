#!/usr/bin/env python3
# coding: utf-8
# Learn2Clean-compatible UnitConverter (rewritten)

import warnings
import time
import pandas as pd
pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class UnitConverter():
    """
    Learn2Clean-compatible Unit Converter.

    Convert numeric values in a column using:  x_new = x * multiplier + offset

    Parameters
    ----------
    dataset : dict-like with keys {'train'} (DataFrame), optionally {'test'}
    strategy : {'NONE_convert','UC'}
        - 'NONE_convert' : no-op
        - 'UC'           : apply conversion
    column : str | None
        Column to apply conversion on (must be numeric).
    multiplier : float, default=1.0
    offset : float, default=0.0
    exclude : str | None
        Column to safeguard (restored afterward if present).
    verbose : bool
    threshold : unused (kept for API compatibility)
    """

    def __init__(self, dataset, strategy='NONE_convert', column=None,
                 multiplier=1.0, offset=0.0,
                 exclude=None, verbose=False, threshold=None):

        self.dataset = dataset
        self.strategy = str(strategy).upper().strip()
        self.column = column
        self.multiplier = float(multiplier)
        self.offset = float(offset)
        self.exclude = exclude
        self.verbose = bool(verbose)
        self.threshold = threshold

    # ---------------- Learn2Clean API ----------------

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
        # normalize after setting
        self.strategy = str(self.strategy).upper().strip()

    # ---------------- strategies ----------------

    def NONE(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print("No unit conversion applied (strategy='NONE_convert').")
        return df.sort_index()

    def UC(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if self.column is None:
            warnings.warn("UnitConverter: `column` is None. Skipping conversion.")
            return out

        if self.column not in out.columns:
            warnings.warn(f"UnitConverter: column '{self.column}' not found. Skipping conversion.")
            return out

        if not pd.api.types.is_numeric_dtype(out[self.column]):
            warnings.warn(f"UnitConverter: column '{self.column}' is not numeric. Skipping conversion.")
            return out

        if self.verbose:
            print(f"Converting '{self.column}' with multiplier={self.multiplier}, offset={self.offset}")

        out[self.column] = out[self.column] * self.multiplier + self.offset
        return out.sort_index()

    # ---------------- driver ----------------

    def transform(self):
        start_time = time.time()
        outd = self.dataset

        print(">>Unit Conversion ")

        for key in ['train']:

            if isinstance(self.dataset, dict) and key in self.dataset and (not isinstance(self.dataset[key], dict)):

                d = self.dataset[key]
                print("* For", key, "dataset")

                # safeguard exclude column if requested
                excl_backup = None
                if self.exclude is not None and self.exclude in d.columns:
                    excl_backup = d[self.exclude].copy()

                if self.strategy in ("NONE_CONVERT", "NONE"):
                    dn = self.NONE(d.copy())
                elif self.strategy == "UC":
                    dn = self.UC(d.copy())
                else:
                    raise ValueError("Unknown strategy. Choose 'NONE_convert' (or 'NONE') or 'UC'.")

                # restore exclude column if needed
                if excl_backup is not None and self.exclude in dn.columns:
                    dn[self.exclude] = excl_backup

                outd[key] = dn
                print('...', key, 'dataset')

                # If your pipeline keeps 'test', apply same conversion there too
                if isinstance(self.dataset, dict) and "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                    dt = self.dataset["test"]
                    excl_backup_t = None
                    if self.exclude is not None and self.exclude in dt.columns:
                        excl_backup_t = dt[self.exclude].copy()

                    if self.strategy in ("NONE_CONVERT", "NONE"):
                        dnt = self.NONE(dt.copy())
                    else:
                        dnt = self.UC(dt.copy())

                    if excl_backup_t is not None and self.exclude in dnt.columns:
                        dnt[self.exclude] = excl_backup_t

                    outd["test"] = dnt

            else:
                print('No', key, 'dataset, no unit conversion')

        print("Unit conversion done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return outd
