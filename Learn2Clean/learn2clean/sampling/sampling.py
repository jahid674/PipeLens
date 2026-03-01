#!/usr/bin/env python3
# coding: utf-8
# Author: (adapted to Learn2Clean-compatible format)

import warnings
import time
import math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class DataSampler():
    """
    Data acquisition / sampling module (Learn2Clean compatible).

    Strategies (parameter name: strategy):
      - "FULL" / "NONE" : use the full dataset (no subsampling)
      - "RANDOM"        : uniform random sampling by fraction
      - "SNAPSHOT"      : take a head snapshot (supports fraction or absolute count)
      - "STRATIFIED"    : balanced sampling over a grouping variable (y or sensitive)

    Input:
      - dataset: dict-like with keys at least {'train'} (DataFrame), optionally {'test'}
      - Optional columns inside train/test:
          - if stratify_col == "y": needs column named `y_col` in df
          - if stratify_col == "sensitive": needs column named `sensitive_col` in df

    Output:
      - returns dataset dict with sampled train (and sampled test if present)
      - indices reset to 0..n-1 after sampling

    Important note:
      Learn2Clean modules typically operate on dataset dict only. Since your original
      sampler takes (X, y, sensitive) separately, this compatible version supports
      BOTH modes:
        1) If y_col/sensitive_col exist inside the DataFrame splits: it will sample
           and keep them aligned automatically.
        2) If they do not exist, it just samples rows of X (the DataFrame) only.
    """

    def __init__(self, dataset, strategy="FULL",
                 random_frac=0.2, snapshot_size=0.2,
                 stratify_col="sensitive", stratify_n_per_group=3000,
                 y_col="y", sensitive_col="sensitive",
                 random_state=42, verbose=False, exclude=None, threshold=None):

        self.dataset = dataset

        self.strategy = "FULL" if strategy is None else str(strategy).upper().strip()

        self.random_frac = float(random_frac) if random_frac is not None else 1.0
        self.snapshot_size = snapshot_size
        self.stratify_col = str(stratify_col).lower().strip()  # 'y' or 'sensitive'
        self.stratify_n_per_group = stratify_n_per_group
        self.y_col = str(y_col)
        self.sensitive_col = str(sensitive_col)

        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self.exclude = exclude  # unused, API compatibility
        self.threshold = threshold  # unused, API compatibility

        if self.strategy not in ("FULL", "NONE", "RANDOM", "SNAPSHOT", "STRATIFIED"):
            raise ValueError("Strategy invalid. Please choose between "
                             "'FULL'/'NONE', 'RANDOM', 'SNAPSHOT', or 'STRATIFIED'.")

    # ------------------- Learn2Clean API -------------------

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'random_frac': self.random_frac,
            'snapshot_size': self.snapshot_size,
            'stratify_col': self.stratify_col,
            'stratify_n_per_group': self.stratify_n_per_group,
            'y_col': self.y_col,
            'sensitive_col': self.sensitive_col,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'exclude': self.exclude,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for DataSampler. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`data_sampler.get_params().keys()`")
            else:
                setattr(self, k, v)

        self.strategy = "FULL" if self.strategy is None else str(self.strategy).upper().strip()

    # ------------------- helpers -------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _reset(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def _full(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._reset(df)

    def _random_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        n_rows = len(df)
        if n_rows == 0:
            return self._reset(df)

        frac = self.random_frac
        if frac is None or frac <= 0 or frac > 1:
            frac = 1.0

        df_s = df.sample(frac=frac, random_state=self.random_state)
        return self._reset(df_s)

    def _snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        n_rows = len(df)
        if n_rows == 0:
            return self._reset(df)

        ss = self.snapshot_size
        if ss is None or ss <= 0:
            return self._full(df)

        # interpret snapshot_size
        if isinstance(ss, float):
            if ss <= 1.0:
                size = int(math.ceil(ss * n_rows))
            else:
                size = int(ss)
        else:
            size = int(ss)

        size = max(1, min(size, n_rows))
        return self._reset(df.iloc[:size])

    def _stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stratified sampling over a grouping variable column inside df:
          - stratify_col == 'y'        -> uses df[y_col] if exists
          - stratify_col == 'sensitive'-> uses df[sensitive_col] if exists
        If grouping column missing, falls back to random sampling.
        """
        if len(df) == 0:
            return self._reset(df)

        if self.stratify_col == "y" and self.y_col in df.columns:
            group_series = df[self.y_col]
        elif self.stratify_col == "sensitive" and self.sensitive_col in df.columns:
            group_series = df[self.sensitive_col]
        else:
            self._log("Stratified sampling: grouping column not found; falling back to RANDOM.")
            return self._random_sample(df)

        tmp = df.copy()
        tmp["_group"] = group_series

        counts = tmp["_group"].value_counts(dropna=False)
        if counts.empty:
            return self._full(df)

        if self.stratify_n_per_group is None or self.stratify_n_per_group <= 0:
            n_per_group = int(counts.min())
        else:
            n_per_group = int(min(int(self.stratify_n_per_group), int(counts.min())))

        if n_per_group <= 0:
            return self._full(df)

        sampled_idx = (
            tmp.groupby("_group", group_keys=False)
            .apply(lambda g: g.sample(n=n_per_group, random_state=self.random_state))
            .index
        )

        df_new = df.loc[sampled_idx]
        return self._reset(df_new)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.strategy in ("FULL", "NONE"):
            self._log("Sampling: using full dataset.")
            return self._full(df)

        if self.strategy == "RANDOM":
            self._log(f"Sampling: random sampling frac={self.random_frac}.")
            return self._random_sample(df)

        if self.strategy == "SNAPSHOT":
            self._log(f"Sampling: snapshot size={self.snapshot_size}.")
            return self._snapshot(df)

        if self.strategy == "STRATIFIED":
            self._log(f"Sampling: stratified on '{self.stratify_col}' "
                      f"(n_per_group={self.stratify_n_per_group}).")
            return self._stratified_sample(df)

        raise ValueError("Invalid sampling strategy. Choose from "
                         "'FULL'/'NONE', 'RANDOM', 'SNAPSHOT', 'STRATIFIED'.")

    # ------------------- driver -------------------

    def transform(self):

        start_time = time.time()
        outd = self.dataset

        print(">>Data sampling ")

        for key in ['train']:

            if (isinstance(self.dataset, dict)
                    and key in self.dataset
                    and (not isinstance(self.dataset[key], dict))):

                d = self.dataset[key].copy()
                print("* For", key, "dataset")

                outd[key] = self._apply(d)

                # Apply the SAME sampling strategy to test (independently).
                # If you need "sample train and apply same indices to test", say so.
                if "test" in self.dataset and self.dataset["test"] is not None and not isinstance(self.dataset["test"], dict):
                    outd["test"] = self._apply(self.dataset["test"].copy())

            else:
                print("No", key, "dataset, no data sampling")

        print("Data sampling done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()

        return outd
