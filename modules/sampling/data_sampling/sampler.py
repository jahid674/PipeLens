# =========================
# FILE: modules/sampling/data_sampling/sampler.py
# =========================
import math
import pandas as pd
import numpy as np
np.random.seed(42)


class DataSampler:
    """
    Data acquisition / sampling module.

    Strategies:
    - 'full' / None / 'none': use the full dataset (no subsampling)
    - 'random': uniform random sampling by fraction
    - 'snapshot': take a head snapshot (supports fraction or absolute count)
    - 'stratified': balanced sampling over a grouping variable (y or sensitive)

    Returns aligned (X_new, y_new, sensitive_new) with reset indices.
    """

    def __init__(
        self,
        dataset,
        strategy=None,
        random_frac=0.2,
        snapshot_size=0.2,
        stratify_col="sensitive",
        stratify_n_per_group=3000,
        random_state=42,
        verbose=False,
    ):
        self.dataset = dataset.copy()

        if strategy is None:
            self.strategy = "full"
        else:
            self.strategy = str(strategy).lower()

        self.random_frac = random_frac
        self.snapshot_size = snapshot_size
        self.stratify_col = stratify_col  # 'y' or 'sensitive'
        self.stratify_n_per_group = stratify_n_per_group
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _full(self, df, y=None, sensitive=None):
        df_new = df.reset_index(drop=True)
        y_new = y.reset_index(drop=True) if y is not None else None
        s_new = sensitive.reset_index(drop=True) if sensitive is not None else None
        return df_new, y_new, s_new

    def _random_sample(self, df, y=None, sensitive=None):
        n_rows = len(df)
        if n_rows == 0:
            return df, y, sensitive

        frac = self.random_frac
        if frac is None or frac <= 0 or frac > 1:
            frac = 1.0

        df_sampled = df.sample(frac=frac, random_state=self.random_state)
        idx = df_sampled.index

        y_new = y.loc[idx] if y is not None else None
        s_new = sensitive.loc[idx] if sensitive is not None else None

        df_sampled = df_sampled.reset_index(drop=True)
        if y_new is not None:
            y_new = y_new.reset_index(drop=True)
        if s_new is not None:
            s_new = s_new.reset_index(drop=True)

        return df_sampled, y_new, s_new

    def _snapshot(self, df, y=None, sensitive=None):
        """
        snapshot_size can be:
          - float in (0,1]: fraction of rows (e.g., 0.2 means 20%)
          - int >= 1: absolute row count (e.g., 5000)
        """
        n_rows = len(df)
        if n_rows == 0:
            return df, y, sensitive

        ss = self.snapshot_size
        if ss is None or ss <= 0:
            return self._full(df, y, sensitive)

        # interpret snapshot_size
        if isinstance(ss, float):
            if ss <= 1.0:
                size = int(math.ceil(ss * n_rows))
            else:
                # defensive: float but > 1 => treat as count
                size = int(ss)
        else:
            # int-like (including numpy ints)
            size = int(ss)

        size = max(1, min(size, n_rows))

        df_snap = df.iloc[:size]
        idx = df_snap.index

        y_new = y.loc[idx] if y is not None else None
        s_new = sensitive.loc[idx] if sensitive is not None else None

        df_snap = df_snap.reset_index(drop=True)
        if y_new is not None:
            y_new = y_new.reset_index(drop=True)
        if s_new is not None:
            s_new = s_new.reset_index(drop=True)

        return df_snap, y_new, s_new

    def _stratified_sample(self, df, y=None, sensitive=None):
        """
        Stratified sampling over a grouping variable:
        - if stratify_col == 'y'        => group by y
        - if stratify_col == 'sensitive'=> group by sensitive
        else falls back to random sampling.
        """
        if len(df) == 0:
            return df, y, sensitive

        # determine grouping series
        if self.stratify_col == "y" and y is not None:
            group_series = y
        elif self.stratify_col == "sensitive" and sensitive is not None:
            group_series = sensitive
        else:
            return self._random_sample(df, y, sensitive)

        data = df.copy()
        data["_group"] = group_series

        counts = data["_group"].value_counts(dropna=False)
        if self.stratify_n_per_group is None or self.stratify_n_per_group <= 0:
            n_per_group = int(counts.min())
        else:
            n_per_group = int(min(self.stratify_n_per_group, counts.min()))

        if n_per_group <= 0:
            return self._full(df, y, sensitive)

        sampled_idx = (
            data.groupby("_group", group_keys=False)
            .apply(lambda g: g.sample(n=n_per_group, random_state=self.random_state))
            .index
        )

        df_new = df.loc[sampled_idx]
        y_new = y.loc[sampled_idx] if y is not None else None
        s_new = sensitive.loc[sampled_idx] if sensitive is not None else None

        df_new = df_new.reset_index(drop=True)
        if y_new is not None:
            y_new = y_new.reset_index(drop=True)
        if s_new is not None:
            s_new = s_new.reset_index(drop=True)

        return df_new, y_new, s_new

    def transform(self, y=None, sensitive=None):
        df = self.dataset.copy()

        if self.strategy in ["full", "none"]:
            self._log("Sampling: using full dataset.")
            return self._full(df, y, sensitive)

        if self.strategy == "random":
            self._log(f"Sampling: random sampling frac={self.random_frac}.")
            return self._random_sample(df, y, sensitive)

        if self.strategy == "snapshot":
            self._log(f"Sampling: snapshot size={self.snapshot_size}.")
            return self._snapshot(df, y, sensitive)

        if self.strategy == "stratified":
            self._log(
                f"Sampling: stratified on '{self.stratify_col}' "
                f"(n_per_group={self.stratify_n_per_group})."
            )
            return self._stratified_sample(df, y, sensitive)

        raise ValueError(
            f"Invalid sampling strategy: {self.strategy}. "
            f"Choose from None/'full', 'random', 'snapshot', 'stratified'."
        )
