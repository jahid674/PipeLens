import warnings
import time
import pandas as pd
pd.options.mode.chained_assignment = None


class Deduplicator():
    """
    Deduplicate the dataset.

    Parameters
    ----------
    * strategy : str, default = 'FIRST'
      Available strategies:
        - 'FIRST' : remove duplicates, keeping the first occurrence
        - 'NONE'  : do nothing (no deduplication)

    * subset : list[str] or None, default=None
      Columns to consider when identifying duplicates. If None, use all columns.

    * exclude : variable to exclude from processing
      (restored from original dataset after processing if present)

    * verbose : bool, default=False
      Print information about deduplication.

    * threshold : float or None
      Unused, for API compatibility.
    """

    def __init__(self, dataset, strategy='FIRST', subset=None,
                 exclude=None, verbose=False, threshold=None):

        self.dataset = dataset
        self.strategy = strategy.upper()
        self.subset = subset
        self.exclude = exclude
        self.verbose = verbose
        self.threshold = threshold

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'subset': self.subset,
            'exclude': self.exclude,
            'verbose': self.verbose,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for deduplicator. "
                              "Parameter(s) IGNORED. "
                              "Check with `deduplicator.get_params().keys()`")
            else:
                setattr(self, k, v)

    # === Strategy: deduplicate keeping first occurrence ===
    def FIRST(self, dataset):
        d = dataset
        if self.verbose:
            print("Deduplicating (keep='first')...")

        if not isinstance(dataset, pd.core.series.Series):
            before = len(dataset)
            df = dataset.drop_duplicates(subset=self.subset, keep='first')
            after = len(df)

            if self.exclude in list(df.columns.values):
                df[self.exclude] = d[self.exclude].loc[df.index]

            if self.verbose:
                print(f"Removed {before - after} duplicate rows.")
        else:
            # Series case
            before = len(dataset)
            mask = ~dataset.duplicated(keep='first')
            df = dataset[mask]
            after = len(df)
            if self.verbose:
                print(f"Removed {before - after} duplicate entries in Series.")

        return df.sort_index()

    # === Strategy: no deduplication applied ===
    def NONE(self, dataset):
        if self.verbose:
            print("No deduplication applied.")
        return dataset.sort_index()

    def transform(self):
        outd = self.dataset
        start_time = time.time()
        print(">>Deduplication ")

        for key in ['train']:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                print("* For", key, "dataset")

                if self.strategy == "FIRST":
                    dn = self.FIRST(d)
                elif self.strategy == "NONE_DEDUP":
                    dn = self.NONE(d)
                else:
                    raise ValueError("Unknown strategy. Choose 'FIRST' or 'NONE_dedup'.")

                if (self.exclude in list(pd.DataFrame(d).columns.values)):
                    dn[self.exclude] = d[self.exclude]

                outd[key] = dn
                print('...', key, 'dataset')
            else:
                outd[key] = self.dataset[key]
                print('No', key, 'dataset, no deduplication')

        print("Deduplication done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()
        return outd
