import pandas as pd

class Deduplicator:
    def __init__(self, dataset, strategy='dd', subset=None, verbose=False):
        self.dataset = dataset.copy()
        self.subset = subset
        self.verbose = verbose
        self.strategy = strategy.lower()

    def transform(self, y=None, sensitive=None):
        before = len(self.dataset)
        self.dataset = self.dataset.drop_duplicates(subset=self.subset).reset_index(drop=True)
        after = len(self.dataset)

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.reset_index(drop=True)
                y = y.loc[self.dataset.index].reset_index(drop=True)
            else:
                y = pd.Series(y).loc[self.dataset.index].reset_index(drop=True)

        if sensitive is not None:
            sensitive = sensitive.reset_index(drop=True)
            sensitive = sensitive.loc[self.dataset.index].reset_index(drop=True)

        if self.verbose:
            print(f"Removed {before - after} duplicate rows.")

        return self.dataset, y, sensitive

