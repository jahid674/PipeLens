import pandas as pd

class Deduplicator:
    def __init__(self, dataset, strategy='dd', subset=None, verbose=False):
        self.dataset = dataset.copy()
        self.subset = subset
        self.verbose = verbose
        self.strategy = strategy.lower()

    def transform(self, y=None, sensitive=None):
        before_all = len(self.dataset)
        na_mask = self.dataset.notna().all(axis=1)
        self.dataset = self.dataset.loc[na_mask].reset_index(drop=True)

        if y is not None:
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y = y.loc[na_mask].reset_index(drop=True)
            else:
                y = pd.Series(y).loc[na_mask].reset_index(drop=True)

        if sensitive is not None:
            sensitive = sensitive.loc[na_mask].reset_index(drop=True)

        if self.verbose:
            print(f"Dropped {before_all - len(self.dataset)} rows with missing values.")

        before = len(self.dataset)

        if self.strategy == 'dd':
            self.dataset = self.dataset.drop_duplicates().reset_index(drop=True)
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
        elif self.strategy == 'none':
            if self.verbose:
                print("No deduplication applied (strategy='NONE').")
            return self.dataset, y, sensitive

        return self.dataset, y, sensitive
