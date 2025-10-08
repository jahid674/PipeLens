import pandas as pd
import warnings

class WhitespaceCleaner:
    def __init__(self, dataset, strategy='wc', verbose=False):
        self.dataset = dataset.copy()
        self.text_column = self.dataset.select_dtypes(include=['object', 'string']).columns
        self.verbose = verbose
        self.strategy = strategy.lower()

    def transform(self):
        df=self.dataset
        if self.text_column is None or len(self.text_column) == 0:
            warnings.warn("No Text column. Skipping whitespace removal.")
            return df

        # No text columns? return as-is

        if self.verbose:
            print(f"Stripping extra whitespace from column(s) '{list(self.text_column)}'...")

        if self.strategy == 'wc':
            df[self.text_column] = df[self.text_column].str.replace(r'\s+', ' ', regex=True).str.strip()
        elif self.strategy == 'none':
            return df

        return df
