import pandas as pd
import warnings

class Lowercaser:
    def __init__(self, dataset, strategy='lc', text_column=None, verbose=False):
        self.dataset = dataset.copy()
        self.text_column = text_column
        self.verbose = verbose
        self.strategy = strategy.lower()

    def transform(self):
        df = self.dataset.copy()
        if self.text_column not in df.columns:
            warnings.warn(f"Column '{self.text_column}' not found. Lowercasing skipped.")
            return df

        if not pd.api.types.is_string_dtype(df[self.text_column]):
            warnings.warn(f"Column '{self.text_column}' is not a string type. Lowercasing skipped.")
            return df

        if self.verbose:
            print(f"Lowercasing column '{self.text_column}'...")

        if self.strategy == 'lc':
            df[self.text_column] = df[self.text_column].str.lower()
        elif self.strategy == 'none':
            return df
        
        return df