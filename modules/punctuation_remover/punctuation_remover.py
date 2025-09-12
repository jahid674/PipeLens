import re
import pandas as pd
import warnings

class PunctuationRemover:
    def __init__(self, dataset, strategy='pr', verbose=False):
        self.dataset = dataset.copy()
        self.text_column = self.dataset.select_dtypes(include=['object', 'string']).columns
        self.verbose = verbose
        self.strategy = strategy.lower()

    def transform(self):
        df = self.dataset.copy()
        if self.text_column is None or len(self.text_column) == 0:
            warnings.warn("No Text column. Skipping punctuation removal.")
            return df

        '''if not pd.api.types.is_string_dtype(df[self.text_column]):
            warnings.warn(f"Column '{self.text_column}' is not a string type. Punctuation removal skipped.")
            return df'''

        if self.verbose:
            print(f"Removing punctuation from column '{self.text_column}'...")
            
        if self.strategy == 'pr':
            df[self.text_column] = df[self.text_column].str.replace(r'[^\w\s]', '', regex=True)
        elif self.strategy == 'none':
            return df
        return df