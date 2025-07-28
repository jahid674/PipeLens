import pandas as pd
import re
import warnings

class SpecialCharRemover:
    def __init__(self, dataset, strategy='sc', verbose=False):
        self.dataset = dataset.copy()
        self.text_column = self.dataset.select_dtypes(include=['object', 'string']).columns
        self.verbose = verbose
        self.strategy = strategy.lower()

    def transform(self):
        df = self.dataset.copy()
        if self.text_column is None or len(self.text_column) == 0:
            warnings.warn("No Text column. Skipping special char removal.")
            return df

        df[self.text_column] = df[self.text_column].astype(str).apply(
            lambda x: re.sub(r'[^\w\s]', '', x.encode('ascii', 'ignore').decode('ascii'))
        )
        return df