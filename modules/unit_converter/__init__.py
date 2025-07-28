import pandas as pd
import warnings

class UnitConverter:
    def __init__(self, dataset, column=None, multiplier=1.0, offset=0.0, verbose=False):
        self.dataset = dataset.copy()
        self.column = column
        self.multiplier = multiplier
        self.offset = offset
        self.verbose = verbose

    def transform(self):
        df = self.dataset.copy()
        if self.column not in df.columns:
            warnings.warn(f"Column '{self.column}' not found. Unit conversion skipped.")
            return df

        if self.verbose:
            print(f"Converting units in column '{self.column}' with multiplier={self.multiplier} and offset={self.offset}...")

        try:
            df[self.column] = pd.to_numeric(df[self.column], errors='coerce')
            df[self.column] = df[self.column] * self.multiplier + self.offset
        except Exception as e:
            warnings.warn(f"Unit conversion failed: {e}")
        return df