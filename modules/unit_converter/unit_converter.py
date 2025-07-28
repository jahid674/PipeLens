import pandas as pd

class UnitConverter:
    def __init__(self, dataset, strategy='uc', verbose=False, column=None, multiplier=1.0, offset=0.0): 
        self.dataset = dataset.copy()
        self.verbose = verbose
        self.strategy = strategy.lower()
        self.column = column
        self.multiplier = multiplier
        self.offset = offset

    def transform(self):
        df = self.dataset.copy()

        if self.column not in df.columns:
            print(f"Warning: Column '{self.column}' not found. Skipping unit conversion.")
            return df

        if not pd.api.types.is_numeric_dtype(df[self.column]):
            print(f"Warning: Column '{self.column}' is not numeric. Skipping unit conversion.")
            return df

        # Apply transformation: new_value = old_value * multiplier + offset
        df[self.column] = df[self.column] * self.multiplier + self.offset

        return df
