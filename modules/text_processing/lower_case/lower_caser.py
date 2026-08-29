# =========================
# modules/text_processing/lower_case/lower_caser.py
# Updated to mirror WhitespaceCleaner behavior:
# - Auto-detect ALL text columns (object/string)
# - If no text columns, skip cleanly
# - Apply lowercase to all detected text columns when strategy == 'lc'
# =========================

import pandas as pd
import warnings
import numpy as np
np.random.seed(42)

class Lowercaser:
    def __init__(self, dataset, strategy='lc', verbose=False):
        self.dataset = dataset.copy()
        # Auto-detect text columns (same style as WhitespaceCleaner)
        self.text_columns = self.dataset.select_dtypes(include=['object', 'string']).columns
        self.verbose = verbose
        self.strategy = str(strategy).lower()

    def transform(self):
        df = self.dataset  # keep reference style like your WhitespaceCleaner

        # No text columns? skip
        if self.text_columns is None or len(self.text_columns) == 0:
            warnings.warn("No Text column. Skipping lowercasing.")
            return df

        if self.verbose:
            print(f"Lowercasing column(s) '{list(self.text_columns)}'...")

        if self.strategy == 'lc':
            # lowercase all detected text columns
            df[self.text_columns] = df[self.text_columns].apply(lambda col: col.str.lower())
        elif self.strategy == 'none':
            return df

        return df
