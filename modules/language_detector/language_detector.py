import pandas as pd
import warnings
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # For consistent results

class LanguageDetector:
    def __init__(self, dataset, text_column=None, result_column='language', verbose=False):
        self.dataset = dataset.copy()
        self.text_column = text_column
        self.result_column = result_column
        self.verbose = verbose

    def transform(self):
        df = self.dataset.copy()

        if self.text_column not in df.columns:
            warnings.warn(f"Column '{self.text_column}' not found. Language detection skipped.")
            return df

        if not pd.api.types.is_string_dtype(df[self.text_column]):
            warnings.warn(f"Column '{self.text_column}' is not a string type. Language detection skipped.")
            return df

        if self.verbose:
            print(f"Detecting language for column '{self.text_column}'...")

        df[self.result_column] = df[self.text_column].apply(lambda x: detect(x) if isinstance(x, str) and x.strip() else 'unknown')
        return df