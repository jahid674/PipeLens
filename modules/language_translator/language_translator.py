import pandas as pd
import warnings
from deep_translator import GoogleTranslator

class LanguageTranslator:
    def __init__(self, dataset, text_column=None, source='auto', target='en', verbose=False):
        self.dataset = dataset.copy()
        self.text_column = text_column
        self.source = source
        self.target = target
        self.verbose = verbose

    def transform(self):
        df = self.dataset.copy()

        if self.text_column not in df.columns:
            warnings.warn(f"Column '{self.text_column}' not found. Translation skipped.")
            return df

        if not pd.api.types.is_string_dtype(df[self.text_column]):
            warnings.warn(f"Column '{self.text_column}' is not a string type. Translation skipped.")
            return df

        if self.verbose:
            print(f"Translating column '{self.text_column}' from {self.source} to {self.target}...")

        translator = GoogleTranslator(source=self.source, target=self.target)
        df[self.text_column] = df[self.text_column].apply(lambda x: translator.translate(x) if isinstance(x, str) and x.strip() else x)
        return df