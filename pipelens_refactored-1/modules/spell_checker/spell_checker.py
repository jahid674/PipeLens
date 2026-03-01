import pandas as pd
import warnings
from textblob import TextBlob

class SpellChecker:
    def __init__(self, dataset, strategy='sc',verbose=False):
        self.dataset = dataset.copy()
        self.text_column = self.dataset.select_dtypes(include=['object', 'string']).columns
        self.verbose = verbose
        self.strategy = strategy.lower()

    def _identify_text_columns(self):
        text_candidates = self.dataset.select_dtypes(include=['object', 'string']).columns
        return list(text_candidates)

    def _correct_spelling(self, text):
        try:
            if pd.isnull(text):
                return text
            return str(TextBlob(text).correct())
        except Exception as e:
            warnings.warn(f"Spell check failed for entry: {text}. Error: {e}")
            return text

    def transform(self):
        if self.verbose:
            print("---------->> Started Spell Checking <<-----------")

        df = self.dataset.copy()
        if self.strategy == 'sc':
            for col in self.text_column:
                if self.verbose:
                    print(f"Spell checking column: {col}")
                df[col] = df[col].astype(str).apply(self._correct_spelling)
        elif self.strategy == 'none':
            if self.verbose:
                print("No spell checking applied (strategy='NONE').")
            return df

        if self.verbose:
            print("---------->> Spell Checking Completed <<-----------")

        return df
