import pandas as pd
import warnings

class QuoteRemover:
    def __init__(self, dataset, text_column=None, verbose=False):
        self.dataset = dataset.copy()
        self.text_column = text_column
        self.verbose = verbose

    def transform(self):
        df = self.dataset.copy()
        if self.text_column not in df.columns:
            warnings.warn(f"Column '{self.text_column}' not found. Quote removal skipped.")
            return df

        if self.verbose:
            print(f"Removing quotes from column '{self.text_column}'...")

        df[self.text_column] = df[self.text_column].str.replace('"', '').str.replace("'", '')
        return df


# File: modules/text_processing/date_separator_replacer.py
import pandas as pd
import warnings
from dateutil.parser import parse

class DateSeparatorReplacer:
    def __init__(self, dataset, text_column=None, from_sep='-', to_sep='/', verbose=False):
        self.dataset = dataset.copy()
        self.text_column = text_column
        self.from_sep = from_sep
        self.to_sep = to_sep
        self.verbose = verbose

    def is_date(self, string):
        try:
            parse(string)
            return True
        except (ValueError, TypeError):
            return False

    def transform(self):
        df = self.dataset.copy()
        if self.text_column not in df.columns:
            warnings.warn(f"Column '{self.text_column}' not found. Date separator replacement skipped.")
            return df

        if self.verbose:
            print(f"Replacing date separator '{self.from_sep}' with '{self.to_sep}' in column '{self.text_column}'...")

        df[self.text_column] = df[self.text_column].apply(
            lambda x: x.replace(self.from_sep, self.to_sep) if isinstance(x, str) and self.is_date(x) else x
        )
        return df