import warnings
import time
import pandas as pd
from textblob import TextBlob
pd.options.mode.chained_assignment = None


class SpellChecker():
    """
    Spell-check and correct text in dataset columns using TextBlob.

    Parameters
    ----------
    * strategy : str, default = 'NONE'
      Available strategies:
        - 'NONE' : do nothing (no spell checking)
        - 'SC'   : correct spelling using TextBlob

    * text_column : str or list[str] or None
      Columns to apply spell checking on. If None, auto-detects object/string columns.

    * exclude : str or None
      Column to safeguard from processing (restored afterward if present).

    * verbose : bool, default=False
      Print information about spell checking.

    * threshold : float or None
      Unused; for compatibility with Learn2Clean components.
    """

    def __init__(self, dataset, strategy='NONE_spell', text_column=None,
                 exclude=None, verbose=False, threshold=None):

        self.dataset = dataset
        self.strategy = strategy.upper()
        self.text_column = text_column
        self.exclude = exclude
        self.verbose = verbose
        self.threshold = threshold

        if isinstance(self.text_column, str):
            self.text_column = [self.text_column]

        # Auto-detect text columns if not provided
        try:
            # Prefer train split if dataset is a dict
            df = dataset['train'] if isinstance(dataset, dict) else dataset
            if self.text_column is None and isinstance(df, pd.DataFrame):
                self.text_column = df.select_dtypes(include=['object', 'string']).columns.tolist()
        except Exception:
            # If anything goes wrong, leave text_column as-is (can be handled later)
            pass

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'text_column': self.text_column,
            'exclude': self.exclude,
            'verbose': self.verbose,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for SpellChecker. "
                              "Parameter(s) IGNORED. "
                              "Check with `spellchecker.get_params().keys()`")
            else:
                setattr(self, k, v)

    # === Strategy: do nothing ===
    def NONE(self, dataset):
        if self.verbose:
            print("No spell checking applied (strategy='NONE').")
        return dataset.sort_index()

    # === Strategy: spell checking with TextBlob ===
    def SC(self, dataset):
        d = dataset
        if not self.text_column or len(self.text_column) == 0:
            warnings.warn("No text columns found. Skipping spell checking.")
            return d

        if self.verbose:
            print("---------->> Started Spell Checking <<-----------")

        for col in self.text_column:
            if pd.api.types.is_string_dtype(d[col]):
                if self.verbose:
                    print(f"Spell checking column: {col}")
                d[col] = d[col].astype(str).apply(self._correct_spelling)

        if self.verbose:
            print("---------->> Spell Checking Completed <<-----------")

        if (self.exclude in list(d.columns.values)):
            d[self.exclude] = dataset[self.exclude]

        return d.sort_index()

    def _correct_spelling(self, text):
        try:
            if pd.isnull(text):
                return text
            return str(TextBlob(text).correct())
        except Exception as e:
            warnings.warn(f"Spell check failed for entry: {text}. Error: {e}")
            return text

    def transform(self):
        outd = self.dataset
        start_time = time.time()
        print(">>Spell Checking ")

        for key in ['train']:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                print("* For", key, "dataset")

                if self.strategy == "NONE_SPELL":
                    dn = self.NONE(d)
                elif self.strategy == "SC":
                    dn = self.SC(d)
                else:
                    raise ValueError("Unknown strategy. Choose 'NONE_spell' or 'SC'.")

                if (self.exclude in list(pd.DataFrame(d).columns.values)):
                    dn[self.exclude] = d[self.exclude]

                outd[key] = dn
                print('...', key, 'dataset')
            else:
                outd[key] = self.dataset[key]
                print('No', key, 'dataset, no spell checking')

        print("Spell checking done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()
        return outd
