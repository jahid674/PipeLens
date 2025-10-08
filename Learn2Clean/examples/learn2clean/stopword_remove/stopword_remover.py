import warnings
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
pd.options.mode.chained_assignment = None


class StopwordRemover():
    """
    Remove stopwords from text columns using NLTK.

    Parameters
    ----------
    * strategy : str, default = 'NONE'
      Available strategies:
        - 'NONE' : do nothing (no stopword removal)
        - 'SW'   : remove stopwords with NLTK

    * text_column : str or list[str] or None
      Columns to apply stopword removal on. If None, auto-detects object/string columns.

    * exclude : str or None
      Column to safeguard from processing (restored afterward if present).

    * verbose : bool, default=False
      Print information about processing.

    * threshold : float or None
      Unused; kept for compatibility with Learn2Clean components.
    """

    def __init__(self, dataset, strategy='NONE_stopword', text_column=None,
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

        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

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
                warnings.warn("Invalid parameter(s) for StopwordRemover. "
                              "Parameter(s) IGNORED. "
                              "Check with `stopword_remover.get_params().keys()`")
            else:
                setattr(self, k, v)

    # === Strategy: do nothing ===
    def NONE(self, dataset):
        if self.verbose:
            print("No stopword removal applied (strategy='NONE').")
        return dataset.sort_index()

    # === Strategy: stopword removal ===
    def SW(self, dataset):
        d = dataset
        if not self.text_column or len(self.text_column) == 0:
            warnings.warn("No text columns found. Skipping stopword removal.")
            return d

        if self.verbose:
            print(f"Removing stopwords from columns {self.text_column}...")

        for col in self.text_column:
            if pd.api.types.is_string_dtype(d[col]):
                d[col] = d[col].apply(
                    lambda x: ' '.join(
                        [word for word in str(x).split() if word.lower() not in self.stop_words]
                    )
                )

        if (self.exclude in list(d.columns.values)):
            d[self.exclude] = dataset[self.exclude]

        return d.sort_index()

    def transform(self):
        outd = self.dataset
        start_time = time.time()
        print(">>Stopword Removal ")

        for key in ['train']:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                print("* For", key, "dataset")

                if self.strategy == "NONE_STOPWORD":
                    dn = self.NONE(d)
                elif self.strategy == "SW":
                    dn = self.SW(d)
                else:
                    raise ValueError("Unknown strategy. Choose 'NONE_stopword' or 'SW'.")

                if (self.exclude in list(pd.DataFrame(d).columns.values)):
                    dn[self.exclude] = d[self.exclude]

                outd[key] = dn
                print('...', key, 'dataset')
            else:
                outd[key] = self.dataset[key]
                print('No', key, 'dataset, no stopword removal')

        print("Stopword removal done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()
        return outd
