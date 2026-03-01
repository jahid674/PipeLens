import warnings
import time
import pandas as pd
from deep_translator import GoogleTranslator
pd.options.mode.chained_assignment = None


class LanguageTranslator():
    """
    Translate text in a dataset column using GoogleTranslator.

    Parameters
    ----------
    * strategy : str, default = 'NONE'
      Available strategies:
        - 'NONE' : do nothing (no translation)
        - 'GT'   : translate using GoogleTranslator

    * text_column : str or None, default=None
      Column to apply translation on.

    * source : str, default='auto'
      Source language ('auto' to auto-detect).

    * target : str, default='en'
      Target language code (e.g., 'en').

    * exclude : str or None
      Column to safeguard from processing (restored afterward if present).

    * verbose : bool, default=False
      Print information about translation.

    * threshold : float or None
      Unused; kept for compatibility with Learn2Clean components.
    """

    def __init__(self, dataset, strategy='NONE_translate', text_column=None,
                 source='auto', target='en',
                 exclude=None, verbose=False, threshold=None):

        self.dataset = dataset
        self.strategy = strategy.upper()
        self.text_column = text_column
        self.source = source
        self.target = target
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
            'source': self.source,
            'target': self.target,
            'exclude': self.exclude,
            'verbose': self.verbose,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for LanguageTranslator. "
                              "Parameter(s) IGNORED. Check with "
                              "`translator.get_params().keys()`")
            else:
                setattr(self, k, v)

    # === Strategy: do nothing ===
    def NONE(self, dataset):
        if self.verbose:
            print("No translation applied (strategy='NONE').")
        return dataset.sort_index()

    # === Strategy: Google Translator ===
    def GT(self, dataset):
        d = dataset

        if not pd.api.types.is_string_dtype(d[self.text_column]):
            warnings.warn(f"Column '{self.text_column}' is not string type. Translation skipped.")
            return d

        if self.verbose:
            print(f"Translating column '{self.text_column}' from {self.source} to {self.target}...")

        translator = GoogleTranslator(source=self.source, target=self.target)
        d[self.text_column] = d[self.text_column].apply(
            lambda x: translator.translate(x) if isinstance(x, str) and x.strip() else x
        )

        if (self.exclude in list(d.columns.values)):
            d[self.exclude] = dataset[self.exclude]

        return d.sort_index()

    def transform(self):
        outd = self.dataset
        start_time = time.time()
        print(">>Translation ")

        for key in ['train']:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                print("* For", key, "dataset")

                if self.strategy == "NONE_TRANSLATE":
                    dn = self.NONE(d)
                elif self.strategy == "GT":
                    dn = self.GT(d)
                else:
                    raise ValueError("Unknown strategy. Choose 'NONE_translate' or 'GT'.")

                if (self.exclude in list(pd.DataFrame(d).columns.values)):
                    dn[self.exclude] = d[self.exclude]

                outd[key] = dn
                print('...', key, 'dataset')
            else:
                outd[key] = self.dataset[key]
                print('No', key, 'dataset, no translation')

        print("Translation done -- CPU time: %s seconds" % (time.time() - start_time))
        print()
        return outd
