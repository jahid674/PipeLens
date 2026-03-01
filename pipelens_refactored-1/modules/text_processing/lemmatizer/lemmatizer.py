# modules/text/lemmatizer.py

import pandas as pd
import numpy as np
import time

try:
    import spacy
except ImportError:
    spacy = None

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Lemmatizer:
    """
    Text lemmatization module.

    Strategies:
      - "wordnet" : NLTK WordNet lemmatizer (lightweight)
      - "spacy"   : spaCy lemmatizer (POS-aware, higher quality)

    Behavior:
      - Automatically detects text columns (object / string dtype)
      - Applies lemmatization column-wise
      - Preserves non-text columns
      - Fits nothing (stateless transform)
    """

    def __init__(
        self,
        dataset,
        strategy="wordnet",
        language="en",
        verbose=False,
        exclude=None,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.strategy = strategy.lower()
        self.language = language
        self.verbose = verbose
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

        if self.strategy == "spacy":
            if spacy is None:
                raise ImportError("spaCy not installed. Run `pip install spacy`.")
            try:
                self._nlp = spacy.load(f"{language}_core_web_sm", disable=["ner", "parser"])
            except Exception:
                raise RuntimeError(
                    f"spaCy model '{language}_core_web_sm' not found. "
                    f"Run: python -m spacy download {language}_core_web_sm"
                )
        else:
            self._lemmatizer = WordNetLemmatizer()

    def _detect_text_columns(self, df: pd.DataFrame):
        return df.select_dtypes(include=["object", "string"]).columns.tolist()

    def _lemmatize_wordnet(self, text: str):
        tokens = word_tokenize(text)
        return " ".join(self._lemmatizer.lemmatize(tok) for tok in tokens)

    def _lemmatize_spacy(self, text: str):
        doc = self._nlp(text)
        return " ".join(tok.lemma_ for tok in doc)

    def _transform_df(self, df: pd.DataFrame):
        df_out = df.copy()

        text_cols = self._detect_text_columns(df_out)
        text_cols = [c for c in text_cols if c not in self.exclude]

        for col in text_cols:
            series = df_out[col].fillna("").astype(str)

            if self.strategy == "spacy":
                df_out[col] = series.apply(self._lemmatize_spacy)
            else:
                df_out[col] = series.apply(self._lemmatize_wordnet)

        return df_out

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Lemmatization -----")

        if isinstance(self.dataset, dict):
            out = {"train": self._transform_df(self.dataset["train"].copy())}
            if "test" in self.dataset and self.dataset["test"] is not None:
                out["test"] = self._transform_df(self.dataset["test"].copy())

        elif isinstance(self.dataset, pd.DataFrame):
            out = self._transform_df(self.dataset.copy())

        else:
            raise TypeError("dataset must be a DataFrame or dict {'train','test'}")

        if self.verbose:
            print(f"Lemmatization completed in {time.time() - start_time:.2f}s")

        return out
