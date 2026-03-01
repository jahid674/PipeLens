
import pandas as pd
import warnings
import re

class Tokenizer:
    def __init__(self, dataset, strategy='whitespace', verbose=False): 

        self.dataset = dataset.copy()
        self.text_column = self.dataset.select_dtypes(include=['object', 'string']).columns
        self.verbose = verbose
        self.strategy = strategy.lower()

    def _tokenize_whitespace(self, text):
        return text.split()

    def _tokenize_nltk(self, text):
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except ImportError:
            warnings.warn("NLTK not found. Falling back to whitespace tokenizer.")
            return self._tokenize_whitespace(text)

    def transform(self):
        df = self.dataset.copy()
        if self.text_column is None or len(self.text_column) == 0:
            warnings.warn("No Text column. Skipping tokenization.")
            return df

        if self.strategy == 'whitespace':
            self.df[self.text_column] = self.df[self.text_column].astype(str).apply(self._tokenize_whitespace)
        elif self.strategy == 'nltk':
            self.df[self.text_column] = self.df[self.text_column].astype(str).apply(self._tokenize_nltk)
        elif self.strategy == 'none':
            return df
        else:
            warnings.warn(f"Unsupported tokenization method: {self.strategy}. Skipped.")
        return self.df