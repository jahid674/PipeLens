import pandas as pd
import nltk
from nltk.corpus import stopwords
import warnings

class StopwordRemover:
    def __init__(self, dataset, strategy='sw', verbose=False):
        self.dataset = dataset.copy()
        self.text_column = self.dataset.select_dtypes(include=['object', 'string']).columns
        self.verbose = verbose
        self.strategy = strategy.lower()

        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def transform(self):
        df = self.dataset.copy()
        if self.text_column is None or len(self.text_column) == 0:
            warnings.warn("No Text column. Skipping punctuation removal.")
            return df
        
        if self.strategy == 'sw':
            df[self.text_column] = df[self.text_column].apply(
            lambda x: ' '.join([word for word in str(x).split() if word.lower() not in self.stop_words]))
        elif self.strategy == 'none':
            return df
        
        return df