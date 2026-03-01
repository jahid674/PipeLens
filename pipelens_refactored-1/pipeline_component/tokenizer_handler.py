# File: components/tokenizer_handler.py

from modules.tokenization.tokenizer import Tokenizer

class TokenizerHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.tk_strategy = config['tokenization_strategy']  # can be 'whitespace' or 'nltk'

    def apply(self, X, y=None, sensitive=None):
        strat = self.tk_strategy[self.strategy]
        processor = Tokenizer(X, strategy=strat)
        X = processor.transform()
        return X, y, sensitive
