from modules.whitespace_cleaning.whitespace_cleaner import WhitespaceCleaner

class WhitespaceHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.wc_strategy = config['whitespace_strategy']

    def apply(self, X, y=None, sensitive=None):
        strat = self.wc_strategy[self.strategy]
        processor = WhitespaceCleaner(X, strategy=strat)
        X = processor.transform()
        return X, y, sensitive