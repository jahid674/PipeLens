from modules.punctuation_remover.punctuation_remover import PunctuationRemover

class PunctuationHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.pr_strategy = config['punctuation_strategy']

    def apply(self, X, y=None, sensitive=None):
        strat = self.pr_strategy[self.strategy]
        processor = PunctuationRemover(X, strategy=strat)
        X = processor.transform()
        return X, y, sensitive