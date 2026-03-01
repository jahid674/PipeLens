from modules.stopword_remover.stopword_remover import StopwordRemover

class StopwordHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.sw_strategy = config['stopword_strategy']

    def apply(self, X, y=None, sensitive=None):
        strat = self.sw_strategy[self.strategy]
        processor = StopwordRemover(X, strategy=strat)
        X = processor.transform()
        return X, y, sensitive