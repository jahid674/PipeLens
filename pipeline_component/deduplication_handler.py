from modules.data_preparation.deduplication.deduplication import Deduplicator


class DeduplicationHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.subset = None  # If None, drop duplicates on all columns
        self.dd_strategy = config['deduplication_strategy']


    def apply(self, X, y=None, sensitive=None):
        strat = self.dd_strategy[self.strategy]
        processor = Deduplicator(X, strategy=strat)
        X, y, sensitive = processor.transform(y, sensitive)
        return X, y, sensitive

