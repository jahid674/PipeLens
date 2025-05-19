from modules.normalization.normalizer import Normalizer

class NormalizationHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.norm_strategy = config['norm_strategy']

    def apply(self, X, y, sensitive):
        strat = self.norm_strategy[self.strategy]
        normalizer = Normalizer(X, strategy=strat, verbose=False)
        X_new = normalizer.transform()
        return X_new, y, sensitive
