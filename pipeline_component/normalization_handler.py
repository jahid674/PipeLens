from modules.normalization.normalizer import Normalizer
from modules.profiling.profile import Profile

class NormalizationHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.norm_strategy = config['norm_strategy']
        self.p = Profile()
    def apply(self, X, y, sensitive):
        self.outlier_before_norm_start=self.p.get_fraction_of_outlier(X)
        strat = self.norm_strategy[self.strategy]
        normalizer = Normalizer(X, strategy=strat, verbose=False)
        X_new = normalizer.transform()
        return X_new, y, sensitive
    
    def get_outlier_bef_normalization_strat(self):
        return self.outlier_before_norm_start
