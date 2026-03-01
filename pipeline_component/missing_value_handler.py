from modules.data_preparation.missing_value.imputer import Imputer

class MissingValueHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.mv_strategy = config['mv_strategy']
        self.knn_k_list = config['knn_k_lst']

    def apply(self, X, y, sensitive):
        if self.strategy < len(self.mv_strategy) - 1:
            strat = self.mv_strategy[self.strategy]
            imputer = Imputer(X, strategy=strat, verbose=False)
            if strat == 'drop':
                return imputer.transform(y, sensitive)
            else:
                return imputer.transform(y, sensitive), y, sensitive
        else:
            k = self.knn_k_list[self.strategy - (len(self.mv_strategy) - 1)]
            imputer = Imputer(X, strategy='knn', k=k, verbose=False)
            return imputer.transform(y, sensitive), y, sensitive
