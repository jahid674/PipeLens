from modules.data_preparation.binning.binning import Binner

class BinnerHandler:
    def __init__(self, config):
        self.column = config['column']
        self.strategy = config.get('strategy', 'uniform')
        self.n_bins = config.get('n_bins', 5)
        self.encode = config.get('encode', 'ordinal')
        self.verbose = config.get('verbose', False)

    def apply(self, X, y=None, sensitive=None):
        processor = Binner(
            dataset=X,
            column=self.column,
            strategy=self.strategy,
            n_bins=self.n_bins,
            encode=self.encode,
            verbose=self.verbose
        )
        X_new = processor.transform()
        return X_new, y, sensitive
