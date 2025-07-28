from modules.unit_converter.unit_converter import UnitConverter

class UnitConverterHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.uc_strategy = config['unit_converter_strategy']
        #self.column = config['column']
        self.multiplier = config.get('multiplier', 1.0)
        self.offset = config.get('offset', 0.0)

    def apply(self, X, y=None, sensitive=None):
        strat = self.uc_strategy[self.strategy]
        processor = UnitConverter(X, strategy=strat)
        X = processor.transform()
        return X, y, sensitive