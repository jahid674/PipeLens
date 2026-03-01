from modules.special_char_remove.special_character_remover import SpecialCharRemover

class SpecialCharacterHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.sc_strategy = config['specialchar_strategy']

    def apply(self, X, y=None, sensitive=None):
        strat = self.sc_strategy[self.strategy]
        processor = SpecialCharRemover(X, strategy=strat)
        X = processor.transform()
        return X, y, sensitive
