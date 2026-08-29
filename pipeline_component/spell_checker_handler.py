from modules.text_processing.spell_checker.spell_checker import SpellChecker

class SpellCheckerHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.sc_strategy = config['spellchecker_strategy']

    def apply(self, X, y=None, sensitive=None):
        strat = self.sc_strategy[self.strategy]
        processor = SpellChecker(X, strategy=strat)
        X_new = processor.transform()
        return X_new, y, sensitive
