# =========================
# pipeline_component/lowercase_handler.py
# Updated to mirror WhitespaceHandler:
# - Accepts (strategy, config)
# - Pulls strategy list from config['lowercase_strategy']
# - Does NOT require text_column in config anymore
# =========================

from modules.text_processing.lower_case.lower_caser import Lowercaser

class LowercaseHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.lowercase_strategy = config['lowercase_strategy']  # e.g., ['none','lc']

    def apply(self, X, y=None, sensitive=None):
        strat = self.lowercase_strategy[self.strategy]  # 0-based index
        processor = Lowercaser(X, strategy=strat)
        X = processor.transform()
        return X, y, sensitive
