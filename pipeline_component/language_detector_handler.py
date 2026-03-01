from modules.text_processing.language_detector.language_detector import LanguageDetector

class LanguageDetectorHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.ld_strategy=config["language_detector_strategy"]
        self.result_column = config.get('result_column', 'language')

    def apply(self, X, y=None, sensitive=None):
        strat = self.lr_strategy[self.strategy]
        processor = LanguageDetector(X, strategy=strat, text_column=self.text_column, result_column=self.result_column)
        X = processor.transform()
        return X, y, sensitive