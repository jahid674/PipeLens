from modules.language_detector.language_detector import LanguageDetector

class LanguageDetectorHandler:
    def __init__(self, config):
        self.text_column = config['text_column']
        self.result_column = config.get('result_column', 'language')

    def apply(self, X, y=None, sensitive=None):
        processor = LanguageDetector(X, text_column=self.text_column, result_column=self.result_column)
        X = processor.transform()
        return X, y, sensitive