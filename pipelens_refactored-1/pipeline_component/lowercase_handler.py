from modules.lower_case.lower_caser import Lowercaser

class LowercaserHandler:
    def __init__(self, config):
        self.text_column = config['text_column']

    def apply(self, X, y=None, sensitive=None):
        processor = Lowercaser(X, text_column=self.text_column)
        X = processor.transform()
        return X, y, sensitive