from modules.text_processing.date_separator.date_separator import DateSeparatorReplacer

class DateSeparatorHandler:
    def __init__(self, config):
        self.text_column = config['text_column']
        self.from_sep = config.get('from_sep', '-')
        self.to_sep = config.get('to_sep', '/')

    def apply(self, X, y=None, sensitive=None):
        processor = DateSeparatorReplacer(
            X, text_column=self.text_column, from_sep=self.from_sep, to_sep=self.to_sep
        )
        X = processor.transform()
        return X, y, sensitive