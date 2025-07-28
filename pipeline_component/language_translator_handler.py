from modules.language_translator.language_translator import LanguageTranslator

class LanguageTranslatorHandler:
    def __init__(self, config):
        self.text_column = config['text_column']
        self.source = config.get('source_lang', 'auto')
        self.target = config.get('target_lang', 'en')

    def apply(self, X, y=None, sensitive=None):
        processor = LanguageTranslator(X, text_column=self.text_column, source=self.source, target=self.target)
        X = processor.transform()
        return X, y, sensitive