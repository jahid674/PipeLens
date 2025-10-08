from modules.language_translator.language_translator import LanguageTranslator

class LanguageTranslatorHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.lr_strategy=config["language_translator_strategy"]
        self.source = config.get('source_lang', 'auto')
        self.target = config.get('target_lang', 'en')

    def apply(self, X, y=None, sensitive=None):
        strat = self.lr_strategy[self.strategy]
        processor = LanguageTranslator(X, strategy=strat, source=self.source, target=self.target)
        X = processor.transform()
        return X, y, sensitive