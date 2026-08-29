# =========================
# pipeline_component/lang_retrieval_wrapper_handler.py
# =========================

from modules.rag.language_wrapper.language_wrapper import LangRetrievalWrapper

class LanguagelWrapperHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.lang_retrieval_wrapper_strategy = config["lang_retrieval_wrapper_strategy"]

    def apply(self, X, y=None, sensitive=None):
        strat = self.lang_retrieval_wrapper_strategy[self.strategy]
        processor = LangRetrievalWrapper(X, strategy=strat)
        X = processor.transform()
        return X, y, sensitive
