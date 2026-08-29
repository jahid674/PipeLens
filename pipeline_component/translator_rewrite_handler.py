# =========================
# pipeline_component/translator_rewriter_handler.py
# =========================

from modules.rag.translator_rewrite.translator_rewrite import TranslatorRewriter

class TranslatorRewriteHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.translator_rewriter_strategy = config["translator_rewriter_strategy"]
        self.query_col = config.get("rag_query_col", "query")

    def apply(self, X, y=None, sensitive=None):
        strat = self.translator_rewriter_strategy[self.strategy]
        processor = TranslatorRewriter(X, strategy=strat, query_col=self.query_col)
        X = processor.transform()
        return X, y, sensitive
