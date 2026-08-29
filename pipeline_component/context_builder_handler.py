# =========================
# pipeline_component/context_builder_handler.py
# =========================

from modules.rag.context_builder.context_builder import ContextBuilder

class ContextBuilderHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.context_builder_strategy = config["context_builder_strategy"]
        self.context_col = config.get("rag_context_col", "context")

    def apply(self, X, y=None, sensitive=None):
        strat = self.context_builder_strategy[self.strategy]
        processor = ContextBuilder(X, strategy=strat, context_col=self.context_col)
        X = processor.transform()
        return X, y, sensitive
