# =========================
# pipeline_component/reranker_handler.py
# =========================

from modules.rag.reranker.reranker import Reranker

class RerankerHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.reranker_strategy = config["reranker_strategy"]
        self.query_col = config.get("rag_query_col", "query")

    def apply(self, X, y=None, sensitive=None):
        strat = self.reranker_strategy[self.strategy]
        processor = Reranker(X, strategy=strat, query_col=self.query_col)
        X = processor.transform()
        return X, y, sensitive
