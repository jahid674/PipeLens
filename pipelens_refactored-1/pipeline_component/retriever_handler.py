# =========================
# pipeline_component/retriever_handler.py
# =========================

from modules.rag.retriever.hybrid_retriever import HybridRetriever

class RetrieverHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.retriever_strategy = config["retriever_strategy"]
        self.query_col = config.get("rag_query_col", "query")

    def apply(self, X, y=None, sensitive=None):
        strat = self.retriever_strategy[self.strategy]  # 0-based index
        processor = HybridRetriever(X, strategy=strat, query_col=self.query_col)
        X = processor.transform()
        return X, y, sensitive
