# =========================
# pipeline_component/query_normalize_handler.py
# =========================

from modules.rag.query_normalizer.query_normalizer import QueryNormalizer

class QueryNormalizeHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.query_normalize_strategy = config["query_normalize_strategy"]
        self.query_col = config.get("rag_query_col", "query")

    def apply(self, X, y=None, sensitive=None):
        strat = self.query_normalize_strategy[self.strategy]  # 0-based index
        processor = QueryNormalizer(X, strategy=strat, query_col=self.query_col)
        X = processor.transform()
        return X, y, sensitive
