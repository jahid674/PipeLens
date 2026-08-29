# =========================
# pipeline_component/generator_handler.py
# - This is the final step: returns utility(float) to match your PipelineExecutor expectations
# - Also implements get_profile_metric(...) to prevent run_pipeline_glass from breaking
# =========================

from modules.rag.generator.generator import Generator
import numpy as np
import re

class GeneratorHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.generator_strategy = config["generator_strategy"]
        self.query_col = config.get("rag_query_col", "query")
        self.context_col = config.get("rag_context_col", "context")
        self._last_utility = None

    def _utility_stub(self, X):
        """
        Utility proxy (dependency-free):
        - Higher if query tokens overlap with context tokens
        - Penalize empty context
        """
        if self.query_col not in X.columns or self.context_col not in X.columns:
            return 0.0

        qs = X[self.query_col].fillna("").astype(str).tolist()
        cs = X[self.context_col].fillna("").astype(str).tolist()

        scores = []
        for q, c in zip(qs, cs):
            if len(c.strip()) == 0:
                scores.append(0.0)
                continue
            qtok = set(re.findall(r"\w+", q.lower()))
            ct = set(re.findall(r"\w+", c.lower()))
            if len(qtok) == 0:
                scores.append(0.0)
                continue
            overlap = len(qtok & ct) / (len(qtok) + 1e-12)
            scores.append(float(overlap))
        return float(np.mean(scores)) if len(scores) else 0.0

    def apply(self, X, y=None, sensitive=None):
        strat = self.generator_strategy[self.strategy]
        processor = Generator(X, strategy=strat, query_col=self.query_col, context_col=self.context_col)
        X = processor.transform()

        # return float utility as the pipeline output
        self._last_utility = self._utility_stub(X)
        return float(self._last_utility)

    def get_profile_metric(self, y, sens):
        # Keep compatible with your run_pipeline_glass() expectation
        headers = []
        sens_data = []
        return headers, sens_data
