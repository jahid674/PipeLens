# =========================
# modules/rag/retriever/hybrid_retriever.py
# - Minimal, dependency-free retrieval stub
# - Strategies encoded like: "bm25@10", "dense@20", "hybrid_rrf@10"
# - Produces X['retrieved'] (list[str]) and X['retrieval_scores'] (list[float])
# - If no corpus column exists, creates empty retrieval and skips safely
# =========================

import pandas as pd
import numpy as np
import warnings
import re

class HybridRetriever:
    def __init__(self, dataset, strategy="bm25@10", query_col="query", verbose=False):
        self.dataset = dataset.copy()
        self.strategy = str(strategy).lower()
        self.query_col = query_col
        self.verbose = verbose

    def _parse(self):
        # "bm25@10" -> ("bm25", 10)
        if "@" in self.strategy:
            name, k = self.strategy.split("@", 1)
            try:
                k = int(k)
            except Exception:
                k = 10
            return name, k
        return self.strategy, 10

    def _tokenize(self, s: str):
        return re.findall(r"\w+", str(s).lower())

    def _score_bm25ish(self, q_tokens, doc_text):
        # very rough scoring: token overlap count
        d_tokens = set(self._tokenize(doc_text))
        return float(sum(1 for t in q_tokens if t in d_tokens))

    def transform(self):
        df = self.dataset

        if self.query_col not in df.columns:
            warnings.warn(f"[Retriever] Missing query column '{self.query_col}'. Skipping.")
            df["retrieved"] = [[] for _ in range(len(df))]
            df["retrieval_scores"] = [[] for _ in range(len(df))]
            return df

        # Identify a corpus column (pick the first that exists)
        corpus_col = None
        for c in ["docs", "doc", "passage", "text", "document"]:
            if c in df.columns:
                corpus_col = c
                break

        if corpus_col is None:
            warnings.warn("[Retriever] No corpus column found (docs/doc/text). Producing empty retrieval.")
            df["retrieved"] = [[] for _ in range(len(df))]
            df["retrieval_scores"] = [[] for _ in range(len(df))]
            return df

        name, topk = self._parse()

        if self.verbose:
            print(f"[Retriever] strategy='{name}' topk={topk} corpus_col='{corpus_col}'")

        retrieved_all = []
        scores_all = []

        for _, row in df.iterrows():
            q = row.get(self.query_col, "")
            q_tokens = self._tokenize(q)

            docs = row.get(corpus_col)
            # allow docs to be a list; if string, treat as single-doc corpus
            if isinstance(docs, list):
                doc_list = docs
            else:
                doc_list = [docs]

            # scoring
            scores = []
            for d in doc_list:
                if name in ("bm25", "hybrid_rrf"):
                    s = self._score_bm25ish(q_tokens, d)
                elif name in ("dense",):
                    # fake dense score: cosine-ish by char overlap
                    s = float(len(set(str(q)) & set(str(d))))
                else:
                    s = self._score_bm25ish(q_tokens, d)
                scores.append(s)

            # topk selection
            idx = np.argsort(scores)[::-1][:topk] if len(scores) > 0 else []
            retrieved = [doc_list[i] for i in idx] if len(idx) > 0 else []
            sel_scores = [float(scores[i]) for i in idx] if len(idx) > 0 else []

            # hybrid_rrf: re-score via simple reciprocal rank fusion (still stub)
            if name == "hybrid_rrf" and len(sel_scores) > 0:
                rrf = []
                for rank, _ in enumerate(sel_scores, start=1):
                    rrf.append(1.0 / (60.0 + rank))
                sel_scores = rrf

            retrieved_all.append(retrieved)
            scores_all.append(sel_scores)

        df["retrieved"] = retrieved_all
        df["retrieval_scores"] = scores_all
        return df
