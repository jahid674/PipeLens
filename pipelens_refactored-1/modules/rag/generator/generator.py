# =========================
# modules/rag/generator/generator.py
# - Strategies: ["grounded_concise", "grounded_detailed"]
# - constraint: "context-only grounding"
# - Produces X['answer'] (string)
# =========================

import pandas as pd
import warnings
import re

class Generator:
    def __init__(self, dataset, strategy="grounded_concise", query_col="query", context_col="context", verbose=False):
        self.dataset = dataset.copy()
        self.strategy = str(strategy).lower()
        self.query_col = query_col
        self.context_col = context_col
        self.verbose = verbose

    def _extract_keywords(self, q: str):
        return re.findall(r"\w+", str(q).lower())[:8]

    def transform(self):
        df = self.dataset

        if self.query_col not in df.columns or self.context_col not in df.columns:
            warnings.warn("[Generator] Missing query/context. Skipping generation.")
            df["answer"] = ""
            return df

        answers = []
        for _, row in df.iterrows():
            q = str(row.get(self.query_col, ""))
            ctx = str(row.get(self.context_col, ""))

            # context-only grounding (no external knowledge)
            if len(ctx.strip()) == 0:
                answers.append("I don't have enough context to answer from the provided documents.")
                continue

            kw = self._extract_keywords(q)

            if self.strategy == "grounded_concise":
                # concise: quote-like snippet behavior (but no verbatim long quote)
                snippet = " ".join(ctx.split()[:40])
                answers.append(f"Based on the provided context, key terms {kw}: {snippet} ...")

            elif self.strategy == "grounded_detailed":
                snippet = " ".join(ctx.split()[:120])
                answers.append(f"Using only the retrieved context, here is a grounded answer.\n\nKey terms: {kw}\n\nContext summary: {snippet} ...")

            else:
                warnings.warn(f"[Generator] Unknown strategy '{self.strategy}'. Returning empty answer.")
                answers.append("")

        df["answer"] = answers
        return df
