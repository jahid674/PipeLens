# =========================
# modules/rag/context_builder/context_builder.py
# - Strategies encoded like: "fixed_chunk_top@1500", "mmr_diverse@3000"
# - Reads from X['reranked'] else X['retrieved']
# - Produces X['context'] (string)
# =========================

import pandas as pd
import warnings
import re
import numpy as np

class ContextBuilder:
    def __init__(self, dataset, strategy="fixed_chunk_top@1500", context_col="context", verbose=False):
        self.dataset = dataset.copy()
        self.strategy = str(strategy).lower()
        self.context_col = context_col
        self.verbose = verbose

    def _parse(self):
        # "mmr_diverse@3000" -> ("mmr_diverse", 3000)
        if "@" in self.strategy:
            name, b = self.strategy.split("@", 1)
            try:
                b = int(b)
            except Exception:
                b = 1500
            return name, b
        return self.strategy, 1500

    def _approx_tokens(self, s: str) -> int:
        # approx tokens by whitespace count (cheap + stable)
        return max(1, len(str(s).split()))

    def transform(self):
        df = self.dataset
        name, budget = self._parse()

        # pick candidate docs
        src_col = None
        for c in ["reranked", "retrieved"]:
            if c in df.columns:
                src_col = c
                break

        if src_col is None:
            warnings.warn("[ContextBuilder] No 'reranked'/'retrieved' found. Creating empty context.")
            df[self.context_col] = ""
            return df

        contexts = []
        for _, row in df.iterrows():
            docs = row.get(src_col, [])
            if not isinstance(docs, list):
                docs = [docs]

            if name == "fixed_chunk_top":
                # take top docs until token budget filled
                parts, used = [], 0
                for d in docs:
                    t = self._approx_tokens(d)
                    if used + t > budget:
                        break
                    parts.append(str(d))
                    used += t
                contexts.append("\n\n".join(parts))

            elif name == "mmr_diverse":
                # stub: diversity = alternate docs (even/odd) until budget
                parts, used = [], 0
                order = []
                evens = list(range(0, len(docs), 2))
                odds = list(range(1, len(docs), 2))
                order = evens + odds

                for i in order:
                    d = docs[i]
                    t = self._approx_tokens(d)
                    if used + t > budget:
                        break
                    parts.append(str(d))
                    used += t
                contexts.append("\n\n".join(parts))
            else:
                warnings.warn(f"[ContextBuilder] Unknown strategy '{self.strategy}'. Using empty context.")
                contexts.append("")

        df[self.context_col] = contexts
        return df
