# =========================
# modules/rag/translator_rewriter/translator_rewriter.py
# - Strategy string format:
#   "mt_generic|fr->en|literal"
#   "llm_translate|dual(fr+en)|search_optimized"
# - Skips cleanly if no query column
# =========================

import pandas as pd
import warnings

class TranslatorRewriter:
    def __init__(self, dataset, strategy="mt_generic|fr->en|literal", query_col="query", verbose=False):
        self.dataset = dataset.copy()
        self.strategy = str(strategy).lower()
        self.query_col = query_col
        self.verbose = verbose

    def _parse(self):
        # translator|mode|rewrite_style
        parts = self.strategy.split("|")
        while len(parts) < 3:
            parts.append("literal")
        translator, mode, rewrite_style = parts[0], parts[1], parts[2]
        return translator, mode, rewrite_style

    def transform(self):
        df = self.dataset

        if self.query_col not in df.columns:
            warnings.warn(f"[TranslatorRewriter] Missing '{self.query_col}'. Skipping.")
            return df

        if not pd.api.types.is_string_dtype(df[self.query_col]):
            warnings.warn(f"[TranslatorRewriter] '{self.query_col}' is not string dtype. Skipping.")
            return df

        translator, mode, rewrite_style = self._parse()

        # NOTE: dependency-free placeholder:
        # In real system you call MT/LLM here.
        def _fake_translate_fr_to_en(q):
            return f"[EN_TRANSLATED]{q}"

        def _fake_search_opt(q):
            # light rewrite: add keyword-style hint
            return f"{q} (keywords)"

        q = df[self.query_col].fillna("").astype(str)

        if mode in ("fr->en", "fr→en"):
            q_en = q.apply(_fake_translate_fr_to_en)
            if rewrite_style == "search_optimized":
                q_en = q_en.apply(_fake_search_opt)
            df[self.query_col] = q_en

        elif mode.startswith("dual"):
            q_en = q.apply(_fake_translate_fr_to_en)
            if rewrite_style == "search_optimized":
                q_en = q_en.apply(_fake_search_opt)
            # dual: keep both
            df[self.query_col] = (q.astype(str) + " || " + q_en.astype(str))

        else:
            warnings.warn(f"[TranslatorRewriter] Unknown mode '{mode}'. Skipping.")
            return df

        return df
