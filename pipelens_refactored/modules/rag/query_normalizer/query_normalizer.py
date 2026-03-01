# =========================
# modules/rag/query_normalize/query_normalizer.py
# - Operates on X['query']
# - Strategies: ["none", "lower+strip_punct", "spellfix_light"]
# - If no query column or non-string, skip cleanly
# =========================

import pandas as pd
import re
import warnings

class QueryNormalizer:
    def __init__(self, dataset, strategy="none", query_col="query", verbose=False):
        self.dataset = dataset.copy()
        self.strategy = str(strategy).lower()
        self.query_col = query_col
        self.verbose = verbose

    def _strip_punct(self, s: str) -> str:
        # keep letters/numbers/spaces; drop punctuation-like chars
        return re.sub(r"[^\w\s]", " ", s)

    def _spellfix_light(self, s: str) -> str:
        """
        Very light "spellfix" that is dependency-free:
        - collapse repeated chars (coooool -> cool)
        - normalize whitespace
        NOTE: this is NOT a real spellchecker; it’s a lightweight stand-in.
        """
        s = re.sub(r"(.)\1{2,}", r"\1\1", s)      # aaa -> aa
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def transform(self):
        df = self.dataset

        if self.query_col not in df.columns:
            warnings.warn(f"[QueryNormalizer] Column '{self.query_col}' not found. Skipping.")
            return df

        # Must be string-like
        if not pd.api.types.is_string_dtype(df[self.query_col]):
            warnings.warn(f"[QueryNormalizer] Column '{self.query_col}' is not string dtype. Skipping.")
            return df

        if self.verbose:
            print(f"[QueryNormalizer] Applying strategy='{self.strategy}' on '{self.query_col}'")

        if self.strategy == "none":
            return df

        q = df[self.query_col].fillna("").astype(str)

        if self.strategy == "lower+strip_punct":
            q = q.str.lower().apply(self._strip_punct)
            q = q.apply(lambda s: re.sub(r"\s+", " ", s).strip())

        elif self.strategy == "spellfix_light":
            # keep case, do minimal normalization
            q = q.apply(self._spellfix_light)

        else:
            warnings.warn(f"[QueryNormalizer] Unknown strategy '{self.strategy}'. Skipping.")
            return df

        df[self.query_col] = q
        return df
