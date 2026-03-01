# modules/text/vocab_pruner.py

import pandas as pd
import numpy as np
import time


class VocabularyPruner:
    """
    Vocabulary pruning for text columns.

    Goal:
      Reduce vocabulary explosion and noise BEFORE embedding/vectorization.

    Strategies:
      - "min_df"       : keep tokens appearing in >= min_df documents
      - "max_df"       : keep tokens appearing in <= max_df fraction/doc count
      - "min_max_df"   : apply both min_df and max_df
      - "top_k"        : keep top_k tokens by document frequency (DF)
      - "top_k_tfidf"  : keep top_k tokens by aggregated TF-IDF score (train only)

    Behavior:
      - Auto-detects text columns (object/string) unless text_cols provided
      - Fits token stats on TRAIN only (if dataset is dict)
      - Transforms by removing pruned tokens from text (reconstructs cleaned text)
      - Does NOT change number of rows; y and sensitive unchanged

    Tokenization:
      - simple whitespace tokenization (fast + deterministic)
      - optional lowercase + basic character filtering
    """

    def __init__(
        self,
        dataset,
        strategy="min_df",
        text_cols=None,
        lowercase=True,
        strip_punct=True,
        # pruning params
        min_df=2,                # int document count
        max_df=0.95,             # float fraction in (0,1] or int doc count
        top_k=20000,
        # tf-idf aggregation params
        tfidf_smooth_idf=True,
        tfidf_sublinear_tf=True,
        # general
        verbose=False,
        exclude=None,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.strategy = str(strategy).lower().strip()

        self.text_cols = text_cols
        self.lowercase = bool(lowercase)
        self.strip_punct = bool(strip_punct)

        self.min_df = int(min_df)
        self.max_df = max_df
        self.top_k = int(top_k)

        self.tfidf_smooth_idf = bool(tfidf_smooth_idf)
        self.tfidf_sublinear_tf = bool(tfidf_sublinear_tf)

        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

        # learned vocab to KEEP (from train)
        self._keep_vocab = None
        self._fitted_text_cols_ = None

    def _detect_text_columns(self, df: pd.DataFrame):
        cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        cols = [c for c in cols if c not in self.exclude]
        return cols

    def _normalize_text(self, s: str) -> str:
        if s is None:
            s = ""
        s = str(s)
        if self.lowercase:
            s = s.lower()
        if self.strip_punct:
            # simple punctuation stripping (keeps alnum and spaces)
            # avoids regex dependency; deterministic
            cleaned = []
            for ch in s:
                if ch.isalnum() or ch.isspace():
                    cleaned.append(ch)
                else:
                    cleaned.append(" ")
            s = "".join(cleaned)
        # collapse spaces
        s = " ".join(s.split())
        return s

    def _tokenize(self, s: str):
        s = self._normalize_text(s)
        if not s:
            return []
        return s.split()

    def _build_documents(self, df: pd.DataFrame, text_cols):
        # concatenate all text columns into one doc per row
        parts = []
        for c in text_cols:
            parts.append(df[c].fillna("").astype(str))
        if len(parts) == 0:
            return pd.Series([""] * len(df), index=df.index)

        doc = parts[0]
        for p in parts[1:]:
            doc = doc.str.cat(p, sep=" ")
        return doc

    def _compute_df_counts(self, docs: pd.Series):
        # document frequency: number of docs containing token at least once
        df_counts = {}
        n_docs = len(docs)
        for text in docs.values:
            toks = set(self._tokenize(text))
            for t in toks:
                df_counts[t] = df_counts.get(t, 0) + 1
        return df_counts, n_docs

    def _compute_tfidf_scores(self, docs: pd.Series):
        """
        Aggregated TF-IDF over corpus (train):
          score(token) = sum_over_docs( tf(token,doc) * idf(token) )

        Implemented simply (no sklearn) to keep dependencies minimal.
        """
        df_counts, n_docs = self._compute_df_counts(docs)

        # idf
        idf = {}
        for t, dfc in df_counts.items():
            if self.tfidf_smooth_idf:
                # smooth: log((1+n)/(1+df)) + 1
                idf[t] = np.log((1.0 + n_docs) / (1.0 + dfc)) + 1.0
            else:
                idf[t] = np.log(n_docs / max(dfc, 1))

        scores = {t: 0.0 for t in df_counts.keys()}

        # accumulate tf-idf
        for text in docs.values:
            toks = self._tokenize(text)
            if not toks:
                continue
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            for t, c in tf.items():
                val = np.log(1 + c) if self.tfidf_sublinear_tf else float(c)
                scores[t] += val * idf.get(t, 0.0)

        return scores, n_docs, df_counts

    def _select_keep_vocab(self, docs: pd.Series):
        strategy = self.strategy

        if strategy in ("top_k_tfidf", "tfidf_top_k"):
            scores, n_docs, df_counts = self._compute_tfidf_scores(docs)
            # keep top_k by score
            sorted_tokens = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            keep = set([t for t, _ in sorted_tokens[: self.top_k]])
            return keep

        # DF-based strategies
        df_counts, n_docs = self._compute_df_counts(docs)

        # interpret max_df
        max_df = self.max_df
        if isinstance(max_df, float):
            if not (0 < max_df <= 1.0):
                raise ValueError("max_df as float must be in (0,1].")
            max_df_count = int(np.floor(max_df * n_docs))
        else:
            max_df_count = int(max_df)

        if strategy in ("min_df",):
            keep = {t for t, c in df_counts.items() if c >= self.min_df}
            return keep

        if strategy in ("max_df",):
            keep = {t for t, c in df_counts.items() if c <= max_df_count}
            return keep

        if strategy in ("min_max_df", "minmax_df"):
            keep = {t for t, c in df_counts.items() if (c >= self.min_df and c <= max_df_count)}
            return keep

        if strategy in ("top_k", "topk"):
            sorted_tokens = sorted(df_counts.items(), key=lambda x: x[1], reverse=True)
            keep = set([t for t, _ in sorted_tokens[: self.top_k]])
            return keep

        raise ValueError(
            "Invalid strategy. Choose from {'min_df','max_df','min_max_df','top_k','top_k_tfidf'}."
        )

    def _prune_text(self, text: str):
        toks = self._tokenize(text)
        if not toks:
            return ""
        kept = [t for t in toks if t in self._keep_vocab]
        return " ".join(kept)

    def _transform_df(self, df: pd.DataFrame):
        df_out = df.copy()
        text_cols = self._fitted_text_cols_ if self._fitted_text_cols_ is not None else (
            self._detect_text_columns(df_out) if self.text_cols is None else self.text_cols
        )
        text_cols = [c for c in text_cols if c in df_out.columns and c not in self.exclude]

        for col in text_cols:
            series = df_out[col].fillna("").astype(str)
            df_out[col] = series.apply(self._prune_text)

        return df_out

    def _fit(self, X_train: pd.DataFrame):
        df = X_train.copy()
        text_cols = self._detect_text_columns(df) if self.text_cols is None else self.text_cols
        text_cols = [c for c in text_cols if c in df.columns and c not in self.exclude]
        self._fitted_text_cols_ = text_cols

        docs = self._build_documents(df, text_cols)
        self._keep_vocab = self._select_keep_vocab(docs)

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Vocabulary Pruning -----")

        if isinstance(self.dataset, dict):
            self._fit(self.dataset["train"].copy())

            out = {"train": self._transform_df(self.dataset["train"].copy())}
            if "test" in self.dataset and self.dataset["test"] is not None:
                out["test"] = self._transform_df(self.dataset["test"].copy())

        elif isinstance(self.dataset, pd.DataFrame):
            # fit on full df (no split available)
            self._fit(self.dataset.copy())
            out = self._transform_df(self.dataset.copy())

        else:
            raise TypeError("dataset must be a DataFrame or dict {'train','test'}")

        if self.verbose:
            keep_sz = len(self._keep_vocab) if self._keep_vocab is not None else 0
            print(f"Vocab pruning completed in {time.time() - start_time:.2f}s; kept_vocab={keep_sz}")

        return out
