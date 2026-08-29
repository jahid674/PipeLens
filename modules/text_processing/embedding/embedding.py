# modules/text/embedding.py

import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer


class TextEmbedder:
    """
    Text embedding / vectorization module (stateless except for fitted vocab in TF-IDF / Count).

    Strategies:
      - "tfidf"   : TfidfVectorizer (fit on train)
      - "count"   : CountVectorizer (fit on train)
      - "hash"    : HashingVectorizer (no fitting; fixed dimensionality)

    Behavior:
      - Automatically detects text columns (object/string dtype)
      - Concatenates all detected text columns into one document per row
      - Produces a numeric feature matrix and returns it as a DataFrame
      - By default, DROPS the original text columns and keeps non-text columns
      - Works with either a DataFrame or dict {"train","test"} (fit on train, transform both)

    Notes:
      - Output can be high-dimensional; for TF-IDF/Count you can cap with max_features.
      - HashingVectorizer gives stable dimension and avoids train/test vocab mismatch.
    """

    def __init__(
        self,
        dataset,
        strategy="tfidf",
        # auto column detection
        text_cols=None,               # if None, auto-detect object/string cols
        drop_text_cols=True,          # drop original text columns from output
        keep_non_text_cols=True,      # keep remaining columns
        # vectorizer options (shared)
        lowercase=True,
        stop_words=None,              # e.g., "english" or None
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        max_features=None,
        # tf-idf specific
        sublinear_tf=False,
        # hashing specific
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
        # general
        verbose=False,
        exclude=None,                 # columns to exclude from processing (rare)
        random_state=42,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.strategy = str(strategy).lower().strip()

        self.text_cols = text_cols  # can be None; will auto-detect
        self.drop_text_cols = bool(drop_text_cols)
        self.keep_non_text_cols = bool(keep_non_text_cols)

        self.lowercase = bool(lowercase)
        self.stop_words = stop_words
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

        self.sublinear_tf = bool(sublinear_tf)

        self.n_features = int(n_features)
        self.alternate_sign = bool(alternate_sign)
        self.norm = norm

        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []
        self.random_state = int(random_state)

        self._vectorizer = None
        self._fitted_text_cols_ = None

    def _detect_text_columns(self, df: pd.DataFrame):
        cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        cols = [c for c in cols if c not in self.exclude]
        return cols

    def _build_vectorizer(self):
        s = self.strategy
        common = dict(
            lowercase=self.lowercase,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
        )

        if s in ("tfidf", "tf-idf"):
            return TfidfVectorizer(
                **common,
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.max_features,
                sublinear_tf=self.sublinear_tf,
            )

        if s in ("count", "bow", "bagofwords"):
            return CountVectorizer(
                **common,
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.max_features,
            )

        if s in ("hash", "hashing"):
            # No fitting; stable dimensionality
            return HashingVectorizer(
                **common,
                n_features=self.n_features,
                alternate_sign=self.alternate_sign,
                norm=self.norm,
            )

        raise ValueError("Invalid strategy. Choose from {'tfidf','count','hash'}.")

    def _make_documents(self, df: pd.DataFrame, text_cols):
        # Combine multiple text columns into one document per row
        # Fill NaNs with empty strings
        parts = []
        for c in text_cols:
            parts.append(df[c].fillna("").astype(str))
        if len(parts) == 0:
            return pd.Series([""] * len(df), index=df.index)
        doc = parts[0]
        for p in parts[1:]:
            doc = doc.str.cat(p, sep=" ")
        return doc

    def _fit(self, X_train: pd.DataFrame):
        df = X_train.copy()
        if self.text_cols is None:
            text_cols = self._detect_text_columns(df)
        else:
            text_cols = [c for c in self.text_cols if c in df.columns and c not in self.exclude]

        self._fitted_text_cols_ = text_cols

        self._vectorizer = self._build_vectorizer()

        # HashingVectorizer does not need fitting
        if self.strategy in ("hash", "hashing"):
            return

        docs = self._make_documents(df, text_cols)
        self._vectorizer.fit(docs.values)

    def _transform_one(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        text_cols = self._fitted_text_cols_ if self._fitted_text_cols_ is not None else (
            self._detect_text_columns(df) if self.text_cols is None else self.text_cols
        )
        text_cols = [c for c in text_cols if c in df.columns and c not in self.exclude]

        docs = self._make_documents(df, text_cols)
        mat = self._vectorizer.transform(docs.values)

        # Dense is easier for your downstream components; keep it explicit
        if hasattr(mat, "toarray"):
            mat = mat.toarray()

        # Column names
        if hasattr(self._vectorizer, "get_feature_names_out"):
            try:
                feat_names = self._vectorizer.get_feature_names_out()
            except Exception:
                feat_names = np.array([f"emb_{i}" for i in range(mat.shape[1])])
        else:
            feat_names = np.array([f"emb_{i}" for i in range(mat.shape[1])])

        emb_df = pd.DataFrame(mat, columns=[f"text_{n}" for n in feat_names], index=df.index)

        # Optionally keep non-text cols
        if self.keep_non_text_cols:
            non_text_cols = [c for c in df.columns if c not in text_cols]
            kept = df[non_text_cols].copy()
            out = pd.concat([kept, emb_df], axis=1)
        else:
            out = emb_df

        # Optionally drop text cols already handled (usually True)
        if not self.drop_text_cols and self.keep_non_text_cols:
            # If user wants to keep original text columns too, reattach them
            out = pd.concat([df[text_cols].copy(), out], axis=1)

        # Sanity: convert inf to nan (rare), but keep consistent with your style
        out = out.replace([np.inf, -np.inf], np.nan)

        return out

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Text Embedding -----")

        if isinstance(self.dataset, dict):
            X_tr = self.dataset["train"].copy()
            self._fit(X_tr)

            out = {"train": self._transform_one(self.dataset["train"].copy())}
            if "test" in self.dataset and self.dataset["test"] is not None:
                out["test"] = self._transform_one(self.dataset["test"].copy())

        elif isinstance(self.dataset, pd.DataFrame):
            X_tr = self.dataset.copy()
            self._fit(X_tr)
            out = self._transform_one(X_tr)

        else:
            raise TypeError("dataset must be a pandas DataFrame or a dict with keys {'train','test'}.")

        if self.verbose:
            print(f"Text embedding completed in {time.time() - start_time:.2f} seconds.")

        return out
