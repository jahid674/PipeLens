# handlers/text/embedding_handler.py

from modules.text_processing.embedding.embedding import TextEmbedder


class EmbeddingHandler:
    """
    Handler for TextEmbedder with integer strategy indexing.

    Config keys (suggested):
      - emb_strategy: ["tfidf","hash","count"]

      TF-IDF / Count:
        - emb_ngram_lst: [(1,1), (1,2)]
        - emb_max_features_lst: [5000, 20000]
        - emb_min_df_lst: [1, 2]
        - emb_max_df_lst: [1.0, 0.95]
        - tfidf_sublinear_tf_lst: [False, True]

      Hashing:
        - hash_n_features_lst: [2**16, 2**18]
        - hash_alternate_sign_lst: [False, True]
        - hash_norm_lst: ["l2", None]

      Shared:
        - emb_stop_words: None or "english"
        - emb_lowercase: True/False
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)
        self.config = config

        self.emb_strategy = [s.lower().strip() for s in config.get("emb_strategy", ["tfidf"])]

        # shared
        self.stop_words = config.get("emb_stop_words", None)
        self.lowercase = bool(config.get("emb_lowercase", True))

        # tfidf/count grids
        self.ngram_lst = list(config.get("emb_ngram_lst", [(1, 1)]))
        self.max_features_lst = list(config.get("emb_max_features_lst", [None]))
        self.min_df_lst = list(config.get("emb_min_df_lst", [1]))
        self.max_df_lst = list(config.get("emb_max_df_lst", [1.0]))
        self.sublinear_tf_lst = list(config.get("tfidf_sublinear_tf_lst", [False]))

        # hashing grids
        self.hash_n_features_lst = list(config.get("hash_n_features_lst", [2**18]))
        self.hash_alternate_sign_lst = list(config.get("hash_alternate_sign_lst", [False]))
        self.hash_norm_lst = list(config.get("hash_norm_lst", ["l2"]))

        self._catalog = self._build_catalog()

        if self.strategy < 0 or self.strategy >= len(self._catalog):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._catalog)-1})")

    def _build_catalog(self):
        cat = []

        # TF-IDF variants
        if "tfidf" in self.emb_strategy or "tf-idf" in self.emb_strategy:
            for ng in self.ngram_lst:
                for mf in self.max_features_lst:
                    for mindf in self.min_df_lst:
                        for maxdf in self.max_df_lst:
                            for stf in self.sublinear_tf_lst:
                                cat.append(
                                    {
                                        "strategy": "tfidf",
                                        "ngram_range": tuple(ng),
                                        "max_features": mf,
                                        "min_df": mindf,
                                        "max_df": maxdf,
                                        "sublinear_tf": bool(stf),
                                    }
                                )

        # Count variants
        if "count" in self.emb_strategy or "bow" in self.emb_strategy:
            for ng in self.ngram_lst:
                for mf in self.max_features_lst:
                    for mindf in self.min_df_lst:
                        for maxdf in self.max_df_lst:
                            cat.append(
                                {
                                    "strategy": "count",
                                    "ngram_range": tuple(ng),
                                    "max_features": mf,
                                    "min_df": mindf,
                                    "max_df": maxdf,
                                }
                            )

        # Hashing variants
        if "hash" in self.emb_strategy or "hashing" in self.emb_strategy:
            for nf in self.hash_n_features_lst:
                for als in self.hash_alternate_sign_lst:
                    for nm in self.hash_norm_lst:
                        cat.append(
                            {
                                "strategy": "hash",
                                "ngram_range": (1, 2),  # hashing commonly benefits from char/word ngrams; keep simple
                                "n_features": int(nf),
                                "alternate_sign": bool(als),
                                "norm": nm,
                            }
                        )

        if len(cat) == 0:
            raise ValueError("EmbeddingHandler: empty catalog; check emb_strategy config.")
        return cat

    def apply(self, X, y, sensitive):
        """
        Uniform pipeline interface: returns (X_new, y, sensitive).
        Embedding changes columns but not rows.
        """
        spec = self._catalog[self.strategy]

        embedder = TextEmbedder(
            X,
            strategy=spec["strategy"],
            text_cols=None,  # auto-detect
            drop_text_cols=True,
            keep_non_text_cols=True,
            lowercase=self.lowercase,
            stop_words=self.stop_words,
            ngram_range=spec.get("ngram_range", (1, 1)),
            min_df=spec.get("min_df", 1),
            max_df=spec.get("max_df", 1.0),
            max_features=spec.get("max_features", None),
            sublinear_tf=spec.get("sublinear_tf", False),
            n_features=spec.get("n_features", 2**18),
            alternate_sign=spec.get("alternate_sign", False),
            norm=spec.get("norm", "l2"),
            verbose=False,
            exclude=None,
        )

        X_new = embedder.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
