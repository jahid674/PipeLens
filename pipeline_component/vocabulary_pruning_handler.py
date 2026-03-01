# handlers/text/vocab_pruner_handler.py

from modules.text_processing.vocabulary_pruning.vocabulary_pruning import VocabularyPruner


class VocabularyPruningHandler:
    """
    Handler for VocabularyPruner with integer strategy indexing.

    Suggested config keys:
      - vocab_prune_strategy: ["min_df","min_max_df","top_k","top_k_tfidf"]
      - vocab_min_df_lst: [2, 5]
      - vocab_max_df_lst: [0.95, 0.90]  # floats in (0,1] or ints
      - vocab_top_k_lst: [5000, 20000]
      - vocab_lowercase: True/False
      - vocab_strip_punct: True/False
      - tfidf_smooth_idf: True/False
      - tfidf_sublinear_tf: True/False

    Strategy indexing scheme:
      Enumerate deterministic combinations for each prune strategy.
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.strategy_list = [s.lower().strip() for s in config.get(
            "vocab_prune_strategy", ["min_df"]
        )]

        self.min_df_lst = list(config.get("vocab_min_df_lst", [2]))
        self.max_df_lst = list(config.get("vocab_max_df_lst", [0.95]))
        self.top_k_lst = list(config.get("vocab_top_k_lst", [20000]))

        self.lowercase = bool(config.get("vocab_lowercase", True))
        self.strip_punct = bool(config.get("vocab_strip_punct", True))

        self.tfidf_smooth_idf = bool(config.get("tfidf_smooth_idf", True))
        self.tfidf_sublinear_tf = bool(config.get("tfidf_sublinear_tf", True))

        self._catalog = self._build_catalog()

        if self.strategy < 0 or self.strategy >= len(self._catalog):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._catalog)-1})")

    def _build_catalog(self):
        cat = []

        for strat in self.strategy_list:
            if strat in ("min_df",):
                for mindf in self.min_df_lst:
                    cat.append({"strategy": "min_df", "min_df": int(mindf)})

            elif strat in ("max_df",):
                for maxdf in self.max_df_lst:
                    cat.append({"strategy": "max_df", "max_df": maxdf})

            elif strat in ("min_max_df", "minmax_df"):
                for mindf in self.min_df_lst:
                    for maxdf in self.max_df_lst:
                        cat.append({"strategy": "min_max_df", "min_df": int(mindf), "max_df": maxdf})

            elif strat in ("top_k", "topk"):
                for k in self.top_k_lst:
                    cat.append({"strategy": "top_k", "top_k": int(k)})

            elif strat in ("top_k_tfidf", "tfidf_top_k"):
                for k in self.top_k_lst:
                    cat.append({"strategy": "top_k_tfidf", "top_k": int(k)})

            else:
                raise ValueError(f"Unknown vocab pruning strategy: {strat}")

        if len(cat) == 0:
            raise ValueError("VocabularyPruningHandler: empty catalog; check vocab_prune_strategy config.")
        return cat

    def apply(self, X, y, sensitive):
        spec = self._catalog[self.strategy]

        pruner = VocabularyPruner(
            X,
            strategy=spec["strategy"],
            text_cols=None,  # auto-detect
            lowercase=self.lowercase,
            strip_punct=self.strip_punct,
            min_df=spec.get("min_df", 2),
            max_df=spec.get("max_df", 0.95),
            top_k=spec.get("top_k", 20000),
            tfidf_smooth_idf=self.tfidf_smooth_idf,
            tfidf_sublinear_tf=self.tfidf_sublinear_tf,
            verbose=False,
            exclude=None,
        )

        X_new = pruner.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
