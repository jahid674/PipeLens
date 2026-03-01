# handlers/text/ngram_extractor_handler.py

from modules.text_processing.ngram_extractor import NGramExtractor


class NGramExtractorHandler:
    """
    Handler for NGramExtractor with integer strategy indexing.

    Suggested config keys:
      - ngram_strategy: ["word","char"]
      - ngram_range_lst: [(1,1), (1,2), (2,3)]
      - ngram_lowercase: True/False
      - ngram_strip_punct: True/False
      - ngram_drop_original_text: True/False
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.strategy_list = [s.lower().strip() for s in config.get("ngram_strategy", ["word"])]
        self.range_list = list(config.get("ngram_range_lst", [(1, 2)]))

        self.lowercase = bool(config.get("ngram_lowercase", True))
        self.strip_punct = bool(config.get("ngram_strip_punct", True))
        self.drop_original_text = bool(config.get("ngram_drop_original_text", False))

        self._catalog = self._build_catalog()

        if self.strategy < 0 or self.strategy >= len(self._catalog):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._catalog)-1})")

    def _build_catalog(self):
        cat = []
        for strat in self.strategy_list:
            for rg in self.range_list:
                cat.append(
                    {
                        "strategy": strat,
                        "ngram_range": tuple(rg),
                    }
                )
        if len(cat) == 0:
            raise ValueError("NGramExtractionHandler: empty catalog; check ngram_strategy config.")
        return cat

    def apply(self, X, y, sensitive):
        spec = self._catalog[self.strategy]

        extractor = NGramExtractor(
            X,
            strategy=spec["strategy"],
            text_cols=None,  # auto-detect
            ngram_range=spec["ngram_range"],
            lowercase=self.lowercase,
            strip_punct=self.strip_punct,
            drop_original_text=self.drop_original_text,
            verbose=False,
            exclude=None,
        )

        X_new = extractor.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
