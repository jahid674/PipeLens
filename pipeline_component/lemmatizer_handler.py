# handlers/text/lemmatizer_handler.py

from modules.text_processing.lemmatizer.lemmatizer import Lemmatizer


class LemmatizerHandler:
    """
    Handler for lemmatization with integer strategy indexing.

    Config keys:
      - lemma_strategy: ["wordnet", "spacy"]
      - lemma_language: "en"
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.strategy_list = [
            s.lower() for s in config.get("lemma_strategy", ["wordnet"])
        ]
        self.language = config.get("lemma_language", "en")

        if self.strategy < 0 or self.strategy >= len(self.strategy_list):
            raise ValueError(
                f"strategy index {self.strategy} out of range (0..{len(self.strategy_list)-1})"
            )

    def apply(self, X, y, sensitive):
        lemma_strategy = self.strategy_list[self.strategy]

        lemmatizer = Lemmatizer(
            X,
            strategy=lemma_strategy,
            language=self.language,
            verbose=False,
            exclude=None,
        )

        X_new = lemmatizer.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
