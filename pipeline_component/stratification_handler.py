# handlers/sampling/stratification_split_handler.py

from modules.sampling.stratification.stratification import StratificationSplitter


class StratificationSplitHandler:
    """
    Handler for stratification split with integer strategy indexing.

    Expected config keys (suggested):
      - split_strategy: ["random", "stratified"]
      - test_size_lst: [0.2, 0.3]   (optional grid)
      - random_state: 42
      - shuffle: True

    Strategy indexing:
      - enumerate all (strategy, test_size) combinations
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.split_strategy = [s.lower().strip() for s in config.get("split_strategy", ["random", "stratified"])]
        self.test_size_list = list(config.get("test_size_lst", [0.2]))

        self.shuffle = bool(config.get("shuffle", True))
        self.random_state = int(config.get("random_state", 42))

        self._catalog = self._build_catalog()

        if self.strategy < 0 or self.strategy >= len(self._catalog):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._catalog)-1})")

    def _build_catalog(self):
        catalog = []
        for strat in self.split_strategy:
            for ts in self.test_size_list:
                catalog.append(
                    {
                        "strategy": strat,
                        "test_size": float(ts),
                    }
                )
        return catalog

    def apply(self, X, y, sensitive):
        """
        Returns:
          X_new: {"train":..., "test":...}
          y_new: {"train":..., "test":...}
          s_new: {"train":..., "test":...} or None

        NOTE: This changes the container type from DataFrame -> dict.
        Downstream modules must support dict inputs (as your other modules do).
        """
        spec = self._catalog[self.strategy]

        splitter = StratificationSplitter(
            X,
            strategy=spec["strategy"],
            test_size=spec["test_size"],
            shuffle=self.shuffle,
            random_state=self.random_state,
            verbose=False,
            exclude=None,
        )

        X_new, y_new, s_new = splitter.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y_new, s_new
