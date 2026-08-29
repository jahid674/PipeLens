from modules.data_preparation.multicollinearity.multicollinearity import VIFMulticollinearityCleaner


class MulticollinearityHandler:
    """
    Handler for VIF multicollinearity.

    Strategy index:
      0 -> none
      1 -> drop_high_vif
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.strategy_list = ["none", "drop_high_vif"]

        if self.strategy < 0 or self.strategy >= len(self.strategy_list):
            raise ValueError(
                f"strategy index {self.strategy} out of range (0..{len(self.strategy_list)-1})"
            )

        # defaults defined HERE
        self.vif_threshold = float(config.get("vif_threshold", 10.0))
        self.max_iter = int(config.get("vif_max_iter", 50))
        self.min_features = int(config.get("vif_min_features", 2))
        self.standardize = bool(config.get("vif_standardize", True))

    def apply(self, X, y, sensitive):
        strat = self.strategy_list[self.strategy]

        # --------------------------------------------------
        # Step 1: Drop rows with any NaN in X
        # --------------------------------------------------
        if hasattr(X, "isna"):
            valid_idx = X.dropna().index
            X = X.loc[valid_idx]

            if hasattr(y, "loc"):
                y = y.loc[valid_idx]

            if sensitive is not None and hasattr(sensitive, "loc"):
                sensitive = sensitive.loc[valid_idx]

        # --------------------------------------------------
        # Step 2: Run VIF cleaner
        # --------------------------------------------------
        cleaner = VIFMulticollinearityCleaner(
            X,
            strategy=strat,
            vif_threshold=self.vif_threshold,
            max_iter=self.max_iter,
            min_features=self.min_features,
            standardize=self.standardize,
            verbose=False,
        )

        X_new = cleaner.transform(y_train=y, sensitive_attr_train=sensitive)

        return X_new, y, sensitive

