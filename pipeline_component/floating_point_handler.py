from modules.data_preparation.floating_point_stabilization.floating_point_stabilization import FloatingPointStabilizer


class FloatingPointHandler:
    def __init__(self, strategy, config):
        # strategy is an index (0-based) passed by PipelineExecutor
        self.strategy = int(strategy)

        # list of strategy names defined in executor
        self.fp_strategy = config["floating_point_strategy"]

        # optional hyperparams (single values recommended)
        # if you want fixed ones, keep these as constants or config values
        self.decimals = int(config.get("fp_decimals", 6))
        self.tol = float(config.get("fp_tol", 1e-8))

        if self.strategy < 0 or self.strategy >= len(self.fp_strategy):
            raise ValueError(
                f"FloatingPointHandler: strategy index {self.strategy} out of range "
                f"(0..{len(self.fp_strategy)-1}). fp_strategy={self.fp_strategy}"
            )

    def apply(self, X, y, sensitive):
        strat = self.fp_strategy[self.strategy]  # e.g., "none", "snap", "round", "both"
        stab = FloatingPointStabilizer(X, strategy=strat, decimals=self.decimals, tol=self.tol, verbose=False)
        X_new = stab.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
