# handlers/numerical/distribution_shape_correction_handler.py

from modules.data_preparation.distribution_shape.distribution_shape_corrector import DistributionShapeCorrector


class DistributionShapeHandler:
    """
    Handler for DistributionShapeCorrector with integer strategy indexing.

    Suggested config keys:
      - shape_strategy: ["log1p","sqrt","yeojohnson","boxcox"]
      - shape_standardize_lst: [False, True]     # for yeojohnson/boxcox only
      - shape_epsilon: 1e-6
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.strategy_list = [s.lower().strip() for s in config.get(
            "shape_strategy", ["none", "log1p", "sqrt", "boxcox"]
        )]
        self.standardize_list = list(config.get("shape_standardize_lst", [False]))
        self.epsilon = float(config.get("shape_epsilon", 1e-6))

        self._catalog = self._build_catalog()
        #print(f"DistributionShapeHandler catalog: {self._catalog}")
        #print(f"strategy", self.strategy)

        if self.strategy < 0 or self.strategy >= len(self._catalog):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._catalog)-1})")

    def _build_catalog(self):
        cat = []
        for strat in self.strategy_list:
            if strat in ("yeojohnson", "boxcox"):
                for std in self.standardize_list:
                    cat.append({"strategy": strat, "standardize": bool(std)})
            else:
                cat.append({"strategy": strat, "standardize": False})
        if len(cat) == 0:
            raise ValueError("DistributionShapeCorrectionHandler: empty catalog; check shape_strategy config.")
        return cat

    def apply(self, X, y, sensitive):
        spec = self._catalog[self.strategy]
        print(f"Applying DistributionShapeCorrector with spec: {spec}")

        corrector = DistributionShapeCorrector(
            X,
            strategy=spec["strategy"],
            standardize=spec.get("standardize", False),
            epsilon=self.epsilon,
            verbose=False,
            exclude=None,
        )

        X_new = corrector.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
