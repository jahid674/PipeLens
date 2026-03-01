# pipeline_component/nonlinear_transform_handler.py

from modules.feature_engineering.nonlinear_feature.nonlinear_feature_transformer import NonLinearTransformer


class NonlinearTransformHandler:
    """
    Handler for NonLinearTransformer with EXACTLY 3 options:
      0 -> none
      1 -> quantile
      2 -> power  (Yeo-Johnson)

    Also provides get_config() to return the resolved spec.
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)
        self.config = config or {}

        # keep ONLY: none + both strategies
        self._choices = ["none", "quantile", "power"]

        if self.strategy < 0 or self.strategy >= len(self._choices):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._choices)-1})")

        # fixed safe defaults (you can expose later if you want)
        self.random_state = int(self.config.get("random_state", 42))

        # resolved spec
        self._spec = {
            "strategy": self._choices[self.strategy],

            # quantile (fixed safe)
            "n_quantiles": int(self.config.get("nl_qt_n_quantiles", 1000)),
            "output_distribution": str(self.config.get("nl_qt_output_distribution", "normal")),
            "subsample": int(self.config.get("nl_qt_subsample", 1_000_000)),

            # power (fixed safe)
            "power_method": str(self.config.get("nl_pt_method", "yeo-johnson")),
            "standardize": bool(self.config.get("nl_pt_standardize", True)),

            "random_state": self.random_state,
        }

    def get_config(self):
        """Return the resolved configuration used by this handler instance."""
        return dict(self._spec)

    def apply(self, X, y, sensitive):
        transformer = NonLinearTransformer(
            X,
            strategy=self._spec["strategy"],

            # quantile
            n_quantiles=self._spec["n_quantiles"],
            output_distribution=self._spec["output_distribution"],
            subsample=self._spec["subsample"],
            random_state=self._spec["random_state"],

            # power
            power_method=self._spec["power_method"],
            standardize=self._spec["standardize"],

            verbose=False,
            exclude=None,
            keep_non_numeric=True,
        )

        X_new = transformer.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
