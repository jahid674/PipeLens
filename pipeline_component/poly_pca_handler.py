# handlers/feature_engineering/poly_pca_handler.py

from modules.feature_engineering.polynomial_feature_generation.polynomial_feature import PolyPCATransformer


class PolyPcaHandler:
    """
    EXACTLY one option per reducer strategy + none:

      0 -> none
      1 -> pca
      2 -> sparsepca
      3 -> minibatchsparsepca
      4 -> kernelpca

    The hyperparameters are fixed to one value each (safe defaults),
    but can be overridden via config keys if you want.

    Also exposes get_config() returning resolved spec.
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)
        self.config = config or {}

        self._choices = ["none", "pca", "sparsepca", "minibatchsparsepca", "kernelpca"]

        if self.strategy < 0 or self.strategy >= len(self._choices):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._choices)-1})")

        # fixed single-choice params (override-able)
        self._spec = {
            "degree": int(self.config.get("poly_degree", 2)),
            "include_bias": bool(self.config.get("poly_include_bias", False)),
            "interaction_only": bool(self.config.get("poly_interaction_only", False)),

            "reducer": self._choices[self.strategy],
            "n_components": self.config.get("poly_n_components", None),  # None => keep all

            # KernelPCA (one fixed choice)
            "kernel": str(self.config.get("kpca_kernel", "rbf")),
            "gamma": self.config.get("kpca_gamma", None),
            "coef0": float(self.config.get("kpca_coef0", 1.0)),
            "fit_inverse_transform": bool(self.config.get("kpca_fit_inverse_transform", False)),

            # SparsePCA variants (one fixed choice)
            "alpha": float(self.config.get("spca_alpha", 1.0)),
            "ridge_alpha": float(self.config.get("spca_ridge_alpha", 0.01)),
            "batch_size": int(self.config.get("mbspca_batch_size", 256)),
            "max_iter": int(self.config.get("spca_max_iter", 1000)),
            "tol": float(self.config.get("spca_tol", 1e-3)),

            "random_state": int(self.config.get("random_state", 42)),
        }

    def get_config(self):
        return dict(self._spec)

    def apply(self, X, y, sensitive):
        transformer = PolyPCATransformer(
            X,
            degree=self._spec["degree"],
            include_bias=self._spec["include_bias"],
            interaction_only=self._spec["interaction_only"],

            reducer=self._spec["reducer"],
            n_components=self._spec["n_components"],

            # KernelPCA
            kernel=self._spec["kernel"],
            gamma=self._spec["gamma"],
            coef0=self._spec["coef0"],
            kpca_fit_inverse_transform=self._spec["fit_inverse_transform"],

            # SparsePCA variants
            alpha=self._spec["alpha"],
            ridge_alpha=self._spec["ridge_alpha"],
            batch_size=self._spec["batch_size"],
            max_iter=self._spec["max_iter"],
            tol=self._spec["tol"],

            random_state=self._spec["random_state"],
            verbose=False,
            exclude=None,
            keep_non_numeric=True,
        )

        X_new = transformer.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive

