# handlers/sampling/oversampling_smote_handler.py

from modules.sampling.oversampling.oversampling import SMOTEOversampler


class OversamplingHandler:
    """
    Handler for SMOTE oversampling with integer strategy indexing.

    Expected config keys (suggested, consistent with your style):
      - smote_k_lst:         e.g., [3, 5, 7]
      - smote_sampling_lst:  e.g., ["auto", 0.5, 1.0]
      - random_state:        e.g., 42

    Strategy indexing scheme:
      Enumerate all (sampling_strategy, k_neighbors) pairs in deterministic order.
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.k_list = list(config.get("smote_k_lst", [5]))
        self.sampling_list = list(config.get("smote_sampling_lst", ["auto"]))

        self.random_state = int(config.get("random_state", 42))

        self._catalog = self._build_catalog()

        if self.strategy < 0 or self.strategy >= len(self._catalog):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._catalog)-1})")

    def _build_catalog(self):
        catalog = []
        for samp in self.sampling_list:
            for k in self.k_list:
                catalog.append(
                    {
                        "sampling_strategy": samp,
                        "k_neighbors": int(k),
                    }
                )
        return catalog

    def apply(self, X, y, sensitive):
        """
        Uniform pipeline interface: returns (X_new, y_new, sensitive_new).

        SMOTE changes the number of rows, so y and sensitive are resampled too.
        """
        spec = self._catalog[self.strategy]

        sm = SMOTEOversampler(
            X,
            sampling_strategy=spec["sampling_strategy"],
            k_neighbors=spec["k_neighbors"],
            random_state=self.random_state,
            verbose=False,
            exclude=None,
        )

        X_new, y_new, s_new = sm.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y_new, s_new
