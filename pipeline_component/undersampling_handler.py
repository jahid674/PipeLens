# handlers/sampling/undersampling_allknn_handler.py

from modules.sampling.undersampling.undersampling import AllKNNUndersampler


class UndersamplingHandler:
    """
    Handler for AllKNN undersampling with integer strategy indexing.

    Expected config keys (suggested):
      - allknn_k_lst:            e.g., [3, 5, 7]
      - allknn_kind_sel_lst:     e.g., ["all", "mode"]  (depends on imblearn version)
      - allknn_allow_minority_lst: [True, False]
      - allknn_sampling_lst:     e.g., ["auto"]
      - random_state:            e.g., 42

    Strategy indexing scheme:
      Enumerate all combinations in deterministic order.
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.k_list = list(config.get("allknn_k_lst", [3]))
        self.kind_sel_list = [s.lower().strip() for s in config.get("allknn_kind_sel_lst", ["all"])]
        self.allow_minority_list = list(config.get("allknn_allow_minority_lst", [True]))
        self.sampling_list = list(config.get("allknn_sampling_lst", ["auto"]))

        self.random_state = int(config.get("random_state", 42))
        self.n_jobs = config.get("n_jobs", None)

        self._catalog = self._build_catalog()

        if self.strategy < 0 or self.strategy >= len(self._catalog):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._catalog)-1})")

    def _build_catalog(self):
        catalog = []
        for samp in self.sampling_list:
            for k in self.k_list:
                for kind_sel in self.kind_sel_list:
                    for allow_minority in self.allow_minority_list:
                        catalog.append(
                            {
                                "sampling_strategy": samp,
                                "n_neighbors": int(k),
                                "kind_sel": kind_sel,
                                "allow_minority": bool(allow_minority),
                            }
                        )
        return catalog

    def apply(self, X, y, sensitive):
        """
        Uniform pipeline interface: returns (X_new, y_new, sensitive_new).

        AllKNN changes the number of rows, so y and sensitive are resampled too.
        """
        spec = self._catalog[self.strategy]

        sampler = AllKNNUndersampler(
            X,
            n_neighbors=spec["n_neighbors"],
            kind_sel=spec["kind_sel"],
            allow_minority=spec["allow_minority"],
            sampling_strategy=spec["sampling_strategy"],
            n_jobs=self.n_jobs,
            verbose=False,
            exclude=None,
        )

        X_new, y_new, s_new = sampler.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y_new, s_new
