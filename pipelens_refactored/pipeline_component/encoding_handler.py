# handlers/encoding/encoder_auto_handler.py

from modules.encoding.encoding import EncoderModuleAuto


class EncodingHandler:
    """
    Handler for EncoderModuleAuto with integer strategy indexing.

    Suggested config keys:
      - enc_strategy: ["binarizer","kbins","onehot","label_encoder","label_binarizer"]
      - cat_max_unique: None or int (e.g., 20)   # treat low-cardinality numeric as categorical

      Binarizer:
        - bin_threshold_lst

      KBins:
        - kbins_n_bins_lst
        - kbins_encode_lst
        - kbins_strategy_lst

      OneHot:
        - ohe_drop_lst
        - ohe_handle_unknown
        - ohe_sparse_output
        - ohe_min_frequency
        - ohe_max_categories

      LabelBinarizer:
        - lb_drop_first_lst: [True, False]
    """

    def __init__(self, strategy, config):
        self.strategy = int(strategy)

        self.enc_strategy = [s.lower().strip() for s in config.get(
            "enc_strategy",
            ["binarizer", "kbins", "onehot", "label_encoder", "label_binarizer"]
        )]

        self.cat_max_unique = config.get("cat_max_unique", None)
        self.cat_min_unique = config.get("cat_min_unique", None)

        # binarizer
        self.bin_threshold_lst = list(config.get("bin_threshold_lst", [0.0]))

        # kbins
        self.kbins_n_bins_lst = list(config.get("kbins_n_bins_lst", [5]))
        self.kbins_encode_lst = list(config.get("kbins_encode_lst", ["onehot-dense"]))
        self.kbins_strategy_lst = list(config.get("kbins_strategy_lst", ["quantile"]))

        # onehot
        self.ohe_drop_lst = list(config.get("ohe_drop_lst", [None]))
        self.ohe_handle_unknown = config.get("ohe_handle_unknown", "ignore")
        self.ohe_sparse_output = bool(config.get("ohe_sparse_output", False))
        self.ohe_min_frequency = config.get("ohe_min_frequency", None)
        self.ohe_max_categories = config.get("ohe_max_categories", None)

        # label binarizer
        self.lb_drop_first_lst = list(config.get("lb_drop_first_lst", [True]))

        self._catalog = self._build_catalog()

        if self.strategy < 0 or self.strategy >= len(self._catalog):
            raise ValueError(f"strategy index {self.strategy} out of range (0..{len(self._catalog)-1})")

    def _build_catalog(self):
        cat = []

        if "binarizer" in self.enc_strategy:
            for thr in self.bin_threshold_lst:
                cat.append({"strategy": "binarizer", "threshold": float(thr)})

        if "kbins" in self.enc_strategy:
            for nb in self.kbins_n_bins_lst:
                for enc in self.kbins_encode_lst:
                    for st in self.kbins_strategy_lst:
                        cat.append(
                            {
                                "strategy": "kbins",
                                "n_bins": int(nb),
                                "kbins_encode": enc,
                                "kbins_strategy": st,
                            }
                        )

        if "onehot" in self.enc_strategy:
            for drop in self.ohe_drop_lst:
                cat.append(
                    {
                        "strategy": "onehot",
                        "drop": drop,
                        "handle_unknown": self.ohe_handle_unknown,
                        "sparse_output": self.ohe_sparse_output,
                        "min_frequency": self.ohe_min_frequency,
                        "max_categories": self.ohe_max_categories,
                    }
                )

        if "label_encoder" in self.enc_strategy:
            cat.append({"strategy": "label_encoder"})

        if "label_binarizer" in self.enc_strategy:
            for d in self.lb_drop_first_lst:
                cat.append({"strategy": "label_binarizer", "lb_drop_first": bool(d)})

        if len(cat) == 0:
            raise ValueError("EncodingAutoHandler: empty strategy catalog; check enc_strategy config.")
        return cat

    def apply(self, X, y, sensitive):
        """
        Uniform pipeline interface: returns (X_new, y, sensitive).
        Encoding changes columns but not rows.
        """
        spec = self._catalog[self.strategy]

        encoder = EncoderModuleAuto(
            X,
            strategy=spec["strategy"],
            cat_max_unique=self.cat_max_unique,
            cat_min_unique=self.cat_min_unique,
            keep_other_cols=True,
            verbose=False,
            exclude=None,

            # binarizer
            threshold=spec.get("threshold", 0.0),

            # kbins
            n_bins=spec.get("n_bins", 5),
            kbins_encode=spec.get("kbins_encode", "onehot-dense"),
            kbins_strategy=spec.get("kbins_strategy", "quantile"),

            # onehot
            drop=spec.get("drop", None),
            handle_unknown=spec.get("handle_unknown", "ignore"),
            min_frequency=spec.get("min_frequency", None),
            max_categories=spec.get("max_categories", None),
            sparse_output=spec.get("sparse_output", False),

            # label binarizer
            lb_drop_first=spec.get("lb_drop_first", True),
        )

        X_new = encoder.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
