from modules.data_preparation.invalid_value.invalid_value import InvalidValueRepair


class InvalidValueHandler:
    """
    Handler for InvalidValueRepair with integer strategy indexing.

    Strategies (index → behavior):
      0 -> "none"
      1 -> "sentinel"
      2 -> "regex"
      3 -> "both"

    Defaults are defined HERE (not in executor).
    """

    def __init__(self, strategy, config=None):
        self.strategy = int(strategy)

        # ---- strategy list (fixed, simple) ----
        self.inv_strategy = ["none", "sentinel", "regex", "both"]

        # ---- DEFAULTS DEFINED HERE ----
        self.numeric_sentinels = [-999, -9999, 999, 9999, 99999]

        self.string_patterns = [
            r"^\s*$",          # empty
            r"^na$", r"^n/a$", r"^none$", r"^null$", r"^nan$",
            r"^unknown$", r"^missing$",
            r"^\?$"
        ]

        self.case_insensitive = True
        self.strip_whitespace = True

        # ---- bounds check ----
        if self.strategy < 0 or self.strategy >= len(self.inv_strategy):
            raise ValueError(
                f"InvalidValueHandler: strategy index {self.strategy} out of range "
                f"(0..{len(self.inv_strategy)-1})"
            )

    def apply(self, X, y, sensitive):
        strat = self.inv_strategy[self.strategy]

        repair = InvalidValueRepair(
            X,
            strategy=strat,
            numeric_sentinels=self.numeric_sentinels,
            string_patterns=self.string_patterns,
            case_insensitive=self.case_insensitive,
            strip_whitespace=self.strip_whitespace,
            verbose=False,
            exclude=None,
        )

        X_new = repair.transform(y_train=y, sensitive_attr_train=sensitive)
        return X_new, y, sensitive
