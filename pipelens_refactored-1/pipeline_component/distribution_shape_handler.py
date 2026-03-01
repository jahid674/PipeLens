# handlers/numerical/distribution_shape_correction_handler.py

from modules.data_preparation.distribution_shape.distribution_shape_corrector import DistributionShapeCorrector
import numpy as np

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
            "shape_strategy", ["none", "log1p", "sqrt", "yeojohnson"]
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

        def _clean_align_df(X_df, y_obj, s_obj):
            # Replace inf with nan, then drop rows with any nan in X
            Xc = X_df.replace([np.inf, -np.inf], np.nan)

            # Build mask for rows to keep
            mask = Xc.notna().all(axis=1)

            # Also ensure y/sensitive not missing (if pandas)
            if hasattr(y_obj, "reindex"):
                y_aligned = y_obj.reindex(Xc.index)
                mask = mask & y_aligned.notna()
            else:
                y_aligned = y_obj

            if hasattr(s_obj, "reindex"):
                s_aligned = s_obj.reindex(Xc.index)
                mask = mask & s_aligned.notna()
            else:
                s_aligned = s_obj

            kept_idx = Xc.index[mask]

            # Filter + reset
            X_out = Xc.loc[kept_idx].reset_index(drop=True)

            if hasattr(y_aligned, "reindex"):
                y_out = y_aligned.reindex(kept_idx).reset_index(drop=True)
            else:
                # fallback: positional indexing (assumes same order/length)
                y_out = y_aligned[mask.to_numpy()]

            if hasattr(s_aligned, "reindex"):
                s_out = s_aligned.reindex(kept_idx).reset_index(drop=True)
            else:
                s_out = s_aligned[mask.to_numpy()]

            return X_out, y_out, s_out

        # -------------------------
        # Clean + align first
        # -------------------------
        if isinstance(X, dict):
            # clean train
            X_train, y_new, sens_new = _clean_align_df(X["train"], y, sensitive)
            X_clean = {"train": X_train}

            # clean test independently (no y/sensitive assumed here)
            if "test" in X and X["test"] is not None:
                X_test = X["test"].replace([np.inf, -np.inf], np.nan)
                X_clean["test"] = X_test.dropna(axis=0, how="any").reset_index(drop=True)
            else:
                X_clean["test"] = None

            # Now transform consistently (fit on train inside corrector)
            corrector = DistributionShapeCorrector(
                X_clean,
                strategy=spec["strategy"],
                standardize=spec.get("standardize", False),
                epsilon=self.epsilon,
                verbose=False,
                exclude=None,
            )
            X_new = corrector.transform(y_train=y_new, sensitive_attr_train=sens_new)
            return X_new, y_new, sens_new

        else:
            # DataFrame case
            X_clean_df, y_new, sens_new = _clean_align_df(X, y, sensitive)

            corrector = DistributionShapeCorrector(
                X_clean_df,
                strategy=spec["strategy"],
                standardize=spec.get("standardize", False),
                epsilon=self.epsilon,
                verbose=False,
                exclude=None,
            )
            X_new = corrector.transform(y_train=y_new, sensitive_attr_train=sens_new)
            return X_new, y_new, sens_new
