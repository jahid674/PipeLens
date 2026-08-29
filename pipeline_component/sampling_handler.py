# =========================
# FILE: pipeline_component/sampling_handler.py
# =========================
from modules.sampling.data_sampling.sampler import DataSampler


class SamplingHandler:
    """
    Handler for Sampling, mirroring other handlers.

    IMPORTANT: PipelineExecutor loads handlers as:
        handler_class(strategy=<int>, config=<dict>)
    so this __init__ MUST accept named args: strategy, config.
    """

    def __init__(self, strategy, config, **kwargs):
        # strategy is 0-based index coming from PipelineExecutor._safe_param_index(...)
        self.strategy_idx = int(strategy)

        # strategies list must be present in shared_config
        self.sampling_strategy = config.get("sampling_strategy", ["full"])

        # --- defaults (kept inside handler) ---
        self.random_frac = float(config.get("sampling_random_frac", 0.30))
        self.snapshot_size = config.get("sampling_snapshot_size", 0.20)  # float fraction or int count
        self.stratify_n_per_group = int(config.get("sampling_stratify_n_per_group", 6000))
        self.random_state = int(config.get("sampling_random_state", 42))

        # --- the ONLY one you asked to keep configurable ---
        # You can set this in sampling_config and merge into shared_config:
        #   sampling_stratify_col: "sensitive" or "y"
        self.stratify_col = config.get("sampling_stratify_col", "sensitive")

    def apply(self, X, y, sensitive=None):
        # choose strategy name from index
        if not self.sampling_strategy:
            strat_name = "full"
        else:
            if self.strategy_idx < 0 or self.strategy_idx >= len(self.sampling_strategy):
                # defensive: out-of-range should have been caught earlier
                strat_name = "full"
            else:
                strat_name = self.sampling_strategy[self.strategy_idx]

        sampler = DataSampler(
            dataset=X,
            strategy=strat_name,
            random_frac=self.random_frac,
            snapshot_size=self.snapshot_size,
            stratify_col=self.stratify_col,
            stratify_n_per_group=self.stratify_n_per_group,
            random_state=self.random_state,
            verbose=False,
        )
        return sampler.transform(y=y, sensitive=sensitive)
