# modules/sampling/undersampling_allknn.py

import pandas as pd
import numpy as np
import time

try:
    from imblearn.under_sampling import AllKNN
except ImportError:
    AllKNN = None


class AllKNNUndersampler:
    """
    Undersampling using AllKNN.

    Key properties (aligned with your framework):
    - Takes X as a DataFrame (or dict {'train','test'} but only undersamples train).
    - Requires y_train to undersample.
    - Returns (X_new, y_new, sensitive_new) because row count can change.
    - Assumes numeric features (typical after encoding). Raises if non-numeric columns exist.

    Notes on excluded columns:
    - If `exclude` columns are provided, they are removed before AllKNN and then reattached
      using the selected indices returned by the sampler (best-effort).
    """

    def __init__(
        self,
        dataset,
        n_neighbors=3,
        kind_sel="all",     # {"all","mode"} depending on imblearn version
        allow_minority=True,
        sampling_strategy="auto",
        n_jobs=None,
        verbose=False,
        exclude=None,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.n_neighbors = int(n_neighbors)
        self.kind_sel = str(kind_sel).lower().strip()
        self.allow_minority = bool(allow_minority)
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs
        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

        if AllKNN is None:
            raise ImportError(
                "imblearn is required for AllKNN but is not installed. "
                "Install with: pip install imbalanced-learn"
            )

        self._sampler = None

    def _check_numeric(self, X: pd.DataFrame):
        non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
        if len(non_numeric) > 0:
            raise ValueError(
                f"AllKNNUndersampler requires all features to be numeric. "
                f"Non-numeric columns found: {non_numeric}. "
                f"Encode them (e.g., one-hot) before undersampling."
            )

    def _sanitize(self, X: pd.DataFrame):
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isnull().any().any():
            raise ValueError("AllKNNUndersampler: NaNs present in X. Run missing-value handling first.")
        return X

    def _build_sampler(self):
        # AllKNN signature can vary slightly across imblearn versions.
        # We'll set common args and fall back if some aren't supported.
        kwargs = dict(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
        )

        # allow_minority / kind_sel sometimes exist depending on version
        # We'll try to include them; if not supported, we retry without.
        try:
            return AllKNN(allow_minority=self.allow_minority, kind_sel=self.kind_sel, **kwargs)
        except TypeError:
            try:
                return AllKNN(kind_sel=self.kind_sel, **kwargs)
            except TypeError:
                try:
                    return AllKNN(allow_minority=self.allow_minority, **kwargs)
                except TypeError:
                    return AllKNN(**kwargs)

    def _fit_resample_train(self, X_train: pd.DataFrame, y_train, sensitive_attr_train=None):
        X_train = X_train.copy()

        excluded_cols = X_train[self.exclude].copy() if self.exclude else pd.DataFrame(index=X_train.index)
        X_core = X_train.drop(columns=self.exclude, errors="ignore")

        X_core = self._sanitize(X_core)
        self._check_numeric(X_core)

        y_arr = np.asarray(y_train)
        if y_arr.ndim != 1:
            y_arr = y_arr.reshape(-1)

        self._sampler = self._build_sampler()

        X_res, y_res = self._sampler.fit_resample(X_core.values, y_arr)

        # Determine which original samples were kept (best-effort).
        # imblearn samplers often expose `sample_indices_` for under-sampling
        # (or can infer via length change + internal state).
        kept_idx = None
        if hasattr(self._sampler, "sample_indices_"):
            kept_idx = self._sampler.sample_indices_
        elif hasattr(self._sampler, "sample_indices"):
            kept_idx = getattr(self._sampler, "sample_indices")
        # If we can't get indices, we can't reliably reattach excluded cols or sensitive values.
        # We'll still return X_res/y_res; excluded/sensitive will be None.

        # Rebuild X DataFrame
        X_res_df = pd.DataFrame(X_res, columns=X_core.columns)

        # Reattach excluded columns using kept indices (if available)
        if not excluded_cols.empty:
            if kept_idx is None:
                # best-effort fallback: drop excluded and warn via error message
                raise RuntimeError(
                    "AllKNNUndersampler could not recover kept indices to reattach excluded columns. "
                    "Either set exclude=None or upgrade/downgrade imbalanced-learn to a version "
                    "that exposes sample_indices_."
                )
            excl_res = excluded_cols.iloc[kept_idx].reset_index(drop=True)
            X_res_df = pd.concat([X_res_df.reset_index(drop=True), excl_res], axis=1)
            X_res_df = X_res_df[X_train.columns]

        # y as Series
        y_res_ser = pd.Series(y_res)

        # sensitive as Series, using kept indices (if available)
        sens_res_ser = None
        if sensitive_attr_train is not None:
            sens = pd.Series(sensitive_attr_train).reset_index(drop=True)
            if kept_idx is None:
                raise RuntimeError(
                    "AllKNNUndersampler could not recover kept indices to resample sensitive attributes. "
                    "Upgrade/downgrade imbalanced-learn or pass sensitive_attr_train=None."
                )
            sens_res_ser = sens.iloc[kept_idx].reset_index(drop=True)

        return X_res_df, y_res_ser, sens_res_ser

    def transform(self, y_train=None, sensitive_attr_train=None):
        """
        Returns:
          - If dataset is DataFrame: (X_resampled, y_resampled, sensitive_resampled)
          - If dataset is dict {'train','test'}: returns (
                {'train': X_resampled_train, 'test': X_test_unchanged},
                y_resampled, sensitive_resampled
            )

        Note: Only TRAIN is undersampled. TEST is kept unchanged.
        """
        start_time = time.time()
        if self.verbose:
            print("----- Starting AllKNN Undersampling -----")

        if y_train is None:
            raise ValueError("AllKNNUndersampler.transform requires y_train (labels).")

        if isinstance(self.dataset, dict):
            X_train = self.dataset["train"].copy()
            X_test = self.dataset.get("test", None)

            X_res, y_res, s_res = self._fit_resample_train(X_train, y_train, sensitive_attr_train)

            out = {"train": X_res}
            if X_test is not None:
                out["test"] = X_test.copy()

            if self.verbose:
                print(f"AllKNN done in {time.time() - start_time:.2f} seconds. "
                      f"Train size: {len(X_train)} -> {len(X_res)}")

            return out, y_res, s_res

        elif isinstance(self.dataset, pd.DataFrame):
            X_train = self.dataset.copy()
            X_res, y_res, s_res = self._fit_resample_train(X_train, y_train, sensitive_attr_train)

            if self.verbose:
                print(f"AllKNN done in {time.time() - start_time:.2f} seconds. "
                      f"Size: {len(X_train)} -> {len(X_res)}")

            return X_res, y_res, s_res

        else:
            raise TypeError("dataset must be a pandas DataFrame or a dict with keys {'train','test'}.")
