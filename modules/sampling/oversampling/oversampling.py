# modules/sampling/oversampling_smote.py

import pandas as pd
import numpy as np
import time

try:
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    SMOTE = None


class SMOTEOversampler:
    """
    Oversampling using SMOTE (Synthetic Minority Over-sampling Technique).

    Key properties (aligned with your framework):
    - Takes X as a DataFrame (or dict {'train','test'} but only oversamples train).
    - Requires y_train (Series/1D array) to oversample.
    - If you pass sensitive_attr_train, it is oversampled using the SAME indices produced by SMOTE
      (via `sample_indices_` / resample mapping) when possible.

    IMPORTANT:
    - SMOTE needs numeric features. If non-numeric columns exist, this module raises an error.
      (Use encoding before sampling.)
    """

    def __init__(
        self,
        dataset,
        sampling_strategy="auto",
        k_neighbors=5,
        random_state=42,
        verbose=False,
        exclude=None,
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = int(k_neighbors)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

        if SMOTE is None:
            raise ImportError(
                "imblearn is required for SMOTE but is not installed. "
                "Install with: pip install imbalanced-learn"
            )

        self._smote = None

    def _check_numeric(self, X: pd.DataFrame):
        non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
        if len(non_numeric) > 0:
            raise ValueError(
                f"SMOTEOversampler requires all features to be numeric. "
                f"Non-numeric columns found: {non_numeric}. "
                f"Encode them (e.g., one-hot) before SMOTE."
            )

    def _fit_resample_train(self, X_train: pd.DataFrame, y_train, sensitive_attr_train=None):
        X_train = X_train.copy()

        excluded_cols = X_train[self.exclude].copy() if self.exclude else pd.DataFrame(index=X_train.index)
        X_core = X_train.drop(columns=self.exclude, errors="ignore")

        # sanitize
        X_core = X_core.replace([np.inf, -np.inf], np.nan)
        if X_core.isnull().any().any():
            raise ValueError("SMOTEOversampler: NaNs present in X. Run missing-value handling first.")

        self._check_numeric(X_core)

        y_arr = np.asarray(y_train)
        if y_arr.ndim != 1:
            y_arr = y_arr.reshape(-1)

        self._smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )

        X_res, y_res = self._smote.fit_resample(X_core.values, y_arr)

        # Rebuild X as DataFrame
        X_res_df = pd.DataFrame(X_res, columns=X_core.columns)

        # Reattach excluded columns (best-effort):
        # SMOTE synthesizes new rows for core features only. For excluded columns, we:
        # - take original excluded rows aligned to SMOTE sampling indices, then
        # - for synthetic samples: copy excluded values from the base sample used.
        #
        # imblearn stores indices of selected samples for resampling in some versions via `sample_indices_`.
        # If unavailable, we default synthetic excluded values to NaN.
        excl_res_df = None
        if not excluded_cols.empty:
            if hasattr(self._smote, "sample_indices_"):
                base_idx = self._smote.sample_indices_
                # base_idx length equals X_res rows, mapping each resampled row -> an original row index
                excl_vals = excluded_cols.iloc[base_idx].reset_index(drop=True)
                excl_res_df = excl_vals
            else:
                # fallback: keep original excluded for original part; NaN for synthetic
                n_orig = len(excluded_cols)
                n_total = len(X_res_df)
                pad = pd.DataFrame(np.nan, columns=excluded_cols.columns, index=range(n_total))
                pad.iloc[:n_orig, :] = excluded_cols.reset_index(drop=True).values
                excl_res_df = pad

            X_res_df = pd.concat([X_res_df, excl_res_df.reset_index(drop=True)], axis=1)
            # restore original column order
            X_res_df = X_res_df[X_train.columns]

        # Rebuild y as Series
        y_res_ser = pd.Series(y_res)

        # Oversample sensitive attributes (best-effort) using same mapping as excluded columns
        sens_res_ser = None
        if sensitive_attr_train is not None:
            sens = pd.Series(sensitive_attr_train).reset_index(drop=True)
            if hasattr(self._smote, "sample_indices_"):
                base_idx = self._smote.sample_indices_
                sens_res_ser = sens.iloc[base_idx].reset_index(drop=True)
            else:
                # fallback: keep originals; NaN for synthetic (rare)
                n_orig = len(sens)
                n_total = len(y_res_ser)
                sens_res_ser = pd.Series([np.nan] * n_total)
                sens_res_ser.iloc[:n_orig] = sens.values

        return X_res_df, y_res_ser, sens_res_ser

    def transform(self, y_train=None, sensitive_attr_train=None):
        """
        Returns:
          - If dataset is DataFrame: (X_resampled, y_resampled, sensitive_resampled)
          - If dataset is dict {'train','test'}: returns (
                {'train': X_resampled_train, 'test': X_test_unchanged},
                y_resampled, sensitive_resampled
            )

        Note: Only TRAIN is oversampled. TEST is kept unchanged.
        """
        start_time = time.time()
        if self.verbose:
            print("----- Starting SMOTE Oversampling -----")

        if y_train is None:
            raise ValueError("SMOTEOversampler.transform requires y_train (labels).")

        if isinstance(self.dataset, dict):
            X_train = self.dataset["train"].copy()
            X_test = self.dataset.get("test", None)
            X_res, y_res, s_res = self._fit_resample_train(X_train, y_train, sensitive_attr_train)

            out = {"train": X_res}
            if X_test is not None:
                out["test"] = X_test.copy()

            if self.verbose:
                print(f"SMOTE done in {time.time() - start_time:.2f} seconds. "
                      f"Train size: {len(X_train)} -> {len(X_res)}")

            return out, y_res, s_res

        elif isinstance(self.dataset, pd.DataFrame):
            X_train = self.dataset.copy()
            X_res, y_res, s_res = self._fit_resample_train(X_train, y_train, sensitive_attr_train)

            if self.verbose:
                print(f"SMOTE done in {time.time() - start_time:.2f} seconds. "
                      f"Size: {len(X_train)} -> {len(X_res)}")

            return X_res, y_res, s_res

        else:
            raise TypeError("dataset must be a pandas DataFrame or a dict with keys {'train','test'}.")
