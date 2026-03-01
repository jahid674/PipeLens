import numpy as np
import pandas as pd
import re

class NoiseInjector:
    """
    Noise injection utility for testing cleaning modules.

    Supports (X-only noises):
      - outlier
      - missing
      - invalid_value
      - duplicate_rows
      - floating_point
      - distribution_shape
      - multicollinearity

    Supports (X,y noise):
      - class_imbalance

    Key design:
      - Deterministic via per-instance RNG (np.random.default_rng(seed))
      - inject_noise(...) accepts a single noise_type OR a list of noise_types
      - inject_multiple_noises(...) applies in order and ALWAYS returns (X, y)
      - STRICT: floating_point noise only on numeric columns where there is already a NON-ZERO fractional part
    """

    def __init__(self, pipeline_type, dataset_name, target_variable_name=None, seed: int = 42):
        self.pipeline_type = pipeline_type
        self.dataset_name = dataset_name
        self.target_variable_name = target_variable_name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.outlier_indices = None  # set by inject_outliers

    # -------------------------
    # Helpers
    # -------------------------
    def _numeric_cols(self, X: pd.DataFrame, exclude=None):
        exclude = set(exclude or [])
        cols = X.select_dtypes(include=["int", "float", "number"]).columns.tolist()
        return [c for c in cols if c not in exclude]

    def _string_cols(self, X: pd.DataFrame, exclude=None):
        exclude = set(exclude or [])
        cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
        return [c for c in cols if c not in exclude]

    def _choose_frac_indices(self, X: pd.DataFrame, frac: float, eligible_idx=None):
        frac = float(frac)
        if frac <= 0:
            return np.array([], dtype=int)

        if eligible_idx is None:
            eligible_idx = np.asarray(X.index)
        else:
            eligible_idx = np.asarray(eligible_idx)

        if eligible_idx.size == 0:
            return np.array([], dtype=int)

        n = max(1, int(np.ceil(frac * eligible_idx.size)))
        n = min(n, eligible_idx.size)
        return self.rng.choice(eligible_idx, size=n, replace=False)

    def _ensure_series(self, y):
        if y is None:
            return None, None
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y DataFrame must have exactly one column.")
            return y.iloc[:, 0].copy(), "dataframe"
        if isinstance(y, pd.Series):
            return y.copy(), "series"
        if isinstance(y, np.ndarray):
            return pd.Series(y), "ndarray"
        raise TypeError("y must be a Series, 1-col DataFrame, or ndarray.")

    def _restore_y_type(self, y_series: pd.Series, original_type: str, y_orig):
        if original_type == "dataframe":
            return pd.DataFrame(y_series, columns=y_orig.columns)
        if original_type == "ndarray":
            return y_series.to_numpy()
        return y_series

    # -------------------------
    # Existing noises
    # -------------------------
    def inject_outliers(
        self,
        X: pd.DataFrame,
        frac=0.3,
        multiplier=5.0,
        numeric_col=None,
        only_sensitive_value=None,
        sensitive_col="Sex"
    ):
        """
        Inject IQR-based extreme low outliers into one numeric column.
        """
        X_modified = X.copy()

        # choose numeric column
        if numeric_col is not None:
            col = numeric_col
        else:
            if self.dataset_name == "hmda":
                col = "lien_status"
            elif self.dataset_name == "housing":
                col = "OverallQual"
            else:
                num_cols = self._numeric_cols(X_modified)
                if not num_cols:
                    self.outlier_indices = np.array([], dtype=int)
                    return X_modified
                col = num_cols[0]

        if col not in X_modified.columns:
            self.outlier_indices = np.array([], dtype=int)
            return X_modified
        if not pd.api.types.is_numeric_dtype(X_modified[col]):
            self.outlier_indices = np.array([], dtype=int)
            return X_modified

        # eligible indices
        eligible = X_modified.index
        if only_sensitive_value is not None and sensitive_col in X_modified.columns:
            eligible = X_modified.index[X_modified[sensitive_col] == only_sensitive_value]

        self.outlier_indices = self._choose_frac_indices(X_modified, frac=frac, eligible_idx=eligible)
        if len(self.outlier_indices) == 0:
            return X_modified

        Q1 = X_modified[col].quantile(0.25)
        Q3 = X_modified[col].quantile(0.75)
        IQR = Q3 - Q1
        injected_value = Q1 - float(multiplier) * IQR
        X_modified.loc[self.outlier_indices, col] = injected_value
        return X_modified

    def inject_missing_values(self, X: pd.DataFrame, frac=0.1, col=None):
        """
        Inject NaNs into a column (user-specified or dataset default).
        """
        X_modified = X.copy()

        if col is None:
            if self.dataset_name == "hmda":
                col = "lien_status"
            elif self.dataset_name == "adult":
                col = "Martial_Status"
            elif self.dataset_name == "housing":
                col = "OverallQual"
            else:
                col = X_modified.columns[0] if len(X_modified.columns) else None

        if col is None or col not in X_modified.columns:
            return X_modified

        mv_idx = self._choose_frac_indices(X_modified, frac=frac, eligible_idx=X_modified.index)
        X_modified.loc[mv_idx, col] = np.nan
        return X_modified

    def inject_class_imbalance(self, X, y, use_outlier_indices=True, frac_random=0.2):
        """
        Flip labels at outlier_indices (default) and/or random indices.
        Returns (X, y_modified).
        """
        y_series, original_type = self._ensure_series(y)
        if y_series is None:
            raise ValueError("y is required for class_imbalance.")

        uniq = pd.unique(y_series.dropna())
        if len(uniq) != 2:
            raise ValueError(f"Labels must be binary; got {len(uniq)} unique values: {uniq}")

        a, b = uniq[0], uniq[1]
        flip_map = {a: b, b: a}

        flip_idx = pd.Index([])
        if use_outlier_indices and self.outlier_indices is not None and len(self.outlier_indices) > 0:
            flip_idx = y_series.index.intersection(pd.Index(self.outlier_indices))

        if frac_random and float(frac_random) > 0:
            rnd_idx = self._choose_frac_indices(pd.DataFrame(index=y_series.index), frac=frac_random, eligible_idx=y_series.index)
            flip_idx = flip_idx.union(pd.Index(rnd_idx))

        if len(flip_idx) == 0:
            return X, y

        y_series.loc[flip_idx] = y_series.loc[flip_idx].map(flip_map)
        y_modified = self._restore_y_type(y_series, original_type, y)
        return X, y_modified

    # -------------------------
    # New noises
    # -------------------------
    def inject_invalid_values(
        self,
        X: pd.DataFrame,
        frac=0.05,
        numeric_sentinels=(-999, -9999, 999, 9999),
        string_tokens=("N/A", "na", "unknown", "?", ""),
        frac_numeric_cols=0.5,
        frac_string_cols=0.5,
        exclude=None,
    ):
        """
        Inject invalid tokens that InvalidValueRepair can fix.
        """
        X_mod = X.copy()
        exclude = exclude or []

        num_cols = self._numeric_cols(X_mod, exclude=exclude)
        str_cols = self._string_cols(X_mod, exclude=exclude)

        # pick columns to target
        tgt_num_cols = []
        if num_cols and frac_numeric_cols > 0:
            k = max(1, int(np.ceil(frac_numeric_cols * len(num_cols))))
            tgt_num_cols = self.rng.choice(num_cols, size=min(k, len(num_cols)), replace=False).tolist()

        tgt_str_cols = []
        if str_cols and frac_string_cols > 0:
            k = max(1, int(np.ceil(frac_string_cols * len(str_cols))))
            tgt_str_cols = self.rng.choice(str_cols, size=min(k, len(str_cols)), replace=False).tolist()

        for c in tgt_num_cols:
            idx = self._choose_frac_indices(X_mod, frac=frac, eligible_idx=X_mod.index)
            if len(idx) == 0:
                continue
            sentinel = self.rng.choice(list(numeric_sentinels))
            X_mod.loc[idx, c] = sentinel

        for c in tgt_str_cols:
            idx = self._choose_frac_indices(X_mod, frac=frac, eligible_idx=X_mod.index)
            if len(idx) == 0:
                continue
            token = self.rng.choice(list(string_tokens))
            X_mod[c] = X_mod[c].astype("object")
            X_mod.loc[idx, c] = token

        return X_mod

    def inject_duplicate_rows(self, X: pd.DataFrame, y=None, frac=0.05, replace=False):
        """
        Append duplicates of random rows (tests deduplication).
        IMPORTANT: If y is provided, duplicates y using the SAME sampled indices.
        Returns:
        - if y is None: X_out
        - else: (X_out, y_out)
        """
        X_mod = X.copy()
        frac = float(frac)

        if frac <= 0 or len(X_mod) == 0:
            return (X_mod, y) if y is not None else X_mod

        n_add = max(1, int(np.ceil(frac * len(X_mod))))
        n_add = min(n_add, len(X_mod)) if not replace else n_add

        dup_idx = self.rng.choice(X_mod.index.to_numpy(), size=n_add, replace=replace)

        # duplicate X
        dup_rows = X_mod.loc[dup_idx].copy()
        X_out = pd.concat([X_mod, dup_rows], axis=0, ignore_index=True)

        # duplicate y (if provided)
        if y is None:
            return X_out

        y_series, original_type = self._ensure_series(y)
        dup_y = y_series.loc[dup_idx].copy()
        y_out_series = pd.concat([y_series, dup_y], axis=0, ignore_index=True)
        y_out = self._restore_y_type(y_out_series, original_type, y)

        return X_out, y_out


    def inject_floating_point_noise(
        self,
        X: pd.DataFrame,
        noise_scale: float = 1e-9,
        frac_rows: float = 0.3,
        exclude=None,
        tol: float = 1e-12
    ):
        """
        STRICT:
        - Only modifies entries that ALREADY have non-zero fractional part.
        - So it will NOT touch integer-valued columns like Sex (0/1).
        """
        X_mod = X.copy()
        exclude_set = set(exclude or [])

        # numeric candidates (int/float). We'll filter by fractional part per-entry.
        num_cols = X_mod.select_dtypes(include=["number"]).columns.tolist()
        num_cols = [c for c in num_cols if c not in exclude_set]

        for c in num_cols:
            s = pd.to_numeric(X_mod[c], errors="coerce")
            vals = s.to_numpy(dtype=float, copy=True)

            finite = np.isfinite(vals)
            if not finite.any():
                continue

            frac_mask = np.abs(vals - np.rint(vals)) > float(tol)
            eligible = np.where(finite & frac_mask)[0]
            if eligible.size == 0:
                continue  # nothing fractional here

            k = max(1, int(np.ceil(float(frac_rows) * eligible.size)))
            chosen = self.rng.choice(eligible, size=min(k, eligible.size), replace=False)
            vals[chosen] = vals[chosen] + self.rng.normal(0.0, float(noise_scale), size=chosen.size)

            X_mod[c] = vals

        return X_mod

    def inject_distribution_shape_noise(
        self,
        X: pd.DataFrame,
        frac_cols=0.5,
        frac_rows=1.0,
        mode="skew_right",
        strength=2.0,
        exclude=None
    ):
        """
        Distort distribution shape of numeric columns.
        Also supports targeting only a fraction of rows to make it easier to see.

        mode:
          - skew_right
          - skew_left
          - heavy_tail
        """
        X_mod = X.copy()
        exclude = exclude or []
        num_cols = self._numeric_cols(X_mod, exclude=exclude)
        if not num_cols:
            return X_mod

        k = max(1, int(np.ceil(float(frac_cols) * len(num_cols))))
        tgt_cols = self.rng.choice(num_cols, size=min(k, len(num_cols)), replace=False)

        mode = str(mode).lower().strip()
        strength = float(strength)

        for c in tgt_cols:
            col = pd.to_numeric(X_mod[c], errors="coerce").astype(float)

            # choose rows to modify (so you can see effect without fully overwriting)
            row_idx = self._choose_frac_indices(X_mod, frac=frac_rows, eligible_idx=X_mod.index)
            if len(row_idx) == 0:
                continue

            mu = np.nanmean(col.loc[row_idx])
            sd = np.nanstd(col.loc[row_idx])
            if not np.isfinite(sd) or sd == 0:
                continue

            z = (col.loc[row_idx] - mu) / sd

            if mode == "skew_right":
                out = np.exp(strength * z)
            elif mode == "skew_left":
                out = -np.exp(strength * z)
            elif mode == "heavy_tail":
                out = z * np.exp(np.abs(z) / max(1e-6, strength))
            else:
                raise ValueError("mode must be one of {'skew_right','skew_left','heavy_tail'}")

            # normalize then rescale to original magnitude range on affected rows
            out = (out - np.nanmean(out)) / (np.nanstd(out) + 1e-12)
            out = out * sd + mu

            X_mod.loc[row_idx, c] = out

        return X_mod

    def inject_multicollinearity(self, X: pd.DataFrame, frac_pairs=0.3, noise_std=1e-5, exclude=None):
        """
        Create multicollinearity by making some numeric columns near-linear copies of others.
        """
        X_mod = X.copy()
        exclude = exclude or []
        num_cols = self._numeric_cols(X_mod, exclude=exclude)
        if len(num_cols) < 2:
            return X_mod

        m = len(num_cols)
        n_pairs = max(1, int(np.ceil(float(frac_pairs) * (m // 2))))
        n_pairs = min(n_pairs, m // 2)

        cols_shuffled = self.rng.permutation(num_cols)
        pairs = [(cols_shuffled[2*i], cols_shuffled[2*i+1]) for i in range(n_pairs)]

        for src, dst in pairs:
            src_col = pd.to_numeric(X_mod[src], errors="coerce")
            if src_col.isna().all():
                continue
            eps = self.rng.normal(0.0, float(noise_std), size=len(src_col))
            X_mod[dst] = 1.0 * src_col + eps

        return X_mod

    # -------------------------
    # Main entry points
    # -------------------------
    def inject_noise(self, X, y=None, noise_type="outlier", **kwargs):
        """
        Single-noise interface.

        Returns:
          - for class_imbalance: (X, y_modified)
          - otherwise: X_modified
        """
        noise_type = str(noise_type).lower().strip()

        if noise_type == "outlier":
            return self.inject_outliers(X, **kwargs)
        if noise_type == "missing":
            return self.inject_missing_values(X, **kwargs)
        if noise_type == "invalid_value":
            return self.inject_invalid_values(X, **kwargs)
        if noise_type == "duplicate_rows":
            return self.inject_duplicate_rows(X, y=y, **kwargs)
        if noise_type == "floating_point":
            return self.inject_floating_point_noise(X, **kwargs)
        if noise_type == "distribution_shape":
            return self.inject_distribution_shape_noise(X, **kwargs)
        if noise_type == "multicollinearity":
            return self.inject_multicollinearity(X, **kwargs)
        if noise_type == "class_imbalance":
            if y is None:
                raise ValueError("y is required for class_imbalance.")
            return self.inject_class_imbalance(X, y, **kwargs)

        raise ValueError(f"Unsupported noise_type '{noise_type}'.")

    def inject_multiple_noises(self, X, y=None, noise_types=None, noise_params=None):
        """
        Apply multiple noise types sequentially.
        Always returns (X_noisy, y_noisy).
        """
        if noise_types is None or not isinstance(noise_types, (list, tuple)) or len(noise_types) == 0:
            raise ValueError("noise_types must be a non-empty list/tuple of noise names.")

        noise_params = noise_params or {}
        X_mod = X.copy()
        y_mod = y

        for nt in noise_types:
            nt = str(nt).lower().strip()
            kwargs = noise_params.get(nt, {})

            if nt == "class_imbalance":
                if y_mod is None:
                    raise ValueError("y is required for class_imbalance.")
                X_mod, y_mod = self.inject_class_imbalance(X_mod, y_mod, **kwargs)
            else:
                out = self.inject_noise(X_mod, y=y_mod, noise_type=nt, **kwargs)
                if isinstance(out, tuple) and len(out) == 2:
                    X_mod, y_mod = out
                else:
                    X_mod = out


        return X_mod, y_mod

    def inject_noise_flexible(self, X, y=None, noise_type="outlier", noise_params=None, **kwargs):
        """
        Accept either a single string or a list.
        - if noise_type is str -> uses inject_noise
        - if noise_type is list -> uses inject_multiple_noises

        Returns:
          - if single str:
              * class_imbalance -> (X,y)
              * else -> X
          - if list -> (X,y)
        """
        if isinstance(noise_type, (list, tuple)):
            return self.inject_multiple_noises(X, y=y, noise_types=list(noise_type), noise_params=noise_params)
        return self.inject_noise(X, y=y, noise_type=noise_type, **(noise_params or {}), **kwargs)


# # -------------------------
# # Example usage
# # -------------------------
# if __name__ == "__main__":
#     df = pd.DataFrame({
#         "A": np.linspace(0, 1, 20),                # float, has fractional
#         "B": np.linspace(1, 2, 20) + 1e-6,         # float, has fractional
#         "Sex": [0, 1] * 10,                        # int, should never be affected by floating noise
#         "Martial_Status": ["married", "single"] * 10
#     })
#     y = pd.Series([0, 1] * 10, name="label")

#     inj = NoiseInjector(pipeline_type="ml", dataset_name="adult", seed=42)

#     # Example 1: distribution shape (clear demo)
#     noise_types = ["distribution_shape"]
#     noise_params = {
#         "distribution_shape": {
#             "frac_cols": 1.0,        # distort all numeric cols (A,B,Sex is numeric but warp applies via numeric_cols;
#                                     # if you want to exclude Sex, add exclude=["Sex"]
#             "frac_rows": 1.0,        # distort all rows for visibility
#             "mode": "skew_right",
#             "strength": 2.0,
#             "exclude": ["Sex"],      # IMPORTANT: keep Sex safe for shape noise too
#         }
#     }

#     X_noisy, y_noisy = inj.inject_multiple_noises(df, y=y, noise_types=noise_types, noise_params=noise_params)

#     print("=== Distribution shape noise injected ===")
#     print("Original A (rounded):", df["A"].round(3).tolist())
#     print("Noisy    A (rounded):", X_noisy["A"].round(3).tolist())
#     print("Sex unchanged? ->", (df["Sex"].values == X_noisy["Sex"].values).all())

#     # Example 2: combine multiple noises including strict floating point noise
#     noise_types2 = ["invalid_value", "missing", "floating_point", "multicollinearity", "duplicate_rows"]
#     noise_params2 = {
#         "invalid_value": {"frac": 0.10},
#         "missing": {"frac": 0.10, "col": "Martial_Status"},
#         "floating_point": {
#             "noise_scale": 1e-6,
#             "frac_rows": 0.50
#         },
#         "multicollinearity": {"frac_pairs": 0.5, "noise_std": 1e-6, "exclude": ["Sex"]},
#         "duplicate_rows": {"frac": 0.5},
#     }

#     X_noisy2, y_noisy2 = inj.inject_multiple_noises(df, y=y, noise_types=noise_types2, noise_params=noise_params2)

#     print("\n=== Multiple noises injected ===")
#     print(X_noisy2)
#     print("Rows:", len(df), "->", len(X_noisy2))
#     print("Sex column still integer values? ->", set(pd.to_numeric(X_noisy2["Sex"], errors="coerce").dropna().unique()))
