# modules/encoding/encoder_auto.py

import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import Binarizer, KBinsDiscretizer, LabelBinarizer, LabelEncoder, OneHotEncoder


class EncoderModuleAuto:
    """
    Encoding / discretization module with multiple strategies (auto target cols):

      - "binarizer"          : applies to numeric columns
      - "kbins"              : applies to numeric columns
      - "onehot"             : applies to categorical columns (auto-detected)
      - "label_encoder"      : applies LabelEncoder to each categorical column (auto-detected)
      - "label_binarizer"    : applies LabelBinarizer to each categorical column (auto-detected)

    Auto-detection rule (default):
      - categorical cols = non-numeric dtypes
      - (optional) treat low-cardinality numeric columns as categorical via `cat_max_unique`

    Accepts either:
      - DataFrame, or
      - dict {'train','test'} (fit on train, transform both)

    Returns:
      - transformed X (same container type as input)
      - does not change rows => y and sensitive unchanged
    """

    def __init__(
        self,
        dataset,
        strategy="onehot",
        # auto column detection
        cat_max_unique=None,        # e.g., 20 => numeric col with <=20 uniques treated categorical
        cat_min_unique=None,        # optional, e.g., 2
        keep_other_cols=True,
        verbose=False,
        exclude=None,

        # Binarizer
        threshold=0.0,

        # KBinsDiscretizer
        n_bins=5,
        kbins_encode="onehot-dense",     # {"onehot-dense","onehot","ordinal"}
        kbins_strategy="quantile",       # {"uniform","quantile","kmeans"}

        # OneHotEncoder
        drop=None,
        handle_unknown="ignore",
        min_frequency=None,
        max_categories=None,
        sparse_output=False,

        # LabelBinarizer
        lb_drop_first=True,              # drop first class to reduce redundancy (optional)
    ):
        self.dataset = dataset.copy() if isinstance(dataset, dict) else dataset.copy()
        self.strategy = str(strategy).lower().strip()

        self.cat_max_unique = cat_max_unique
        self.cat_min_unique = cat_min_unique
        self.keep_other_cols = bool(keep_other_cols)
        self.verbose = bool(verbose)
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []

        # params
        self.threshold = float(threshold)
        self.n_bins = int(n_bins)
        self.kbins_encode = str(kbins_encode)
        self.kbins_strategy = str(kbins_strategy)

        self.drop = drop
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.sparse_output = bool(sparse_output)

        self.lb_drop_first = bool(lb_drop_first)

        # fitted artifacts
        self._model = None                     # for onehot/kbins/binarizer
        self._models_per_col = {}              # for label_encoder / label_binarizer
        self._mode_cols_ = None
        self._feature_names_ = None

        # learned column sets (from train)
        self._num_cols_ = None
        self._cat_cols_ = None

    def _split_cols(self, df: pd.DataFrame):
        df_work = df.copy()

        excluded_cols = df_work[self.exclude].copy() if self.exclude else pd.DataFrame(index=df_work.index)
        df_work = df_work.drop(columns=self.exclude, errors="ignore")

        num_cols = df_work.select_dtypes(include=["number"]).columns.tolist()
        non_num_cols = [c for c in df_work.columns if c not in num_cols]

        # optionally treat low-cardinality numeric columns as categorical
        cat_cols = list(non_num_cols)
        if self.cat_max_unique is not None:
            for c in num_cols:
                nunique = df_work[c].nunique(dropna=True)
                if nunique <= int(self.cat_max_unique) and (self.cat_min_unique is None or nunique >= int(self.cat_min_unique)):
                    cat_cols.append(c)

        # final categorical and numeric partitions (avoid overlap)
        cat_cols = sorted(list(dict.fromkeys(cat_cols)))
        num_cols_final = [c for c in num_cols if c not in cat_cols]

        return df_work, excluded_cols, num_cols_final, cat_cols

    def _sanitize_numeric(self, X_num: pd.DataFrame):
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        if X_num.isnull().any().any():
            raise ValueError("EncoderModuleAuto: NaNs in numeric cols. Run missing-value handling first.")
        return X_num

    def _build_onehot(self):
        try:
            return OneHotEncoder(
                drop=self.drop,
                handle_unknown=self.handle_unknown,
                min_frequency=self.min_frequency,
                max_categories=self.max_categories,
                sparse_output=self.sparse_output,
            )
        except TypeError:
            return OneHotEncoder(
                drop=self.drop,
                handle_unknown=self.handle_unknown,
                min_frequency=self.min_frequency,
                max_categories=self.max_categories,
                sparse=bool(self.sparse_output),
            )

    def _fit(self, X_train: pd.DataFrame):
        df_work, _, num_cols, cat_cols = self._split_cols(X_train)
        self._num_cols_, self._cat_cols_ = num_cols, cat_cols

        s = self.strategy

        # ---------- Numeric transforms ----------
        if s in ("binarizer", "bin"):
            self._mode_cols_ = num_cols
            if len(num_cols) == 0:
                self._model = None
                return
            self._model = Binarizer(threshold=self.threshold)
            X_num = self._sanitize_numeric(df_work[num_cols])
            self._model.fit(X_num.values)

        elif s in ("kbins", "kbinsdiscretizer", "discretizer"):
            self._mode_cols_ = num_cols
            if len(num_cols) == 0:
                self._model = None
                return
            self._model = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode=self.kbins_encode,
                strategy=self.kbins_strategy,
            )
            X_num = self._sanitize_numeric(df_work[num_cols])
            self._model.fit(X_num.values)

            if hasattr(self._model, "get_feature_names_out"):
                try:
                    self._feature_names_ = self._model.get_feature_names_out(num_cols)
                except Exception:
                    self._feature_names_ = None

        # ---------- Categorical transforms ----------
        elif s in ("onehot", "onehotencoder", "ohe"):
            self._mode_cols_ = cat_cols
            if len(cat_cols) == 0:
                self._model = None
                return
            self._model = self._build_onehot()
            X_cat = df_work[cat_cols].astype(str)
            self._model.fit(X_cat)

            if hasattr(self._model, "get_feature_names_out"):
                self._feature_names_ = self._model.get_feature_names_out(cat_cols)

        elif s in ("label_encoder", "labelencoder", "le"):
            self._mode_cols_ = cat_cols
            if len(cat_cols) == 0:
                return
            # fit one LabelEncoder per col
            self._models_per_col = {}
            for c in cat_cols:
                le = LabelEncoder()
                le.fit(df_work[c].astype(str).values)
                self._models_per_col[c] = le

        elif s in ("label_binarizer", "labelbinarizer", "lb"):
            self._mode_cols_ = cat_cols
            if len(cat_cols) == 0:
                return
            self._models_per_col = {}
            for c in cat_cols:
                lb = LabelBinarizer()
                lb.fit(df_work[c].astype(str).values)
                self._models_per_col[c] = lb

        else:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. Choose from "
                f"{{'binarizer','kbins','onehot','label_encoder','label_binarizer'}}."
            )

    def _transform_one(self, X: pd.DataFrame) -> pd.DataFrame:
        df_work, excluded_cols, num_cols, cat_cols = self._split_cols(X)

        # Use train-learned column sets to avoid train/test drift
        num_cols = self._num_cols_ if self._num_cols_ is not None else num_cols
        cat_cols = self._cat_cols_ if self._cat_cols_ is not None else cat_cols

        s = self.strategy

        kept_df = df_work.drop(columns=(num_cols if s in ("binarizer","bin","kbins","kbinsdiscretizer","discretizer") else cat_cols),
                              errors="ignore") if self.keep_other_cols else pd.DataFrame(index=df_work.index)

        # ---------- Numeric ----------
        if s in ("binarizer", "bin"):
            if self._model is None or len(num_cols) == 0:
                out_df = df_work
            else:
                X_num = self._sanitize_numeric(df_work[num_cols])
                arr = self._model.transform(X_num.values)
                out_enc = pd.DataFrame(arr, columns=[f"bin_{c}" for c in num_cols], index=df_work.index)
                out_df = pd.concat([kept_df, out_enc], axis=1)

        elif s in ("kbins", "kbinsdiscretizer", "discretizer"):
            if self._model is None or len(num_cols) == 0:
                out_df = df_work
            else:
                X_num = self._sanitize_numeric(df_work[num_cols])
                arr = self._model.transform(X_num.values)
                if hasattr(arr, "toarray"):
                    arr = arr.toarray()

                cols = list(self._feature_names_) if (self._feature_names_ is not None and len(self._feature_names_) == arr.shape[1]) \
                    else [f"kbins_{i}" for i in range(arr.shape[1])]
                out_enc = pd.DataFrame(arr, columns=cols, index=df_work.index)
                out_df = pd.concat([kept_df, out_enc], axis=1)

        # ---------- OneHot ----------
        elif s in ("onehot", "onehotencoder", "ohe"):
            if self._model is None or len(cat_cols) == 0:
                out_df = df_work
            else:
                X_cat = df_work[cat_cols].astype(str)
                arr = self._model.transform(X_cat)
                if hasattr(arr, "toarray"):
                    arr = arr.toarray()
                cols = list(self._feature_names_) if self._feature_names_ is not None else [f"ohe_{i}" for i in range(arr.shape[1])]
                out_enc = pd.DataFrame(arr, columns=cols, index=df_work.index)
                out_df = pd.concat([kept_df, out_enc], axis=1)

        # ---------- LabelEncoder (multi-col) ----------
        elif s in ("label_encoder", "labelencoder", "le"):
            if len(cat_cols) == 0:
                out_df = df_work
            else:
                enc_parts = []
                for c in cat_cols:
                    le = self._models_per_col.get(c)
                    if le is None:
                        # unseen column (shouldn't happen if train/test consistent) => passthrough
                        enc_parts.append(df_work[[c]].copy())
                        continue
                    vals = df_work[c].astype(str).values
                    # handle unseen labels in test: map to -1
                    mapping = {cls: i for i, cls in enumerate(le.classes_)}
                    enc = np.array([mapping.get(v, -1) for v in vals], dtype=int)
                    enc_parts.append(pd.DataFrame(enc, columns=[f"{c}__le"], index=df_work.index))
                out_enc = pd.concat(enc_parts, axis=1)
                out_df = pd.concat([kept_df, out_enc], axis=1)

        # ---------- LabelBinarizer (multi-col) ----------
        elif s in ("label_binarizer", "labelbinarizer", "lb"):
            if len(cat_cols) == 0:
                out_df = df_work
            else:
                enc_parts = []
                for c in cat_cols:
                    lb = self._models_per_col.get(c)
                    if lb is None:
                        enc_parts.append(df_work[[c]].copy())
                        continue
                    vals = df_work[c].astype(str).values
                    arr = lb.transform(vals)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)

                    classes = getattr(lb, "classes_", None)
                    if classes is None:
                        cols = [f"{c}__lb_{i}" for i in range(arr.shape[1])]
                    else:
                        cols = [f"{c}__{cls}" for cls in classes]

                    # optional: drop first to reduce redundancy
                    if self.lb_drop_first and arr.shape[1] > 1:
                        arr = arr[:, 1:]
                        cols = cols[1:]

                    enc_parts.append(pd.DataFrame(arr, columns=cols, index=df_work.index))

                out_enc = pd.concat(enc_parts, axis=1)
                out_df = pd.concat([kept_df, out_enc], axis=1)

        else:
            raise RuntimeError("Unexpected strategy.")

        # Reattach excluded columns
        if not excluded_cols.empty:
            out_df = pd.concat([out_df, excluded_cols], axis=1)

        return out_df

    def transform(self, y_train=None, sensitive_attr_train=None):
        start_time = time.time()
        if self.verbose:
            print("----- Starting Auto Encoding Module -----")

        if isinstance(self.dataset, dict):
            self._fit(self.dataset["train"].copy())
            out = {"train": self._transform_one(self.dataset["train"].copy())}
            if "test" in self.dataset and self.dataset["test"] is not None:
                out["test"] = self._transform_one(self.dataset["test"].copy())

        elif isinstance(self.dataset, pd.DataFrame):
            self._fit(self.dataset.copy())
            out = self._transform_one(self.dataset.copy())
        else:
            raise TypeError("dataset must be a pandas DataFrame or a dict with keys {'train','test'}.")

        if self.verbose:
            print(f"Encoding completed in {time.time() - start_time:.2f} seconds.")

        return out
