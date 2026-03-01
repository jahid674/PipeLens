import os
import pandas as pd
import numpy as np
import logging
import random
import math
import importlib
import itertools
import time

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import ElasticNetCV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from regression import Regression
from LoadDataset import LoadDataset
from noise_injection import NoiseInjector
from modules.profiling.profile import Profile
from modules.data_preparation.outlier_detection.outlier_detector import OutlierDetector
from pipeline_component.swapping_handler import SwapHandler
from similarity_metric import compute_similarity

np.random.seed(42)


class PipelineExecutor:
    def __init__(self, pipeline_type, dataset_name, metric_type, pipeline_ord,
                 execution_type='pass', h_sample_frac=None, scalability_bool=False):

        self.pipeline_type = pipeline_type  # 'ml' or 'em'
        self.dataset_name = dataset_name
        self.metric_type = metric_type
        self.pipeline_order = pipeline_ord  # always keep model at the end
        self.execution_type = execution_type  # 'pass' or 'fail'
        self.h_sample_frac = h_sample_frac
        self.scalability_bool = scalability_bool
        self.utiliy_threshold = 190

        if self.pipeline_type == 'ml':

            self.mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
            self.norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm']
            self.od_strategy = ['none', 'if', 'iqr', 'lof']
            self.fselection_strategy = ['none','va']
            self.sampling_strategy = ['full', 'random', 'stratified']
            self.swapping_strategy = [i for i in range(len(self.pipeline_order)-2)]
            self.model_selection = ['lr']  # rf, 'reg' # 'nb' #nn
            self.knn_k_lst = [1, 5, 10, 20, 30]
            self.lof_k_lst = [1, 5, 10, 20, 30]
            self.distribution_shape_strategy = ['none','log1p', 'sqrt', 'yeojohnson']
            self.floating_point_strategy = ["none","snap","round","both"]
            self.invalid_value_strategy = ['none', 'sentinel', 'regex', 'both']
            self.multicollinearity_strategy = ['none', 'drop_high_vif']
            self.nonlinear_transform_strategy = ["none", "quantile"]# "power"]
            self.poly_pca_strategy = ["none", "pca", "sparsepca", "minibatchsparsepca", "kernelpca"]
            self.sampling_strategy = ['full', 'random', 'snapshot', 'stratified']


            # ---- RAG strategies (base pipeline) ----
            self.query_normalizer_strategy = ["none", "lower+strip_punct", "spellfix_light"]

            # Retriever strategies encode (type + topk) in ONE list (since your framework uses single strategy index)
            self.retriever_strategy = [
                "bm25@10", "bm25@20",
                "dense@10", "dense@20",
                "hybrid_rrf@10", "hybrid_rrf@20"
            ]

            self.reranker_strategy = ["none", "cross_encoder@20", "cross_encoder@50"]

            self.context_builder_strategy = [
                "fixed_chunk_top@1500", "fixed_chunk_top@3000",
                "mmr_diverse@1500", "mmr_diverse@3000"
            ]

            self.generator_strategy = ["grounded_concise", "grounded_detailed"]

            # ---- insertion space ----
            self.translator_rewriter_strategy = [
                "mt_generic|fr->en|literal"
            ]

            self.lang_retrieval_wrapper_strategy = [
                "use_multilingual_dense|rrf"
            ]




            # text strategies
            self.pr_strategy = ['none','pr']
            self.lowercase_strategy = ['none','lc']
            self.spellcheck_strategy = ['none','sc']
            self.whitespace_strategy = ['none','wc']
            self.unit_converter_strategy = ['none','uc']
            self.tokenization_strategy = ['none','whitespace', 'nltk']
            self.stopword_strategy = ['none','sw']
            self.specialchar_strategy = ['none','sc']
            self.deduplication_strategy = ['none','dd']
            

            # data
            loader = LoadDataset(self.dataset_name)
            self.dataset, self.X_train, self.y_train, self.X_test, self.y_test = loader.load()
            self.h_sample_bool = False
            self.h_sample = None
            if h_sample_frac is not None and 0 < h_sample_frac < 1.0:
                self.h_sample = h_sample_frac
                self.h_sample_bool = True
                if self.execution_type == 'pass':
                    sample_idx = random.sample(list(range(len(self.X_train))), math.ceil(self.h_sample * len(self.X_train)))
                    self.X_train = self.X_train.iloc[sample_idx].reset_index(drop=True)
                    self.y_train = self.y_train.iloc[sample_idx].reset_index(drop=True)
                else:
                    sample_idx = random.sample(list(range(len(self.X_test))), math.ceil(self.h_sample * len(self.X_test)))
                    self.X_test = self.X_test.iloc[sample_idx].reset_index(drop=True)
                    self.y_test = self.y_test.iloc[sample_idx].reset_index(drop=True)

            self.sensitive_var = loader.get_sensitive_variable()
            self.target_variable_name = self.y_train.name
            self.noise_injector = NoiseInjector(pipeline_type="ml", dataset_name=self.dataset_name, seed=42)

            # strategy counts (for search space sizes)
            self.strategy_counts = {
                'missing_value': len(self.mv_strategy) + len(self.knn_k_lst) - 1 if 'knn' in self.mv_strategy else len(self.mv_strategy),
                'normalization': len(self.norm_strategy),
                'outlier': len(self.od_strategy) + len(self.lof_k_lst) - 1 if 'lof' in self.od_strategy else len(self.od_strategy),
                'model': len(self.model_selection),
                'fselection': len(self.fselection_strategy),
                'sampling': len(self.sampling_strategy),

                # text
                'punctuation': len(self.pr_strategy),
                'whitespace': len(self.whitespace_strategy),
                'unit_converter': len(self.unit_converter_strategy),
                'tokenizer': len(self.tokenization_strategy),
                'lowercase': len(self.lowercase_strategy),
                'stopword': len(self.stopword_strategy),
                'spell_checker': len(self.spellcheck_strategy),
                'special_character': len(self.specialchar_strategy),
                'deduplication': len(self.deduplication_strategy),
                'swapping': len(self.swapping_strategy),
                'distribution_shape': len(self.distribution_shape_strategy),
                'floating_point': len(self.floating_point_strategy),
                'invalid_value': len(self.invalid_value_strategy),
                'multicollinearity': len(self.multicollinearity_strategy),
                'nonlinear_transform': len(self.nonlinear_transform_strategy),
                'poly_pca': len(self.poly_pca_strategy),
                'sampling': len(self.sampling_strategy),

                'query_normalize': len(self.query_normalizer_strategy),
                'retriever': len(self.retriever_strategy),
                'reranker': len(self.reranker_strategy),
                'context_builder': len(self.context_builder_strategy),
                'generator': len(self.generator_strategy),
                'translator_rewriter': len(self.translator_rewriter_strategy),
                'lang_retrieval_wrapper': len(self.lang_retrieval_wrapper_strategy),

            }

            # dataset-specific thresholds
            if dataset_name == 'adult':
                if self.execution_type == 'pass':
                    self.tau = 0.1
                    self.contamination = 0.2
                    self.contamination_lof = 'auto'
                else:
                    self.tau = 0.1
                    self.contamination = 0.2
                    self.contamination_lof = 'auto'
            elif dataset_name == 'hmda':
                if self.execution_type == 'pass':
                    self.tau = 0.2
                    self.contamination = 0.3
                    self.contamination_lof = 0.3
                else:
                    self.tau = 0.2
                    self.contamination = 0.2
                    self.contamination_lof = 0.2
            elif dataset_name == 'housing':
                if self.execution_type == 'pass':
                    self.tau = 0.2
                    self.contamination = 0.3
                    self.contamination_lof = 0.3
                else:
                    self.tau = 0.1
                    self.contamination = 0.2
                    self.contamination_lof = 0.2
            else:
                raise ValueError("Invalid dataset name for ML pipeline. Supported datasets are: 'adult', 'hmda', 'housing'.")

            # shared config passed into handlers
            self.shared_config = {
                'mv_strategy': self.mv_strategy,
                'knn_k_lst': self.knn_k_lst,
                'norm_strategy': self.norm_strategy,
                'od_strategy': self.od_strategy,
                "fselection_stategy":self.fselection_strategy,
                'lof_k_lst': self.lof_k_lst,
                'contamination': self.contamination,
                'contamination_lof': self.contamination_lof,
                'model_selection': self.model_selection,
                'metric_type': self.metric_type,
                'sensitive_var': self.sensitive_var,
                'target_var': self.target_variable_name,
                #structured
                'distribution_shape_strategy': self.distribution_shape_strategy,
                # text
                'punctuation_strategy': self.pr_strategy,
                'whitespace_strategy': self.whitespace_strategy,
                'unit_converter_strategy': self.unit_converter_strategy,
                'tokenization_strategy': self.tokenization_strategy,
                'lowercase_strategy': self.lowercase_strategy,
                'stopword_strategy': self.stopword_strategy,
                'spellchecker_strategy': self.spellcheck_strategy,
                'specialchar_strategy': self.specialchar_strategy,
                'deduplication_strategy': self.deduplication_strategy,
                'swapping_strategy': self.swapping_strategy,
                'sampling_strategy': self.sampling_strategy,
                'floating_point_strategy': self.floating_point_strategy,
                'invalid_value_strategy': self.invalid_value_strategy,
                'multicollinearity_strategy': self.multicollinearity_strategy,
                'nonlinear_transform_strategy': self.nonlinear_transform_strategy,
                'poly_pca_strategy': self.poly_pca_strategy,
                'sampling_strategy': self.sampling_strategy,
                #
                'query_normalize_strategy': self.query_normalizer_strategy,
                'retriever_strategy': self.retriever_strategy,
                'reranker_strategy': self.reranker_strategy,
                'context_builder_strategy': self.context_builder_strategy,
                'generator_strategy': self.generator_strategy,
                'translator_rewriter_strategy': self.translator_rewriter_strategy,
                'lang_retrieval_wrapper_strategy': self.lang_retrieval_wrapper_strategy,

                # column convention
                'rag_query_col': 'query',
                'rag_context_col': 'context',


            }
        elif self.pipeline_type == 'em':
            print('pipeline_type is em')

        self.noise_types = ["invalid_value", "missing", "floating_point", "class_imbalance", "multicollinearity"]
        self.noise_params = {
            "invalid_value": {"frac": 0.10},
            "missing": {"frac": 0.10, "col": None},
            "outlier": {"frac": 0.20},
            "class_imbalance": {"frac_random": 0.2},
            # "floating_point": {
            #     "noise_scale": 1e-6,
            #     "frac_rows": 0.50
            # },
            # "multicollinearity": {"frac_pairs": 0.5, "noise_std": 1e-6, "exclude": ["Sex"]},
            "duplicate_rows": {"frac": 0.15},
        }
        

    # ---------- small helpers ----------
    def set_dataset(self, dataset):
        self.numerical_columns = dataset.select_dtypes(include=['int', 'float']).columns
        self.categorical_columns = dataset.select_dtypes(include=['object']).columns

    def getIdxSensitive(self, df, sensitive_var):
        priv_idx = df.index[df[sensitive_var] == 1]
        unpriv_idx = df.index[df[sensitive_var] == 0]
        sensitive_attr = df[sensitive_var]
        return priv_idx, unpriv_idx, sensitive_attr

    def _load_handler(self, step, strategy_index):
        module_name = f"pipeline_component.{step}_handler"
        class_name = ''.join(w.capitalize() for w in step.split('_')) + "Handler"
        try:
            handler_module = importlib.import_module(module_name)
            handler_class = getattr(handler_module, class_name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Error loading handler for '{step}': {e}")
        return handler_class(strategy=strategy_index, config=self.shared_config)

    def _safe_param_index(self, step: str, raw_value: int) -> int:
        """
        STRICT index conversion:
        - Accepts 1-based (1..n) and converts to 0-based (0..n-1)
        - Accepts 0-based already (0..n-1)
        - Raises if out of range
        """
        n = self.strategy_counts.get(step)
        if n is None or n <= 0:
            raise ValueError(f"Unknown or empty strategy space for step '{step}'")

        v = int(raw_value)

        # user gives 1-based
        if 1 <= v <= n:
            return v - 1

        # already 0-based
        if 0 <= v < n:
            return v

        # strict: do not clamp
        raise ValueError(
            f"Parameter index out of range for step='{step}'. "
            f"Got {v}, but valid are 1..{n} (1-based) or 0..{n-1} (0-based)."
        )


    def _apply_step(self, handler, X, y, sens):
        result = handler.apply(X, y, sens)

        frac_header, frac_value = None, None
        method_name = f'get_outlier_bef_{handler.__class__.__name__.replace("Handler","").lower()}_strat'
        if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
            frac_value = getattr(handler, method_name)()
            step_name = handler.__class__.__name__.replace("Handler", "")
            frac_header = f'outlier_bef_{step_name.lower()}_strat'

        method_name2 = f'get_{handler.__class__.__name__.replace("Handler","").lower()}'
        if hasattr(handler, method_name2) and callable(getattr(handler, method_name2)):
            fraction_outlier = getattr(handler, method_name2)()
        else:
            detector = OutlierDetector(X)
            _, _, _ = detector.transform(y, sensitive_attr_train=None)
            fraction_outlier = detector.get_frac()

        if isinstance(result, (float, int)):
            utility = result
        elif isinstance(result, tuple):
            X, y, sens = result
            utility = None
        else:
            raise ValueError(f"Expected tuple output from {handler.__class__.__name__}.apply")

        return X, y, sens, utility, fraction_outlier, frac_header, frac_value
    
    def fuse_scores(self, similarity, utility, method='geometric', wS=1.0, wU=1.0,
                normalize=True):

        S = np.asarray(similarity, dtype=float)
        U = np.asarray(utility, dtype=float)
        if S.shape != U.shape:
            raise ValueError("similarity and utility must have the same shape")
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)

        eps = 1e-12

        def _minmax(x):
            x_min = np.min(x)
            x_max = np.max(x)
            if not np.isfinite(x_min) or not np.isfinite(x_max):
                return np.zeros_like(x)
            rng = x_max - x_min
            if rng < eps:
                return np.zeros_like(x)
            return (x - x_min) / (rng + eps)

        if normalize:
            S_hat = _minmax(S)
            U_hat = 1.0 - _minmax(U)
        else:
            U_pos = U - np.min(U)
            U_hat = 1.0 / (1.0 + U_pos + eps)
            S_hat = S

        wS = max(0.0, float(wS))
        wU = max(0.0, float(wU))
        w_sum = max(eps, wS + wU)

        method = str(method).lower()
        if method in ('geometric', 'geo'):
            S_clip = np.clip(S_hat, eps, 1.0)
            U_clip = np.clip(U_hat, eps, 1.0)
            fused = (np.power(S_clip, wS) * np.power(U_clip, wU)) ** (1.0 / w_sum)

        elif method in ('arithmetic', 'arith', 'mean', 'avg'):
            fused = (wS * S_hat + wU * U_hat) / w_sum

        elif method in ('harmonic', 'harm'):
            S_clip = np.clip(S_hat, eps, None)
            U_clip = np.clip(U_hat, eps, None)
            fused = (w_sum) / (wS / S_clip + wU / U_clip)

        else:
            raise ValueError("Unknown method. Use 'geometric', 'arithmetic', or 'harmonic'.")

        return fused


    # ---------- opaque pipeline ----------
    def run_pipeline_opaque(self, file_name):
        if self.execution_type == 'pass':
            X_copy, y_copy = self.X_train.copy(), self.y_train.copy()
            _,_,sensitive_attr_train = self.getIdxSensitive(X_copy, self.sensitive_var)
        else:
            X_copy, y_copy, sensitive_attr_train = self.get_injected_data()

        if os.path.exists(file_name):
            self.param_lst_df = pd.read_csv(file_name)[self.pipeline_order + [f'utility_{self.metric_type}']]
            return

        param_ranges = [range(self.strategy_counts[step]) for step in self.pipeline_order]
        param_combinations = itertools.product(*param_ranges)

        param_lst = []
        print("Running pipeline combinations...")

        for combo in param_combinations:
            if self.pipeline_type == 'ml':
                X, y, sens = X_copy.copy(), y_copy.copy(), sensitive_attr_train.copy()

            param_record = []
            utility = None
            for i, step in enumerate(self.pipeline_order):
                
                if step in ['swapping']:
                    pipeline = [m for m in self.pipeline_order if m != "swapping"]
                    handler = SwapHandler(strategy=i-2, config={"verbose": True})
                    new_pipeline, new_params= handler.apply_with_params(pipeline, cur_par)
                    utility = self.current_par_lookup(new_pipeline, new_params)

                else:
                    handler = self._load_handler(step, combo[i])
                    X, y, sens, util_tmp, _, _, _ = self._apply_step(handler, X, y, sens)
                if util_tmp is not None:
                    utility = util_tmp
                param_record.append(combo[i] + 1)
                print('dataset size after ', step, X.shape)

            param_lst.append(param_record + [utility])
            print(param_record + [utility])

        self.param_lst_df = pd.DataFrame(param_lst, columns=self.pipeline_order + [f'utility_{self.metric_type}'])
        self.param_lst_df.to_csv(file_name, index=False)

    # ---------- glass pipeline ----------
    def run_pipeline_glass(self, file_name):
        import time
        t0 = time.perf_counter()
        try:
            if self.execution_type == 'pass':
                X_copy, y_copy = self.X_train.copy(), self.y_train.copy()
            else:
                X_copy, y_copy = self.X_test.copy(), self.y_test.copy()
                #X_copy = self.noise_injector.inject_noise(X_copy, noise_type='outlier', frac=self.tau)
                #X_copy, y_copy = self.noise_injector.inject_noise(X_copy, y_copy, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_copy, self.sensitive_var)
                X, y, sens = X_copy.copy(), y_copy.copy(), sensitive.copy()
                #X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
                X, y = self.noise_injector.inject_multiple_noises(X, y=y, noise_types=self.noise_types, noise_params=self.noise_params)

            if os.path.exists(file_name):
                self.param_lst_df = pd.read_csv(file_name)[self.pipeline_order + [f'utility_{self.metric_type}']]
                return

            if self.pipeline_type == 'ml':
                X_copy = self.noise_injector.inject_noise(X_copy, noise_type='missing', frac=self.tau)
                _, _, sensitive_attr_train = self.getIdxSensitive(X_copy, self.sensitive_var)

            p = Profile()
            param_ranges = [range(self.strategy_counts[step]) for step in self.pipeline_order]
            param_combinations = itertools.product(*param_ranges)
            numerical_columns = X_copy.select_dtypes(include=['int', 'float']).columns
            param_lst = []
            print("Running pipeline combinations...")

            for combo in param_combinations:
                if self.pipeline_type == 'ml':
                    X, y, sens = X_copy.copy(), y_copy.copy(), sensitive_attr_train.copy()

                param_record, frac_data = [], []
                self.frac_header = []
                utility, fraction_outlier = None, None
                last_handler = None

                for i, step in enumerate(self.pipeline_order):
                    handler = self._load_handler(step, combo[i])
                    last_handler = handler
                    X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)
                    if frac_header is not None:
                        frac_data.append(frac_value)
                        self.frac_header.append(frac_header)
                    if util_tmp is not None:
                        utility = util_tmp
                    param_record.append(combo[i] + 1)
                print(param_record)

                self.headers, sens_data = last_handler.get_profile_metric(y, sens)
                prof_data = frac_data + sens_data
                profile_gen, key_profile = p.populate_profiles(
                    pd.concat([X, y], axis=1),
                    numerical_columns,
                    self.target_variable_name,
                    fraction_outlier,
                    self.metric_type
                )
                param_lst.append(param_record + prof_data + profile_gen + [utility])

            cols = self.pipeline_order + self.frac_header + self.headers + key_profile + [f'utility_{self.metric_type}']
            self.profile_param_lst_df = pd.DataFrame(param_lst, columns=cols)
            self.profile_param_lst_df.to_csv(file_name, index=False)
            pd.DataFrame(param_lst, columns=cols).to_csv(file_name, index=False)
        finally:
            elapsed = time.perf_counter() - t0
            print(f"[run_pipeline_glass] runtime: {elapsed:.3f} s")
            try:
                logging.info(f"run_pipeline_glass_runtime_seconds={elapsed:.6f}")
            except Exception:
                pass

    def run_pipeline_glass_sample(self, file_name, n=1000, seed=42):

        # ---------- 0) Load existing ----------
        if os.path.exists(file_name):
            print(f"[GlassRandom] File already exists -> skipping generation: {file_name}")
            self.profile_param_lst_df = pd.read_csv(file_name)
            return self.profile_param_lst_df

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        # ---------- 1) Enforce full component set + model last ----------
        if not isinstance(self.pipeline_order, (list, tuple)) or len(self.pipeline_order) == 0:
            raise ValueError("pipeline_order must be a non-empty list/tuple.")

        # remove duplicates but preserve order
        seen = set()
        canonical = []
        for s in self.pipeline_order:
            if s not in seen:
                canonical.append(s)
                seen.add(s)

        if "model" not in canonical:
            canonical.append("model")
        else:
            # move model to end
            canonical = [c for c in canonical if c != "model"] + ["model"]

        eval_order = canonical  # THIS is what we will execute & sample over

        # sanity: every step must have a strategy space
        missing_counts = [s for s in eval_order if s not in self.strategy_counts]
        if missing_counts:
            raise ValueError(
                f"Missing strategy_counts for steps: {missing_counts}. "
                f"Add them to self.strategy_counts before sampling."
            )

        # ---------- 2) Data setup (same spirit as your original) ----------
        if self.execution_type == 'pass':
            X_copy, y_copy = self.X_train.copy(), self.y_train.copy()
            X_copy, y_copy = self.noise_injector.inject_multiple_noises(
                X=X_copy, y=y_copy, noise_types=self.noise_types, noise_params=self.noise_params
            )
        else:
            X_copy, y_copy = self.X_test.copy(), self.y_test.copy()
            _, _, sensitive = self.getIdxSensitive(X_copy, self.sensitive_var)
            X_tmp, y_tmp, sens_tmp = X_copy.copy(), y_copy.copy(), sensitive.copy()
            X_copy, y_copy = self.noise_injector.inject_multiple_noises(
                X=X_tmp, y=y_tmp, noise_types=self.noise_types, noise_params=self.noise_params
            )

        if self.pipeline_type == 'ml':
            #X_copy = self.noise_injector.inject_noise(X_copy, noise_type='missing', frac=self.tau)
            _, _, sensitive_attr_train = self.getIdxSensitive(X_copy, self.sensitive_var)

        p = Profile()
        numerical_columns = X_copy.select_dtypes(include=['int', 'float']).columns

        # ---------- 3) Sampling space ----------
        sizes = [int(self.strategy_counts[step]) for step in eval_order]
        if any(s <= 0 for s in sizes):
            raise ValueError(f"Invalid strategy space sizes for eval_order={eval_order}: {sizes}")

        total_space = 1
        for s in sizes:
            total_space *= s

        if n <= 0:
            raise ValueError("n must be a positive integer.")
        n = int(min(n, total_space))

        # ---------- 4) Sample unique combos (0-based for handlers; store 1-based in CSV) ----------
        sampled = []
        sampled_keys = set()
        max_attempts = 10 * n + 10_000
        attempts = 0

        while len(sampled) < n and attempts < max_attempts:
            attempts += 1
            combo_0b = [int(rng.integers(0, sizes[i])) for i in range(len(sizes))]
            combo_1b = [c + 1 for c in combo_0b]
            key = ",".join(map(str, combo_1b))
            if key in sampled_keys:
                continue
            sampled_keys.add(key)
            sampled.append(combo_0b)

        if len(sampled) < n:
            print(f"[WARN] Only sampled {len(sampled)} unique combos (requested {n}).")

        # ---------- 5) Evaluate sampled combos ----------
        param_lst = []
        utility_col = f'utility_{self.metric_type}'
        # print(f"[GlassRandom] Evaluating {len(sampled)} random combinations (seed={seed})...")
        # print(f"[GlassRandom] eval_order (model forced last): {eval_order}")

        for run_idx, combo_0b in enumerate(sampled, start=1):
            if self.pipeline_type == 'ml':
                X, y, sens = X_copy.copy(), y_copy.copy(), sensitive_attr_train.copy()
            else:
                raise ValueError("run_pipeline_glass_sample currently assumes pipeline_type == 'ml'.")

            param_record, frac_data = [], []
            self.frac_header = []
            utility, fraction_outlier = None, None
            last_handler = None

            for step_i, step in enumerate(eval_order):
                handler = self._load_handler(step, combo_0b[step_i])
                last_handler = handler

                X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)

                if frac_header is not None:
                    frac_data.append(frac_value)
                    self.frac_header.append(frac_header)

                if util_tmp is not None:
                    utility = util_tmp

                # store 1-based param id
                param_record.append(combo_0b[step_i] + 1)

            # profile metrics
            self.headers, sens_data = last_handler.get_profile_metric(y, sens)
            prof_data = frac_data + sens_data

            profile_gen, key_profile = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            param_lst.append(param_record + prof_data + profile_gen + [utility])
            if run_idx % 25 == 0 or run_idx == 1 or run_idx == len(sampled):
                print(f"[Sample] {run_idx}/{len(sampled)} | params={param_record} | utility={utility}")

        # ---------- 6) Write output ----------
        cols = eval_order + self.frac_header + self.headers + key_profile + [utility_col]
        df_new = pd.DataFrame(param_lst, columns=cols)

        out_dir = os.path.dirname(file_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df_new.to_csv(file_name, index=False)

        elapsed = time.perf_counter() - t0
        print(f"[run_pipeline_glass_sample] wrote {len(df_new)} rows to {file_name} | runtime={elapsed:.3f}s")
        try:
            logging.info(f"run_pipeline_glass_sample_runtime_seconds={elapsed:.6f}")
        except Exception:
            pass

        self.profile_param_lst_df = df_new
        return df_new



    # ---------- scoring ----------
    def score_parameter(self, param_lst_df):
        if self.pipeline_type != 'ml':
            return None, None

        y = param_lst_df[f'utility_{self.metric_type}']
        X = param_lst_df.drop([f'utility_{self.metric_type}'], axis=1)

        if self.h_sample_bool:
            print(" ------ Sampling --------", self.h_sample)
            random.seed(42)
            sample_idx = random.sample(list(range(len(X))), math.ceil(self.h_sample * len(X)))
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]

        reg = Regression()
        model = reg.generate_regression(X, y)
        coefs = model.coef_
        print('coefficient', coefs)
        print('Intercept', model.intercept_)
        coef_rank = np.argsort(np.abs(coefs)).tolist()[::-1]
        print('ranking', coef_rank)
        logging.info(f'coef {coefs}')
        return coefs, coef_rank

    def get_injected_data(self):
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        X_test, y_test = self.noise_injector.inject_multiple_noises(X=X_test, y=y_test, noise_types=self.noise_types, noise_params=self.noise_params)  
        _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
        X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
        return X, y, sens
        
    # ---------- look up current pipeline’s utility ----------
    def current_par_lookup(self, pipeline, cur_par=[], fixed_data=None):
        if fixed_data is None:
            X, y, sens = self.get_injected_data()
        else:
            X, y, sens = fixed_data

        for i, step in enumerate(pipeline):
            param_index = self._safe_param_index(step, int(cur_par[i]))
            handler = self._load_handler(step, param_index)
            X, y, sens, util_tmp, _, _, _ = self._apply_step(handler, X, y, sens)
            if isinstance(util_tmp, (float, int)):
                return util_tmp

        raise ValueError("No output returned from final pipeline component.")


    # ---------- headers / profiles (the later-defined versions are preserved) ----------
    def get_header(self, file_name):
        df = pd.read_csv(file_name)
        known_cols = set(self.pipeline_order + [f'utility_{self.metric_type}'])
        extra_cols = [col for col in df.columns
                      if col not in known_cols and not col.startswith('cov') and
                      not col.startswith('outlier') and not col.startswith('insertion_pos')]
        self.headers = extra_cols
        return self.headers

    def rank_profile_new_comp(self, file_name, new_comp):
        param_lst_df = pd.read_csv(file_name)
        profiles = self.get_header(file_name)
        print("[INFO] Profiles (features):", profiles)
        if self.model_selection == 'reg':
            y = param_lst_df[new_comp]
            t = StandardScaler().fit(param_lst_df).transform(param_lst_df)
            X = pd.DataFrame(data=t, columns=param_lst_df.columns)[profiles]
        else:
            y = param_lst_df[new_comp]
            X = param_lst_df.copy()[profiles]

        if self.h_sample_bool:
            print(" ------ Sampling --------", self.h_sample)
            random.seed(1000)
            sample_idx = random.sample(list(range(len(X))), math.ceil(self.h_sample * len(X)))
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]

        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        clf.fit(X, y)

        coef_matrix = np.abs(clf.coef_)
        coef_df = pd.DataFrame(coef_matrix, columns=profiles)
        avg_coefs = coef_df.mean(axis=0)
        profile_ranking = avg_coefs.sort_values(ascending=False).index.tolist()
        profile_coefs = avg_coefs.sort_values(ascending=False).values
        return coef_df, profile_ranking

    def rank_profile_parameter(self, file_name):
        param_lst_df = pd.read_csv(file_name)
        self.profiles = self.get_header(file_name)

        if self.model_selection == 'reg':
            y = param_lst_df[f'utility_{self.metric_type}']
            t = StandardScaler().fit(param_lst_df).transform(param_lst_df)
            X = pd.DataFrame(data=t, columns=param_lst_df.columns)[self.profiles]
        else:
            y = param_lst_df[f'utility_{self.metric_type}']
            X = param_lst_df.copy()[self.profiles]

        if self.h_sample_bool:
            print(" ------ Sampling --------", self.h_sample)
            random.seed(1000)
            sample_idx = random.sample(list(range(len(X))), math.ceil(self.h_sample * len(X)))
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]

        reg = Regression()
        model = reg.generate_regression(X, y)
        coefs = model.coef_
        print(coefs)
        print(model.intercept_)

        self.profile_ranking = np.argsort(np.abs(coefs))[::-1]
        self.profile_coefs = coefs
        print(self.profile_coefs)

        # ranking parameter
        param_columns = self.pipeline_order
        self.ranking_param = {}
        self.param_coeff = {}
        for index, elem in enumerate(self.profiles):
            y = param_lst_df[elem]
            X = param_lst_df.copy()[param_columns]
            if self.h_sample_bool:
                X = X.iloc[sample_idx]
                y = y.iloc[sample_idx]
            reg = Regression()
            model = reg.generate_regression(X, y)
            coefs = model.coef_
            self.ranking_param[elem] = np.argsort(np.abs(coefs))[::-1]
            print(self.ranking_param[elem])
            self.param_coeff[elem] = coefs
            print(f'name : {elem} {self.param_coeff[elem]}')

        for idx, profile_index in enumerate(self.profile_ranking):
            print(self.profiles[profile_index])

        return self.profile_coefs, self.profile_ranking, self.param_coeff, self.ranking_param
    
    # ---------- get passing pipeline from passing dataset ----------
    def get_passing_pipeline(self, train_file, target_column, threshold_val):
        df_train = pd.read_csv(train_file)
        if target_column not in df_train.columns:
            raise ValueError(f"'{target_column}' not found in training file.")
        df_filtered = df_train[np.abs(df_train[target_column]) < threshold_val]
        if df_filtered.empty:
            print("No rows satisfy the threshold filter.")
            return None
        best_row = df_filtered.loc[df_filtered[target_column].idxmin()]
        best_param = [best_row[step] for step in self.pipeline_order]
        best_param = [int(x) for x in best_param]
        return best_param

    # ---------- similarity utilities ----------
    def profile_similarity_df(self, file_train, file_test, param, profile_cols, metric='cosine'):
        df1 = pd.read_csv(file_train)
        df2 = pd.read_csv(file_test)

        # Filter training rows by pipeline parameters
        param_dict = dict(zip(self.pipeline_order, param))
        for col, val in param_dict.items():
            if col in df1.columns:
                df1 = df1[df1[col].astype(str) == str(val)]

        if df1.empty or df2.empty:
            print("Pipeline parameter not found in one or both datasets.")
            return None

        # --- STRICT: remove pipeline-order columns before similarity ---
        drop_cols = [c for c in self.pipeline_order if c in df1.columns]

        df1_prof = df1.drop(columns=drop_cols, errors="ignore")
        df2_prof = df2.drop(columns=drop_cols, errors="ignore")

        # Keep only profile columns (intersection for safety)
        prof_cols = [c for c in profile_cols if c in df1_prof.columns and c in df2_prof.columns]

        if len(prof_cols) == 0:
            raise ValueError("No valid profile columns left after dropping pipeline parameters.")

        v1 = df1_prof[prof_cols].iloc[0].astype(float).values.reshape(1, -1)
        v2 = df2_prof[prof_cols].iloc[0].astype(float).values.reshape(1, -1)

        similarity = compute_similarity(v1, v2, metric=metric)
        print(f"{metric.capitalize()} similarity:", similarity)

        return similarity

    

    def profile_similarity_df_average(self, file_train, file_test,
                                      profile_cols, threshold_col, threshold_val, metric='cosine'):
        df_train = pd.read_csv(file_train)
        df_test = pd.read_csv(file_test)

        if threshold_col not in df_train.columns:
            raise ValueError(f"'{threshold_col}' not found in training file.")
        df_train_filtered = df_train[df_train[threshold_col] < threshold_val]
        if df_train_filtered.empty:
            print("No training rows satisfy the threshold filter.")
            return None

        train_num = df_train_filtered[profile_cols].apply(pd.to_numeric, errors='coerce')
        train_mean_profile = train_num.mean(skipna=True)

        if df_test.shape[0] == 0:
            print("Test dataset is empty.")
            return None
        test_num = df_test.iloc[0][profile_cols].apply(pd.to_numeric, errors='coerce')

        test_num = test_num.fillna(train_mean_profile)
        train_mean_profile = train_mean_profile.fillna(0.0)
        test_num = test_num.fillna(0.0)

        test_profile = test_num.values.reshape(1, -1)
        train_profile = train_mean_profile.values.reshape(1, -1)
        sim = compute_similarity(test_profile, train_profile, metric=metric)
        return sim

    def profile_similarity_df_filtered(self, file_train, file_test, param, profile_cols,
                                       threshold_col, threshold_val, output_file, metric='cosine'):
        df_train = pd.read_csv(file_train)
        df_test = pd.read_csv(file_test)

        df_train_filtered = df_train[df_train[threshold_col] < threshold_val]
        param_dict = dict(zip(self.pipeline_order, param))
        for col, val in param_dict.items():
            if col in df_test.columns:
                df_test = df_test[df_test[col].astype(str) == str(val)]

        if df_test.empty:
            print("Parameter combination not found in test dataset.")
            return None

        test_profile = df_test[profile_cols].iloc[0].astype(float).values.reshape(1, -1)

        results = []
        for idx, row in df_train_filtered.iterrows():
            try:
                train_profile = row[profile_cols].astype(float).values.reshape(1, -1)
                sim = compute_similarity(test_profile, train_profile, metric=metric)
                param_values = [row.get(key, None) for key in self.pipeline_order]
                results.append((param_values, sim))
            except Exception as e:
                print(f"Error in row {idx}: {e}")
                continue

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(f"# Test parameter values: {param}\n")
            f.write("param_values,similarity\n")
            for param_list, similarity in results:
                f.write(f"{param_list},{similarity:.6f}\n")

        print(f"Similarity results saved to: {output_file}")
        return results

    def profile_similarity_all_rows(self, file_train, file_test, profile_cols, output_file):
        df1 = pd.read_csv(file_train)
        df2 = pd.read_csv(file_test)

        df1.dropna(axis=1, how='any', inplace=True)
        df2.dropna(axis=1, how='any', inplace=True)

        df1['param_key'] = df1[self.pipeline_order].astype(str).agg(','.join, axis=1)
        df2['param_key'] = df2[self.pipeline_order].astype(str).agg(','.join, axis=1)

        shared_profile_cols = [col for col in profile_cols if col in df1.columns and col in df2.columns]
        if not shared_profile_cols:
            raise ValueError("No shared profile columns found between the datasets.")

        rows = []
        grouped1 = df1.groupby('param_key')
        grouped2 = df2.groupby('param_key')

        keys = set(grouped1.groups.keys()) & set(grouped2.groups.keys())
        for key in keys:
            g1 = grouped1.get_group(key)
            g2 = grouped2.get_group(key)
            for idx1, row1 in g1.iterrows():
                v1 = row1[shared_profile_cols].astype(float).values.reshape(1, -1)
                for idx2, row2 in g2.iterrows():
                    v2 = row2[shared_profile_cols].astype(float).values.reshape(1, -1)
                    sim = cosine_similarity(v1, v2)[0][0]
                    rows.append({
                        **{k: row1[k] for k in self.pipeline_order},
                        'index_df1': idx1,
                        'index_df2': idx2,
                        'similarity': sim
                    })

        df_sim = pd.DataFrame(rows)
        df_sim.sort_values(by=self.pipeline_order, inplace=True)
        df_sim.to_csv(output_file, index=False)

    # ---------- intervention evaluation ----------
    def evaluate_interventions_pred_and_similarity(self, cur_par, filename_training, new_components, wS=1.0, wU=1.0):
        np.random.seed(42)
        

        # ---------- 0) Train predictor on training CSV ----------
        if not os.path.exists(filename_training):
            self.run_pipeline_glass(filename_training)

        df_train = pd.read_csv(filename_training)
        utility_col = f'utility_{self.metric_type}'
        if utility_col not in df_train.columns:
            raise ValueError(f"Training file must contain column '{utility_col}'")
        
        prof_cols = self.get_header(filename_training)
        passing_par = self.get_passing_pipeline(filename_training, f'utility_{self.metric_type}', threshold_val=self.utiliy_threshold)
        #passing_par = cur_par

        feature_cols = [c for c in df_train.columns if c != utility_col]
        X_all = df_train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        y_all = pd.to_numeric(df_train[utility_col], errors='coerce').fillna(0.0).values

        #X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.20, random_state=42)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_scaled = scaler.fit_transform(X_all)

        poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=False)
        X_tr_poly = poly.fit_transform(X_tr_scaled)

        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-4, 2, 30),
            cv=5,
            max_iter=10000,
            random_state=42
        ).fit(X_tr_poly, y_all)

        # quick hold-out report
        #y_val_pred = np.ravel(enet.predict(poly.transform(scaler.transform(X_val))))
        #print(f"[Pred+Sim(optpos)] Hold-out R^2={r2_score(y_val, y_val_pred):.4f} | RMSE={np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}")
        #print(f"[Pred+Sim(optpos)] ElasticNet alpha={enet.alpha_:.6g} | l1_ratio={enet.l1_ratio_}")

        # ---------- helpers ----------
        def _build_feature_row(eval_order, eval_params):
            if self.pipeline_type == 'ml':
                X,y,sens=self.get_injected_data()


            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns

            param_record, frac_data = [], []
            self.frac_header = []
            utility = None
            fraction_outlier = None
            #last_handler = None

            for i, step in enumerate(eval_order[:-1]):
                param_index = self._safe_param_index(step, int(eval_params[i]))
                handler = self._load_handler(step, param_index)
                #last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)
                if frac_header is not None:
                    frac_data.append(frac_value)
                    self.frac_header.append(frac_header)
                if util_tmp is not None:
                    utility = util_tmp
                else:
                    utility = 0.0
                param_record.append(param_index + 1)

            model_param_1b = 1
            param_record.append(model_param_1b)

            # ---- set last_handler manually to model (0-based) ----
            last_handler = self._load_handler("model", 0)

            self.headers, sens_data = last_handler.get_profile_metric(y, sens)
            prof_data = frac_data + sens_data

            profile_gen, key_profile = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            out_cols = eval_order
            row = param_record + prof_data + profile_gen + [utility]
            col_headers = out_cols + self.frac_header + self.headers + key_profile + [f'utility_{self.metric_type}']
            row_df = pd.DataFrame([row], columns=col_headers)
            return row_df

        def _align_features(test_row_df):
            missing = [c for c in feature_cols if c not in test_row_df.columns]
            for c in missing:
                test_row_df[c] = 0.0
            extras = [c for c in test_row_df.columns if c not in feature_cols and c != utility_col]
            if extras:
                test_row_df = test_row_df.drop(columns=extras)
            test_row_df = test_row_df.reindex(columns=feature_cols)
            test_row_df = test_row_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            return test_row_df

        def _predict_from_row_df(row_df):
            X_feat = _align_features(row_df.drop(columns=[utility_col], errors='ignore'))
            X_poly = poly.transform(scaler.transform(X_feat.values))
            return float(np.ravel(enet.predict(X_poly))[0])

        def _similarity_from_row_df(row_df):
            test_file = f'historical_data/insertion/test_row_profile.csv'
            row_df.to_csv(test_file, index=False)
            return self.profile_similarity_df(filename_training, test_file, passing_par, prof_cols, metric='cosine')

        # ensure rank_profile exists (as in your eval funcs)
        try:
            _, self.rank_profile = self.rank_profile_new_comp(filename_training, ['outlier'])
        except Exception:
            pass

        original_order = self.pipeline_order
        insertion_positions = list(range(2, len(original_order)))  # we will choose best pos per (comp,strat)
        fused_rows = []  # (comp, strat, sim, y_pred, pos_or_None, fused)

        # ---------- A) Existing components: alternate strategies ----------
        for i, component in enumerate(original_order):
            num_strategies = self.strategy_counts[component]
            current_strategy = cur_par[i]
            for new_strategy in range(1, num_strategies + 1):
                if new_strategy == current_strategy:
                    continue
                new_cur_par = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                row_df = _build_feature_row(original_order, new_cur_par)
                sim = _similarity_from_row_df(row_df)
                if sim is None:
                    continue
                y_pred = _predict_from_row_df(row_df)
                y_truth = row_df[f'utility_{self.metric_type}'].values[0]
                fused_rows.append((component, new_strategy, float(sim), float(y_pred), None))

        # ---------- B) New components: for each (comp,strat) keep only BEST position by similarity ----------
        for comp in new_components:
            comp_ranges = self.strategy_counts[comp]
            for strat_idx in range(1,comp_ranges+1):
                if comp in ["swapping"]:
                    utility_bef = self.current_par_lookup(original_order, cur_par)
                    pipeline = [m for m in original_order if m != "swapping"]
                    handler = SwapHandler(strategy=strat_idx-1, config={"verbose": True})
                    new_pipeline, new_params= handler.apply_with_params(pipeline, cur_par)
                    row_df = _build_feature_row(new_pipeline, new_params)
                    sim = _similarity_from_row_df(row_df)
                    if sim is None:
                        continue
                    y_pred = _predict_from_row_df(row_df)
                    utility_aft = self.current_par_lookup(new_pipeline, new_params)
                    fused_rows.append((comp, strat_idx, float(sim), float(y_pred), 1))
                else:
                    best_for_this_strat = None  # (sim, y_pred, pos)
                    for insert_pos in insertion_positions:
                        new_order   = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                        new_cur_par = cur_par[:insert_pos] + [strat_idx] + cur_par[insert_pos:]
                        row_df = _build_feature_row(new_order, new_cur_par)
                        sim = _similarity_from_row_df(row_df)
                        if sim is None:
                            continue
                        y_pred = _predict_from_row_df(row_df)
                        y_truth = row_df[f'utility_{self.metric_type}'].values[0]
                        if (best_for_this_strat is None) or (sim > best_for_this_strat[0]):
                            best_for_this_strat = (float(sim), float(y_pred), insert_pos)

                    if best_for_this_strat is not None:
                        sim_star, ypred_star, pos_star = best_for_this_strat
                        fused_rows.append((comp, strat_idx, float(sim_star), float(ypred_star), pos_star))

        # ---------- Fuse & rank ----------
        if not fused_rows:
            print("[Pred+Sim(optpos)] No valid candidates.")
            return []
        

        S = np.array([t[2] for t in fused_rows], dtype=float)  # similarity ↑
        U = np.array([t[3] for t in fused_rows], dtype=float)  # utility ↓

        fused = self.fuse_scores(S, U, method='arith', wS=wS, wU=wU)
        fused_ranking = [(*t, float(f)) for t, f in zip(fused_rows, fused)]
        #fused_ranking.sort(key=lambda x: x[-1], reverse=True)
        fused_ranking.sort(key=lambda x: x[2], reverse=True)
        

        # # Pretty print
        # for idx, (component, strategy, sim, y_pred, pos, _) in enumerate(fused_ranking, start=1):
        #     if pos is not None:
        #         print(f"[{idx}] NEW {component}@{pos}-->strat={strategy} | sim={sim} | uti={y_pred:.4f} | truth={y_truth:.4f}")
        #     else:
        #         print(f"[{idx}] EXIST {component}-->strat={strategy} | sim={sim} | uti={y_pred:.4f} | truth={y_truth:.4f}")

        return fused_ranking

    
    def run_pipeline_opaque_with_positions(self, file_name):
        import time
        from itertools import product, permutations

        t0 = time.perf_counter()
        try:
            # ---------- Data setup (opaque-style) ----------
            if self.execution_type == 'pass':
                X_base, y_base = self.X_train.copy(), self.y_train.copy()
            else:
                X_base, y_base, sens_base = self.get_injected_data()

            if self.pipeline_type != 'ml':
                raise ValueError("This method assumes pipeline_type == 'ml'.")

            # inject missingness like run_pipeline_opaque
            X_base = self.noise_injector.inject_noise(X_base, noise_type='missing', frac=self.tau)
            _, _, sens_base = self.getIdxSensitive(X_base, self.sensitive_var)

            if 'model' not in self.pipeline_order:
                raise ValueError("Expected 'model' in pipeline_order.")

            # ---------- Schema ----------
            strategy_cols = list(self.pipeline_order)                 # canonical strategies (includes 'model')
            base_components = [c for c in self.pipeline_order if c != 'model']
            pos_cols = [f"{c}_pos" for c in base_components]          # NO model_pos
            utility_col = f'utility_{self.metric_type}'
            final_cols = strategy_cols + pos_cols + [utility_col]     # enforced column ordering

            # If file exists: load, normalize cols, sort by strategies then positions, save, return.
            if os.path.exists(file_name):
                df_old = pd.read_csv(file_name)

                # Drop any legacy model_pos if present
                if "model_pos" in df_old.columns:
                    df_old = df_old.drop(columns=["model_pos"])

                # Ensure required columns exist with defaults
                for c in strategy_cols:
                    if c not in df_old.columns:
                        df_old[c] = 1
                for c in pos_cols:
                    if c not in df_old.columns:
                        df_old[c] = len(base_components)  # default "last" among non-models
                if utility_col not in df_old.columns:
                    df_old[utility_col] = float("nan")

                # Reorder columns and sort rows to group "serially"
                df_old = df_old.reindex(columns=final_cols).sort_values(by=strategy_cols + pos_cols, kind="mergesort")
                df_old.to_csv(file_name, index=False)
                self.param_lst_df = df_old[strategy_cols + [utility_col]]
                return

            # ---------- Enumerate strategy combos (outer), then position perms (inner) ----------
            # Build 0-based strategy ranges for each canonical component (incl. model)
            strat_ranges = {comp: range(self.strategy_counts[comp]) for comp in self.pipeline_order}

            rows = []
            print("[Opaque+Pos] Serial order: strategies outer, positions inner (model fixed last, no model_pos).")

            # OUTER: all strategy tuples in CANONICAL order (incl. model)
            canonical_ranges = [strat_ranges[c] for c in strategy_cols]
            for combo in product(*canonical_ranges):
                # Fixed canonical strategies for this block:
                # map component -> 1-based strategy
                combo_map = {comp: int(idx) + 1 for comp, idx in zip(strategy_cols, combo)}

                # INNER: try all permutations of positions for non-model components
                for perm in permutations(base_components):
                    eval_order = list(perm) + ['model']  # apply steps in this order

                    print(eval_order)
                    X, y, sens = X_base.copy(), y_base.copy(), sens_base.copy()
                    utility = None
                    pos_record_map = {}  # non-model component -> 1-based position

                    # Apply in permuted order, but use strategies from the fixed combo_map
                    for pos_idx, step in enumerate(eval_order, start=1):
                        # combo_map is 1-based; handler expects 0-based
                        strat_1b = combo_map[step]
                        handler = self._load_handler(step, strat_1b - 1)

                        if step != 'model':
                            pos_record_map[step] = pos_idx  # record position of non-model only

                        X, y, sens, util_tmp, _, _, _ = self._apply_step(handler, X, y, sens)
                        if util_tmp is not None:
                            utility = util_tmp

                    if utility is None:
                        utility = float("nan")

                    # Build row in enforced order:
                    # 1) strategies (canonical order)  2) positions (base_components order)  3) utility
                    strategy_vals = [combo_map[c] for c in strategy_cols]
                    pos_vals = [pos_record_map.get(c, len(base_components)) for c in base_components]
                    rows.append(strategy_vals + pos_vals + [utility])

            # ---------- Save in enforced order & already-grouped row order ----------
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            df_out = pd.DataFrame(rows, columns=final_cols)
            # Defensive stable sort to preserve grouping if needed
            df_out = df_out.sort_values(by=strategy_cols + pos_cols, kind="mergesort")
            df_out.to_csv(file_name, index=False)

            # Minimal view for downstream utilities
            self.param_lst_df = df_out[strategy_cols + [utility_col]]
            print(f"[Opaque+Pos] wrote {len(df_out)} rows to {file_name}")

        finally:
            elapsed = time.perf_counter() - t0
            print(f"[run_pipeline_opaque_with_positions] runtime: {elapsed:.3f} s")
            try:
                logging.info(f"run_pipeline_opaque_with_positions_runtime_seconds={elapsed:.6f}")
            except Exception:
                pass





    # ---------- misc ----------
    def get_profile(self):
        return getattr(self, 'profile', None)

    def run_pipeline(self, alg_type, file_name):
        if alg_type == 'opaque':
            self.run_pipeline_opaque(file_name)
        elif alg_type == 'glass':
            self.run_pipeline_glass(file_name)


# dataset_name = 'adult'
# metric_type = 'sp'
# model_type = 'lr'

# pipeline_order = ["sampling","invalid_value","missing_value","floating_point","unit_converter","distribution_shape","multicollinearity","normalization","outlier","punctuation","stopword", "lowercase","whitespace","model"]

# #pipeline_order = ["sampling","missing_value","nonlinear_transform","model"]
# # pipeline_order = [
# #   "query_normalize",
# #   "retriever",
# #   "reranker",
# #   "context_builder",
# #   "generator"
# # ]

# # new_comp=['swapping']

# # output = f'historical_data/similarity/Noisy_vs_orig_{model_type}_{metric_type}_{dataset_name}.csv'
# # output1 = f'historical_data/similarity/Noisy_vs_filterd_orig_{model_type}_{metric_type}_{dataset_name}.csv'
# # filename_test = f'historical_data/noise/1111BugDoc_profile_{model_type}_{metric_type}_{dataset_name}.csv'
# # #filename_test = 'class_sim_historical_data_test_profile_lr_sp_adult.csv'
# # filename_train = f'historical_data/revision/2000_sample_historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
# # # filename_train1 = f'historical_data/historical_data_train_profile{model_type}_{metric_type}_{dataset_name}.csv'
# executor = PipelineExecutor(
#                  pipeline_type='ml',
#                  dataset_name=dataset_name,
#                  metric_type=metric_type,
#                  pipeline_ord=pipeline_order,
#                  execution_type='fail',
#              )

# fixed_data = executor.get_injected_data()

# X, y, sens = fixed_data

# df_injected = X.copy()
# df_injected[y.name] = y.values
# df_injected[executor.sensitive_var] = sens.values

# out_path = f"historical_data/noise_injected_{dataset_name}.csv"
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# df_injected.to_csv(out_path, index=False)

# print("Saved:", out_path, "| shape:", df_injected.shape)



# # #cur_par = [2, 2, 2, 1]  # example current parameter combination (1-based indexing)
# # for k in ['pass']:
# #     executor = PipelineExecutor(
# #                  pipeline_type='ml',
# #                  dataset_name=dataset_name,
# #                  metric_type=metric_type,
# #                  pipeline_ord=pipeline_order,
# #                  execution_type=k,
# #              )
# #     if k=='pass':
# #         j='train'
# #     else:
# #         j='test'
# #     fixed_data = executor.get_injected_data()
# #     for i in [500, 1000, 1500, 2000]:
# #         filename_train = f'historical_data/revision/{i}_sample_historical_data_{j}_profile_{model_type}_{metric_type}_{dataset_name}.csv'
# #         executor.run_pipeline_glass_sample(filename_train, n=i)
# # #_, rank_profile=executor.rank_profile_new_comp(filename_train1, new_comp)
# #rank_profile.remove('corr_Country')
# #executor.evaluate_interventions_pred_and_similarity(cur_par, filename_train, new_components=new_comp)
# #executor.evaluate_with_component_insertion(cur_par, new_components=new_comp)
# #executor.evaluate_interventions_swap_only(cur_par, filename_train)
# #print(rank)
# #executor.profile_similarity_all_rows(filename_test, filename_test, pd.read_csv(filename_train), output1)
# #executor.profile_similarity_df(filename_train, filename_test, cur_par, pd.read_csv(filename_test), metric='cosine')
# #Change the cur_par_lookUp fucntion
# #print(executor.current_par_lookup(pipeline_order, cur_par, fixed_data))
# #u1 = executor.current_par_lookup(pipeline_order, cur_par, fixed_data=fixed_data)
# #u2 = executor.current_par_lookup(pipeline_order, cur_par, fixed_data=fixed_data)
# #print(u1, u2)
# #similarities = executor.profile_similarity_df_filtered(filename_train,filename_test,param=cur_par,profile_cols=pd.read_csv(filename_train).columns,threshold_col='utility_sp',threshold_val=0.055,output_file=output1,metric='cosine')'''
