import os
import pandas as pd
import numpy as np
import logging
import random
import math
import importlib
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from regression import Regression
from LoadDataset import LoadDataset
from noise_injection import NoiseInjector
from modules.profiling.profile import Profile
from modules.outlier_detection.outlier_detector import OutlierDetector
from similarity_metric import compute_similarity


class PipelineExecutor:
    def __init__(self, pipeline_type, dataset_name, metric_type, pipeline_ord,
                 execution_type='pass', h_sample_bool=False, scalability_bool=False):

        self.pipeline_type = pipeline_type  # 'ml' or 'em'
        self.dataset_name = dataset_name
        self.metric_type = metric_type
        self.pipeline_order = pipeline_ord  # always keep model at the end
        self.execution_type = execution_type  # 'pass' or 'fail'
        self.h_sample_bool = h_sample_bool
        self.scalability_bool = scalability_bool
        if self.h_sample_bool:
            self.h_sample = 0.005

        if self.pipeline_type == 'ml':
            # structured strategies
            self.mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
            self.norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm']
            self.od_strategy = ['none', 'if', 'iqr', 'lof']
            self.model_selection = ['lr']  # rf, 'reg' # 'nb'
            self.knn_k_lst = [1, 5, 10, 20, 30]
            self.lof_k_lst = [1, 5, 10, 20, 30]

            # text strategies
            self.pr_strategy = ['pr']
            self.lowercase_strategy = ['lc']
            self.spellcheck_strategy = ['sc']
            self.whitespace_strategy = ['wc']
            self.unit_converter_strategy = ['uc']
            self.tokenization_strategy = ['whitespace', 'nltk']
            self.stopword_strategy = ['sw']
            self.specialchar_strategy = ['sc']
            self.deduplication_strategy = ['dd']

            # data
            loader = LoadDataset(self.dataset_name)
            self.dataset, self.X_train, self.y_train, self.X_test, self.y_test = loader.load()
            self.sensitive_var = loader.get_sensitive_variable()
            self.target_variable_name = self.y_train.name

            self.noise_injector = NoiseInjector(self.pipeline_type, self.dataset_name, self.target_variable_name)

            # strategy counts (for search space sizes)
            self.strategy_counts = {
                'missing_value': len(self.mv_strategy) + len(self.knn_k_lst) - 1 if 'knn' in self.mv_strategy else len(self.mv_strategy),
                'normalization': len(self.norm_strategy),
                'outlier': len(self.od_strategy) + len(self.lof_k_lst) - 1 if 'lof' in self.od_strategy else len(self.od_strategy),
                'model': len(self.model_selection),
                # text
                'punctuation': len(self.pr_strategy),
                'whitespace': len(self.whitespace_strategy),
                'unit_converter': len(self.unit_converter_strategy),
                'tokenizer': len(self.tokenization_strategy),
                'stopword': len(self.stopword_strategy),
                'spell_checker': len(self.spellcheck_strategy),
                'special_character': len(self.specialchar_strategy),
                'deduplication': len(self.deduplication_strategy),
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
                    self.tau = 0.05
                    self.contamination = 0.1
                    self.contamination_lof = 0.1
                else:
                    self.tau = 0.1
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
                'lof_k_lst': self.lof_k_lst,
                'contamination': self.contamination,
                'contamination_lof': self.contamination_lof,
                'model_selection': self.model_selection,
                'metric_type': self.metric_type,
                'sensitive_var': self.sensitive_var,
                'target_var': self.target_variable_name,
                # text
                'punctuation_strategy': self.pr_strategy,
                'whitespace_strategy': self.whitespace_strategy,
                'unit_converter_strategy': self.unit_converter_strategy,
                'tokenization_strategy': self.tokenization_strategy,
                'stopword_strategy': self.stopword_strategy,
                'spellchecker_strategy': self.spellcheck_strategy,
                'specialchar_strategy': self.specialchar_strategy,
                'deduplication_strategy': self.deduplication_strategy,
            }
        elif self.pipeline_type == 'em':
            print('pipeline_type is em')

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

    def _apply_step(self, handler, X, y, sens):
        """Apply handler and return (X,y,sens, utility, fraction_outlier, frac_data_header, frac_data_value)."""
        result = handler.apply(X, y, sens)

        # optional 'before' outlier fraction accessor
        frac_header, frac_value = None, None
        method_name = f'get_outlier_bef_{handler.__class__.__name__.replace("Handler","").lower()}_strat'
        if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
            frac_value = getattr(handler, method_name)()
            step_name = handler.__class__.__name__.replace("Handler", "")
            frac_header = f'outlier_bef_{step_name.lower()}_strat'

        # generic outlier fraction accessor
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
    
    def fused_geometric_mean(self, similarity, utility, wS=1.0, wU=1.0):    
        S = np.array(similarity, dtype=float)
        U = np.array(utility, dtype=float)
        S_hat = (S - S.min()) / (S.max() - S.min() + 1e-12)
        U_hat = 1 - (U - U.min()) / (U.max() - U.min() + 1e-12)
        fused = (S_hat**wS * U_hat**wU) ** (1.0 / (wS + wU))
        return fused

    # ---------- opaque pipeline ----------
    def run_pipeline_opaque(self, file_name):
        if self.execution_type == 'pass':
            X_copy, y_copy = self.X_train.copy(), self.y_train.copy()
        else:
            X_copy, y_copy = self.X_test.copy(), self.y_test.copy()

        if os.path.exists(file_name):
            self.param_lst_df = pd.read_csv(file_name)[self.pipeline_order + [f'utility_{self.metric_type}']]
            return

        if self.pipeline_type == 'ml':
            X_copy = self.inject_missing_values(X_copy, self.tau)
            _, _, sensitive_attr_train = self.getIdxSensitive(X_copy, self.sensitive_var)

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
                handler = self._load_handler(step, combo[i])
                X, y, sens, util_tmp, _, _, _ = self._apply_step(handler, X, y, sens)
                if util_tmp is not None:
                    utility = util_tmp
                param_record.append(combo[i] + 1)

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
                # X_copy= self.noise_injector.inject_noise(X_copy,noise_type='outlier', frac = self.tau)
                # X_copy, y_copy = self.noise_injector.inject_noise(X_copy, y_copy, noise_type='class_imbalance', frac=0.49)
                X_copy = self.noise_injector.inject_noise(X_copy, noise_type='outlier', frac=self.tau)
                X_copy, y_copy = self.noise_injector.inject_noise(X_copy, y_copy, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_copy, self.sensitive_var)
                X, y, sens = X_copy.copy(), y_copy.copy(), sensitive.copy()
                #X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)

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

    # ---------- look up current pipeline’s utility ----------
    def current_par_lookup(self, pipeline, cur_par=[]):
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()

        if self.pipeline_type == 'ml':
            X_test = self.inject_missing_values(X_test, self.tau)
            _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
            X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()

        for i, step in enumerate(pipeline):
            param_index = int(cur_par[i]) - 1
            handler = self._load_handler(step, param_index)
            result = handler.apply(X, y, sens)
            if isinstance(result, (float, int)):
                return result
            else:
                X, y, sens = result

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
        logging.basicConfig(
            filename='logs/franking' + "_" + dataset_name + '_' + model_type + '_' + metric_type + '.log',
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
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

    # ---------- similarity utilities ----------
    def profile_similarity_df(self, file_train, file_test, param, profile_cols, metric='cosine'):
        df1 = pd.read_csv(file_train)
        df2 = pd.read_csv(file_test)

        param_dict = dict(zip(self.pipeline_order, param))
        for col, val in param_dict.items():
            if col in df1.columns:
                df1 = df1[df1[col].astype(str) == str(val)]
        if df1.empty or df2.empty:
            print("Pipeline parameter not found in one or both datasets.")
            return None

        v1 = df1[profile_cols].iloc[0].astype(float).values.reshape(1, -1)
        v2 = df2[profile_cols].iloc[0].astype(float).values.reshape(1, -1)
        similarity = compute_similarity(v1, v2, metric=metric)
        print("Cosine Similarity:", similarity)
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
    def evaluate_interventions(self, cur_par, filename_training, new_components):
        filename_training = f'historical_data/historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
        original_order = self.pipeline_order
        insertion_positions = list(range(len(original_order)))
        _, self.rank_profile = self.rank_profile_new_comp(filename_training, ['outlier'])

        best_sim = -float('inf')
        best_component = None
        best_insert_pos = None
        best_result = None
        best_utility = None
        global_ranking = []

        logging.basicConfig(
            filename=f'logs/finsertion_{dataset_name}_{model_type}_{metric_type}.log',
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        def _eval_config(eval_order, eval_params, tag_component, tag_strategy, position=None):
            X_test = self.X_test.copy()
            y_test = self.y_test.copy()

            if self.pipeline_type == 'ml':
                X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
            else:
                X, y, sens = X_test.copy(), y_test.copy(), None

            param_record, frac_data = [], []
            self.frac_header = []
            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns
            utility = None
            fraction_outlier = None
            last_handler = None
            '''first_component = eval_order[0]
            if first_component.lower() != 'missing':
                X = X.dropna(axis=1)'''

            for i, step in enumerate(eval_order):
                param_index = int(eval_params[i]) - 1
                handler = self._load_handler(step, param_index)
                last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)
                if frac_header is not None:
                    frac_data.append(frac_value)
                    self.frac_header.append(frac_header)
                if util_tmp is not None:
                    utility = util_tmp
                param_record.append(param_index + 1)

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

            df = pd.DataFrame([row], columns=col_headers)
            test_file = f'historical_data/insertion/historical_data_test_profile_{model_type}_{metric_type}_{dataset_name}.csv'
            df.to_csv(test_file, index=False)

            sim = self.profile_similarity_df(filename_training, test_file, cur_par, self.rank_profile, metric='cosine')
            logging.info(f'Intervention: component={tag_component}, strategy={tag_strategy}, similarity={sim}, utility={utility}')

            if sim is not None:
                global_ranking.append((tag_component, tag_strategy, sim, utility, position))

            return sim, utility

        # (A) try alternate strategies for existing steps
        for i, component in enumerate(original_order):
            num_strategies = self.strategy_counts[component]
            current_strategy = cur_par[i]
            for new_strategy in range(1, num_strategies + 1):
                if new_strategy == current_strategy:
                    continue
                new_cur_par = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                _eval_config(original_order, new_cur_par, component, new_strategy)

        # (B) try inserting new components at positions
        for comp in new_components:
            comp_ranges = self.strategy_counts[comp]
            for strat_idx in range(comp_ranges):
                for insert_pos in insertion_positions:
                    new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                    new_cur_par = cur_par[:insert_pos] + [strat_idx + 1] + cur_par[insert_pos:]
                    sim, utility = _eval_config(new_order, new_cur_par, comp, strat_idx + 1, insert_pos)
                    if sim is not None and sim > best_sim:
                        best_component = comp
                        best_result = new_cur_par
                        best_sim = sim
                        best_utility = utility
                        best_insert_pos = insert_pos

        global_ranking.sort(key=lambda x: x[2], reverse=True)
        for idx, (component, strategy, sim, utility, pos) in enumerate(global_ranking):
            if pos is not None:
                print(f"New component: {component} @ {pos}, -- Strategy -> {strategy}, Similarity={sim:.4f}, Utility={utility:.4f}")
            else:
                print(f"Existing component: {component}, -- Strategy -> {strategy}, Similarity={sim:.4f}, Utility={utility:.4f}")
        return global_ranking
    
    def evaluate_interventions1(self, cur_par, filename_training, new_components):
        import time, logging   # <-- add time (and logging if not already imported)

        t0 = time.perf_counter()   # <-- start timer
        try:
            # ====== your existing body starts here ======

            filename_training = f'historical_data/historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
            original_order = self.pipeline_order
            insertion_positions = list(range(len(original_order)))
            _, self.rank_profile = self.rank_profile_new_comp(filename_training, ['outlier'])

            best_sim = -float('inf')
            best_component = None
            best_insert_pos = None
            best_result = None
            best_utility = None
            global_ranking = []

            logging.basicConfig(
                filename=f'logs/finsertion_{dataset_name}_{model_type}_{metric_type}.log',
                filemode='w',
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

            def _eval_config(eval_order, eval_params, tag_component, tag_strategy, position=None):
                X_test = self.X_test.copy()
                y_test = self.y_test.copy()

                if self.pipeline_type == 'ml':
                    X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                    X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                    _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                    X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                    X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
                else:
                    X, y, sens = X_test.copy(), y_test.copy(), None

                param_record, frac_data = [], []
                self.frac_header = []
                p = Profile()
                numerical_columns = X.select_dtypes(include=['int', 'float']).columns
                utility = None
                fraction_outlier = None
                last_handler = None

                for i, step in enumerate(eval_order):
                    param_index = int(eval_params[i]) - 1
                    handler = self._load_handler(step, param_index)
                    last_handler = handler
                    X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)
                    if frac_header is not None:
                        frac_data.append(frac_value)
                        self.frac_header.append(frac_header)
                    if util_tmp is not None:
                        utility = util_tmp
                    param_record.append(param_index + 1)

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

                df = pd.DataFrame([row], columns=col_headers)
                test_file = f'historical_data/insertion/historical_data_test_profile_{model_type}_{metric_type}_{dataset_name}.csv'
                df.to_csv(test_file, index=False)

                sim = self.profile_similarity_df(filename_training, test_file, cur_par, self.rank_profile, metric='cosine')
                logging.info(f'Intervention: component={tag_component}, strategy={tag_strategy}, position={position}, similarity={sim}, utility={utility}')
                return sim, utility

            # (A) try alternate strategies for existing steps (append directly)
            for i, component in enumerate(original_order):
                num_strategies = self.strategy_counts[component]
                current_strategy = cur_par[i]
                for new_strategy in range(1, num_strategies + 1):
                    if new_strategy == current_strategy:
                        continue
                    new_cur_par = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                    sim, utility = _eval_config(original_order, new_cur_par, component, new_strategy, position=None)
                    if sim is not None:
                        global_ranking.append((component, new_strategy, sim, utility, None))
                        if sim > best_sim:
                            best_component = component
                            best_result = new_cur_par
                            best_sim = sim
                            best_utility = utility
                            best_insert_pos = None

            # (B) try inserting new components — ONLY keep best insertion position per (component, strategy)
            for comp in new_components:
                comp_ranges = self.strategy_counts[comp]
                for strat_idx in range(comp_ranges):
                    best_for_this_strat = None  # (sim, utility, pos)
                    for insert_pos in insertion_positions:
                        new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                        new_cur_par = cur_par[:insert_pos] + [strat_idx + 1] + cur_par[insert_pos:]
                        sim, utility = _eval_config(new_order, new_cur_par, comp, strat_idx + 1, position=insert_pos)

                        if sim is None:
                            continue

                        # Track best over positions for this (comp, strategy)
                        if (best_for_this_strat is None) or (sim > best_for_this_strat[0]):
                            best_for_this_strat = (sim, utility, insert_pos)

                        if sim > best_sim:
                            best_component = comp
                            best_result = new_cur_par
                            best_sim = sim
                            best_utility = utility
                            best_insert_pos = insert_pos

                    if best_for_this_strat is not None:
                        sim_star, util_star, pos_star = best_for_this_strat
                        global_ranking.append((comp, strat_idx + 1, sim_star, util_star, pos_star))

            # sort and print

            S = np.array([t[2] for t in global_ranking], dtype=float)  # similarity (↑ better)
            U = np.array([t[3] for t in global_ranking], dtype=float)  # utility (↓ better)

            fused = self.fused_geometric_mean(S, U, wS=1.0, wU=1.0)

            #global_ranking = [(*t, float(f)) for t, f in zip(global_ranking, fused)]
            #global_ranking.sort(key=lambda x: x[3])

            # Pretty print
            for idx, (component, strategy, sim, utility, pos) in enumerate(global_ranking, start=1):
                if pos is not None:
                    print(f"New component: {component} @ {pos}, Strategy {strategy}, utility={utility:.4f}")
                else:
                    print(f"Existing component: {component}, Strategy {strategy}, utility={utility:.4f}")

            return global_ranking

            # ====== your existing body ends here ======
        finally:
            elapsed = time.perf_counter() - t0
            print(f"[evaluate_interventions1] runtime: {elapsed:.3f} s")
            try:
                logging.info(f"evaluate_interventions1_runtime_seconds={elapsed:.6f}")
            except Exception:
                pass


    
    def evaluate_interventions_ba(self, cur_par, filename_training, new_components):
        """
        Evaluate two kinds of interventions:
        (A) Parameter changes for existing steps  -> single similarity (full pipeline).
        (B) New component insertions              -> BEFORE and AFTER similarity (around the inserted step only).

        Returns
        -------
        param_change_ranking : list[tuple]
            (component, strategy, similarity, utility) sorted by similarity desc.
        insertion_before_ranking : list[tuple]
            (component, strategy, insert_pos, similarity_before) sorted by similarity desc.
        insertion_after_ranking : list[tuple]
            (component, strategy, insert_pos, similarity_after) sorted by similarity desc.
        """
        import os
        import logging
        import pandas as pd
        import numpy as np

        # Use your standard training file convention (keep behavior)
        filename_training = f'historical_data/historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
        original_order = self.pipeline_order
        insertion_positions = list(range(len(original_order)))  # keep your original rule

        # Learn profile ranking used for similarity and capture training header set
        _, self.rank_profile = self.rank_profile_new_comp(filename_training, ['outlier'])
        train_headers = self.get_header(filename_training)  # includes sens/profile cols present in training

        logging.basicConfig(
            filename=f'logs/finsertion_{dataset_name}_{model_type}_{metric_type}.log',
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        out_dir = 'historical_data/insertion'
        os.makedirs(out_dir, exist_ok=True)

        # -------------------- helpers --------------------

        def _safe_get_sens_metrics(handler, y, sens):
            """Return (sens_headers, sens_values) if the handler provides them; else empty."""
            if handler is not None and hasattr(handler, "get_profile_metric") and callable(handler.get_profile_metric):
                return handler.get_profile_metric(y, sens)
            return [], []

        def _align_to_training_schema(df_row: pd.DataFrame, eval_order_cols, frac_headers, sens_headers, key_profile):
            """
            Ensure the emitted row has the same shape/order as training:
            [pipeline params] + frac_headers + sens_headers + profile + utility.
            Adds any missing training headers with 0.0 and drops extras.
            """
            # Add any training headers missing from this snapshot (fill with 0.0)
            for c in train_headers:
                if c not in df_row.columns and c not in eval_order_cols and c not in frac_headers:
                    df_row[c] = 0.0

            # Build profile block in a stable order: sens + key_profile + (remaining train headers)
            profile_block = []
            for c in list(sens_headers) + list(key_profile):
                if c not in profile_block:
                    profile_block.append(c)
            for c in train_headers:
                if (c not in profile_block) and (c not in eval_order_cols) and (c not in frac_headers):
                    profile_block.append(c)

            final_cols = list(eval_order_cols) + list(frac_headers) + profile_block + [f'utility_{self.metric_type}']

            # Ensure all final columns exist
            for c in final_cols:
                if c not in df_row.columns:
                    df_row[c] = 0.0

            # Keep only final columns, in order; sanitize NaN/inf
            df_row = df_row[final_cols]
            df_row = df_row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return df_row

        def _emit_snapshot_row(X, y, sens, fraction_outlier,
                            eval_order_cols, param_record_vals,
                            frac_headers, frac_values,
                            sens_headers, sens_values,
                            key_profile, profile_values,
                            tag, utility=None):
            """
            Emit a single-row CSV snapshot with full schema (pipeline + frac + sens + profile + utility)
            and align it to training schema.
            """
            df = pd.DataFrame(
                [list(param_record_vals) + list(frac_values) + list(sens_values) + list(profile_values) + [utility]],
                columns=list(eval_order_cols) + list(frac_headers) + list(sens_headers) + list(key_profile) + [f'utility_{self.metric_type}']
            )
            df = _align_to_training_schema(df, eval_order_cols, frac_headers, sens_headers, key_profile)
            path = os.path.join(out_dir, f'{tag}.csv')
            df.to_csv(path, index=False)
            return path

        # ---------- A) Parameter changes: full-pipeline single similarity (same behavior as original) ----------
        def _eval_config_full(eval_order, eval_params, tag_component, tag_strategy):
            X_test = self.X_test.copy()
            y_test = self.y_test.copy()

            if self.pipeline_type == 'ml':
                X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
            else:
                X, y, sens = X_test.copy(), y_test.copy(), None

            param_record, frac_data = [], []
            self.frac_header = []
            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns
            utility = None
            fraction_outlier = None
            last_handler = None

            for i, step in enumerate(eval_order):
                param_index = int(eval_params[i]) - 1
                handler = self._load_handler(step, param_index)
                last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, bef_hdr, bef_val = self._apply_step(handler, X, y, sens)
                if bef_hdr is not None:
                    self.frac_header.append(bef_hdr)
                    frac_data.append(bef_val)
                if util_tmp is not None:
                    utility = util_tmp
                param_record.append(param_index + 1)

            # sens metrics from last handler (safely)
            sens_headers, sens_values = _safe_get_sens_metrics(last_handler, y, sens)

            # profile block
            profile_values, key_profile = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            # Emit a full-pipeline snapshot aligned to training
            test_file = _emit_snapshot_row(
                X, y, sens, fraction_outlier,
                eval_order_cols=eval_order,
                param_record_vals=param_record,
                frac_headers=self.frac_header,
                frac_values=frac_data,
                sens_headers=sens_headers,
                sens_values=sens_values,
                key_profile=key_profile,
                profile_values=profile_values,
                tag=f'full_{tag_component}_{tag_strategy}',
                utility=utility
            )

            sim = self.profile_similarity_df(filename_training, test_file, cur_par, self.rank_profile, metric='cosine')
            logging.info(f'[PARAM-CHANGE] component={tag_component}, strategy={tag_strategy}, similarity={sim}, utility={utility}')
            return sim, utility

        # ---------- B) Insertions: BEFORE/AFTER around inserted step only ----------
        def _profile_before_after(eval_order, eval_params, insert_pos, label):
            """
            Run steps up to the inserted component, snapshot BEFORE with sens/profile,
            apply inserted step once, snapshot AFTER with sens/profile. Only up to that point.
            """
            X_test = self.X_test.copy()
            y_test = self.y_test.copy()

            if self.pipeline_type == 'ml':
                X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
            else:
                X, y, sens = X_test.copy(), y_test.copy(), None

            param_record, frac_data, frac_headers = [], [], []
            fraction_outlier = None
            last_handler = None
            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns

            # Run up to (not including) inserted component
            for j, step in enumerate(eval_order[:insert_pos]):
                param_index = int(eval_params[j]) - 1
                handler = self._load_handler(step, param_index)
                last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, bef_hdr, bef_val = self._apply_step(handler, X, y, sens)
                if bef_hdr is not None:
                    frac_headers.append(bef_hdr)
                    frac_data.append(bef_val)
                param_record.append(param_index + 1)
            fraction_outlier=0.13
            # BEFORE snapshot sens/profile (safe)
            sens_headers_bef, sens_values_bef = _safe_get_sens_metrics(last_handler, y, sens)
            profile_values_bef, key_profile = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            pre_params_full = param_record + [int(v) for v in eval_params[len(param_record):]]
            before_file = _emit_snapshot_row(
                X, y, sens, fraction_outlier,
                eval_order_cols=eval_order,
                param_record_vals=pre_params_full,
                frac_headers=frac_headers,
                frac_values=frac_data,
                sens_headers=sens_headers_bef,
                sens_values=sens_values_bef,
                key_profile=key_profile,
                profile_values=profile_values_bef,
                tag=f'before_{label}',
                utility=None
            )

            # Apply inserted component once
            step = eval_order[insert_pos]
            param_index = int(eval_params[insert_pos]) - 1
            handler = self._load_handler(step, param_index)
            X, y, sens, util_tmp, fraction_outlier, bef_hdr, bef_val = self._apply_step(handler, X, y, sens)
            if bef_hdr is not None:
                frac_headers.append(bef_hdr)
                frac_data.append(bef_val)
            param_record.append(param_index + 1)

            # AFTER snapshot sens/profile (from inserted handler)
            fraction_outlier=0.13
            sens_headers_aft, sens_values_aft = _safe_get_sens_metrics(handler, y, sens)
            profile_values_aft, _ = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            post_params_full = param_record + [int(v) for v in eval_params[len(param_record):]]
            after_file = _emit_snapshot_row(
                X, y, sens, fraction_outlier,
                eval_order_cols=eval_order,
                param_record_vals=post_params_full,
                frac_headers=frac_headers,
                frac_values=frac_data,
                sens_headers=sens_headers_aft,
                sens_values=sens_values_aft,
                key_profile=key_profile,
                profile_values=profile_values_aft,
                tag=f'after_{label}',
                utility=util_tmp if isinstance(util_tmp, (float, int)) else None
            )

            return before_file, after_file

        # -------------------- collect rankings --------------------
        param_change_ranking = []      # (component, strategy, sim, utility)
        insertion_before_ranking = []  # (component, strategy, insert_pos, sim_before)
        insertion_after_ranking  = []  # (component, strategy, insert_pos, sim_after)

        # (A) PARAMETER CHANGES — full pipeline single similarity
        for i, component in enumerate(original_order):
            num_strategies = self.strategy_counts[component]
            current_strategy = cur_par[i]
            for new_strategy in range(1, num_strategies + 1):
                if new_strategy == current_strategy:
                    continue
                new_params = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                sim, utility = _eval_config_full(original_order, new_params, component, new_strategy)
                if sim is not None:
                    param_change_ranking.append((component, new_strategy, float(sim), utility))

        # (B) NEW COMPONENT INSERTIONS — BEFORE/AFTER around the inserted step
        for comp in new_components:
            comp_ranges = self.strategy_counts[comp]
            for strat_idx in range(comp_ranges):
                for insert_pos in insertion_positions:
                    new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                    new_params = cur_par[:insert_pos] + [strat_idx + 1] + cur_par[insert_pos:]

                    label = f'{comp}_ins{insert_pos}_{strat_idx + 1}'
                    before_file, after_file = _profile_before_after(
                        eval_order=new_order,
                        eval_params=new_params,
                        insert_pos=insert_pos,
                        label=label
                    )
                    filename_test = f"historical_data/noise/test_sim_historical_data_test_profile_{model_type}_{metric_type}_{dataset_name}.csv"
                    sim_before = self.profile_similarity_df(
                        filename_training, before_file, cur_par, self.rank_profile, metric='cosine'
                    )
                    sim_after = self.profile_similarity_df(
                        filename_training, after_file, cur_par, self.rank_profile, metric='cosine'
                    )

                    if sim_before is not None:
                        insertion_before_ranking.append((comp, strat_idx + 1, insert_pos, float(sim_before)))
                    if sim_after is not None:
                        insertion_after_ranking.append((comp, strat_idx + 1, insert_pos, float(sim_after)))

                    logging.info(f'[INSERT] comp={comp}@{insert_pos}, strat={strat_idx + 1}, '
                                f'sim_before={sim_before}, sim_after={sim_after}')

        # -------------------- sort & print --------------------
        param_change_ranking.sort(key=lambda x: x[2], reverse=True)
        insertion_before_ranking.sort(key=lambda x: x[3], reverse=True)
        insertion_after_ranking.sort(key=lambda x: x[3], reverse=True)

        print("\n=== Parameter-change ranking (by similarity, desc) ===")
        for comp, strat, sim, util in param_change_ranking:
            util_str = f"{util:.4f}" if isinstance(util, (float, int)) else "NA"
            print(f"{comp} -> {strat}, Similarity={sim:.4f}, Utility={util_str}")

        print("\n=== Insertion ranking: BEFORE (by similarity, desc) ===")
        for comp, strat, pos, sim in insertion_before_ranking:
            print(f"{comp}@{pos} -> {strat}, SimilarityBefore={sim:.4f}")

        print("\n=== Insertion ranking: AFTER (by similarity, desc) ===")
        for comp, strat, pos, sim in insertion_after_ranking:
            print(f"{comp}@{pos} -> {strat}, SimilarityAfter={sim:.4f}")

        return param_change_ranking, insertion_before_ranking, insertion_after_ranking




    
    def evaluate_interventions_predicted_utility(self, cur_par, filename_training, new_components):
        """
        Learn a polynomial regression model (PolynomialFeatures + linear reg) on 80% of training data,
        report R^2 and RMSE on the 20% hold-out, then predict utility for each intervention and rank.
        """
        import pandas as pd
        import numpy as np
        import logging
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error

        df_train = pd.read_csv(filename_training)
        utility_col = f'utility_{self.metric_type}'
        if utility_col not in df_train.columns:
            raise ValueError(f"Training file must contain column '{utility_col}'")

        feature_cols = [c for c in df_train.columns if c != utility_col]
        X_all = df_train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        y_all = pd.to_numeric(df_train[utility_col], errors='coerce').fillna(0.0).values

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42
        )

        base_scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_scaled = base_scaler.fit_transform(X_tr)

        poly_degree = 2
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
        X_tr_poly = poly.fit_transform(X_tr_scaled)

        reg = Regression()
        model = reg.generate_regression(X_tr_poly, y_tr)

        X_val_scaled = base_scaler.transform(X_val)
        X_val_poly = poly.transform(X_val_scaled)
        y_val_pred = np.ravel(model.predict(X_val_poly))

        r2_val = r2_score(y_val, y_val_pred)
        rmse_val = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))

        print(f"[Polynomial utility model] Hold-out (20%)  R^2 = {r2_val:.4f} | RMSE = {rmse_val:.4f}")

        logging.basicConfig(
            filename=f'logs/fpred_{self.dataset_name}_{self.model_selection[0]}_{self.metric_type}.log',
            filemode='w', level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Hold-out (20%%) performance: R^2={r2_val:.6f}, RMSE={rmse_val:.6f}")

        # ---- helpers (same as before), but they will use the *trained* scaler & poly ----
        def _build_feature_row(eval_order, eval_params):
            X_test = self.X_test.copy()
            y_test = self.y_test.copy()

            if self.pipeline_type == 'ml':
                X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
            else:
                X, y, sens = X_test.copy(), y_test.copy(), None

            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns

            param_record, frac_data = [], []
            self.frac_header = []
            utility = None
            fraction_outlier = None
            last_handler = None

            for i, step in enumerate(eval_order):
                param_index = int(eval_params[i]) - 1
                handler = self._load_handler(step, param_index)
                last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)
                if frac_header is not None:
                    frac_data.append(frac_value)
                    self.frac_header.append(frac_header)
                if util_tmp is not None:
                    utility = util_tmp
                param_record.append(param_index + 1)

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
            return row_df, out_cols

        def _align_features(test_row_df):
            # align columns to training feature set (pre-expansion)
            missing = [c for c in feature_cols if c not in test_row_df.columns]
            for c in missing:
                test_row_df[c] = 0.0
            extras = [c for c in test_row_df.columns if c not in feature_cols and c != utility_col]
            if extras:
                test_row_df = test_row_df.drop(columns=extras)
            test_row_df = test_row_df.reindex(columns=feature_cols)
            test_row_df = test_row_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            return test_row_df

        def _to_poly_space(X_df):
            X_scaled_row = base_scaler.transform(X_df.values)
            X_poly_row = poly.transform(X_scaled_row)
            return X_poly_row

        # ---- 6) Evaluate all interventions with the trained model ----
        original_order = self.pipeline_order
        insertion_positions = list(range(2, len(original_order)))
        results = []  # (component, strategy, insert_pos or None, y_pred)

        # (A) Alternate strategies for existing components
        for i, component in enumerate(original_order):
            num_strategies = self.strategy_counts[component]
            current_strategy = cur_par[i]
            for new_strategy in range(1, num_strategies + 1):
                if new_strategy == current_strategy:
                    continue
                new_cur_par = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                test_row_df, _ = _build_feature_row(original_order, new_cur_par)
                X_test = _align_features(test_row_df.drop(columns=[utility_col], errors='ignore'))
                X_poly_row = _to_poly_space(X_test)
                y_pred = float(np.ravel(model.predict(X_poly_row))[0])
                logging.info(f'[ALT] {component} -> {new_strategy}, predicted_utility={y_pred:.6f}')
                results.append((component, new_strategy, None, y_pred))

        # (B) Insert new components
        for comp in new_components:
            comp_ranges = self.strategy_counts[comp]
            for strat_idx in range(comp_ranges):
                for insert_pos in insertion_positions:
                    new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                    new_cur_par = cur_par[:insert_pos] + [strat_idx + 1] + cur_par[insert_pos:]
                    test_row_df, _ = _build_feature_row(new_order, new_cur_par)
                    X_test = _align_features(test_row_df.drop(columns=[utility_col], errors='ignore'))
                    X_poly_row = _to_poly_space(X_test)
                    y_pred = float(np.ravel(model.predict(X_poly_row))[0])
                    logging.info(f'[INS] {comp}@{insert_pos} -> {strat_idx + 1}, predicted_utility={y_pred:.6f}')
                    results.append((comp, strat_idx + 1, insert_pos, y_pred))

        # ---- 7) Rank & print (lower predicted utility is better) ----
        results.sort(key=lambda x: abs(x[3]))
        for component, strategy, pos, y_pred in results:
            if pos is None:
                print(f"{component} -> {strategy}, PredictedUtility={y_pred:.4f}")
            else:
                print(f"{component}@{pos} -> {strategy}, PredictedUtility={y_pred:.4f}")

        return results





    # ---------- misc ----------
    def get_profile(self):
        return getattr(self, 'profile', None)

    def run_pipeline(self, alg_type, file_name):
        if alg_type == 'opaque':
            self.run_pipeline_opaque(file_name)
        elif alg_type == 'glass':
            self.run_pipeline_glass(file_name)

    # kept to preserve behavior where used
    def inject_missing_values(self, X, tau):
        """Wrapper to match existing calls. Delegates to noise_injector for 'missing'."""
        return self.noise_injector.inject_noise(X, noise_type='missing', frac=tau)



dataset_name = 'adult'
metric_type = 'sp'
model_type = 'lr'
pipeline_order = ['missing_value','normalization','model']
new_comp=['punctuation','outlier','stopword']
cur_par=[2,1,1]


output = f'historical_data/similarity/Noisy_vs_orig_{model_type}_{metric_type}_{dataset_name}.csv'
output1 = f'historical_data/similarity/Noisy_vs_filterd_orig_{model_type}_{metric_type}_{dataset_name}.csv'
filename_test = f"historical_data/noise/test_sim_historical_data_test_profile_{model_type}_{metric_type}_{dataset_name}.csv"
filename_train = f'historical_data/partial_pipeline/sim_historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
filename_train1 = f'historical_data/historical_data_train_profile{model_type}_{metric_type}_{dataset_name}.csv'



executor = PipelineExecutor(
                pipeline_type='ml',
                dataset_name=dataset_name,
                metric_type=metric_type,
                pipeline_ord=pipeline_order,
                execution_type='pass',
            )


#executor.run_pipeline_glass(filename_test)
#_, rank_profile=executor.rank_profile_new_comp(filename_train1, new_comp)
#rank_profile.remove('corr_Country')

#executor.evaluate_parameter_intervention(cur_par, filename_train)
#executor.evaluate_with_component_insertion(cur_par, new_components=new_comp)
#executor.evaluate_interventions1(cur_par, filename_train, new_components=new_comp)
#print(rank)
#executor.profile_similarity_all_rows(filename_test, filename_test, pd.read_csv(filename_train), output1)
#executor.profile_similarity_df(filename_train, filename_test, cur_par, pd.read_csv(filename_test), metric='cosine')
#Change the cur_par_lookUp fucntion


#similarities = executor.profile_similarity_df_filtered(filename_train,filename_test,param=cur_par,profile_cols=pd.read_csv(filename_train).columns,threshold_col='utility_sp',threshold_val=0.055,output_file=output1,metric='cosine')'''
