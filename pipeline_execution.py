import os
import pandas as pd
import numpy as np
import logging
import random
import math
from sklearn.preprocessing import StandardScaler
from regression import Regression
from modules.outlier_detection.outlier_detector import OutlierDetector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from LoadDataset import LoadDataset
from modules.profiling.profile import Profile
import importlib
import itertools
from sklearn.metrics.pairwise import cosine_similarity

class PipelineExecutor:
    def __init__(self, pipeline_type, dataset_name, metric_type, pipeline_ord, execution_type='pass', h_sample_bool=False, scalability_bool=False):
        
        self.pipeline_type = pipeline_type # 'ml' or 'em'
        if self.pipeline_type == 'ml':
            self.dataset_name = dataset_name
            self.metric_type = metric_type
            self.h_sample_bool = h_sample_bool
            if self.h_sample_bool:
                self.h_sample = 0.005 # make it as a paramater
            self.scalability_bool = scalability_bool
            self.mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
            self.norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm']
            self.fselection_strategy = ['mu']#['none', 'va', 'mu']
            self.od_strategy = ['none', 'if', 'lof']
            self.model_selection = ['lr'] # rf, 'reg' # 'nb'
            self.knn_k_lst = [1, 5, 10, 20, 30]
            self.lof_k_lst = [1, 5, 10, 20, 30]
            self.pipeline_order = pipeline_ord # always keep model at the end
            self.execution_type = execution_type # 'pass' or 'fail'

            loader = LoadDataset(self.dataset_name)
            self.dataset, self.X_train, self.y_train, self.X_test, self.y_test = loader.load()
            self.sensitive_var = loader.get_sensitive_variable()
            self.target_variable_name = self.y_train.name

            #known domain
            self.strategy_counts = {
                'missing_value': len(self.mv_strategy) + len(self.knn_k_lst) - 1 if 'knn' in self.mv_strategy else len(self.mv_strategy),
                'normalization': len(self.norm_strategy),
                'outlier': len(self.od_strategy) + len(self.lof_k_lst) - 1 if 'lof' in self.od_strategy else len(self.od_strategy),
                'fselection': len(self.fselection_strategy),
                'model': len(self.model_selection)
            }


            if dataset_name == 'adult':
                    if self.execution_type == 'pass':
                        self.tau = 0.1 # fraction of missing values
                        self.contamination = 0.2
                        self.contamination_lof = 'auto'
                    else:
                        self.tau = 0.1
                        self.contamination = 0.2
                        self.contamination_lof = 'auto'
            elif dataset_name == 'hmda':
                    if self.execution_type == 'pass':
                        self.tau = 0.05 # fraction of missing values
                        self.contamination = 0.1
                        self.contamination_lof = 0.1
                    else:
                        self.tau = 0.1
                        self.contamination = 0.2
                        self.contamination_lof = 0.2

            elif dataset_name == 'housing':
                    
                    if self.execution_type == 'pass':
                        self.tau = 0.2 # fraction of missing values
                        self.contamination = 0.3
                        self.contamination_lof = 0.3
                    else:
                        self.tau = 0.1
                        self.contamination = 0.2
                        self.contamination_lof = 0.2
            else:
                raise ValueError("Invalid dataset name for ML pipeline. Supported datasets are: 'adult', 'hmda', 'housing'.")
            
            self.shared_config = {
                'mv_strategy': self.mv_strategy,
                'fs_strategy': self.fselection_strategy,
                'knn_k_lst': self.knn_k_lst,
                'norm_strategy': self.norm_strategy,
                'od_strategy': self.od_strategy,
                'lof_k_lst': self.lof_k_lst,
                'contamination': self.contamination,
                'contamination_lof': self.contamination_lof,
                'model_selection': self.model_selection,
                'metric_type': self.metric_type,
                'sensitive_var': self.sensitive_var,
                'target_var': self.target_variable_name
            }
        
        elif self.pipeline_type == 'em':
            print('pipeline_type is em')
    
    def set_dataset(self, dataset):
        self.numerical_columns = dataset.select_dtypes(include=['int', 'float']).columns
        self.categorical_columns = dataset.select_dtypes(include=['object']).columns

    def getIdxSensitive(self, df, sensitive_var):
        if self.metric_type == 'sp':
            sensitive_attr = df[sensitive_var]
        else:
            sensitive_attr = None
        return sensitive_attr


    def inject_missing_values(self, X, frac):
        X_modified = X.copy()
        idx_train = np.arange(0, len(X_modified), 1)
        mv_train = pd.DataFrame(idx_train).sample(frac=frac, replace=False, random_state=1).index
        if self.pipeline_type == 'ml':
            if self.dataset_name == 'hmda':
                X_modified.loc[mv_train, 'lien_status'] = np.nan
            elif self.dataset_name == 'adult':
                X_modified.loc[mv_train, 'Martial_Status'] = np.nan
            elif self.dataset_name == 'housing':
                X_modified.loc[mv_train, 'OverallQual'] = np.nan
            return X_modified
        
    '''def inject_outliers(self, X, frac, multiplier=5.0):
        X_modified = X.copy()
        idx_train = np.arange(len(X_modified))
        outlier_indices = pd.DataFrame(idx_train).sample(frac=frac, replace=False, random_state=42).index

        if self.pipeline_type == 'ml':
            if self.dataset_name == 'hmda':
                col = 'lien_status'
            elif self.dataset_name == 'adult':
                col = 'Education_Num'
            elif self.dataset_name == 'housing':
                col = 'OverallQual'
            else:
                return X_modified

            if not pd.api.types.is_numeric_dtype(X_modified[col]):
                print(f"[WARNING] Column {col} is not numeric. Outlier injection skipped.")
                return X_modified

            Q1 = X_modified[col].quantile(0.25)
            Q3 = X_modified[col].quantile(0.75)
            IQR = Q3 - Q1
            high_outlier = Q3 + multiplier * IQR
            low_outlier = Q1 - multiplier * IQR

            for i, idx in enumerate(outlier_indices):
                X_modified.at[idx, col] = high_outlier if i % 2 == 0 else low_outlier

        return X_modified'''
    
    def inject_outliers(self, X, frac, multiplier=5.0):
        X_modified = X.copy()
        
        idx_train = np.arange(len(X_modified))
        outlier_indices = pd.DataFrame(idx_train).sample(frac=frac, replace=False, random_state=42).index

        # === Step 1: Select column based on pipeline type and dataset ===
        if self.pipeline_type == 'ml':
            if self.dataset_name == 'hmda':
                col = 'lien_status'
            elif self.dataset_name == 'adult':
                col = 'Education_Num'
            elif self.dataset_name == 'housing':
                col = 'OverallQual'
            else:
                return X_modified

            if not pd.api.types.is_numeric_dtype(X_modified[col]):
                print(f"[WARNING] Column {col} is not numeric. Outlier injection skipped.")
                return X_modified

            # === Step 2: SVM fit BEFORE injection ===
            values = X_modified[col].values.reshape(-1, 1)
            scaler = StandardScaler()
            values_scaled = scaler.fit_transform(values)
            svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
            svm.fit(values_scaled)
            pred_before = svm.predict(values_scaled)
            outlier_frac_before = round((pred_before == -1).sum() / len(pred_before) * 100, 4)


            print(f"[Before Injection] Outliers detected: {outlier_frac_before}%")

            '''# === Step 3: Inject synthetic outliers ===
            col_mean = np.mean(values_scaled)
            col_std = np.std(values_scaled)
            synthetic_outliers_scaled = np.random.choice([-1, 1], size=len(outlier_indices)) * (multiplier * col_std)
            synthetic_outliers = scaler.inverse_transform(synthetic_outliers_scaled.reshape(-1, 1)).flatten()

            for i, idx in enumerate(outlier_indices):
                X_modified.at[idx, col] = synthetic_outliers[i]

            # === Step 4: SVM fit AFTER injection ===
            values_after = X_modified[col].values.reshape(-1, 1)
            values_after_scaled = scaler.transform(values_after)
            pred_after = svm.predict(values_after_scaled)
            outlier_frac_after = round((pred_after == -1).sum() / len(pred_after) * 100, 4)'''
            Q1 = X_modified[col].quantile(0.25)
            Q3 = X_modified[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
               
            high_outlier_value = Q3 +  multiplier* IQR
            X_modified.loc[outlier_indices, col] = high_outlier_value

            '''values = X_modified.values.reshape(-1, 1)
            scaler = StandardScaler()
            values_scaled = scaler.fit_transform(values)
            svm.fit(values_scaled)
            pred_before = svm.predict(values_scaled)
            outlier_frac_after = round((pred_before == -1).sum() / len(pred_before) * 100, 4)'''
            
            
            #print(f"[After Injection] Outliers detected: {outlier_frac_after}%")

        return X_modified


    def run_pipeline_opaque(self, file_name):
        if self.execution_type == 'pass':
            X_copy, y_copy = self.X_train.copy(), self.y_train.copy()
        elif self.execution_type == 'fail':
            X_copy, y_copy = self.X_test.copy(), self.y_test.copy()

        if os.path.exists(file_name):
            self.param_lst_df = pd.read_csv(file_name)[self.pipeline_order + [f'utility_{self.metric_type}']]
            return
        
        if self.pipeline_type == 'ml':
            X_copy = self.inject_missing_values(X_copy, self.tau)
            sensitive_attr_train = self.getIdxSensitive(X_copy, self.sensitive_var)

        param_ranges = [range(self.strategy_counts[step]) for step in self.pipeline_order]
        param_combinations = itertools.product(*param_ranges)

        param_lst = []
        print("Running pipeline combinations...")

        for combo in param_combinations:
            if self.pipeline_type == 'ml':
                X, y, sens = X_copy.copy(), y_copy.copy(), sensitive_attr_train.copy()
            
            param_record = []

            for i, step in enumerate(self.pipeline_order):
                param_index = combo[i]
                module_name = f"pipeline_component.{step}_handler"
                class_name = ''.join(w.capitalize() for w in step.split('_')) + "Handler"

                try:
                    handler_module = importlib.import_module(module_name)
                    handler_class = getattr(handler_module, class_name)
                except (ModuleNotFoundError, AttributeError) as e:
                    raise ImportError(f"Error loading handler for '{step}': {e}")

                handler = handler_class(strategy=param_index, config=self.shared_config)
                result = handler.apply(X, y, sens)

                if isinstance(result, float) or isinstance(result, int):
                    utility = result
                elif isinstance(result, tuple):
                    X, y, sens = result
                else:
                    raise ValueError(f"Expected tuple output from {class_name}.apply")

                param_record.append(param_index + 1)

            param_lst.append(param_record + [utility])
            print(param_record + [utility])

        self.param_lst_df = pd.DataFrame(param_lst, columns=self.pipeline_order + [f'utility_{self.metric_type}'])
        self.param_lst_df.to_csv(file_name, index=False)


    def score_parameter(self, param_lst_df):
        if self.pipeline_type == 'ml':
            param_lst_df = param_lst_df.copy()

            y = param_lst_df[f'utility_{self.metric_type}']
            X = param_lst_df.drop([f'utility_{self.metric_type}'], axis=1)

            if self.h_sample_bool:
                import math
                print(" ------ Sampling --------", self.h_sample)
                random.seed(42)
                sample_idx = random.sample(list(range(len(X))), math.ceil(self.h_sample * len(X)))
                X = X.iloc[sample_idx]
                y = y.iloc[sample_idx]

            reg = Regression()
            model = reg.generate_regression(X, y)
            coefs = model.coef_
            print('coefficient', coefs)
            print('Intercept',model.intercept_)
            coefs = coefs
            coef_rank = np.argsort(np.abs(coefs)).tolist()[::-1]
            print('ranking',coef_rank)
            logging.info(f'coef {coefs}')

            return coefs, coef_rank


    def current_par_lookup(self, cur_par=[]):
        if len(cur_par) != len(self.pipeline_order):
            raise ValueError("Mismatch between cur_par and pipeline_order.")

        X_test = self.X_test.copy()
        #X_test= self.inject_outliers(self.X_test.copy(), frac=.5, multiplier=3.0)
        y_test = self.y_test.copy()

        if self.pipeline_type== 'ml':
            X_test = self.inject_missing_values(X_test, self.tau)
            sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
            X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()

        for i, step in enumerate(self.pipeline_order):
            param_index = int(cur_par[i]) - 1
            module_name = f"pipeline_component.{step}_handler"
            class_name = ''.join(w.capitalize() for w in step.split('_')) + "Handler"

            try:
                handler_module = importlib.import_module(module_name)
                handler_class = getattr(handler_module, class_name)
            except (ModuleNotFoundError, AttributeError) as e:
                raise ImportError(f"Error loading handler for '{step}': {e}")

            handler = handler_class(strategy=param_index, config=self.shared_config)
            result = handler.apply(X, y, sens)

            if isinstance(result, float) or isinstance(result, int):
                return result
            else:
                X, y, sens = result

        raise ValueError("No output returned from final pipeline component.")
                    
    def run_pipeline_glass(self, file_name):
        if self.execution_type == 'pass':
            X_copy, y_copy = self.X_train.copy(), self.y_train.copy()
        elif self.execution_type == 'fail':
            X_copy, y_copy = self.X_test.copy(), self.y_test.copy()
            X_copy= self.inject_outliers(X_copy, frac=.2, multiplier=5.0)

        if os.path.exists(file_name):
            self.param_lst_df = pd.read_csv(file_name)
            return
        
        if self.pipeline_type == 'ml':
            X_copy = self.inject_missing_values(X_copy, self.tau)
            sensitive_attr_train = self.getIdxSensitive(X_copy, self.sensitive_var)
        
        p = Profile()

        param_ranges = [range(self.strategy_counts[step]) for step in self.pipeline_order]
        param_combinations = itertools.product(*param_ranges)
        
        param_lst = []
        print("Running pipeline combinations...")
        
        for combo in param_combinations:
            if self.pipeline_type == 'ml':
                X, y, sens = X_copy.copy(), y_copy.copy(), sensitive_attr_train.copy()
            
            param_record = []
            frac_data = []
            self.frac_header = []
            

            for i, step in enumerate(self.pipeline_order):
                param_index = combo[i]
                module_name = f"pipeline_component.{step}_handler"
                class_name = ''.join(w.capitalize() for w in step.split('_')) + "Handler"

                try:
                    handler_module = importlib.import_module(module_name)
                    handler_class = getattr(handler_module, class_name)
                except (ModuleNotFoundError, AttributeError) as e:
                    raise ImportError(f"Error loading handler for '{step}': {e}")

                handler = handler_class(strategy=param_index, config=self.shared_config)
                result = handler.apply(X, y, sens)

                method_name = f'get_outlier_bef_{step}_strat'
                if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
                    method = getattr(handler, method_name)
                    frac = method()
                    frac_data.append(frac)
                    self.frac_header.append(f'outlier_bef_{step}_strat')
                
                method_name = f'get_outlier_after_{step}_strat'
                if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
                    method = getattr(handler, method_name)
                    frac = method()
                    frac_data.append(frac)
                    self.frac_header.append(f'outlier_after_{step}_strat')
                print(self.frac_header)
                

                method_name = f'get_{step}'
                if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
                    method = getattr(handler, method_name)
                    fraction_outlier = method()
                else:
                    detector = OutlierDetector(X)
                    _, _, _ = detector.transform(y, sensitive_attr_train=None)
                    fraction_outlier = detector.get_frac()

                if isinstance(result, float) or isinstance(result, int):
                    utility = result
                elif isinstance(result, tuple):
                    X, y, sens = result
                else:
                    raise ValueError(f"Expected tuple output from {class_name}.apply")
                param_record.append(param_index + 1)
            print(param_record)
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns

            self.headers, sens_data = handler.get_profile_metric(self.y_train)
            prof_data = frac_data + sens_data
            profile_gen, key_profile = p.populate_profiles(pd.concat([X, y], axis=1), numerical_columns, self.target_variable_name, fraction_outlier, self.metric_type)
            param_lst.append(param_record + prof_data + profile_gen + [utility])


        profile_param_lst_df = pd.DataFrame(param_lst, columns= self.pipeline_order + self.frac_header + self.headers +key_profile+[f'utility_{self.metric_type}'])
        profile_param_lst_df = profile_param_lst_df.loc[:, (profile_param_lst_df != 0).any(axis=0)]
        profile_param_lst_df.to_csv(file_name, index=False)

    
    def get_header(self, file_name):
        df = pd.read_csv(file_name)
        known_cols = set(self.pipeline_order + [f'utility_{self.metric_type}'])
        extra_cols = [
            col for col in df.columns
            if col not in known_cols and not col.startswith('ot_') and not col.startswith('cov') and not col.startswith('outlier') and not col.startswith('insertion_pos')
        ]

        self.headers = extra_cols

        return self.headers
    


    def rank_profile_parameter(self, file_name):
        param_lst_df = pd.read_csv(file_name)
        #if self.h_sample_bool and self: #need to update according to logic #From SHAP
        profiles = self.get_header(file_name)
        if self.model_selection == 'reg':
            y = param_lst_df[f'utility_{self.metric_type}']                  
            t = StandardScaler().fit(param_lst_df).transform(param_lst_df)
            X = pd.DataFrame(data = t, columns = param_lst_df.columns)[profiles]
        else :
            y = param_lst_df[f'utility_{self.metric_type}']
            X = param_lst_df.copy()[profiles]       
                
        if (self.h_sample_bool):
            print(" ------ Sampling --------", self.h_sample)
            random.seed(1000)
            sample_idx = random.sample(list(range(len(X))), math.ceil(self.h_sample * len(X)))
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]

        reg = Regression()
        model = reg.generate_regression(X, y)
        coefs = model.coef_
        #print(coefs)
        #print(model.intercept_)
                
        self.profile_ranking = np.argsort(np.abs(coefs))[::-1]
        self.profile_coefs = coefs
        #print(self.profile_coefs)

        #ranking parameter 
        param_columns=self.pipeline_order
        self.ranking_param ={}
        self.param_coeff  = {}
        for index, elem in enumerate(profiles):
                y = param_lst_df[elem]
                X = param_lst_df.copy()[param_columns]
                if self.h_sample_bool:
                            X = X.iloc[sample_idx]
                            y = y.iloc[sample_idx]
                reg = Regression()
                model = reg.generate_regression(X, y)
                coefs = model.coef_
                # print(model.intercept_)
                self.ranking_param[elem] =  np.argsort(np.abs(coefs))[::-1]
                #print(self.ranking_param[elem])
                        
                self.param_coeff[elem] =  coefs
                #print(f'name : {elem} {self.param_coeff[elem]}')

        #for idx,profile_index in enumerate(self.profile_ranking):
                #print(self.profiles[profile_index])

        return self.profile_coefs, self.profile_ranking, self.param_coeff, self.ranking_param
    



    # Insertion of new component in the pipeline

    def rank_profile_new_comp(self, file_name, new_comp):
        param_lst_df = pd.read_csv(file_name)
        profiles = self.get_header(file_name)
        print("[INFO] Profiles (features):", profiles)
        logging.basicConfig(filename='logs/franking'+"_"+dataset_name+'_'+model_type+'_'+metric_type+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
        #print("Coefficient Matrix:")
        #print(coef_df)

        avg_coefs = coef_df.mean(axis=0)
        profile_ranking = avg_coefs.sort_values(ascending=False).index.tolist()
        profile_coefs = avg_coefs.sort_values(ascending=False).values

        #We could find the top K

        print("[INFO] Ranked Profiles:", profile_ranking)
        print("[INFO] Corresponding Average Coefficients:", profile_coefs)
        logging.info(f'Ranked Profiles: {profile_ranking}')
        logging.info(f'Corresponding Average Coefficients: {profile_coefs}')
        return coef_df, profile_ranking


    def profile_similarity_df(self, file_train, file_test, param, profile_cols):

        df1 = pd.read_csv(file_train)
        df2 = pd.read_csv(file_test)


        param_dict = dict(zip(self.pipeline_order, param))


        for col, val in param_dict.items():
            if col in df1.columns:
                df1 = df1[df1[col].astype(str) == str(val)]
            if col in df2.columns:
                df2 = df2[df2[col].astype(str) == str(val)]
        if df1.empty or df2.empty:
            print("Pipeline parameter not found in one or both datasets.")
            return None
        
        
        v1 = df1[profile_cols].iloc[0].astype(float).values.reshape(1, -1)
        
        v2 = df2[profile_cols].iloc[0].astype(float).values.reshape(1, -1)
        logging.info(f'profile for passing: {v2}')
        logging.info(f'profile for failing: {v1}')
        similarity = cosine_similarity(v1, v2)[0][0]
        print("Cosine Similarity:", similarity)
        return similarity

    
    def profile_similarity_all_rows(self, file_train, file_test, profile_cols, output_file):
        df1 = pd.read_csv(file_train)
        df2 = pd.read_csv(file_test)
        
        df1['param_key'] = df1.iloc[:, :len(self.pipeline_order)-1].astype(str).agg(','.join, axis=1)
        df2['param_key'] = df2.iloc[:, :len(self.pipeline_order)-1].astype(str).agg(','.join, axis=1)

        # Keep only columns in profile_cols that exist in both dataframes
        shared_profile_cols = [col for col in profile_cols if col in df1.columns and col in df2.columns]
        
        if not shared_profile_cols:
            raise ValueError("No shared profile columns found between the datasets.")

        similarities = []

        for idx, row in df1.iterrows():
            param = row.iloc[:len(self.pipeline_order)].astype(str).tolist()
            
            key1 = ','.join(param[:2])  # You might want to generalize this if param[:2] is not flexible
            key2 = ','.join(param[:2])

            row1 = df1[df1['param_key'] == key1]
            row2 = df2[df2['param_key'] == key2]

            if row1.empty or row2.empty:
                print(f"Row {idx}: Pipeline parameter not found.")
                similarities.append(None)
                continue

            v1 = row1[shared_profile_cols].iloc[0].values.astype(float).reshape(1, -1)
            v2 = row2[shared_profile_cols].iloc[0].values.astype(float).reshape(1, -1)

            similarity = cosine_similarity(v1, v2)[0][0]
            similarities.append(similarity)

        param_cols = df1.columns[:len(self.pipeline_order)].tolist()
        df_similarity = df1[param_cols].copy()
        df_similarity['similarities'] = similarities
        df_similarity.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    
    def evaluate_with_component_insertion(self, cur_par, new_components):
        #filename_training = pd.read_csv(file_name)
        filename_training = f'historical_data/historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
        
        original_order = self.pipeline_order
        insertion_positions = list(range(1, len(original_order)))
        _, self.rank_profile=self.rank_profile_new_comp(filename_training, ['outlier'])
        best_result = None
        best_utility = -float('inf')
        best_insert_pos = None
        logging.basicConfig(filename='logs/finsertion'+"_"+dataset_name+'_'+model_type+'_'+'metric_type'+'.log', filemode = 'w',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f'original order: {original_order}')
        logging.info(f'Initial Pipeline: {cur_par}')
        cur_f=self.current_par_lookup(cur_par)
        logging.info(f'Initial Utility: {cur_f}')
        for comp in new_components:
            logging.info(f'New component: {comp}')
            comp_ranges = self.strategy_counts[comp]
            for idx in range(comp_ranges):
                for insert_pos in insertion_positions:
                    new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                    new_cur_par = cur_par[:insert_pos] + [idx+1] + cur_par[insert_pos:]
                    if len(new_cur_par) != len(new_order):
                        raise ValueError("Parameter and pipeline length mismatch after insertion.")

                    #X_test = self.X_test.copy()
                    y_test = self.y_test.copy()
                    X_test= self.inject_outliers(self.X_test, frac=.1, multiplier=3.0)
                    

                    if self.pipeline_type == 'ml':
                        X_test = self.inject_missing_values(X_test, self.tau)
                        sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                        X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()

                    param_record = []
                    frac_data = []
                    self.frac_header = []

                    p = Profile()
                    numerical_columns = X.select_dtypes(include=['int', 'float']).columns

                    for i, step in enumerate(new_order):
                        param_index = new_cur_par[i]-1
                        module_name = f"pipeline_component.{step}_handler"
                        class_name = ''.join(w.capitalize() for w in step.split('_')) + "Handler"

                        try:
                            handler_module = importlib.import_module(module_name)
                            handler_class = getattr(handler_module, class_name)
                        except (ModuleNotFoundError, AttributeError) as e:
                            raise ImportError(f"Error loading handler for '{step}': {e}")

                        handler = handler_class(strategy=param_index, config=self.shared_config)
                        result = handler.apply(X, y, sens)

                        method_name = f'get_outlier_bef_{step}_strat'
                        if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
                            method = getattr(handler, method_name)
                            frac = method()
                            frac_data.append(frac)
                            self.frac_header.append(f'outlier_bef_{step}_strat')
                        

                        method_name = f'get_{step}'
                        if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
                            method = getattr(handler, method_name)
                            fraction_outlier = method()
                        else:
                            detector = OutlierDetector(X)
                            _, _, _ = detector.transform(y, sensitive_attr_train=None)
                            fraction_outlier = detector.get_frac()

                        if isinstance(result, float) or isinstance(result, int):
                            utility = result
                        elif isinstance(result, tuple):
                            X, y, sens = result
                        else:
                            raise ValueError(f"Expected tuple output from {class_name}.apply")

                        param_record.append(param_index + 1)
                    print(param_record)
                    self.headers, sens_data = handler.get_profile_metric(self.y_test)
                    prof_data = frac_data + sens_data
                    profile_gen, key_profile = p.populate_profiles(pd.concat([X, y], axis=1), numerical_columns, self.target_variable_name, fraction_outlier, self.metric_type)
                    row = param_record + prof_data + profile_gen + [utility] + [insert_pos]
                    
                    profile_data = []
                    profile_data.append(row)
                    out_cols = original_order[:insert_pos] + new_components + original_order[insert_pos:]
                    col_headers = out_cols + self.frac_header + self.headers + key_profile + [f'utility_{self.metric_type}', 'insertion_pos']
                    df = pd.DataFrame(profile_data, columns=col_headers)
                    file=f'historical_data/insertion/historical_data_test_profile_{model_type}_{metric_type}_{dataset_name}.csv'
                    
                    #profile similarity

                    df.to_csv(file, index=False)
                    filename_training = f'historical_data/historical_data_sim_profile_{model_type}_{metric_type}_{dataset_name}.csv'
                    sim=self.profile_similarity_df(filename_training, file, cur_par, self.rank_profile)
                    logging.info(f'New pipeline: {new_order} and new configuration: {new_cur_par}')
                    logging.info(f'similarity is - {sim}, Utility is - {utility}')

                    #evaluation

                    if utility > best_utility:
                        best_result = param_record
                        best_utility = utility
                        best_insert_pos = insert_pos

        '''print("Best result:")
        print("Best Insertion position:", best_insert_pos)
        print("Passing Params:", best_result)
        print("Utility:", best_utility)
        out_cols = original_order[:best_insert_pos] + new_components + original_order[best_insert_pos:]
        col_headers = out_cols + self.frac_header + self.headers + key_profile + [f'utility_{self.metric_type}', 'insertion_pos']
        df = pd.DataFrame(profile_data, columns=col_headers)

        #profile similarity

        df.to_csv(file_name, index=False)
        df1 = pd.read_csv(f'historical_data/historical_data_trainoooo_profile_{model_type}_{metric_type}_{dataset_name}.csv')
        similarity= self.profile_similarity(df, df1, cur_par, self.rank_profile)
        print(f"[INFO] Cosine Similarity with historical profile: {similarity}")'''


        return #best_insert_pos, best_result, best_utility
    
    def get_profile(self):
        return self.profile
    
    def run_pipeline(self, alg_type, file_name):
        if alg_type == 'opaque':
            self.run_pipeline_opaque(file_name)
        elif alg_type == 'glass':
            self.run_pipeline_glass(file_name)
    


dataset_name = 'housing'
metric_type = 'rmse'
model_type = 'reg'
pipeline_order = ['missing_value','normalization','outlier','model']
new_comp=['outlier']#['fselection']
cur_par=[6, 1, 1] #make sure it passes the threshold
#new_comp_par=[1]



#filename_train = f'historical_data/historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
#filename_test = f'historical_data/historical_data_sim_test_profile_{model_type}_{metric_type}_{dataset_name}.csv'
output = f'historical_data/similarity/P2_F2_{model_type}_{metric_type}_{dataset_name}.csv'
filename_test = f'historical_data/noise/full_exp_historical_data_test_profile_{model_type}_{metric_type}_{dataset_name}.csv'
filename_train = f'historical_data/historical_data_profile_{model_type}_{metric_type}_{dataset_name}.csv'
filename_train1 = f'historical_data/historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
#filename_test = f'historical_data/fselection/mu_historical_data_test_profile_{model_type}_{metric_type}_{dataset_name}.csv'
#filename_test = f'historical_data/noise/full_exp_historical_data_test_profile_{model_type}_{metric_type}_{dataset_name}.csv'

executor = PipelineExecutor(
                pipeline_type='ml',
                dataset_name=dataset_name,
                metric_type=metric_type,
                pipeline_ord=pipeline_order,
                execution_type='fail',
            )

executor.run_pipeline_glass(filename_train) #chagne the function
#_, rank_profile=executor.rank_profile_new_comp(filename_train1, new_comp)
#rank_profile.remove('corr_Race')
#rank_profile.remove('corr_Country')
#executor.inject_outliers(frac=.5, multiplier=5.0) #inject outliers
#filename_train = f'historical_data/historical_data_sim_profile_{model_type}_{metric_type}_{dataset_name}.csv'
#filename_test = f'historical_data/historical_data_sim_test_profile_{model_type}_{metric_type}_{dataset_name}.csv'
#output = f'historical_data/similarity/cosine_similarity_alter_{model_type}_{metric_type}_{dataset_name}.csv'
#print(executor.get_header(filename_train))
#executor.evaluate_with_component_insertion(cur_par, new_components=new_comp)
#executor.profile_similarity_all_rows(filename_test, filename_train, rank_profile, output)
#executor.profile_similarity_df(filename_train, filename_test, cur_par, rank_profile)
#Change the cur_par_lookUp fucntion