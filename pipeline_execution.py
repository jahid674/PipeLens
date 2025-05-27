import os
import pandas as pd
import numpy as np
import logging
import random
import math
from sklearn.preprocessing import StandardScaler
from regression import Regression
from LoadDataset import LoadDataset
from modules.profiling.profile import Profile
import importlib
import itertools

class PipelineExecutor:
    def __init__(self, pipeline_type, dataset_name, metric_type, pipeline_ord, execution_type='pass', h_sample_bool=False, scalability_bool=False):
        
        self.pipeline_type = pipeline_type # 'ml' or 'em'
        if self.pipeline_type == 'ml':
            self.dataset_name = dataset_name
            self.metric_type = metric_type
            self.h_sample_bool = h_sample_bool
            if self.h_sample_bool:
                self.h_sample = 0.005
            self.scalability_bool = scalability_bool
            self.mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
            self.norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm']
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
        priv_idx = df.index[df[sensitive_var] == 1]
        unpriv_idx = df.index[df[sensitive_var] == 0]
        sensitive_attr = df[sensitive_var]
        return priv_idx, unpriv_idx, sensitive_attr


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
            _, _, sensitive_attr_train = self.getIdxSensitive(X_copy, self.sensitive_var)

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
        y_test = self.y_test.copy()

        if self.pipeline_type== 'ml':
            X_test = self.inject_missing_values(X_test, self.tau)
            _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
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

        if os.path.exists(file_name):
            self.param_lst_df = pd.read_csv(file_name)[self.pipeline_order + [f'utility_{self.metric_type}']]
            return
        
        if self.pipeline_type == 'ml':
            X_copy = self.inject_missing_values(X_copy, self.tau)
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

                method_name = f'get_{step}'
                if hasattr(handler, method_name) and callable(getattr(handler, method_name)):
                    method = getattr(handler, method_name)
                    fraction_outlier = method()

                if isinstance(result, float) or isinstance(result, int):
                    utility = result
                elif isinstance(result, tuple):
                    X, y, sens = result
                else:
                    raise ValueError(f"Expected tuple output from {class_name}.apply")
                param_record.append(param_index + 1)

            self.headers, sens_data = handler.get_profile_metric()
            prof_data = frac_data + sens_data
            profile_gen,key_profile = p.populate_profiles(pd.concat([X, y], axis=1), numerical_columns, self.target_variable_name, fraction_outlier, self.metric_type)
            param_lst.append(param_record + prof_data + profile_gen + [utility])
            #print(param_lst)

        self.profile_param_lst_df = pd.DataFrame(param_lst, columns= self.pipeline_order + self.frac_header + self.headers +key_profile+[f'utility_{self.metric_type}'])
        self.profile_param_lst_df.to_csv(file_name, index=False)

    #this 
    def rank_profile_parameter(self):
        param_lst_df=self.profile_param_lst_df.copy()
        key_profile = self.headers
        for column in self.X_train:
            key_profile.append('corr_'+column)
        self.profile=key_profile+self.frac_header
        #if self.h_sample_bool and self: #need to update according to logic #From SHAP
        if self.model_selection == 'reg':
            y = param_lst_df[f'utility_{self.metric_type}']                  
            t = StandardScaler().fit(param_lst_df).transform(param_lst_df)
            X = pd.DataFrame(data = t, columns = param_lst_df.columns)[self.profiles]
        else :
            y = param_lst_df[f'utility_{self.metric_type}']
            X = param_lst_df.copy()[self.profiles]       
                
        if (self.h_sample_bool):
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

        #ranking parameter 
        self.ranking_param ={}
        self.param_coeff  = {}
        for index, elem in enumerate(self.profiles):
                y = param_lst_df[elem]
                X = param_lst_df.copy()[self.param_columns]
                if self.h_sample_bool:
                            X = X.iloc[sample_idx]
                            y = y.iloc[sample_idx]
                reg = Regression()
                model = reg.generate_regression(X, y)
                coefs = model.coef_
                # print(model.intercept_)
                self.ranking_param[elem] =  np.argsort(np.abs(coefs))[::-1]
                print(self.ranking_param[elem])
                        
                self.param_coeff[elem] =  coefs
                print(f'name : {elem} {self.param_coeff[elem]}')

        for idx,profile_index in enumerate(self.profile_ranking):
                print(self.profiles[profile_index])

        return self.profile_coefs, self.profile_ranking, self.param_coeff, self.ranking_param
    
    def run_pipeline(self, alg_type, file_name):
        if alg_type == 'opaque':
            self.run_pipeline_opaque(file_name)
        elif alg_type == 'glass':
            self.run_pipeline_glass(file_name)
    
    def get_profile(self):
        return self.profile




'''dataset_name = 'adult'
metric_type = 'sp'
model_type = 'lr'
pipeline_order = ['missing_value', 'normalization', 'outlier', 'model']


filename_train = f'historical_data/historical_data_train_{model_type}_{metric_type}_{dataset_name}.csv'
executor = PipelineExecutor(
                pipeline_type='ml',
                dataset_name=dataset_name,
                metric_type=metric_type,
                pipeline_ord=pipeline_order,
                execution_type='pass',
            )
executor.run_pipeline_opaque(filename_train)

cur_par=[1, 1, 1, 1]

utility= executor.current_par_lookup(cur_par=cur_par)
print('utility:', utility)'''
