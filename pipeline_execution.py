import os
import pandas as pd
import numpy as np
import logging
import random
import sys
from modules.models.model import ModelTrainer
from modules.models.metric import MetricEvaluator
from modules.normalization.normalizer import Normalizer
from modules.missing_value.imputer import Imputer
from modules.outlier_detection.outlier_detector import OutlierDetector
from sklearn.preprocessing import StandardScaler
from regression import Regression
from LoadDataset import LoadDataset
from modules.profiling.profile import Profile
import math
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
            self.model_selection = ['lr'] # 
            self.knn_k_lst = [1, 5, 10, 20, 30]
            self.lof_k_lst = [1, 5, 10, 20, 30]
            self.pipeline_order = pipeline_ord # always keep model at the end
            self.execution_type = execution_type # 'pass' or 'fail'

            loader = LoadDataset(self.dataset_name)
            self.dataset, self.X_train, self.y_train, self.X_test, self.y_test = loader.load()
            self.sensitive_var = loader.get_sensitive_variable()

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
                'metric_type': self.metric_type
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

        if self.dataset_name == 'hmda':
            X_modified.loc[mv_train, 'lien_status'] = np.nan
        elif self.dataset_name == 'adult':
            X_modified.loc[mv_train, 'Martial_Status'] = np.nan
        elif self.dataset_name == 'housing':
            X_modified.loc[mv_train, 'OverallQual'] = np.nan
        return X_modified


    def run_pipeline(self, file_name):
        if self.execution_type == 'pass':
            X_copy, y_copy = self.X_train.copy(), self.y_train.copy()
        elif self.execution_type == 'fail':
            X_copy, y_copy = self.X_test.copy(), self.y_test.copy()

        if os.path.exists(file_name):
            self.param_lst_df = pd.read_csv(file_name)[self.pipeline_order + [f'utility_{self.metric_type}']]
            return

        X_copy = self.inject_missing_values(X_copy, self.tau)
        _, _, sensitive_attr_train = self.getIdxSensitive(X_copy, self.sensitive_var)

        param_ranges = [range(self.strategy_counts[step]) for step in self.pipeline_order]
        param_combinations = itertools.product(*param_ranges)

        param_lst = []
        print("Running pipeline combinations...")

        for combo in param_combinations:
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


    def run_pipeline_algo2(self, file_name, X_train, y_train,
                     pipeline_order=['missing_value', 'normalization', 'outlier', 'model']):
        param_lst_df = None
        key_profile = []
        p = Profile()

        if os.path.exists(file_name):
            param_lst_df = pd.read_csv(file_name)
        else:
            X_train = self.inject_missing_values(X_train, self.tau)

            mv_params = len(self.mv_strategy) + len(self.knn_k_lst) - 1 if 'knn' in self.mv_strategy else len(self.mv_strategy)
            od_params = len(self.od_strategy) + len(self.lof_k_lst) - 1 if 'lof' in self.od_strategy else len(self.od_strategy)
            norm_params = len(self.norm_strategy)
            model_params = len(self.model_selection)
            numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns
            #sensitive_var = self.get_sensitive_variable()
            priv_idx_train, unpriv_idx_train, sensitive_attr_train = self.getIdxSensitive(X_train, self.sensitive_var)

            param_lst = []
            sens_attr_name = ''
            target_variable_name = ''
            print("Running pipeline combinations...")
            
            sens_attr_name = sensitive_attr_train.name
            target_variable_name = y_train.name
            
            for param1 in range(mv_params):
                for param2 in range(norm_params):
                    for param3 in range(od_params):
                        for param4 in range(model_params):
                            X_processed = X_train.copy()
                            y_processed = y_train.copy()
                            sensitive_processed = sensitive_attr_train.copy()

                            mv_param, norm_param, od_param, model_param = [], [], [], []

                            for step in pipeline_order:
                                if step == 'missing_value':
                                    if param1 < len(self.mv_strategy) - 1:
                                        strategy = self.mv_strategy[param1]
                                        imputer = Imputer(X_processed, strategy=strategy, verbose=False)
                                        if strategy == 'drop':
                                            X_processed, y_processed, sensitive_processed = imputer.transform(y_processed, sensitive_processed)
                                        else:
                                            X_processed = imputer.transform(y_processed, sensitive_processed)
                                        y_processed = y_processed.copy()
                                        sensitive_processed = sensitive_processed.copy()
                                        mv_param = [param1 + 1]
                                    else:
                                        k = self.knn_k_lst[param1 - (len(self.mv_strategy) - 1)]
                                        imputer = Imputer(X_processed, strategy='knn', k=k, verbose=False)
                                        X_processed = imputer.transform(y_processed, sensitive_processed)
                                        y_processed = y_processed.copy()
                                        sensitive_processed = sensitive_processed.copy()
                                        mv_param = [param1 + 1]
                                    out_before_norm_strat = p.get_fraction_of_outlier(X_processed)

                                elif step == 'normalization':
                                    strategy = self.norm_strategy[param2]
                                    normalizer = Normalizer(X_processed, strategy=strategy, verbose=False)
                                    X_processed = normalizer.transform()
                                    norm_param = [param2 + 1]
                                    out_before_out_strat = p.get_fraction_of_outlier(X_processed)
                                    

                                elif step == 'outlier':
                                    if param3 < len(self.od_strategy) - 1:
                                        od_choice = self.od_strategy[param3]
                                        if od_choice == 'none':
                                            outlier_detector = OutlierDetector(X_processed, strategy=od_choice)
                                        elif od_choice == 'if':
                                            outlier_detector = OutlierDetector(X_processed, strategy=od_choice,
                                                                               contamination=self.contamination, verbose=False)
                                    else:
                                        k = self.lof_k_lst[param3 - (len(self.od_strategy) - 1)]
                                        outlier_detector = OutlierDetector(X_processed, strategy='lof', k=k,
                                                                           contamination=self.contamination_lof, verbose=False)

                                    X_processed, y_processed, sensitive_processed, fraction_out = outlier_detector.transform(
                                        y_processed, sensitive_processed)
                                    od_param = [param3 + 1]


                            #fraction_out = round((p.get_fraction_of_outlier(X_processed))*100, 4)

                            #model training
                            trainer = ModelTrainer(self.model_selection[param4])
                            model = trainer.train(X_processed, y_processed)
                            model_param = [param4 + 1]
                            y_pred = model.predict(X_processed)

                            priv_idx = [i for i, val in enumerate(sensitive_processed) if val == 1]
                            unpriv_idx = [i for i, val in enumerate(sensitive_processed) if val == 0]
                            
                            if(self.metric_type=='sp' or self.metric_type=='accuracy_score'):
                                concat_X_y  =  pd.concat([sensitive_processed, y_processed], axis=1)
                                                        
                                y_pred_priv = len(concat_X_y[(concat_X_y[sens_attr_name] == 1) & (concat_X_y[target_variable_name] == 1)])/ len(concat_X_y[(concat_X_y[sens_attr_name] == 1)])
                                y_pred_unpriv = len(concat_X_y[(concat_X_y[sens_attr_name] == 0) & (concat_X_y[target_variable_name] == 1)])/ len(concat_X_y[(concat_X_y[sens_attr_name] == 0)])
                                diff_sensitive_attr = round(y_pred_priv - y_pred_unpriv,5)
                                ratio_sensitive_attr =  round(len(concat_X_y[(concat_X_y[sens_attr_name] == 1)] )/len(concat_X_y[(concat_X_y[sens_attr_name] == 0)]),5)
                                cov = concat_X_y[sens_attr_name].cov(concat_X_y[y_processed.name])
                                class_imbalance_ratio = round((y_processed == 1).sum()/len(y_train),5)
                                sens_data =[out_before_out_strat,out_before_norm_strat,diff_sensitive_attr,ratio_sensitive_attr,cov,class_imbalance_ratio]
                                if(self.metric_type=='accuracy_score'):
                                        sens_data =[out_before_out_strat,out_before_norm_strat,class_imbalance_ratio]
                            else:
                                    profile_median = y_processed.median()
                                    sens_data =[out_before_out_strat,out_before_norm_strat,profile_median]


                            metric_evaluator = MetricEvaluator(self.metric_type)
                            outc = metric_evaluator.compute(
                                y_true=y_processed,
                                y_pred=y_pred,
                                priv_idx=priv_idx,
                                unpriv_idx=unpriv_idx
                            )

                            f = [outc]
                            profile_gen,key_profile = p.populate_profiles(pd.concat([X_processed, y_processed], axis=1), numerical_columns, target_variable_name, fraction_out, self.metric_type)
                                                        
                            param_lst.append(mv_param + norm_param + od_param + model_param + sens_data + profile_gen + f)
                            print(str(mv_param + norm_param + od_param + model_param + f))

            param_column = ["missing_value","normalization","outlier", "model"]
            if(self.metric_type=='sp'):
                    param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['out_before_out_strat','out_before_norm_strat',"diff_sensitive_attr","ratio_sensitive_attr","cov","class_imbalance_ratio"]+key_profile+["fairness"])
            elif(self.metric_type=='accuracy_score'):
                    param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['out_before_out_strat','out_before_norm_strat','class_imbalance_ratio']+key_profile+["fairness"])
            else:
                    param_lst_df = pd.DataFrame(param_lst, columns= param_column   + ['out_before_out_strat','out_before_norm_strat','profile_median'] + key_profile + ["fairness"])
                               
            param_lst_df.to_csv(file_name, index=False)


        # Need to understand the logic of this part
        self.param_columns = ["missing_value","normalization","outlier", "model"]
        if(self.dataset_name=='hmda'):
                key_profile = ['class_imbalance_ratio']
        elif self.dataset_name=='housing':
                key_profile = ['profile_median']
        elif self.dataset_name=='adult':
                key_profile = ["diff_sensitive_attr","ratio_sensitive_attr","class_imbalance_ratio"]
        for column in X_train.columns:
                key_profile.append('corr_' + column)
        self.profiles = key_profile + ['out_before_out_strat','out_before_norm_strat']
        # self.profiles = key_profile
        if self.h_sample_bool and self.dataset == 'adult':
                self.profiles = ['diff_sensitive_attr', 'class_imbalance_ratio','ratio_sensitive_attr','corr_Country','corr_Workclass']
                
        #rank profile first
        if(self.dataset_name=='housing'):
            y = param_lst_df['fairness']
                        
            t = StandardScaler().fit(param_lst_df).transform(param_lst_df)
            # t = param_lst_df
            X = pd.DataFrame(data = t, columns = param_lst_df.columns)[self.profiles]
        else :
            y = param_lst_df['fairness']
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
                        
                # X = StandardScaler().fit(X).transform(X)
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
        print('33')
        
        return self.profile_coefs, self.profile_ranking, self.param_coeff, self.ranking_param



dataset_name = 'adult'
metric_type = 'sp'
model_type = 'lr'
pipeline_order = ['missing_value', 'normalization', 'outlier', 'model']


filename_train = f'historical_data/historical_data_test_{model_type}_{metric_type}_{dataset_name}.csv'
executor = PipelineExecutor(
                pipeline_type='ml',
                dataset_name=dataset_name,
                metric_type=metric_type,
                pipeline_ord=pipeline_order,
                execution_type='fail',
            )
executor.run_pipeline(filename_train)

historical_data = pd.read_csv(filename_train)
_, _ = executor.score_parameter(historical_data)


cur_par=[1, 1, 1, 1]

utility= executor.current_par_lookup(cur_par=cur_par)
print('utility:', utility)
