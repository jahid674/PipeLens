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

class PipelineExecutor:
    def __init__(self, dataset_name, metric_type,
                 mv_strategy, norm_strategy, od_strategy, model_selection,
                 knn_k_lst, lof_k_lst, tau_train,
                 contamination_train, contamination_train_lof, h_sample_bool=False):
        self.dataset_name = dataset_name
        self.metric_type = metric_type
        self.mv_strategy = mv_strategy
        self.norm_strategy = norm_strategy
        self.od_strategy = od_strategy
        self.model_selection = model_selection
        self.knn_k_lst = knn_k_lst
        self.lof_k_lst = lof_k_lst
        self.tau_train = tau_train
        self.contamination_train = contamination_train
        self.contamination_train_lof = contamination_train_lof
        self.h_sample_bool = h_sample_bool
        if self.h_sample_bool:
            self.h_sample = 0.005

    def get_sensitive_variable(self):
        if self.dataset_name == 'adult':
            return 'Sex'
        elif self.dataset_name == 'hmda':
            return 'race'
        return None
    
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
    
    def run_pipeline(self, file_name, X_train, y_train,
                     pipeline_order=['missing_value', 'normalization', 'outlier', 'model']):
        param_lst_df = None
        if os.path.exists(file_name):
            param_lst_df = pd.read_csv(file_name)[list(pipeline_order) + ["fairness"]]
        else:
            X_train = self.inject_missing_values(X_train, self.tau_train)

            mv_params = len(self.mv_strategy) + len(self.knn_k_lst) - 1 if 'knn' in self.mv_strategy else len(self.mv_strategy)
            od_params = len(self.od_strategy) + len(self.lof_k_lst) - 1 if 'lof' in self.od_strategy else len(self.od_strategy)
            norm_params = len(self.norm_strategy)
            model_params = len(self.model_selection)

            sensitive_var = self.get_sensitive_variable()
            priv_idx_train, unpriv_idx_train, sensitive_attr_train = self.getIdxSensitive(X_train, sensitive_var)

            param_lst = []
            print("Running pipeline combinations...")

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

                                elif step == 'normalization':
                                    strategy = self.norm_strategy[param2]
                                    normalizer = Normalizer(X_processed, strategy=strategy, verbose=False)
                                    X_processed = normalizer.transform()
                                    norm_param = [param2 + 1]

                                elif step == 'outlier':
                                    if param3 < len(self.od_strategy) - 1:
                                        od_choice = self.od_strategy[param3]
                                        if od_choice == 'none':
                                            outlier_detector = OutlierDetector(X_processed, strategy=od_choice)
                                        elif od_choice == 'if':
                                            outlier_detector = OutlierDetector(X_processed, strategy=od_choice,
                                                                               contamination=self.contamination_train, verbose=False)
                                    else:
                                        k = self.lof_k_lst[param3 - (len(self.od_strategy) - 1)]
                                        outlier_detector = OutlierDetector(X_processed, strategy='lof', k=k,
                                                                           contamination=self.contamination_train_lof, verbose=False)

                                    X_processed, y_processed, sensitive_processed, _, _ = outlier_detector.transform(
                                        y_processed, sensitive_processed)
                                    od_param = [param3 + 1]

                            trainer = ModelTrainer(self.model_selection[param4])
                            model = trainer.train(X_processed, y_processed)
                            model_param = [param4 + 1]
                            y_pred = model.predict(X_processed)

                            priv_idx = [i for i, val in enumerate(sensitive_processed) if val == 1]
                            unpriv_idx = [i for i, val in enumerate(sensitive_processed) if val == 0]

                            metric_evaluator = MetricEvaluator(self.metric_type)
                            outc = metric_evaluator.compute(
                                y_true=y_processed,
                                y_pred=y_pred,
                                priv_idx=priv_idx,
                                unpriv_idx=unpriv_idx
                            )

                            param_lst.append(mv_param + norm_param + od_param + model_param + [outc])
                            print(mv_param + norm_param + od_param + model_param + [outc])

            param_lst_df = pd.DataFrame(param_lst, columns=pipeline_order + ["fairness"])
            param_lst_df.to_csv(file_name, index=False)
 
        y = param_lst_df['fairness']
        X = param_lst_df.drop('fairness', axis=1)

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
        print(coefs)
        print(model.intercept_)

        coef_rank = np.argsort(np.abs(coefs)).tolist()[::-1]
        logging.info(f'coef {coefs}')
        print(coef_rank)
        return param_lst_df, coefs, coef_rank
    
    def current_par_lookup(self, X_train, y_train,
                     pipeline_order=['missing_value', 'normalization', 'outlier', 'model'], cur_par=[]):
        X_train = self.inject_missing_values(X_train, self.tau_train)

        sensitive_var = self.get_sensitive_variable()
        _, _, sensitive_attr_train = self.getIdxSensitive(X_train, sensitive_var)

        param_lst = []
        print("Running pipeline combinations...")

        for param1 in [int(cur_par[0])-1]:
            for param2 in [int(cur_par[1])-1]:
                for param3 in [int(cur_par[2])-1]:
                     for param4 in [int(cur_par[3])-1]:
                        X_processed = X_train.copy()
                        y_processed = y_train.copy()
                        sensitive_processed = sensitive_attr_train.copy()

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
                                else:
                                    k = self.knn_k_lst[param1 - (len(self.mv_strategy) - 1)]
                                    imputer = Imputer(X_processed, strategy='knn', k=k, verbose=False)
                                    X_processed = imputer.transform(y_processed, sensitive_processed)
                                    y_processed = y_processed.copy()
                                    sensitive_processed = sensitive_processed.copy()

                            elif step == 'normalization':
                                strategy = self.norm_strategy[param2]
                                normalizer = Normalizer(X_processed, strategy=strategy, verbose=False)
                                X_processed = normalizer.transform()

                            elif step == 'outlier':
                                if param3 < len(self.od_strategy) - 1:
                                    od_choice = self.od_strategy[param3]
                                    if od_choice == 'none':
                                        outlier_detector = OutlierDetector(X_processed, strategy=od_choice)
                                    elif od_choice == 'if':
                                        outlier_detector = OutlierDetector(X_processed, strategy=od_choice,
                                                                               contamination=self.contamination_train, verbose=False)
                                else:
                                    k = self.lof_k_lst[param3 - (len(self.od_strategy) - 1)]
                                    outlier_detector = OutlierDetector(X_processed, strategy='lof', k=k,
                                                                           contamination=self.contamination_train_lof, verbose=False)

                                X_processed, y_processed, sensitive_processed, _, _ = outlier_detector.transform(
                                        y_processed, sensitive_processed)

                        trainer = ModelTrainer(self.model_selection[param4])
                        model = trainer.train(X_processed, y_processed)
                        y_pred = model.predict(X_processed)

                        priv_idx = [i for i, val in enumerate(sensitive_processed) if val == 1]
                        unpriv_idx = [i for i, val in enumerate(sensitive_processed) if val == 0]

                        metric_evaluator = MetricEvaluator(self.metric_type)
                        outc = metric_evaluator.compute(
                            y_true=y_processed,
                            y_pred=y_pred,
                            priv_idx=priv_idx,
                            unpriv_idx=unpriv_idx
                            )      
        return outc
    def run_pipeline_algo2(self, file_name, X_train, y_train,
                     pipeline_order=['missing_value', 'normalization', 'outlier', 'model']):
        param_lst_df = None
        key_profile = []
        p = Profile()

        if os.path.exists(file_name):
            param_lst_df = param_lst_df = pd.read_csv(file_name)
        else:
            X_train = self.inject_missing_values(X_train, self.tau_train)

            mv_params = len(self.mv_strategy) + len(self.knn_k_lst) - 1 if 'knn' in self.mv_strategy else len(self.mv_strategy)
            od_params = len(self.od_strategy) + len(self.lof_k_lst) - 1 if 'lof' in self.od_strategy else len(self.od_strategy)
            norm_params = len(self.norm_strategy)
            model_params = len(self.model_selection)

            sensitive_var = self.get_sensitive_variable()
            priv_idx_train, unpriv_idx_train, sensitive_attr_train = self.getIdxSensitive(X_train, sensitive_var)

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
                                                                               contamination=self.contamination_train, verbose=False)
                                    else:
                                        k = self.lof_k_lst[param3 - (len(self.od_strategy) - 1)]
                                        outlier_detector = OutlierDetector(X_processed, strategy='lof', k=k,
                                                                           contamination=self.contamination_train_lof, verbose=False)

                                    X_processed, y_processed, sensitive_processed, _, _ = outlier_detector.transform(
                                        y_processed, sensitive_processed)
                                    od_param = [param3 + 1]


                            fraction_out = round((p.get_fraction_of_outlier(X_processed))*100, 4)

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
                                if(metric_type=='accuracy_score'):
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
                            profile_gen,key_profile = p.populate_profiles(pd.concat([X_processed, y_processed], axis=1), self.numerical_columns, target_variable_name, fraction_out, self.metric_type)
                                                        
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



'''dataset_name = 'adult'
metric_type = 'sp'
modelType= 'lr'
filename_train = f'historical_data/historical_data_train_profile_{modelType}_{metric_type}_{dataset_name}.csv'


mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm']
od_strategy = ['none', 'if', 'lof']
model_selection = ['lr']
knn_k_lst = [1, 5, 10, 20, 30]
lof_k_lst = [1, 5, 10, 20, 30]


loader = LoadDataset(dataset_name)
dataset, X_train, y_train, X_test, y_test = loader.load()


tau_train = 0.1
contamination_train = 0.2
contamination_train_lof = 'auto'


executor = PipelineExecutor(
    dataset_name=dataset_name,
    metric_type=metric_type,
    mv_strategy=mv_strategy,
    norm_strategy=norm_strategy,
    od_strategy=od_strategy,
    model_selection=model_selection,
    knn_k_lst=knn_k_lst,
    lof_k_lst=lof_k_lst,
    tau_train=tau_train,
    contamination_train=contamination_train,
    contamination_train_lof=contamination_train_lof
)


sensitive_var = executor.get_sensitive_variable()
_, _, sensitive_attr_train = executor.getIdxSensitive(X_train, sensitive_var)
pipeline_df, coefs, coef_rank = executor.run_pipeline(filename_train, X_train, y_train)
print(pipeline_df)
print(coefs)
print(coef_rank)
print("Pipeline execution completed.")
cur_par=[0, 0, 0, 0]

utility= executor.current_par_lookup(X_train, y_train,
                     pipeline_order=['missing_value', 'normalization', 'outlier', 'model'], cur_par=cur_par)
print('utility:', utility)'''


'''dataset_name = 'adult'
metric_type = 'sp'
modelType= 'lr'
filename_train = f'historical_data/historical_data_train_profile_{modelType}_{metric_type}_{dataset_name}.csv'


mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm']
od_strategy = ['none', 'if', 'lof']
model_selection = ['lr']
knn_k_lst = [1, 5, 10, 20, 30]
lof_k_lst = [1, 5, 10, 20, 30]


#loader = LoadDataset(dataset_name)
#dataset, X_train, y_train, X_test, y_test = loader.load()


tau_train = 0.1
contamination_train = 0.2
contamination_train_lof = 'auto'


executor = PipelineExecutor(
    dataset_name=dataset_name,
    metric_type=metric_type,
    mv_strategy=mv_strategy,
    norm_strategy=norm_strategy,
    od_strategy=od_strategy,
    model_selection=model_selection,
    knn_k_lst=knn_k_lst,
    lof_k_lst=lof_k_lst,
    tau_train=tau_train,
    contamination_train=contamination_train,
    contamination_train_lof=contamination_train_lof
)
loader = LoadDataset(dataset_name)
datasets, X_train, y_train, X_test, y_test = loader.load()
executor.set_dataset(X_train)
sensitive_var = executor.get_sensitive_variable()
_, _, sensitive_attr_train = executor.getIdxSensitive(X_train, sensitive_var)
pro_coef, pro_coef_rank, par_coef, par_coef_rank = executor.run_pipeline_algo2(filename_train, X_train, y_train)
print(pro_coef)
print(pro_coef_rank)
print(par_coef)
print(par_coef_rank)
print("Pipeline execution completed.")'''
'''cur_par=[0, 0, 0, 0]

utility= executor.current_par_lookup(X_train, y_train,
                     pipeline_order=['missing_value', 'normalization', 'outlier', 'model'], cur_par=cur_par)
print('utility:', utility)'''