from modules.models.model import ModelTrainer
from modules.models.metric import MetricEvaluator
import pandas as pd
from collections import Counter

class ModelHandler:
    def __init__(self, strategy, config):
        self.model_selection = config['model_selection']
        self.metric_type = config['metric_type']
        self.model_index = strategy
        self.target_variable_name = config['target_var']
        self.sens_attr_name = config['sensitive_var']

    def apply(self, X, y, sensitive):
        trainer = ModelTrainer(self.model_selection[self.model_index])
        if hasattr(X, "dropna") and hasattr(y, "__len__"):
            kept_idx = X.dropna().index
            X = X.loc[kept_idx].reset_index(drop=True)
            y = y.loc[kept_idx].reset_index(drop=True) if hasattr(y, "loc") else y[kept_idx]
            sensitive=sensitive.loc[kept_idx].reset_index(drop=True)
        model = trainer.train(X, y)
        y_pred = model.predict(X)
        if self.metric_type == 'sp':
            priv_idx = [i for i, val in enumerate(sensitive) if val == 1]
            unpriv_idx = [i for i, val in enumerate(sensitive) if val == 0]
        else:
            priv_idx = None
            unpriv_idx = None

        evaluator = MetricEvaluator(self.metric_type)

        self._X = X
        self._y = y
        self._sensitive = sensitive
        self._y_pred = y_pred

        return evaluator.compute(y_true=y, y_pred=y_pred, priv_idx=priv_idx, unpriv_idx=unpriv_idx)
    
    def get_profile_metric(self, y_train, sensitive=None):
        y = y_train.reset_index(drop=True)
        if self.metric_type == 'sp' or 'accuracy_score':
            sensitive = sensitive.reset_index(drop=True)
        else:
            sensitive = None

        if self.metric_type in ['sp', 'accuracy_score']:
            concat_X_y = pd.concat([sensitive, y], axis=1)
            concat_X_y.columns = [self.sens_attr_name, self.target_variable_name]
            class_imbalance_ratio= None
            target_counts = Counter(y_train)
            majority_class = max(target_counts.values())
            minority_class = min(target_counts.values())
            class_imbalance_ratio= majority_class / minority_class

        if self.metric_type in ['sp']:
            
            y_pred_priv = len(concat_X_y[(concat_X_y[self.sens_attr_name] == 1) &
                                        (concat_X_y[self.target_variable_name] == 1)]) / \
                        len(concat_X_y[concat_X_y[self.sens_attr_name] == 1])

            y_pred_unpriv = len(concat_X_y[(concat_X_y[self.sens_attr_name] == 0) &
                                        (concat_X_y[self.target_variable_name] == 1)]) / \
                            len(concat_X_y[concat_X_y[self.sens_attr_name] == 0])

            diff_sensitive_attr = round(y_pred_priv - y_pred_unpriv, 5)
            ratio_sensitive_attr = round(len(concat_X_y[concat_X_y[self.sens_attr_name] == 1]) /
                                        len(concat_X_y[concat_X_y[self.sens_attr_name] == 0]), 5)
            cov = concat_X_y[self.sens_attr_name].cov(concat_X_y[self.target_variable_name])
            return ['diff_sensitive_attr', 'ratio_sensitive_attr', 'cov', 'class_imbalance_ratio'], \
                    [diff_sensitive_attr, ratio_sensitive_attr, cov, class_imbalance_ratio]
            #class_imbalance_ratio = round((y == 1).sum() / len(y_train),5) if y is not None else None
        elif self.metric_type == 'accuracy_score':
            return ['class_imbalance_ratio'], [class_imbalance_ratio]
        else:
            profile_median = y.median()
            return ['profile_median'], [profile_median]



