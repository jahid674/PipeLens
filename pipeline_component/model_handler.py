from modules.models.model import ModelTrainer
from modules.models.metric import MetricEvaluator
import pandas as pd

class ModelHandler:
    def __init__(self, strategy, config):
        self.model_selection = config['model_selection']
        self.metric_type = config['metric_type']
        self.model_index = strategy
        self.target_variable_name = config['target_var']
        self.sens_attr_name = config['sensitive_var']

    def apply(self, X, y, sensitive):
        trainer = ModelTrainer(self.model_selection[self.model_index])
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
    
    def get_profile_metric(self,y_train):
        y = self._y.reset_index(drop=True)
        if self.metric_type == 'sp':
            sensitive = self._sensitive.reset_index(drop=True)
        else:
            sensitive = None
        y_pred = self._y_pred
        if self.metric_type == 'sp':
            concat_X_y = pd.concat([sensitive, y], axis=1)
            concat_X_y.columns = [self.sens_attr_name, self.target_variable_name]

        if self.metric_type in ['sp', 'accuracy_score']:
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
            class_imbalance_ratio = round((y == 1).sum() / len(y_train), 5) if y is not None else None

            if self.metric_type == 'accuracy_score':
                return ['class_imbalance_ratio'], [class_imbalance_ratio]
            else:
                return ['diff_sensitive_attr', 'ratio_sensitive_attr', 'cov', 'class_imbalance_ratio'], \
                    [diff_sensitive_attr, ratio_sensitive_attr, cov, class_imbalance_ratio]

        else:
            profile_median = y.median()
            return ['profile_median'], [profile_median]



