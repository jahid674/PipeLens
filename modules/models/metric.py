# modules/metric/evaluator.py

from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error, mean_absolute_error, mean_squared_error
import numpy as np

class MetricEvaluator:
    def __init__(self, metric_type):
        self.metric_type = metric_type

    def compute(self, y_true, y_pred, priv_idx=None, unpriv_idx=None):
        if self.metric_type == 'sp':
            return self.computeStatisticalParity(y_pred[priv_idx], y_pred[unpriv_idx])

        elif self.metric_type == 'f-1':
            return 1 - f1_score(y_true, y_pred)

        elif self.metric_type == 'accuracy_score':
            return 1 - accuracy_score(y_true, y_pred)

        elif self.metric_type == 'mae':
            return mean_absolute_error(y_true, y_pred)

        elif self.metric_type == 'rmse':
            return np.sqrt(root_mean_squared_error(y_true, y_pred))

        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def computeStatisticalParity(self, p_priv, p_unpriv):
        diff = p_priv.mean() - p_unpriv.mean()
        return diff
