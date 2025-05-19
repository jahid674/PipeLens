from modules.models.model import ModelTrainer
from modules.models.metric import MetricEvaluator

class ModelHandler:
    def __init__(self, strategy, config):
        self.model_selection = config['model_selection']
        self.metric_type = config['metric_type']
        self.model_index = strategy

    def apply(self, X, y, sensitive):
        trainer = ModelTrainer(self.model_selection[self.model_index])
        model = trainer.train(X, y)
        y_pred = model.predict(X)

        priv_idx = [i for i, val in enumerate(sensitive) if val == 1]
        unpriv_idx = [i for i, val in enumerate(sensitive) if val == 0]

        evaluator = MetricEvaluator(self.metric_type)
        return evaluator.compute(y_true=y, y_pred=y_pred, priv_idx=priv_idx, unpriv_idx=unpriv_idx)
