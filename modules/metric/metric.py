
from sklearn.metrics import  confusion_matrix
import pandas as pd
class metric():


    def __init__(self,estimator=None):
        print("Metric ....")
        self.estimator = None

    def equalized_odds(estimator, X_test, y_true):
        y_pred = estimator.predict(X_test)
        sensitive_features = X_test['gender']
        assert len(y_true) == len(y_pred) == len(sensitive_features), "Input array lengths must be equal"

        sensitive_groups = pd.unique(sensitive_features)

        disparities = []
        try:
            for group in sensitive_groups:
                group_mask = sensitive_features == group
                cm = confusion_matrix(y_true[group_mask], y_pred[group_mask])
                tn, fp, fn, tp = cm.ravel()
                equalized_odds_disparity = (tp / (tp + fn)) - (tn / (tn + fp))
                disparities.append(equalized_odds_disparity)
            score = max(disparities) - min(disparities)
        except Exception as exception:
            print("An exception occurred:", exception)
            return 0
        
        return score
    def calculate_accuracy(estimator,X_test, predicted_labels): 
        true_labels = estimator.predict(X_test)
        if len(true_labels) != len(predicted_labels):
            raise ValueError("Input lists must have the same length.")

        correct_predictions = 0
        for true, predicted in zip(true_labels, predicted_labels):
            if true == predicted:
                correct_predictions += 1
        accuracy = correct_predictions / len(true_labels)
        return accuracy