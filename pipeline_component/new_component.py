class NewComponent:
    def __init__(self, executor, strategy):
        self.strategy = strategy
        # use executor to access shared settings like contamination, model list, etc.

    def apply(self, X, y, sensitive):
        # If preprocessing:
        #return X_new, y_new, sensitive_new

        # If final model evaluation:
        return X
