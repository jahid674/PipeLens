class NewComponent:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self. new_component_strategy = config['new_component_strategy']

    def apply(self, X, y, sensitive):
        # If preprocessing:
        #return X_new, y_new, sensitive_new

        # If final model evaluation:
        return X
