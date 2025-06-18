from modules.features_selection.fselector import FeatureSelector


class FeatureHandler:
    def __init__(self, strategy, config):
        """
        - strategy: index into the fs_strategy list
        - config: dictionary with the following keys:
            - 'fs_strategy': list of strategies: ['none', 'variance', 'mutual_info']
            - 'variance_threshold': (float) threshold for VarianceThreshold
            - 'mutual_info_top_k': (int or None) number of features or None to select 80%
        """
        self.strategy = strategy
        self.fs_strategy = config['fs_strategy']
        self.threshold = config.get('variance_threshold', 0.01)
        self.top_k = config.get('mutual_info_top_k', None)
        self.selected_features = None

    def apply(self, X, y):
        fs_choice = self.fs_strategy[self.strategy]

        if fs_choice == 'none':
            selector = FeatureSelector(X, strategy='none')
        elif fs_choice == 'variance':
            selector = FeatureSelector(X, strategy='variance', threshold=self.threshold, verbose=False)
        elif fs_choice == 'mutual_info':
            selector = FeatureSelector(X, strategy='mutual_info', top_k=self.top_k, verbose=False)
        else:
            raise ValueError("Unsupported feature selection strategy.")

        X_selected = selector.transform(y_train=y)
        self.selected_features = selector.selected_features
        return X_selected

    def get_selected_features(self):
        if self.selected_features is None:
            raise ValueError("No features selected yet. Please run apply() first.")
        return self.selected_features

