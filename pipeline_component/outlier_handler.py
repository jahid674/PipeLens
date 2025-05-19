from modules.outlier_detection.outlier_detector import OutlierDetector

class OutlierHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.od_strategy = config['od_strategy']
        self.lof_k_list = config['lof_k_lst']
        self.contamination = config['contamination']
        self.contamination_lof = config['contamination_lof']

    def apply(self, X, y, sensitive):
        if self.strategy < len(self.od_strategy) - 1:
            od_choice = self.od_strategy[self.strategy]
            if od_choice == 'none':
                detector = OutlierDetector(X, strategy=od_choice)
            elif od_choice == 'if':
                detector = OutlierDetector(X, strategy=od_choice, contamination=self.contamination, verbose=False)
        else:
            k = self.lof_k_list[self.strategy - (len(self.od_strategy) - 1)]
            detector = OutlierDetector(X, strategy='lof', k=k, contamination=self.contamination_lof, verbose=False)
        X_out, y_out, sens_out, _ = detector.transform(y, sensitive)
        return X_out, y_out, sens_out
