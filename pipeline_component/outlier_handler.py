from modules.outlier_detection.outlier_detector import OutlierDetector
from modules.profiling.profile import Profile

class OutlierHandler:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.od_strategy = config['od_strategy']
        self.lof_k_list = config['lof_k_lst']
        self.contamination = config['contamination']
        self.contamination_lof = config['contamination_lof']
        self.p = Profile()

    def apply(self, X, y, sensitive):
        self.outlier_before_out_start=self.p.get_fraction_of_outlier(X)
        if self.strategy < len(self.od_strategy) - 1:
            od_choice = self.od_strategy[self.strategy]
            if od_choice == 'none':
                detector = OutlierDetector(X, strategy=od_choice)
            elif od_choice == 'if':
                detector = OutlierDetector(X, strategy=od_choice, contamination=self.contamination, verbose=False)
        else:
            k = self.lof_k_list[self.strategy - (len(self.od_strategy) - 1)]
            detector = OutlierDetector(X, strategy='lof', k=k, contamination=self.contamination_lof, verbose=False)
        X_out, y_out, sens_out = detector.transform(y, sensitive)
        self.get_outlier_frac=detector.get_frac()
        #self.outlier_after_out_start=self.p.get_fraction_of_outlier(X_out)
        return X_out, y_out, sens_out
    
    def get_outlier_bef_outlier_strat(self):
        return self.outlier_before_out_start
    
    #def get_outlier_after_outlier_strat(self):
    #    return self.outlier_after_out_start
    
    def get_outlier(self):
        return self.get_outlier_frac
        
