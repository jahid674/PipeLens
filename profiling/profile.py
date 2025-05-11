import numpy as np
import operator
import random
import pandas as pd
import os
import statistics
from scipy.stats import pearsonr, chi2_contingency
from scipy import stats
from sklearn.svm import OneClassSVM

#from modules.matching.perfectmatching import PerfectMatching
#from modules.matching.jaccardmatching import JaccardMatching

random.seed(0)


class Profile:
    profile_lst = []

    def __init__(self, df):
        self.df = df

    def outlier(self, lst):
        mean = statistics.mean(lst)
        std = statistics.stdev(lst)
        count = sum(1 for v in lst if v > mean + 2 * std or v < mean - 2 * std)
        return count / len(lst)

    def missing(self, lst):
        count = 0
        for v in lst:
            try:
                if np.isnan(v):
                    count += 1
            except:
                if hasattr(v, '__len__') and len(v) == 0:
                    count += 1
        return count / len(lst)

    def correlation(self, lst1, lst2):
        try:
            r, _ = pearsonr(lst1, lst2)
            return r
        except Exception as e:
            print("Correlation error:", e)
            return 0

    def categorical_correlation(self, lst1, lst2):
        cross_tab = pd.crosstab(lst1, lst2)
        chi2, _, _, _ = chi2_contingency(cross_tab)
        return chi2

    def categorical_numerical_correlation(self, lst1, lst2):
        (chi2, _) = stats.f_oneway(lst1, lst2)
        return chi2
    
    def get_fraction_of_outlier(self,data):
        svm_model = OneClassSVM(kernel='rbf')
        svm_model.fit(data)
        predicted_labels = svm_model.predict(data)
        n_outliers = (predicted_labels == -1).sum()
        fraction_outliers = n_outliers / len(data)
        return fraction_outliers

    def populate_profiles(self, data_final, numerical_columns, target_column, outlier, metric_type):
        scaling_factor = 1
        profile = {}

        for column in data_final.columns:
            if column == target_column:
                continue

            if metric_type in ('rmse', 'mae'):
                if column in numerical_columns:
                    corr = self.correlation(data_final[column], data_final[target_column])
                else:
                    corr = self.categorical_numerical_correlation(data_final[column], data_final[target_column])
            else:
                if column in numerical_columns:
                    corr = self.categorical_numerical_correlation(data_final[column], data_final[target_column])
                else:
                    corr = self.categorical_correlation(data_final[column], data_final[target_column])

            profile[(f'corr_{column}', f'ot_{column}')] = [
                column,
                round(corr * scaling_factor, 5),
                round(outlier * scaling_factor, 5)
            ]

        dd = []
        keys = []
        for val in profile:
            # import pdb;pdb.set_trace()
            dd.append(profile[val][1])
            dd.append(profile[val][2])
            keys.append(val[0])
            keys.append(val[1])

        return dd, keys

'''
data_final = pd.DataFrame({
    'age': [25, 30, 22, 35, 40, 28],
    'income': [50000, 60000, 55000, 65000, 70000, 62000],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F'],
    'purchased': [1, 0, 0, 1, 1, 0]
})

numerical_columns = ['age', 'income']

target_column = 'purchased'

outlier_fraction = 0.1
profiler = Profile(data_final)

results, keys = profiler.populate_profiles(
    data_final=data_final,
    numerical_columns=numerical_columns,
    target_column=target_column,
    outlier=outlier_fraction,
    metric_type='classification'
)

print("Keys:", keys)
print("Results:", results)
'''