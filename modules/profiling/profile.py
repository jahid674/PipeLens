import numpy as np
import operator
import random
import pandas as pd
import os
import statistics
from scipy.stats import pearsonr, chi2_contingency  
from scipy import stats
from sklearn.svm import OneClassSVM

from modules.matching.perfectmatching import PerfectMatching
from modules.matching.jaccardmatching import JaccardMatching

random.seed(0)

# did merge both of the profile class 

class Profile:
    def __init__(self):

        self.profiles = {}
        self.df_list = {}
        self.attributes = {}
        self.stopword_lst = {}
        self.sim = {}
        self.gt_data = []

        self.profile_lst = []

    # EM Profiling
    def random_sample(self):
        gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
        id1_list = []
        id2_list = []
        for [id1, id2] in gt.values.tolist():
            if random.random() <= 0.1:
                self.gt_data.append((id1, id2))
                id1_list.append(id1)
                id2_list.append(id2)
        dblp = pd.read_csv('DBLP-ACM/noisy20.csv', encoding="latin-1")
        acm = pd.read_csv('DBLP-ACM/noisy21.csv')
        sample_dblp = dblp[dblp['id'].isin(id1_list)]
        sample_acm = acm[acm['id'].isin(id2_list)]
        if not os.path.exists('DBLP-ACM/noisyDBLP2_Sample.csv'):
            sample_dblp.to_csv('DBLP-ACM/noisyDBLP2_Sample.csv')
        if not os.path.exists('DBLP-ACM/noisyACM_Sample.csv'):
            sample_acm.to_csv('DBLP-ACM/noisyACM_Sample.csv')
        gt_df = pd.DataFrame.from_records(self.gt_data, columns=['idDBLP', 'idACM'])
        if not os.path.exists('DBLP-ACM/noisy_perfectMappingSample.csv'):
            gt_df.to_csv('DBLP-ACM/noisy_perfectMappingSample.csv')

    def generate_bbprofiles(self, df_list, attribute_lst):
        self.df_list = df_list
        self.attributes = attribute_lst
        self.profiles['dataset_size'] = self.generate_dsize(df_list)
        rlen_distr = self.generate_rlens(df_list)
        stopword_distr = self.generate_stopdistr(df_list)
        self.profiles['record_length_q1'] = rlen_distr[0]
        self.profiles['record_length_q2'] = rlen_distr[1]
        self.profiles['record_length_q3'] = rlen_distr[2]
        self.profiles['percent_stop_word_q1'] = stopword_distr[0]
        self.profiles['percent_stop_word_q2'] = stopword_distr[1]
        self.profiles['percent_stop_word_q3'] = stopword_distr[2]
        self.profiles['num_attributes'] = self.generate_attributes()
        return self.profiles

    def generate_abprofiles(self, pairs, theta):
        avsim_distr = self.generate_avsim(pairs)
        sim_distr = self.generate_simdistr()
        stop_cnt = self.generate_stopcnt(pairs)
        f_score = self.generate_fscore(pairs, theta)
        for attribute in self.attributes:
            self.profiles['stopcnt_' + attribute] = stop_cnt[attribute]
            self.profiles['avsim_' + attribute] = avsim_distr[attribute]
            self.profiles['sim_' + attribute + '_q1'] = sim_distr[attribute][0]
            self.profiles['sim_' + attribute + '_q2'] = sim_distr[attribute][1]
            self.profiles['sim_' + attribute + '_q3'] = sim_distr[attribute][2]
        self.profiles['f-score'] = f_score
        return self.profiles

    def generate_dsize(self, df_list):
        return sum(len(df) for df in df_list)

    def generate_rlens(self, df_list):
        rlens = [len(rec['title']) for df in df_list for _, rec in df.iterrows()]
        return np.percentile(rlens, [25, 50, 75], interpolation='midpoint')

    def generate_stopdistr(self, df_list):
        token_count = []
        stop_words = []
        for df in df_list:
            token_freq = {}
            for _, rec in df.iterrows():
                for token in rec['title'].lower().split():
                    token_freq[token] = token_freq.get(token, 0) + 1
            sorted_token = sorted(token_freq.items(), key=operator.itemgetter(1))
            stop_words.append(sorted_token[-max(1, int(0.1 * len(sorted_token))):])
            token_count.append(token_freq)
        self.stopword_lst = stop_words

        stopword_percents = []
        for df, sw in zip(df_list, stop_words):
            stopword_set = {x[0] for x in sw}
            for _, rec in df.iterrows():
                tokens = rec['title'].lower().split()
                stopword_cnt = sum(1 for token in tokens if token in stopword_set)
                stopword_percents.append(stopword_cnt / len(tokens))
        return np.percentile(stopword_percents, [25, 50, 75], interpolation='midpoint')

    def generate_attributes(self):
        return len(self.attributes)

    def generate_sim(self, pairs):
        self.sim = {attribute: [] for attribute in self.attributes}
        t1, t2 = self.df_list[0], self.df_list[1]
        for (id1, id2) in pairs:
            for attribute in self.attributes:
                a1 = t1[t1['id']==id1][attribute].tolist()[0]
                a2 = t2[t2['id']==id2][attribute].tolist()[0]
                self.sim[attribute].append(self.get_sim(a1, a2))

    def generate_stopcnt(self, pairs):
        stop_words = {x[0] for sw in self.stopword_lst for x in sw}
        stop_cnt = {attribute: 0 for attribute in self.attributes}
        t1, t2 = self.df_list[0], self.df_list[1]
        for (id1, id2) in pairs:
            for attribute in self.attributes:
                tokens1 = t1[t1['id']==id1][attribute].tolist()[0].split()
                tokens2 = t2[t2['id']==id2][attribute].tolist()[0].split()
                stop_cnt[attribute] += sum(token in stop_words for token in tokens1)
                stop_cnt[attribute] += sum(token in stop_words for token in tokens2)
        return stop_cnt

    def generate_avsim(self, pairs):
        self.generate_sim(pairs)
        return {attribute: sum(self.sim[attribute]) / len(pairs) for attribute in self.sim} if pairs else -1

    def generate_simdistr(self):
        return {attribute: np.percentile(self.sim[attribute], [25, 50, 75], interpolation='midpoint') for attribute in self.sim}

    def get_sim(self, text1, text2):
        try:
            l1 = set(text1.lower().replace(',', '').split())
            l2 = set(text2.lower().replace(',', '').split())
            return len(l1 & l2) / len(l1.union(l2))
        except:
            return 0

    def generate_fscore(self, pairs, theta):
        Jm = JaccardMatching(theta)
        gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMappingSample.csv')
        gt.drop(gt.columns[[0]], axis=1, inplace=True)
        tp, fp, tn, fn = Jm.pair_matching(pairs, self.df_list, gt.values.tolist())
        if tp == 0:
            return 0
        fn = len(self.gt_data) - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

    # ML Profiling
    def outlier(self, lst):
        mean = statistics.mean(lst)
        std = statistics.stdev(lst)
        return sum(1 for v in lst if v > mean + 2 * std or v < mean - 2 * std) / len(lst)

    def missing(self, lst):
        return sum(1 for v in lst if pd.isnull(v) or (hasattr(v, '__len__') and len(v) == 0)) / len(lst)

    def correlation(self, lst1, lst2):
        try:
            r, _ = pearsonr(lst1, lst2)
            return r
        except:
            return 0

    def categorical_correlation(self, lst1, lst2):
        cross_tab = pd.crosstab(lst1, lst2)
        chi2, _, _, _ = chi2_contingency(cross_tab)
        return chi2

    def categorical_numerical_correlation(self, lst1, lst2):
        chi2, p = stats.f_oneway(lst1, lst2)
        return chi2

    def get_fraction_of_outlier(self, data):
        svm_model = OneClassSVM(kernel='rbf')
        svm_model.fit(data)
        return (svm_model.predict(data) == -1).sum() / len(data)

    def populate_profiles(self, data_final, outlier, dataset, metric_type, numerical_columns):
        profile = {}
        target = {'hmda': 'action_taken', 'adult': 'income', 'housing': 'SalePrice'}.get(dataset)

        for column in data_final.columns:
            if column == target:
                continue
            if metric_type in ('rmse', 'mae'):
                corr = self.correlation(data_final[column], data_final[target]) if column in numerical_columns else self.categorical_numerical_correlation(data_final[column], data_final[target])
            else:
                corr = self.categorical_numerical_correlation(data_final[column], data_final[target]) if column in numerical_columns else self.categorical_correlation(data_final[column], data_final[target])
            profile[(f'corr_{column}', f'ot_{column}')] = [column, round(corr,5), round(outlier,5)]

        dd, keys = [], []
        for val in profile:
            dd.append(profile[val][1])
            dd.append(profile[val][2])
            keys.append(val[0])
            keys.append(val[1])
        return dd, keys
