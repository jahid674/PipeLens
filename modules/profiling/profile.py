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


class Profile:
    profile_lst = []

    def __init__(self):
        #self.df = df
        self.profiles = {}
        self.df_list = {}
        self.attributes = {}
        self.stopword_lst = {}
        self.sim = {}
        self.gt_data = []
    
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
        if not(os.path.exists('DBLP-ACM/noisyDBLP2_Sample.csv')):
            sample_dblp.to_csv('DBLP-ACM/noisyDBLP2_Sample.csv')
        if not(os.path.exists('DBLP-ACM/noisyACM_Sample.csv')):
            sample_acm.to_csv('DBLP-ACM/noisyACM_Sample.csv')
        gt_df = pd.DataFrame.from_records(self.gt_data, columns=['idDBLP', 'idACM'])
        if not(os.path.exists('DBLP-ACM/noisy_perfectMappingSample.csv')):
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
        #missing_vals = self.generate_missingvals(pairs, gt_list)
        for attribute in self.attributes:
            self.profiles['stopcnt_' + attribute] = stop_cnt[attribute]
            self.profiles['avsim_' + attribute] = avsim_distr[attribute]
            self.profiles['sim_' + attribute + '_q1'] = sim_distr[attribute][0]
            self.profiles['sim_' + attribute + '_q2'] = sim_distr[attribute][1]
            self.profiles['sim_' + attribute + '_q3'] = sim_distr[attribute][2]
        self.profiles['f-score'] = f_score
        #self.profiles['missing_vals'] = missing_vals
        return self.profiles

    def generate_dsize(self, df_list):
        size = 0
        for df in df_list:
            size += len(df)
        return size
    
    def generate_rlens(self, df_list):
        rlens = []
        for df in df_list:
            for index, rec in df.iterrows():
                rlens.append(len(rec['title']))
        distr = (np.percentile(rlens, [25, 50, 75], interpolation='midpoint'))
        return distr
    
    def generate_stopdistr(self, df_list):
        token_count = []
        stop_words = []
        for i in range(len(df_list)):
            token_count.append({})
            for index, rec in df_list[i].iterrows():
                cur_token = rec['title'].lower()
                for token in cur_token.split():
                    if token not in token_count[i]:
                        token_count[i][token] = 0
                    token_count[i][token] += 1
            #print(len(token_count[i]))
            sorted_token = sorted(token_count[i].items(), key=operator.itemgetter(1))
            #print(len(sorted_token[-int((0.1)*len(sorted_token)):]))
            stop_words.append(sorted_token[-int((0.1)*len(sorted_token)):])
        self.stopword_lst = stop_words
        stopword_percents = []
        for i in range(len(df_list)):
            stopword_lst = [x[0] for x in stop_words[i]]
            for index, rec in df_list[i].iterrows():
                cur_token = rec['title'].lower()
                stopword_cnt = 0
                token_cnt = 0
                for token in cur_token.split():
                    if token in stopword_lst:
                        stopword_cnt += 1
                    token_cnt += 1
                stopword_percents.append(stopword_cnt / token_cnt)
        distr = (np.percentile(stopword_percents, [25, 50, 75], interpolation='midpoint'))
        return distr
    
    def generate_attributes(self):
        return len(self.attributes)
    
    def generate_sim(self, pairs):
       sim = {}
       t1 = self.df_list[0]
       t2 = self.df_list[1]
       for attribute in self.attributes:
           sim[attribute] = []
       for (id1, id2) in pairs:
           for attribute in self.attributes:
              a1 = t1[t1['id']==id1][attribute].tolist()[0]
              a2 = t2[t2['id']==id2][attribute].tolist()[0]
              #print(a1, a2)
              sim[attribute].append(self.get_sim(a1, a2))
       self.sim = sim
    
    def generate_stopcnt(self, pairs):
        stop_words = set()
        for lst in self.stopword_lst:
            stop_words.update([x[0] for x in lst])
        stop_cnt = {}
        for attribute in self.attributes:
            stop_cnt[attribute] = 0
        t1 = self.df_list[0]
        t2 = self.df_list[1]
        for (id1, id2) in pairs:
           for attribute in self.attributes:
              for token in t1[t1['id']==id1][attribute].tolist()[0].split():
                  if token in stop_words:
                      stop_cnt[attribute] += 1
              for token in t2[t2['id']==id2][attribute].tolist()[0].split():
                  if token in stop_words:
                      stop_cnt[attribute] += 1
        return stop_cnt
    
    def generate_avsim(self, pairs):
        self.generate_sim(pairs)
        av_sim = {}
        if len(pairs) == 0:
            return -1
        for attribute in self.sim.keys():
            av_sim[attribute] = sum(self.sim[attribute]) / len(pairs)
        return av_sim
    
    def generate_simdistr(self):
        distr = {}
        for attribute in self.sim.keys():
            distr[attribute] = np.percentile(self.sim[attribute], [25, 50, 75], interpolation='midpoint')
        return distr
       
    #Calculates similarity between set of words in text1 and text2
    def get_sim(self, text1,text2):
        try:
            l1=set(text1.lower().replace(',','').split())
            l2=set(text2.lower().replace(',','').split())
        except:
            l1=set([text1])
            l2=set([text2])
        return len(list(l1&l2))*1.0/len(list(l1.union(l2)))
    
    def generate_fscore(self, pairs, theta):
        Jm = JaccardMatching(theta)
        gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMappingSample.csv')
        gt.drop(gt.columns[[0]], axis=1, inplace=True)
        (tp,fp,tn,fn) = Jm.pair_matching(pairs,self.df_list,gt.values.tolist())
        if tp == 0:
            return 0
        fn = len(self.gt_data) - tp
        cur_p = round(tp / (tp + fp), 5)
        cur_r = round(tp / (tp + fn), 5)
        f_score = round((2 * cur_p * cur_r) / (cur_p + cur_r), 5)
        return f_score
    
    def generate_missingvals(self, pairs, gt_list):
        pm = PerfectMatching()
        (_,_,_,fn) = pm.pair_matching(pairs, gt_list)
        return fn
    
    def outlier(self,lst):
        mean=statistics.mean(lst)
        std=statistics.stdev(lst)
        count=0
        for v in lst:
            if v>mean+2*std or v<mean-2*std:
                count+=1
            return count*1.0/len(lst)

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

    def correlation(self,lst1,lst2):
        try:
            (r,p)= pearsonr(lst1,lst2)
            if True:
                    return r
            else:
                    return 0
        except:
            print("efe")

    def categorical_correlation(self,lst1,lst2):
        cross_tab=pd.crosstab(lst1,lst2)
        chi2, p, dof, ex=chi2_contingency(cross_tab)
        return chi2

    def categorical_numerical_correlation(self, lst1, lst2):
        (chi2, p) = stats.f_oneway(lst1, lst2)
        return chi2
    

    
    def get_fraction_of_outlier(self,data):
        svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
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

            name = column
            tuple = ('corr_' + name,  'ot_' + name)
            profile[tuple]= [column,round(corr*scaling_factor,5),round(outlier*scaling_factor,5)]
            #i+=1

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