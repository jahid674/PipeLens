import pandas as pd
from modules.blocking.token import TokenBlocking
from modules.blocking.qgram import QGramBlocking
from modules.matching.jaccardmatching import JaccardMatching

#datasets to apply record linkage
t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
t2=pd.read_csv('DBLP-ACM/ACM.csv')

#ground truth
gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
gt_list = gt.values.tolist()

def token_blocking_metrics():
  #token blocking metrics
  token_blocking_data = []
  #very small theta for token blocking results in no block pruning and extremely long runtimes
  for i in range(5,100):
    for j in range(100):
        cur_tk = TokenBlocking(theta = i * 0.01)
        cur_jm = JaccardMatching(theta = j * 0.01)
        (tp,fp,tn,fn) = cur_jm.pair_matching(cur_tk.generate_pairs_from_dataframe([t1,t2],['title']),[t1,t2],gt_list)
        fn = len(gt) - tp
        cur_p = round(tp / (tp + fp), 5)
        cur_r = round(tp / (tp + fn), 5)
        cur_f = round((2 * cur_p * cur_r) / (cur_p + cur_r), 5)
        cur_a = round((tp + tn) / (tp + fp + tn + fn), 5)
        token_blocking_data.append((round(i * 0.01, 5), round(j * 0.01, 5), cur_p, cur_r, cur_f, cur_a))
  token_blocking_df = pd.DataFrame.from_records(token_blocking_data, columns=['blocking threshold', 'match threshold', 'precision', 'recall', 'f-score', 'acc'])
  token_blocking_df.to_csv('ERmetrics/token_blocking.csv')

def qgram_blocking_metrics():
   #qgram blocking metrics
  qgram_blocking_data = []
  #very small theta for token blocking results in no block pruning and extremely long runtimes
  for i in range(5,20):
    for j in range(100):
        for k in range(4,7):
          cur_qb = QGramBlocking(qsize = k, theta = i * 0.05)
          cur_jm = JaccardMatching(theta = j * 0.01)
          (tp,fp,tn,fn) = cur_jm.pair_matching(cur_qb.generate_pairs_from_dataframe([t1,t2],['title']),[t1,t2],gt_list)
          fn = len(gt) - tp
          cur_p = round(tp / (tp + fp), 5)
          cur_r = round(tp / (tp + fn), 5)
          cur_f = round((2 * cur_p * cur_r) / (cur_p + cur_r), 5)
          cur_a = round((tp + tn) / (tp + fp + tn + fn), 5)
          qgram_blocking_data.append((k, round(i * 0.05, 5), round(j * 0.01, 5), cur_p, cur_r, cur_f, cur_a))
  qgram_blocking_df = pd.DataFrame.from_records(qgram_blocking_data, columns=['q-gram size','blocking threshold', 'match threshold', 'precision', 'recall', 'f-score', 'acc'])
  qgram_blocking_df.to_csv('ERmetrics/qgram_blocking.csv')

qgram_blocking_metrics()