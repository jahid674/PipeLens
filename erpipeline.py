import pandas as pd
from modules.blocking.token import TokenBlocking
from modules.matching.jaccardmatching import JaccardMatching

#Blocking options: Standard/token based, jaccard, meta blocking

#Qgram: Change the add record function!!!!!

#ER: Jaccard and its variations, learning based approaches, embedding based similarity

#Option 1: Train on a subset and test on validation as passing
#Failing dataset is the test data where we evaluate some metric on deployment


#Option 2: We train on a subset and test on validation every 24 hrs
#We evaluate validation accuracy and see how bad it is!


#Twitter data download https://snap.stanford.edu/data/bigdata/twitter7/tweets2009-06.txt.gz
#Look for others


#Amazon data has something called dedup!




t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
t2=pd.read_csv('DBLP-ACM/ACM.csv')

gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
gt_list = gt.values.tolist()

tk = TokenBlocking()

pairs=tk.generate_pairs_from_dataframe([t1,t2],['title'])


jm = JaccardMatching()


print(len(pairs))

(tp,fp,tn,fn) = jm.pair_matching(pairs,[t1,t2],gt_list)

print (tp,fp,tn,fn)