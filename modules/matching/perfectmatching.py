'''
Perfect matching with knowledge of ground truth
Assumption: each record is a dictionary with a schema!

TODO: Add parameters here
'''

class PerfectMatching:
    def __init__(self):
        print ("initializing matching component")
    
    '''
    Assumes records is a pair of dataframes, where each dataframe has a column called id
    '''
    def pair_matching (self, pair_lst, gt):
        tp=[]
        fp=[]
        tn=[]
        fn=[]
        for (id1,id2) in pair_lst:
          if [id1,id2] in gt:
              tp.append((id1,id2))
          else:
              fp.append((id1,id2))
                                
        tp=len(list(set(tp)))
        fp=len(list(set(fp)))
        tn=len(list(set(tn)))
        #account for pairs in gt not originally in pair list
        fn=len(gt) - tp
        return tp,fp,tn,fn

   