'''
Jaccard similarity based matching for Entity Resolution
Assumption: each record is a dictionary with a schema!

TODO: Add parameters here
'''


class JaccardMatching:
    def __init__(self,theta=0.75):
        print ("initializing matching component")
        self.match_thres=theta
        self.parameters=[self.match_thres]
    
    #Calculates similarity between set of words in text1 and text2
    def get_sim(self, text1,text2):
        try:
            l1=set(text1.lower().replace(',','').split())
            l2=set(text2.lower().replace(',','').split())
        except:
            l1=set([text1])
            l2=set([text2])
        #print (l1,l2, l1&l2, l1.union(l2))
        return len(list(l1&l2))*1.0/len(list(l1.union(l2)))

    '''
    Assumes records is a pair of dataframes, where each dataframe has a column called id
    '''
    def pair_matching (self, pair_lst, records, gt):
        tp=[]
        fp=[]
        tn=[]
        fn=[]
        t1=records[0]
        t2=records[1]
        iter=0
        for (id1,id2) in pair_lst:
            if iter%100000==0:
                print (iter)
            r1 = t1[t1['id']==id1].values[0]
            r2 = t2[t2['id']==id2].values[0]

            #print (r1,r2)

            #compare titles, author, year why not venue?
            avg_sim =  (self.get_sim(r1[1],r2[1])+self.get_sim(r1[-1],r2[-1])+self.get_sim(r1[2],r2[2]))/3

            if avg_sim<self.match_thres:
                if [id1,id2] in gt:
                    fn.append((id1,id2))
                else:
                    tn.append((id1,id2))
            else:
                if [id1,id2] in gt:
                    tp.append((id1,id2))
                else:
                    fp.append((id1,id2))
                                
            iter+=1
                
        tp=len(list(set(tp)))
        fp=len(list(set(fp)))
        tn=len(list(set(tn)))
        #account for pairs in gt not originally in pair list
        fn=len(gt) - tp
        return tp,fp,tn,fn

if __name__ == "__main__":
    jm=JaccardMatching()
   