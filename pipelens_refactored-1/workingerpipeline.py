'''
Working version of pipeline w/ all existing parameters
'''

import operator
import math
import pandas as pd
import copy

'''
BlockBuilding: Parameter qsize. If qsize > 0, use qgram blocking with q-gram size qsize. Otherwise, use token blocking
'''
class BlockBuilding:
    def __init__(self,qsize):
        print ("initializing block building")
        self.blocks=[]
        self.qsize = qsize

    #Standardizes record to lowercase
    def clean_record (self, r):
        r=r.lower()
        return r

    #Create blocks from dataframe function!
    #Assumes a column called id
    def create_blocks_from_dataframe(self, df_lst, attribute_lst):
        for i in range(len(df_lst)):
            self.blocks.append({})
        for attribute in attribute_lst:
            iter=0
            for df in df_lst:
                for index,rec in df.iterrows():
                    self.add_record(rec[attribute],rec['id'],iter)
                iter+=1
        return self.blocks

    
    #Assumes a record r and its id  is a text
    def add_record(self, r, rid, table_id=0):
        r = self.clean_record(r)

        if self.qsize <= 0:
            for token in r.split():
              lst=[]
              if token in self.blocks[table_id].keys():
                # add to existing list of rids associated w/ token
                lst = self.blocks[table_id][token]
              if rid not in lst:
                lst.append(rid)
              self.blocks[table_id][token] = lst

        else:
          iter=0
          while iter<len(r)-self.qsize:
            token = r[iter:iter+self.qsize]
            lst=[]
            if token in self.blocks[table_id].keys():
                lst = self.blocks[table_id][token]
            if rid not in lst:
                lst.append(rid)
            self.blocks[table_id][token] = lst
            iter+=1

class BlockCleaning:
   #pf is boolean param that represents block purging if true and block filtering if false
   def __init__(self,pf,thres):
        print ("initializing block cleaning")
        self.pf = pf
        self.thres = thres

   def clean_blocks(self, blocks):
      if self.pf:
        return self.purge_blocks(blocks)
      return self.filter_blocks(blocks)
   
   def purge_blocks(self, input_blocks):
      blocks = copy.deepcopy(input_blocks)
      size_dic={}
      for block_token in blocks[0].keys():
        size = 1
        for bl in blocks:
            if block_token in bl.keys():
                    # multiply size by num rids in corresponding block token of both datasets
                    size *= len(bl[block_token])
            if size>0:
                size_dic[block_token]=size
      sorted_size = sorted(size_dic.items(), key=operator.itemgetter(1))
        
      # prune last theta blocks (block, num rids)
      final_blocks=sorted_size[:int((1-self.thres)*len(sorted_size))]
      final_lst=set()
      for (bl,blsize) in final_blocks:
        final_lst.add(bl)

      for table in range(len(blocks)):
          for block_token in blocks[table].copy().keys():
              if block_token not in final_lst:
                  blocks[table].pop(block_token)

      return blocks
      
   
   def filter_blocks(self, input_blocks):
      blocks = copy.deepcopy(input_blocks)
      for table in range(len(blocks)):
        # sort blocks by descending importance
        size_dic={}
        for block_token in blocks[table].keys():
            if len(blocks[table][block_token])>0:
                size_dic[block_token]=len(blocks[table][block_token])
        sorted_size = sorted(size_dic.items(), key=operator.itemgetter(1))
        # limit per profile
        thresholds={}
        counter={}
        for block_token in blocks[table].keys():
            for rid in blocks[table][block_token]:
                if rid in thresholds:
                    thresholds[rid] += 1
                else:
                    thresholds[rid] = 1
                    counter[rid] = 0

        for (bl,blsize) in sorted_size:
            for rid in blocks[table][bl].copy():
                if counter[rid] > thresholds[rid] * (1 - self.thres):
                    blocks[table][bl].remove(rid)
                else:
                    counter[rid] += 1
            if len(blocks[table][bl]) < 2:
                blocks[table].pop(bl)
      return blocks
   
class ComparisonCleaning:
    def __init__(self, weighting_scheme):
        print ("initializing comparison cleaning")
        self.weighting = weighting_scheme

    def gen_block_list(self, blocks):
        lst = {}
        for table in range(len(blocks)):
            for bl in blocks[table].keys():
                for id in blocks[table][bl]:
                    if id not in lst.keys():
                        lst[id] = []
                    lst[id].append(bl)

        return lst

    def generate_pairs(self, blocks):
        pairs_identified=set()
        # deduplication
        if len(blocks)==1:
            for bl in blocks[0].keys():
                rec_lst=blocks[0][bl]
                for i in range(len(rec_lst)):
                    for j in range(i,len(rec_lst)):
                        if rec_lst[i]<rec_lst[j]:
                            pairs_identified.add((rec_lst[i],rec_lst[j]))
                        else:
                            pairs_identified.add((rec_lst[j],rec_lst[i]))
        # record linkage
        else:
            for bl in blocks[0].keys():
                if bl in blocks[1].keys():
                    rec_lst1=blocks[0][bl]
                    rec_lst2=blocks[1][bl]
                    for i in range(len(rec_lst1)):
                        for j in range(len(rec_lst2)):
                            pairs_identified.add((rec_lst1[i],rec_lst2[j]))
        return self.clean_pairs(pairs_identified, blocks)

    def clean_pairs(self, pairs, blocks):
        id_list = []
        if len(blocks) == 1:
            ids = set()
            for (r1, r2) in pairs:
                ids.add(r1)
                ids.add(r2)
            id_list = list(ids)
        else:
            ids1 = set()
            ids2 = set()
            for (r1, r2) in pairs:
                ids1.add(r1)
                ids2.add(r2)
            id_list = list(ids1) + list(ids2)

        map = {}
        for x in range(len(id_list)):
            map[id_list[x]] = x
        #adjacency matrix representation of graph
        distinct_edges = 0
        total_weight = 0
        adj_matrix = [[0]*len(id_list) for i in range(len(id_list))]
        block_lists = self.gen_block_list(blocks)
        for (r1, r2) in pairs:
            common_blocks = []
            bl_i = block_lists[r1]
            bl_j = block_lists[r2]
            for bl in bl_i:
                if bl in bl_j:
                    common_blocks.append(bl)
            if adj_matrix[map[r1]][map[r2]] == 0:
                weight = self.calculate_weights(common_blocks, bl_i, bl_j, blocks)
                adj_matrix[map[r1]][map[r2]] = weight
                adj_matrix[map[r2]][map[r1]] = weight
                distinct_edges += 1
                total_weight += weight
        
        new_pairs = set()
        for i in range(len(id_list)):
            for j in range(len(id_list)):
                if i < j:
                    # WEP w/ minimum edge weight as average of all edge weights
                    if round(adj_matrix[i][j], 5) >= round(total_weight/distinct_edges, 5):
                        new_pairs.add((id_list[i], id_list[j]))
                        #print(str(id_list[i]) + ", " + str(id_list[j]))
        return new_pairs
    
    def calculate_weights(self, common_blocks, bl_i, bl_j, blocks):
        #ARCS
        if self.weighting == 0:
            weight = 0
            for bl in common_blocks:
                av_cardinality = 0
                for table in range(len(blocks)):
                    av_cardinality += len(blocks[table][bl])
                weight += 1/(av_cardinality / len(blocks))
            return weight
        #CBS
        elif self.weighting == 1:
            return len(common_blocks)
        #ECBS
        elif self.weighting == 2:
            total_blocks = 0
            for table in range(len(blocks)):
                total_blocks += len(blocks[table].keys())
            return len(common_blocks) * math.log2(total_blocks / len(bl_i)) * math.log2(total_blocks / len(bl_j))
        #JS
        else:
            return len(common_blocks) / (len(bl_i) + len(bl_j) - len(common_blocks))
        
class Matching:
    def __init__(self,theta):
        print ("initializing matching component")
        self.match_thres=theta
    
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
        fn=len(list(set(fn)))
        return tp,fp,tn,fn

if __name__ == "__main__":
      t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
      t2=pd.read_csv('DBLP-ACM/ACM.csv')

      gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
      gt_list = gt.values.tolist()

      q_size = 0
      pf = False
      block_clean_thres = 0.4
      comparison_weighting_scheme = 1
      matching_thres = 0.75

      BuBl = BlockBuilding(0)
      BlCl = BlockCleaning(pf,block_clean_thres)
      CoCl = ComparisonCleaning(comparison_weighting_scheme)
      Jm = Matching(matching_thres)

      pairs = CoCl.generate_pairs(BlCl.clean_blocks(BuBl.create_blocks_from_dataframe([t1,t2],['title'])))

      (tp,fp,tn,fn) = Jm.pair_matching(pairs,[t1,t2],gt_list)

      print (tp,fp,tn,fn)
      print ((tp + tn)/(tp + fp + tn + fn))


        