'''
Implement standard blocking described by Papadakis et al. w/ 
block filtering (BlFi) for block cleaning and weighted edge pruning (WEP) with 
common blocks scheme (CBS) for comparison cleaning

TODO: Add parameters here
'''
import operator

class StandardBlocking:
    def __init__(self,n_tables=1,theta=0):
        print ("initializing standard blocking")
        self.blocks=[]

        #parameter to denote the percentage to filter
        self.theta=theta

        self.parameters=[self.theta]
        
    
    '''
    Cleans the record r (makes it lowercase)
    '''
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
                    # title, id, csv 1 or csv 2 in erpipeline
                    self.add_record(rec[attribute],rec['id'],iter)

                iter+=1
        for x in range(len(df_lst)):
            self.clean_blocks(x)


    def generate_pairs_from_dataframe(self, df_lst, attribute_lst):
        self.create_blocks_from_dataframe(df_lst,attribute_lst)
        pairs_identified=[]
        # deduplication
        if len(self.blocks)==1:
            for bl in self.blocks[0].keys():
                if bl in self.final_blocks:
                    rec_lst=self.blocks[0][bl]
                    for i in range(len(rec_lst)):
                        for j in range(i,len(rec_lst)):
                            if rec_lst[i]<rec_lst[j]:
                                pairs_identified.append((rec_lst[i],rec_lst[j]))
                            else:
                                pairs_identified.append((rec_lst[j],rec_lst[i]))
        # record linkage
        else:
            for bl in self.blocks[0].keys():
                if bl in self.final_blocks and bl in self.blocks[1].keys():
                    rec_lst1=self.blocks[0][bl]
                    rec_lst2=self.blocks[1][bl]
                    for i in range(len(rec_lst1)):
                        for j in range(len(rec_lst2)):
                            pairs_identified.append((rec_lst1[i],rec_lst2[j]))
        return self.clean_pairs(pairs_identified, len(self.blocks))
    
    def clean_pairs(self, pairs, num_tables):
        id_list = []
        if num_tables == 1:
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
        adj_matrix = [[0]*len(id_list) for i in range(len(id_list))]
        for (r1, r2) in pairs:
            if adj_matrix[map[r1]][map[r2]] == 0:
                distinct_edges += 1
            adj_matrix[map[r1]][map[r2]] += 1
            adj_matrix[map[r2]][map[r1]] += 1
        new_pairs = set()
        for i in range(len(id_list)):
            for j in range(len(id_list)):
                if i < j:
                    # WEP w/ minimum edge weight as average of all edge weights
                    if adj_matrix[i][j] >= len(pairs)/distinct_edges:
                        new_pairs.add((id_list[i], id_list[j]))
        return new_pairs
        

    def clean_blocks(self, table):
        # sort blocks by descending importance
        size_dic={}
        for block_token in self.blocks[table].keys():
            if len(self.blocks[table][block_token])>0:
                size_dic[block_token]=len(self.blocks[table][block_token])
        sorted_size = sorted(size_dic.items(), key=operator.itemgetter(1))
        # limit per profile
        thresholds={}
        counter={}
        for block_token in self.blocks[table].keys():
            for rid in self.blocks[table][block_token]:
                if rid in thresholds:
                    thresholds[rid] += 1
                else:
                    thresholds[rid] = 1
                    counter[rid] = 0

        self.final_blocks = []

        for (bl,blsize) in sorted_size:
            for rid in self.blocks[table][bl].copy():
                if counter[rid] > thresholds[rid] * (1 - self.theta):
                    self.blocks[table][bl].remove(rid)
                else:
                    counter[rid] += 1
            if len(self.blocks[table][bl]) >= 2:
                self.final_blocks.append(bl)
       
        for x in range(10):
            if (len(sorted_size) - 1 - x) >= 0:
              self.print_block(sorted_size[len(sorted_size) - 1 - x][0], table)


    '''
    Assumes a dictionary that maps record id to record, where each record is a text attribute
    table_id denotes the corresponding table number for recordlst
    '''
    def create_blocks(self, record_lst,table_id=0):
        for r_id in record_lst.keys():
            self.add_record(record_lst[r_id],r_id)

        self.clean_blocks()


    '''
    Assumes a record r and its id  is a text
    '''
    def add_record(self, r, rid, table_id=0):
        r = self.clean_record(r)
        for token in r.split():
            lst=[]
            if token in self.blocks[table_id].keys():
                # add to existing list of rids associated w/ token
                lst = self.blocks[table_id][token]
            if rid not in lst:
                lst.append(rid)
            self.blocks[table_id][token] = lst
    '''             
    def merge_blocks(self, cur_block, iter):
        if iter == 0:
            self.blocks = cur_block
        else:
            cur_block_keys = []
            self_blocks_keys = []
            for bl in self.blocks.copy().keys():
                self_blocks_keys.append((bl, 0))
            for bl in cur_block.keys():
                cur_block_keys.append((bl, 0))
            c_df = pd.DataFrame.from_records(cur_block_keys, columns=['cur-tokens', 'empty'])
            c_df.to_csv('ERmetrics/c_blocking.csv')
            s_df = pd.DataFrame.from_records(self_blocks_keys, columns=['self-tokens', 'empty'])
            s_df.to_csv('ERmetrics/s_blocking.csv')
            
            for bl in self.blocks.copy().keys():
                if bl in cur_block.keys():
                    self.blocks[bl] = self.blocks[bl].union(cur_block[bl])
                else:
                    #print(bl)
                    self.blocks.pop(bl)
    '''  

    def print_block(self, token, table):
         print ("Block name:", token, len(self.blocks[table][token]))

