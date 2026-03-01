'''
Metablocking for Entity Resolution with weighted edge pruning (WEP) under common blocks scheme (CBS)

TODO: Add parameters here
'''
import operator

class MetaBlocking:
    def __init__(self,n_tables=1):
        print ("initializing metablocking")
        self.blocks=[]

        for i in range(n_tables):
            self.blocks.append({})
    
    '''
    Cleans the record r
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
                    self.add_record(rec[attribute],rec['id'],iter)
                iter+=1

    def generate_pairs_from_dataframe(self, df_lst, attribute_lst):
        self.create_blocks_from_dataframe(df_lst,attribute_lst)
        pairs_identified=[]
        if len(self.blocks)==1:
            for bl in self.blocks[0].keys():
                    rec_lst=self.blocks[0][bl]
                    for i in range(len(rec_lst)):
                        for j in range(i,len(rec_lst)):
                            if rec_lst[i]<rec_lst[j]:
                                pairs_identified.append((rec_lst[i],rec_lst[j]))
                            else:
                                pairs_identified.append((rec_lst[j],rec_lst[i]))
        else:
            for bl in self.blocks[0].keys():
                if bl in self.blocks[1].keys():
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

    '''
    Assumes a dictionary that maps record id to record, where each record is a text attribute
    table_id denotes the corresponding table number for recordlst
    '''
    def create_blocks(self, record_lst,table_id=0):
        for r_id in record_lst.keys():
            self.add_record(record_lst[r_id],r_id)

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
