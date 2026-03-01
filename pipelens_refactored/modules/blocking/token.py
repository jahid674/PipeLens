'''
Token based blocking for Entity Resolution

TODO: Add parameters here
'''
import operator

class TokenBlocking:
    def __init__(self,n_tables=1,theta=0):
        print ("initializing token based blocking")
        self.blocks=[]

        #parameter to denote the percentage of blocks that will be pruned
        self.theta=theta


        self.parameters=[self.theta]
        for i in range(n_tables):
            self.blocks.append({})
    
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

        self.clean_blocks()

    def generate_pairs_from_dataframe(self, df_lst, attribute_lst):
        self.create_blocks_from_dataframe(df_lst,attribute_lst)
        pairs_identified=set()
        # deduplication
        if len(self.blocks)==1:
            for bl in self.blocks[0].keys():
                if bl in self.final_blocks:
                    rec_lst=self.blocks[0][bl]
                    for i in range(len(rec_lst)):
                        for j in range(i,len(rec_lst)):
                            if rec_lst[i]<rec_lst[j]:
                                pairs_identified.add((rec_lst[i],rec_lst[j]))
                            else:
                                pairs_identified.add((rec_lst[j],rec_lst[i]))
        # record linkage
        else:
            for bl in self.blocks[0].keys():
                if bl in self.final_blocks and bl in self.blocks[1].keys():
                    rec_lst1=self.blocks[0][bl]
                    rec_lst2=self.blocks[1][bl]
                    for i in range(len(rec_lst1)):
                        for j in range(len(rec_lst2)):
                            pairs_identified.add((rec_lst1[i],rec_lst2[j]))
        return pairs_identified


    def clean_blocks(self):
        size_dic={}
        for block_token in self.blocks[0].keys():
            size = 1
            for bl in self.blocks:
                if block_token in bl.keys():
                    # multiply size by num rids in corresponding block token of both datasets
                    size *= len(bl[block_token])
            if size>0:
                size_dic[block_token]=size
        sorted_size = sorted(size_dic.items(), key=operator.itemgetter(1))
        
        # prune last theta blocks (block, num rids)
        self.final_blocks=sorted_size[:int((1-self.theta)*len(sorted_size))]
        final_lst=[]
        for (bl,blsize) in self.final_blocks:
            final_lst.append(bl)
        self.final_blocks=final_lst
        print(len(self.final_blocks),len(sorted_size),sorted_size[:10],sorted_size[-10:])


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


    def print_blocks(self,table_id=0):
        for token in self.blocks[table_id].keys():
            if len(self.blocks[table_id][token])>1:
                print ("Block name:", token, len(self.blocks[table_id][token]))


if __name__ == "__main__":
    tk=TokenBlocking(2)
    tk.create_blocks({1:"AAAA hdjsk fhdsjk hello", 2:"hello hi"},0)
    tk.print_blocks()