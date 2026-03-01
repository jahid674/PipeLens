import pandas as pd
import random
import string
import os
from workingerpipeline import BlockBuilding, BlockCleaning, ComparisonCleaning, Matching

class Noise:
    # alpha represents percent of values to add noise to
    def __init__(self, alpha):
        print("generating noisy dataset")
        self.percent = alpha
    
    # adds noise to attribute for each df in df_lst
    def generate_noise(self, df_lst, attribute):
        for idx, df in enumerate(df_lst):
            df[attribute] = df[attribute].map(self.replace_spaces)
                
            if not(os.path.exists('DBLP-ACM/noisy2' + str(idx) + '.csv')):
                df.to_csv('DBLP-ACM/noisy2' + str(idx) + '.csv')

    def replace_spaces(self,title):
        result = title.replace(" ", "-")
        return result
        
    def generate_typo(self,title):
        if random.random() <= self.percent:
            print(title)
            title = list(title)
            #num typos
            n_chars_to_flip = round(len(title) * 0.2)

            #characters to add typos to
            pos_error = []
            for i in range(n_chars_to_flip):
                pos_error.append(random.randint(0, len(title) - 1))

            # insert typos
            for pos in pos_error:
                # try-except in case of special characters
                try:
                    # typo error
                    if random.random() <= 0.7:
                      title[pos] = random.choice(string.ascii_lowercase)
                    else:
                      # swap error
                      if pos != 0:
                          title[pos], title[pos-1] = title[pos - 1], title[pos]
                except:
                    break
            # recombine the message into a string
            title = ''.join(title)
            print(title)
            return title
        else:
            return title
    
if __name__ == "__main__":
    t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
    t2=pd.read_csv('DBLP-ACM/ACM.csv')
    gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
    gt_list = gt.values.tolist()

    N = Noise(alpha=0.3)
    N.generate_noise([t1,t2],['title'])

    n1=pd.read_csv('DBLP-ACM/noisy0.csv',encoding="latin-1")
    n2=pd.read_csv('DBLP-ACM/noisy1.csv')

    t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
    t2=pd.read_csv('DBLP-ACM/ACM.csv')
    
    q_size = 0
    pf = True
    block_clean_thres = 0.05
    comparison_weighting_scheme = 1
    matching_thres = 0.69286

    BuBl = BlockBuilding(0)
    BlCl = BlockCleaning(pf,block_clean_thres)
    CoCl = ComparisonCleaning(comparison_weighting_scheme)
    Jm = Matching(matching_thres)

    pairs = CoCl.generate_pairs(BlCl.clean_blocks(BuBl.create_blocks_from_dataframe([n1,n2],['title'])))

    (tp,fp,tn,fn) = Jm.pair_matching(pairs,[n1,n2],gt_list)

    print (tp,fp,tn,fn)
    print ((tp + tn)/(tp + fp + tn + fn))


