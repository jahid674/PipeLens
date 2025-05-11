import pandas as pd
from sklearn.linear_model import LinearRegression
from modules.profiling.profile import Profile
from workingerpipeline import BlockBuilding, BlockCleaning, ComparisonCleaning, Matching

class Regression:
    def __init__(self):
        print("initializing regression")
    def generate_regression(self, xvals, yvals):
        model = LinearRegression().fit(xvals, yvals)
        return model
    def test(self):
        token_stats = pd.read_csv('ERmetrics/token_blocking.csv')
        X = token_stats[['blocking threshold', 'match threshold']]
        y = token_stats['f-score']
        model = self.generate_regression(X, y)
        print(model.coef_)
        print(model.intercept_)

    
if __name__ == "__main__":
    reg = Regression()
    #reg.test()
    profile = Profile(df=None)

    t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
    t2=pd.read_csv('DBLP-ACM/ACM.csv')

    profile.generate_bbprofiles([t1, t2], ['title'])

    gt = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
    gt_list = gt.values.tolist()

    q_size = 0
    BuBl = BlockBuilding(0)
    blocks = BuBl.create_blocks_from_dataframe([t1,t2],['title'])
    
    matching_thres = 0.75
    Jm = Matching(matching_thres)

    profile_data = []
    for i in range(2):
        pf = False if i == 0 else True
        for j in range(4):
            block_clean_thres = 0.05 + (0.225 * j)
            for k in range(4):
              BlCl = BlockCleaning(pf,block_clean_thres)
              CoCl = ComparisonCleaning(k)
              pairs = CoCl.generate_pairs(BlCl.clean_blocks(blocks))
              stop_cnt = 0
              av_sim = 0
              sim_q1 = 0
              sim_q2 = 0
              sim_q3 = 0
              cur_f = 0
              if len(pairs) > 0:
                ab_profiles = profile.generate_abprofiles(pairs)
                stop_cnt = round(ab_profiles['stopcnt_title'], 5)
                av_sim = round(ab_profiles['avsim_title'], 5)
                sim_q1 = round(ab_profiles['sim_title_q1'], 5)
                sim_q2 = round(ab_profiles['sim_title_q2'], 5)
                sim_q3 = round(ab_profiles['sim_title_q3'], 5)
                (tp,fp,tn,fn) = Jm.pair_matching(pairs,[t1,t2],gt_list)
                cur_p = round(tp / (tp + fp), 5)
                cur_r = round(tp / (tp + fn), 5)
                cur_f = round((2 * cur_p * cur_r) / (cur_p + cur_r), 5)
              profile_data.append((i, block_clean_thres, k, stop_cnt, av_sim, sim_q1, sim_q2, sim_q3, cur_f))
    profiles_df = pd.DataFrame.from_records(profile_data, columns=['pf','block clean threshold', 'weighting schema', 'stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3', 'f-score'])
    profiles_df.to_csv('ERmetrics/profiles.csv')

    print("beginning regression generation")
    X = profiles_df[['stop count', 'av sim', 'sim q1', 'sim q2', 'sim q3']]
    y = profiles_df['f-score']
    model = reg.generate_regression(X, y)
    print(model.coef_)
    print(model.intercept_)