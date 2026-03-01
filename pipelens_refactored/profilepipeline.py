import pandas as pd
from modules.profiling.profile import Profile
from workingerpipeline import BlockBuilding, BlockCleaning, ComparisonCleaning

profile = Profile()

t1=pd.read_csv('DBLP-ACM/DBLP2.csv',encoding="latin-1")
t2=pd.read_csv('DBLP-ACM/ACM.csv')

bb_profiles = profile.generate_bbprofiles([t1, t2], ['title'])

print(bb_profiles)

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

pairs = CoCl.generate_pairs(BlCl.clean_blocks(BuBl.create_blocks_from_dataframe([t1,t2],['title'])))

ab_profiles = profile.generate_abprofiles(pairs)

print(ab_profiles)