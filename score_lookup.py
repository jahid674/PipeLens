import itertools
import operator
import numpy as np
import pandas as pd

class ScoreLookup:
    def __init__(self, pipeline_order, metric_type):
        self.pipeline_order = pipeline_order
        self.metric_type = metric_type

    '''def utility_look_up(self, profiles_df, elem):
        try:
            mask = True
            for col, val in zip(self.pipeline_order, elem):
                mask &= (profiles_df[col] == val)

            result_row = profiles_df.loc[mask]
            if not result_row.empty:
                return round(result_row.iloc[0][f'utility_{self.metric_type}'], 5)
            else:
                return None
            
        except Exception as e:
            print(f"[ERROR] utility_look_up failed: {e}")
            return None'''
        
    def utility_look_up(self,profiles_df,elem):
                column_names = self.pipeline_order + [f'utility_{self.metric_type}']
                try:
                        return round(profiles_df.loc[(profiles_df[column_names[0]] == elem[0]) & (profiles_df[column_names[1]] == elem[1] ) 
                                               & (profiles_df[column_names[2]] == elem[2]) & (profiles_df[column_names[3]] == elem[3])].iloc[0][f'utility_{self.metric_type}'],5)
                except Exception as e :
                        print(e)
                        import pdb;pdb.set_trace()



    def identify_param(self, rank_list, comb_size):
                return list(itertools.combinations(rank_list, comb_size))
        
    def score_values(self, historical_data, coefs, comb_size, comb):
            i=0
            coef_lst=[]
            score = {}
            while i<comb_size:
                    coef_lst.append(coefs[comb[i]])
                    i+=1
                                        
            for param in historical_data:
                    result = [param[i] for i in list(comb)]
                    score[tuple(param)] = sum([x*y for x,y in zip( coef_lst,result)])
            sorted_params = sorted(score.items(), key=operator.itemgetter(1))

            return sorted_params

    
    