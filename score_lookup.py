import itertools
import operator
import numpy as np
import pandas as pd

class ScoreLookup:
    def __init__(self, pipeline_order, metric_type):
        self.pipeline_order = pipeline_order
        self.metric_type = metric_type

    def f_score_look_up2(self,profiles_df,elem):
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
    
    def write_quartiles(csv_writer, f_goal, f_goals, algorithm, metric, quartiles, dataset_name):
        if dataset_name in ['adult', 'hmda']:
            fair_label = round(1 - f_goal, 2)
        else:
            fair_label = round(1 - (f_goal - min(f_goals)) / min(f_goals), 2)

        for i, q in enumerate(quartiles, 1):
            csv_writer.writerow([fair_label, algorithm, f"{metric} q{i}", round(q, 5)])

    
    