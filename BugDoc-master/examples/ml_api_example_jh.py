import time
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
def f_score_look_up2(profiles_df, elem, threshold):
    column_names =['missing_value', 'normalization', 'outlier', 'whitespace','punctuation','stopword','unit_converter','fairness']
    try:
        metric_value = profiles_df.loc[(profiles_df[column_names[0]] == elem[0]) 
                               & (profiles_df[column_names[1]] == elem[1]) 
                               & (profiles_df[column_names[2]] == elem[2]) 
                            ].iloc[0]['fairness']
        return metric_value < threshold
    except Exception as e :
        print(e)
        # import pdb;pdb.set_trace()
        
def execute_pipeline(configuration, historical_data, threshold):
    elem = []
    elem.append(int(configuration['missing_value']))
    elem.append(int(configuration['normalization']))
    elem.append(int(configuration['outlier']))
    elem.append(int(configuration['whitespace']))
    elem.append(int(configuration['punctuation']))
    elem.append(int(configuration['stopword']))
    elem.append(int(configuration['unit_converter']))
    return f_score_look_up2(historical_data, elem, threshold)