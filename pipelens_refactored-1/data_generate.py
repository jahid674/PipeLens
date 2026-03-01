import pandas as pd

df = pd.read_csv('historical_data/historical_data_sim_profile_lr_sp_adult.csv')
df.insert(2, 'unit_converter', 1)
df.to_csv('historical_data/component_data/unit_converter_historical_data_train_profile_lr_sp_adult.csv', index=False)
