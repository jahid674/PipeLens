import pandas as pd

path = "BugDoc-master/examples/bugdoc_test_sim_historical_data_test_profile_lr_rmse_housing.csv"
data = pd.read_csv(path)

if 'model' in data.columns:
    data = data.drop(columns=['model'])

data.to_csv(path, index=False)


