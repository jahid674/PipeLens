import pandas as pd

path = "BugDoc-master/examples/Bugdoc_test_lr_sp_adult.csv"
data = pd.read_csv(path)

if 'model' in data.columns:
    data = data.drop(columns=['model'])

data.to_csv(path, index=False)


