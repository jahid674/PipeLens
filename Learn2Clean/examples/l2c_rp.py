import pandas as pd
import learn2clean.loading.reader as rd 
import learn2clean.normalization.normalizer as nl 
import learn2clean.loading.reader as rd 
import learn2clean.qlearning.qlearner as ql
import learn2clean.imputation.imputer as imp
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import sys
import csv

class Reader():

    def __init__(self, train_path,test_path):
        self.train_path = train_path
        self.test_path = test_path


    def load_data(self):
        train = pd.DataFrame(pd.read_csv(self.train_path))
        test = pd.DataFrame(pd.read_csv(self.test_path))
        return train,test


data='housing' # 'housing', 'hmda'
metric_type='rmse' # 'rmse', 'accuracy'
tau = 0.1
if data == 'hmda':
    dataset = ["../../data/hmda/hmda_Orleans_X_test_1.csv", "../../data/hmda/hmda_Orleans_X_test_1.csv"]
    hr=rd.Reader(sep=',',verbose=False, encoding=False) 
    dataset=hr.train_test_split(dataset, 'action_taken')

    print(len(dataset['test']))
    # import pdb; pdb.set_trace();

    idx_train = np.arange(0, len(dataset['train']), 1)
    mv_train = pd.DataFrame(idx_train).sample(frac=tau, replace=False, random_state=1).index
    dataset['train']['lien_status'][mv_train] = np.NaN
                        
    idx_test = np.arange(0, len(dataset['train']), 1)
    mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
    dataset['test']['lien_status'][mv_test] = np.NaN

    metric_path  = '../../metric/metric_l2c_lr_'+metric_type+'_'+data+'.csv'
elif data == 'housing':
    d2 = pd.DataFrame(pd.read_csv('../datasets/house/housing_test.csv'))
    dataset = {
         'train': d2,
         'test':d2,
         'target':d2['SalePrice']
         }

    metric_path = '../../metric/metric_l2c_reg_'+metric_type+'_'+data+'.csv'

    selected_features = ['OverallQual', 'GarageFinish', 'GarageArea', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', 'YearRemodAdd', 'GrLivArea', 'GarageCars', 'FullBath', 'Fireplaces', 'BsmtQual', 'KitchenQual', 'ExterQual', 'TotRmsAbvGrd']
    print("These features have been selected as the KBest for passing data in our case.")

    missing_colmn_categorical = ['Electrical','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','GarageType','GarageFinish','GarageQual','GarageCond']
    missing_colmn_numerical  = ['LotFrontage','GarageYrBlt','MasVnrArea']
    for column in missing_colmn_categorical:
        most_frequent = dataset['train'][column].mode()[0]
        dataset['train'][column].fillna(most_frequent, inplace=True)
    for column in missing_colmn_numerical:
        median_value = dataset['train'][column].median()
        dataset['train'][column].fillna(median_value, inplace=True)

    selected_features.append('SalePrice')
    dataset['train'] = dataset['train'][selected_features]

    categorical_columns = dataset['train'].select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        dataset['train'][column] = le.fit_transform(dataset['train'][column]) 
        dataset['train'][column] = dataset['train'][column].astype('category')

    idx = np.arange(0, len(dataset['train']), 1)
    mv_idx = pd.DataFrame(idx).sample(frac=tau, replace=False, random_state=1).index
    dataset['train']['OverallQual'][mv_idx] = np.NaN
    
    # print("Modified dataset", dataset['train'])
    # X_train = dataset['train'].select_dtypes(['number']).dropna()
    # X_train = X_train.drop('SalePrice', axis=1)
    # y_train = dataset['train']['SalePrice'].loc[X_train.index]
    # model = LinearRegression()
    # y_pred = model.fit(X_train, y_train).predict(X_train)
    # rmse = np.sqrt(root_mean_squared_error(y_train, y_pred)) 
    # print("RP RMSE ------------", rmse)


def write_quartiles(csv_writer, algorithm, metric, quartiles, f_goal):
    if data in ['adult', 'hmda']:
            csv_writer.writerow([round(f_goal, 2), algorithm, f"{metric} q1", round(quartiles[0], 5)])
            csv_writer.writerow([round(f_goal, 2), algorithm, f"{metric} q2", round(quartiles[1], 5)])
            csv_writer.writerow([round(f_goal, 2), algorithm, f"{metric} q3", round(quartiles[2], 5)])
            csv_writer.writerow([round(f_goal, 2), algorithm, f"{metric} q4", round(quartiles[3], 5)])
    else:
            csv_writer.writerow([round(1 - (f_goal - min(goals))/min(goals), 2), algorithm, f"{metric} q1", round(quartiles[0], 5)])
            csv_writer.writerow([round(1 - (f_goal - min(goals))/min(goals), 2), algorithm, f"{metric} q2", round(quartiles[1], 5)])
            csv_writer.writerow([round(1 - (f_goal - min(goals))/min(goals), 2), algorithm, f"{metric} q3", round(quartiles[2], 5)])
            csv_writer.writerow([round(1 - (f_goal - min(goals))/min(goals), 2), algorithm, f"{metric} q4", round(quartiles[3], 5)])

f = sys.stdout

f = open(metric_path, 'w')

csv_writer = csv.writer(f)

if data == 'hmda':
    goals = [0.91, 0.92, 0.93, 0.94]
    # goals = [0.94]
elif data == 'housing':
    goals = [162, 170, 180, 185]

import random
random_seeds = random.sample(range(0, 1000000), 315)

for g in goals:
    iterations = []
    for seed in random_seeds:
    # for seed in [1999]:
        # print(seed)
        if data == 'hmda':
            l2c_c1assification1=ql.Qlearner(dataset = dataset,goal='LR',target_goal='action_taken',
                                    target_prepare='action_taken', verbose = False, f_goal=g)
            result = l2c_c1assification1.learn2clean(r_state = seed)
            if result[0]:
                iterations.append(result[1])
            else:
                iterations.append(-1) # not achieved
        elif data == 'housing':
            l2c_regression1=ql.Qlearner(dataset = dataset, goal='MARS', target_goal='SalePrice',
                                    target_prepare='SalePrice', verbose = False, f_goal=g)
            result = l2c_regression1.learn2clean(r_state = seed)
            if result[0]:
                iterations.append(result[1])
            else:
                iterations.append(-1) # not achieved

    rank_iquartiles = np.percentile(iterations, [25, 50, 75,100], interpolation='midpoint')
    write_quartiles(csv_writer, "l2c", "iterations", rank_iquartiles, g)
    csv_writer.writerow([])

f.close()

print("Goals : ", goals)
print("Iterations : ", iterations)
