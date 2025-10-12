#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille
import time
import warnings
from sklearn.model_selection import cross_val_score
# import pyearth
# from pyearth import Earth
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from scipy.stats import skew
import statsmodels.api as sm
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def LT_log_transform_skew_features(dataset):

    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index

    Y = dataset.select_dtypes(['object'])

    skewed_feats = dataset[numeric_feats].apply(
        lambda x: skew(x.dropna()))  # compute skewness

    skewed_feats = skewed_feats[skewed_feats >= 0.75]

    skewed_feats = skewed_feats.index

    dataset[skewed_feats] = np.log1p(dataset[skewed_feats])

    return dataset[skewed_feats].join(Y)


class Regressor():
    """
    Regression task
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * strategy: str, default = 'MARS'
        The choice for the regression method:
            - 'MARS, 'LASSO or 'OLS'

   * target: str, name of the target numerical variable from  dataset['target']
       pandas DataSeries

   * k_folds: int, default = 10, number of folds for cross-validation

   * verbose: Boolean,  default = 'False' otherwise display the list of
       duplicate rows that have been removed
   """

    def __init__(self, dataset, target, strategy='LASSO',
                 k_folds=10, verbose=False):
        self.Earth = None

        self.dataset = dataset

        self.target = target

        self.strategy = strategy

        self.k_folds = k_folds

        self.verbose = verbose

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'target': self.target,

                'k_folds': self.k_folds,

                'verbose': self.verbose}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for clusterer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`regressor.get_params().keys()`")

            else:

                setattr(self, k, v)

    def OLS_regression(self, dataset, target):  # quality metrics : accuracy
        # requires no missing values

        k = self.k_folds

        X_train = dataset['train'].select_dtypes(['number']).dropna()

        X_test = dataset['test'].select_dtypes(['number']).dropna()

        y_train = dataset['target'].loc[X_train.index]

        y_test = dataset['target_test']

        if (len(X_train.columns) <= 1) or (len(X_train) < k):

            print('Error: Need at least one continous variable and ',
                  k, ' observations for regression')

            mse = None

        else:

            X1Train = sm.add_constant(X_train)

            reg = sm.OLS(y_train, X1Train)

            resReg = reg.fit()

            X1Test = sm.add_constant(X_test)

            ypReg = reg.predict(resReg.params, X1Test)

            if self.verbose:

                print(resReg.summary())

            if len(y_test) == 0:

                mse = resReg.mse_total

            else:

                y_test = dataset['target_test'].loc[X_test.index]

                print("MSE of OLS with", k, " folds for cross-validation:",
                      mean_squared_error(y_test, ypReg))

                mse = mean_squared_error(y_test, ypReg)

        return mse

    def LASSO_regression(self, dataset, target):  # quality metrics : accuracy
        # requires no missing value
        
        k = self.k_folds

        X_train = dataset['train'].select_dtypes(['number']).dropna()

        X_test = dataset['test'].select_dtypes(['number']).dropna()

        y_test = dataset['target_test']

        y_train = dataset['target'].loc[X_train.index]

        if (len(X_train.columns) <= 1) or (len(X_train) < k):

            print('Error: Need at least one continous variable and ',
                  k, ' observations for regression')

            mse = None

        else:

            my_alphas = np.array(
                [0.001, 0.01, 0.02, 0.025, 0.05, 0.1,
                 0.25, 0.5, 0.8, 1.0, 1.2])

            lcv = LassoCV(alphas=my_alphas, normalize=False,
                          fit_intercept=False, random_state=0,
                          cv=k, tol=0.0001)

            lcv.fit(X_train, y_train)

            # MSE values of cross validation
            if self.verbose:

                print("MSE values of cross validation")

                print(lcv.mse_path_)
            # avg mse of cross validation for each alpha

            avg_mse = np.mean(lcv.mse_path_, axis=1)
            # alphas vs. MSE in cross-validation
            if self.verbose:

                print("alphas vs. MSE in cross-validation")

                print(pd.DataFrame({'alpha': lcv.alphas_, 'MSE': avg_mse}))

            print("Best alpha = ", lcv.alpha_)

            if len(y_test) == 0:

                mse = min(avg_mse)

                print("MSE of LASSO with", k,
                      " folds for cross-validation:", mse)

            else:

                y_test = dataset['target_test'].loc[X_test.index]

                ypLasso = lcv.predict(X_test)

                mse = mean_squared_error(y_test, ypLasso)

                print("MSE of LASSO with", k,
                      " folds for cross-validation:", mse)
        return mse

    def MARS_regression(self, dataset, target):
        # requires no missing value
        X = dataset['train'].select_dtypes(['number']).dropna()
        
        if target not in X.columns:
            raise ValueError(f"Target '{target}' not found. Available: {list(X.columns)}")
        X = X.drop(columns=[target])
        y = dataset['train'][target].loc[X.index]
        # missing_indices = y_test.index.difference(X.index)
        # y_test =y_test.drop(index=missing_indices)
        # X_test.reset_index(drop=True, inplace=True)
        # y_test.reset_index(drop=True, inplace=True)
        # dataset['test'] = X_test;
        # dataset['target'] = y_test
        # dataset['SalePrice'] = y_test

        # import pdb;pdb.set_trace()
        # X_train = LT_log_transform_skew_features(X_train)
        cv_mars = None
        # if (len(X_test.columns) <= 1) or (len(X_test) < k):

        #     print('Error: Need at least one continous variable and ',
        #           k, ' observations for regression')

        #     cv_mars = None

        #     # y_train = X_train[target]

        #     # X_train = X_train.drop([target], 1)

        # else:
        if True:

            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()

            
            y_pred = model.fit(X, y).predict(X)

            # def rmse_cv(model):

            #     rmse = np.sqrt(-cross_val_score(model, X_train,
            #                                     np.log1p(y_train),
            #                                     scoring="neg_mean_"
            #                                     "squared_error", cv=k))

            #     return(rmse)

            # cv_mars = rmse_cv(model).mean()

            # if self.verbose:

            #     print(model.summary())

            # print("MSE of MARS with", k, "folds "
            #       "for cross-validation:", cv_mars)
            # import pdb;pdb.set_trace()
            cv_mars = np.sqrt(root_mean_squared_error(y, y_pred)) 
        
        print(f'MSE {cv_mars}')
        return cv_mars
        

    def transform(self):

        start_time = time.time()

        d = self.dataset

        if self.target == d['target'].name:

            print()

            print(">>Regression task")

            if (self.strategy == "OLS"):

                dn = self.OLS_regression(d, self.target)

            elif (self.strategy == "LASSO"):

                dn = self.LASSO_regression(d, self.target)

            elif (self.strategy == "MARS"):
                
                dn = self.MARS_regression(d, self.target)
                print(dn)

            else:

                raise ValueError(
                    "The regression function should be OLS, LASSO, or MARS")

            print("Regression done -- CPU time: %s seconds" %

                  (time.time() - start_time))

        else:

            raise ValueError("Target variable invalid.")

        return {'quality_metric': dn}
