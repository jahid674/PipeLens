import itertools
from modules.outlier_detection.outlier_detector import OutlierDetector
from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
from modules.normalization.normalizer import DataNormalizer
from modules.metric.metric import metric
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from regression import Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import csv
import os
import pandas as pd
import numpy as np
from itertools import product
import random
from itertools import cycle
import sys
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from modules.outlier_detection.outlier_detector import OutlierDetector
from modules.missing_value.imputer import DataImputer
from modules.Util.reader import Reader
from modules.normalization.normalizer import DataNormalizer
from sklearn.naive_bayes import GaussianNB
from regression import Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import KNNImputer, SimpleImputer
from random import sample 
import time
import os
import pandas as pd
import numpy as np
from itertools import product
import random
import copy
import csv 
from sklearn.metrics import f1_score,accuracy_score
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder

tau = 0.1 # fraction of missing values
# knn_k = 1 # knn number of neighbors
# lof_k = 50 # number of neighbors for local outlier factor
lof_contamination = 'auto' 
knn_k_lst = [1, 5, 10, 20, 30]
lof_k_lst = [1, 5, 10, 20, 30]
len_knn = len(knn_k_lst)
len_lof = len(lof_k_lst)
norm_strategy = ['none', 'ss', 'rs', 'ma', 'mm'] # standard scaler, robust scaler, max absolute scaler, minmax scaler
mv_strategy = ['drop', 'mean', 'median', 'most_frequent', 'knn']
# od_strategy = ['zs', 'iqr', 'if', 'lof'] # local outlier factor, z-score, interquartile range, isolation forest
od_strategy = ['none', 'if', 'lof'] # local outlier factor, isolation forest
feature_strategy = ['l1', 'imp'],
dataset = 'housing'
modelType = 'lr' #'lr' # 'nb' Logistic Regression or Gaussian Naive Bayes
metric_type = 'rmse'


column_names = ['Age','Workclass','fnlwgt','Education','Education_Num','Martial_Status','Occupation','Relationship','Race','Sex','Capital_Gain','Capital_Loss','Hours_per_week','Country','income']
categorical_cols = ['Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
if dataset == 'hdma':
    hdma_train  = "data/hmda/hmda_Orleans_X_train_1.csv"
    hdma_test = "data/hmda/hmda_Calcasieu_X_test_1.csv"
    # hdma_train = "data/hmda/hmda_Calcasieu_X_test_1.csv"
    train,test = Reader(hdma_train,hdma_test).load_data()
    y_train = train['action_taken']
    X_train = train.drop('action_taken', axis=1)

    y_test = test['action_taken']
    X_test = test.drop('action_taken', axis=1)
elif dataset == 'adult':
    adult_train  = "data/adult/adult_train.csv"
    adult_test = "data/adult/adult_test.csv"
    train,test = Reader(adult_train,adult_test).load_data()
    categorical_columns = train.select_dtypes(include=['object']).columns
    label_encoders = {}
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        le = LabelEncoder()
        train[column] = label_encoder.fit_transform(train[column]) 
        test[column] = label_encoder.fit_transform(test[column]) 
        label_encoders[column] = le

    
    y_train = train['income']
    X_train = train.drop('income', axis=1)

    y_test = test['income']
    X_test = test.drop('income', axis=1)
    unique_labels = np.unique(y_test)
    print("Unique labels in y_test:", unique_labels)
elif dataset == 'housing':
    house_train  = "data/house/house_train.csv"
    house_test = "data/house/test.csv"
    train,test = Reader(house_train,house_test).load_data()
#     import pdb;pdb.set_trace()
    y = train['SalePrice']
    categorical_columns = ['SaleType',
    'MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','TotRmsAbvGrd','Functional','FireplaceQu','GarageType','GarageFinish',
    'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleCondition']
    X = train.drop('SalePrice', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#     y_train = train['SalePrice']
    label_encoder = LabelEncoder()
    label_encoders = {}
    
    for column in categorical_columns:
        le = LabelEncoder()
        X_train[column] = label_encoder.fit_transform(X_train[column]) 
        X_test[column] = label_encoder.fit_transform(X_test[column]) 
        label_encoders[column] = le

#     y_test = test['SalePrice']
#     X_test = test.drop('SalePrice', axis=1)

if(metric_type=='sp'):
        if modelType == 'lr':
                model = LogisticRegression(random_state=0).fit(X_train, y_train)
        elif modelType == 'nb':
                model = GaussianNB().fit(X_train, y_train)
        print("Training accuracy : " + str(round(model.score(X_train, y_train), 4)))
        print("Test accuracy : " + str(round(model.score(X_test, y_test), 4)))

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        p_train = model.predict_proba(X_train)
        p_test = model.predict_proba(X_test)

class base:
        def __init__(self):
                self.f = 'tdst';
                self.ranking = None
                self.ranges = {}
                self.imputer_strategies = ['drop', 'mean', 'median', 'most_frequent', 'knn']
                self.mv_name_mapping = {'drop': 'mv_drop', 'mean': 'mv_mean', 'median': 'mv_median', 'most_frequent': 'mv_mode', 'knn': 'mv_knn'}
                self.fail = 0
                self.pass_ = 0
                #KNN 
                self.outlier_strategies = ['none', 'if', 'lof']
                self.ot_name_mapping = {'none': 'od_none', 'if': 'od_if', 'lof': 'od_lof'}

                #1.15 paramater 
                self.normalizer_strategies = ['none', 'ss', 'rs', 'ma', 'mm']
                self.no_name_mapping = {'none': 'norm_none', 'ss': 'norm_ss', 'rs': 'norm_rs', 'ma': 'norm_ma', 'mm': 'norm_mm'}

                #normalization 
                #

                self.ranges[0] = [1]
                self.ranges[1] = [1, 5, 10, 20, 30]
                self.ranges[2] = [1, 5, 10, 20, 30]

                self.base_strategies = ["mv_drop","mv_mean","mv_median","mv_mode","mv_knn",
                                                                "norm_none","norm_ss","norm_rs","norm_ma","norm_mm",
                                                                "od_none", "od_if","od_lof"]



                self.historical_data = []   
                self.historical_data_pd = []
                self.gs_idistr = []
                self.gs_fdistr = []
                self.k = {}
                self.k[0] = 2
                self.k[1] = 8
                self.k[2] = 4
                self.k[3] = 8
                self.k[4] = 6
                self.k[5] = 8
                self.k[6] = 8
                self.k[7] = 8
                self.sublist_1  = []
                self.sublist_2 = []
                self.sublist_3 = [] 
                self.column_name = ['normal','imputer','outlier_strategy' ]
        def getIdxSensitive(self, df, dataset):
                if dataset == 'hdma':
                        priv_idx = df.index[df['race']==1]
                        unpriv_idx = df.index[df['race']==0]
                        sensitive_attr = df['race']
                elif dataset == 'adult':
                        priv_idx = df.index[df['Sex']==1]
                        unpriv_idx = df.index[df['Sex']==0]
                        sensitive_attr = df['Sex']
                return priv_idx, unpriv_idx, sensitive_attr
        def computeStatisticalParity(self, p_priv, p_unpriv):
                # p_priv = pd.DataFrame(p_priv)[1]
                # p_unpriv = pd.DataFrame(p_unpriv)[1]
                diff = p_priv.mean() - p_unpriv.mean()
                return diff
        def computeEqualizedOdds(self, y_pred, y_true, priv_idx, unpriv_idx):
                # import pdb;pdb.set_trace()
                try:
                        cm = confusion_matrix(y_true[priv_idx], y_pred[priv_idx])
                        # print(cm)
                        tn, fp, fn, tp = cm.ravel()
                        tpr_priv = tp / (tp + fn)
                        fpr_priv = tn / (tn + fp)

                        cm = confusion_matrix(y_true[unpriv_idx], y_pred[unpriv_idx])
                        #a print(cm)
                        tn, fp, fn, tp = cm.ravel()
                        tpr_unpriv = tp / (tp + fn)
                        fpr_unpriv = tn / (tn + fp)

                        tpr_diff = tpr_priv - tpr_unpriv
                        fpr_diff = fpr_priv - fpr_unpriv
                except:
                        import pdb;pdb.set_trace()
                # fpr_diff = 0 # equality of opportunity, comparing tprs only

                return max(tpr_diff, fpr_diff)
        
        def inject_null_values(self):
                ## Injected missing values with 5% of data 
                np.random.seed(42)

                for col in ['lien_status']:
                        mask_train = np.random.rand(len(self.dataset['train'])) < 0.05
                        mask_test = np.random.rand(len(self.dataset['test'])) < 0.05
                
                
                self.dataset['train'].loc[mask_train, col] = np.nan
                
                self.dataset['test'].loc[mask_test, col] = np.nan

                #Imputed summary 
                print(self.dataset['train'].isnull().sum())
                print(self.dataset['test'].isnull().sum())
                ##end 

                
        def load_data(self):
                hdma_train  = "data/hmda/hmda_Orleans_X_train_1.csv"
                hdma_test = "data/hmda/hmda_Calcasieu_X_test_1.csv"
                train,test = Reader(hdma_train,hdma_test).load_data()

                self. dataset = {
                        'train': train,
                        'test': test,
                }

        def generate_combination_of_all_transformer(self):
                self.all_binary_vectors = []
                final_strategies = self.normalizer_strategies + self.imputer_strategies + self.outlier_strategies
                for nor in self.normalizer_strategies:
                        for impt in self.imputer_strategies:
                                for neighbors in self.outlier_strategies:
                                        # Select one strategy for each category
                                        selected_imputer_strategy = impt
                                        selected_outlier_strategy = neighbors
                                        selected_normalizer_strategy = nor 

                                        # Create binary vectors for each category
                                        imputer_vector = [1 if strategy == selected_imputer_strategy else 0 for strategy in self.imputer_strategies]
                                        outlier_vector = [1 if strategy == selected_outlier_strategy else 0 for strategy in self.outlier_strategies]
                                        normalizer_vector = [1 if strategy == selected_normalizer_strategy else 0 for strategy in self.normalizer_strategies]
                                        binary_vector = normalizer_vector + outlier_vector + imputer_vector  
                                        self.all_binary_vectors.append(binary_vector)

       
                # Print all generated binary vectors
                for binary_vector in self.all_binary_vectors:
                        print(binary_vector)


 
        def optimize(self, init_params, f_goal):
                #print(f'St#arting Grasp search for fairness score:{f_goal}')
                #======Intitalize variable start here =====
                # test = [0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 30.0, 0.2296946068875893]
                test = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.06203095684803]
                self.rank_iter = 0  #Iteration count by ranking algorithm
                self.rank_f = 0    # Iteration count whn we found the fairness score less than the calculated fairness from seed value 
                iter_size = 0   # Total iteration allowed in ranking algorithm , falllback is after 1 to grid search 
                cur_params = init_params.copy()
                #========end=========
                # self.fail = 0
                # self.pass_ = 0
                
                #  From seed value calculate the new fairness score
                #=========Code Start========
                
                cur_params_init = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params[:len(self.base_strategies)])}
                positive_index = [index for index, value in enumerate(cur_params_init.values()) if value > 0]
                i_imputer = positive_index[0]
                i_normal  = positive_index[1]-5
                i_outlier = positive_index[2]-10
                # import pdb;pdb.set_trace()
                opt_f  = self.f_score_look_up2(self.historical_data_pd,init_params)
                #print(f'Fairness score from seed value : {opt_f}')
                #======Code end ============
                # print(init_params)
                # if(init_params==test):
                #         import pdb;pdb.set_trace()
                seen = set()
                # import pdb;pdb.set_trace()
                seen.add(tuple(cur_params))
                # while ( iter_size < 9 ):
                #         iter_size += 1      
                #         index = 0
                iter_size  = 1
                if iter_size == 1:
                        goal_pipline = False
                        
                        
                        #while(not goal_pipline):
                        #print('Entered loop')
                        inc_imp  = 1
                        inc_nor = 1
                        inc_out = 1

                        
                        nor_type = self.coef_normalization_rank.index(i_normal)
                        imputer_type = self.coef_imputer_rank.index(i_imputer)
                        out_type = self.coef_outlier_rank.index(i_outlier)
                        

                        len_nor = len(self.coef_normalization_rank)
                        len_out = len(self.coef_outlier_rank)
                        len_imp = len(self.coef_imputer_rank)


                        if(imputer_type!=0):
                                inc_imp = -1
                                len_imp = 0
                        if(out_type!=0):
                                inc_out = -1
                                len_out = 0
                                
                        if(nor_type!=0):
                                inc_nor = -1
                                len_nor = 0
                        # import pdb;pdb.set_trace()
                        # self.historical_data_pd = pd.read_csv(filename_test);
                        for norm_it in range( nor_type , len_nor, inc_nor):
                                ds_val  = [1]
                                # cur_params_init = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params[:len(self.base_strategies)])}
                                for imput_itr in range(imputer_type,len_imp,inc_imp):
                                        knn_val = [1]
                                        
                                        if(cur_params_init['mv_knn']!=0):
                                                knn_val =  self.ranges[1]
                                                try:
                                                        
                                                        
                                                        value_out  = cur_params_init['mv_knn']

                                                
                                                        indx = knn_val.index(value_out)
                                                        if(self.coef_imputer[self.coef_imputer_rank[imput_itr]]>0):
                                                                knn_val =  knn_val[indx:len(knn_val)]
                                                        else:
                                                                knn_val =  knn_val[0:indx]
                                                                knn_val = knn_val[::-1]
                                                except Exception as e :
                                                        print(e)
                                                        import pdb;pdb.set_trace()
                                                
                                                
                                        
                                        for outlier_itr in range(out_type,len_out,inc_out):
                                                lof_val = [1]
                                                try:
                                                        if cur_params_init['od_lof']!=0:
                                                                lof_val = self.ranges[2]

                                                                value_out  = cur_params_init['od_lof']
                                                                indx = lof_val.index(value_out)
                                                                
                                                                if(self.coef_outlier[self.coef_outlier_rank[outlier_itr]]>0):
                                                                        lof_val =  lof_val[indx:len(lof_val)]
                                                                else:
                                                                        lof_val =  lof_val[0:indx]
                                                                        lof_val = lof_val[::-1]
                                                except Exception as e :
                                                        print(e)
                                                        import pdb;pdb.set_trace()

                                                # print(self.normalizer_strategies[self.coef_normalization_rank[norm_it]])
                                                # print(self.imputer_strategies[self.coef_imputer_rank[imput_itr]])
                                                # print(self.outlier_strategies[self.coef_outlier[outlier_itr]])
                                                # import pdb;pdb.set_trace()
                                                # try:
                                                #         if not self.coef_outlier[outlier_itr]>0:

                                                #                 lof_val = lof_val[::-1]
                                                                
                                                        
                                                #         if not self.coef_imputer[outlier_itr]>0:
                                                #                 knn_val = knn_val[::-1]
                                                # except Exception as e :
                                                #         print(e)
                                                #         import pdb;pdb.set_trace()
                                                        

                                
                                                for p_normal in ds_val:
                                                        for p_imputer in knn_val:
                                                                for p_outlier in lof_val:
                                                                        try:
                                                                                cur_params_init = {'mv_drop': 0, 'mv_mean': 0, 'mv_median': 0, 'mv_mode': 0, 'mv_knn': 0, 'norm_none': 0, 'norm_ss': 0, 'norm_rs': 0, 'norm_ma': 0, 'norm_mm': 0, 'od_none': 0, 'od_if': 0, 'od_lof': 0}
                                                                                cur_params = {
                                                                                        self.no_name_mapping[self.normalizer_strategies[self.coef_normalization_rank[norm_it]]]:p_normal,
                                                                                        self.mv_name_mapping[self.imputer_strategies[self.coef_imputer_rank[imput_itr]]]:p_imputer,
                                                                                        self.ot_name_mapping[self.outlier_strategies[self.coef_outlier_rank[outlier_itr]]]:p_outlier
                                                                                }
                                                                                
                                                                                if   tuple(cur_params.items()) in seen:
                                                                                        continue
                                                                                else:
                                                                                        
                                                                                        
                                                                                        # cur_params_init = {}
                                                                                        for key, value in cur_params.items():
                                                                                                cur_params_init[key] = value
                                                        
                                                                                        # import pdb;pdb.set_trace()
                                                                                        # cur_f = self.get_fairness_score_from_test_data(cur_params,self.imputer_strategies[i_imputer],self.outlier_strategies[i_outlier]
                                                                                        #                                 ,self.normalizer_strategies[i_normal])
                                                                                        
                                                                                        
                                                                                        if(tuple(cur_params_init.items())) in seen:
                                                                                                continue

                                                                                        seen.add(tuple(cur_params_init.items()))
                                                                                        # if(init_params==test):
                                                                                        #         print(cur_params_init)
                                                                                        cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params_init.values()))
                                                                                        self.rank_iter += 1
                                                                                        # print(f'Fairness score found for tupple {tuple(cur_params.items())}: {cur_f}')
                                                                                       
                                                                                        if metric_type=='sp' or metric_type=='mae' or metric_type=='rmse':                                                                           
                                                                                                # if(cur_f==0):
                                                                                                #         import pdb;pdb.set_trace()
                                                                                                if cur_f <= f_goal:
                                                                                                        # cur_params = test_params.copy()
                                                                                                # print(f'Found Fairness score found for tupple {tuple(cur_params.items())}: {cur_f}')
                                                                                                        self.rank_f = cur_f
                                                                                                        f_goal_found = True
                                                                                                        #print('Found')
                                                                                                        self.pass_ = self.pass_  + 1
                                                                                                        return cur_params.values() # early exit when f_goal obtained
                                                                        
                                                                                                elif cur_f < opt_f:
                                                                                                        # cur_params = cur_params.values()
                                                                                                        opt_f = cur_f
                                                                                        elif metric_type=='f-1':
                                                                                                
                                                                                                if(cur_f==0):
                                                                                                        import pdb;pdb.set_trace()
                                                                                                if cur_f >= f_goal:
                                                                                                        # cur_params = test_params.copy()
                                                                                                # print(f'Found Fairness score found for tupple {tuple(cur_params.items())}: {cur_f}')
                                                                                                        self.rank_f = cur_f
                                                                                                        f_goal_found = True
                                                                                                        #print('Found')
                                                                                                        self.pass_ = self.pass_  + 1
                                                                                                        return cur_params.values() # early exit when f_goal obtained
                                                                        
                                                                                                elif cur_f > opt_f:
                                                                                                        # cur_params = cur_params.values()
                                                                                                        opt_f = cur_f

                                                                        except Exception as e : 
                                                                                print(e)
                                                                                import pdb;pdb.set_trace()
                                                                                print('help')
                                        
                                        for val in self.ot_name_mapping.values() :
                                                cur_params_init[val] = 0
                                        for val in self.mv_name_mapping.values() :
                                                        cur_params_init[val] = 0
                                for val in self.no_name_mapping.values() :
                                                        cur_params_init[val] = 0

                        iter_size = iter_size + 1


                if(iter_size>1):
                        #print("Grasp algo not found ,Starting grid search")
                        # import pdb;pdb.set_trace()
                        # print('yes')
                        self.fail = self.fail  + 1
                        # print(init_params)
                        self.rank_iter,self.rank_f   = self.grid_search(f_goal,1)
                return cur_params


        def f_score_look_up2(self,profiles_df,elem):
                column_names =self.base_strategies
                # return elem[-1]
                # import pdb;pdb.set_trace()
                try:
                        return profiles_df.loc[(profiles_df[column_names[0]] == elem[0]) & (profiles_df[column_names[1]] == elem[1] ) 
                                               & (profiles_df[column_names[2]] == elem[2]) &
                                               (profiles_df[column_names[3]] == elem[3]) & (profiles_df[column_names[4]] == elem[4] ) 
                                               & (profiles_df[column_names[5]] == elem[5]) &
                                                (profiles_df[column_names[6]] == elem[6]) & (profiles_df[column_names[7]] == elem[7] ) 
                                               & (profiles_df[column_names[8]] == elem[8]) 
                                               & (profiles_df[column_names[9]] == elem[9]) 
                                               & (profiles_df[column_names[10]] == elem[10]) 
                                               & (profiles_df[column_names[11]] == elem[11]) 
                                               & (profiles_df[column_names[12]] == elem[12]) 
                                               
                                               ].iloc[0]['fairness']
                except Exception as e :
                        print(e)
                        import pdb;pdb.set_trace()


        def grid_search(self, f_goal, iterations):
                self.gs_idistr = []
                self.gs_fdistr = []
                gs_iter = 0     
                gs_f = 0
                iter_lst = []
                #default iteration is 100 
                # import pdb;pdb.set_trace()
                for i in range(iterations):
                        gs_iter = 0
                        gs_f = 0
                        cur_order = self.historical_data
                        
                        #Added randomness for grid search 
                        random.shuffle(cur_order)
                        
                        for elem in cur_order:
                                # dict_strategy = {strategy: selection for strategy, selection in zip(self.base_strategies, elem[:len(self.base_strategies)])}
                                
                                # positive_positions = [index for index, value in enumerate(dict_strategy.values()) if value > 0]
                                
                                # cur_f = self.get_fairness_score_from_test_data(dict_strategy,normalizer_strat = self.base_strategies[positive_positions[0]],imputer_strat = self.base_strategies[positive_positions[1]],outlier_strat = self.base_strategies[positive_positions[2]])
                                # import pdb;pdb.set_trace()
                                cur_f = self.f_score_look_up2(self.historical_data_pd,elem)
                                # ielmport pdb;pdb.set_trace()
                                if(cur_f==0):
                                        print('Found zero fairness value ')
        
                                gs_iter += 1
                                if metric_type=='sp' or metric_type=='mae' or metric_type=='rmse':
                                        if cur_f <= f_goal:
                                                gs_f = cur_f
                                                cur_params = []
                                                cur_params.append(elem[0]) 
                                                cur_params.append(elem[1])
                                                cur_params.append(elem[2])
                                                self.gs_fdistr.append(gs_f)
                                                self.gs_idistr.append(gs_iter)
                                                return gs_iter,gs_f
                                elif metric_type=='f-1':
                                        # print('i am in f1 score')
                                        if cur_f >= f_goal:
                                                gs_f = cur_f
                                                cur_params = []
                                                cur_params.append(elem[0]) 
                                                cur_params.append(elem[1])
                                                cur_params.append(elem[2])
                                                self.gs_fdistr.append(gs_f)
                                                self.gs_idistr.append(gs_iter)
                                                return gs_iter,gs_f


        def create_historic_data(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
                if not(os.path.exists(file_name)):
                        idx_train = np.arange(0, len(X_train), 1)
                        mv_train = pd.DataFrame(idx_train).sample(frac=tau, replace=False, random_state=1).index
                        if (dataset == 'hdma'):
                                X_train['lien_status'][mv_train] = np.NaN
                        elif (dataset == 'adult'):
                                # 
                                X_train['Martial_Status'][mv_train] = np.NaN

                        # idx_test = np.arange(0, len(X_test), 1)
                        # mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
                        # X_test['lien_status'][mv_test] = np.NaN
                        # # X_test['income_brackets'][mv_test] = np.NaN

                        params_metrics = []
                        print("Running pipeline combinations ...")
                        
                        
                        
                        mv_params = len(mv_strategy)
                        od_params = len(od_strategy)
                        norm_params = len(norm_strategy)
                        if 'knn' in mv_strategy:
                                mv_params += len(knn_k_lst) - 1
                        if 'lof' in od_strategy:
                                od_params += len(lof_k_lst) - 1

                        mv_param = ''
                        norm_param = ''
                        od_param = ''

                        param_lst = []
                        if(metric_type=='sp'):
                                priv_idx_train, unpriv_idx_train, sensitive_attr_train = self.getIdxSensitive(X_train, dataset)
                                train_eqOdds = self.computeEqualizedOdds(y_pred_train, y_train, priv_idx_train, unpriv_idx_train)
                                print("Training fairness : " + str(round(train_eqOdds, 4)))

                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)
                                test_eqOdds = self.computeEqualizedOdds(y_pred_test, y_test, priv_idx_test, unpriv_idx_test)
                                print("Test fairness : " + str(round(test_eqOdds, 4)))

                        for param1 in range(mv_params):
                                for param2 in range(norm_params):
                                        for param3 in range(od_params):
                                                # import pdb;pdb.set_trace()
                                                if param1 < len(mv_strategy) - 1:
                                                        
                                                        if mv_strategy[param1] == 'drop':
                                        # print("Dropping missing values ...")
                                                                mv_idx = []
                                                                if dataset == 'hdma':
                                                                        mv_idx = X_train[X_train['lien_status'].isna()].index.tolist()
                                                                elif dataset == 'adult':
                                                                        mv_idx = X_train[X_train['Martial_Status'].isna()].index.tolist()
                                                                else:
                                                                        mv_idx = X_train[X_train.isna().any(axis=1)].index.tolist()
                                                                        
                                                                # import pdb;pdb.set_trace()
                                                                
                                                                
                                                                imputed_X_train = X_train.drop(mv_idx)
                                                                imputed_X_train.reset_index(drop=True, inplace=True)
                                                                updated_y_train = y_train.copy()
                                                                for idx in sorted(mv_idx, reverse=True):
                                                                        del updated_y_train[idx]
                                                                updated_y_train.reset_index(drop=True, inplace=True)
                                                                if metric_type=='sp':
                                                                        
                                                                        updated_sensitive_attr_train = sensitive_attr_train.drop(mv_idx)
                                                                        updated_sensitive_attr_train.reset_index(drop=True, inplace=True)
                                                                mv_param = [1, 0, 0, 0, 0]
                                                        elif mv_strategy[param1] in ['mean', 'median', 'most_frequent']:
                                                                imputed_X_train = SimpleImputer(missing_values=np.nan, strategy=mv_strategy[param1]).fit(X_train).transform(X_train)
                                                                updated_y_train = y_train.copy()
                                                                if metric_type=='sp':
                                                                        updated_sensitive_attr_train = sensitive_attr_train.copy()
                                                                if mv_strategy[param1] == 'mean':
                                                                        mv_param = [0, 1, 0, 0, 0]
                                                                if mv_strategy[param1] == 'median':
                                                                        mv_param = [0, 0, 1, 0, 0]
                                                                if mv_strategy[param1] == 'most_frequent':
                                                                        mv_param = [0, 0, 0, 1, 0]
                                                else:
                                                        k = knn_k_lst[param1-4] # start accessing number of neighbors in knn
                                                        imputed_X_train = KNNImputer(n_neighbors=k).fit_transform(X_train)
                                                        mv_param = [0, 0, 0, 0, k]
                                
                                
                                                if norm_strategy[param2] == 'none':
                                                        scaled_X_train = imputed_X_train.copy()
                                                        norm_param = [1, 0, 0, 0, 0]
                                                elif norm_strategy[param2] == 'ss':
                                                        scaled_X_train = StandardScaler().fit(imputed_X_train).transform(imputed_X_train) 
                                                        norm_param = [0, 1, 0, 0, 0]
                                                elif norm_strategy[param2] == 'rs':
                                                        scaled_X_train = RobustScaler().fit(imputed_X_train).transform(imputed_X_train) 
                                                        norm_param = [0, 0, 1, 0, 0]
                                                elif norm_strategy[param2] == 'ma':
                                                        scaled_X_train = MaxAbsScaler().fit(imputed_X_train).transform(imputed_X_train) 
                                                        norm_param = [0, 0, 0, 1, 0]
                                                elif norm_strategy[param2] == 'mm':
                                                        scaled_X_train = MinMaxScaler().fit(imputed_X_train).transform(imputed_X_train) 
                                                        norm_param = [0, 0, 0, 0, 1]
                                
                                                if param3 < len(od_strategy) - 2:
                                                        if od_strategy[param3] == 'none':
                                                                outlier_y_pred = np.ones(len(scaled_X_train))
                                                                od_param = [1, 0, 0]
                                                        if od_strategy[param3] == 'if':
                                                                outlier_y_pred = IsolationForest(max_samples=100, random_state=0).fit_predict(scaled_X_train)
                                                                od_param = [0, 1, 0]
                                                else:
                                                        k = lof_k_lst[param3 - 2] # start accessing number of neighbors in lof
                                                        outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=lof_contamination).fit_predict(scaled_X_train)
                                                        od_param = [0, 0, k]
                                                mask = outlier_y_pred != -1
                                # print("Fraction of outliers: ", round((1 - sum(mask)/len(outlier_y_pred)) * 100, 4))

                                # import pdb; pdb.set_trace();
                                                outlier_X_train = scaled_X_train.copy()
                                                outlier_y_train = updated_y_train.copy()
                                                if metric_type=='sp':
                                                        outlier_sensitive_train = updated_sensitive_attr_train.copy()
                                                        if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                outlier_X_train, outlier_y_train, outlier_sensitive_train = scaled_X_train[mask], updated_y_train[mask], updated_sensitive_attr_train[mask]
                                                                outlier_y_train.reset_index(drop=True, inplace=True)
                                                                outlier_sensitive_train.reset_index(drop=True, inplace=True)          
                                                        priv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 1]
                                                        unpriv_idx_train = [i for i, val in enumerate(outlier_sensitive_train) if val == 0]
                                                
                                                updated_model = None
                                                # import pdb;pdb.set_trace()
                                                if modelType == 'lr':
                                                        updated_model = LogisticRegression(random_state=0).fit(outlier_X_train, outlier_y_train)
                                                elif modelType == 'nb':
                                                        updated_model = GaussianNB().fit(outlier_X_train, outlier_y_train)

                                                y_pred = updated_model.predict(outlier_X_train)
                                                # import pdb;pdb.set_trace()
                                                outc = None
                                                if metric_type== 'sp':
                                                        outc = self.computeStatisticalParity(y_pred[priv_idx_test],y_pred[unpriv_idx_test])
                                                elif metric_type=='f-1':
                                                        outc = f1_score(outlier_y_train,y_pred)
                                                elif metric_type=='mae':
                                                        outc = mean_absolute_error(outlier_y_train, y_pred)
                                                elif metric_type=='rmse':
                                                        outc = np.sqrt(mean_squared_error(outlier_y_train, y_pred)) 

                                                       
                                                f = [outc]
                                                param_lst.append(mv_param + norm_param + od_param + f)
                                                # if eqOdds_train == 0.0:
                                                print(str(mv_param + norm_param + od_param + f))
                                        # print(len(unpriv_idx_train))

                        param_lst_df = pd.DataFrame(param_lst, columns=["mv_drop","mv_mean","mv_median","mv_mode","mv_knn",
                                                                        "norm_none","norm_ss","norm_rs","norm_ma","norm_mm",
                                                                        "od_none", "od_if","od_lof","fairness"])
                        # import pdb; pdb.set_trace()
                        param_lst_df.to_csv(file_name, index=False)
                        # import pdb;pdb.set_trace()
                else :
                        param_lst_df = pd.read_csv(file_name)
                  
                y = param_lst_df['fairness']
                X = param_lst_df.copy().drop("fairness", axis=1)
                reg = Regression()
                model = reg.generate_regression(X, y)
                coefs = model.coef_
                print(coefs)
                print(model.intercept_)
                self.coefs = coefs
                sublist_size = len(coefs) // 3
                self.coef_normalization = coefs[5:10]
                self.coef_imputer = coefs[0:5]
                self.coef_outlier = coefs[10:13]
                self.coef_normalization_rank = np.argsort(np.abs( coefs[5:10])).tolist()
                self.coef_imputer_rank = np.argsort(np.abs(coefs[0:5])).tolist()
                self.coef_outlier_rank = np.argsort(np.abs(coefs[10:13])).tolist()
                # Calculate ranks
                # original_list = [self.coef_normalization[self.coef_normalization_rank[0]],self.coef_imputer[self.coef_imputer_rank],self.coef_outlier[self.coef_outlier_rank]]
                # ranks = rankdata(original_list)

                # # Convert ranks to integers
                # self.comp_rank = ranks.astype(int)
        def create_historic_data_test(self,file_name):
                # inject missing values in the most important column
                param_lst_df = None
                if not(os.path.exists(file_name)):
                        idx_test = np.arange(0, len(X_test), 1)
                        mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
                        if (dataset == 'hdma'):
                                X_test['lien_status'][mv_test] = np.NaN
                        elif (dataset == 'adult'):
                                X_test['Martial_Status'][mv_test] = np.NaN

                        # idx_test = np.arange(0, len(X_test), 1)
                        # mv_test = pd.DataFrame(idx_test).sample(frac=tau, replace=False, random_state=1).index
                        # X_test['lien_status'][mv_test] = np.NaN
                        # # X_test['income_brackets'][mv_test] = np.NaN

                        params_metrics = []
                        print("Running pipeline combinations ...")
                        
                        mv_params = len(mv_strategy)
                        od_params = len(od_strategy)
                        norm_params = len(norm_strategy)
                        if 'knn' in mv_strategy:
                                mv_params += len(knn_k_lst) - 1
                        if 'lof' in od_strategy:
                                od_params += len(lof_k_lst) - 1

                        mv_param = ''
                        norm_param = ''
                        od_param = ''

                        param_lst = []
                        if metric_type == 'sp':
                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)
                                # test_eqOdds = self.computeEqualizedOdds(y_pred_test, y_test, priv_idx_test, unpriv_idx_test)
                                # print("testing fairness : " + str(round(test_eqOdds, 4)))

                                priv_idx_test, unpriv_idx_test, sensitive_attr_test = self.getIdxSensitive(X_test, dataset)
                                # test_eqOdds = self.computeEqualizedOdds(y_pred_test, y_test, priv_idx_test, unpriv_idx_test)
                                # print("Test fairness : " + str(round(test_eqOdds, 4)))
                        for feature in range(len(feature_strategy)):
                                for param1 in range(mv_params):
                                        for param2 in range(norm_params):
                                                for param3 in range(od_params):
                                                        
                                                        if param1 < len(mv_strategy) - 1:
                                                                if mv_strategy[param1] == 'drop':
                                                # print("Dropping missing values ...")
                                                                        if dataset == 'hdma':
                                                                                mv_idx = X_test[X_test['lien_status'].isna()].index.tolist()
                                                                        elif dataset == 'adult':
                                                                                mv_idx = X_test[X_test['Martial_Status'].isna()].index.tolist()
                                                                        elif dataset == 'housing':
                                                                                mv_idx = X_test[X_test.isna().any(axis=1)].index.tolist()
                                                                        imputed_X_test = X_test.drop(mv_idx)
                                                                        imputed_X_test.reset_index(drop=True, inplace=True)
                                                                        updated_y_test = y_test.copy()
                                                                        for idx in sorted(mv_idx, reverse=True):
                                                                                del updated_y_test[idx]
                                                                        updated_y_test.reset_index(drop=True, inplace=True)
                                                                        if metric_type=='sp':
                                                                                updated_sensitive_attr_test = sensitive_attr_test.drop(mv_idx)
                                                                                updated_sensitive_attr_test.reset_index(drop=True, inplace=True)
                                                                        mv_param = [1, 0, 0, 0, 0]
                                                                elif mv_strategy[param1] in ['mean', 'median', 'most_frequent']:
                                                                        imputed_X_test = SimpleImputer(missing_values=np.nan, strategy=mv_strategy[param1]).fit(X_test).transform(X_test)
                                                                        updated_y_test = y_test.copy()
                                                                        if metric_type=='sp':
                                                                                updated_sensitive_attr_test = sensitive_attr_test.copy()
                                                                        if mv_strategy[param1] == 'mean':
                                                                                mv_param = [0, 1, 0, 0, 0]
                                                                        if mv_strategy[param1] == 'median':
                                                                                mv_param = [0, 0, 1, 0, 0]
                                                                        if mv_strategy[param1] == 'most_frequent':
                                                                                mv_param = [0, 0, 0, 1, 0]
                                                                                # import pdb;pdb.set_trace()
                                                        else:
                                                                k = knn_k_lst[param1-4] # start accessing number of neighbors in knn
                                                                imputed_X_test = KNNImputer(n_neighbors=k).fit_transform(X_test)
                                                                mv_param = [0, 0, 0, 0, k]
                                        
                                                        # import pdb;pdb.set_trace()
                                                        if norm_strategy[param2] == 'none':
                                                                scaled_X_test = imputed_X_test.copy()
                                                                norm_param = [1, 0, 0, 0, 0]
                                                        elif norm_strategy[param2] == 'ss':
                                                                # column  = imputed_X_test.columns
                                                                scaled_X_test = StandardScaler().fit(imputed_X_test).transform(imputed_X_test) 
                                                                norm_param = [0, 1, 0, 0, 0]
                                                        elif norm_strategy[param2] == 'rs':
                                                                scaled_X_test = RobustScaler().fit(imputed_X_test).transform(imputed_X_test) 
                                                                norm_param = [0, 0, 1, 0, 0]
                                                        elif norm_strategy[param2] == 'ma':
                                                                scaled_X_test = MaxAbsScaler().fit(imputed_X_test).transform(imputed_X_test) 
                                                                norm_param = [0, 0, 0, 1, 0]
                                                        elif norm_strategy[param2] == 'mm':
                                                                scaled_X_test = MinMaxScaler().fit(imputed_X_test).transform(imputed_X_test) 
                                                                norm_param = [0, 0, 0, 0, 1]
                                        
                                                        if param3 < len(od_strategy) - 2:
                                                                if od_strategy[param3] == 'none':
                                                                        outlier_y_pred = np.ones(len(scaled_X_test))
                                                                        od_param = [1, 0, 0]
                                                                if od_strategy[param3] == 'if':
                                                                        outlier_y_pred = IsolationForest(max_samples=100, random_state=0).fit_predict(scaled_X_test)
                                                                        od_param = [0, 1, 0]
                                                        else:
                                                                k = lof_k_lst[param3 - 2] # start accessing number of neighbors in lof
                                                                outlier_y_pred = LocalOutlierFactor(n_neighbors=k, contamination=lof_contamination).fit_predict(scaled_X_test)
                                                                od_param = [0, 0, k]
                                                        mask = outlier_y_pred != -1
                                        # print("Fraction of outliers: ", round((1 - sum(mask)/len(outlier_y_pred)) * 100, 4))

                                        # import pdb; pdb.set_trace();
                                                        outlier_X_test = scaled_X_test.copy()
                                                        outlier_y_test = updated_y_test.copy()
                                                        priv_idx_test = None
                                                        unpriv_idx_test = None
                                                        if(metric_type=='sp'):
                                                                outlier_sensitive_test = updated_sensitive_attr_test.copy()
                                                                if (sum(mask) > 0 and sum(mask) < len(outlier_y_pred)): # at least one outlier
                                                                        # import pdb;pdb.set_trace()
                                                                        outlier_X_test, outlier_y_test, outlier_sensitive_test = scaled_X_test[mask], updated_y_test[mask], updated_sensitive_attr_test[mask]
                                                                        outlier_y_test.reset_index(drop=True, inplace=True)
                                                                        outlier_sensitive_test.reset_index(drop=True, inplace=True)          
                                                                priv_idx_test = [i for i, val in enumerate(outlier_sensitive_test) if val == 1]
                                                                unpriv_idx_test = [i for i, val in enumerate(outlier_sensitive_test) if val == 0]

                                                        updated_model = None
                                                        # feature_arr = [0,0]
                                                        # if feature == 0:
                                                        #         feature_arr = [1,0]
                                                        # else:
                                                        #         feature_arr = [0,1]

                                                        # feature_method = feature_strategy[feature]
                                                        # print(outlier_X_test.columns)
                                                        #outlier_X_test  = self.feature_selection(feature_method,outlier_X_test.copy())
                                                        #import pdb;pdb.set_trace()
                                                        if modelType == 'lr':
                                                                updated_model = LogisticRegression(random_state=0).fit(outlier_X_test, outlier_y_test)
                                                        elif modelType == 'nb':
                                                                updated_model = GaussianNB().fit(outlier_X_test, outlier_y_test)

                                                        y_pred = updated_model.predict(outlier_X_test)
                                                        # eqOdds_test = self.computeEqualizedOdds(y_pred, outlier_y_test, priv_idx_test, unpriv_idx_test)
                                                        outc = None
                                                        # import pdb;pdb.set_trace()
                                                        if metric_type== 'sp':
                                                                outc = self.computeStatisticalParity(y_pred[priv_idx_test],y_pred[unpriv_idx_test])
                                                        elif metric_type=='f-1':
                                                                outc = f1_score(outlier_y_test,y_pred)
                                                        elif metric_type=='mae':
                                                                outc = mean_absolute_error(outlier_y_test, y_pred)
                                                        elif metric_type=='rmse':
                                                                outc = np.sqrt(mean_squared_error(outlier_y_test, y_pred)) 
                                                        # import pdb;pdb.set_trace()
                                                        # if(outc == 0):
                                                        #         import pdb;pdb.set_trace()
                                                        f = [outc]
                                                        param_lst.append(mv_param + norm_param + od_param + f)
                                                        # if eqOdds_test == 0.0:
                                                        print(str(mv_param + norm_param + od_param + f))
                                        # print(len(unpriv_idx_test))

                        param_lst_df = pd.DataFrame(param_lst, columns=["mv_drop","mv_mean","mv_median","mv_mode","mv_knn",
                                                                        "norm_none","norm_ss","norm_rs","norm_ma","norm_mm",
                                                                        "od_none", "od_if","od_lof","fairness"])
                        # import pdb; pdb.set_trace()
                        param_lst_df.to_csv(file_name,index=False)
                # else:
                #         param_lst_df = pd.read_csv(file_name)
                # y = param_lst_df['fairness']
                # X = param_lst_df.copy().drop("fairness", axis=1)
                # reg = Regression()
                # model = reg.generate_regression(X, y)
                # coefs = model.coef_
                # print(coefs)
                # print(model.intercept_)
                # self.coefs = coefs
                # sublist_size = len(coefs) // 3
                # self.coef_normalization = coefs[5:10]
                # self.coef_imputer = coefs[0:5]
                # self.coef_outlier = coefs[10:13]
                # self.coef_normalization_rank = np.argsort(np.abs( coefs[5:10])).tolist()
                # self.coef_imputer_rank = np.argsort(np.abs(coefs[0:5])).tolist()
                # self.coef_outlier_rank = np.argsort(np.abs(coefs[10:13])).tolist()
        def feature_selection(self,strategy,dn):
                        #import pdb;pdb.set_trace()
                        if (strategy == 'L1'):

                                from sklearn.linear_model import Lasso

                                model = Lasso(alpha=100.0, tol=0.01, random_state=0)

                                model.fit(dn, dn)

                                coef = np.abs(model.coef_)

                                abstract_threshold = np.percentile(coef, 100. * .3)

                                to_discard = dn.columns[coef < abstract_threshold]

                                dn = dn.drop(to_discard, axis=1)
                                return dn
                        else:
                  

                                from sklearn.ensemble import RandomForestRegressor

                                model = RandomForestRegressor(n_estimators=50,
                                                        n_jobs=-1,
                                                        random_state=0)

                                model.fit(dn, dn)

                                coef = model.feature_importances_
                                
                                abstract_threshold = np.percentile(coef, 100. * .3)
                                try:
                                        to_discard = dn.columns[coef < abstract_threshold]
                                except:
                                        import pdb;pdb.set_trace()

                                dn = dn.drop(to_discard, axis=1)
                                return dn
                    
        def write_quartiles(self,csv_writer, algorithm, metric, quartiles):
                csv_writer.writerow([f_goal, algorithm, f"{metric} q1", round(quartiles[0], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q2", round(quartiles[1], 5)])
                csv_writer.writerow([f_goal, algorithm, f"{metric} q3", round(quartiles[2], 5)])
                # csv_writer.writerow([f_goal, algorithm, f"{metric} q4", round(quartiles[3], 5)])
p = base()



filename_test = 'historical_data/historical_data_test_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
filename_train = 'historical_data/historical_data_train_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
p.create_historic_data(filename_train)

p.create_historic_data_test(filename_test)
f_goals = [19200,20000,14000,16000,18000]


# #Read from historical data gererated on training data 
historical_data = pd.read_csv(filename_test)
p.historical_data_pd = historical_data;
# #Convert to list of list containing all the combination of transformers in a fixed pipeline
p.historical_data = historical_data.values.tolist();


f = sys.stdout
metric_path  = 'metric/metric_'+modelType+'_'+metric_type+'_'+dataset+'.csv'
f = open(metric_path, 'w')
# f.write("Ranking  and Grid search statistic ")
# f.write('\n')
gg  = historical_data.values.tolist()
csv_writer = csv.writer(f)

for f_goal in f_goals:
        print(f'Fairness goal {f_goal}')
        rank_idistr = []
        rank_fdistr = []
        gs_idistr = []
        gs_fdistr = []

        random.shuffle(gg)
        failures = 0
        for seed_ in gg:
                p.grid_search(f_goal, 1)
                # print(p.gs_idistr)
                gs_idistr.append(p.gs_idistr[0])
                gs_fdistr.append(p.gs_fdistr[0])

                p.optimize(seed_, f_goal)

                if p.rank_iter != -1:
                        # print('Rank : ',p.rank_f)
                        # print('Rank : ',p.rank_iter)
                        rank_idistr.append(p.rank_iter)
                        rank_fdistr.append(p.rank_f)
                else:
                        failures += 1
        # print('rank appended',rank_idistr)
        print(p.fail)
        print(p.pass_)
        # import pdb;pdb.set_trace()
        rank_iquartiles = np.percentile(rank_idistr, [25, 50, 75,100], interpolation='midpoint')
        rank_fquartiles = np.percentile(rank_fdistr, [25, 50, 75,100], interpolation='midpoint')
        g_iquartiles = np.percentile(gs_idistr, [25, 50, 75,100], interpolation='midpoint')
        g_fquartiles = np.percentile(gs_fdistr, [25, 50, 75,100], interpolation='midpoint')
        print("Fairness goal stats: " + str(f_goal))
        # f.write("stats for Fairness goal: " + str(f_goal))
        # f.write('\n')
        # print(rank_iquartiles)
        # f.write("ranking algorithm iterations q1: " + str(round(rank_iquartiles[0], 5)))
        # f.write('\n')
        # f.write("ranking algorithm iterations q2: " + str(round(rank_iquartiles[1], 5)))
        # f.write('\n')
        # f.write("ranking algorithm iterations q3: " + str(round(rank_iquartiles[2], 5)))
        # f.write('\n')
        # print(rank_fquartiles)
        # f.write("ranking algorithm Fairness q1: " + str(round(rank_fquartiles[0], 5)))
        # f.write('\n')
        # f.write("ranking algorithm Fairness q2: " + str(round(rank_fquartiles[1], 5)))
        # f.write('\n')
        # f.write("ranking algorithm Fairness q3: " + str(round(rank_fquartiles[2], 5)))
        # f.write('\n')
        # print(failures)
        # f.write("ranking algorithm failures: " + str(failures))
        # f.write('\n')
        # print(g_iquartiles)
        # f.write("grid search iterations q1: " + str(round(g_iquartiles[0], 5)))
        # f.write('\n')
        # f.write("grid search iterations q2: " + str(round(g_iquartiles[1], 5)))
        # f.write('\n')
        # f.write("grid search iterations q3: " + str(round(g_iquartiles[2], 5)))
        # f.write('\n')
        # print(g_fquartiles)
        # f.write("grid search Fairness q1: " + str(round(g_fquartiles[0], 5)))
        # f.write('\n')
        # f.write("grid search Fairness q2: " + str(round(g_fquartiles[1], 5)))
        # f.write('\n')
        # f.write("grid search Fairness q3: " + str(round(g_fquartiles[2], 5)))
        # f.write('\n')



# Write header
        csv_writer.writerow(["Fairness Goal", "Grid search", "Iteration", "Value"])

        # Write data for ranking algorithm
        p.write_quartiles(csv_writer, "ranking", "iterations", rank_iquartiles)
        p.write_quartiles(csv_writer, "ranking", "Fairness", rank_fquartiles)
        csv_writer.writerow([])

        # Write data for grid search algorithm
        p.write_quartiles(csv_writer, "grid search", "iterations", g_iquartiles)
        p.write_quartiles(csv_writer, "grid search", "Fairness", g_fquartiles)
        csv_writer.writerow([])
        #import pdb;pdb.set_trace()
f.close()


