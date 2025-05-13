import numpy as np
import operator
import random
import pandas as pd
import os
import statistics
from scipy.stats import pearsonr, chi2_contingency
from scipy import stats
from sklearn.svm import OneClassSVM

from modules.matching.perfectmatching import PerfectMatching
from modules.matching.jaccardmatching import JaccardMatching

random.seed(0)

class Profile:
  profile_lst=[]
  def __init__(df, self):
    pass
  def outlier(self,lst):
    mean=statistics.mean(lst)
    std=statistics.stdev(lst)
    count=0
    for v in lst:
      if v>mean+2*std or v<mean-2*std:
        count+=1
    return count*1.0/len(lst)
  def missing(self,lst):
    count = 0
    for v in lst:
      try:
        if np.isnan(v):
          count += 1
      except:
        if len(v) == 0:
          count += 1
    return count*1.0/len(lst)
  
  def correlation(self,lst1,lst2):
    try:
        (r,p)= pearsonr(lst1,lst2)
        if True:
                return r
        else:
                return 0
    except:
           print("efe")
    
  def categorical_correlation(self,lst1,lst2):
    cross_tab=pd.crosstab(lst1,lst2)
    chi2, p, dof, ex=chi2_contingency(cross_tab)
    return chi2

  def categorical_numerical_correlation(self,lst1,lst2):
    (chi2,p)=stats.f_oneway(lst1,lst2)
    return chi2
  def get_fraction_of_outlier(self,data):
        svm_model = OneClassSVM(kernel='rbf')  # You can adjust the parameters as needed
        svm_model.fit(data)

        # Step 2: Predict the labels of your data points
        predicted_labels = svm_model.predict(data)

        # Step 3: Count the number of predicted outliers
        n_outliers = (predicted_labels == -1).sum()

        # Step 4: Calculate the fraction of outliers
        fraction_outliers = n_outliers / len(data)
        return fraction_outliers
  
  def populate_profiles(self,data_final,outlier, numerical_columns, target, metric_type):
    scaling_factor = 1
    
    profile_map={}
    # import pdb;pdb.set_trace()


    categorical_values={}
    #Partition each column as categorical, numerical and textual
    #Each profilehas four parameters where 3rd is conditional attribute 4th is value
#     le = LabelEncoder()
    
#     for column in categorical_columns:
      
#       data_final[column] = le.fit_transform(data_final[column]) 
      
    profile = {}
    i = 0
    # import pdb;pdb.set_trace()

       
    for column in data_final.columns:

        if(column==target):
          continue
        if(metric_type=='rmse' or metric_type=='mae'):
                if column in numerical_columns :
                        #pearson -  regression 
                        corr = self.correlation(data_final[column],data_final[target])
                else:
                        #categorical_numerical_correlation - regression
                        corr = self.categorical_numerical_correlation(data_final[column],data_final[target])
        else:
                if column in numerical_columns :
                       corr = self.categorical_numerical_correlation(data_final[column],data_final[target])
                else:
                        #categorical_numerical_correlation - regression
                        corr = self.categorical_correlation(data_final[column],data_final[target])
               
        # missing_value = self.missing(self.df[categorical_columns[i]])
        #outlier  = self.outlier(self.df[categorical_columns[i]])
        
        name = column
        tuple = ('corr_' + name,  'ot_' + name)
        profile[tuple]= [column,round(corr*scaling_factor,5),round(outlier*scaling_factor,5)]
        i+=1
    dd = []
    keys = []
    for val in profile:
        # import pdb;pdb.set_trace()
        dd.append(profile[val][1])
        dd.append(profile[val][2])
        keys.append(val[0])
        keys.append(val[1])
    return dd,keys
  
data_final = pd.DataFrame({
    'age': [25, 30, 22, 35, 40, 28],
    'income': [50000, 60000, 55000, 65000, 70000, 62000],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F'],
    'purchased': [1, 0, 0, 1, 1, 0]
})

numerical_columns = ['age', 'income']

target_column = 'purchased'

outlier_fraction = 0.1
profiler = Profile(data_final)

results, keys = profiler.populate_profiles(
    data_final=data_final,
    outlier=outlier_fraction,
    numerical_columns=numerical_columns,
    target=target_column,
    metric_type='classification'
)

print("Keys:", keys)
print("Results:", results)