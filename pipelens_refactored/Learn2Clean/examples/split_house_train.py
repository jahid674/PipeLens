import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
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


train = "../datasets/house/house_train.csv"
dataset = pd.DataFrame(pd.read_csv(train))

train, test = train_test_split(
    dataset, test_size=0.5, random_state=42)

train.to_csv('../datasets/house/housing_train.csv', index=False)  
test.to_csv('../datasets/house/housing_test.csv', index=False)  