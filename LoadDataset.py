import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder
from modules.Util.reader import Reader

class LoadDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def get_sensitive_variable(self):
        if self.dataset_name == 'adult':
            return 'Sex'
        elif self.dataset_name == 'housing':
            return 'OverallQual'
        elif self.dataset_name == 'hmda':
            return 'race'

    def load(self):
        if self.dataset_name == 'hmda':
            train_file = "data/hmda/hmda_Orleans_X_train_1.csv"
            test_file = "data/hmda/hmda_Calcasieu_X_test_1.csv"
            train, test = Reader(train_file, test_file).load_data()

            

            self.y_train = train['action_taken']
            self.X_train = train.drop('action_taken', axis=1)
            self.y_test = test['action_taken']
            self.X_test = test.drop('action_taken', axis=1)

        elif self.dataset_name == 'adult':
            train_file = "data/adult/adult_test.csv"
            test_file = "data/adult/adult_train.csv"
            train, test = Reader(train_file, test_file).load_data()

            '''if 'fnlwgt' in train.columns:
                train = train.drop('fnlwgt', axis=1)
            if 'fnlwgt' in test.columns:
                test = test.drop('fnlwgt', axis=1)'''

            categorical_columns = train.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                column_unique = pd.unique(list(train[column]) + list(test[column]))
                #print(f"Encoding column: {column} with unique values: {column_unique}")
                le.fit(column_unique)
                train[column] = le.transform(train[column])
                test[column] = le.transform(test[column])
                train[column] = train[column].astype('category')
                test[column] = test[column].astype('category')

            self.y_train = train['income']
            self.X_train = train.drop('income', axis=1)
            self.y_test = test['income']
            self.X_test = test.drop('income', axis=1)


        elif self.dataset_name == 'housing':
            train_file = "data/house/housing_train.csv"
            test_file = "data/house/housing_test.csv"
            train, test = Reader(train_file, test_file).load_data()

            missing_percentage = (train.isnull().sum() / len(train)) * 100
            train.drop(columns=missing_percentage[missing_percentage > 40].index.tolist(), inplace=True)

            cat_cols = ['Electrical','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','GarageType','GarageFinish','GarageQual','GarageCond']
            num_cols = ['LotFrontage','GarageYrBlt','MasVnrArea']
            for col in cat_cols:
                if col in train.columns:
                    train[col].fillna(train[col].mode()[0], inplace=True)
            for col in num_cols:
                if col in train.columns:
                    train[col].fillna(train[col].median(), inplace=True)

            categorical_columns = train.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                column_unique = pd.unique(list(train[column]) + list(test[column]))
                le.fit(column_unique)
                train[column] = le.transform(train[column])
                test[column] = le.transform(test[column])
                train[column] = train[column].astype('category')
                test[column] = test[column].astype('category')

            self.X_train = train.copy()
            self.y_train = self.X_train.pop('SalePrice')
            self.X_test = test.copy()
            self.y_test = self.X_test.pop('SalePrice')

            selector = SelectKBest(score_func=f_regression, k=15)
            selector.fit(self.X_train, self.y_train)
            selected_indices = selector.get_support(indices=True)
            selected_features = self.X_train.columns[selected_indices]

            self.X_train = self.X_train[selected_features]
            self.X_test = self.X_test[selected_features]


        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

        self.dataset = {
            'train': train,
            'test': test,
        }

        return self.dataset, self.X_train, self.y_train, self.X_test, self.y_test


        # frequency Encoding for categorical variables
        '''elif self.dataset_name == 'adult':
            train_file = "data/adult/adult_test.csv"
            test_file = "data/adult/adult_train.csv"
            train, test = Reader(train_file, test_file).load_data()

            categorical_columns = train.select_dtypes(include=['object']).columns

            for column in categorical_columns:
                # Combine train and test for consistent frequency mapping
                combined = pd.concat([train[column], test[column]])
                freq = combined.value_counts(normalize=True)

                # Map frequencies
                train[column] = train[column].map(freq).fillna(0)
                test[column] = test[column].map(freq).fillna(0)

            self.y_train = train['income']
            self.X_train = train.drop('income', axis=1)
            self.y_test = test['income']
            self.X_test = test.drop('income', axis=1)'''


'''dataset='housing'
loader = LoadDataset(dataset)
dataset, X_train, y_train, X_test, y_test = loader.load()
x_dedup=X_test.drop_duplicates().reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
y_test=y_test.loc[x_dedup.index].reset_index(drop=True)

print(f"Training data shape: {X_test.dtypes}")
print(f"Dedup ratio: {x_dedup.shape[0]}")
print(f"Test data shape: {y_test.shape[0]}")'''