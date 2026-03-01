from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from regression import Regression  # Custom regression module

class ModelTrainer:
    def __init__(self, model_type):
        """
        Parameters:
            model_type (str): One of ['lr', 'nb', 'rf', 'reg']
        """
        self.model_type = model_type.lower()
        if self.model_type not in ['lr', 'nb', 'rf', 'reg', 'dt', 'svm']:
            raise ValueError("Invalid model type. Choose from 'lr', 'nb', 'rf', 'reg'.")
        self.model = None

    def train(self, X, y):
        # Drop rows with any NaN in X, then align y using the same row indices
        if hasattr(X, "dropna") and hasattr(y, "__len__"):
            kept_idx = X.dropna().index
            X = X.loc[kept_idx].reset_index(drop=True)
            y = y.loc[kept_idx].reset_index(drop=True) if hasattr(y, "loc") else y[kept_idx]

        if self.model_type == 'lr':
            self.model = LogisticRegression(random_state=0).fit(X, y)
        elif self.model_type == 'nb':
            self.model = GaussianNB().fit(X, y)
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(random_state=0).fit(X, y)
        elif self.model_type == 'dt':
            self.model = DecisionTreeClassifier(random_state=0).fit(X, y)
        elif self.model_type == 'svm':
            self.model = SVC(random_state=0, kernel='rbf', probability=True).fit(X, y)
        elif self.model_type == 'reg':
            self.model = Regression().generate_regression(X, y)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return self.model

    def get_model(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model
