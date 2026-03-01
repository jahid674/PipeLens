from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier  # ✅ add this
from regression import Regression  # Custom regression module
import numpy as np
np.random.seed(42)


class ModelTrainer:
    def __init__(self, model_type):
        """
        Parameters:
            model_type (str): One of ['lr', 'nb', 'rf', 'dt', 'svm', 'nn', 'reg']
        """
        self.model_type = model_type.lower()
        if self.model_type not in ['lr', 'nb', 'rf', 'reg', 'dt', 'svm', 'nn']:
            raise ValueError("Invalid model type. Choose from 'lr', 'nb', 'rf', 'dt', 'svm', 'nn', 'reg'.")
        self.model = None

    def train(self, X, y):
        # Drop rows with any NaN in X, then align y using the same row indices
        if hasattr(X, "dropna") and hasattr(y, "__len__"):
            kept_idx = X.dropna().index
            X = X.loc[kept_idx].reset_index(drop=True)
            y = y.loc[kept_idx].reset_index(drop=True) if hasattr(y, "loc") else y[kept_idx]

        if self.model_type == 'lr':
            self.model = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)

        elif self.model_type == 'nb':
            self.model = GaussianNB().fit(X, y)

        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(random_state=42).fit(X, y)

        elif self.model_type == 'dt':
            self.model = DecisionTreeClassifier(random_state=42).fit(X, y)

        elif self.model_type == 'svm':
            self.model = SVC(random_state=42, kernel='rbf', probability=True).fit(X, y)

        elif self.model_type == 'nn':
            # Neural network tuned for Adult / HMDA tabular fairness datasets
            self.model = Pipeline([
                ("scaler", StandardScaler()),

                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(16,),     # single small layer (acts like nonlinear LR)
                    activation="relu",
                    solver="adam",

                    # regularization (VERY important for HMDA)
                    alpha=0.01,                   # strong L2 regularization

                    # stable learning
                    learning_rate_init=0.0005,
                    batch_size=64,

                    # convergence control
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,

                    # stability across runs (important for fairness experiments)
                    tol=1e-4,
                    random_state=42
                ))
            ])

            self.model.fit(X, y)



        elif self.model_type == 'reg':
            self.model = Regression().generate_regression(X, y)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return self.model

    def get_model(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model
