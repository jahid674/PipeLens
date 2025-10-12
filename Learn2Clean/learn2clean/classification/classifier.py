#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille

import time
import warnings
import sklearn.exceptions
from typing import Optional, Tuple
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class Classifier:
    """
    Classification task using a particular method.

    dataset: {'train': DataFrame, 'test': DataFrame, 'target': Series, (optional) 'target_test': Series}
    strategy: one of {'NB','LDA','CART','MNB','LR'}
    target: name of the target variable (must equal dataset['target'].name)
    dataset_name: string used to toggle fairness logic (e.g., 'adult')
    """

    def __init__(self, dataset, target, strategy='NB', k_folds=10,
                 verbose=False, dataset_name: Optional[str] = None):

        self.dataset = dataset
        self.target = target
        self.strategy = strategy
        self.k_folds = k_folds
        self.verbose = verbose
        self.dataset_name = (dataset_name or "").lower()

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'target': self.target,
            'k_folds': self.k_folds,
            'verbose': self.verbose,
            'dataset_name': self.dataset_name,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for classifier. "
                              "Parameter(s) IGNORED. "
                              "Check with `classifier.get_params().keys()`")
            else:
                setattr(self, k, v)

    # ----------------- Helpers ----------------- #

    def _drop_if_present(self, X, col):
        if col in X.columns.values:
            return X.drop(columns=[col])
        return X

    def _compute_stat_parity(self, y_pred: pd.Series, sensitive_series: pd.Series) -> Optional[float]:
        """
        Statistical Parity (difference):
        P(Ŷ=1 | A=1) - P(Ŷ=1 | A≠1), where privileged A==1
        Returns absolute value (non-negative).
        """
        try:
            s = sensitive_series.astype(float)
            priv_mask = (s == 1)
            unpriv_mask = (s != 1) & (~s.isna())

            if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
                return None

            p_priv = (y_pred[priv_mask] == 1).mean()
            p_unpriv = (y_pred[unpriv_mask] == 1).mean()
            return float(abs(p_priv - p_unpriv))
        except Exception:
            return None

    # ----------------- Models ----------------- #

    def LDA_classification(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):
            print('Error: Need at least one continous variable and', k, 'observations for classification')
            return None

        y_train = dataset['target'].loc[X_train.index]
        X_test = dataset['test'].select_dtypes(['number']).dropna()
        y_test = dataset.get('target_test', dataset['target']).loc[X_test.index]

        X_train = self._drop_if_present(X_train, target)
        X_test = self._drop_if_present(X_test, target)

        if dataset['target'].nunique() < k:
            k = dataset['target'].nunique()

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
        model = LinearDiscriminantAnalysis(n_components=2)
        gs = GridSearchCV(model, cv=skf, param_grid={}, scoring='accuracy')
        gs.fit(X_train, y_train)

        results = gs.cv_results_
        clf = gs.best_estimator_
        best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
        accuracy = results['mean_test_score'][best_index]

        if target in X_test.columns.values:
            accuracy = clf.score(X_test, y_test)
            if self.verbose:
                y_true, y_pred = y_test, clf.predict(X_test)
                print("Detailed classification report:")
                print(classification_report(y_true, y_pred))

        print(f"\nAccuracy of LDA result for {self.k_folds} cross-validation : {accuracy}\n")
        return accuracy

    def CART_classification(self, dataset, target):
        NUM_TRIALS = 10
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):
            print('Error: Need at least one continous variable and', k, 'observations for classification')
            return None

        y_train = dataset['target'].loc[X_train.index]
        X_test = dataset['test'].select_dtypes(['number']).dropna()
        y_test = dataset.get('target_test', dataset['target']).loc[X_test.index]

        X_train = self._drop_if_present(X_train, target)
        X_test = self._drop_if_present(X_test, target)

        params = {'max_depth': [3, 5, 7, 9, 10]}
        for i in range(1, NUM_TRIALS):
            inner_cv = KFold(n_splits=k, shuffle=True, random_state=i)
            outer_cv = KFold(n_splits=k, shuffle=True, random_state=i)
            model = DecisionTreeClassifier(random_state=i)
            gs = GridSearchCV(model, cv=inner_cv, param_grid=params, scoring='accuracy')
            gs.fit(X_train, y_train)
            _ = cross_val_score(gs, X=X_train, y=y_train, cv=outer_cv)

        clf = gs.best_estimator_
        results = gs.cv_results_
        best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
        accuracy = results['mean_test_score'][best_index]

        if target in X_test.columns.values:
            accuracy = clf.score(X_test, y_test)
            if self.verbose:
                y_true, y_pred = y_test, clf.predict(X_test)
                print("Detailed classification report:")
                print(classification_report(y_true, y_pred))

        print(f"Avg accuracy of CART classification for {k} cross-validation : {accuracy}\n")
        return accuracy

    def NB_classification(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):
            print('Error: Need at least one continous variable and', k, 'observations for classification')
            return None

        y_train = dataset['target'].loc[X_train.index]
        X_test = dataset['test'].select_dtypes(['number']).dropna()
        y_test = dataset.get('target_test', dataset['target']).loc[X_test.index]

        X_train = self._drop_if_present(X_train, target)
        X_test = self._drop_if_present(X_test, target)

        skf = StratifiedKFold(n_splits=k)
        model = GaussianNB()
        gs = GridSearchCV(model, cv=skf, param_grid={}, scoring='accuracy')
        gs.fit(X_train, y_train)

        clf = gs.best_estimator_
        results = gs.cv_results_
        best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
        accuracy = results['mean_test_score'][best_index]

        if target in X_test.columns.values:
            accuracy = clf.score(X_test, y_test)
            if self.verbose:
                y_true, y_pred = y_test, clf.predict(X_test)
                print("Detailed classification report:")
                print(classification_report(y_true, y_pred))

        print(f"Accuracy of Naive Naive Bayes classification for {k} cross-validation : {accuracy}\n")
        return accuracy

    def MNB_classification(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):
            print('Error: Need at least one continous variable and', k, 'observations for classification')
            return None

        y_train = dataset['target'].loc[X_train.index]
        X_test = dataset['test'].select_dtypes(['number']).dropna()
        y_test = dataset.get('target_test', dataset['target']).loc[X_test.index]

        X_train = self._drop_if_present(X_train, target)
        X_test = self._drop_if_present(X_test, target)

        skf = StratifiedKFold(n_splits=k)
        params = {"alpha": np.arange(0.001, 1, 0.01)}
        model = MultinomialNB()

        gs = GridSearchCV(model, cv=skf, param_grid=params, scoring='accuracy')
        gs.fit(X_train, y_train)

        clf = gs.best_estimator_
        results = gs.cv_results_
        best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
        accuracy = results['mean_test_score'][best_index]

        if target in X_test.columns.values:
            accuracy = clf.score(X_test, y_test)
            if self.verbose:
                y_true, y_pred = y_test, clf.predict(X_test)
                print("Detailed classification report:")
                print(classification_report(y_true, y_pred))
                print("Best alpha:", gs.best_params_)

        print(f"Accuracy of Multinomial Naive Bayes classification for {k} cross-validation : {accuracy:.3f}\n")
        return accuracy

    def LR_classification(self, dataset, target) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns (accuracy, abs_stat_parity) where SP is computed
        only when dataset_name == 'adult' and a Sex column exists.
        Accuracy is measured on TRAIN.
        """
        k = self.k_folds
        if self.dataset_name == 'adult':
            X_train = dataset['train'].dropna()
            X_test = dataset['test'].dropna()
        else:
            X_train = dataset['train'].select_dtypes(['number']).dropna()
            X_test = dataset['test'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):
            print('Error: Need at least one continous variable and', k, 'observations for classification')
            return None, None

        y_train = dataset['target'].loc[X_train.index]

        # keep parity with original signature even if not used
        X_test = dataset['test'].dropna()
        _ = dataset['target'].loc[X_test.index]

        X_train = self._drop_if_present(X_train, target)
        X_test = self._drop_if_present(X_test, target)

        model = LogisticRegression(random_state=0, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        accuracy = float(accuracy_score(y_train, y_pred_train))


        # Statistical parity (absolute) for Adult only
        abs_sp = None
        if self.dataset_name == 'adult':
            sensitive = dataset['train'].loc[X_train.index, 'Sex']
            abs_sp = self._compute_stat_parity(
                    pd.Series(y_pred_train, index=X_train.index), sensitive
                )

        return accuracy, abs_sp

    # ----------------- Public API ----------------- #

    def transform(self):
        start_time = time.time()

        d = self.dataset
        if self.target != d['target'].name:
            raise ValueError("Target variable invalid.")

        print("\n>>Classification task")

        extra = {}
        if self.strategy == "LDA":
            dn = self.LDA_classification(dataset=d, target=self.target)
        elif self.strategy == "CART":
            dn = self.CART_classification(dataset=d, target=self.target)
        elif self.strategy == "NB":
            dn = self.NB_classification(dataset=d, target=self.target)
        elif self.strategy == "MNB":
            dn = self.MNB_classification(dataset=d, target=self.target)
        elif self.strategy == "LR":
            print("WE are sailing into LR")
            acc, abs_sp = self.LR_classification(dataset=d, target=self.target)
            print(f"fairness metric (|SP|) = {abs_sp}")
            print(f"accuracy = {acc}")

            # === Objective (quality_metric) ===
            if self.dataset_name == 'adult':
                # Use a higher-is-better fairness metric: 1 - |SP|
                # If abs_sp is None, keep quality_metric as None to avoid false success.
                dn = None if abs_sp is None else float(1.0 - abs_sp)
                if acc is not None:
                    extra['accuracy'] = float(acc)
                if abs_sp is not None:
                    extra['abs_statistical_parity'] = float(abs_sp)
            else:
                # Others => accuracy objective (higher is better)
                dn = None if acc is None else float(acc)
        else:
            raise ValueError("The classification function should be LDA, CART, NB, MNB, or LR.")

        print("Classification done -- CPU time: %s seconds" % (time.time() - start_time))
        out = {'quality_metric': dn}
        out.update(extra)
        return out
