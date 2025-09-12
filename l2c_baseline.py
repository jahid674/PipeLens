# l2c_baseline.py
import numpy as np
import pandas as pd
import logging

# L2C imports guarded to avoid hard failure if not installed
try:
    import Learn2Clean.python_package.learn2clean.loading.reader as rd
    import Learn2Clean.python_package.learn2clean.qlearning.qlearner as ql
    L2C_AVAILABLE = True
except Exception as e:
    logging.warning(f"[L2C] learn2clean import failed: {e}")
    L2C_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder

class L2CBaselineRunner:
    """
    Wraps your previous Learn2Clean script into a reusable baseline.
    Produces per-goal iteration counts (or -1 if goal not achieved).
    """

    def __init__(self, dataset_name: str, metric_type: str, tau: float = 0.1):
        self.dataset_name = dataset_name.lower()
        self.metric_type = metric_type.lower()
        self.tau = float(tau)

    def _prepare_hmda(self):
        # NOTE: keep paths identical to your original snippet
        dataset_files = ["../../data/hmda/hmda_Orleans_X_test_1.csv",
                         "../../data/hmda/hmda_Orleans_X_test_1.csv"]
        hr = rd.Reader(sep=',', verbose=False, encoding=False)
        dataset = hr.train_test_split(dataset_files, 'action_taken')

        # Inject MCAR missingness into a single column, mirroring your code
        rng = np.random.default_rng(1)
        n_tr = len(dataset['train'])
        n_ts = len(dataset['test'])
        mv_tr_idx = rng.choice(n_tr, size=int(self.tau * n_tr), replace=False)
        mv_ts_idx = rng.choice(n_ts, size=int(self.tau * n_ts), replace=False)

        dataset['train'].loc[mv_tr_idx, 'lien_status'] = np.nan
        dataset['test'].loc[mv_ts_idx, 'lien_status'] = np.nan

        return dataset

    def _prepare_housing(self):
        # Keep behavior identical to your script, but self-contained
        d2 = pd.read_csv('../datasets/house/housing_test.csv')
        dataset = {
            'train': d2.copy(),
            'test': d2.copy(),
            'target': d2['SalePrice'].copy(),
        }

        selected_features = [
            'OverallQual','GarageFinish','GarageArea','YearBuilt','TotalBsmtSF',
            '1stFlrSF','YearRemodAdd','GrLivArea','GarageCars','FullBath',
            'Fireplaces','BsmtQual','KitchenQual','ExterQual','TotRmsAbvGrd'
        ]
        missing_cat = ['Electrical','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure',
                       'BsmtFinType1','GarageType','GarageFinish','GarageQual','GarageCond']
        missing_num = ['LotFrontage','GarageYrBlt','MasVnrArea']

        # Fill missing values as before
        for col in missing_cat:
            most_f = dataset['train'][col].mode(dropna=True)
            if len(most_f) > 0:
                dataset['train'][col] = dataset['train'][col].fillna(most_f.iloc[0])
        for col in missing_num:
            med_v = dataset['train'][col].median(skipna=True)
            dataset['train'][col] = dataset['train'][col].fillna(med_v)

        selected_features = selected_features + ['SalePrice']
        dataset['train'] = dataset['train'][selected_features].copy()

        # Label encode categorical cols (as in your code)
        cat_cols = dataset['train'].select_dtypes(include=['object']).columns
        for c in cat_cols:
            le = LabelEncoder()
            dataset['train'][c] = le.fit_transform(dataset['train'][c].astype(str))
            dataset['train'][c] = dataset['train'][c].astype('category')

        # Inject some missingness into OverallQual (as before)
        rng = np.random.default_rng(1)
        n_tr = len(dataset['train'])
        mv_idx = rng.choice(n_tr, size=int(self.tau * n_tr), replace=False)
        dataset['train'].loc[mv_idx, 'OverallQual'] = np.nan

        return dataset

    def _build_dataset(self):
        if not L2C_AVAILABLE:
            logging.warning("[L2C] learn2clean unavailable; skipping baseline.")
            return None

        if self.dataset_name == 'hmda':
            return self._prepare_hmda()
        elif self.dataset_name == 'housing':
            return self._prepare_housing()
        else:
            logging.warning(f"[L2C] Unsupported dataset for baseline: {self.dataset_name}")
            return None

    def run(self, goals, random_seeds):
        """
        Returns:
            dict: goal -> list of iteration counts (ints), -1 if not achieved.
        """
        results = {g: [] for g in goals}
        dataset = self._build_dataset()
        if dataset is None:
            # No L2C; mark all as not run
            for g in goals:
                results[g] = []
            return results

        for g in goals:
            iters = []
            for seed in random_seeds:
                if self.dataset_name == 'hmda':
                    l2c = ql.Qlearner(dataset=dataset, goal='LR',
                                      target_goal='action_taken', target_prepare='action_taken',
                                      verbose=False, f_goal=g)
                else:  # housing
                    l2c = ql.Qlearner(dataset=dataset, goal='MARS',
                                      target_goal='SalePrice', target_prepare='SalePrice',
                                      verbose=False, f_goal=g)

                ok, it = l2c.learn2clean(r_state=int(seed))
                iters.append(it if ok else -1)
            results[g] = iters
        return results
