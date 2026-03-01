import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import KBinsDiscretizer

class Binner:
    def __init__(self, dataset, column, strategy='uniform', n_bins=5, encode='ordinal', verbose=False):
        """
        Parameters:
        - dataset: input DataFrame
        - column: column name to bin
        - strategy: 'uniform', 'quantile', or 'kmeans'
        - n_bins: number of bins to divide the data into
        - encode: 'ordinal', 'onehot', or 'onehot-dense'
        - verbose: print logs if True
        """
        self.dataset = dataset.copy()
        self.column = column
        self.strategy = strategy
        self.n_bins = n_bins
        self.encode = encode
        self.verbose = verbose

    def transform(self):
        start_time = time.time()
        df = self.dataset.copy()

        if self.column not in df.columns:
            if self.verbose:
                print(f"Column '{self.column}' not found. Skipping binning.")
            return df

        if not np.issubdtype(df[self.column].dtype, np.number):
            if self.verbose:
                print(f"Column '{self.column}' is not numeric. Skipping binning.")
            return df

        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy)
        binned = discretizer.fit_transform(df[[self.column]])

        if self.encode == 'ordinal':
            df[self.column + '_binned'] = binned.astype(int)
        else:
            # One-hot/dense produces multiple columns
            binned_df = pd.DataFrame(binned.toarray() if hasattr(binned, 'toarray') else binned,
                                     columns=[f"{self.column}_bin{i}" for i in range(binned.shape[1])])
            df = df.drop(columns=[self.column])
            df = pd.concat([df.reset_index(drop=True), binned_df.reset_index(drop=True)], axis=1)

        if self.verbose:
            print(f"Binning applied to column '{self.column}' using strategy '{self.strategy}' in {time.time() - start_time:.2f} seconds.")

        return df