import pandas as pd

class Reader():

    def __init__(self, train_path,test_path):
        self.train_path = train_path
        self.test_path = test_path


    def load_data(self):
        train = pd.DataFrame(pd.read_csv(self.train_path))
        test = pd.DataFrame(pd.read_csv(self.test_path))
        # import pdb;pdb.set_trace()
        return train,test
