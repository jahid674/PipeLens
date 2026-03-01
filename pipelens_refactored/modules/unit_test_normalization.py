
import unittest
import pandas as pd
from .profiling.profile import PerfectMatching

# import pdb;pdb.set_trace()
class TestNormalizer(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset for testing
        data = {
            'feature1': [100, 200, 300, 400],
            'feature2': [2, 4, 6, 8],
        }
        self.dataset = {
            'train': pd.DataFrame(data),
            'test': pd.DataFrame(data),
        }

    def test_zs_normalization(self):
        normalizer = Normalizer(self.dataset, strategy='ZS')
        normalized_data = normalizer.transform()

        # Add your assertions here to check if the normalization is correct
        self.assertAlmostEqual(normalized_data['train']['feature1'].mean(), 0.0, delta=1e-6)
        self.assertAlmostEqual(normalized_data['train']['feature1'].std(), 1.0, delta=1e-6)
        self.assertAlmostEqual(normalized_data['train']['feature2'].mean(), 0.0, delta=1e-6)
        self.assertAlmostEqual(normalized_data['train']['feature2'].std(), 1.0, delta=1e-6)

    def test_mm_normalization(self):
        normalizer = Normalizer(self.dataset, strategy='MM')
        normalized_data = normalizer.transform()

        # Add your assertions here to check if the normalization is correct
        self.assertAlmostEqual(normalized_data['train']['feature1'].min(), 0.0, delta=1e-6)
        self.assertAlmostEqual(normalized_data['train']['feature1'].max(), 1.0, delta=1e-6)
        self.assertAlmostEqual(normalized_data['train']['feature2'].min(), 0.0, delta=1e-6)
        self.assertAlmostEqual(normalized_data['train']['feature2'].max(), 1.0, delta=1e-6)

if __name__ == '__main__':
    unittest.main()
