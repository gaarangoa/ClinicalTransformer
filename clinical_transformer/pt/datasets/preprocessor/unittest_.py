import unittest
import numpy as np

from clinical_transformer.pt.datasets.preprocessor.tabular import Preprocessor

class MockAnnData:
    def __init__(self, X, var_names):
        self.X = X
        self.var_names = var_names

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.prep = Preprocessor(
            categorical_features=['cat1'],
            numerical_features=['num1']
        )
        # Simulate fit
        self.prep.features = {
            'cat1': {'type': 'cat', 'min': 0., 'max': 1., 'encode': {'A': 0, 'B': 1}},
            'num1': {'type': 'num', 'min': 0., 'max': 10.}
        }
        self.prep.feature_encoder = {'cat1': 3, 'num1': 4}
    
    def test_transform_from_ad(self):
        # 2 samples, 2 features
        X = np.array([
            [0, 'A'],   # num1=0 (should be ignored), cat1='A'
            [5, 'B']    # num1=5, cat1='B'
        ], dtype=object)
        adata = MockAnnData(X, ['num1', 'cat1'])
        result = self.prep.transform_from_ad(adata)
        # First sample: only cat1 should be present
        self.assertEqual(result[0][0], [3])  # tokens
        self.assertEqual(result[0][1], [0.0])  # values
        # Second sample: both features
        self.assertEqual(set(result[1][0]), {3, 4})
        self.assertAlmostEqual(result[1][1][result[1][0].index(4)], 0.5)  # num1 scaled
        self.assertEqual(result[1][1][result[1][0].index(3)], 1.0)        # cat1 scaled

if __name__ == '__main__':
    unittest.main()