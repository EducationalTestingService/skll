
import scipy.sparse as sp
import sys
import os
import numpy as np
import classifier
from unittest import TestCase, main

_my_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(_my_path)

from run_experiment import load_featureset


class TestSklearnWrapper(TestCase):
    def setUp(self):
        pass

    def test_SelectByMinCount(self):
        m2 = [[0.001,   0.0,    0.0,    0.0],
              [0.00001, -2.0,   0.0,    0.0],
              [0.001,   0.0,    0.0,    4.0], 
              [0.0101,  -200.0, 0.0,    0.0]]

        # default should keep all nonzero features (i.e., ones that appear 1+ times)
        feat_selector = classifier.SelectByMinCount()
        expected = np.array([[0.001, 0.0, 0.0], [0.00001, -2.0, 0.0], [0.001, 0.0, 4.0], [0.0101, -200.0, 0.0]])
        self.assertTrue(np.array_equal(feat_selector.fit_transform(np.array(m2)), expected))
        self.assertTrue(np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected))

        # keep features that happen 2+ times
        feat_selector = classifier.SelectByMinCount(min_count=2)
        expected = np.array([[0.001, 0.0], [0.00001, -2.0], [0.001, 0.0], [0.0101, -200.0]])
        self.assertTrue(np.array_equal(feat_selector.fit_transform(np.array(m2)), expected))
        self.assertTrue(np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected))

        # keep features that happen 3+ times
        feat_selector = classifier.SelectByMinCount(min_count=3)
        expected = np.array([[0.001], [0.00001], [0.001], [0.0101]])
        self.assertTrue(np.array_equal(feat_selector.fit_transform(np.array(m2)), expected))
        self.assertTrue(np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected))


    def test_input_checking(self):
        dirpath = os.path.join(_my_path, 'tests')
        featureset = ['']
        suffix = '.jsonlines'
    
        featureset = ['test_input_2examples_1', 'test_input_3examples_1']
        with self.assertRaises(ValueError):
            load_featureset(dirpath, featureset, suffix)


        featureset = ['test_input_3examples_1', 'test_input_3examples_1']
        with self.assertRaises(ValueError):
            load_featureset(dirpath, featureset, suffix)

        featureset = ['test_input_3examples_1', 'test_input_3examples_2']
        feats = load_featureset(dirpath, featureset, suffix)
        self.assertTrue(len(feats) == 3, "Wrong number of examples.")
       
       
if __name__ == '__main__':
    main()
