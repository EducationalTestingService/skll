
import scipy.sparse as sp
import sys
import os
import numpy as np
import classifier

_my_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(_my_path)

from run_experiment import load_featureset
from nose.tools import *


def test_SelectByMinCount():
    m2 = [[0.001,   0.0,    0.0,    0.0],
          [0.00001, -2.0,   0.0,    0.0],
          [0.001,   0.0,    0.0,    4.0], 
          [0.0101,  -200.0, 0.0,    0.0]]

    # default should keep all nonzero features (i.e., ones that appear 1+ times)
    feat_selector = classifier.SelectByMinCount()
    expected = np.array([[0.001, 0.0, 0.0], [0.00001, -2.0, 0.0], [0.001, 0.0, 4.0], [0.0101, -200.0, 0.0]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)

    # keep features that happen 2+ times
    feat_selector = classifier.SelectByMinCount(min_count=2)
    expected = np.array([[0.001, 0.0], [0.00001, -2.0], [0.001, 0.0], [0.0101, -200.0]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)

    # keep features that happen 3+ times
    feat_selector = classifier.SelectByMinCount(min_count=3)
    expected = np.array([[0.001], [0.00001], [0.001], [0.0101]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)


@raises(ValueError)
def test_input_checking1():
    dirpath = os.path.join(_my_path, 'tests')
    featureset = ['']
    suffix = '.jsonlines'

    featureset = ['test_input_2examples_1', 'test_input_3examples_1']
    load_featureset(dirpath, featureset, suffix)


@raises(ValueError)
def test_input_checking2():
    dirpath = os.path.join(_my_path, 'tests')
    featureset = ['']
    suffix = '.jsonlines'

    featureset = ['test_input_3examples_1', 'test_input_3examples_1']
    load_featureset(dirpath, featureset, suffix)


def test_input_checking3():
    dirpath = os.path.join(_my_path, 'tests')
    featureset = ['']
    suffix = '.jsonlines'

    featureset = ['test_input_3examples_1', 'test_input_3examples_2']
    feats = load_featureset(dirpath, featureset, suffix)
    assert len(feats) == 3
