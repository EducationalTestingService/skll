# License: BSD 3 clause
"""
Tests related to data preprocessing options with run_experiment.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import json
import os
import re
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_array_equal, assert_almost_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.datasets.samples_generator import make_classification
from skll.data import FeatureSet, NDJWriter
from skll.experiments import run_configuration
from skll.learner import Learner, SelectByMinCount
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import fill_in_config_paths


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
SCORE_OUTPUT_RE = re.compile(r'Objective Function Score \(Test\) = '
                             r'([\-\d\.]+)')
_my_dir = abspath(dirname(__file__))


def setup():
    """
    Create necessary directories for testing.
    """
    train_dir = join(_my_dir, 'train')
    if not exists(train_dir):
        os.makedirs(train_dir)
    test_dir = join(_my_dir, 'test')
    if not exists(test_dir):
        os.makedirs(test_dir)
    output_dir = join(_my_dir, 'output')
    if not exists(output_dir):
        os.makedirs(output_dir)


def tearDown():
    """
    Clean up after tests.
    """
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')

    for output_file in glob.glob(join(output_dir, 'test_class_map_*')):
        os.unlink(output_file)

    if exists(join(train_dir, 'test_class_map.jsonlines')):
        os.unlink(join(train_dir, 'test_class_map.jsonlines'))

    if exists(join(test_dir, 'test_class_map.jsonlines')):
        os.unlink(join(test_dir, 'test_class_map.jsonlines'))

    config_files = ['test_class_map.cfg',
                    'test_class_map_feature_hasher.cfg']
    for cf in config_files:
        if exists(join(config_dir, cf)):
            os.unlink(join(config_dir, cf))


def test_SelectByMinCount():
    """ Test SelectByMinCount feature selector """
    m2 = [[0.001, 0.0, 0.0, 0.0],
          [0.00001, -2.0, 0.0, 0.0],
          [0.001, 0.0, 0.0, 4.0],
          [0.0101, -200.0, 0.0, 0.0]]

    # default should keep all nonzero features (i.e. ones that appear 1+ times)
    feat_selector = SelectByMinCount()
    expected = np.array([[0.001, 0.0, 0.0],
                         [0.00001, -2.0, 0.0],
                         [0.001, 0.0, 4.0],
                         [0.0101, -200.0, 0.0]])
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(feat_selector.fit_transform(
        sp.csr_matrix(m2)).todense(),
        expected)

    # keep features that happen 2+ times
    feat_selector = SelectByMinCount(min_count=2)
    expected = np.array([[0.001, 0.0],
                         [0.00001, -2.0],
                         [0.001, 0.0],
                         [0.0101, -200.0]])
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(
        feat_selector.fit_transform(sp.csr_matrix(m2)).todense(),
        expected)

    # keep features that happen 3+ times
    feat_selector = SelectByMinCount(min_count=3)
    expected = np.array([[0.001], [0.00001], [0.001], [0.0101]])
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(
        feat_selector.fit_transform(sp.csr_matrix(m2)).todense(),
        expected)


def make_class_map_data():
    # Create training file
    train_path = join(_my_dir, 'train', 'test_class_map.jsonlines')
    ids = []
    labels = []
    features = []
    class_names = ['beagle', 'cat', 'dachsund', 'cat']
    for i in range(1, 101):
        y = class_names[i % 4]
        ex_id = "{}{}".format(y, i)
        # note that f1 and f5 are missing in all instances but f4 is not
        x = {"f2": i + 1, "f3": i + 2, "f4": i + 5}
        ids.append(ex_id)
        labels.append(y)
        features.append(x)
    train_fs = FeatureSet('train_class_map', ids, features=features,
                          labels=labels)
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Create test file
    test_path = join(_my_dir, 'test', 'test_class_map.jsonlines')
    ids = []
    labels = []
    features = []
    for i in range(1, 51):
        y = class_names[i % 4]
        ex_id = "{}{}".format(y, i)
        # f1 and f5 are not missing in any instances here but f4 is
        x = {"f1": i, "f2": i + 2, "f3": i % 10, "f5": i * 2}
        ids.append(ex_id)
        labels.append(y)
        features.append(x)
    test_fs = FeatureSet('test_class_map', ids, features=features,
                         labels=labels)
    writer = NDJWriter(test_path, test_fs)
    writer.write()


def test_class_map():
    """
    Test class maps
    """

    make_class_map_data()

    config_template_path = join(
        _my_dir,
        'configs',
        'test_class_map.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output', ('test_class_map_test_class_map_Logistic'
                                       'Regression.results.json'))) as f:
        outd = json.loads(f.read())
        # outstr = f.read()
        # logistic_result_score = float(
            # SCORE_OUTPUT_RE.search(outstr).groups()[0])
        logistic_result_score = outd[0]['score']

    assert_almost_equal(logistic_result_score, 0.5)


def test_class_map_feature_hasher():
    """
    Test class maps with feature hashing
    """

    make_class_map_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_class_map_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output', ('test_class_map_test_class_map_'
                                       'LogisticRegression.results.'
                                       'json'))) as f:
        # outstr = f.read()
        outd = json.loads(f.read())
        # logistic_result_score = float(
        #     SCORE_OUTPUT_RE.search(outstr).groups()[0])
        logistic_result_score = outd[0]['score']

    assert_almost_equal(logistic_result_score, 0.5)


def make_scaling_data(use_feature_hashing=False):

    X, y = make_classification(n_samples=1000, n_classes=2,
                               n_features=5, n_informative=5,
                               n_redundant=0, random_state=1234567890)

    # we want to arbitrary scale the various features to test the scaling
    scalers = np.array([1, 10, 100, 1000, 10000])
    X = X * scalers

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 1001)]

    # create a list of dictionaries as the features
    feature_names = ['f{}'.format(n) for n in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # split everything into training and testing portions
    train_features, test_features = features[:800], features[800:]
    train_y, test_y = y[:800], y[800:]
    train_ids, test_ids = ids[:800], ids[800:]

    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    train_fs = FeatureSet('train_scaling', train_ids,
                          features=train_features, labels=train_y,
                          vectorizer=vectorizer)
    test_fs = FeatureSet('test_scaling', test_ids,
                         features=test_features, labels=test_y,
                         vectorizer=vectorizer)

    return (train_fs, test_fs)


def check_scaling_features(use_feature_hashing=False, use_scaling=False):
    train_fs, test_fs = make_scaling_data(use_feature_hashing=use_feature_hashing)

    # create a Linear SVM with the value of scaling as specified
    feature_scaling = 'both' if use_scaling else 'none'
    learner = Learner('SGDClassifier', feature_scaling=feature_scaling,
                      pos_label_str=1)

    # train the learner on the training set and test on the testing set
    learner.train(train_fs)
    test_output = learner.evaluate(test_fs)
    fmeasures = [test_output[2][0]['F-measure'],
                 test_output[2][1]['F-measure']]

    # these are the expected values of the f-measures, sorted
    if not use_feature_hashing:
        expected_fmeasures = ([0.77319587628865982, 0.78640776699029125] if
                              not use_scaling else
                              [0.94930875576036866, 0.93989071038251359])
    else:
        expected_fmeasures = ([0.42774566473988435, 0.5638766519823788] if
                              not use_scaling else
                              [0.87323943661971837, 0.85561497326203206])

    assert_almost_equal(expected_fmeasures, fmeasures)


def test_scaling():
    yield check_scaling_features, False, False
    yield check_scaling_features, False, True
    yield check_scaling_features, True, False
    yield check_scaling_features, True, True
