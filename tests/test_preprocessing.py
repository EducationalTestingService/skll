# License: BSD 3 clause
"""
Tests related to data preprocessing options with run_experiment.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import os
import re
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
import scipy.sparse as sp
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.datasets.samples_generator import make_classification
from skll.data import FeatureSet, NDJWriter
from skll.experiments import run_configuration, _setup_config_parser
from skll.learner import Learner, SelectByMinCount
from skll.learner import _DEFAULT_PARAM_GRIDS


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
SCORE_OUTPUT_RE = re.compile(r'Objective Function Score \(Test\) = '
                             r'([\-\d\.]+)')
GRID_RE = re.compile(r'Grid Objective Score \(Train\) = ([\-\d\.]+)')
_my_dir = abspath(dirname(__file__))


def setup():
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


def fill_in_config_paths(config_template_path):
    """
    Add paths to train, test, and output directories to a given config template
    file.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path)

    task = config.get("General", "task")
    # experiment_name = config.get("General", "experiment_name")

    config.set("Input", "train_location", train_dir)

    to_fill_in = ['log', 'predictions']

    if task != 'cross_validate':
        to_fill_in.append('models')

    if task == 'evaluate' or task == 'cross_validate':
        to_fill_in.append('results')

    for d in to_fill_in:
        config.set("Output", d, join(output_dir))

    if task == 'cross_validate':
        cv_folds_location = config.get("Input", "cv_folds_location")
        if cv_folds_location:
            config.set("Input", "cv_folds_location",
                       join(train_dir, cv_folds_location))

    if task == 'predict' or task == 'evaluate':
        config.set("Input", "test_location", test_dir)

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}.cfg'.format(config_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


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
                                       'Regression.results'))) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

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
                                       'LogisticRegression.results'))) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

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
    train_fs, test_fs = make_scaling_data(
        use_feature_hashing=use_feature_hashing)

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
        expected_fmeasures = ([0.7979797979797979, 0.80198019801980192] if
                              not use_scaling else
                              [0.94883720930232551, 0.94054054054054048])
    else:
        expected_fmeasures = ([0.83962264150943389, 0.81914893617021278] if
                              not use_scaling else
                              [0.88038277511961716, 0.86910994764397898])

    for expected, actual in zip(expected_fmeasures, fmeasures):
        assert_almost_equal(expected, actual)


def test_scaling():
    yield check_scaling_features, False, False
    yield check_scaling_features, False, True
    yield check_scaling_features, True, False
    yield check_scaling_features, True, True
