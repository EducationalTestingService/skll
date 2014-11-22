# License: BSD 3 clause
"""
Tests related to classification experiments.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import glob
import itertools
import json
import os
import re
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
from nose.tools import eq_, assert_almost_equal, raises
from sklearn.base import RegressorMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.datasets.samples_generator import make_classification

from skll.data import FeatureSet
from skll.data.writers import NDJWriter
from skll.experiments import (_parse_config_file, _setup_config_parser,
                              run_configuration)
from skll.learner import Learner
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import make_classification_data, make_regression_data


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
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
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')

    if exists(join(train_dir, 'train_single_file.jsonlines')):
        os.unlink(join(train_dir, 'train_single_file.jsonlines'))

    if exists(join(test_dir, 'test_single_file.jsonlines')):
        os.unlink(join(test_dir, 'test_single_file.jsonlines'))

    if exists(join(output_dir, 'rare_class.predictions')):
        os.unlink(join(output_dir, 'rare_class.predictions'))

    for output_file in glob.glob(join(output_dir, 'train_test_single_file_*')):
        os.unlink(output_file)

    config_file = join(config_dir, 'test_single_file.cfg')
    if exists(config_file):
        os.unlink(config_file)


def fill_in_config_paths_for_single_file(config_template_path, train_file,
                                         test_file, train_directory='',
                                         test_directory=''):
    """
    Add paths to train and test files, and output directories to a given config
    template file.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path)

    task = config.get("General", "task")

    config.set("Input", "train_file", join(train_dir, train_file))
    if task == 'predict' or task == 'evaluate':
        config.set("Input", "test_file", join(test_dir, test_file))

    if train_directory:
        config.set("Input", "train_directory", join(train_dir, train_directory))

    if test_directory:
        config.set("Input", "test_directory", join(test_dir, test_directory))

    to_fill_in = ['log', 'predictions']

    if task != 'cross_validate':
        to_fill_in.append('models')

    if task == 'evaluate' or task == 'cross_validate':
        to_fill_in.append('results')

    for d in to_fill_in:
        config.set("Output", d, join(output_dir))

    if task == 'cross_validate':
        cv_folds_file = config.get("Input", "cv_folds_file")
        if cv_folds_file:
            config.set("Input", "cv_folds_file",
                       join(train_dir, cv_folds_file))

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}.cfg'.format(config_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def check_predict(model, use_feature_hashing=False):
    """
    This tests whether predict task runs and generates the same
    number of predictions as samples in the test set. The specified
    model indicates whether to generate random regression
    or classification data.
    """

    # create the random data for the given model
    if issubclass(model, RegressorMixin):
        train_fs, test_fs, _ = \
            make_regression_data(use_feature_hashing=use_feature_hashing,
                                 feature_bins=5)
    # feature hashing will not work for Naive Bayes since it requires
    # non-negative feature values
    elif model.__name__ == 'MultinomialNB':
        train_fs, test_fs = \
            make_classification_data(use_feature_hashing=False,
                                     non_negative=True)
    else:
        train_fs, test_fs = \
            make_classification_data(use_feature_hashing=use_feature_hashing,
                                     feature_bins=25)

    # create the learner with the specified model
    learner = Learner(model.__name__)

    # now train the learner on the training data and use feature hashing when
    # specified and when we are not using a Naive Bayes model
    learner.train(train_fs, grid_search=False)

    # now make predictions on the test set
    predictions = learner.predict(test_fs)

    # make sure we have the same number of outputs as the
    # number of test set samples
    eq_(len(predictions), test_fs.features.shape[0])


# the runner function for the prediction tests
def test_predict():
    for model, use_feature_hashing in \
            itertools.product(_ALL_MODELS, [True, False]):
        yield check_predict, model, use_feature_hashing


# the function to create data with rare labels for cross-validation
def make_rare_class_data():
    """
    We want to create data that has five instances per class, for three labels
    and for each instance within the group of 5, there's only a single feature
    firing
    """

    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 16)]
    y = [0] * 5 + [1] * 5 + [2] * 5
    X = np.vstack([np.identity(5), np.identity(5), np.identity(5)])
    feature_names = ['f{}'.format(i) for i in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    return FeatureSet('rare-class', ids, features=features, labels=y)


def test_rare_class():
    """
    Test cross-validation when some labels are very rare
    """

    rare_class_fs = make_rare_class_data()
    prediction_prefix = join(_my_dir, 'output', 'rare_class')
    learner = Learner('LogisticRegression')
    learner.cross_validate(rare_class_fs,
                           grid_objective='unweighted_kappa',
                           prediction_prefix=prediction_prefix)

    with open(prediction_prefix + '.predictions', 'r') as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        pred = [row[1] for row in reader]

        eq_(len(pred), 15)


def make_sparse_data(use_feature_hashing=False):
    """
    Function to create sparse data with two features always zero
    in the training set and a different one always zero in the
    test set
    """
    # Create training data
    X, y = make_classification(n_samples=500, n_features=3,
                               n_informative=3, n_redundant=0,
                               n_classes=2, random_state=1234567890)

    # we need features to be non-negative since we will be
    # using naive bayes laster
    X = np.abs(X)

    # make sure that none of the features are zero
    X[np.where(X == 0)] += 1

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 501)]

    # create a list of dictionaries as the features
    # with f1 and f5 always 0
    feature_names = ['f{}'.format(n) for n in range(1, 6)]
    features = []
    for row in X:
        row = [0] + row.tolist() + [0]
        features.append(dict(zip(feature_names, row)))

    # use a FeatureHasher if we are asked to do feature hashing
    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    train_fs = FeatureSet('train_sparse', ids,
                          features=features, labels=y,
                          vectorizer=vectorizer)

    # now create the test set with f4 always 0 but nothing else
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=4, n_redundant=0,
                               n_classes=2, random_state=1234567890)
    X = np.abs(X)
    X[np.where(X == 0)] += 1
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 101)]

    # create a list of dictionaries as the features
    # with f4 always 0
    feature_names = ['f{}'.format(n) for n in range(1, 6)]
    features = []
    for row in X:
        row = row.tolist()
        row = row[:3] + [0] + row[3:]
        features.append(dict(zip(feature_names, row)))

    test_fs = FeatureSet('test_sparse', ids,
                         features=features, labels=y,
                         vectorizer=vectorizer)

    return train_fs, test_fs


def check_sparse_predict(use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(
        use_feature_hashing=use_feature_hashing)

    # train a linear SVM on the training data and evalute on the testing data
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False)
    test_score = learner.evaluate(test_fs)[1]
    expected_score = 0.51 if use_feature_hashing else 0.45
    assert_almost_equal(test_score, expected_score)


def test_sparse_predict():
    yield check_sparse_predict, False
    yield check_sparse_predict, True


def check_sparse_predict_sampler(use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(
        use_feature_hashing=use_feature_hashing)

    if use_feature_hashing:
        sampler = 'RBFSampler'
        sampler_parameters = {"gamma": 1.0, "n_components": 50}
    else:
        sampler = 'Nystroem'
        sampler_parameters = {"gamma": 1.0, "n_components": 50,
                              "kernel": 'rbf'}

    learner = Learner('LogisticRegression',
                      sampler=sampler,
                      sampler_kwargs=sampler_parameters)

    learner.train(train_fs, grid_search=False)
    test_score = learner.evaluate(test_fs)[1]

    expected_score = 0.4 if use_feature_hashing else 0.48999999999999999
    assert_almost_equal(test_score, expected_score)


def test_sparse_predict_sampler():
    yield check_sparse_predict_sampler, False
    yield check_sparse_predict_sampler, True


def make_single_file_featureset_data():
    """
    Write a training file and a test file for tests that check whether
    specifying train_file and test_file actually works.
    """
    train_fs, test_fs = make_classification_data(num_examples=600,
                                                 train_test_ratio=0.8,
                                                 num_labels=2,
                                                 num_features=3,
                                                 non_negative=False)

    # Write training feature set to a file
    train_path = join(_my_dir, 'train', 'train_single_file.jsonlines')
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = join(_my_dir, 'test', 'test_single_file.jsonlines')
    writer = NDJWriter(test_path, test_fs)
    writer.write()


def test_train_file_test_file():
    """
    Test that train_file and test_file experiments work
    """
    # Create data files
    make_single_file_featureset_data()

    # Run experiment
    config_path = fill_in_config_paths_for_single_file(join(_my_dir, "configs",
                                                            "test_single_file"
                                                            ".template.cfg"),
                                                       join(_my_dir, 'train',
                                                            'train_single_file'
                                                            '.jsonlines'),
                                                       join(_my_dir, 'test',
                                                            'test_single_file.'
                                                            'jsonlines'))
    run_configuration(config_path, quiet=True)

    # Check results
    with open(join(_my_dir, 'output', ('train_test_single_file_train_train_'
                                       'single_file.jsonlines_test_test_single'
                                       '_file.jsonlines_RandomForestClassifier'
                                       '.results.json'))) as f:
        result_dict = json.load(f)[0]

    assert_almost_equal(result_dict['score'], 0.925)


@raises(ValueError)
def test_train_file_and_train_directory():
    """
    Test that train_file + train_directory = ValueError
    """
    # Run experiment
    config_path = fill_in_config_paths_for_single_file(join(_my_dir, "configs",
                                                            "test_single_file"
                                                            ".template.cfg"),
                                                       join(_my_dir, 'train',
                                                            'train_single_file'
                                                            '.jsonlines'),
                                                       join(_my_dir, 'test',
                                                            'test_single_file.'
                                                            'jsonlines'),
                                                       train_directory='foo')
    _parse_config_file(config_path)


@raises(ValueError)
def test_test_file_and_test_directory():
    """
    Test that test_file + test_directory = ValueError
    """
    # Run experiment
    config_path = fill_in_config_paths_for_single_file(join(_my_dir, "configs",
                                                            "test_single_file"
                                                            ".template.cfg"),
                                                       join(_my_dir, 'train',
                                                            'train_single_file'
                                                            '.jsonlines'),
                                                       join(_my_dir, 'test',
                                                            'test_single_file.'
                                                            'jsonlines'),
                                                       test_directory='foo')
    _parse_config_file(config_path)
