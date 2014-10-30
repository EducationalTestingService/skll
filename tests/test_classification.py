# License: BSD 3 clause
'''
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import itertools
import os
import re
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
from nose.tools import eq_, assert_almost_equal
from sklearn.base import RegressorMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.datasets.samples_generator import make_classification
from skll.data import FeatureSet
from skll.learner import Learner
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import make_classification_data, make_regression_data


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


def check_predict(model, use_feature_hashing=False):
    '''
    This tests whether predict task runs and generates the same
    number of predictions as samples in the test set. The specified
    model indicates whether to generate random regression
    or classification data.
    '''

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
    learner.train(train_fs, grid_search=False,
                  feature_hasher=(use_feature_hashing
                                  and model.__name__ != 'MultinomialNB'))

    # now make predictions on the test set
    predictions = learner.predict(test_fs, \
        feature_hasher=(use_feature_hashing and
                        model.__name__ != 'MultinomialNB'))

    # make sure we have the same number of outputs as the
    # number of test set samples
    eq_(len(predictions), test_fs.features.shape[0])


# the runner function for the prediction tests
def test_predict():
    for model, use_feature_hashing in \
            itertools.product(_ALL_MODELS, [True, False]):
        yield check_predict, model, use_feature_hashing


# the function to create data with rare classes for cross-validation
def make_rare_class_data():
    '''
    We want to create data that has five instances per class, for three classes
    and for each instance within the group of 5, there's only a single feature firing
    '''

    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 16)]
    y = [0]*5 + [1]*5 + [2]*5
    X = np.vstack([np.identity(5), np.identity(5), np.identity(5)])
    feature_names = ['f{}'.format(i) for i in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    return FeatureSet('rare-class', ids, features=features, classes=y)

def test_rare_class():
    '''
    Test cross-validation when some classes are very rare
    '''

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
    '''
    Function to create sparse data with two features always zero
    in the training set and a different one always zero in the
    test set
    '''
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
                          features=features, classes=y,
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
                         features=features, classes=y,
                         vectorizer=vectorizer)

    return train_fs, test_fs


def check_sparse_predict(use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(use_feature_hashing=use_feature_hashing)

    # train a linear SVM on the training data and evalute on the testing data
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False, feature_hasher=use_feature_hashing)
    test_score = learner.evaluate(test_fs, feature_hasher=use_feature_hashing)[1]

    expected_score = 0.51 if use_feature_hashing else 0.45
    assert_almost_equal(test_score, expected_score)


def test_sparse_predict():
    yield check_sparse_predict, False
    yield check_sparse_predict, True


def check_sparse_predict_sampler(use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(use_feature_hashing=use_feature_hashing)

    if use_feature_hashing:
        sampler = 'RBFSampler'
        sampler_parameters = {"gamma": 1.0, "n_components":50}
    else:
        sampler = 'Nystroem'
        sampler_parameters = {"gamma": 1.0, "n_components":50, "kernel":'rbf'}

    learner = Learner('LogisticRegression',
                      sampler=sampler,
                      sampler_kwargs=sampler_parameters)

    learner.train(train_fs, grid_search=False, feature_hasher=use_feature_hashing)
    test_score = learner.evaluate(test_fs, feature_hasher=use_feature_hashing)[1]

    expected_score = 0.4 if use_feature_hashing else 0.48999999999999999
    assert_almost_equal(test_score, expected_score)

def test_sparse_predict_sampler():
    yield check_sparse_predict_sampler, False
    yield check_sparse_predict_sampler, True
