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
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
from nose.tools import eq_, assert_almost_equal, raises

from skll.data import FeatureSet
from skll.data.writers import NDJWriter
from skll.config import _parse_config_file
from skll.experiments import run_configuration
from skll.learner import Learner
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import (make_classification_data, make_regression_data,
                   make_sparse_data, fill_in_config_paths_for_single_file)


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

    if exists(join(output_dir, 'float_class.predictions')):
        os.unlink(join(output_dir, 'float_class.predictions'))

    for output_file in glob.glob(join(output_dir, 'train_test_single_file_*')):
        os.unlink(output_file)

    config_file = join(config_dir, 'test_single_file.cfg')
    if exists(config_file):
        os.unlink(config_file)


def check_predict(model, use_feature_hashing=False):
    """
    This tests whether predict task runs and generates the same
    number of predictions as samples in the test set. The specified
    model indicates whether to generate random regression
    or classification data.
    """

    # create the random data for the given model
    if model._estimator_type == 'regressor':
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


def check_sparse_predict(learner_name, expected_score, use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(
        use_feature_hashing=use_feature_hashing)

    # train a logistic regression classifier on the training
    # data and evalute on the testing data
    learner = Learner(learner_name)
    learner.train(train_fs, grid_search=False)
    test_score = learner.evaluate(test_fs)[1]
    assert_almost_equal(test_score, expected_score)


def test_sparse_predict():
    for learner_name, expected_scores in zip(['LogisticRegression',
                                              'DecisionTreeClassifier',
                                              'RandomForestClassifier',
                                              'AdaBoostClassifier',
                                              'MultinomialNB',
                                              'KNeighborsClassifier'],
                                             [(0.45, 0.51), (0.5, 0.51),
                                              (0.46, 0.46), (0.5, 0.5),
                                              (0.44, 0), (0.51, 0.43)]):
        yield check_sparse_predict, learner_name, expected_scores[0], False
        if learner_name != 'MultinomialNB':
            yield check_sparse_predict, learner_name, expected_scores[1], True


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

    expected_score = 0.44 if use_feature_hashing else 0.48999999999999999
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

    # Check results for objective functions ["accuracy", "f1"]

    # objective function accuracy
    with open(join(_my_dir, 'output', ('train_test_single_file_train_train_'
                                       'single_file.jsonlines_test_test_single'
                                       '_file.jsonlines_RandomForestClassifier'
                                       '_accuracy.results.json'))) as f:
        result_dict = json.load(f)[0]
    assert_almost_equal(result_dict['score'], 0.925)

    # objective function f1
    with open(join(_my_dir, 'output', ('train_test_single_file_train_train_'
                                       'single_file.jsonlines_test_test_single'
                                       '_file.jsonlines_RandomForestClassifier'
                                       '_f1.results.json'))) as f:
        result_dict = json.load(f)[0]
    assert_almost_equal(result_dict['score'], 0.928)


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


def check_adaboost_predict(base_estimator, algorithm, expected_score):
    train_fs, test_fs = make_sparse_data()

    # train an AdaBoostClassifier on the training data and evalute on the
    # testing data
    learner = Learner('AdaBoostClassifier', model_kwargs={'base_estimator': base_estimator,
                                                          'algorithm': algorithm})
    learner.train(train_fs, grid_search=False)
    test_score = learner.evaluate(test_fs)[1]
    assert_almost_equal(test_score, expected_score)


def test_adaboost_predict():
    for base_estimator_name, algorithm, expected_score in zip(['MultinomialNB',
                                                               'DecisionTreeClassifier',
                                                               'SGDClassifier',
                                                               'SVC'],
                                                              ['SAMME.R', 'SAMME.R',
                                                               'SAMME', 'SAMME'],
                                                              [0.45, 0.5, 0.45, 0.43]):
        yield check_adaboost_predict, base_estimator_name, algorithm, expected_score


def check_results_with_unseen_labels(res, n_labels, new_label_list):
    (confusion_matrix,
     score,
     result_dict,
     model_params,
     grid_score) = res

    # check that the new label is included into the results
    for output in [confusion_matrix, result_dict]:
        eq_(len(output), n_labels)

    # check that all metrics for new label are 0
    for label in new_label_list:
        for metric in ['Precision', 'Recall', 'F-measure']:
            eq_(result_dict[label][metric], 0)


def test_new_labels_in_test_set():
    """
    Test classification experiment with an unseen label in the test set.
    """
    train_fs, test_fs = make_classification_data(num_labels=3,
                                                 train_test_ratio=0.8)
    # add new labels to the test set
    test_fs.labels[-3:] = 3

    learner = Learner('SVC')
    learner.train(train_fs, grid_search=False)
    res = learner.evaluate(test_fs)
    yield check_results_with_unseen_labels, res, 4, [3]
    yield assert_almost_equal, res[1], 0.3


def test_new_labels_in_test_set_change_order():
    """
    Test classification with an unseen label in the test set when the new label falls between the existing labels
    """
    train_fs, test_fs = make_classification_data(num_labels=3,
                                                 train_test_ratio=0.8)
    # change train labels to create a gap
    train_fs.labels = train_fs.labels*10
    # add new test labels
    test_fs.labels = test_fs.labels*10
    test_fs.labels[-3:] = 15

    learner = Learner('SVC')
    learner.train(train_fs, grid_search=False)
    res = learner.evaluate(test_fs)
    yield check_results_with_unseen_labels, res, 4, [15]
    yield assert_almost_equal, res[1], 0.3


def test_all_new_labels_in_test():
    """
    Test classification with all labels in test set unseen
    """
    train_fs, test_fs = make_classification_data(num_labels=3,
                                                 train_test_ratio=0.8)
    # change all test labels
    test_fs.labels = test_fs.labels+3

    learner = Learner('SVC')
    learner.train(train_fs, grid_search=False)
    res = learner.evaluate(test_fs)
    yield check_results_with_unseen_labels, res, 6, [3, 4, 5]
    yield assert_almost_equal, res[1], 0


# the function to create data with labels that look like floats
def make_float_class_data():
    """
    We want to create data that has labels that look like
    floats to make sure they are preserved correctly
    """

    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 76)]
    y = [1.2] * 25 + [1.5] * 25 + [1.8] * 25
    X = np.vstack([np.identity(25), np.identity(25), np.identity(25)])
    feature_names = ['f{}'.format(i) for i in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    return FeatureSet('float-classes', ids, features=features, labels=y)


def test_float_classes():
    """
    Test classification with labels that look like floats
    Make sure that they have been converted to strings as expected
    """

    float_class_fs = make_float_class_data()
    prediction_prefix = join(_my_dir, 'output', 'float_class')
    learner = Learner('LogisticRegression')
    learner.cross_validate(float_class_fs,
                           grid_objective='accuracy',
                           prediction_prefix=prediction_prefix)

    with open(prediction_prefix + '.predictions', 'r') as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        pred = [row[1] for row in reader]
        for p in pred:
            assert p in ['1.2', '1.5', '1.8']


