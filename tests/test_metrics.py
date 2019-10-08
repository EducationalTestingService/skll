# License: BSD 3 clause
"""
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

import json
import os
import re

from glob import glob
from itertools import product
from os.path import abspath, dirname, exists, join

from nose.tools import raises
from numpy.testing import assert_almost_equal

from scipy.stats import kendalltau, pearsonr, spearmanr

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as sk_shuffle

from skll import run_configuration
from skll.config import _load_cv_folds
from skll.data import CSVReader, FeatureSet
from skll.learner import FilteredLeaveOneGroupOut, Learner
from skll.learner import _DEFAULT_PARAM_GRIDS
from skll.metrics import (_CLASSIFICATION_ONLY_METRICS,
                          _CORRELATION_METRICS,
                          _REGRESSION_ONLY_METRICS,
                          _UNWEIGHTED_KAPPA_METRICS,
                          _WEIGHTED_KAPPA_METRICS,
                          kappa,
                          use_score_func)

from tests.utils import (fill_in_config_options,
                         make_classification_data,
                         make_regression_data)

_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
_CLF_METRICS_REGEXP = re.compile(r'clf_metrics_(true|false)_titanic_LogisticRegression_(.*?).results.json')

# Inputs derived from Ben Hamner's unit tests for his
# kappa implementation as part of the ASAP competition
_KAPPA_INPUTS = [([1, 2, 3], [1, 2, 3]),
                 ([1, 2, 1], [1, 2, 2]),
                 ([1, 2, 3, 1, 2, 2, 3], [1, 2, 3, 1, 2, 3, 2]),
                 ([1, 2, 3, 3, 2, 1], [1, 1, 1, 2, 2, 2]),
                 ([-1, 0, 1, 2], [-1, 0, 0, 2]),
                 ([5, 6, 7, 8], [5, 6, 6, 8]),
                 ([1, 1, 2, 2], [3, 3, 4, 4]),
                 ([1, 1, 3, 3], [2, 2, 4, 4]),
                 ([1, 1, 4, 4], [2, 2, 3, 3]),
                 ([1, 2, 4], [1, 2, 4]),
                 ([1, 2, 4], [1, 2, 2])]

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


# def tearDown():
#     """
#     Clean up any files/directories created during testing.
#     """
#     config_dir = join(_my_dir, 'configs')
#     output_dir = join(_my_dir, 'output')

#     # delete the config files we created
#     for config_file in glob(join(config_dir,
#                                  'test_other_metrics_for_classification_prob_*.cfg')):
#         os.unlink(config_file)

#     for output_file in glob(join(output_dir, 'clf_metrics_*')):
#         os.unlink(output_file)


def check_kappa(y_true, y_pred, weights, allow_off_by_one, expected):
    assert_almost_equal(kappa(y_true, y_pred, weights=weights,
                              allow_off_by_one=allow_off_by_one), expected)


def test_quadratic_weighted_kappa():
    outputs = [1.0, 0.4, 0.75, 0.0, 0.9, 0.9, 0.11111111, 0.6666666666667, 0.6,
               1.0, 0.4]

    for (y_true, y_pred), expected in zip(_KAPPA_INPUTS, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', False, expected

    # Swap y_true and y_pred and test again
    for (y_pred, y_true), expected in zip(_KAPPA_INPUTS, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', False, expected


def test_allow_off_by_one_qwk():
    outputs = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.3333333333333333, 1.0, 1.0,
               1.0, 0.5]

    for (y_true, y_pred), expected in zip(_KAPPA_INPUTS, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', True, expected

    # Swap y_true and y_pred and test again
    for (y_pred, y_true), expected in zip(_KAPPA_INPUTS, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', True, expected


def test_linear_weighted_kappa():
    outputs = [1.0, 0.4, 0.65, 0.0, 0.8, 0.8, 0.0, 0.3333333, 0.3333333, 1.0,
               0.4]

    for (y_true, y_pred), expected in zip(_KAPPA_INPUTS, outputs):
        yield check_kappa, y_true, y_pred, 'linear', False, expected

    # Swap y_true and y_pred and test again
    for (y_pred, y_true), expected in zip(_KAPPA_INPUTS, outputs):
        yield check_kappa, y_true, y_pred, 'linear', False, expected


def test_unweighted_kappa():
    outputs = [1.0, 0.4, 0.5625, 0.0, 0.6666666666667, 0.6666666666667,
               0.0, 0.0, 0.0, 1.0, 0.5]

    for (y_true, y_pred), expected in zip(_KAPPA_INPUTS, outputs):
        yield check_kappa, y_true, y_pred, None, False, expected

    # Swap y_true and y_pred and test again
    for (y_pred, y_true), expected in zip(_KAPPA_INPUTS, outputs):
        yield check_kappa, y_true, y_pred, None, False, expected


@raises(ValueError)
def test_invalid_weighted_kappa():
    kappa([1, 2, 1], [1, 2, 1], weights='invalid', allow_off_by_one=False)
    kappa([1, 2, 1], [1, 2, 1], weights='invalid', allow_off_by_one=True)


@raises(ValueError)
def test_invalid_lists_kappa():
    kappa(['a', 'b', 'c'], ['a', 'b', 'c'])


@raises(ValueError)
def check_invalid_regression_grid_objective(learner, grid_objective):
    """
    Checks whether the grid objective function is valid for this regressor
    """
    (train_fs, _, _) = make_regression_data()
    clf = Learner(learner)
    clf.train(train_fs, grid_objective=grid_objective)


def test_invalid_regression_grid_objective():
    for learner in ['AdaBoostRegressor', 'BayesianRidge',
                    'DecisionTreeRegressor', 'ElasticNet',
                    'GradientBoostingRegressor', 'HuberRegressor',
                    'KNeighborsRegressor', 'Lars', 'Lasso',
                    'LinearRegression', 'MLPRegressor',
                    'RandomForestRegressor', 'RANSACRegressor',
                    'Ridge', 'LinearSVR', 'SVR', 'SGDRegressor',
                    'TheilSenRegressor']:
        for metric in _CLASSIFICATION_ONLY_METRICS:
            yield check_invalid_regression_grid_objective, learner, metric


# @raises(ValueError)
def check_invalid_classification_grid_objective(learner, fs, grid_objective):
    """
    Checks whether the grid objective function is valid for this classifier
    """
    clf = Learner(learner)
    clf.train(fs, grid_objective=grid_objective)


def test_invalid_classification_grid_objective():
    for (learner,
         num_labels,
         label_type) in product(['AdaBoostClassifier', 'DecisionTreeClassifier',
                                 'GradientBoostingClassifier', 'KNeighborsClassifier',
                                 'MLPClassifier', 'MultinomialNB',
                                 'RandomForestClassifier', 'LogisticRegression',
                                 'LinearSVC', 'SVC', 'SGDClassifier'],
                                [2, 4],
                                ['string', 'integer']):

        bad_objectives = set(_REGRESSION_ONLY_METRICS)
        if label_type == 'string':
            bad_objectives.update(_WEIGHTED_KAPPA_METRICS, _CORRELATION_METRICS)

        string_label_list = ['yes', 'no'] if num_labels == 2 else ['A', 'B', 'C', 'D']

        train_fs, _ = make_classification_data(num_labels=num_labels,
                                               string_label_list=string_label_list)
        for metric in bad_objectives:
            yield check_invalid_classification_grid_objective, learner, train_fs, metric


def check_other_metrics_for_classification(results_json_path):

    # define the directories we need
    train_dir = join(_my_dir, 'train')

    # get the metric name and probability status from the path name
    (use_probabilities, metric_name) = re.findall(_CLF_METRICS_REGEXP,
                                                  results_json_path)[0]

    # load the json file into an object and get the grid search results
    results_obj = json.load(open(results_json_path, 'r'))
    grid_search_cv_results = results_obj['grid_search_cv_results']

    # load in the featureset to get the feature matrix (X) and the
    # labels array (y) we need to pass to scikit-learn; we also need
    # to shuffle after we load in the matrix since that's what SKLL does
    fs = CSVReader.for_path(join(train_dir, 'titanic_combined_features.csv'),
                            id_col='PassengerId',
                            label_col='Survived').read()
    ids, labels, features = sk_shuffle(fs.ids,
                                       fs.labels,
                                       fs.features,
                                       random_state=123456789)
    shuffled_fs = FeatureSet(fs.name,
                             ids,
                             labels=labels,
                             features=features,
                             vectorizer=fs.vectorizer)
    X = shuffled_fs.features
    y = shuffled_fs.labels

    # instantiate and save two different sklearn LogisticRegression
    # models for each value of C that was in the SKLL grid and with the
    # same other fixed parameters that we used for SKLL
    models_with_C_values = {}
    for param_value in grid_search_cv_results['params']:
        model_kwargs = param_value
        model_kwargs.update({'max_iter': 1000,
                             'solver': 'liblinear',
                             'multi_class': 'auto',
                             'random_state': 123456789})
        sklearn_learner = LogisticRegression(**model_kwargs)
        models_with_C_values[param_value['C']] = sklearn_learner

    # now let's split the featureset the same way SKLL would have
    # done using the folds file
    folds_dict = _load_cv_folds(join(train_dir, 'titanic_grid_search_folds.csv'))
    dummy_label = next(iter(folds_dict.values()))
    fold_groups = [folds_dict.get(curr_id, dummy_label) for curr_id in shuffled_fs.ids]
    kfold = FilteredLeaveOneGroupOut(folds_dict, shuffled_fs.ids)
    fold_train_test_ids = list(kfold.split(shuffled_fs.features, shuffled_fs.labels, fold_groups))

    # generate predictions on the test split of the each fold, and
    # then compute the objectives appropriately (using either the
    # labels or the probabilities) and store them for comparison
    metric_values_dict = {}

    # compute values for each fold
    for fold_id in [0, 1]:

        metric_values_dict[fold_id] = []

        fold_train_ids = fold_train_test_ids[fold_id][0]
        fold_test_ids = fold_train_test_ids[fold_id][1]

        X_fold_train = X[fold_train_ids, :]
        X_fold_test = X[fold_test_ids, :]

        y_fold_train = y[fold_train_ids, ]
        y_fold_test = y[fold_test_ids, ]

        # iterate over the two trained sklearn models;
        # one for each point on the C grid
        for C_value in models_with_C_values:

            sklearn_learner = models_with_C_values[C_value]
            sklearn_learner.fit(X_fold_train, y_fold_train)

            # if the SKLL experiment used probabilities and
            # then compute the probabilties with sklearn too
            if use_probabilities == 'true' and metric_name in _CORRELATION_METRICS:
                yhat_fold_test = sklearn_learner.predict_proba(X_fold_test)[:, 1]
            else:
                yhat_fold_test = sklearn_learner.predict(X_fold_test)

            # compute the metric value; for correlation metrics
            # we use `pearsonr`, `spearmanr`, directly from scipy
            # but for kappa metrics, we have to use SKLL versions
            if metric_name == 'pearson':
                metric_value = pearsonr(y_fold_test, yhat_fold_test)[0]
            elif metric_name == 'spearman':
                metric_value = spearmanr(y_fold_test, yhat_fold_test)[0]
            elif metric_name == 'kendall_tau':
                metric_value = kendalltau(y_fold_test, yhat_fold_test)[0]
            else:
                metric_value = use_score_func(metric_name,
                                              y_fold_test,
                                              yhat_fold_test)

            # save the computed metric values for the fold
            metric_values_dict[fold_id].append(metric_value)

    # compare SKLL grid-search values with sklearn values for each fold
    assert_almost_equal(grid_search_cv_results['split0_test_score'],
                        metric_values_dict[0])

    assert_almost_equal(grid_search_cv_results['split1_test_score'],
                        metric_values_dict[1])


def test_other_metrics_for_classification():

    # Test that the metrics that aren't directly meant for classifiction yield
    # expected results. For example, correlations with/without probabilities etc.
    # Here's how the test works:

    # 1. We first run probabilistic classification experiments with:
    #    (a) each metrics as the tuning objective
    #    (b) only two points on the grid for a single hyperparametr
    #    (b) 2 externally specified grid search folds
    #    (c) probability either true or false
    #
    # 2. We get the specific values of the objective on the test split
    #    for each point on the grid.
    #
    # 3. We run a separate experiment in scikit-learn space where:
    #    (a) for of the two points on the grid, we explicitly train
    #        the same model on the train split
    #    (b) compute the two trained model's predictions on the respective
    #        test splits
    #    (c) compute the two values of the objective using these predictions
    #
    # 4. We then compare values in 2 and 3 to verify that they are almost equal.

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    config_template_path = join(_my_dir,
                                'configs',
                                'test_other_metrics_for_classification.template.cfg')

    metrics_to_test = set(_CORRELATION_METRICS)
    metrics_to_test.update(_UNWEIGHTED_KAPPA_METRICS, _WEIGHTED_KAPPA_METRICS)

    static_values_dict = {'train_file': join(train_dir, "titanic_combined_features.csv"),
                          'task': 'train',
                          'log': output_dir,
                          'models': output_dir,
                          'objectives': "{}".format(list(metrics_to_test)),
                          'folds_file': join(train_dir,
                                             "titanic_grid_search_folds.csv"),
                          'results': output_dir}

    results_json_paths = []
    for use_probabilities in ['true', 'false']:

        values_to_fill_dict = static_values_dict.copy()
        values_to_fill_dict.update({'experiment_name': 'clf_metrics_{}'.format(use_probabilities), 'probability': use_probabilities})
        config_path = fill_in_config_options(config_template_path,
                                             values_to_fill_dict,
                                             'prob_{}'.format(use_probabilities),
                                             good_probability_option=True)

        results_json_path = run_configuration(config_path, quiet=True, local=True)
        results_json_paths.extend(results_json_path)

    for results_json_path in results_json_paths:
        yield check_other_metrics_for_classification, results_json_path
