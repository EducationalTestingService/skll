# License: BSD 3 clause
"""
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import os
import re
import warnings

from glob import glob
from itertools import product
from os.path import abspath, dirname, join, exists

from nose.tools import eq_, assert_almost_equal

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import assert_greater, assert_less

from skll.data import FeatureSet, NDJWriter
from skll.config import _setup_config_parser
from skll.experiments import run_configuration
from skll.learner import Learner
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import make_regression_data, fill_in_config_paths_for_fancy_output

_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
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
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')

    train_file = join(train_dir, 'fancy_train.jsonlines')
    if exists(train_file):
        os.unlink(train_file)

    test_file = join(test_dir, 'fancy_test.jsonlines')
    if exists(test_file):
        os.unlink(test_file)

    for output_file in glob(join(output_dir, 'regression_fancy_output_*')):
        os.unlink(output_file)

    config_file = join(config_dir, 'test_regression_fancy_output.cfg')
    if exists(config_file):
        os.unlink(config_file)

    config_file = join(config_dir, 'test_int_labels_cv.cfg')
    if exists(config_file):
        os.unlink(config_file)


# a utility function to check rescaling for linear models
def check_rescaling(name, grid_search=False):

    train_fs, test_fs, _ = make_regression_data(num_examples=2000,
                                                sd_noise=4,
                                                num_features=3)

    # instantiate the given learner and its rescaled counterpart
    learner = Learner(name)
    rescaled_learner = Learner('Rescaled' + name)

    # train both the regular regressor and the rescaled regressor
    # with and without using grid search
    if grid_search:
        learner.train(train_fs, grid_objective='pearson')
        rescaled_learner.train(train_fs, grid_objective='pearson')
    else:
        learner.train(train_fs, grid_search=False)
        rescaled_learner.train(train_fs, grid_search=False)

    # now generate both sets of predictions on the test feature set
    predictions = learner.predict(test_fs)
    rescaled_predictions = rescaled_learner.predict(test_fs)

    # ... and on the training feature set
    train_predictions = learner.predict(train_fs)
    rescaled_train_predictions = rescaled_learner.predict(train_fs)

    # make sure that both sets of correlations are close to perfectly
    # correlated, since the only thing different is that one set has been
    # rescaled
    assert_almost_equal(pearsonr(predictions, rescaled_predictions)[0],
                        1.0,
                        places=3)

    # make sure that the standard deviation of the rescaled test set
    # predictions is higher than the standard deviation of the regular test set
    # predictions
    p_std = np.std(predictions)
    rescaled_p_std = np.std(rescaled_predictions)
    assert_greater(rescaled_p_std, p_std)

    # make sure that the standard deviation of the rescaled predictions
    # on the TRAINING set (not the TEST) is closer to the standard
    # deviation of the training set labels than the standard deviation
    # of the regular predictions.
    train_y_std = np.std(train_fs.labels)
    train_p_std = np.std(train_predictions)
    rescaled_train_p_std = np.std(rescaled_train_predictions)
    assert_less(abs(rescaled_train_p_std - train_y_std),
                      abs(train_p_std - train_y_std))


def test_rescaling():
    for regressor_name in ['BayesianRidge',
                           'ElasticNet',
                           'HuberRegressor',
                           'Lars',
                           'Lasso',
                           'LinearRegression',
                           'LinearSVR',
                           'RANSACRegressor',
                           'Ridge',
                           'SGDRegressor',
                           'SVR',
                           'TheilSenRegressor']:
        for do_grid_search in [True, False]:
            yield check_rescaling, regressor_name, do_grid_search


# the utility function to run the linear regession tests
def check_linear_models(name,
                        use_feature_hashing=False,
                        use_rescaling=False):

    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        (train_fs,
         test_fs,
         weightdict) = make_regression_data(num_examples=5000,
                                            num_features=10,
                                            use_feature_hashing=True,
                                            feature_bins=5)
    else:
        train_fs, test_fs, weightdict = make_regression_data(num_examples=2000,
                                                             num_features=3)

    # create the learner
    if use_rescaling:
        name = 'Rescaled' + name
    learner = Learner(name)

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_objective='pearson')

    # make sure that the weights are close to the weights
    # that we got from make_regression_data. Take the
    # ceiling before comparing since just comparing
    # the ceilings should be enough to make sure nothing
    # catastrophic happened. Note though that we cannot
    # test feature weights if we are using feature hashing
    # since model_params is not defined with a featurehasher.
    if not use_feature_hashing:

        # get the weights for this trained model
        learned_weights = learner.model_params[0]

        for feature_name in learned_weights:
            learned_w = math.ceil(learned_weights[feature_name])
            given_w = math.ceil(weightdict[feature_name])
            eq_(learned_w, given_w)

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.labels)
    expected_cor_range = [0.7, 0.8] if use_feature_hashing else [0.9, 1.0]
    assert_greater(cor, expected_cor_range[0])
    assert_less(cor, expected_cor_range[1])


# the runner function for linear regression models
def test_linear_models():

    for (regressor_name,
         use_feature_hashing,
         use_rescaling) in product(['BayesianRidge',
                                    'ElasticNet',
                                    'HuberRegressor',
                                    'Lars',
                                    'Lasso',
                                    'LinearRegression',
                                    'Ridge',
                                    'LinearSVR',
                                    'SGDRegressor',
                                    'TheilSenRegressor'],
                                   [False, True],
                                   [False, True]):

        yield (check_linear_models,
               regressor_name,
               use_feature_hashing,
               use_rescaling)


# the utility function to run the non-linear tests
def check_non_linear_models(name,
                            use_feature_hashing=False,
                            use_rescaling=False):

    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        train_fs, test_fs, weightdict = make_regression_data(num_examples=5000,
                                                             num_features=10,
                                                             use_feature_hashing=True,
                                                             feature_bins=5)
    else:
        train_fs, test_fs, weightdict = make_regression_data(num_examples=2000,
                                                             num_features=3)

    # create the learner
    if use_rescaling:
        name = 'Rescaled' + name
    learner = Learner(name)

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_objective='pearson')

    # Note that we cannot check the feature weights here
    # since `model_params()` is not defined for non-linear
    # kernels.

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.labels)
    expected_cor_range = [0.7, 0.8] if use_feature_hashing else [0.9, 1.0]
    assert_greater(cor, expected_cor_range[0])
    assert_less(cor, expected_cor_range[1])


# the runner function for linear regression models
def test_non_linear_models():

    for (regressor_name,
         use_feature_hashing,
         use_rescaling) in product(['SVR'],
                                   [False, True],
                                   [False, True]):

        yield (check_non_linear_models,
               regressor_name,
               use_feature_hashing,
               use_rescaling)

# the utility function to run the tree-based regression tests


def check_tree_models(name,
                      use_feature_hashing=False,
                      use_rescaling=False):

    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        train_fs, test_fs, _ = make_regression_data(num_examples=5000,
                                                    num_features=10,
                                                    use_feature_hashing=True,
                                                    feature_bins=5)
    else:
        train_fs, test_fs, _ = make_regression_data(num_examples=2000,
                                                    num_features=3)

    # create the learner
    if use_rescaling:
        name = 'Rescaled' + name
    learner = Learner(name)

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_objective='pearson')

    # make sure that the feature importances are as expected.
    if name.endswith('DecisionTreeRegressor'):
        expected_feature_importances = ([0.37483895,
                                         0.08816508,
                                         0.25379838,
                                         0.18337128,
                                         0.09982631] if use_feature_hashing else
                                        [0.08926899,
                                         0.15585068,
                                         0.75488033])
        expected_cor_range = [0.5, 0.6] if use_feature_hashing else [0.9, 1.0]
    else:
        expected_feature_importances = ([0.40195798,
                                         0.06702903,
                                         0.25816559,
                                         0.18185518,
                                         0.09099222] if use_feature_hashing else
                                        [0.07974267,
                                         0.16121895,
                                         0.75903838])
        expected_cor_range = [0.7, 0.8] if use_feature_hashing else [0.9, 1.0]

    feature_importances = learner.model.feature_importances_
    assert_allclose(feature_importances, expected_feature_importances,
                    atol=1e-2, rtol=0)

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, expected_cor_range[0])
    assert_less(cor, expected_cor_range[1])


# the runner function for tree-based regression models
def test_tree_models():

    for (regressor_name,
         use_feature_hashing,
         use_rescaling) in product(['DecisionTreeRegressor',
                                    'RandomForestRegressor'],
                                   [False, True],
                                   [False, True]):

        yield (check_tree_models, regressor_name, use_feature_hashing,
               use_rescaling)


# the utility function to run the ensemble-based regression tests
def check_ensemble_models(name,
                          use_feature_hashing=False,
                          use_rescaling=False):

    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        train_fs, test_fs, _ = make_regression_data(num_examples=5000,
                                                    num_features=10,
                                                    use_feature_hashing=True,
                                                    feature_bins=5)
    else:
        train_fs, test_fs, _ = make_regression_data(num_examples=2000,
                                                    num_features=3)

    # create the learner
    if use_rescaling:
        name = 'Rescaled' + name
    learner = Learner(name)

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_objective='pearson')

    # make sure that the feature importances are as expected.
    if name.endswith('AdaBoostRegressor'):
        if use_feature_hashing:
            expected_feature_importances = [0.33718443,
                                            0.07810721,
                                            0.25621769,
                                            0.19489766,
                                            0.13359301]
        else:
            expected_feature_importances = [0.10266744, 0.18681777, 0.71051479]
    else:
        expected_feature_importances = ([0.204,
                                         0.172,
                                         0.178,
                                         0.212,
                                         0.234] if use_feature_hashing else
                                        [0.262,
                                         0.288,
                                         0.45])

    feature_importances = learner.model.feature_importances_
    assert_allclose(feature_importances, expected_feature_importances,
                    atol=1e-2, rtol=0)

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.labels)
    expected_cor_range = [0.7, 0.8] if use_feature_hashing else [0.9, 1.0]
    assert_greater(cor, expected_cor_range[0])
    assert_less(cor, expected_cor_range[1])


# the runner function for ensemble regression models
def test_ensemble_models():

    for (regressor_name,
         use_feature_hashing,
         use_rescaling) in product(['AdaBoostRegressor',
                                    'GradientBoostingRegressor'],
                                   [False, True],
                                   [False, True]):

        yield (check_ensemble_models, regressor_name, use_feature_hashing,
               use_rescaling)


def test_int_labels():
    """
    Testing that SKLL can take integer input.
    This is just to test that SKLL can take int labels in the input
    (rather than floats or strings).  For v1.0.0, it could not because the
    json package doesn't know how to serialize numpy.int64 objects.
    """
    config_template_path = join(_my_dir, 'configs',
                                'test_int_labels_cv.template.cfg')
    config_path = join(_my_dir, 'configs', 'test_int_labels_cv.cfg')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path, validate=False)
    config.set("Input", "train_file",
               join(_my_dir, 'other', 'test_int_labels_cv.jsonlines'))
    config.set("Output", "results", output_dir)
    config.set("Output", "log", output_dir)
    config.set("Output", "predictions", output_dir)

    with open(config_path, 'w') as new_config_file:
        config.write(new_config_file)

    run_configuration(config_path, quiet=True)


def test_additional_metrics():
    """
    Test additional metrics in the results file for a regressor
    """
    train_fs, test_fs, _ = make_regression_data(num_examples=2000,
                                                num_features=3)

    # train a regression model using the train feature set
    learner = Learner('LinearRegression')
    learner.train(train_fs, grid_objective='pearson')

    # evaluate the trained model using the test feature set
    results = learner.evaluate(test_fs, output_metrics=['spearman',
                                                        'kendall_tau'])

    # check that the values for the additional metrics are as expected
    additional_scores_dict = results[-1]
    assert_almost_equal(additional_scores_dict['spearman'], 0.9996, places=4)
    assert_almost_equal(additional_scores_dict['kendall_tau'], 0.9846, places=4)


def test_fancy_output():
    """
    Test the descriptive statistics output in the results file for a regressor
    """
    train_fs, test_fs, _ = make_regression_data(num_examples=2000,
                                                num_features=3)

    # train a regression model using the train feature set
    learner = Learner('LinearRegression')
    learner.train(train_fs, grid_objective='pearson')

    # evaluate the trained model using the test feature set
    resultdict = learner.evaluate(test_fs)
    actual_stats_from_api = dict(resultdict[2]['descriptive']['actual'])
    pred_stats_from_api = dict(resultdict[2]['descriptive']['predicted'])

    # write out the training and test feature set
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    train_writer = NDJWriter(join(train_dir, 'fancy_train.jsonlines'),
                             train_fs)
    train_writer.write()
    test_writer = NDJWriter(join(test_dir, 'fancy_test.jsonlines'), test_fs)
    test_writer.write()

    # now get the config file template, fill it in and run it
    # so that we can get a results file
    config_template_path = join(_my_dir, 'configs',
                                'test_regression_fancy_output.template.cfg')
    config_path = fill_in_config_paths_for_fancy_output(config_template_path)

    run_configuration(config_path, quiet=True)

    # read in the results file and get the descriptive statistics
    actual_stats_from_file = {}
    pred_stats_from_file = {}
    with open(join(output_dir, ('regression_fancy_output_train_fancy_train.'
                                'jsonlines_test_fancy_test.jsonlines'
                                '_LinearRegression.results')),
              'r') as resultf:

        result_output = resultf.read().strip().split('\n')
        for desc_stat_line in result_output[26:30]:
            desc_stat_line = desc_stat_line.strip()
            if not desc_stat_line:
                continue
            else:
                m = re.search(r'([A-Za-z]+)\s+=\s+(-?[0-9]+.?[0-9]*)\s+'
                              r'\((actual)\),\s+(-?[0-9]+.?[0-9]*)\s+'
                              r'\((predicted)\)', desc_stat_line)
                stat_type, actual_value, _, pred_value, _ = m.groups()
                actual_stats_from_file[stat_type.lower()] = float(actual_value)
                pred_stats_from_file[stat_type.lower()] = float(pred_value)

    for stat_type in actual_stats_from_api:

        assert_almost_equal(actual_stats_from_file[stat_type],
                            actual_stats_from_api[stat_type],
                            places=4)

        assert_almost_equal(pred_stats_from_file[stat_type],
                            pred_stats_from_api[stat_type],
                            places=4)


def check_adaboost_regression(base_estimator):
    train_fs, test_fs, _ = make_regression_data(num_examples=2000,
                                                sd_noise=4,
                                                num_features=3)

    # train an AdaBoostRegressor on the training data and evalute on the
    # testing data
    learner = Learner('AdaBoostRegressor', model_kwargs={'base_estimator':
                                                         base_estimator})
    learner.train(train_fs, grid_search=False)

    # now generate the predictions on the test set
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, 0.95)


def test_adaboost_regression():
    for base_estimator_name in ['DecisionTreeRegressor', 'SGDRegressor', 'SVR']:
        yield check_adaboost_regression, base_estimator_name


def check_ransac_regression(base_estimator, pearson_value):
    train_fs, test_fs, _ = make_regression_data(num_examples=2000,
                                                sd_noise=4,
                                                num_features=3)

    # train a RANSACRegressor on the training data and evalute on the
    # testing data
    model_kwargs = {'base_estimator': base_estimator} if base_estimator else {}
    learner = Learner('RANSACRegressor', model_kwargs=model_kwargs)
    learner.train(train_fs, grid_search=False)

    # now generate the predictions on the test set
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated and the value
    # of the correlation is as expected
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, pearson_value)


def test_ransac_regression():
    for (base_estimator_name, pearson_value) in zip([None,
                                                     'SGDRegressor',
                                                     'DecisionTreeRegressor',
                                                     'SVR'],
                                                     [0.95, 0.45, 0.75, 0.65]):
        yield check_ransac_regression, base_estimator_name, pearson_value


def check_mlp_regression(use_rescaling=False):
    train_fs, test_fs, _ = make_regression_data(num_examples=500,
                                                sd_noise=4,
                                                num_features=5)

    # train an MLPRegressor on the training data and evalute on the
    # testing data
    name = 'MLPRegressor' if use_rescaling else 'RescaledMLPRegressor'
    learner = Learner(name)
    # we don't want to see any convergence warnings during the grid search
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        learner.train(train_fs, grid_search=True, grid_objective='pearson')

    # now generate the predictions on the test set
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, 0.98)


def test_mlp_regression():
    yield check_mlp_regression, False
    yield check_mlp_regression, True


def check_dummy_regressor_predict(model_args, train_labels, expected_output):

    # create hard-coded featuresets with the given labels
    train_fs = FeatureSet('regression_train',
                          ['TrainExample{}'.format(i) for i in range(20)],
                          labels=train_labels,
                          features=[{"feature": i} for i in range(20)])

    test_fs = FeatureSet('regression_test',
                         ['TestExample{}'.format(i) for i in range(10)],
                         features=[{"feature": i} for i in range(20, 30)])

    # Ensure predictions are as expectedfor the given strategy
    learner = Learner('DummyRegressor', model_kwargs=model_args)
    learner.train(train_fs, grid_search=False)
    predictions = learner.predict(test_fs)
    eq_(np.array_equal(expected_output, predictions), True)


def test_dummy_regressor_predict():

    # create a hard-coded set of labels
    prng = np.random.RandomState(123456789)
    train_labels = prng.random_sample(20)

    for (model_args, expected_output) in zip([{"strategy": "mean"},
                                              {"strategy": "median"},
                                              {"strategy": "quantile", "quantile": 0.5},
                                              {"strategy": "quantile", "quantile": 0.0},
                                              {"strategy": "quantile", "quantile": 1.0},
                                              {"strategy": "constant", "constant": 1}],
                                             [np.ones(10)*np.mean(train_labels),
                                              np.ones(10)*np.median(train_labels),
                                              np.ones(10)*np.median(train_labels),
                                              np.ones(10)*np.min(train_labels),
                                              np.ones(10)*np.max(train_labels),
                                              np.ones(10)]):
        yield check_dummy_regressor_predict, model_args, train_labels, expected_output
