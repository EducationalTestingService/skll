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
import re

from itertools import product
from os.path import abspath, dirname

from nose.tools import eq_, assert_almost_equal

import numpy as np
from numpy.testing import assert_allclose
from sklearn.utils.testing import assert_greater, assert_less
from skll.data import FeatureSet
from skll.learner import Learner
from skll.learner import _DEFAULT_PARAM_GRIDS
from scipy.stats import pearsonr

from utils import make_regression_data


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
SCORE_OUTPUT_RE = re.compile(r'Objective Function Score \(Test\) = '
                             r'([\-\d\.]+)')
GRID_RE = re.compile(r'Grid Objective Score \(Train\) = ([\-\d\.]+)')
_my_dir = abspath(dirname(__file__))


# a utility function to check rescaling for linear models
def check_rescaling(name):

    train_fs, test_fs, weightdict = make_regression_data(num_examples=2000,
                                                         sd_noise=4,
                                                         num_features=3)

    # instantiate the given learner and its rescaled counterpart
    learner = Learner(name)
    rescaled_learner = Learner('Rescaled' + name)

    # train both the regular regressor and the rescaled regressor
    learner.train(train_fs, grid_objective='pearson')
    rescaled_learner.train(train_fs, grid_objective='pearson')

    # now generate both sets of predictions on the test feature set
    predictions = learner.predict(test_fs)
    rescaled_predictions = rescaled_learner.predict(test_fs)

    # ... and on the training feature set
    train_predictions = learner.predict(train_fs)
    rescaled_train_predictions = rescaled_learner.predict(train_fs)

    # make sure that both sets of correlations are close to perfectly correlated
    # since the only thing different is that one set has been rescaled
    assert_almost_equal(pearsonr(predictions, rescaled_predictions)[0], 1.0, places=3)

    # make sure that the standard deviation of the rescaled test set predictions
    # is higher than the standard deviation of the regular test set predictions
    p_std = np.std(predictions)
    rescaled_p_std = np.std(rescaled_predictions)
    assert_greater(rescaled_p_std, p_std)

    # make sure that the standard deviation of the rescaled predictions
    # on the TRAINING set (not the TEST) is closer to the standard
    # deviation of the training set labels than the standard deviation
    # of the regular predictions.
    train_y_std = np.std(train_fs.classes)
    train_p_std = np.std(train_predictions)
    rescaled_train_p_std = np.std(rescaled_train_predictions)
    assert_less(abs(rescaled_train_p_std - train_y_std), abs(train_p_std - train_y_std))


def test_rescaling():
    for regressor_name in ['ElasticNet', 'Lasso',
                           'LinearRegression', 'Ridge',
                           'SVR', 'SGDRegressor']:

        yield check_rescaling, regressor_name


# the utility function to run the linear regression tests
def check_linear_models(name,
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
    learner.train(train_fs,
                  grid_objective='pearson',
                  feature_hasher=use_feature_hashing)

    # make sure that the weights are close to the weights
    # that we got from make_regression_data. Take the
    # ceiling before  comparing since just comparing
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
    predictions = learner.predict(test_fs, feature_hasher=use_feature_hashing)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.classes)
    expected_cor_range = [0.7, 0.8] if use_feature_hashing else [0.9, 1.0]
    assert_greater(cor, expected_cor_range[0])
    assert_less(cor, expected_cor_range[1])


# the runner function for linear regression models
def test_linear_models():

    for (regressor_name,
        use_feature_hashing,
        use_rescaling) in product(['ElasticNet', 'Lasso',
                                   'LinearRegression', 'Ridge',
                                   'SVR', 'SGDRegressor'],
                                  [False, True],
                                  [False, True]):

        yield check_linear_models, regressor_name, use_feature_hashing, use_rescaling


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
    learner.train(train_fs, grid_objective='pearson', feature_hasher=use_feature_hashing)

    # make sure that the feature importances are as expected.
    if name.endswith('DecisionTreeRegressor'):
        expected_feature_importances = [0.37331461,
                                        0.08572699,
                                        0.2543484,
                                        0.1841172,
                                        0.1024928] if use_feature_hashing else [0.08931994,
                                                                                0.15545093,
                                                                                0.75522913]
        expected_cor_range = [0.5, 0.6] if use_feature_hashing else [0.9, 1.0]
    else:
        expected_feature_importances = [0.40195655,
                                        0.06702161,
                                        0.25814858,
                                        0.18183947,
                                        0.09103379] if use_feature_hashing else [0.07975691,
                                                                                 0.16122862,
                                                                                 0.75901447]
        expected_cor_range = [0.7, 0.8] if use_feature_hashing else [0.9, 1.0]

    feature_importances = learner.model.feature_importances_
    assert_allclose(feature_importances, expected_feature_importances, rtol=1e-2)


    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs, feature_hasher=use_feature_hashing)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.classes)
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

        yield check_tree_models, regressor_name, use_feature_hashing, use_rescaling


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
    learner.train(train_fs, grid_objective='pearson', feature_hasher=use_feature_hashing)

    # make sure that the feature importances are as expected.
    if name.endswith('AdaBoostRegressor'):
        expected_feature_importances = [0.33260501,
                                        0.07685393,
                                        0.25858443,
                                        0.19214259,
                                        0.13981404] if use_feature_hashing else [0.10266744,
                                                                                 0.18681777,
                                                                                 0.71051479]
    else:
        expected_feature_importances = [0.204,
                                        0.172,
                                        0.178,
                                        0.212,
                                        0.234] if use_feature_hashing else [0.262,
                                                                            0.288,
                                                                            0.45]

    feature_importances = learner.model.feature_importances_
    assert_allclose(feature_importances, expected_feature_importances, rtol=1e-2)

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs, feature_hasher=use_feature_hashing)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.classes)
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

        yield check_ensemble_models, regressor_name, use_feature_hashing, use_rescaling

