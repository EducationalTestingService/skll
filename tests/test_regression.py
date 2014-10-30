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

import math
import re
from os.path import abspath, dirname

from nose.tools import eq_
from sklearn.utils.testing import assert_greater
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


# the utility function to run the basic regression tests
# with or without feature hashing
def check_regression(use_feature_hashing=False):
    # This is a bit of a contrived test, but it should fail if anything drastic
    # happens to the regression code.

    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        train_fs, test_fs, weightdict = make_regression_data(num_examples=5000,
                                                             num_features=10,
                                                             use_feature_hashing=True,
                                                             feature_bins=25)
    else:
        train_fs, test_fs, weightdict = make_regression_data(num_examples=2000,
                                                             num_features=3)

    # create a LinearRegression learner
    learner = Learner('LinearRegression')

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_objective='pearson', feature_hasher=use_feature_hashing)

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
    assert_greater(cor, 0.95)

# the runner function for the regression tests
def test_regression():
    # without feature hashing
    yield check_regression

    # with feature hashing
    yield check_regression, True
