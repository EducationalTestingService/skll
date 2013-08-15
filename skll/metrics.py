# Copyright (C) 2012-2013 Educational Testing Service

# This file is part of SciKit-Learn Laboratory.

# SciKit-Learn Laboratory is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

# SciKit-Learn Laboratory is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# SciKit-Learn Laboratory.  If not, see <http://www.gnu.org/licenses/>.

'''
This module contains a bunch of evaluation metrics that can be used to
evaluate the performance of learners.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
'''

from __future__ import print_function, unicode_literals

import logging

import ml_metrics
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, SCORERS


# Constants
_CORRELATION_METRICS = frozenset(['kendall_tau', 'spearman', 'pearson'])


def kappa(y_true, y_pred, weights=None):
    '''
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating

    Simple kappa function adapted from yorchopolis's kappa-stats project on
    Github.

    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'squared' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimension


    :type weights: str or numpy array
    '''

    assert(len(y_true) == len(y_pred))
    conf_matrix = confusion_matrix(y_true, y_pred)
    num_ratings = len(conf_matrix)
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if (weights is None) or (weights == 'squared') or (weights == 'linear'):
        weighted = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                if weights is None:
                    weighted[i, j] = (i != j)
                elif weights == 'squared':
                    weighted[i, j] = abs(i - j) ** 2
                else:  # linear
                    weighted[i, j] = abs(i - j)

    # Figure out observed/expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))
    hist_y_true = np.bincount(y_true)[min_rating: max_rating + 1]
    hist_y_pred = np.bincount(y_pred)[min_rating: max_rating + 1]

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_y_true[i] * hist_y_pred[j]
                              / num_scored_items)
            d = weighted[i, j]
            numerator += d * conf_matrix[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


# First we have to define the functions, and then we add them to SCORERS at the
# end
def quadratic_weighted_kappa(y_true, y_pred):
    '''
    Returns the quadratic weighted kappa.
    This rounds the inputs before passing them to the ml_metrics module.
    '''

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    try:
        y_true_rounded = [int(round(float(y))) for y in y_true]
        y_pred_rounded = [int(round(float(y))) for y in y_pred]
    except ValueError as e:
        logging.error("For kappa, the labels should be integers or strings " +
                      "that can be converted to ints (E.g., '4.0' or '3').")
        raise e

    res = kappa(y_true_rounded, y_pred_rounded, weights='squared')
    return res


def unweighted_kappa(y_true, y_pred):
    '''
    Returns the unweighted Cohen's kappa.
    '''
    # See quadratic_weighted_kappa for comments about the typecasting below.
    try:
        y_true_rounded = [int(round(float(y))) for y in y_true]
        y_pred_rounded = [int(round(float(y))) for y in y_pred]
    except ValueError as e:
        logging.error("For kappa, the labels should be integers or strings " +
                      "that can be converted to ints (E.g., '4.0' or '3').")
        raise e

    res = kappa(y_true_rounded, y_pred_rounded)
    return res


def kendall_tau(y_true, y_pred):
    '''
    Optimize the hyperparameter values during the grid search based on
    Kendall's tau.

    This is useful in cases where you want to use the actual probabilities of
    the different classes after the fact, and not just the optimize based on
    the classification accuracy.
    '''
    ret_score = kendalltau(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def spearman(y_true, y_pred):
    '''
    Optimize the hyperparameter values during the grid search based on
    Spearman rank correlation.

    This is useful in cases where you want to use the actual probabilities of
    the different classes after the fact, and not just the optimize based on
    the classification accuracy.
    '''
    ret_score = spearmanr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def pearson(y_true, y_pred):
    '''
    Optimize the hyperparameter values during the grid search based on Pearson
    correlation.
    '''
    ret_score = pearsonr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def f1_score_least_frequent(y_true, y_pred):
    '''
    Optimize the hyperparameter values during the grid search based on the F1
    measure of the least frequent class.

    This is mostly intended for use when you're doing binary classification
    and your data is highly skewed. You should probably use f1_score_macro if
    your data is skewed and you're doing multi-class classification.
    '''
    least_frequent = np.bincount(y_true).argmin()
    return f1_score(y_true, y_pred, average=None)[least_frequent]


# Store our scorers in a dictionary and then update the scikit-learn SCORERS
_scorers = {'f1_score_micro': make_scorer(f1_score, average='micro'),
            'f1_score_macro': make_scorer(f1_score, average='macro'),
            'f1_score_least_frequent': make_scorer(f1_score_least_frequent),
            'pearson': make_scorer(pearson),
            'spearman': make_scorer(spearman),
            'kendall_tau': make_scorer(kendall_tau),
            'unweighted_kappa': make_scorer(unweighted_kappa),
            'quadratic_weighted_kappa': make_scorer(quadratic_weighted_kappa)}
SCORERS.update(_scorers)
