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

import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six import string_types
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix, f1_score


# Constants
_CORRELATION_METRICS = frozenset(['kendall_tau', 'spearman', 'pearson'])


def kappa(y_true, y_pred, weights=None):
    '''
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.


    :type weights: str or numpy array
    '''

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    try:
        y_true = [int(round(float(y))) for y in y_true]
        y_pred = [int(round(float(y))) for y in y_pred]
    except ValueError as e:
        logging.error("For kappa, the labels should be integers or strings " +
                      "that can be converted to ints (E.g., '4.0' or '3').")
        raise e

    # Build the observed/confusion matrix
    observed = confusion_matrix(y_true, y_pred)
    num_ratings = len(observed)
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                if wt_scheme == 'linear':
                    weights[i, j] = abs(i - j)
                elif wt_scheme == 'quadratic':
                    weights[i, j] = abs(i - j) ** 2
                else:  # unweighted
                    weights[i, j] = (i != j)

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))
    hist_true = np.bincount(y_true, minlength=(max_rating + 1))
    hist_true = hist_true[min_rating: max_rating + 1] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=(max_rating + 1))
    hist_pred = hist_pred[min_rating: max_rating + 1] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    k = 1.0 - (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k


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
