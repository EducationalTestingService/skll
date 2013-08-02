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
import sys

import ml_metrics
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn import metrics as sk_metrics


# Constants
_CORRELATION_METRICS = frozenset(['kendall_tau', 'spearman', 'pearson'])


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


    res = ml_metrics.quadratic_weighted_kappa(y_true_rounded, y_pred_rounded)
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

    res = ml_metrics.kappa(y_true_rounded, y_pred_rounded)
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
    return sk_metrics.f1_score(y_true, y_pred, average=None)[least_frequent]


def f1_score_macro(y_true, y_pred):
    '''
    Use the macro-averaged F1 measure to select hyperparameter values during
    the cross-validation grid search during training.

    This method averages over classes (does not take imbalance into account).
    You should use this if each class is equally important.
    '''
    return sk_metrics.f1_score(y_true, y_pred, average="macro")


def f1_score_micro(y_true, y_pred):
    '''
    Use the micro-averaged F1 measure to select hyperparameter values during
    the cross-validation grid search during training.

    This method averages over instances (takes imbalance into account). This
    implies that precision == recall == F1.
    '''
    return sk_metrics.f1_score(y_true, y_pred, average="micro")


def accuracy(y_true, y_pred):
    '''
    Use the overall accuracy to select hyperparameter values during the cross-
    validation grid search during training.
    '''
    return sk_metrics.accuracy_score(y_true, y_pred)
