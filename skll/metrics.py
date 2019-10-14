# License: BSD 3 clause
"""
This module contains a bunch of evaluation metrics that can be used to
evaluate the performance of learners.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import (confusion_matrix,
                             f1_score,
                             SCORERS)


# Useful constants
_CORRELATION_METRICS = set(['kendall_tau', 'pearson', 'spearman'])

_REGRESSION_ONLY_METRICS = set(['explained_variance',
                                'max_error',
                                'neg_mean_squared_error',
                                'neg_mean_absolute_error',
                                'r2'])

_CLASSIFICATION_ONLY_METRICS = set(['accuracy',
                                    'average_precision',
                                    'balanced_accuracy',
                                    'f1',
                                    'f1_score_least_frequent',
                                    'f1_score_macro',
                                    'f1_score_micro',
                                    'f1_score_weighted',
                                    'neg_log_loss',
                                    'precision',
                                    'recall',
                                    'roc_auc'])

_PROBABILISTIC_METRICS = frozenset(['average_precision',
                                    'neg_log_loss',
                                    'roc_auc'])

_UNWEIGHTED_KAPPA_METRICS = set(['unweighted_kappa',
                                 'uwk_off_by_one'])

_WEIGHTED_KAPPA_METRICS = set(['linear_weighted_kappa',
                               'lwk_off_by_one',
                               'quadratic_weighted_kappa',
                               'qwk_off_by_one'])


def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in ``y_true`` and ``y_pred`` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    Parameters
    ----------
    y_true : array-like of float
        The true/actual/gold labels for the data.
    y_pred : array-like of float
        The predicted/observed labels for the data.
    weights : str or np.array, optional
        Specifies the weight matrix for the calculation.
        Options are ::

            -  None = unweighted-kappa
            -  'quadratic' = quadratic-weighted kappa
            -  'linear' = linear-weighted kappa
            -  two-dimensional numpy array = a custom matrix of

        weights. Each weight corresponds to the
        :math:`w_{ij}` values in the wikipedia description
        of how to calculate weighted Cohen's kappa.
        Defaults to None.
    allow_off_by_one : bool, optional
        If true, ratings that are off by one are counted as
        equal, and all other differences are reduced by
        one. For example, 1 and 2 will be considered to be
        equal, whereas 1 and 3 will have a difference of 1
        for when building the weights matrix.
        Defaults to False.

    Returns
    -------
    k : float
        The kappa score, or weighted kappa score.

    Raises
    ------
    AssertionError
        If ``y_true`` != ``y_pred``.
    ValueError
        If labels cannot be converted to int.
    ValueError
        If invalid weight scheme.
    """

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
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError:
        raise ValueError("For kappa, the labels should be integers or strings "
                         "that can be converted to ints (E.g., '4.0' or '3').")

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, str):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k


def correlation(y_true, y_pred, corr_type='pearson'):
    """
    Calculate given correlation between ``y_true`` and ``y_pred``. ``y_pred``
    can be multi-dimensional. If ``y_pred`` is 1-dimensional, it may either
    contain probabilities, most-likely classification labels, or regressor
    predictions. In that case, we simply return the correlation between
    ``y_true`` and ``y_pred``. If ``y_pred`` is multi-dimensional,
    it contains probabilties for multiple classes in which case, we infer
    the most likely labels and then compute the correlation between those
    and ``y_true``.

    Parameters
    ----------
    y_true : array-like of float
        The true/actual/gold labels for the data.
    y_pred : array-like of float
        The predicted/observed labels for the data.
    corr_type : str, optional
        Which type of correlation to compute. Possible
        choices are ``pearson``, ``spearman``,
        and ``kendall_tau``.
        Defaults to ``pearson``.

    Returns
    -------
    ret_score : float
        correlation value if well-defined, else 0.0
    """

    # get the correlation function to use based on the given type
    corr_func = pearsonr
    if corr_type == 'spearman':
        corr_func = spearmanr
    elif corr_type == 'kendall_tau':
        corr_func = kendalltau

    # convert to numpy array in case we are passed a list
    y_pred = np.array(y_pred)

    # multi-dimensional -> probability array -> get label
    if y_pred.ndim > 1:
        labels = np.argmax(y_pred, axis=1)
        ret_score = corr_func(y_true, labels)[0]
    # 1-dimensional -> probabilities/labels -> use as is
    else:
        ret_score = corr_func(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def f1_score_least_frequent(y_true, y_pred):
    """
    Calculate the F1 score of the least frequent label/class in ``y_true`` for
    ``y_pred``.

    Parameters
    ----------
    y_true : array-like of float
        The true/actual/gold labels for the data.
    y_pred : array-like of float
        The predicted/observed labels for the data.

    Returns
    -------
    ret_score : float
        F1 score of the least frequent label.
    """
    least_frequent = np.bincount(y_true).argmin()
    return f1_score(y_true, y_pred, average=None)[least_frequent]


def use_score_func(func_name, y_true, y_pred):
    """
    Call the scoring function in ``sklearn.metrics.SCORERS`` with the given name.
    This takes care of handling keyword arguments that were pre-specified when
    creating the scorer. This applies any sign-flipping that was specified by
    ``make_scorer()`` when the scorer was created.

    Parameters
    ----------
    func_name : str
        The name of the objective function to use from SCORERS.
    y_true : array-like of float
        The true/actual/gold labels for the data.
    y_pred : array-like of float
        The predicted/observed labels for the data.

    Returns
    -------
    ret_score : float
        The scored result from the given scorer.
    """
    scorer = SCORERS[func_name]
    return scorer._sign * scorer._score_func(y_true, y_pred, **scorer._kwargs)
