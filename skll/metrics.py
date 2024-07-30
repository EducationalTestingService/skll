# License: BSD 3 clause
"""
Metrics that can be used to evaluate the performance of learners.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
"""

import copy
import sys
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    get_scorer,
    get_scorer_names,
    make_scorer,
)

from skll.types import PathOrStr


def kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[Union[str, np.ndarray]] = None,
    allow_off_by_one: bool = False,
) -> float:
    """
    Calculate the kappa inter-rater agreement.

    The agreement is calculated between the gold standard and the predicted
    ratings. Potential values range from -1 (representing complete disagreement)
    to 1 (representing complete agreement).  A kappa value of 0 is expected if
    all agreement is due to chance.

    In the course of calculating kappa, all items in ``y_true`` and ``y_pred`` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    Parameters
    ----------
    y_true : numpy.ndarray
        The true/actual/gold labels for the data.
    y_pred : numpy.ndarray
        The predicted/observed labels for the data.
    weights : Optional[Union[str, numpy.ndarray]], default=None
        Specifies the weight matrix for the calculation.
        Possible values are: ``None`` (unweighted-kappa), ``"quadratic"``
        (quadratically weighted kappa), ``"linear"`` (linearly weighted kappa),
        and a two-dimensional numpy array (a custom matrix of weights). Each
        weight in this array corresponds to the :math:`w_{ij}` values in the
        Wikipedia description of how to calculate weighted Cohen's kappa.
    allow_off_by_one : bool, default=False
        If true, ratings that are off by one are counted as
        equal, and all other differences are reduced by
        one. For example, 1 and 2 will be considered to be
        equal, whereas 1 and 3 will have a difference of 1
        for when building the weights matrix.

    Returns
    -------
    float
        The weighted or unweighted kappa score.

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
    assert len(y_true) == len(y_pred)

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    try:
        y_true = np.array([int(np.round(float(y))) for y in y_true])
        y_pred = np.array([int(np.round(float(y))) for y in y_pred])
    except ValueError:
        raise ValueError(
            "For kappa, the labels should be integers or strings"
            " that can be converted to ints (E.g., '4.0' or "
            "'3')."
        )

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = y_true - min_rating
    y_pred = y_pred - min_rating

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, str):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ""

    if weights is None:
        kappa_weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == "linear":
                    kappa_weights[i, j] = diff
                elif wt_scheme == "quadratic":
                    kappa_weights[i, j] = diff**2
                elif not wt_scheme:  # unweighted
                    kappa_weights[i, j] = bool(diff)
                else:
                    raise ValueError("Invalid weight scheme specified for " f"kappa: {wt_scheme}")
    else:
        kappa_weights = weights

    hist_true: np.ndarray = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[:num_ratings] / num_scored_items
    hist_pred: np.ndarray = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[:num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(kappa_weights):
        observed_sum = np.sum(kappa_weights * observed)
        expected_sum = np.sum(kappa_weights * expected)
        k -= np.sum(observed_sum) / np.sum(expected_sum)

    return k


def correlation(y_true: np.ndarray, y_pred: np.ndarray, corr_type: str = "pearson") -> float:
    """
    Calculate given correlation type between ``y_true`` and ``y_pred``.

    ``y_pred`` can be multi-dimensional. If ``y_pred`` is 1-dimensional, it
    may either contain probabilities, most-likely classification labels, or
    regressor predictions. In that case, we simply return the correlation
    between ``y_true`` and ``y_pred``. If ``y_pred`` is multi-dimensional,
    it contains probabilties for multiple classes in which case, we infer the
    most likely labels and then compute the correlation between those and
    ``y_true``.

    Parameters
    ----------
    y_true : numpy.ndarray
        The true/actual/gold labels for the data.
    y_pred : numpy.ndarray
        The predicted/observed labels for the data.
    corr_type : str, default="pearson"
        Which type of correlation to compute. Possible
        choices are "pearson", "spearman", and "kendall_tau".

    Returns
    -------
    float
        correlation value if well-defined, else 0.0

    """
    # get the correlation function to use based on the given type
    corr_func = pearsonr
    if corr_type == "spearman":
        corr_func = spearmanr
    elif corr_type == "kendall_tau":
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
    return ret_score


def f1_score_least_frequent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score of the least frequent label/class.

    Parameters
    ----------
    y_true : numpy.ndarray
        The true/actual/gold labels for the data.
    y_pred : numpy.ndarray
        The predicted/observed labels for the data.

    Returns
    -------
    float
        F1 score of the least frequent label.

    """
    least_frequent = np.bincount(y_true).argmin()
    return f1_score(y_true, y_pred, average=None)[least_frequent]


def register_custom_metric(custom_metric_path: PathOrStr, custom_metric_name: str):
    """
    Import, load, and register the custom metric function from the given path.

    Parameters
    ----------
    custom_metric_path : :class:`skll.types.PathOrStr`
        The path to a custom metric.
    custom_metric_name : str
        The name of the custom metric function to load. This function must take
        only two array-like arguments: the true labels and the predictions,
        in that order.

    Raises
    ------
    ValueError
        If the custom metric path does not end in '.py'.
    NameError
        If the name of the custom metric file conflicts
        with an already existing attribute in ``skll.metrics``
        or if the custom metric name conflicts with a scikit-learn
        or SKLL metric.

    """
    if not custom_metric_path:
        raise ValueError(
            f"custom metric path was not set and " f"metric {custom_metric_name} was not found."
        )

    custom_metric_path = Path(custom_metric_path)
    if not custom_metric_path.exists():
        raise ValueError(f"custom metric path '{custom_metric_path}' " f"does not exist.")

    if custom_metric_path.suffix != ".py":
        raise ValueError(
            f"custom metric path must end in .py, you specified " f"{custom_metric_path}"
        )

    # get the name of the module containing the custom metric
    custom_metric_module_name = custom_metric_path.stem

    # once we know that the module name is okay, we need to make sure
    # that the metric function name is also okay
    if custom_metric_name in get_scorer_names() or custom_metric_name in _CUSTOM_METRICS:
        raise NameError(
            f"a metric called '{custom_metric_name}' already "
            f"exists; rename the metric function in "
            f"{custom_metric_module_name}.py and try again."
        )

    # dynamically import the module unless we have already done it
    if custom_metric_module_name not in sys.modules:
        sys.path.append(str(custom_metric_path.resolve().parent))
        metric_module = import_module(custom_metric_module_name)

        # this statement is only necessary so that if we end
        # up using multiprocessing parallelization backend,
        # things are serialized properly
        globals()[custom_metric_module_name] = metric_module

    # get the metric function from this imported module
    metric_func = getattr(sys.modules[custom_metric_module_name], custom_metric_name)
    # again, we need this for multiprocessing serialization
    metric_func.__module__ = f"skll.metrics.{custom_metric_module_name}"

    # extract any "special" keyword arguments from the metric function
    metric_func_parameters = signature(metric_func).parameters
    make_scorer_kwargs = {}
    for make_scorer_kwarg in ["greater_is_better", "response_method"]:
        if make_scorer_kwarg in metric_func_parameters:
            parameter = metric_func_parameters.get(make_scorer_kwarg)
            if parameter is not None:
                parameter_value = parameter.default
                make_scorer_kwargs.update({make_scorer_kwarg: parameter_value})

    # make the scorer function with the extracted keyword arguments
    _CUSTOM_METRICS[f"{custom_metric_name}"] = make_scorer(metric_func, **make_scorer_kwargs)

    return metric_func


def use_score_func(func_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Call the given scoring function.

    This takes care of handling keyword arguments that were pre-specified
    when creating the scorer. This applies any sign-flipping that was
    specified by ``make_scorer()`` when the scorer was created.

    Parameters
    ----------
    func_name : str
        The name of the objective function to use.
    y_true : numpy.ndarray
        The true/actual/gold labels for the data.
    y_pred : numpy.ndarray
        The predicted/observed labels for the data.

    Returns
    -------
    float
        The scored result from the given scorer.

    """
    try:
        scorer = get_scorer(func_name)
    except ValueError:
        scorer = _CUSTOM_METRICS[func_name]

    return scorer._sign * scorer._score_func(y_true, y_pred, **scorer._kwargs)


# a dictionary that maps pre-defined custom metric names to their scorer functions
# this is a private variable only meant for internal use
_PREDEFINED_CUSTOM_METRICS = {
    "f1_score_micro": make_scorer(f1_score, average="micro"),
    "f1_score_macro": make_scorer(f1_score, average="macro"),
    "f1_score_weighted": make_scorer(f1_score, average="weighted"),
    "f1_score_least_frequent": make_scorer(f1_score_least_frequent),
    "f05": make_scorer(fbeta_score, beta=0.5, average="binary"),
    "f05_score_micro": make_scorer(fbeta_score, beta=0.5, average="micro"),
    "f05_score_macro": make_scorer(fbeta_score, beta=0.5, average="macro"),
    "f05_score_weighted": make_scorer(fbeta_score, beta=0.5, average="weighted"),
    "pearson": make_scorer(correlation, corr_type="pearson"),
    "spearman": make_scorer(correlation, corr_type="spearman"),
    "kendall_tau": make_scorer(correlation, corr_type="kendall_tau"),
    "unweighted_kappa": make_scorer(kappa),
    "quadratic_weighted_kappa": make_scorer(kappa, weights="quadratic"),
    "linear_weighted_kappa": make_scorer(kappa, weights="linear"),
    "qwk_off_by_one": make_scorer(kappa, weights="quadratic", allow_off_by_one=True),
    "lwk_off_by_one": make_scorer(kappa, weights="linear", allow_off_by_one=True),
    "uwk_off_by_one": make_scorer(kappa, allow_off_by_one=True),
}

# now create a new dictionary that contains all of the above metrics but
# will also contain any user-defined custom metrics
_CUSTOM_METRICS = copy.deepcopy(_PREDEFINED_CUSTOM_METRICS)
