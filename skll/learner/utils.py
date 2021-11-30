"""
Utility classes and functions for SKLL learners.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
:organization: ETS
"""

import inspect
import logging
import os
import sys
from collections import Counter, defaultdict
from csv import DictWriter, excel_tab
from functools import wraps
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    ShuffleSplit,
    StratifiedKFold,
)

from skll.data import FeatureSet
from skll.metrics import _CUSTOM_METRICS, use_score_func
from skll.utils.constants import (
    CLASSIFICATION_ONLY_METRICS,
    CORRELATION_METRICS,
    REGRESSION_ONLY_METRICS,
    UNWEIGHTED_KAPPA_METRICS,
    WEIGHTED_KAPPA_METRICS,
)
from skll.version import VERSION


class Densifier(BaseEstimator, TransformerMixin):
    """
    A custom pipeline stage that will be inserted into the
    learner pipeline attribute to accommodate the situation
    when SKLL needs to manually convert feature arrays from
    sparse to dense. For example, when features are being hashed
    but we are also doing centering using the feature means.
    """

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


class FilteredLeaveOneGroupOut(LeaveOneGroupOut):

    """
    Version of ``LeaveOneGroupOut`` cross-validation iterator that only outputs
    indices of instances with IDs in a prespecified set.

    Parameters
    ----------
    keep : set of str
        A set of IDs to keep.
    example_ids : list of str, of length n_samples
        A list of example IDs.
    """

    def __init__(self, keep, example_ids, logger=None):
        super(FilteredLeaveOneGroupOut, self).__init__()
        self.keep = keep
        self.example_ids = example_ids
        self._warned = False
        self.logger = logger if logger else logging.getLogger(__name__)

    def split(self, X, y, groups):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, with shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        -------
        train_index : np.array
            The training set indices for that split.
        test_index : np.array
            The testing set indices for that split.
        """
        for train_index, test_index in super(FilteredLeaveOneGroupOut,
                                             self).split(X, y, groups):
            train_len = len(train_index)
            test_len = len(test_index)
            train_index = [i for i in train_index if self.example_ids[i] in
                           self.keep]
            test_index = [i for i in test_index if self.example_ids[i] in
                          self.keep]
            if not self._warned and (train_len != len(train_index) or
                                     test_len != len(test_index)):
                self.logger.warning('Feature set contains IDs that are not '
                                    'in folds dictionary. Skipping those IDs.')
                self._warned = True

            yield train_index, test_index


class SelectByMinCount(SelectKBest):

    """
    Select features occurring in more (and/or fewer than) than a specified
    number of examples in the training data (or a CV training fold).

    Parameters
    ----------
    min_count : int, optional
        The minimum feature count to select.
        Defaults to 1.
    """

    def __init__(self, min_count=1):
        self.min_count = min_count
        self.scores_ = None

    def fit(self, X, y=None):
        """
        Fit the SelectByMinCount model.

        Parameters
        ----------
        X : array-like, with shape (n_samples, n_features)
            The training data to fit.
        y : Ignored

        Returns
        -------
        self
        """

        # initialize a list of counts of times each feature appears
        col_counts = [0 for _ in range(X.shape[1])]

        if sp.issparse(X):
            # find() is scipy.sparse's equivalent of nonzero()
            _, col_indices, _ = sp.find(X)
        else:
            # assume it's a numpy array (not a numpy matrix)
            col_indices = X.nonzero()[1].tolist()

        for i in col_indices:
            col_counts[i] += 1

        self.scores_ = np.array(col_counts)
        return self

    def _get_support_mask(self):
        """
        Returns an indication of which features to keep.
        Adapted from ``SelectKBest``.

        Returns
        -------
        mask : np.array
            The mask with features to keep set to True.
        """
        mask = np.zeros(self.scores_.shape, dtype=bool)
        mask[self.scores_ >= self.min_count] = True
        return mask


def add_unseen_labels(train_label_dict, test_label_list):
    """
    Merge test set labels that are not seen in the training data with seen ones.

    Parameters
    ----------
    train_label_dict : dict
        Dictionary mapping training set class labels to class indices.
    test_label_list : list
        List containing labels in the test set.

    Returns
    -------
    train_and_test_label_dict : dict
        Dictionary mapping merged labels from both the training and test sets
        to indices.
    """
    # get the list of labels that were in the training set
    train_label_list = list(train_label_dict.keys())

    # identify any unseen labels in the test set
    unseen_test_label_list = [label for label in test_label_list
                              if label not in train_label_list]

    # create a new dictionary for these unseen labels with label indices
    # for them starting _after_ those for the training set labels
    unseen_label_dict = {label: i for i, label in enumerate(unseen_test_label_list,
                                                            start=len(train_label_list))}

    # combine the train label dictionary with this unseen label one & return
    train_and_test_label_dict = train_label_dict.copy()
    train_and_test_label_dict.update(unseen_label_dict)
    return train_and_test_label_dict


def compute_evaluation_metrics(metrics,
                               labels,
                               predictions,
                               model_type,
                               label_dict=None,
                               grid_objective=None,
                               probability=False,
                               logger=None):
    """
    Compute given metrics to evaluate the given predictions generated
    by the given type of estimator against the given true labels.

    Parameters
    ----------
    metrics : list of str
        List of metrics to compute.
    labels : array-like
        True labels to be used for computing the metrics.
    predictions : array-like
        The predictions to be used for computing the metrics.
    model_type : str
        One of "classifier" or "regressor".
    label_dict : dict, optional
        Dictionary mapping class labels to indices for classification.
        Defaults to ``None``.
    grid_objective : str, optional
        The objective used for tuning the hyper-parameters of the model
        that generated the predictions. If ``None``, it means that no
        grid search was done.
        Defaults to ``None``.
    probability : bool, optional
        Does the model output class probabilities?
        Defaults to ``False``.
    logger : logging.Logger, optional
        A logger instance to use for logging messages and warnings.
        If ``None``, a new one is created.
        Defaults to ``None``.

    Returns
    -------
    res : 5-tuple
        The confusion matrix, the overall accuracy, the per-label
        PRFs, the grid search objective function score, and the
        additional evaluation metrics, if any. For regressors, the
        first two elements are ``None``.
    """
    # set up the logger
    logger = logger if logger else logging.getLogger(__name__)

    # warn if grid objective was also specified in metrics
    if len(metrics) > 0 and grid_objective in metrics:
        logger.warning(f"The grid objective '{grid_objective}' is also "
                       "specified as an evaluation metric. Since its value "
                       "is already included in the results as the objective "
                       "score, it will not be printed again in the list of "
                       "metrics.")
        metrics = [metric for metric in metrics if metric != grid_objective]

    # initialize a dictionary that will hold all of the metric scores
    metric_scores = {metric: None for metric in metrics}

    # if we are doing classification and are a probablistic
    # learner or a soft-voting meta learner, then `yhat` are
    # probabilities so we need to compute the class indices
    # separately and save them too
    if model_type == 'classifier' and probability:
        class_probs = predictions
        predictions = np.argmax(class_probs, axis=1)
    # if we are a regressor or classifier not in probability
    # mode, then we have the class indices already and there
    # are no probabilities
    else:
        class_probs = None

    # make a single list of metrics including the grid objective
    # since it's easier to compute everything together
    metrics_to_compute = [grid_objective] + metrics
    for metric in metrics_to_compute:

        # skip the None if we are not doing grid search
        if not metric:
            continue

        # CASE 1: in probability mode for classification which means we
        # need to either use the probabilities directly or infer the labels
        # from them depending on the metric
        if probability:

            # there are three possible cases here:
            # (a) if we are using a correlation metric or
            #     `average_precision` or `roc_auc` in a binary
            #      classification scenario, then we need to explicitly
            #     pass in the probabilities of the positive class.
            # (b) if we are using `neg_log_loss`, then we
            #     just pass in the full probability array
            # (c) we compute the most likely labels from the
            #     probabilities via argmax and use those
            #     for all other metrics
            if (len(label_dict) == 2 and
                    (metric in CORRELATION_METRICS or
                     metric in ['average_precision', 'roc_auc']) and
                    metric != grid_objective):
                logger.info("using probabilities for the positive class to "
                            f"compute '{metric}' for evaluation.")
                preds_for_metric = class_probs[:, 1]
            elif metric == 'neg_log_loss':
                preds_for_metric = class_probs
            else:
                preds_for_metric = predictions

        # CASE 2: no probability mode for classifier or regressor
        # in which case we just use the predictions as they are
        else:
            preds_for_metric = predictions

        try:
            metric_scores[metric] = use_score_func(metric, labels, preds_for_metric)
        except ValueError:
            metric_scores[metric] = float('NaN')

    # now separate out the grid objective score from the additional metric scores
    # if a grid objective was actually passed in. If no objective was passed in
    # then that score should just be none.
    objective_score = None
    additional_scores = metric_scores.copy()
    if grid_objective:
        objective_score = metric_scores[grid_objective]
        del additional_scores[grid_objective]

    # compute some basic statistics for regressors
    if model_type == 'regressor':
        result_dict = {'descriptive': defaultdict(dict)}
        for table_label, y in zip(['actual', 'predicted'], [labels, predictions]):
            result_dict['descriptive'][table_label]['min'] = min(y)
            result_dict['descriptive'][table_label]['max'] = max(y)
            result_dict['descriptive'][table_label]['avg'] = np.mean(y)
            result_dict['descriptive'][table_label]['std'] = np.std(y)
        result_dict['pearson'] = use_score_func('pearson', labels, predictions)
        res = (None, None, result_dict, objective_score, additional_scores)
    else:
        # compute the confusion matrix and precision/recall/f1
        # note that we are using the class indices here
        # and not the actual class labels themselves
        num_labels = len(label_dict)
        conf_mat = confusion_matrix(labels,
                                    predictions,
                                    labels=list(range(num_labels)))
        # Calculate metrics
        overall_accuracy = accuracy_score(labels, predictions)
        result_matrix = precision_recall_fscore_support(
            labels, predictions, labels=list(range(num_labels)), average=None)

        # Store results
        result_dict = defaultdict(dict)
        for actual_label in sorted(label_dict):
            col = label_dict[actual_label]
            result_dict[actual_label]["Precision"] = result_matrix[0][col]
            result_dict[actual_label]["Recall"] = result_matrix[1][col]
            result_dict[actual_label]["F-measure"] = result_matrix[2][col]

        res = (conf_mat.tolist(), overall_accuracy, result_dict,
               objective_score, additional_scores)

    return res


def compute_num_folds_from_example_counts(cv_folds,
                                          labels,
                                          model_type,
                                          logger=None):
    """
    Calculate the number of folds we should use for cross-validation, based
    on the number of examples we have for each label.

    Parameters
    ----------
    cv_folds : int
        The number of cross-validation folds.
    labels : list
        The example labels.
    model_type : str
        One of "classifier" or "regressor".
    logger : logging.Logger, optional
        A logger instance to use for logging messages and warnings.
        If ``None``, a new one is created.
        Defaults to ``None``.

    Returns
    -------
    cv_folds : int
        The number of folds to use, based on the number of examples
        for each label.

    Raises
    ------
    ValueError
        If ``cv_folds`` is not an integer or if the training set has
        fewer than 2 examples associated with a label (for classification).
    """
    try:
        assert isinstance(cv_folds, int)
    except AssertionError:
        raise ValueError("`cv_folds` must be an integer.")

    # For regression models, we can just return the current cv_folds
    if model_type == 'regressor':
        return cv_folds

    min_examples_per_label = min(Counter(labels).values())
    if min_examples_per_label <= 1:
        raise ValueError(f"The training set has only {min_examples_per_label}"
                         " example for a label.")
    if min_examples_per_label < cv_folds:
        logger.warning("The minimum number of examples per label was "
                       f"{min_examples_per_label}. Setting the number of "
                       "cross-validation folds to that value.")
        cv_folds = min_examples_per_label
    return cv_folds


def contiguous_ints_or_floats(numbers):
    """
    Check whether the given list of numbers contains
    contiguous integers or contiguous integer-like
    floats. For example, [1, 2, 3] or [4.0, 5.0, 6.0]
    are both contiguous but [1.1, 1.2, 1.3] is not.

    Parameters
    ----------
    numbers : array-like of ints or floats
        The numbers we want to check.

    Returns
    -------
    answer : bool
        True if the numbers are contiguous integers
        or contiguous integer-like floats (1.0, 2.0, etc.)

    Raises
    ------
    TypeError
        If ``numbers`` does not contain integers or floating
        point values.
    ValueError
        If ``numbers`` is empty.
    """

    try:

        # make sure that number is not empty
        assert len(numbers) > 0

        # first check that the numbers are all integers
        # or integer-like floats (e.g., 1.0, 2.0 etc.)
        ints_or_int_like_floats = np.all(np.mod(numbers, 1) == 0)

        # next check that the successive differences between
        # the numbers are all 1, i.e., they are nuermicontiguous
        contiguous = np.all(np.diff(numbers) == 1)

    except AssertionError:
        raise ValueError('Input cannot be empty.')

    except TypeError:
        raise TypeError('Input should only contain numbers.')

    # we need both conditions to be true
    return ints_or_int_like_floats and contiguous


def get_acceptable_classification_metrics(label_array):
    """
    Return the set of metrics that are acceptable given the
    the unique set of labels that we are classifying.

    Parameters
    ----------
    label_array : numpy.ndarray
        A sorted numpy array containing the unique labels
        that we are trying to predict. Optional for regressors
        but required for classifiers.

    Returns
    -------
    acceptable_metrics : set
        A set of metric names that are acceptable
        for the given classification scenario.
    """

    # this is a classifier so the acceptable objective
    # functions definitely include those metrics that
    # are specifically for classification and also
    # the unweighted kappa metrics
    acceptable_metrics = CLASSIFICATION_ONLY_METRICS | UNWEIGHTED_KAPPA_METRICS

    # now let us consider which other metrics may also
    # be acceptable depending on whether the labels
    # are strings or (contiguous) integers/floats
    label_type = label_array.dtype.type

    # CASE 1: labels are strings, then no other metrics
    # are acceptable
    if issubclass(label_type, (np.object_, str)):
        pass

    # CASE 2: labels are integers or floats; the way
    # it works in SKLL, it's guaranteed that
    # class indices will be sorted in the same order
    # as the class labels therefore, ranking metrics
    # such as various correlations should work fine.
    elif issubclass(label_type, (int,
                                 np.int32,
                                 np.int64,
                                 float,
                                 np.float32,
                                 np.float64)):
        acceptable_metrics.update(CORRELATION_METRICS)

        # CASE 3: labels are numerically contiguous integers
        # this is a special sub-case of CASE 2 which
        # represents ordinal classification. Only in this
        # case, weighted kappas -- where the distance
        # between the class labels has a special
        # meaning -- can be allowed. This is because
        # class indices are always contiguous and all
        # metrics in SKLL are computed in the index
        # space, not the label space. Note that floating
        # point numbers that are equivalent to integers
        # (e.g., [1.0, 2.0, 3.0]) are also acceptable.
        if contiguous_ints_or_floats(label_array):
            acceptable_metrics.update(WEIGHTED_KAPPA_METRICS)

    # if there are any custom metrics registered, include them too
    if len(_CUSTOM_METRICS) > 0:
        acceptable_metrics.update(_CUSTOM_METRICS)

    return acceptable_metrics


def get_acceptable_regression_metrics():
    """
    Return the set of metrics that are acceptable for regression.
    """

    # it's fairly straightforward for regression since
    # we do not have to check the labels
    acceptable_metrics = (REGRESSION_ONLY_METRICS |
                          UNWEIGHTED_KAPPA_METRICS |
                          WEIGHTED_KAPPA_METRICS |
                          CORRELATION_METRICS)

    # if there are any custom metrics registered, include them too
    if len(_CUSTOM_METRICS) > 0:
        acceptable_metrics.update(_CUSTOM_METRICS)

    return acceptable_metrics


def load_custom_learner(custom_learner_path, custom_learner_name):
    """
    Import and load the custom learner object from the given path.

    Parameters
    ----------
    custom_learner_path : str
        The path to a custom learner.
    custom_learner_name : str
        The name of a custom learner.

    Raises
    ------
    ValueError
        If the custom learner path does not end in '.py'.

    Returns
    -------
    custom_learner_obj : skll.Learner object
        The SKLL learner object loaded from the given path.
    """
    if not custom_learner_path:
        raise ValueError('custom_learner_path was not set and learner '
                         f'{custom_learner_name} was not found.')

    if not custom_learner_path.endswith('.py'):
        raise ValueError('custom_learner_path must end in .py '
                         f'({custom_learner_path})')

    custom_learner_module_name = os.path.basename(custom_learner_path)[:-3]
    sys.path.append(os.path.dirname(os.path.abspath(custom_learner_path)))
    import_module(custom_learner_module_name)
    return getattr(sys.modules[custom_learner_module_name], custom_learner_name)


def get_predictions(learner, xtest):
    """
    Get various different types of predictions generated by the given
    learner (or meta-learner) on the given feature array.

    The various types of predictions include:

    - "raw" predictions which are self-explanatory for regressors; for
      classifiers, these are the indices of the class labels, not the labels
      themselves.
    - "labels": for classifiers, these are the class labels; for regressors
      they are not applicable and represented as ``None``.
    - "probabilities": for classifiers, these are the class probabilities; for
      non-probabilistic classifiers or regressors, they are not applicable and
      represented as ``None``.

    Parameters
    ----------
    learner : skll.Learner
        The already-trained ``Learner`` or ``VotingLearner`` instance that is
        used to generate the predictions.
    xtest : array-like
        Numpy array of features on which the predictions are to be made.

    Returns
    -------
    prediction_dict : dict
        Dictionary containing the three types of predictions as the keys
        and either ``None`` or a numpy array as the value.

    Raises
    ------
    NotImplementedError
        If the scikit-learn model does not implement ``predict_proba()`` to
        get the class probabilities.
    """

    # deferred import to avoid circular dependencies
    from skll.learner.voting import VotingLearner

    # initialize the prediction dictionary
    prediction_dict = {"raw": None, "labels": None, "probabilities": None}

    # first get the raw predictions from the underlying scikit-learn model
    # this works for both classifiers and regressors
    yhat = learner.model.predict(xtest)
    prediction_dict["raw"] = yhat

    # next, if it's a classifier ...
    if learner.model_type._estimator_type == "classifier":

        # get the list of labels from the learner (or meta-learner)
        if isinstance(learner, VotingLearner):
            label_list = learner.learners[0].label_list
        else:
            label_list = learner.label_list

        # get the predicted class labels
        class_labels = np.array([label_list[int(pred)] for pred in yhat])
        prediction_dict["labels"] = class_labels

        # then get the class probabilities too if the learner
        # (or meta-learner) is probabilistic
        if ((hasattr(learner, "probability") and learner.probability) or
                (hasattr(learner, "voting") and learner.voting == "soft")):
            try:
                yhat_probs = learner.model.predict_proba(xtest)
            except NotImplementedError as e:
                learner.logger.error(f"Model type: {learner.model_type.__name__}\n"
                                     f"Model: {learner.model}\n")
                raise e
            else:
                prediction_dict["probabilities"] = yhat_probs

    return prediction_dict


def rescaled(cls):
    """
    Decorator to create regressors that store a min and a max for the training
    data and make sure that predictions fall within that range.  It also stores
    the means and SDs of the gold standard and the predictions on the training
    set to rescale the predictions (e.g., as in e-rater).

    Parameters
    ----------
    cls : BaseEstimator
        An estimator class to add rescaling to.

    Returns
    -------
    cls : BaseEstimator
        Modified version of estimator class with rescaled functions added.

    Raises
    ------
    ValueError
        If classifier cannot be rescaled (i.e. is not a regressor).
    """
    # If this class has already been run through the decorator, return it
    if hasattr(cls, 'rescale'):
        return cls

    # Save original versions of functions to use later.
    orig_init = cls.__init__
    orig_fit = cls.fit
    orig_predict = cls.predict

    if cls._estimator_type == 'classifier':
        raise ValueError('Classifiers cannot be rescaled. Only regressors '
                         'can.')

    # Define all new versions of functions
    @wraps(cls.fit)
    def fit(self, X, y=None):
        """
        Fit a model, then store the mean, SD, max and min of the training set
        and the mean and SD of the predictions on the training set.

        Parameters
        ----------
        X : array-like, with shape (n_samples, n_features)
            The data to fit.
        y : Ignored

        Returns
        -------
        self
        """

        # fit a regular regression model
        orig_fit(self, X, y=y)

        if self.constrain:
            # also record the training data min and max
            self.y_min = min(y)
            self.y_max = max(y)

        if self.rescale:
            # also record the means and SDs for the training set
            y_hat = orig_predict(self, X)
            self.yhat_mean = np.mean(y_hat)
            self.yhat_sd = np.std(y_hat)
            self.y_mean = np.mean(y)
            self.y_sd = np.std(y)

        return self

    @wraps(cls.predict)
    def predict(self, X):
        """
        Make predictions with the super class, and then adjust them using the
        stored min, max, means, and standard deviations.

        Parameters
        ----------
        X : array-like, with shape (n_samples,)
            The data to predict.

        Returns
        -------
        res : array-like
            The prediction results.
        """
        # get the unconstrained predictions
        res = orig_predict(self, X)

        if self.rescale:
            # convert the predictions to z-scores,
            # then rescale to match the training set distribution
            res = (((res - self.yhat_mean) / self.yhat_sd) * self.y_sd) + self.y_mean

        if self.constrain:
            # apply min and max constraints
            res = np.array([max(self.y_min, min(self.y_max, pred))
                            for pred in res])

        return res

    @classmethod
    @wraps(cls._get_param_names)
    def _get_param_names(class_x):
        """
        This is adapted from scikit-learns's ``BaseEstimator`` class.
        It gets the kwargs for the superclass's init method and adds the
        kwargs for newly added ``__init__()`` method.

        Parameters
        ----------
        class_x
            The the superclass from which to retrieve param names.

        Returns
        -------
        args : list
            A list of parameter names for the class's init method.

        Raises
        ------
        RunTimeError
            If `varargs` exist in the scikit-learn estimator.
        """

        # initialize the empty list of parameter names
        args = []

        try:
            # get signature of the original init method
            init = getattr(orig_init, 'deprecated_original', orig_init)
            init_signature = inspect.signature(init)

            # get all parameters excluding 'self'
            original_parameters = [p for p in init_signature.parameters.values()
                                   if p.name != 'self' and p.kind != p.VAR_KEYWORD]

            # there should be no varargs
            for parameter in original_parameters:
                if parameter.kind == parameter.VAR_POSITIONAL:
                    raise RuntimeError(
                        "scikit-learn estimators should always specify their "
                        "parameters in the signature of their __init__ (no "
                        f"varargs). {cls} with constructor {init_signature} "
                        "doesn't follow this convention."
                    )
                else:
                    args.append(parameter.name)

        except TypeError:
            pass

        # now get the additional rescaling arguments
        rescale_args = inspect.getargspec(class_x.__init__)[0]

        # Remove 'self'
        rescale_args.pop(0)

        # add the rescaling arguments to the original arguments and sort
        args += rescale_args
        args.sort()

        return args

    @wraps(cls.__init__)
    def init(self, constrain=True, rescale=True, **kwargs):
        """
        This special init function is used by the decorator to make sure
        that things get initialized in the right order.

        Parameters
        ----------
        constrain : bool, optional
            Whether to constrain predictions within min and max values.
            Defaults to True.
        rescale : bool, optional
            Whether to rescale prediction values using z-scores.
            Defaults to True.
        kwargs : dict, optional
            Arguments for base class.
        """
        # pylint: disable=W0201
        self.constrain = constrain
        self.rescale = rescale
        self.y_min = None
        self.y_max = None
        self.yhat_mean = None
        self.yhat_sd = None
        self.y_mean = None
        self.y_sd = None
        orig_init(self, **kwargs)

    # Override original functions with new ones
    cls.__init__ = init
    cls.fit = fit
    cls.predict = predict
    cls._get_param_names = _get_param_names
    cls.rescale = True

    # Return modified class
    return cls


def setup_cv_fold_iterator(cv_folds,
                           examples,
                           model_type,
                           stratified=False,
                           logger=None):
    """
    Set up a cross-validation fold iterator for the given ``FeatureSet``.

    Parameters
    ----------
    cv_folds : int or dict
        The number of folds to use for cross-validation, or
        a mapping from example IDs to folds.
    examples : skll.FeatureSet
        The ``FeatureSet`` instance for which the CV iterator is to be computed.
    model_type : str
        One of "classifier" or "regressor".
    stratified : bool, optional
        Should the cross-validation iterator be set up in a stratified
        fashion?
        Defaults to ``False``.
    logger : logging.Logger, optional
        A logger instance to use for logging messages and warnings.
        If ``None``, a new one is created.
        Defaults to ``None``.

    Returns
    -------
    res : a 2-tuple
        The first element is the the kfold iterator and the second
        is the list of cross-validation groups.
    """
    # Set up the cross-validation iterator.
    if isinstance(cv_folds, int):
        cv_folds = compute_num_folds_from_example_counts(cv_folds,
                                                         examples.labels,
                                                         model_type,
                                                         logger=logger)

        stratified = (stratified and model_type == 'classifier')
        if stratified:
            kfold = StratifiedKFold(n_splits=cv_folds)
            cv_groups = None
        else:
            kfold = KFold(n_splits=cv_folds)
            cv_groups = None
    # Otherwise cv_folds is a dict
    else:
        # if we have a mapping from IDs to folds, use it for the overall
        # cross-validation as well as the grid search within each
        # training fold.  Note that this means that the grid search
        # will use K-1 folds because the Kth will be the test fold for
        # the outer cross-validation.
        dummy_label = next(iter(cv_folds.values()))
        fold_groups = [cv_folds.get(curr_id, dummy_label) for curr_id in examples.ids]
        # Only retain IDs within folds if they're in cv_folds
        kfold = FilteredLeaveOneGroupOut(cv_folds,
                                         examples.ids,
                                         logger=logger)
        cv_groups = fold_groups

    return kfold, cv_groups


def setup_cv_split_iterator(cv_folds, examples):
    """
    Set up a cross-validation split iterator over the given ``FeatureSet``.

    Parameters
    ----------
    cv_folds : int or dict
        The number of folds to use for cross-validation, or
        a mapping from example IDs to folds.
    examples : skll.FeatureSet
        The given featureset which is to be split.

    Returns
    -------
    res : a 2-tuple
        The first element is an iterator over the train/test featuresets
        and the second is the maximum number of training samples available.
    """
    # seed the random number generator for replicability
    random_state = np.random.RandomState(123456789)

    # set up the cross-validation split iterator with 20% of
    # the data always reserved for testing
    cv = ShuffleSplit(n_splits=cv_folds,
                      test_size=0.2,
                      random_state=random_state)
    cv_iter = list(cv.split(examples.features, examples.labels, None))
    n_max_training_samples = len(cv_iter[0][0])

    # create an iterator over train/test featuresets based on the
    # cross-validation index iterator
    featureset_iter = (FeatureSet.split_by_ids(examples, train, test)
                       for train, test in cv_iter)

    return featureset_iter, n_max_training_samples


def train_and_score(learner,
                    train_examples,
                    test_examples,
                    metric):
    """
    A utility method to train a given learner instance on the given
    training examples, generate predictions on the training set itself
    and also the given test set, and score those predictions using the
    given metric. The method returns the train and test scores.

    If the learner has its ``probability`` attribute set to ``True``, it will
    produce probability values as predictions rather than class indices.
    In this case, this function will compute the argmax over the probability
    values to find the most likely class index and use that.

    Note that this method needs to be a top-level function since it is
    called from within ``joblib.Parallel()`` and, therefore, needs to be
    picklable which it would not be as an instancemethod of the ``Learner``
    class.

    Parameters
    ----------
    learner : skll.Learner
        A SKLL ``Learner`` instance.
    train_examples : array-like, with shape (n_samples, n_features)
        The training examples.
    test_examples : array-like, of length n_samples
        The test examples.
    metric : str
        The scoring function passed to ``use_score_func()``.

    Returns
    -------
    train_score : float
        Output of the score function applied to predictions of
        ``learner`` on ``train_examples``.
    test_score : float
        Output of the score function applied to predictions of
        ``learner`` on ``test_examples``.
    """
    _ = learner.train(train_examples, grid_search=False, shuffle=False)

    # get the train and test class probabilities or indices (not labels)
    train_predictions = learner.predict(train_examples, class_labels=False)
    test_predictions = learner.predict(test_examples, class_labels=False)

    # recall that voting learners return a tuple from `predict()`
    if isinstance(train_predictions, tuple):
        train_predictions = train_predictions[0]
    if isinstance(test_predictions, tuple):
        test_predictions = test_predictions[0]

    # if we got probabilities, then we need to run argmax over them
    # to convert them into indices; this needs to handle both
    # regular learners as well as voting learners
    if ((hasattr(learner, "probability") and learner.probability) or
            (hasattr(learner, "voting") and learner.voting == "soft")):
        train_predictions = np.argmax(train_predictions, axis=1)
        test_predictions = np.argmax(test_predictions, axis=1)

    # now get the training and test labels and convert them to indices
    # but make sure to include any unseen labels in the test data
    if learner.model_type._estimator_type == 'classifier':
        test_label_list = np.unique(test_examples.labels).tolist()
        train_and_test_label_dict = add_unseen_labels(learner.label_dict,
                                                      test_label_list)
        train_labels = np.array([train_and_test_label_dict[label]
                                 for label in train_examples.labels])
        test_labels = np.array([train_and_test_label_dict[label]
                                for label in test_examples.labels])
    else:
        train_labels = train_examples.labels
        test_labels = test_examples.labels

    # now compute and return the scores
    train_score = use_score_func(metric, train_labels, train_predictions)
    test_score = use_score_func(metric, test_labels, test_predictions)
    return train_score, test_score


def write_predictions(example_ids,
                      predictions_to_write,
                      file_prefix,
                      model_type,
                      append=False,
                      label_list=None,
                      probability=False):
    """
    Write example IDs and predictions to a tab-separated file with given prefix.

    Parameters
    ----------
    example_ids : array-like
        The IDs of the examples for which the predictions have been generated.
    predictions_to_write : array-like
        The predictions to write out to the file.
    file_prefix : TYPE
        The prefix for the output file. The output file will be named
        "<file_prefix>_predictions.tsv".
    model_type : str
        One of "classifier" or "regressor".
    append : bool, optional
        Should we append the current predictions to the file if it exists?
        Defaults to ``False``.
    label_list : list of str, optional
        List of class labels, required if ``probability`` is ``True``.
        Defaults to ``None``.
    probability : bool, optional
        Are the predictions class probabilities? If ``True``, requires
        ``label_list`` to be specified.
        Defaults to ``False``.
    """
    # create a new file starting with the given prefix
    prediction_file = f"{file_prefix}_predictions.tsv"
    with open(prediction_file,
              mode='w' if not append else 'a',
              newline='') as predictionfh:

        # create a DictWriter with the appropriate field names
        if predictions_to_write.ndim > 1:
            fieldnames = ["id"] + [label for label in label_list]
        else:
            fieldnames = ["id", "prediction"]
        writer = DictWriter(predictionfh, fieldnames=fieldnames, dialect=excel_tab)

        # write out the header unless we are appending
        if not append:
            writer.writeheader()

        for example_id, pred in zip(example_ids, predictions_to_write):

            # for regressors, we just write out the prediction as-is
            if model_type == "regressor":
                row = {"id": example_id, "prediction": pred}

            # if we have an array as a prediction, it must be
            # a list of probabilities and if not, then it's
            # either a class label or an index
            else:
                if isinstance(pred, np.ndarray):
                    row = {'id': example_id}
                    row.update(dict(zip(label_list, pred)))
                else:
                    row = {"id": example_id, "prediction": pred}

            # write out the row
            writer.writerow(row)


def _save_learner_to_disk(learner, filepath):
    """
    Save the given SKLL learner instance to disk.

    NOTE: This function should only be used by the ``save()`` methods
    for the various learner classes in SKLL.

    Parameters
    ----------
    learner : skll.learner.Learner or skll.learner.voting.VotingLearner
        A ``Learner`` or ``VotingLearner`` instance to save to disk.
    filepath : str
        The path to save the learner instance to.
    """
    # create the directory if it doesn't exist
    learner_dir = Path(filepath).parent
    if not learner_dir.exists():
        learner_dir.mkdir(parents=True)
    # write out the learner to disk
    joblib.dump((VERSION, learner), filepath)


def _load_learner_from_disk(learner_type, filepath, logger):
    """
    Load a saved instance of the given type from disk.

    NOTE: This function should only be used by the ``from_file()``
    methods for the various learner classes in SKLL.

    Parameters
    ----------
    learner_type : skll.learner.Learner or skll.learner.voting.VotingLearner
        The type of learner instance to load from disk.
    filepath : str
        The path to a saved ``Learner`` or ``VotingLearner`` file.
    logger : logging object
        A logging object.

    Returns
    -------
    learner : skll.learner.Learner or skll.learner.voting.VotingLearner
        The ``Learner`` or ``VotingLearner`` instance loaded from the file.

    Raises
    ------
    ValueError
        If the pickled version of the ``Learner`` instance is out of date.
    """
    skll_version, learner = joblib.load(filepath)

    # Check that we've actually loaded an instance of the requested type
    if not isinstance(learner, learner_type):
        raise ValueError(f"'{filepath}' does not contain an object "
                         f"of type '{learner_type.__name__}'.")

    # check that versions are compatible
    elif skll_version < (2, 5, 0):
        model_version_str = '.'.join(map(str, skll_version))
        current_version_str = '.'.join(map(str, VERSION))
        raise ValueError(f"The learner stored in '{filepath}' was "
                         f"created with v{model_version_str} of SKLL, "
                         "which is incompatible with the current "
                         f"v{current_version_str}.")
    else:
        if not hasattr(learner, 'sampler'):
            learner.sampler = None

        # For backward compatibility, convert string model types to actual classes
        if isinstance(learner._model_type, str):
            learner._model_type = globals()[learner._model_type]

        # set the learner logger attribute to the logger that's passed in
        learner.logger = logger

        # if the loaded learner is a `VotingLearner` then we need to attach
        # the same logger to the underlying learners as well
        if learner_type.__name__ == "VotingLearner":
            for underlying_learner in learner._learners:
                underlying_learner.logger = logger

        return learner
