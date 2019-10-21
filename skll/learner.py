# License: BSD 3 clause
"""
Provides easy-to-use wrapper around scikit-learn.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
:organization: ETS
"""
# pylint: disable=F0401,W0622,E1002,E1101

import copy
import inspect
import logging
import os
import sys

from collections import Counter, defaultdict
from functools import wraps
from math import floor, log10
from importlib import import_module
from itertools import combinations
from multiprocessing import cpu_count

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (GridSearchCV,
                                     KFold,
                                     LeaveOneGroupOut,
                                     ShuffleSplit,
                                     StratifiedKFold)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier,
                              AdaBoostRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer as OldDictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
# AdditiveChi2Sampler is used indirectly, so ignore linting message
from sklearn.kernel_approximation import (Nystroem,
                                          RBFSampler,
                                          SkewedChi2Sampler)
from sklearn.linear_model import (BayesianRidge,
                                  ElasticNet,
                                  HuberRegressor,
                                  Lars,
                                  Lasso,
                                  LinearRegression,
                                  LogisticRegression,
                                  RANSACRegressor,
                                  Ridge,
                                  RidgeClassifier,
                                  SGDClassifier,
                                  SGDRegressor,
                                  TheilSenRegressor)
from sklearn.linear_model.base import LinearModel
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle as sk_shuffle

from skll.data import FeatureSet
from skll.data.dict_vectorizer import DictVectorizer
from skll.data.readers import safe_float
from skll.metrics import (_CLASSIFICATION_ONLY_METRICS,
                          _CORRELATION_METRICS,
                          _REGRESSION_ONLY_METRICS,
                          _UNWEIGHTED_KAPPA_METRICS,
                          _WEIGHTED_KAPPA_METRICS,
                          SCORERS,
                          use_score_func)
from skll.version import VERSION

# Constants #
_DEFAULT_PARAM_GRIDS = {AdaBoostClassifier:
                        [{'learning_rate': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        AdaBoostRegressor:
                        [{'learning_rate': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        BayesianRidge:
                        [{'alpha_1': [1e-6, 1e-4, 1e-2, 1, 10],
                          'alpha_2': [1e-6, 1e-4, 1e-2, 1, 10],
                          'lambda_1': [1e-6, 1e-4, 1e-2, 1, 10],
                          'lambda_2': [1e-6, 1e-4, 1e-2, 1, 10]}],
                        DecisionTreeClassifier:
                        [{'max_features': ["auto", None]}],
                        DecisionTreeRegressor:
                        [{'max_features': ["auto", None]}],
                        DummyClassifier:
                        [{}],
                        DummyRegressor:
                        [{}],
                        ElasticNet:
                        [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        GradientBoostingClassifier:
                        [{'max_depth': [1, 3, 5]}],
                        GradientBoostingRegressor:
                        [{'max_depth': [1, 3, 5]}],
                        HuberRegressor:
                        [{'epsilon': [1.05, 1.35, 1.5, 2.0, 2.5, 5.0],
                          'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}],
                        KNeighborsClassifier:
                        [{'n_neighbors': [1, 5, 10, 100],
                          'weights': ['uniform', 'distance']}],
                        KNeighborsRegressor:
                        [{'n_neighbors': [1, 5, 10, 100],
                          'weights': ['uniform', 'distance']}],
                        MLPClassifier:
                        [{'activation': ['logistic', 'tanh', 'relu'],
                          'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                          'learning_rate_init': [0.001, 0.01, 0.1]}],
                        MLPRegressor:
                        [{'activation': ['logistic', 'tanh', 'relu'],
                          'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                          'learning_rate_init': [0.001, 0.01, 0.1]}],
                        MultinomialNB:
                        [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}],
                        Lars:
                        [{}],
                        Lasso:
                        [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        LinearRegression:
                        [{}],
                        LinearSVC:
                        [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        LogisticRegression:
                        [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        SVC:
                        [{'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                          'gamma': ['auto', 'scale', 0.01, 0.1, 1.0, 10.0, 100.0]}],
                        RandomForestClassifier:
                        [{'max_depth': [1, 5, 10, None]}],
                        RandomForestRegressor:
                        [{'max_depth': [1, 5, 10, None]}],
                        RANSACRegressor:
                        [{}],
                        Ridge:
                        [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        RidgeClassifier:
                        [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        SGDClassifier:
                        [{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                          'penalty': ['l1', 'l2', 'elasticnet']}],
                        SGDRegressor:
                        [{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                          'penalty': ['l1', 'l2', 'elasticnet']}],
                        LinearSVR:
                        [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        SVR:
                        [{'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                          'gamma': ['auto', 'scale', 0.01, 0.1, 1.0, 10.0, 100.0]}],
                        TheilSenRegressor:
                        [{}]}

_REQUIRES_DENSE = (BayesianRidge, Lars, TheilSenRegressor)

MAX_CONCURRENT_PROCESSES = int(os.getenv('SKLL_MAX_CONCURRENT_PROCESSES', '3'))


# pylint: disable=W0223,R0903
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
        return X.todense()


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
                self.logger.warning('Feature set contains IDs that are not ' +
                                    'in folds dictionary. Skipping those IDs.')
                self._warned = True

            yield train_index, test_index


def _contiguous_ints_or_floats(numbers):
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


def _find_default_param_grid(cls):
    """
    Finds the default parameter grid for the specified classifier.

    Parameters
    ----------
    cls
        A parent classifier class to check, and find the
        default param grid.

    Returns
    -------
    grid : list of dicts or None
        The parameters grid for a given classifier.
    """
    for key_cls, grid in _DEFAULT_PARAM_GRIDS.items():
        if issubclass(cls, key_cls):
            return grid
    return None


def _import_custom_learner(custom_learner_path, custom_learner_name):
    """
    Does the gruntwork of adding the custom model's module to globals.

    Parameters
    ----------
    custom_learner_path : str
        The path to a custom learner.
    custom_learner_name : str
        The name of a custom learner.

    Raises
    ------
    ValueError
        If the custom learner path is None.
    ValueError
        If the custom learner path does not end in '.py'.
    """
    if not custom_learner_path:
        raise ValueError('custom_learner_path was not set and learner {} '
                         'was not found.'.format(custom_learner_name))

    if not custom_learner_path.endswith('.py'):
        raise ValueError('custom_learner_path must end in .py ({})'
                         .format(custom_learner_path))

    custom_learner_module_name = os.path.basename(custom_learner_path)[:-3]
    sys.path.append(os.path.dirname(os.path.abspath(custom_learner_path)))
    import_module(custom_learner_module_name)
    globals()[custom_learner_name] = \
        getattr(sys.modules[custom_learner_module_name], custom_learner_name)


def _train_and_score(learner,
                     train_examples,
                     test_examples,
                     metric):
    """
    A utility method to train a given learner instance on the given training examples,
    generate predictions on the training set itself and also the given
    test set, and score those predictions using the given metric.
    The method returns the train and test scores.

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
    train_predictions = learner.predict(train_examples)
    test_predictions = learner.predict(test_examples)
    if learner.model_type._estimator_type == 'classifier':
        test_label_list = np.unique(test_examples.labels).tolist()
        unseen_test_label_list = [label for label in test_label_list
                                  if label not in learner.label_list]
        unseen_label_dict = {label: i for i, label in enumerate(unseen_test_label_list,
                                                                start=len(learner.label_list))}
        # combine the two dictionaries
        train_and_test_label_dict = learner.label_dict.copy()
        train_and_test_label_dict.update(unseen_label_dict)
        train_labels = np.array([train_and_test_label_dict[label]
                                 for label in train_examples.labels])
        test_labels = np.array([train_and_test_label_dict[label]
                                for label in test_examples.labels])
    else:
        train_labels = train_examples.labels
        test_labels = test_examples.labels

    train_score = use_score_func(metric, train_labels, train_predictions)
    test_score = use_score_func(metric, test_labels, test_predictions)
    return train_score, test_score


def _get_acceptable_regression_metrics():
    """
    Return the set of metrics that are acceptable for regression.
    """

    # it's fairly straightforward for regression since
    # we do not have to check the labels
    acceptable_metrics = (_REGRESSION_ONLY_METRICS |
                          _UNWEIGHTED_KAPPA_METRICS |
                          _WEIGHTED_KAPPA_METRICS |
                          _CORRELATION_METRICS)
    return acceptable_metrics


def _get_acceptable_classification_metrics(label_array):
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
    acceptable_metrics = _CLASSIFICATION_ONLY_METRICS | _UNWEIGHTED_KAPPA_METRICS

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
        acceptable_metrics.update(_CORRELATION_METRICS)

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
        if _contiguous_ints_or_floats(label_array):
            acceptable_metrics.update(_WEIGHTED_KAPPA_METRICS)

    return acceptable_metrics


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
        raise ValueError('Classifiers cannot be rescaled. ' +
                         'Only regressors can.')

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
        try:
            init = getattr(orig_init, 'deprecated_original', orig_init)

            args, varargs, _, _ = inspect.getargspec(init)
            if varargs is not None:
                raise RuntimeError('scikit-learn estimators should always '
                                   'specify their parameters in the signature'
                                   ' of their init (no varargs).')
            # Remove 'self'
            args.pop(0)
        except TypeError:
            args = []

        rescale_args = inspect.getargspec(class_x.__init__)[0]
        # Remove 'self'
        rescale_args.pop(0)

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


# Rescaled regressors
@rescaled
class RescaledBayesianRidge(BayesianRidge):
    pass


@rescaled
class RescaledAdaBoostRegressor(AdaBoostRegressor):
    pass


@rescaled
class RescaledDecisionTreeRegressor(DecisionTreeRegressor):
    pass


@rescaled
class RescaledElasticNet(ElasticNet):
    pass


@rescaled
class RescaledGradientBoostingRegressor(GradientBoostingRegressor):
    pass


@rescaled
class RescaledHuberRegressor(HuberRegressor):
    pass


@rescaled
class RescaledKNeighborsRegressor(KNeighborsRegressor):
    pass


@rescaled
class RescaledLars(Lars):
    pass


@rescaled
class RescaledLasso(Lasso):
    pass


@rescaled
class RescaledLinearRegression(LinearRegression):
    pass


@rescaled
class RescaledLinearSVR(LinearSVR):
    pass


@rescaled
class RescaledMLPRegressor(MLPRegressor):
    pass


@rescaled
class RescaledRandomForestRegressor(RandomForestRegressor):
    pass


@rescaled
class RescaledRANSACRegressor(RANSACRegressor):
    pass


@rescaled
class RescaledRidge(Ridge):
    pass


@rescaled
class RescaledSGDRegressor(SGDRegressor):
    pass


@rescaled
class RescaledSVR(SVR):
    pass


@rescaled
class RescaledTheilSenRegressor(TheilSenRegressor):
    pass


class Learner(object):
    """
    A simpler learner interface around many scikit-learn classification
    and regression functions.

    Parameters
    ----------
    model_type : str
        Name of estimator to create (e.g., ``'LogisticRegression'``).
        See the skll package documentation for valid options.
    probability : bool, optional
        Should learner return probabilities of all
        labels (instead of just label with highest probability)?
        Defaults to ``False``.
    pipeline : bool, optional
        Should learner contain a pipeline attribute that
        contains a scikit-learn Pipeline object composed
        of all steps including the vectorizer, the feature
        selector, the sampler, the feature scaler, and the
        actual estimator. Note that this will increase the
        size of the learner object in memory and also when
        it is saved to disk.
        Defaults to ``False``.
    feature_scaling : str, optional
        How to scale the features, if at all. Options are
        -  'with_std': scale features using the standard deviation
        -  'with_mean': center features using the mean
        -  'both': do both scaling as well as centering
        -  'none': do neither scaling nor centering
        Defaults to 'none'.
    model_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the
        initializer for the specified model.
        Defaults to ``None``.
    pos_label_str : str, optional
        A string denoting the label of the class to be
        treated as the positive class in a binary classification
        setting. If ``None``, the class represented by the label
        that appears second when sorted is chosen as the positive
        class. For example, if the two labels in data are "A"
        and "B" and ``pos_label_str`` is not specified, "B" will
        be chosen as the positive class.
        Defaults to ``None``.
    min_feature_count : int, optional
        The minimum number of examples a feature
        must have a nonzero value in to be included.
        Defaults to 1.
    sampler : str, optional
        The sampler to use for kernel approximation, if desired.
        Valid values are
        -  'AdditiveChi2Sampler'
        -  'Nystroem'
        -  'RBFSampler'
        -  'SkewedChi2Sampler'
        Defaults to ``None``.
    sampler_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the
        initializer for the specified sampler.
        Defaults to ``None``.
    custom_learner_path : str, optional
        Path to module where a custom classifier is defined.
        Defaults to ``None``.
    logger : logging object, optional
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.
    """

    def __init__(self, model_type, probability=False, pipeline=False,
                 feature_scaling='none', model_kwargs=None, pos_label_str=None,
                 min_feature_count=1, sampler=None, sampler_kwargs=None,
                 custom_learner_path=None, logger=None):
        """
        Initializes a learner object with the specified settings.
        """
        super(Learner, self).__init__()

        self.feat_vectorizer = None
        self.scaler = None
        self.label_dict = None
        self.label_list = None
        self.pos_label_str = safe_float(pos_label_str) if pos_label_str is not None else pos_label_str
        self._model = None
        self._store_pipeline = pipeline
        self._feature_scaling = feature_scaling
        self.feat_selector = None
        self._min_feature_count = min_feature_count
        self._model_kwargs = {}
        self._sampler_kwargs = {}
        self.logger = logger if logger else logging.getLogger(__name__)

        if model_type not in globals():
            # here, we need to import the custom model and add it
            # to the appropriate lists of models.
            _import_custom_learner(custom_learner_path, model_type)
            model_class = globals()[model_type]

            default_param_grid = (model_class.default_param_grid()
                                  if hasattr(model_class, 'default_param_grid')
                                  else [{}])

            # ewww, globals :-(
            global _REQUIRES_DENSE

            _DEFAULT_PARAM_GRIDS.update({model_class: default_param_grid})
            if hasattr(model_class, 'requires_dense') and \
                    model_class.requires_dense():
                _REQUIRES_DENSE = _REQUIRES_DENSE + (model_class,)

        self._model_type = globals()[model_type]
        self._probability = None
        # Use setter to set self.probability
        self.probability = probability
        self._use_dense_features = \
            (issubclass(self._model_type, _REQUIRES_DENSE) or
             self._feature_scaling in {'with_mean', 'both'})

        # Set default keyword arguments for models that we have some for.
        if issubclass(self._model_type, SVC):
            self._model_kwargs['cache_size'] = 1000
            self._model_kwargs['probability'] = self.probability
            self._model_kwargs['gamma'] = 'scale'
            if self.probability:
                self.logger.warning('Because LibSVM does an internal '
                                    'cross-validation to produce probabilities, '
                                    'results will not be exactly replicable when '
                                    'using SVC and probability mode.')
        elif issubclass(self._model_type,
                        (RandomForestClassifier, RandomForestRegressor,
                         GradientBoostingClassifier, GradientBoostingRegressor,
                         AdaBoostClassifier, AdaBoostRegressor)):
            self._model_kwargs['n_estimators'] = 500
        elif issubclass(self._model_type, SVR):
            self._model_kwargs['cache_size'] = 1000
            self._model_kwargs['gamma'] = 'scale'
        elif issubclass(self._model_type, SGDClassifier):
            self._model_kwargs['loss'] = 'log'
            self._model_kwargs['max_iter'] = 1000
            self._model_kwargs['tol'] = 1e-3
        elif issubclass(self._model_type, SGDRegressor):
            self._model_kwargs['max_iter'] = 1000
            self._model_kwargs['tol'] = 1e-3
        elif issubclass(self._model_type, RANSACRegressor):
            self._model_kwargs['loss'] = 'squared_loss'
        elif issubclass(self._model_type, (MLPClassifier, MLPRegressor)):
            self._model_kwargs['learning_rate'] = 'invscaling'
            self._model_kwargs['max_iter'] = 500
        elif issubclass(self._model_type, LogisticRegression):
            self._model_kwargs['max_iter'] = 1000
            self._model_kwargs['solver'] = 'liblinear'
            self._model_kwargs['multi_class'] = 'auto'

        if issubclass(self._model_type,
                      (AdaBoostClassifier, AdaBoostRegressor,
                       DecisionTreeClassifier, DecisionTreeRegressor,
                       DummyClassifier, ElasticNet,
                       GradientBoostingClassifier,
                       GradientBoostingRegressor, Lasso, LinearSVC,
                       LinearSVR, LogisticRegression, MLPClassifier,
                       MLPRegressor, RandomForestClassifier,
                       RandomForestRegressor, RANSACRegressor, Ridge,
                       RidgeClassifier, SGDClassifier, SGDRegressor,
                       SVC, TheilSenRegressor)):
            self._model_kwargs['random_state'] = 123456789

        if sampler_kwargs:
            self._sampler_kwargs.update(sampler_kwargs)
        if sampler:
            sampler_type = globals()[sampler]
            if issubclass(sampler_type, (Nystroem, RBFSampler,
                                         SkewedChi2Sampler)):
                self._sampler_kwargs['random_state'] = 123456789
            self.sampler = sampler_type(**self._sampler_kwargs)
        else:
            self.sampler = None

        if model_kwargs:
            # if the model is an AdaBoost classifier or regressor or
            # a RANSAC regressor, then we need to convert any specified
            # `base_estimator` (a string) into an object before passing
            # it in to the learner constructor. We also need to make sure
            # where appropriate, we set the random state to a fixed seed
            # such that results are replicable
            if issubclass(self._model_type,
                          (AdaBoostRegressor,
                           AdaBoostClassifier,
                           RANSACRegressor)) and ('base_estimator' in model_kwargs):
                base_estimator_name = model_kwargs['base_estimator']
                if base_estimator_name in ['LinearRegression', 'MultinomialNB']:
                    base_estimator_kwargs = {}
                elif base_estimator_name in ['SGDClassifier', 'SGDRegressor']:
                    base_estimator_kwargs = {'max_iter': 1000,
                                             'tol': 0.001,
                                             'random_state': 123456789}
                elif base_estimator_name == 'SVR':
                    base_estimator_kwargs = {'gamma': 'scale'}
                elif base_estimator_name == 'SVC':
                    base_estimator_kwargs = {'gamma': 'scale', 'random_state': 123456789}
                else:
                    base_estimator_kwargs = {'random_state': 123456789}
                base_estimator = globals()[base_estimator_name](**base_estimator_kwargs)
                model_kwargs['base_estimator'] = base_estimator
            self._model_kwargs.update(model_kwargs)

    @classmethod
    def from_file(cls, learner_path, logger=None):
        """
        Load a saved ``Learner`` instance from a file path.

        Parameters
        ----------
        learner_path : str
            The path to a saved ``Learner`` instance file.
        logger : logging object, optional
            A logging object. If ``None`` is passed, get logger from ``__name__``.
            Defaults to ``None``.

        Returns
        -------
        learner : skll.Learner
            The ``Learner`` instance loaded from the file.

        Raises
        ------
        ValueError
            If the pickled object is not a ``Learner`` instance.
        ValueError
            If the pickled version of the ``Learner`` instance is out of date.
        """
        skll_version, learner = joblib.load(learner_path)

        # create the learner logger attribute to the logger that's passed in
        # or if nothing was passed in, then a new logger should be linked
        learner.logger = logger if logger else logging.getLogger(__name__)

        # For backward compatibility, convert string model types to labels.
        if isinstance(learner._model_type, str):
            learner._model_type = globals()[learner._model_type]

        # Check that we've actually loaded a Learner (or sub-class)
        if not isinstance(learner, cls):
            raise ValueError(('The pickle stored at {} does not contain ' +
                              'a {} object.').format(learner_path, cls))
        # Check that versions are compatible. (Currently, this just checks
        # that major versions match)
        elif skll_version >= (0, 9, 17):
            if not hasattr(learner, 'sampler'):
                learner.sampler = None
            # From v0.17.0 onwards, scikit-learn requires all scalers to have
            # the `scale_` instead of the `std_` parameter. So, we need to
            # make all old models adapt to this.
            if hasattr(learner, 'scaler'):
                new_scaler = copy.copy(learner.scaler)
                # We need to use `__dict__` because the `std_` has been
                # overridden to  just return the `scale_` value, and we
                # need the original value of `std_`.
                if (not hasattr(new_scaler, 'scale_') and
                        'std_' in new_scaler.__dict__):
                    new_scaler.scale_ = new_scaler.__dict__['std_']
                    learner.scaler = new_scaler
            return learner
        else:
            raise ValueError(("{} stored in pickle file {} was " +
                              "created with version {} of SKLL, which is " +
                              "incompatible with the current version " +
                              "{}").format(cls, learner_path,
                                           '.'.join(skll_version),
                                           '.'.join(VERSION)))

    @property
    def model_type(self):
        """
        The model type (i.e., the class)
        """
        return self._model_type

    @property
    def model_kwargs(self):
        """
        A dictionary of the underlying scikit-learn model's keyword arguments
        """
        return self._model_kwargs

    @property
    def model(self):
        """
        The underlying scikit-learn model
        """
        return self._model

    def load(self, learner_path):
        """
        Replace the current learner instance with a saved learner.

        Parameters
        ----------
        learner_path : str
            The path to a saved learner object file to load.
        """
        del self.__dict__
        self.__dict__ = Learner.from_file(learner_path).__dict__

    def _convert_coef_array_to_feature_names(self,
                                             coef,
                                             feature_name_prefix=''):
        """
        A helper method used by `model_params` to convert the model
        coefficients array into a dictionary with feature names as
        keys and the coefficients as values.

        Parameters
        ----------
        coef : np.array
            A numpy array with the model coefficients
        feature_name_prefix : str, optional
            An optional string that should be prefixed to the feature
            name, e.g. the name of the class for LogisticRegression
            or the class pair for SVCs with linear kernels.

        Returns
        -------
        res : dict
            A dictionary of labeled weights
        """
        res = {}

        # if we are doing feature hashing, then we need to make up
        # the feature names
        if isinstance(self.feat_vectorizer, FeatureHasher):
            num_features = len(coef)
            index_width_in_feature_name = int(floor(log10(num_features))) + 1
            feature_names = []
            for idx in range(num_features):
                index_str = str(idx + 1).zfill(index_width_in_feature_name)
                feature_names.append('hashed_feature_{}'.format(index_str))
            feature_indices = range(num_features)
            vocabulary = dict(zip(feature_names, feature_indices))

        # otherwise we can just use the DictVectorizer vocabulary
        # to get the feature names
        else:
            vocabulary = self.feat_vectorizer.vocabulary_

        # create the final result dictionary with the prefixed
        # feature names and the corresponding coefficient
        for feat, idx in vocabulary.items():
            if coef[idx]:
                res['{}{}'.format(feature_name_prefix, feat)] = coef[idx]

        return res

    @property
    def model_params(self):
        """
        Model parameters (i.e., weights) for a ``LinearModel`` (e.g., ``Ridge``)
        regression and liblinear models. If the model was trained using feature
        hashing, then names of the form `hashed_feature_XX` are used instead.

        Returns
        -------
        res : dict
            A dictionary of labeled weights.
        intercept : dict
            A dictionary of intercept(s).

        Raises
        ------
        ValueError
            If the instance does not support model parameters.
        """
        res = {}
        intercept = None
        if (isinstance(self._model, LinearModel) or
           (isinstance(self._model, SVR) and
                self._model.kernel == 'linear') or
           isinstance(self._model, SGDRegressor)):
            # also includes RescaledRidge, RescaledSVR, RescaledSGDRegressor

            coef = self.model.coef_
            intercept = {'_intercept_': self.model.intercept_}

            # convert SVR coefficient from a matrix to a 1D array
            # and convert from sparse to dense also if necessary.
            # However, this last bit may not be necessary
            # if we did feature scaling and coef is already dense.
            if isinstance(self._model, SVR):
                if sp.issparse(coef):
                    coef = coef.toarray()
                coef = coef[0]

            # inverse transform to get indices for before feature selection
            coef = coef.reshape(1, -1)
            coef = self.feat_selector.inverse_transform(coef)[0]
            res = self._convert_coef_array_to_feature_names(coef)

        elif isinstance(self._model, LinearSVC) or isinstance(self._model, LogisticRegression):
            label_list = self.label_list

            # if there are only two labels, scikit-learn will only have one
            # set of parameters and they will be associated with label 1 (not
            # 0)
            if len(self.label_list) == 2:
                label_list = self.label_list[-1:]

            if isinstance(self.feat_vectorizer, FeatureHasher):
                self.logger.warning("No feature names are available since this model was trained on hashed features.")

            for i, label in enumerate(label_list):
                coef = self.model.coef_[i]
                coef = coef.reshape(1, -1)
                coef = self.feat_selector.inverse_transform(coef)[0]
                label_res = self._convert_coef_array_to_feature_names(coef, feature_name_prefix='{}\t'.format(label))
                res.update(label_res)

            if isinstance(self.model.intercept_, float):
                intercept = {'_intercept_': self.model.intercept_}
            elif self.model.intercept_.any():
                intercept = dict(zip(label_list, self.model.intercept_))

        # for SVCs with linear kernels, we want to print out the primal
        # weights - that is, the weights for each feature for each one-vs-one
        # binary classifier. These are the weights contained in the `coef_`
        # attribute of the underlying scikit-learn model. This is a matrix that
        # has the shape [(n_classes)*(n_classes -1)/2, n_features] since there
        # are C(n_classes, 2) = n_classes*(n_classes-1)/2 one-vs-one classifiers
        # and each one has weights for each of the features. According to the
        # scikit-learn user guide and the code for the function `_one_vs_one_coef()`
        # in `svm/base.py`, the order of the rows is as follows is "0 vs 1",
        # "0 vs 2", ... "0 vs n", "1 vs 2", "1 vs 3", "1 vs n", ... "n-1 vs n".
        elif isinstance(self._model, SVC) and self._model.kernel == 'linear':
            intercept = {}
            if isinstance(self.feat_vectorizer, FeatureHasher):
                self.logger.warning("No feature names are available since this model was trained on hashed features.")
            for i, class_pair in enumerate(combinations(range(len(self.label_list)), 2)):
                coef = self.model.coef_[i]
                coef = coef.toarray()
                coef = self.feat_selector.inverse_transform(coef)[0]
                class1 = self.label_list[class_pair[0]]
                class2 = self.label_list[class_pair[1]]
                class_pair_res = self._convert_coef_array_to_feature_names(coef, feature_name_prefix='{}-vs-{}\t'.format(class1, class2))
                res.update(class_pair_res)
                intercept['{}-vs-{}'.format(class1, class2)] = self.model.intercept_[i]
        else:
            # not supported
            raise ValueError(("{} is not supported by" +
                              " model_params with its current settings."
                              ).format(self._model_type.__name__))

        return res, intercept

    @property
    def probability(self):
        """
        Should learner return probabilities of all labels (instead of just
        label with highest probability)?
        """
        return self._probability

    @probability.setter
    def probability(self, value):
        """
        Set the probabilities flag (i.e. whether learner
        should return probabilities of all labels).

        Parameters
        ----------
        value : bool
            Whether learner should return probabilities of all labels.
        """
        # LinearSVC doesn't support predict_proba
        self._probability = value
        if not hasattr(self.model_type, "predict_proba") and value:
            self.logger.warning("Probability was set to True, but {} does not have "
                                "a predict_proba() method.".format(self.model_type.__name__))
            self._probability = False

    def __getstate__(self):
        """
        Return the attributes that should be pickled. We need this
        because we cannot pickle loggers.
        """
        attribute_dict = dict(self.__dict__)
        if 'logger' in attribute_dict:
            del attribute_dict['logger']
        return attribute_dict

    def save(self, learner_path):
        """
        Save the ``Learner`` instance to a file.

        Parameters
        ----------
        learner_path : str
            The path to save the ``Learner`` instance to.
        """
        # create the directory if it doesn't exist
        learner_dir = os.path.dirname(learner_path)
        if not os.path.exists(learner_dir):
            os.makedirs(learner_dir)
        # write out the learner to disk
        joblib.dump((VERSION, self), learner_path)

    def _create_estimator(self):
        """
        Create an estimator.

        Returns
        -------
        estimator
            The estimator that was created.
        default_param_grid : list of dicts
            The parameter grid for the estimator.

        Raises
        ------
        ValueError
            If there is no default parameter grid for estimator.
        """
        estimator = None
        default_param_grid = _find_default_param_grid(self._model_type)
        if default_param_grid is None:
            raise ValueError("%s is not a valid learner type." %
                             (self._model_type.__name__,))

        estimator = self._model_type(**self._model_kwargs)

        return estimator, default_param_grid

    def _check_input_formatting(self, examples):
        """
        check that the examples are properly formatted.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to use for training.

        Raises
        ------
        TypeError
            If labels are strings.
        TypeError
            If any features are strings.
        """
        # Make sure the labels for a regression task are not strings.
        if self.model_type._estimator_type == 'regressor':
            for label in examples.labels:
                if isinstance(label, str):
                    raise TypeError("You are doing regression with string "
                                    "labels.  Convert them to integers or "
                                    "floats.")

        # make sure that feature values are not strings
        for val in examples.features.data:
            if isinstance(val, str):
                raise TypeError("You have feature values that are strings.  "
                                "Convert them to floats.")

    def _check_max_feature_value(self, feat_array):
        """
        Check if the the maximum absolute value of any feature is too large

        Parameters
        ----------
        feat_array : np.array
            A numpy array with features.
        """
        max_feat_abs = np.max(np.abs(feat_array.data))
        if max_feat_abs > 1000.0:
            self.logger.warning("You have a feature with a very large absolute "
                                "value ({}).  That may cause the learning "
                                "algorithm to crash or perform "
                                "poorly.".format(max_feat_abs))

    def _create_label_dict(self, examples):
        """
        Creates a dictionary of labels for classification problems.

        Parameters
        ----------
        examples : skll.FeatureSet
            The examples to use for training.
        """
        # We don't need to do this for regression models, so return.
        if self.model_type._estimator_type == 'regressor':
            return

        # extract list of unique labels if we are doing classification;
        # note that the output of np.unique() is sorted
        self.label_list = np.unique(examples.labels).tolist()

        # for binary classification, if one label is specified as
        # the positive class, re-sort the label list to make sure
        # that it is last in the list; for multi-class classification
        # raise a warning and set it back to None, since it does not
        # make any sense anyway
        if self.pos_label_str is not None:
            if len(self.label_list) != 2:
                self.logger.warning('Ignoring value of `pos_label_str` for '
                                    'multi-class classification.')
                self.pos_label_str = None
            else:
                self.label_list = sorted(self.label_list,
                                         key=lambda x: (x == self.pos_label_str,
                                                        x))

        # Given a list of all labels in the dataset and a list of the
        # unique labels in the set, convert the first list to an array of
        # numbers.
        self.label_dict = {label: i for i, label in enumerate(self.label_list)}

    def _train_setup(self, examples):
        """
        Set up the feature vectorizer and the scaler.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to use for training.
        """
        # Check feature values and labels
        self._check_input_formatting(examples)

        # Create feature name -> value mapping
        self.feat_vectorizer = examples.vectorizer

        # initialize feature selector
        self.feat_selector = SelectByMinCount(
            min_count=self._min_feature_count)

        # Create a scaler if we weren't passed one and we are asked
        # to do feature scaling; note that we do not support feature
        # scaling for `MultinomialNB` learners
        if (not issubclass(self._model_type, MultinomialNB) and
                self._feature_scaling != 'none'):
            scale_with_mean = self._feature_scaling in {'with_mean', 'both'}
            scale_with_std = self._feature_scaling in {'with_std', 'both'}
            self.scaler = StandardScaler(copy=True,
                                         with_mean=scale_with_mean,
                                         with_std=scale_with_std)
        else:
            # Doing this is to prevent any modification of feature values
            # using a dummy transformation
            self.scaler = StandardScaler(copy=False,
                                         with_mean=False,
                                         with_std=False)

    def train(self, examples, param_grid=None, grid_search_folds=3,
              grid_search=True, grid_objective=None,
              grid_jobs=None, shuffle=False, create_label_dict=True):
        """
        Train a classification model and return the model, score, feature
        vectorizer, scaler, label dictionary, and inverse label dictionary.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to use for training.
        param_grid : list of dicts, optional
            The parameter grid to search through for grid
            search. If ``None``, a default parameter grid
            will be used.
            Defaults to ``None``.
        grid_search_folds : int or dict, optional
            The number of folds to use when doing the
            grid search, or a mapping from
            example IDs to folds.
            Defaults to 3.
        grid_search : bool, optional
            Should we do grid search?
            Defaults to ``True``.
        grid_objective : str, optional
            The name of the objective function to use when
            doing the grid search. Must be specified if
            ``grid_search`` is ``True``.
            Defaults to ``None``.
        grid_jobs : int, optional
            The number of jobs to run in parallel when doing the
            grid search. If ``None`` or 0, the number of
            grid search folds will be used.
            Defaults to ``None``.
        shuffle : bool, optional
            Shuffle examples (e.g., for grid search CV.)
            Defaults to ``False``.
        create_label_dict : bool, optional
            Should we create the label dictionary?  This
            dictionary is used to map between string
            labels and their corresponding numerical
            values.  This should only be done once per
            experiment, so when ``cross_validate`` calls
            ``train``, ``create_label_dict`` gets set to
            ``False``. This option is only for internal
            use.
            Defaults to ``True``.

        Returns
        -------
        tuple : (float, dict)
            1) The best grid search objective function score, or 0 if
            we're not doing grid search, and 2) a dictionary of grid
            search CV results with keys such as "params",
            "mean_test_score", etc, that are mapped to lists of values
            associated with each hyperparameter set combination, or
            None if not doing grid search.

        Raises
        ------
        ValueError
            If grid_objective is not a valid grid objective or if
            one is not specified when necessary.
        MemoryError
            If process runs out of memory converting training data to dense.
        ValueError
            If FeatureHasher is used with MultinomialNB.
        """

        # get the estimator type since we need it in multiple places below
        estimator_type = self.model_type._estimator_type

        # if we are asked to do grid search, check that the grid objective
        # is specified and that the specified function is valid for the
        # selected learner
        if grid_search:

            if not grid_objective:
                raise ValueError("Grid search is on by default. You must either "
                                 "specify a grid objective or turn off grid search.")

            # get the list of objectives that are acceptable in the current
            # prediction scenario and raise an exception if the current
            # objective is not in this allowed list
            label_type = examples.labels.dtype.type
            if estimator_type == 'classifier':
                sorted_unique_labels = np.unique(examples.labels)
                allowed_objectives = _get_acceptable_classification_metrics(sorted_unique_labels)
            else:
                allowed_objectives = _get_acceptable_regression_metrics()

            if grid_objective not in allowed_objectives:
                raise ValueError("'{}' is not a valid objective "
                                 "function for {} with "
                                 "labels of type {}.".format(grid_objective,
                                                             self._model_type.__name__,
                                                             label_type.__name__))

            # If we're using a correlation metric for doing binary
            # classification and probability is set to true, we assume
            # that the user actually wants the `_with_probabilities`
            # version of the metric
            if (grid_objective in _CORRELATION_METRICS and
                    estimator_type == 'classifier' and
                    self.probability):
                self.logger.info('You specified "{}" as the objective with '
                                 '"probability" set to "true". If this is '
                                 'a binary classification task with integer '
                                 'labels, the probabilities for the positive '
                                 'class will be used to compute the '
                                 'correlation.'.format(grid_objective))
                old_grid_objective = grid_objective
                new_grid_objective = '{}_probs'.format(grid_objective)
                metrics_module = import_module('skll.metrics')
                metric_func = getattr(metrics_module, 'correlation')
                SCORERS[new_grid_objective] = make_scorer(metric_func,
                                                          corr_type=grid_objective,
                                                          needs_proba=True)
                grid_objective = new_grid_objective

        # Shuffle so that the folds are random for the inner grid search CV.
        # If grid search is True but shuffle isn't, shuffle anyway.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if grid_search or shuffle:
            if grid_search and not shuffle:
                self.logger.warning('Training data will be shuffled to randomize '
                                    'grid search folds.  Shuffling may yield '
                                    'different results compared to scikit-learn.')
            ids, labels, features = sk_shuffle(examples.ids, examples.labels,
                                               examples.features,
                                               random_state=123456789)
            examples = FeatureSet(examples.name, ids, labels=labels,
                                  features=features,
                                  vectorizer=examples.vectorizer)

        # call train setup to set up the vectorizer, the labeldict, and the
        # scaler
        if create_label_dict:
            self._create_label_dict(examples)
        self._train_setup(examples)

        # select features
        xtrain = self.feat_selector.fit_transform(examples.features)

        # Convert to dense if necessary
        if self._use_dense_features:
            try:
                xtrain = xtrain.todense()
            except MemoryError:
                if issubclass(self._model_type, _REQUIRES_DENSE):
                    reason = ('{} does not support sparse ' +
                              'matrices.').format(self._model_type.__name__)
                else:
                    reason = ('{} feature scaling requires a dense ' +
                              'matrix.').format(self._feature_scaling)
                raise MemoryError('Ran out of memory when converting training '
                                  'data to dense. This was required because ' +
                                  reason)

        if isinstance(self.feat_vectorizer, FeatureHasher) and \
                issubclass(self._model_type, MultinomialNB):
            raise ValueError('Cannot use FeatureHasher with MultinomialNB '
                             'because MultinomialNB cannot handle negative '
                             'feature values.')

        # Scale features if necessary
        xtrain = self.scaler.fit_transform(xtrain)

        # check whether any feature values are too large
        self._check_max_feature_value(xtrain)

        # Sampler
        if self.sampler is not None and \
                issubclass(self._model_type, MultinomialNB):
            raise ValueError('Cannot use a sampler with MultinomialNB '
                             'because MultinomialNB cannot handle negative '
                             'feature values.')

        if self.sampler:
            self.logger.warning('Sampler converts sparse matrix to dense')
            if isinstance(self.sampler, SkewedChi2Sampler):
                self.logger.warning('SkewedChi2Sampler uses a dense matrix')
                if sp.issparse(xtrain):
                    xtrain = xtrain.todense()
            xtrain = self.sampler.fit_transform(xtrain)

        # use label dict transformed version of examples.labels if doing
        # classification
        if estimator_type == 'classifier':
            labels = np.array([self.label_dict[label] for label in
                               examples.labels])
        else:
            labels = examples.labels

        # Instantiate an estimator and get the default parameter grid to search
        estimator, default_param_grid = self._create_estimator()

        # Use default parameter grid if we weren't passed one
        # In case the default parameter grid is also empty
        # then there's no point doing the grid search at all
        if grid_search and not param_grid:
            if default_param_grid == [{}]:
                self.logger.warning("SKLL has no default parameter grid "
                                    "available for the {} learner and no "
                                    "parameter grids were supplied. Using "
                                    "default values instead of grid "
                                    "search.".format(self._model_type.__name__))
                grid_search = False
            else:
                param_grid = default_param_grid

        # set up a grid searcher if we are asked to
        if grid_search:
            # set up grid search folds
            if isinstance(grid_search_folds, int):
                grid_search_folds = \
                    self._compute_num_folds_from_example_counts(grid_search_folds, labels)

                if not grid_jobs:
                    grid_jobs = grid_search_folds
                else:
                    grid_jobs = min(grid_search_folds, grid_jobs)
                folds = grid_search_folds
            else:
                # use the number of unique fold IDs as the number of grid jobs
                num_specified_folds = len(set(grid_search_folds.values()))
                if not grid_jobs:
                    grid_jobs = num_specified_folds
                else:
                    grid_jobs = min(num_specified_folds, grid_jobs)
                # Only retain IDs within folds if they're in grid_search_folds
                dummy_label = next(iter(grid_search_folds.values()))
                fold_groups = [grid_search_folds.get(curr_id, dummy_label) for
                               curr_id in examples.ids]
                kfold = FilteredLeaveOneGroupOut(grid_search_folds,
                                                 examples.ids,
                                                 logger=self.logger)
                folds = kfold.split(examples.features, examples.labels, fold_groups)

            # limit the number of grid_jobs to be no higher than five or the
            # number of cores for the machine, whichever is lower
            grid_jobs = min(grid_jobs, cpu_count(), MAX_CONCURRENT_PROCESSES)
            grid_searcher = GridSearchCV(estimator,
                                         param_grid,
                                         scoring=grid_objective,
                                         iid=False,
                                         cv=folds,
                                         n_jobs=grid_jobs,
                                         pre_dispatch=grid_jobs)

            # run the grid search for hyperparameters
            grid_searcher.fit(xtrain, labels)
            self._model = grid_searcher.best_estimator_
            grid_score = grid_searcher.best_score_
            grid_cv_results = grid_searcher.cv_results_
        else:
            self._model = estimator.fit(xtrain, labels)
            grid_score = 0.0
            grid_cv_results = None

        # restore the original of the grid objective if we
        # had futzed with it to handle correlation
        # objectives and probability outputs
        if 'old_grid_objective' in locals():
            grid_objective = old_grid_objective
            del SCORERS[new_grid_objective]

        # store a scikit-learn Pipeline in the `pipeline` attribute
        # composed of a copy of the vectorizer, the selector,
        # the sampler, the scaler, and the estimator. This pipeline
        # attribute can then be used by someone who wants to take a SKLL
        # model and then do further analysis using scikit-learn
        # We are using copies since the user might want to play
        # around with the pipeline and we want to let her do that
        # but keep the SKLL model the same
        if self._store_pipeline:

            # initialize the list that will hold the pipeline steps
            pipeline_steps = []

            # start with the vectorizer

            # note that sometimes we may have to end up using dense
            # features or if we were using a SkewedChi2Sampler which
            # requires dense inputs. If this turns out to be the case
            # then let's turn off `sparse` for the vectorizer copy
            # to be stored in the pipeline as well so that it works
            # on the scikit-learn in the same way. However, note that
            # this solution will only work for DictVectorizers. For
            # feature hashers, we manually convert things to dense
            # when we need in SKLL. Therefore, to handle this case,
            # we basically need to create a custom intermediate
            # pipeline stage that will convert the features to dense
            # once the hashing is done since this is what happens
            # in SKLL.
            vectorizer_copy = copy.deepcopy(self.feat_vectorizer)
            if (self._use_dense_features or
                    isinstance(self.sampler, SkewedChi2Sampler)):
                if isinstance(self.feat_vectorizer, DictVectorizer):
                    self.logger.warning("The `sparse` attribute of the "
                                        "DictVectorizer stage will be "
                                        "set to `False` in the pipeline "
                                        "since dense features are "
                                        "required when centering.")
                    vectorizer_copy.sparse = False
                else:
                    self.logger.warning("A custom pipeline stage "
                                        "(`Densifier`) will be inserted "
                                        " in the pipeline since the "
                                        "current SKLL configuration "
                                        "requires dense features.")
                    densifier = Densifier()
                    pipeline_steps.append(('densifier', densifier))
            pipeline_steps.insert(0, ('vectorizer', vectorizer_copy))

            # next add the selector
            pipeline_steps.append(('selector',
                                   copy.deepcopy(self.feat_selector)))

            # next, include the scaler
            pipeline_steps.append(('scaler', copy.deepcopy(self.scaler)))

            # next, include the sampler, if there is one
            if self.sampler:
                pipeline_steps.append(('sampler',
                                       copy.deepcopy(self.sampler)))

            # finish with the estimator
            pipeline_steps.append(('estimator', copy.deepcopy(self.model)))

            self.pipeline = Pipeline(steps=pipeline_steps)

        return grid_score, grid_cv_results

    def evaluate(self, examples, prediction_prefix=None, append=False,
                 grid_objective=None, output_metrics=[]):
        """
        Evaluates a given model on a given dev or test ``FeatureSet``.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to evaluate the performance of the model on.
        prediction_prefix : str, optional
            If saving the predictions, this is the
            prefix that will be used for the filename.
            It will be followed by ``"_predictions.tsv"``
            Defaults to ``None``.
        append : bool, optional
            Should we append the current predictions to the file if
            it exists?
            Defaults to ``False``.
        grid_objective : function, optional
            The objective function that was used when doing
            the grid search.
            Defaults to ``None``.
        output_metrics : list of str, optional
            List of additional metric names to compute in
            addition to grid objective. Empty by default.
            Defaults to an empty list.

        Returns
        -------
        res : 6-tuple
            The confusion matrix, the overall accuracy, the per-label
            PRFs, the model parameters, the grid search objective
            function score, and the additional evaluation metrics, if any.
        """
        # initialize a dictionary that will hold all of the metric scores
        metric_scores = {metric: None for metric in output_metrics}

        # are we in a regressor or a classifier
        estimator_type = self.model_type._estimator_type

        # make the prediction on the test data; note that these
        # are either class indices or class probabilities
        yhat = self.predict(examples,
                            prediction_prefix=prediction_prefix,
                            append=append)

        # if we are a classifier and in probability mode, then
        # `yhat` are probabilities so we need to compute the
        # class indices separately and save them too
        if self.probability and estimator_type == 'classifier':
            yhat_probs = yhat
            yhat = np.argmax(yhat_probs, axis=1)
        # if we are a regressor or classifier not in probability
        # mode, then we have the class indices already and there
        # are no probabilities
        else:
            yhat_probs = None

        # convert the true class labels to indices too for consistency
        # if we are a classifier
        if estimator_type == 'classifier':
            test_label_list = np.unique(examples.labels).tolist()

            # identify unseen test labels if any and add a new dictionary
            # for these  labels
            unseen_test_label_list = [label for label in test_label_list
                                      if label not in self.label_list]
            unseen_label_dict = {label: i for i, label in enumerate(unseen_test_label_list,
                                                                    start=len(self.label_list))}
            # combine the two dictionaries
            train_and_test_label_dict = self.label_dict.copy()
            train_and_test_label_dict.update(unseen_label_dict)
            ytest = np.array([train_and_test_label_dict[label]
                              for label in examples.labels])
        # we are a regressor, so we do not need to do anything else
        else:
            ytest = examples.labels

        # compute the acceptable metrics for our current prediction scenario
        label_type = examples.labels.dtype.type
        if estimator_type == 'classifier':
            sorted_unique_labels = np.unique(examples.labels)
            acceptable_metrics = _get_acceptable_classification_metrics(sorted_unique_labels)
        else:
            acceptable_metrics = _get_acceptable_regression_metrics()

        # check that all of the output metrics are acceptable
        unacceptable_metrics = set(output_metrics).difference(acceptable_metrics)
        if unacceptable_metrics:
            raise ValueError("The following metrics are not valid "
                             "for this learner ({}) with these labels of "
                             "type {}: {}".format(self._model_type.__name__,
                                                  label_type.__name__,
                                                  list(unacceptable_metrics)))

        # if metrics has the objective in it, we will only output
        # that function once as an objective and not include it
        # in the list of additional metrics printed out
        if len(output_metrics) > 0 and grid_objective in output_metrics:
            self.logger.warning('The grid objective "{}" is also specified '
                                'as an evaluation metric. Since its value is '
                                'already included in the results as the '
                                'objective score, it will not be printed '
                                'again in the list of metrics.'.format(grid_objective))
            output_metrics = [metric for metric in output_metrics
                              if metric != grid_objective]

        # make a single list of metrics including the grid objective
        # since it's easier to compute everything together
        metrics_to_compute = [grid_objective] + output_metrics
        for metric in metrics_to_compute:

            # skip the None if we are not doing grid search
            if not metric:
                continue

            # CASE 1: in probability mode for classification which means we
            # need to either use the probabilities directly or infer the labels
            # from them depending on the metric
            if self.probability:

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
                if (len(self.label_list) == 2 and
                        (metric in _CORRELATION_METRICS or
                         metric in ['average_precision', 'roc_auc']) and
                        metric != grid_objective):
                    self.logger.info('using probabilities for the positive class to '
                                     'compute "{}" for evaluation.'.format(metric))
                    yhat_for_metric = yhat_probs[:, 1]
                elif metric == 'neg_log_loss':
                    yhat_for_metric = yhat_probs
                else:
                    yhat_for_metric = yhat

            # CASE 2: no probability mode for classifier or regressor
            # in which case we just use the predictions as they are
            else:
                yhat_for_metric = yhat

            try:
                metric_scores[metric] = use_score_func(metric, ytest, yhat_for_metric)
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

        if estimator_type == 'regressor':
            result_dict = {'descriptive': defaultdict(dict)}
            for table_label, y in zip(['actual', 'predicted'], [ytest, yhat]):
                result_dict['descriptive'][table_label]['min'] = min(y)
                result_dict['descriptive'][table_label]['max'] = max(y)
                result_dict['descriptive'][table_label]['avg'] = np.mean(y)
                result_dict['descriptive'][table_label]['std'] = np.std(y)
            result_dict['pearson'] = use_score_func('pearson', ytest, yhat)
            res = (None, None, result_dict, self._model.get_params(), objective_score,
                   additional_scores)
        else:
            # compute the confusion matrix and precision/recall/f1
            # note that we are using the labels indices here
            # and not the actual class labels themselves
            num_labels = len(train_and_test_label_dict)
            conf_mat = confusion_matrix(ytest, yhat,
                                        labels=list(range(num_labels)))
            # Calculate metrics
            overall_accuracy = accuracy_score(ytest, yhat)
            result_matrix = precision_recall_fscore_support(
                ytest, yhat, labels=list(range(num_labels)), average=None)

            # Store results
            result_dict = defaultdict(dict)
            for actual_label in sorted(train_and_test_label_dict):
                col = train_and_test_label_dict[actual_label]
                result_dict[actual_label]["Precision"] = result_matrix[0][col]
                result_dict[actual_label]["Recall"] = result_matrix[1][col]
                result_dict[actual_label]["F-measure"] = result_matrix[2][col]

            res = (conf_mat.tolist(), overall_accuracy, result_dict,
                   self._model.get_params(), objective_score,
                   additional_scores)
        return res

    def predict(self, examples, prediction_prefix=None, append=False,
                class_labels=False):
        """
        Uses a given model to generate predictions on a given ``FeatureSet``.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to predict labels for.
        prediction_prefix : str, optional
            If saving the predictions, this is the prefix that will be used for
            the filename. It will be followed by ``"_predictions.tsv"``
            Defaults to ``None``.
        append : bool, optional
            Should we append the current predictions to the file if it exists?
            Defaults to ``False``.
        class_labels : bool, optional
            For classifier, should we convert class indices to their (str) labels
            for the returned array? Note that class labels are always written out
            to disk.
            Defaults to ``False``.

        Returns
        -------
        yhat : array-like
            The predictions returned by the ``Learner`` instance.

        Raises
        ------
        MemoryError
            If process runs out of memory when converting to dense.
        """
        example_ids = examples.ids

        # Need to do some transformations so the features are in the right
        # columns for the test set. Obviously a bit hacky, but storing things
        # in sparse matrices saves memory over our old list of dicts approach.

        # We also need to think about the various combinations of the model
        # vectorizer and the vectorizer for the set for which we want to make
        # predictions:

        # 1. Both vectorizers are DictVectorizers. If they use different sets
        # of features, we raise a warning and transform the features of the
        # prediction set from its space to the trained model space.

        # 2. Both vectorizers are FeatureHashers. If they use different number
        # of feature bins, we should just raise an error since there's no
        # inverse_transform() available for a FeatureHasher - the hash function
        # is not reversible.

        # 3. The model vectorizer is a FeatureHasher but the prediction feature
        # set vectorizer is a DictVectorizer. We should be able to handle this
        # case, since we can just call inverse_transform() on the DictVectorizer
        # and then transform() on the FeatureHasher?

        # 4. The model vectorizer is a DictVectorizer but the prediction feature
        # set vectorizer is a FeatureHasher. Again, we should raise an error here
        # since there's no inverse available for the hasher.
        model_is_dict = isinstance(self.feat_vectorizer,
                                   (DictVectorizer, OldDictVectorizer))
        model_is_hasher = isinstance(self.feat_vectorizer, FeatureHasher)
        data_is_dict = isinstance(examples.vectorizer,
                                  (DictVectorizer, OldDictVectorizer))
        data_is_hasher = isinstance(examples.vectorizer, FeatureHasher)

        both_dicts = model_is_dict and data_is_dict
        both_hashers = model_is_hasher and data_is_hasher
        model_hasher_and_data_dict = model_is_hasher and data_is_dict
        model_dict_and_data_hasher = model_is_dict and data_is_hasher

        # 1. both are DictVectorizers
        if both_dicts:
            if (set(self.feat_vectorizer.feature_names_) !=
                    set(examples.vectorizer.feature_names_)):
                self.logger.warning("There is mismatch between the training model "
                                    "features and the data passed to predict. The "
                                    "prediction features will be transformed to "
                                    "the trained model space.")
            if self.feat_vectorizer == examples.vectorizer:
                xtest = examples.features
            else:
                xtest = self.feat_vectorizer.transform(
                    examples.vectorizer.inverse_transform(
                        examples.features))

        # 2. both are FeatureHashers
        elif both_hashers:
            self_feat_vec_tuple = (self.feat_vectorizer.dtype,
                                   self.feat_vectorizer.input_type,
                                   self.feat_vectorizer.n_features)
            example_feat_vec_tuple = (examples.vectorizer.dtype,
                                      examples.vectorizer.input_type,
                                      examples.vectorizer.n_features)

            if self_feat_vec_tuple == example_feat_vec_tuple:
                xtest = examples.features
            else:
                self.logger.error('There is mismatch between the FeatureHasher '
                                  'configuration for the training data and '
                                  'the configuration for the data passed to predict')
                raise RuntimeError('Mismatched hasher configurations')

        # 3. model is a FeatureHasher and test set is a DictVectorizer
        elif model_hasher_and_data_dict:
            xtest = self.feat_vectorizer.transform(
                examples.vectorizer.inverse_transform(
                    examples.features))

        # 4. model is a DictVectorizer and test set is a FeatureHasher
        elif model_dict_and_data_hasher:
            self.logger.error('Cannot predict with a model using a '
                              'DictVectorizer on data that uses '
                              'a FeatureHasher')
            raise RuntimeError('Cannot use FeatureHasher for data')

        # filter features based on those selected from training set
        xtest = self.feat_selector.transform(xtest)

        # Convert to dense if necessary
        if self._use_dense_features and not isinstance(xtest, np.ndarray):
            try:
                xtest = xtest.todense()
            except MemoryError:
                if issubclass(self._model_type, _REQUIRES_DENSE):
                    reason = ('{} does not support sparse ' +
                              'matrices.').format(self._model_type.__name__)
                else:
                    reason = ('{} feature scaling requires a dense ' +
                              'matrix.').format(self._feature_scaling)
                raise MemoryError('Ran out of memory when converting test ' +
                                  'data to dense. This was required because ' +
                                  reason)

        # Scale xtest if necessary
        if not issubclass(self._model_type, MultinomialNB):
            xtest = self.scaler.transform(xtest)

        # Sampler
        if self.sampler:
            self.logger.warning('Sampler converts sparse matrix to dense')
            if isinstance(self.sampler, SkewedChi2Sampler):
                self.logger.warning('SkewedChi2Sampler uses a dense matrix')
                if sp.issparse(xtest):
                    xtest = xtest.todense()
            xtest = self.sampler.fit_transform(xtest)

        # make the prediction on the test data
        try:
            yhat = (self._model.predict_proba(xtest)
                    if (self.probability and
                        not class_labels)
                    else self._model.predict(xtest))
        except NotImplementedError as e:
            self.logger.error("Model type: {}\n"
                              "Model: {}\n"
                              "Probability: {}\n".format(self._model_type.__name__,
                                                         self._model,
                                                         self.probability))
            raise e

        # write out the predictions if we are asked to
        if prediction_prefix is not None:
            prediction_file = '{}_predictions.tsv'.format(prediction_prefix)
            with open(prediction_file,
                      "w" if not append else "a") as predictionfh:
                # header
                if not append:
                    # Output probabilities if we're asked (and able)
                    if self.probability:
                        print('\t'.join(["id"] +
                                        [str(x) for x in self.label_list]),
                              file=predictionfh)
                    else:
                        print('id\tprediction', file=predictionfh)

                if self.probability:
                    for example_id, class_probs in zip(example_ids, yhat):
                        print('\t'.join([str(example_id)] +
                                        [str(x) for x in class_probs]),
                              file=predictionfh)
                else:
                    if self.model_type._estimator_type == 'regressor':
                        for example_id, pred in zip(example_ids, yhat):
                            print('{0}\t{1}'.format(example_id, pred),
                                  file=predictionfh)
                    else:
                        for example_id, pred in zip(example_ids, yhat):
                            print('%s\t%s' % (example_id,
                                              self.label_list[int(pred)]),
                                  file=predictionfh)

        if (class_labels and
                self.model_type._estimator_type == 'classifier'):
            yhat = np.array([self.label_list[int(pred)] for pred in yhat])

        return yhat

    def _compute_num_folds_from_example_counts(self, cv_folds, labels):
        """
        Calculate the number of folds we should use for cross validation, based
        on the number of examples we have for each label.

        Parameters
        ----------
        cv_folds : int
            The number of cross-validation folds.
        labels : list
            The example labels.

        Returns
        -------
        cv_folds : int
            The number of folds to use, based on the number of examples
            for each label.

        Raises
        ------
        AssertionError
            If ```cv_folds``` is not an integer.
        ValueError
            If the training set has less than or equal to one label(s).
        """
        assert isinstance(cv_folds, int)

        # For regression models, we can just return the current cv_folds
        if self.model_type._estimator_type == 'regressor':
            return cv_folds

        min_examples_per_label = min(Counter(labels).values())
        if min_examples_per_label <= 1:
            raise ValueError(('The training set has only {} example for a' +
                              ' label.').format(min_examples_per_label))
        if min_examples_per_label < cv_folds:
            self.logger.warning('The minimum number of examples per label was {}. '
                                'Setting the number of cross-validation folds to '
                                'that value.'.format(min_examples_per_label))
            cv_folds = min_examples_per_label
        return cv_folds

    def cross_validate(self,
                       examples,
                       stratified=True,
                       cv_folds=10,
                       grid_search=True,
                       grid_search_folds=3,
                       grid_jobs=None,
                       grid_objective=None,
                       output_metrics=[],
                       prediction_prefix=None,
                       param_grid=None,
                       shuffle=False,
                       save_cv_folds=False,
                       save_cv_models=False,
                       use_custom_folds_for_grid_search=True):
        """
        Cross-validates a given model on the training examples.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to cross-validate learner performance on.
        stratified : bool, optional
            Should we stratify the folds to ensure an even
            distribution of labels for each fold?
            Defaults to ``True``.
        cv_folds : int, optional
            The number of folds to use for cross-validation, or
            a mapping from example IDs to folds.
            Defaults to 10.
        grid_search : bool, optional
            Should we do grid search when training each fold?
            Note: This will make this take *much* longer.
            Defaults to ``False``.
        grid_search_folds : int or dict, optional
            The number of folds to use when doing the
            grid search, or a mapping from
            example IDs to folds.
            Defaults to 3.
        grid_jobs : int, optional
            The number of jobs to run in parallel when doing the
            grid search. If ``None`` or 0, the number of
            grid search folds will be used.
            Defaults to ``None``.
        grid_objective : str, optional
            The name of the objective function to use when
            doing the grid search. Must be specified if
            ``grid_search`` is ``True``.
            Defaults to ``None``.
        output_metrics : list of str, optional
            List of additional metric names to compute in
            addition to the metric used for grid search. Empty
            by default.
            Defaults to an empty list.
        prediction_prefix : str, optional
            If saving the predictions, this is the
            prefix that will be used for the filename.
            It will be followed by ``"_predictions.tsv"``
            Defaults to ``None``.
        param_grid : list of dicts, optional
            The parameter grid to traverse.
            Defaults to ``None``.
        shuffle : bool, optional
            Shuffle examples before splitting into folds for CV.
            Defaults to ``False``.
        save_cv_folds : bool, optional
             Whether to save the cv fold ids or not?
             Defaults to ``False``.
        save_cv_models : bool, optional
            Whether to save the cv models or not?
            Defaults to ``False``.
        use_custom_folds_for_grid_search : bool, optional
            If ``cv_folds`` is a custom dictionary, but
            ``grid_search_folds`` is not, perhaps due to user
            oversight, should the same custom dictionary
            automatically be used for the inner grid-search
            cross-validation?
            Defaults to ``True``.

        Returns
        -------
        results : list of 6-tuples
            The confusion matrix, overall accuracy, per-label PRFs, model
            parameters, objective function score, and evaluation metrics (if any)
            for each fold.
        grid_search_scores : list of floats
            The grid search scores for each fold.
        grid_search_cv_results_dicts : list of dicts
            A list of dictionaries of grid search CV results, one per fold,
            with keys such as "params", "mean_test_score", etc, that are
            mapped to lists of values associated with each hyperparameter set
            combination.
        skll_fold_ids : dict
            A dictionary containing the test-fold number for each id
            if ``save_cv_folds`` is ``True``, otherwise ``None``.
        models : list of skll.learner.Learner
            A list of skll.learner.Learners, one for each fold if
            ``save_cv_models`` is ``True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If labels are not encoded as strings.
        """

        # Seed the random number generator so that randomized algorithms are
        # replicable.
        random_state = np.random.RandomState(123456789)

        # We need to check whether the labels in the featureset are labels
        # or continuous values. If it's the latter, we need to raise an
        # an exception since the stratified splitting in sklearn does not
        # work with continuous labels. Note that although using random folds
        # _will_ work, we want to raise an error in general since it's better
        # to encode the labels as strings anyway.
        if (self.model_type._estimator_type == 'classifier' and
                type_of_target(examples.labels) not in ['binary', 'multiclass']):
            raise ValueError("Floating point labels must be encoded as strings for cross-validation.")

        # check that we have an objective since grid search is on by default
        # Note that `train()` would raise this error anyway later but it's
        # better to raise this early on so rather than after a whole bunch of
        # stuff has happened
        if grid_search:
            if not grid_objective:
                raise ValueError("Grid search is on by default. You must either "
                                 "specify a grid objective or turn off grid search.")

        # Shuffle so that the folds are random for the inner grid search CV.
        # If grid search is True but shuffle isn't, shuffle anyway.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if grid_search or shuffle:
            if grid_search and not shuffle:
                self.logger.warning('Training data will be shuffled to randomize '
                                    'grid search folds. Shuffling may yield '
                                    'different results compared to scikit-learn.')
            ids, labels, features = sk_shuffle(examples.ids, examples.labels,
                                               examples.features,
                                               random_state=random_state)
            examples = FeatureSet(examples.name, ids, labels=labels,
                                  features=features,
                                  vectorizer=examples.vectorizer)

        # call train setup
        self._create_label_dict(examples)
        self._train_setup(examples)

        # Set up the cross-validation iterator.
        if isinstance(cv_folds, int):
            cv_folds = self._compute_num_folds_from_example_counts(cv_folds,
                                                                   examples.labels)

            stratified = (stratified and
                          self.model_type._estimator_type == 'classifier')
            if stratified:
                kfold = StratifiedKFold(n_splits=cv_folds)
                cv_groups = None
            else:
                kfold = KFold(n_splits=cv_folds, random_state=random_state)
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
                                             logger=self.logger)
            cv_groups = fold_groups

            # If we are planning to do grid search, set the grid search folds
            # to be the same as the custom cv folds unless a flag is set that
            # explicitly tells us not to. Note that this should only happen
            # when we are using the API; otherwise the configparser should
            # take care of this even before this method is called
            if grid_search and use_custom_folds_for_grid_search and grid_search_folds != cv_folds:
                self.logger.warning("The specified custom folds will be used for "
                                    "the inner grid search.")
                grid_search_folds = cv_folds

        # Save the cross-validation fold information, if required
        # The format is that the test-fold that each id appears in is stored
        skll_fold_ids = None
        if save_cv_folds:
            skll_fold_ids = {}
            for fold_num, (_, test_indices) in enumerate(kfold.split(examples.features,
                                                                     examples.labels,
                                                                     cv_groups)):
                for index in test_indices:
                    skll_fold_ids[examples.ids[index]] = str(fold_num)

        # handle each fold separately and accumulate the predictions and the
        # numbers
        results = []
        grid_search_scores = []
        grid_search_cv_results_dicts = []
        append_predictions = False
        models = [] if save_cv_models else None
        for train_index, test_index in kfold.split(examples.features,
                                                   examples.labels,
                                                   cv_groups):
            # Train model
            self._model = None  # prevent feature vectorizer from being reset.
            train_set = FeatureSet(examples.name,
                                   examples.ids[train_index],
                                   labels=examples.labels[train_index],
                                   features=examples.features[train_index],
                                   vectorizer=examples.vectorizer)

            # Set run_create_label_dict to False since we already created the
            # label dictionary for the whole dataset above.
            (grid_search_score,
             grid_search_cv_results) = self.train(train_set,
                                                  grid_search_folds=grid_search_folds,
                                                  grid_search=grid_search,
                                                  grid_objective=grid_objective,
                                                  param_grid=param_grid,
                                                  grid_jobs=grid_jobs,
                                                  shuffle=grid_search,
                                                  create_label_dict=False)
            grid_search_scores.append(grid_search_score)
            if save_cv_models:
                models.append(copy.deepcopy(self))
            grid_search_cv_results_dicts.append(grid_search_cv_results)
            # note: there is no need to shuffle again within each fold,
            # regardless of what the shuffle keyword argument is set to.

            # Evaluate model
            test_tuple = FeatureSet(examples.name,
                                    examples.ids[test_index],
                                    labels=examples.labels[test_index],
                                    features=examples.features[test_index],
                                    vectorizer=examples.vectorizer)
            results.append(self.evaluate(test_tuple,
                                         prediction_prefix=prediction_prefix,
                                         append=append_predictions,
                                         grid_objective=grid_objective,
                                         output_metrics=output_metrics))
            append_predictions = True

        # return list of results/outputs for all folds
        return (results,
                grid_search_scores,
                grid_search_cv_results_dicts,
                skll_fold_ids,
                models)

    def learning_curve(self,
                       examples,
                       metric,
                       cv_folds=10,
                       train_sizes=np.linspace(0.1, 1.0, 5)):
        """
        Generates learning curves for a given model on the training examples
        via cross-validation. Adapted from the scikit-learn code for learning
        curve generation (cf.``sklearn.model_selection.learning_curve``).

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to generate the learning curve on.
        cv_folds : int, optional
            The number of folds to use for cross-validation, or
            a mapping from example IDs to folds.
            Defaults to 10.
        metric : str
            The name of the metric function to use
            when computing the train and test scores
            for the learning curve.
        train_sizes : list of float or int, optional
            Relative or absolute numbers of training examples
            that will be used to generate the learning curve.
            If the type is float, it is regarded as a fraction
            of the maximum size of the training set (that is
            determined by the selected validation method),
            i.e. it has to be within (0, 1]. Otherwise it
            is interpreted as absolute sizes of the training
            sets. Note that for classification the number of
            samples usually have to be big enough to contain
            at least one sample from each class.
            Defaults to  ``np.linspace(0.1, 1.0, 5)``.

        Returns
        -------
        train_scores : list of float
            The scores for the training set.
        test_scores : list of float
            The scores on the test set.
        num_examples : list of int
            The numbers of training examples used to generate
            the curve
        """

        # Seed the random number generator so that randomized algorithms are
        # replicable.
        random_state = np.random.RandomState(123456789)

        # Call train setup before since we need to train
        # the learner eventually
        self._create_label_dict(examples)
        self._train_setup(examples)

        # Set up the cross-validation iterator with 20% of the data
        # always reserved for testing
        cv = ShuffleSplit(n_splits=cv_folds,
                          test_size=0.2,
                          random_state=random_state)
        cv_iter = list(cv.split(examples.features, examples.labels, None))
        n_max_training_samples = len(cv_iter[0][0])

        # Get the _translate_train_sizes() function from scikit-learn
        # since we need it to get the right list of sizes after cross-validation
        _module = import_module('sklearn.model_selection._validation')
        _translate_train_sizes = getattr(_module, '_translate_train_sizes')
        train_sizes_abs = _translate_train_sizes(train_sizes,
                                                 n_max_training_samples)
        n_unique_ticks = train_sizes_abs.shape[0]

        # Create an iterator over train/test featuresets based on the
        # cross-validation index iterator
        featureset_iter = (FeatureSet.split_by_ids(examples, train, test) for train, test in cv_iter)

        # Limit the number of parallel jobs for this
        # to be no higher than five or the number of cores
        # for the machine, whichever is lower
        n_jobs = min(cpu_count(), MAX_CONCURRENT_PROCESSES)

        # Run jobs in parallel that train the model on each subset
        # of the training data and compute train and test scores
        parallel = joblib.Parallel(n_jobs=n_jobs, pre_dispatch=n_jobs)
        out = parallel(joblib.delayed(_train_and_score)(self,
                                                        train_fs[:n_train_samples],
                                                        test_fs,
                                                        metric)
                       for train_fs, test_fs in featureset_iter
                       for n_train_samples in train_sizes_abs)

        # Reshape the outputs
        out = np.array(out)
        n_cv_folds = out.shape[0] // n_unique_ticks
        out = out.reshape(n_cv_folds, n_unique_ticks, 2)
        out = np.asarray(out).transpose((2, 1, 0))

        return list(out[0]), list(out[1]), list(train_sizes_abs)
