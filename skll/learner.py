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

from __future__ import absolute_import, print_function, unicode_literals

import copy
import inspect
import logging
import os
import sys
from collections import Counter, defaultdict
from functools import wraps
from importlib import import_module
from multiprocessing import cpu_count

import joblib
import numpy as np
import scipy.sparse as sp
from six import iteritems, itervalues
from six import string_types
from six.moves import xrange as range
from six.moves import zip
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
from sklearn.feature_selection import SelectKBest
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
from skll.metrics import _CORRELATION_METRICS, use_score_func
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
                          'alpha': [1e-4, 1e-3, 1e-3, 1e-1, 1, 10, 100, 1000]}],
                        KNeighborsClassifier:
                        [{'n_neighbors': [1, 5, 10, 100],
                          'weights': ['uniform', 'distance']}],
                        KNeighborsRegressor:
                        [{'n_neighbors': [1, 5, 10, 100],
                          'weights': ['uniform', 'distance']}],
                        MLPClassifier:
                        [{'activation': ['logistic', 'tanh', 'relu'],
                          'alpha': [1e-4, 1e-3, 1e-3, 1e-1, 1],
                          'learning_rate_init': [0.001, 0.01, 0.1]}],
                        MLPRegressor:
                        [{'activation': ['logistic', 'tanh', 'relu'],
                          'alpha': [1e-4, 1e-3, 1e-3, 1e-1, 1],
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
                          'gamma': ['auto', 0.01, 0.1, 1.0, 10.0, 100.0]}],
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
                          'gamma': ['auto', 0.01, 0.1, 1.0, 10.0, 100.0]}],
                        TheilSenRegressor:
                        [{}]}


# list of valid grid objective functions for regression and classification
# models depending on type of labels

_BINARY_CLASS_OBJ_FUNCS = frozenset(['unweighted_kappa',
                                     'linear_weighted_kappa',
                                     'quadratic_weighted_kappa',
                                     'uwk_off_by_one',
                                     'lwk_off_by_one',
                                     'qwk_off_by_one',
                                     'kendall_tau',
                                     'pearson',
                                     'spearman',
                                     'neg_log_loss'])

_REGRESSION_ONLY_OBJ_FUNCS = frozenset(['r2',
                                        'neg_mean_squared_error'])

_CLASSIFICATION_ONLY_OBJ_FUNCS = frozenset(['accuracy',
                                            'precision',
                                            'recall',
                                            'f1',
                                            'f1_score_micro',
                                            'f1_score_macro',
                                            'f1_score_weighted',
                                            'f1_score_least_frequent',
                                            'average_precision',
                                            'roc_auc',
                                            'neg_log_loss'])

_INT_CLASS_OBJ_FUNCS = frozenset(['unweighted_kappa',
                                  'linear_weighted_kappa',
                                  'quadratic_weighted_kappa',
                                  'uwk_off_by_one',
                                  'lwk_off_by_one',
                                  'qwk_off_by_one',
                                  'neg_log_loss'])

_REQUIRES_DENSE = (BayesianRidge,
                   GradientBoostingClassifier,
                   GradientBoostingRegressor,
                   Lars,
                   TheilSenRegressor)

MAX_CONCURRENT_PROCESSES = int(os.getenv('SKLL_MAX_CONCURRENT_PROCESSES', '3'))


# pylint: disable=W0223,R0903
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

    def __init__(self, keep, example_ids):
        super(FilteredLeaveOneGroupOut, self).__init__()
        self.keep = keep
        self.example_ids = example_ids
        self._warned = False

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
                     objective='f1_score_micro'):
    """
    A utility method to train a given learner instance on the given training examples,
    generate predictions on the training set itself and also the given
    test set, and score those predictions using the given objective function.
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
    objective : str, optional
        The objective function passed to ``use_score_func()``.
        Defaults to ``'f1_score_micro'``.

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
                                  if not label in learner.label_list]
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

    train_score = use_score_func(objective, train_labels, train_predictions)
    test_score = use_score_func(objective, test_labels, test_predictions)
    return train_score, test_score


def _predict_binary(self, X):
    """
    A helper function to allow us to use ``GridSearchCV`` with objective
    functions like Kendall's tau for binary classification problems (where the
    probability of the true class is used as the input to the objective
    function).

    This only works if we've also taken the step of storing the old predict
    function for ``self`` as ``predict_normal``. It's kind of a hack, but it saves
    us from having to override ``GridSearchCV`` to change one little line.

    Parameters
    ----------
    X : array-like
        A set of examples to predict values for.

    Returns
    -------
    res : array-like
        The prediction results.
    """

    if self.coef_.shape[0] == 1:
        res = self.predict_proba(X)[:, 1]
    else:
        res = self.predict_normal(X)
    return res


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
            res = (((res - self.yhat_mean) / self.yhat_sd)
                   * self.y_sd) + self.y_mean

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
        The string for the positive label in the binary
        classification setting.  Otherwise, an arbitrary
        label is picked.
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

    def __init__(self, model_type, probability=False, feature_scaling='none',
                 model_kwargs=None, pos_label_str=None, min_feature_count=1,
                 sampler=None, sampler_kwargs=None, custom_learner_path=None,
                 logger=None):
        """
        Initializes a learner object with the specified settings.
        """
        super(Learner, self).__init__()

        self.feat_vectorizer = None
        self.scaler = None
        self.label_dict = None
        self.label_list = None
        self.pos_label_str = pos_label_str
        self._model = None
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
        elif issubclass(self._model_type, SGDClassifier):
            self._model_kwargs['loss'] = 'log'
        elif issubclass(self._model_type, RANSACRegressor):
            self._model_kwargs['loss'] = 'squared_loss'
        elif issubclass(self._model_type, (MLPClassifier, MLPRegressor)):
            self._model_kwargs['learning_rate'] = 'invscaling'
            self._model_kwargs['max_iter'] = 500


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
                base_estimator_kwargs = {} if base_estimator_name in ['LinearRegression',
                                                                      'MultinomialNB',
                                                                      'SVR'] else {'random_state': 123456789}
                base_estimator = globals()[base_estimator_name](**base_estimator_kwargs)
                model_kwargs['base_estimator'] = base_estimator
            self._model_kwargs.update(model_kwargs)

    @classmethod
    def from_file(cls, learner_path):
        """
        Load a saved ``Learner`` instance from a file path.

        Parameters
        ----------
        learner_path : str
            The path to a saved ``Learner`` instance file.

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

        # For backward compatibility, convert string model types to labels.
        if isinstance(learner._model_type, string_types):
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
                    new_scaler.scale_ =  new_scaler.__dict__['std_']
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

    @property
    def model_params(self):
        """
        Model parameters (i.e., weights) for a ``LinearModel`` (e.g., ``Ridge``)
        regression and liblinear models.

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

            # convert SVR coefficient format (1 x matrix) to array
            if isinstance(self._model, SVR):
                coef = coef.toarray()[0]

            # inverse transform to get indices for before feature selection
            coef = coef.reshape(1, -1)
            coef = self.feat_selector.inverse_transform(coef)[0]
            for feat, idx in iteritems(self.feat_vectorizer.vocabulary_):
                if coef[idx]:
                    res[feat] = coef[idx]

        elif isinstance(self._model, LinearSVC) or isinstance(self._model, LogisticRegression):
            label_list = self.label_list

            # if there are only two labels, scikit-learn will only have one
            # set of parameters and they will be associated with label 1 (not
            # 0)
            if len(self.label_list) == 2:
                label_list = self.label_list[-1:]

            for i, label in enumerate(label_list):
                coef = self.model.coef_[i]
                coef = coef.reshape(1, -1)
                coef = self.feat_selector.inverse_transform(coef)[0]
                for feat, idx in iteritems(self.feat_vectorizer.vocabulary_):
                    if coef[idx]:
                        res['{}\t{}'.format(label, feat)] = coef[idx]

            if isinstance(self.model.intercept_, float):
                intercept = {'_intercept_': self.model.intercept_}
            elif self.model.intercept_.any():
                intercept = dict(zip(label_list, self.model.intercept_))

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
                if isinstance(label, string_types):
                    raise TypeError("You are doing regression with string "
                                    "labels.  Convert them to integers or "
                                    "floats.")

        # make sure that feature values are not strings
        for val in examples.features.data:
            if isinstance(val, string_types):
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

        # extract list of unique labels if we are doing classification
        self.label_list = np.unique(examples.labels).tolist()

        # if one label is specified as the positive class, make sure it's
        # last
        if self.pos_label_str:
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

        # Create scaler if we weren't passed one and it's necessary
        if not issubclass(self._model_type, MultinomialNB):
            if self._feature_scaling != 'none':
                scale_with_mean = self._feature_scaling in {
                    'with_mean', 'both'}
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
              grid_search=True, grid_objective='f1_score_micro',
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
            doing the grid search.
            Defaults to ``'f1_score_micro'``.
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
            ``False``.
            Defaults to ``True``.

        Returns
        -------
        grid_score : float
            The best grid search objective function score, or 0 if we're
            not doing grid search.

        Raises
        ------
        ValueError
            If grid_objective is not a valid grid objective.
        MemoryError
            If process runs out of memory converting training data to dense.
        ValueError
            If FeatureHasher is used with MultinomialNB.
        """

        # if we are asked to do grid search, check that the grid objective
        # function is valid for the selected learner
        if grid_search:
            if self.model_type._estimator_type == 'regressor':
                # types 2-4 are valid for all regression models
                if grid_objective in _CLASSIFICATION_ONLY_OBJ_FUNCS:
                    raise ValueError("{} is not a valid grid objective "
                                     "function for the {} learner"
                                     .format(grid_objective,
                                             self._model_type.__name__))
            elif grid_objective not in _CLASSIFICATION_ONLY_OBJ_FUNCS:
                # This is a classifier. Valid objective functions depend on
                # type of label (int, string, binary)

                if issubclass(examples.labels.dtype.type, int):
                    # If they're ints, class 1 and 2 are valid for classifiers,
                    if grid_objective not in _INT_CLASS_OBJ_FUNCS:
                        raise ValueError("{} is not a valid grid objective "
                                         "function for the {} learner with "
                                         "integer labels"
                                         .format(grid_objective,
                                                 self._model_type.__name__))

                elif issubclass(examples.labels.dtype.type, str):
                    # if all of the labels are strings, only class 1 objectives
                    # are valid (with a classifier).
                    raise ValueError("{} is not a valid grid objective "
                                     "function for the {} learner with string "
                                     "labels".format(grid_objective,
                                                     self._model_type.__name__))

                elif len(set(examples.labels)) == 2:
                    # If there are two labels, class 3 objectives are valid for
                    # classifiers regardless of the type of the label.
                    if grid_objective not in _BINARY_CLASS_OBJ_FUNCS:
                        raise ValueError("{} is not a valid grid objective "
                                         "function for the {} learner with "
                                         "binary labels"
                                         .format(grid_objective,
                                                 self._model_type.__name__))
                elif grid_objective in _REGRESSION_ONLY_OBJ_FUNCS:
                    # simple backoff check for mixed-type labels
                    raise ValueError("{} is not a valid grid objective "
                                     "function for the {} learner"
                                     .format(grid_objective,
                                             self._model_type.__name__))

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
            raise ValueError('Cannot use FeatureHasher with MultinomialNB, '
                             'because MultinomialNB cannot handle negative '
                             'feature values.')

        # Scale features if necessary
        if not issubclass(self._model_type, MultinomialNB):
            xtrain = self.scaler.fit_transform(xtrain)

        # check whether any feature values are too large
        self._check_max_feature_value(xtrain)

        # Sampler
        if self.sampler:
            self.logger.warning('Sampler converts sparse matrix to dense')
            if isinstance(self.sampler, SkewedChi2Sampler):
                self.logger.warning('SkewedChi2Sampler uses a dense matrix')
                xtrain = self.sampler.fit_transform(xtrain.todense())
            else:
                xtrain = self.sampler.fit_transform(xtrain)

        # use label dict transformed version of examples.labels if doing
        # classification
        if self.model_type._estimator_type == 'classifier':
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
                dummy_label = next(itervalues(grid_search_folds))
                fold_groups = [grid_search_folds.get(curr_id, dummy_label) for
                               curr_id in examples.ids]
                kfold = FilteredLeaveOneGroupOut(grid_search_folds, examples.ids)
                folds = kfold.split(examples.features, examples.labels, fold_groups)

            # If we're using a correlation metric for doing binary
            # classification, override the estimator's predict function
            if (grid_objective in _CORRELATION_METRICS and
                    self.model_type._estimator_type == 'classifier'):
                estimator.predict_normal = estimator.predict
                estimator.predict = _predict_binary

            # limit the number of grid_jobs to be no higher than five or the
            # number of cores for the machine, whichever is lower
            grid_jobs = min(grid_jobs, cpu_count(), MAX_CONCURRENT_PROCESSES)

            grid_searcher = GridSearchCV(estimator, param_grid,
                                         scoring=grid_objective,
                                         cv=folds,
                                         n_jobs=grid_jobs,
                                         pre_dispatch=grid_jobs)

            # run the grid search for hyperparameters
            grid_searcher.fit(xtrain, labels)
            self._model = grid_searcher.best_estimator_
            grid_score = grid_searcher.best_score_
        else:
            self._model = estimator.fit(xtrain, labels)
            grid_score = 0.0

        return grid_score

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

        # make the prediction on the test data
        yhat = self.predict(examples,
                            prediction_prefix=prediction_prefix,
                            append=append)

        # make a single list of metrics including the grid objective
        # since it's easier to compute everything together
        metrics_to_compute = [grid_objective] + output_metrics

        # extract actual labels (transformed for classification tasks)
        if self.model_type._estimator_type == 'classifier':
            test_label_list = np.unique(examples.labels).tolist()

            # identify unseen test labels if any and add a new dictionary for these
            # labels
            unseen_test_label_list = [label for label in test_label_list
                                      if not label in self.label_list]
            unseen_label_dict = {label: i for i, label in enumerate(unseen_test_label_list,
                                                                    start=len(self.label_list))}
            # combine the two dictionaries
            train_and_test_label_dict = self.label_dict.copy()
            train_and_test_label_dict.update(unseen_label_dict)
            ytest = np.array([train_and_test_label_dict[label]
                              for label in examples.labels])
        else:
            ytest = examples.labels

        # compute all of the metrics that we need to but save the original
        # predictions since we will need to use those for each metric
        original_yhat = yhat
        for metric in metrics_to_compute:

            # if run in probability mode, convert yhat to list of labels predicted
            if self.probability:
                # if we're using a correlation grid objective, calculate it here
                if metric and metric in _CORRELATION_METRICS:
                    try:
                        metric_scores[metric] = use_score_func(metric, ytest, yhat[:, 1])
                    except ValueError:
                        metric_scores[metric] = float('NaN')

                yhat = np.array([max(range(len(row)),
                                     key=lambda i: row[i])
                                 for row in original_yhat])

            # calculate grid search objective function score, if specified
            if (metric and (metric not in _CORRELATION_METRICS or
                                   not self.probability)):
                try:
                    metric_scores[metric] = use_score_func(metric, ytest, yhat)
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

        if self.model_type._estimator_type == 'regressor':
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
            # compute the confusion matrix
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
            For classifier, should we convert class indices to their (str) labels?
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
        if isinstance(self.feat_vectorizer, FeatureHasher):
            if (self.feat_vectorizer.n_features !=
                    examples.vectorizer.n_features):
                self.logger.warning("There is mismatch between the training model "
                                    "features and the data passed to predict.")

            self_feat_vec_tuple = (self.feat_vectorizer.dtype,
                                   self.feat_vectorizer.input_type,
                                   self.feat_vectorizer.n_features,
                                   self.feat_vectorizer.non_negative)
            example_feat_vec_tuple = (examples.vectorizer.dtype,
                                      examples.vectorizer.input_type,
                                      examples.vectorizer.n_features,
                                      examples.vectorizer.non_negative)

            if self_feat_vec_tuple == example_feat_vec_tuple:
                xtest = examples.features
            else:
                xtest = self.feat_vectorizer.transform(
                    examples.vectorizer.inverse_transform(
                        examples.features))
        else:
            if (set(self.feat_vectorizer.feature_names_) !=
                    set(examples.vectorizer.feature_names_)):
                self.logger.warning("There is mismatch between the training model "
                                    "features and the data passed to predict.")
            if self.feat_vectorizer == examples.vectorizer:
                xtest = examples.features
            else:

                xtest = self.feat_vectorizer.transform(
                    examples.vectorizer.inverse_transform(
                        examples.features))

        # filter features based on those selected from training set
        xtest = self.feat_selector.transform(xtest)

        # Sampler
        if self.sampler:
            self.logger.warning('Sampler converts sparse matrix to dense')
            if isinstance(self.sampler, SkewedChi2Sampler):
                self.logger.warning('SkewedChi2Sampler uses a dense matrix')
                xtest = self.sampler.fit_transform(xtest.todense())
            else:
                xtest = self.sampler.fit_transform(xtest)

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
                       grid_search=False,
                       grid_search_folds=3,
                       grid_jobs=None,
                       grid_objective='f1_score_micro',
                       output_metrics=[],
                       prediction_prefix=None,
                       param_grid=None,
                       shuffle=False,
                       save_cv_folds=False,
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
            doing the grid search.
            Defaults to ``'f1_score_micro'``.
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
        skll_fold_ids : dict
            A dictionary containing the test-fold number for each id
            if ``save_cv_folds`` is ``True``, otherwise ``None``.

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
            dummy_label = next(itervalues(cv_folds))
            fold_groups = [cv_folds.get(curr_id, dummy_label) for curr_id in examples.ids]
            # Only retain IDs within folds if they're in cv_folds
            kfold = FilteredLeaveOneGroupOut(cv_folds, examples.ids)
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
        append_predictions = False
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
            grid_search_score = self.train(train_set,
                                           grid_search_folds=grid_search_folds,
                                           grid_search=grid_search,
                                           grid_objective=grid_objective,
                                           param_grid=param_grid,
                                           grid_jobs=grid_jobs,
                                           shuffle=grid_search,
                                           create_label_dict=False)
            grid_search_scores.append(grid_search_score)
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

        # return list of results for all folds
        return results, grid_search_scores, skll_fold_ids

    def learning_curve(self,
                       examples,
                       cv_folds=10,
                       train_sizes=np.linspace(0.1, 1.0, 5),
                       metric='f1_score_micro'):
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
        metric : str, optional
            The name of the metric function to use
            when computing the train and test scores
            for the learning curve. (default: 'f1_score_micro')
            Defaults to ``'f1_score_micro'``.

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
