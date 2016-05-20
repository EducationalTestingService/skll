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
from sklearn.cross_validation import KFold, LeaveOneLabelOut, StratifiedKFold
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
# AdditiveChi2Sampler is used indirectly, so ignore linting message
from sklearn.kernel_approximation import (AdditiveChi2Sampler, Nystroem,
                                          RBFSampler, SkewedChi2Sampler)
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge, SGDClassifier,
                                  SGDRegressor)
from sklearn.linear_model.base import LinearModel
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, SCORERS)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
                        DecisionTreeClassifier:
                        [{'max_features': ["auto", None]}],
                        DecisionTreeRegressor:
                        [{'max_features': ["auto", None]}],
                        ElasticNet:
                        [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        GradientBoostingClassifier:
                        [{'max_depth': [1, 3, 5]}],
                        GradientBoostingRegressor:
                        [{'max_depth': [1, 3, 5]}],
                        KNeighborsClassifier:
                        [{'n_neighbors': [1, 5, 10, 100],
                          'weights': ['uniform', 'distance']}],
                        KNeighborsRegressor:
                        [{'n_neighbors': [1, 5, 10, 100],
                          'weights': ['uniform', 'distance']}],
                        Lasso:
                        [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        LinearRegression:
                        [{}],
                        LinearSVC:
                        [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        LogisticRegression:
                        [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        SVC: [{'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                               'gamma': ['auto', 0.01, 0.1, 1.0, 10.0, 100.0]}],
                        MultinomialNB:
                        [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}],
                        RandomForestClassifier:
                        [{'max_depth': [1, 5, 10, None]}],
                        RandomForestRegressor:
                        [{'max_depth': [1, 5, 10, None]}],
                        Ridge:
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
                          'gamma': ['auto', 0.01, 0.1, 1.0, 10.0, 100.0]}]}


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
                                     'spearman'])

_REGRESSION_ONLY_OBJ_FUNCS = frozenset(['r2',
                                        'mean_squared_error'])

_CLASSIFICATION_ONLY_OBJ_FUNCS = frozenset(['accuracy',
                                            'precision',
                                            'recall',
                                            'f1',
                                            'f1_score_micro',
                                            'f1_score_macro',
                                            'f1_score_weighted',
                                            'f1_score_least_frequent',
                                            'average_precision',
                                            'roc_auc'])

_INT_CLASS_OBJ_FUNCS = frozenset(['unweighted_kappa',
                                  'linear_weighted_kappa',
                                  'quadratic_weighted_kappa',
                                  'uwk_off_by_one',
                                  'lwk_off_by_one',
                                  'qwk_off_by_one'])

_REQUIRES_DENSE = (GradientBoostingClassifier, GradientBoostingRegressor)

MAX_CONCURRENT_PROCESSES = int(os.getenv('SKLL_MAX_CONCURRENT_PROCESSES', '5'))


# pylint: disable=W0223,R0903
class FilteredLeaveOneLabelOut(LeaveOneLabelOut):

    """
    Version of LeaveOneLabelOut cross-validation iterator that only outputs
    indices of instances with IDs in a prespecified set.
    """

    def __init__(self, labels, keep, examples):
        super(FilteredLeaveOneLabelOut, self).__init__(labels)
        self.keep = keep
        self.examples = examples
        self._warned = False
        self.logger = logging.getLogger(__name__)

    def __iter__(self):
        for train_index, test_index in super(FilteredLeaveOneLabelOut,
                                             self).__iter__():
            train_len = len(train_index)
            test_len = len(test_index)
            train_index = [i for i in train_index if self.examples.ids[i] in
                           self.keep]
            test_index = [i for i in test_index if self.examples.ids[i] in
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
    """
    for key_cls, grid in _DEFAULT_PARAM_GRIDS.items():
        if issubclass(cls, key_cls):
            return grid
    return None


def _import_custom_learner(custom_learner_path, custom_learner_name):
    """
    Does the gruntwork of adding the custom model's module to globals.
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


def _predict_binary(self, X):
    """
    Little helper function to allow us to use `GridSearchCV` with objective
    functions like Kendall's tau for binary classification problems (where the
    probability of the true class is used as the input to the objective
    function).

    This only works if we've also taken the step of storing the old predict
    function for `self` as `predict_normal`. It's kind of a hack, but it saves
    us from having to override GridSearchCV to change one little line.

    :param self: A scikit-learn classifier instance
    :param X: A set of examples to predict values for.
    :type X: array
    """

    if self.coef_.shape[0] == 1:
        res = self.predict_proba(X)[:, 1]
    else:
        res = self.predict_normal(X)
    return res


class SelectByMinCount(SelectKBest):

    """
    Select features ocurring in more (and/or fewer than) than a specified
    number of examples in the training data (or a CV training fold).
    """

    def __init__(self, min_count=1):
        self.min_count = min_count
        self.scores_ = None

    def fit(self, X, y=None):
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
        Adapted from SelectKBest.
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

    :param cls: A regressor to add rescaling to.
    :type cls: BaseEstimator

    :returns: Modified version of class with rescaled functions added.
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
        This is adapted from scikit-learns's BaseEstimator class.
        It gets the kwargs for the superclass's init method and adds the
        kwargs for newly added __init__ method.
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
class RescaledKNeighborsRegressor(KNeighborsRegressor):
    pass


@rescaled
class RescaledLasso(Lasso):
    pass


@rescaled
class RescaledLinearRegression(LinearRegression):
    pass


@rescaled
class RescaledRandomForestRegressor(RandomForestRegressor):
    pass


@rescaled
class RescaledRidge(Ridge):
    pass


@rescaled
class RescaledSVR(SVR):
    pass


@rescaled
class RescaledLinearSVR(LinearSVR):
    pass


@rescaled
class RescaledSGDRegressor(SGDRegressor):
    pass


class Learner(object):

    """
    A simpler learner interface around many scikit-learn classification
    and regression functions.

    :param model_type: Type of estimator to create (e.g., LogisticRegression).
                       See the skll package documentation for valid options.
    :type model_type: str
    :param probability: Should learner return probabilities of all
                        labels (instead of just label with highest
                        probability)?
    :type probability: bool
    :param feature_scaling: how to scale the features, if at all. Options are:
                    'with_std': scale features using the standard deviation,
                    'with_mean': center features using the mean,
                    'both': do both scaling as well as centering,
                    'none': do neither scaling nor centering
    :type feature_scaling: str
    :param model_kwargs: A dictionary of keyword arguments to pass to the
                         initializer for the specified model.
    :type model_kwargs: dict
    :param pos_label_str: The string for the positive label in the binary
                          classification setting.  Otherwise, an arbitrary
                          label is picked.
    :type pos_label_str: str
    :param min_feature_count: The minimum number of examples a feature
                              must have a nonzero value in to be included.
    :type min_feature_count: int
    :param sampler: The sampler to use for kernel approximation, if desired.
                    Valid values are: ``'AdditiveChi2Sampler'``, ``'Nystroem'``,
                    ``'RBFSampler'``, and ``'SkewedChi2Sampler'``.
    :type sampler: str
    :param sampler_kwargs: A dictionary of keyword arguments to pass to the
                          initializer for the specified sampler.
    :type sampler_kwargs: dict
    :param custom_learner_path: Path to module where a custom classifier is
                                defined.
    :type custom_learner_path: str
    """

    def __init__(self, model_type, probability=False, feature_scaling='none',
                 model_kwargs=None, pos_label_str=None, min_feature_count=1,
                 sampler=None, sampler_kwargs=None, custom_learner_path=None):
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
                logger = logging.getLogger(__name__)
                logger.warning('Because LibSVM does an internal ' +
                               'cross-validation to produce probabilities, ' +
                               'results will not be exactly replicable when ' +
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

        if issubclass(self._model_type,
                      (RandomForestClassifier, LinearSVC, LogisticRegression,
                       DecisionTreeClassifier, GradientBoostingClassifier,
                       GradientBoostingRegressor, DecisionTreeRegressor,
                       RandomForestRegressor, SGDClassifier, SGDRegressor,
                       AdaBoostRegressor, AdaBoostClassifier, LinearSVR,
                       Lasso, Ridge, ElasticNet, SVC)):
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
            # if the model is an AdaBoost classifier or regressor, then we
            # need to convert any specified `base_estimator` (a string)
            # into an object before passing it in to the learner constructor.
            # we also need to make sure that if the base estimator is
            # anything other than MultinomialNB, we set the random state
            # to a fixed seed such that results are replicable
            if issubclass(self._model_type,
                          (AdaBoostRegressor, AdaBoostClassifier)) and ('base_estimator' in model_kwargs):
                base_estimator_name = model_kwargs['base_estimator']
                base_estimator_kwargs = {} if base_estimator_name in ['MultinomialNB', 'SVR'] else {'random_state': 123456789}
                base_estimator = globals()[base_estimator_name](**base_estimator_kwargs)
                model_kwargs['base_estimator'] = base_estimator
            self._model_kwargs.update(model_kwargs)

    @classmethod
    def from_file(cls, learner_path):
        """
        :returns: New instance of Learner from the pickle at the specified
                  path.
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
        """ The model type (i.e., the class) """
        return self._model_type

    @property
    def model_kwargs(self):
        """
        A dictionary of the underlying scikit-learn model's keyword arguments
        """
        return self._model_kwargs

    @property
    def model(self):
        """ The underlying scikit-learn model """
        return self._model

    def load(self, learner_path):
        """
        Replace the current learner instance with a saved learner.

        :param learner_path: The path to the file to load.
        :type learner_path: str
        """
        del self.__dict__
        self.__dict__ = Learner.from_file(learner_path).__dict__

    @property
    def model_params(self):
        """
        Model parameters (i.e., weights) for ``LinearModel`` (e.g., ``Ridge``)
        regression and liblinear models.

        :returns: Labeled weights and (labeled if more than one) intercept
                  value(s)
        :rtype: tuple of (``weights``, ``intercepts``), where ``weights`` is a
                dict and ``intercepts`` is a dictionary
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
                              ).format(self._model_type))

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
        # LinearSVC doesn't support predict_proba
        self._probability = value
        if not hasattr(self.model_type, "predict_proba") and value:
            logger = logging.getLogger(__name__)
            logger.warning(("probability was set to True, but {} does not have"
                            " a predict_proba() method.")
                           .format(self.model_type))
            self._probability = False

    def save(self, learner_path):
        """
        Save the learner to a file.

        :param learner_path: The path to where you want to save the learner.
        :type learner_path: str
        """
        # create the directory if it doesn't exist
        learner_dir = os.path.dirname(learner_path)
        if not os.path.exists(learner_dir):
            os.makedirs(learner_dir)
        # write out the files
        joblib.dump((VERSION, self), learner_path)

    def _create_estimator(self):
        """
        :returns: A tuple containing an instantiation of the requested
                  estimator, and a parameter grid to search.
        """
        estimator = None
        default_param_grid = _find_default_param_grid(self._model_type)
        if default_param_grid is None:
            raise ValueError("%s is not a valid learner type." %
                             (self._model_type,))

        estimator = self._model_type(**self._model_kwargs)

        return estimator, default_param_grid

    def _check_input_formatting(self, examples):
        """
        check that the examples are properly formatted.
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

    @staticmethod
    def _check_max_feature_value(feat_array):
        """
        Check if the the maximum absolute value of any feature is too large
        """
        max_feat_abs = np.max(np.abs(feat_array.data))
        if max_feat_abs > 1000.0:
            logger = logging.getLogger(__name__)
            logger.warning(("You have a feature with a very large absolute " +
                            "value (%s).  That may cause the learning " +
                            "algorithm to crash or perform " +
                            "poorly."), max_feat_abs)

    def _create_label_dict(self, examples):
        """
        Creates a dictionary of labels for classification problems.

        :param examples: The examples to use for training.
        :type examples: FeatureSet
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

        :param examples: The examples to use for training.
        :type examples: FeatureSet
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

        :param examples: The examples to train the model on.
        :type examples: FeatureSet
        :param param_grid: The parameter grid to search through for grid
                           search. If unspecified, a default parameter grid
                           will be used.
        :type param_grid: list of dicts mapping from strs to
                          lists of parameter values
        :param grid_search_folds: The number of folds to use when doing the
                                  grid search, or a mapping from
                                  example IDs to folds.
        :type grid_search_folds: int or dict
        :param grid_search: Should we do grid search?
        :type grid_search: bool
        :param grid_objective: The objective function to use when doing the
                               grid search.
        :type grid_objective: function
        :param grid_jobs: The number of jobs to run in parallel when doing the
                          grid search. If unspecified or 0, the number of
                          grid search folds will be used.
        :type grid_jobs: int
        :param shuffle: Shuffle examples (e.g., for grid search CV.)
        :type shuffle: bool
        :param create_label_dict: Should we create the label dictionary?  This
                                  dictionary is used to map between string
                                  labels and their corresponding numerical
                                  values.  This should only be done once per
                                  experiment, so when ``cross_validate`` calls
                                  ``train``, ``create_label_dict`` gets set to
                                  ``False``.
        :type create_label_dict: bool

        :return: The best grid search objective function score, or 0 if we're
                 not doing grid search.
        :rtype: float
        """
        logger = logging.getLogger(__name__)

        # if we are asked to do grid search, check that the grid objective
        # function is valid for the selected learner
        if grid_search:
            if self.model_type._estimator_type == 'regressor':
                # types 2-4 are valid for all regression models
                if grid_objective in _CLASSIFICATION_ONLY_OBJ_FUNCS:
                    raise ValueError("{} is not a valid grid objective "
                                     "function for the {} learner"
                                     .format(grid_objective,
                                             self._model_type))
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
                                                 self._model_type))

                elif issubclass(examples.labels.dtype.type, str):
                    # if all of the labels are strings, only class 1 objectives
                    # are valid (with a classifier).
                    raise ValueError("{} is not a valid grid objective "
                                     "function for the {} learner with string "
                                     "labels".format(grid_objective,
                                                     self._model_type))

                elif len(set(examples.labels)) == 2:
                    # If there are two labels, class 3 objectives are valid for
                    # classifiers regardless of the type of the label.
                    if grid_objective not in _BINARY_CLASS_OBJ_FUNCS:
                        raise ValueError("{} is not a valid grid objective "
                                         "function for the {} learner with "
                                         "binary labels"
                                         .format(grid_objective,
                                                 self._model_type))
                elif grid_objective in _REGRESSION_ONLY_OBJ_FUNCS:
                    # simple backoff check for mixed-type labels
                    raise ValueError("{} is not a valid grid objective "
                                     "function for the {} learner"
                                     .format(grid_objective,
                                             self._model_type))

        # Shuffle so that the folds are random for the inner grid search CV.
        # If grid search is True but shuffle isn't, shuffle anyway.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if grid_search or shuffle:
            if grid_search and not shuffle:
                logger.warning('Training data will be shuffled to randomize '
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
                              'matrices.').format(self._model_type)
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
            logger.warning('Sampler converts sparse matrix to dense')
            if isinstance(self.sampler, SkewedChi2Sampler):
                logger.warning('SkewedChi2Sampler uses a dense matrix')
                xtrain = self.sampler.fit_transform(xtrain.todense())
            else:
                xtrain = self.sampler.fit_transform(xtrain)

        # Instantiate an estimator and get the default parameter grid to search
        estimator, default_param_grid = self._create_estimator()

        # use label dict transformed version of examples.labels if doing
        # classification
        if self.model_type._estimator_type == 'classifier':
            labels = np.array([self.label_dict[label] for label in
                               examples.labels])
        else:
            labels = examples.labels

        # set up a grid searcher if we are asked to
        if grid_search:
            # set up grid search folds
            if isinstance(grid_search_folds, int):
                grid_search_folds = \
                    self._compute_num_folds_from_example_counts(
                        grid_search_folds, labels)

                if not grid_jobs:
                    grid_jobs = grid_search_folds
                else:
                    grid_jobs = min(grid_search_folds, grid_jobs)
                folds = grid_search_folds
            else:
                # use the number of unique fold IDs as the number of grid jobs
                if not grid_jobs:
                    grid_jobs = len(np.unique(grid_search_folds))
                else:
                    grid_jobs = min(len(np.unique(grid_search_folds)),
                                    grid_jobs)
                # Only retain IDs within folds if they're in grid_search_folds
                dummy_label = next(itervalues(grid_search_folds))
                fold_labels = [grid_search_folds.get(curr_id, dummy_label) for
                               curr_id in examples.ids]
                folds = FilteredLeaveOneLabelOut(fold_labels,
                                                 grid_search_folds,
                                                 examples)

            # Use default parameter grid if we weren't passed one
            if not param_grid:
                param_grid = default_param_grid

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
                                         scoring=grid_objective, cv=folds,
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
                 grid_objective=None):
        """
        Evaluates a given model on a given dev or test example set.

        :param examples: The examples to evaluate the performance of the model
                         on.
        :type examples: FeatureSet
        :param prediction_prefix: If saving the predictions, this is the
                                  prefix that will be used for the filename.
                                  It will be followed by ".predictions"
        :type prediction_prefix: str
        :param append: Should we append the current predictions to the file if
                       it exists?
        :type append: bool
        :param grid_objective: The objective function that was used when doing
                               the grid search.
        :type grid_objective: function

        :return: The confusion matrix, the overall accuracy, the per-label
                 PRFs, the model parameters, and the grid search objective
                 function score.
        :rtype: 5-tuple
        """
        # initialize grid score
        grid_score = None

        # make the prediction on the test data
        yhat = self.predict(examples, prediction_prefix=prediction_prefix,
                            append=append)

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

        # if run in probability mode, convert yhat to list of labels predicted
        if self.probability:
            # if we're using a correlation grid objective, calculate it here
            if grid_objective and grid_objective in _CORRELATION_METRICS:
                try:
                    grid_score = use_score_func(grid_objective, ytest,
                                                yhat[:, 1])
                except ValueError:
                    grid_score = float('NaN')

            yhat = np.array([max(range(len(row)),
                                 key=lambda i: row[i])
                             for row in yhat])

        # calculate grid search objective function score, if specified
        if (grid_objective and (grid_objective not in _CORRELATION_METRICS or
                                not self.probability)):
            try:
                grid_score = use_score_func(grid_objective, ytest, yhat)
            except ValueError:
                grid_score = float('NaN')

        if self.model_type._estimator_type == 'regressor':
            result_dict = {'descriptive': defaultdict(dict)}
            for table_label, y in zip(['actual', 'predicted'], [ytest, yhat]):
                result_dict['descriptive'][table_label]['min'] = min(y)
                result_dict['descriptive'][table_label]['max'] = max(y)
                result_dict['descriptive'][table_label]['avg'] = np.mean(y)
                result_dict['descriptive'][table_label]['std'] = np.std(y)
            result_dict['pearson'] = use_score_func('pearson', ytest, yhat)
            res = (None, None, result_dict, self._model.get_params(),
                   grid_score)
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
                   self._model.get_params(), grid_score)
        return res

    def predict(self, examples, prediction_prefix=None, append=False,
                class_labels=False):
        """
        Uses a given model to generate predictions on a given data set

        :param examples: The examples to predict the labels for.
        :type examples: FeatureSet
        :param prediction_prefix: If saving the predictions, this is the
                                  prefix that will be used for the
                                  filename. It will be followed by
                                  ".predictions"
        :type prediction_prefix: str
        :param append: Should we append the current predictions to the file if
                       it exists?
        :type append: bool
        :param class_labels: For classifier, should we convert class
                             indices to their (str) labels?
        :type class_labels: bool

        :return: The predictions returned by the learner.
        :rtype: array
        """
        logger = logging.getLogger(__name__)
        example_ids = examples.ids

        # Need to do some transformations so the features are in the right
        # columns for the test set. Obviously a bit hacky, but storing things
        # in sparse matrices saves memory over our old list of dicts approach.
        if isinstance(self.feat_vectorizer, FeatureHasher):
            if (self.feat_vectorizer.n_features !=
                    examples.vectorizer.n_features):
                logger.warning("There is mismatch between the training model "
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
                logger.warning("There is mismatch between the training model "
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
            logger.warning('Sampler converts sparse matrix to dense')
            if isinstance(self.sampler, SkewedChi2Sampler):
                logger.warning('SkewedChi2Sampler uses a dense matrix')
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
                              'matrices.').format(self._model_type)
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
            logger.error("Model type: %s\nModel: %s\nProbability: %s\n",
                         self._model_type, self._model, self.probability)
            raise e

        # write out the predictions if we are asked to
        if prediction_prefix is not None:
            prediction_file = '{}.predictions'.format(prediction_prefix)
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
            logger = logging.getLogger(__name__)
            logger.warning('The minimum number of examples per label was %s.  '
                           'Setting the number of cross-validation folds to '
                           'that value.', min_examples_per_label)
            cv_folds = min_examples_per_label
        return cv_folds

    def cross_validate(self, examples, stratified=True, cv_folds=10,
                       grid_search=False, grid_search_folds=3, grid_jobs=None,
                       grid_objective='f1_score_micro', prediction_prefix=None,
                       param_grid=None, shuffle=False, save_cv_folds=False):
        """
        Cross-validates a given model on the training examples.

        :param examples: The data to cross-validate learner performance on.
        :type examples: FeatureSet
        :param stratified: Should we stratify the folds to ensure an even
                           distribution of labels for each fold?
        :type stratified: bool
        :param cv_folds: The number of folds to use for cross-validation, or
                         a mapping from example IDs to folds.
        :type cv_folds: int or dict
        :param grid_search: Should we do grid search when training each fold?
                            Note: This will make this take *much* longer.
        :type grid_search: bool
        :param grid_search_folds: The number of folds to use when doing the
                                  grid search (ignored if cv_folds is set to
                                  a dictionary mapping examples to folds).
        :type grid_search_folds: int
        :param grid_jobs: The number of jobs to run in parallel when doing the
                          grid search. If unspecified or 0, the number of
                          grid search folds will be used.
        :type grid_jobs: int
        :param grid_objective: The objective function to use when doing the
                               grid search.
        :type grid_objective: function
        :param param_grid: The parameter grid to search through for grid
                           search. If unspecified, a default parameter
                           grid will be used.
        :type param_grid: list of dicts mapping from strs to
                          lists of parameter values
        :param prediction_prefix: If saving the predictions, this is the
                                  prefix that will be used for the filename.
                                  It will be followed by ".predictions"
        :type prediction_prefix: str
        :param shuffle: Shuffle examples before splitting into folds for CV.
        :type shuffle: bool
        :param save_cv_folds: Whether to save the cv fold ids or not
        :type save_cv_folds: bool

        :return: The confusion matrix, overall accuracy, per-label PRFs, and
                 model parameters for each fold in one list, and another list
                 with the grid search scores for each fold. Also return a
                 dictionary containing the test-fold number for each id
                 if save_cv_folds is True, otherwise None.
        :rtype: (list of 4-tuples, list of float, dict)
        """

        # Seed the random number generator so that randomized algorithms are
        # replicable.
        random_state = np.random.RandomState(123456789)
        # Set up logger.
        logger = logging.getLogger(__name__)

        # Shuffle so that the folds are random for the inner grid search CV.
        # If grid search is True but shuffle isn't, shuffle anyway.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if grid_search or shuffle:
            if grid_search and not shuffle:
                logger.warning('Training data will be shuffled to randomize '
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
            cv_folds = self._compute_num_folds_from_example_counts(
                cv_folds, examples.labels)

            stratified = (stratified and
                          self.model_type._estimator_type == 'classifier')
            if stratified:
                kfold = StratifiedKFold(examples.labels, n_folds=cv_folds)
            else:
                kfold = KFold(len(examples.labels),
                              n_folds=cv_folds,
                              random_state=random_state)
        # Otherwise cv_volds is a dict
        else:
            # if we have a mapping from IDs to folds, use it for the overall
            # cross-validation as well as the grid search within each
            # training fold.  Note that this means that the grid search
            # will use K-1 folds because the Kth will be the test fold for
            # the outer cross-validation.
            # Only retain IDs within folds if they're in grid_search_folds
            dummy_label = next(itervalues(cv_folds))
            fold_labels = [cv_folds.get(curr_id, dummy_label) for curr_id in
                           examples.ids]
            # Only retain IDs within folds if they're in cv_folds
            kfold = FilteredLeaveOneLabelOut(fold_labels, cv_folds, examples)
            grid_search_folds = cv_folds

        # Save the cross-validation fold information, if required
        # The format is that the test-fold that each id appears in is stored
        skll_fold_ids = None
        if save_cv_folds:
            skll_fold_ids = {}
            for fold_num, (_, test_indices) in enumerate(kfold):
                for index in test_indices:
                    skll_fold_ids[examples.ids[index]] = str(fold_num)

        # handle each fold separately and accumulate the predictions and the
        # numbers
        results = []
        grid_search_scores = []
        append_predictions = False
        for train_index, test_index in kfold:
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
                                         grid_objective=grid_objective))
            append_predictions = True

        # return list of results for all folds
        return results, grid_search_scores, skll_fold_ids
