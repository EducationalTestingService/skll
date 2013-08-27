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
Provides easy-to-use wrapper around scikit-learn.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
'''

from __future__ import absolute_import, print_function, unicode_literals

import inspect
import logging
import os
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
import scipy.sparse as sp
from six import iteritems
from six import string_types
from six.moves import cPickle as pickle
from six.moves import xrange as range
from six.moves import zip
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, SCORERS)
from sklearn.preprocessing import StandardScaler
from sklearn.svm.base import BaseLibLinear
from sklearn.utils import shuffle as sk_shuffle
# sklearn models: these are used indirectly, so ignore linting messages
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from skll.data import ExamplesTuple
from skll.metrics import _CORRELATION_METRICS
from skll.version import VERSION


# Constants #
_DEFAULT_PARAM_GRIDS = {'LogisticRegression': [{'C': [0.01, 0.1, 1.0, 10.0,
                                                      100.0]}],
                        'LinearSVC': [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        'SVC': [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        'MultinomialNB': [{'alpha': [0.1, 0.25, 0.5, 0.75,
                                                     1.0]}],
                        'DecisionTreeClassifier': [{'max_features': ["auto",
                                                                     None]}],
                        'RandomForestClassifier': [{'max_depth': [1, 5, 10,
                                                                  None]}],
                        'GradientBoostingClassifier': [{'max_depth': [1, 3, 5],
                                                        'n_estimators': [500]}],
                        'Ridge': [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        'RescaledRidge': [{'alpha': [0.01, 0.1, 1.0, 10.0,
                                                     100.0]}],
                        'SVR': [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        'RescaledSVR': [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}],
                        'GradientBoostingRegressor': [{'max_depth': [1, 3, 5],
                                                       'n_estimators': [500]}]}
_REGRESSION_MODELS = frozenset(['Ridge', 'RescaledRidge', 'SVR',
                                'RescaledSVR', 'GradientBoostingRegressor'])
_REQUIRES_DENSE = frozenset(['MultinomialNB', 'RandomForestClassifier',
                             'GradientBoostingClassifier',
                             'DecisionTreeClassifier',
                             'GradientBoostingRegressor'])
MAX_CONCURRENT_PROCESSES = int(os.getenv('SKLL_MAX_CONCURRENT_PROCESSES', '5'))


def _predict_binary(self, X):
    '''
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
    '''

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
        '''
        Returns an indication of which features to keep.
        Adapted from SelectKBest.
        '''
        mask = np.zeros(self.scores_.shape, dtype=bool)
        mask[self.scores_ >= self.min_count] = True
        return mask


class RescaledRegressionMixin(BaseEstimator):
    '''
    Mixin to create regressors that store a min and a max for the training data
    and make sure that predictions fall within that range.  It also stores the
    means and SDs of the gold standard and the predictions on the training set
    to rescale the predictions (e.g., as in e-rater).
    '''
    def rescale_fit(self, X, y=None):
        '''
        Fit a model, then store the mean, SD, max and min of the training set
        and the mean and SD of the predictions on the training set.
        '''

        # fit a regular regression model
        super(self.__class__, self).fit(X, y=y)

        if self.constrain:
            # also record the training data min and max
            self.y_min = min(y)
            self.y_max = max(y)

        if self.rescale:
            # also record the means and SDs for the training set
            y_hat = super(self.__class__, self).predict(X)
            self.yhat_mean = np.mean(y_hat)
            self.yhat_sd = np.std(y_hat)
            self.y_mean = np.mean(y)
            self.y_sd = np.std(y)

    def rescale_predict(self, X):
        '''
        Make predictions with the super class, and then adjust them using the
        stored min, max, means, and standard deviations.
        '''
        # get the unconstrained predictions
        res = super(self.__class__, self).predict(X)

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

    def rescale_get_param_names(self):
        '''
        This is adapted from scikit-learns's BaseEstimator class.
        It gets the kwargs for the superclass's init method and adds the
        kwargs for the rescale_init method.
        '''
        try:
            init = getattr(super(self.__class__, self).__init__,
                           'deprecated_original',
                           super(self.__class__, self).__init__)

            args, varargs, __, __ = inspect.getargspec(init)
            if not varargs is None:
                raise RuntimeError('scikit-learn estimators should always '
                                   'specify their parameters in the signature'
                                   ' of their init (no varargs).')
            # Remove 'self'
            args.pop(0)
        except TypeError:
            args = []

        rescale_args, __, __, __ = inspect.getargspec(self.rescale_init)
        # Remove 'self'
        rescale_args.pop(0)

        args += rescale_args
        args.sort()
        return args

    def rescale_init(self, constrain=True, rescale=True, **kwargs):
        self.constrain = constrain
        self.rescale = rescale
        self.y_min = None
        self.y_max = None
        self.yhat_mean = None
        self.yhat_sd = None
        self.y_mean = None
        self.y_sd = None
        super(self.__class__, self).__init__(**kwargs)


class RescaledRidge(Ridge, RescaledRegressionMixin):
    def __init__(self, constrain=True, rescale=True, **kwargs):
        self.rescale_init(constrain=constrain, rescale=rescale, **kwargs)

    def _get_param_names(self):
        return self.rescale_get_param_names()

    def fit(self, X, y=None):
        self.rescale_fit(X, y=y)

    def predict(self, X):
        return self.rescale_predict(X)


class RescaledSVR(SVR, RescaledRegressionMixin):
    def __init__(self, constrain=True, rescale=True, **kwargs):
        self.rescale_init(constrain=constrain, rescale=rescale, **kwargs)

    def _get_param_names(self):
        return self.rescale_get_param_names()

    def fit(self, X, y=None):
        self.rescale_fit(X, y=y)

    def predict(self, X):
        return self.rescale_predict(X)


class Learner(object):
    """
    A simpler learner interface around many scikit-learn classification
    and regression functions.

    :param do_scale_features: Should we scale features with this
                              learner?
    :type do_scale_features: bool
    :param model_type: Type of estimator to create. Options are:
                       'LogisticRegression', 'LinearSVC', 'SVC',
                       'MultinomialNB', 'DecisionTreeClassifier',
                       'RandomForestClassifier',
                       'GradientBoostingClassifier', 'Ridge', 'RescaledRidge',
                       'SVR', 'RescaledSVR', and 'GradientBoostingRegressor'
    :type model_type: str
    :param probability: Should learner return probabilities of all
                        classes (instead of just class with highest
                        probability)?
    :type probability: bool
    :param model_kwargs: A dictionary of keyword arguments to pass to the
                         initializer for the specified model.
    :type model_kwargs: dict
    :param pos_label_str: The string for the positive class in the binary
                          classification setting.  Otherwise, an arbitrary
                          class is picked.
    :type pos_label_str: str
    :param use_dense_features: Whether to require conversion to dense
                               feature matrices.
    :type use_dense_features: bool
    :param min_feature_count: The minimum number of examples a feature
                              must have a nonzero value in to be included.
    :type min_feature_count: int
    """

    def __init__(self, probability=False, do_scale_features=False,
                 model_type='LogisticRegression', model_kwargs=None,
                 pos_label_str=None, use_dense_features=False,
                 min_feature_count=1):
        '''
        Initializes a learner object with the specified settings.
        '''
        super(Learner, self).__init__()

        self.feat_vectorizer = None
        self.do_scale_features = do_scale_features
        self.scaler = None
        self.label_dict = None
        self.label_list = None
        self.pos_label_str = pos_label_str
        self._model_type = model_type
        self.probability = probability
        self._model = None
        self._use_dense_features = use_dense_features
        self.feat_selector = None
        self._min_feature_count = min_feature_count
        self._model_kwargs = {}

        self._use_dense_features = (self._model_type in _REQUIRES_DENSE or
                                    self._use_dense_features)

        # Set default keyword arguments for models that we have some for.
        if self._model_type == 'SVC':
            self._model_kwargs['cache_size'] = 1000
            self._model_kwargs['probability'] = self.probability
        elif self._model_type == 'DecisionTreeClassifier':
            self._model_kwargs['criterion'] = 'entropy'
        elif self._model_type in {'RandomForestClassifier',
                                  'GradientBoostingClassifier',
                                  'GradientBoostingRegressor'}:
            self._model_kwargs['n_estimators'] = 500

        if self._model_type in {'RandomForestClassifier',
                                'DecisionTreeClassifier'}:
            self._model_kwargs['compute_importances'] = True

        if self._model_type in {'RandomForestClassifier', 'LinearSVC',
                                'LogisticRegression', 'DecisionTreeClassifier',
                                'GradientBoostingClassifier',
                                'GradientBoostingRegressor'}:
            self._model_kwargs['random_state'] = 123456789

        if self._model_type in {'SVR', 'RescaledSVR'}:
            self._model_kwargs['kernel'] = b'linear'

        if model_kwargs:
            self._model_kwargs.update(model_kwargs)

    @classmethod
    def from_file(cls, learner_path):
        '''
        :returns: New instance of Learner from the pickle at the specified
                  path.
        '''
        with open(learner_path, "rb") as f:
            skll_version, learner = pickle.load(f)
            # Check that versions are compatible. (Currently, this just checks
            # that major versions match)
            if skll_version[0] == VERSION[0]:
                return learner
            else:
                raise Exception(("Learner stored in pickle file {} was " +
                                 "created with version {} of SKLL, which is " +
                                 "incompatible with the current version " +
                                 "{}").format(learner_path,
                                              '.'.join(skll_version),
                                              '.'.join(VERSION)))

    @property
    def model_type(self):
        ''' A string representation of the underlying modeltype '''
        return self._model_type

    @property
    def model_kwargs(self):
        '''
        A dictionary of the underlying scikit-learn model's keyword arguments
        '''
        return self._model_kwargs

    @property
    def model(self):
        ''' The underlying scikit-learn model '''
        return self._model

    def load(self, learner_path):
        '''
        Replace the current learner instance with a saved learner.

        :param learner_path: The path to the file to load.
        :type learner_path: str
        '''
        del self.__dict__
        self.__dict__ = Learner.from_file(learner_path).__dict__

    @property
    def model_params(self):
        '''
        Model parameters (i.e., weights) for Ridge regression and
        liblinear models.
        '''
        res = {}
        if isinstance(self._model, Ridge):
            # also includes RescaledRidge
            coef = self.feat_selector.inverse_transform(self.model.coef_)[0]
            for feat, idx in iteritems(self.feat_vectorizer.vocabulary_):
                if coef[idx]:
                    res[feat] = coef[idx]
        elif isinstance(self._model, BaseLibLinear):

            label_list = self.label_list

            # if there are only two classes, scikit-learn will only have one
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
        else:
            # not supported
            raise ValueError(("{} is not supported by" +
                              " model_params.").format(self._model_type))

        return res

    @property
    def probability(self):
        '''
        Should learner return probabilities of all classes (instead of just
        class with highest probability)?
        '''
        return self._probability

    @probability.setter
    def probability(self, value):
        # LinearSVC doesn't support predict_proba
        self._probability = value and self.model_type != 'LinearSVC'

    def save(self, learner_path):
        '''
        Save the learner to a file.

        :param learner_path: The path to where you want to save the learner.
        :type learner_path: str
        '''
        # create the directory if it doesn't exist
        learner_dir = os.path.dirname(learner_path)
        if not os.path.exists(learner_dir):
            os.makedirs(learner_dir)
        # write out the files
        with open(learner_path, "wb") as f:
            pickle.dump((VERSION, self), f, -1)

    def _create_estimator(self):
        '''
        :returns: A tuple containing an instantiation of the requested
        estimator, and a parameter grid to search.
        '''
        estimator = None
        default_param_grid = None
        if self._model_type in _DEFAULT_PARAM_GRIDS:
            # This crazy looking line creates an estimator based on a string
            estimator = globals()[self._model_type](**self._model_kwargs)
            default_param_grid = _DEFAULT_PARAM_GRIDS[self._model_type]
        else:
            raise ValueError(("{} is not a valid learner " +
                              "type.").format(self._model_type))

        return estimator, default_param_grid

    def _check_input(self, examples):
        '''
        check that the examples are properly formatted.
        '''

        # Make sure the labels for a regression task are not strings.
        if self._model_type in _REGRESSION_MODELS:
            for label in examples.classes:
                if isinstance(label, string_types):
                    raise TypeError("You are doing regression with" +
                                    " string labels.  Convert them to" +
                                    " integers or floats.")

        max_feat_abs = float("-inf")

        # make sure that feature values are not strings
        for val in examples.features.data:
            max_feat_abs = max(max_feat_abs, abs(val))
            if isinstance(val, string_types):
                raise TypeError("You have feature values that are" +
                                " strings.  Convert them to floats.")

        if max_feat_abs > 1000.0:
            logging.warning(("You have a feature with a very large absolute " +
                             "value ({}).  That may cause the learning " +
                             "algorithm to crash or perform " +
                             "poorly.").format(max_feat_abs))

    def _train_setup(self, examples):
        '''
        Set up the feature vectorizer, the scaler and the label dict and
        return the features and the labels.

        :param examples: The examples to use for training.
        :type examples: ExamplesTuple
        '''
        # Check feature values and labels
        self._check_input(examples)

        # Create label_dict if we weren't passed one
        if self._model_type not in _REGRESSION_MODELS:

            # extract list of unique labels if we are doing classification
            self.label_list = np.unique(examples.classes).tolist()

            # if one label is specified as the positive class, make sure it's
            # last
            if self.pos_label_str:
                self.label_list = sorted(self.label_list,
                                         key=lambda x: (x == self.pos_label_str,
                                                        x))

            # Given a list of all labels in the dataset and a list of the
            # unique labels in the set, convert the first list to an array of
            # numbers.
            self.label_dict = {}
            for i, label in enumerate(self.label_list):
                self.label_dict[label] = i

        # Create feature name -> value mapping
        self.feat_vectorizer = examples.feat_vectorizer

        # initialize feature selector
        self.feat_selector = SelectByMinCount(min_count=self._min_feature_count)

        # Create scaler if we weren't passed one and it's necessary
        if self._model_type != 'MultinomialNB':
            if self.do_scale_features:
                self.scaler = StandardScaler(copy=True,
                                             with_mean=self._use_dense_features,
                                             with_std=True)
            else:
                # Doing this is to prevent any modification of feature values
                # using a dummy transformation
                self.scaler = StandardScaler(copy=False,
                                             with_mean=False,
                                             with_std=False)

    def train(self, examples, param_grid=None, grid_search_folds=5,
              grid_search=True, grid_objective='f1_score_micro',
              grid_jobs=None, shuffle=True):
        '''
        Train a classification model and return the model, score, feature
        vectorizer, scaler, label dictionary, and inverse label dictionary.

        :param examples: The examples to train the model on.
        :type examples: ExamplesTuple
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
                          grid search. If unspecified or None, the number of
                          grid search folds will be used.
        :type grid_jobs: int
        :param shuffle: Shuffle examples (e.g., for grid search CV.)
        :type shuffle: bool

        :return: The best grid search objective function score, or 0 if we're
                 not doing grid search.
        :rtype: float
        '''
        # seed the random number generator so that randomized algorithms are
        # replicable
        rand_seed = 123456789
        np.random.seed(rand_seed)

        # Shuffle so that the folds are random for the inner grid search CV.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if shuffle:
            ids, classes, features = sk_shuffle(examples.ids, examples.classes,
                                                examples.features,
                                                random_state=rand_seed)
            examples = ExamplesTuple(ids, classes, features,
                                     examples.feat_vectorizer)

        # call train setup to set up the vectorizer, the labeldict, and the
        # scaler
        self._train_setup(examples)

        # set up grid search folds
        if isinstance(grid_search_folds, int):
            grid_jobs = grid_search_folds
            folds = grid_search_folds
        else:
            # use the number of unique fold IDs as the number of grid jobs
            grid_jobs = len(np.unique(grid_search_folds))
            labels = [grid_search_folds[curr_id] for curr_id in examples.ids]
            folds = LeaveOneLabelOut(labels)

        # limit the number of grid_jobs to be no higher than five or
        # the number of cores for the machine, whichever is lower
        grid_jobs = min(grid_jobs, cpu_count(), MAX_CONCURRENT_PROCESSES)

        # select features
        xtrain = self.feat_selector.fit_transform(examples.features)

        # Convert to dense if necessary
        if self._use_dense_features:
            xtrain = xtrain.todense()

        # Scale features if necessary
        if self._model_type != 'MultinomialNB':
            xtrain = self.scaler.fit_transform(xtrain)

        # set up a grid searcher if we are asked to
        estimator, default_param_grid = self._create_estimator()

        # use label dict transformed version of examples.classes if doing
        # classification
        if self._model_type not in _REGRESSION_MODELS:
            classes = np.array([self.label_dict[label] for label in
                                examples.classes])
        else:
            classes = examples.classes

        if grid_search:
            if not param_grid:
                param_grid = default_param_grid

            if (grid_objective in _CORRELATION_METRICS and
                    self._model_type not in _REGRESSION_MODELS):
                estimator.predict_normal = estimator.predict
                estimator.predict = _predict_binary

            grid_searcher = GridSearchCV(estimator, param_grid,
                                         scoring=grid_objective, cv=folds,
                                         n_jobs=grid_jobs,
                                         pre_dispatch=grid_jobs)

            # run the grid search for hyperparameters
            grid_searcher.fit(xtrain, classes)
            self._model = grid_searcher.best_estimator_
            grid_score = grid_searcher.best_score_
        else:
            self._model = estimator.fit(xtrain, classes)
            grid_score = 0.0

        return grid_score

    def evaluate(self, examples, prediction_prefix=None, append=False,
                 grid_objective=None):
        '''
        Evaluates a given model on a given dev or test example set.

        :param examples: The examples to evaluate the performance of the model
                         on.
        :type examples: ExamplesTuple
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

        :return: The confusion matrix, the overall accuracy, the per-class
                 PRFs, the model parameters, and the grid search objective
                 function score.
        :rtype: 5-tuple
        '''
        # initialize grid score
        grid_score = None

        # make the prediction on the test data
        yhat = self.predict(examples, prediction_prefix=prediction_prefix,
                            append=append)

        # extract actual labels (transformed for classification tasks)
        if self._model_type not in _REGRESSION_MODELS:
            ytest = np.array([self.label_dict[label] for label in
                              examples.classes])
        else:
            ytest = examples.classes

        # if run in probability mode, convert yhat to list of classes predicted
        if self.probability:
            # if we're using a correlation grid objective, calculate it here
            if (grid_objective and grid_objective in _CORRELATION_METRICS):
                grid_score = SCORERS[grid_objective]._score_func(ytest, yhat[:, 1])
            yhat = np.array([max(range(len(row)),
                                 key=lambda i: row[i])
                             for row in yhat])

        # calculate grid search objective function score, if specified
        if (grid_objective and (grid_objective not in _CORRELATION_METRICS or
                                not self.probability)):
            grid_score = SCORERS[grid_objective]._score_func(ytest, yhat)

        if self._model_type in _REGRESSION_MODELS:
            result_dict = {'descriptive': defaultdict(dict), 'comparative': {}}
            for table_label, y in zip(['actual', 'predicted'], [ytest, yhat]):
                result_dict['descriptive'][table_label]['min'] = min(y)
                result_dict['descriptive'][table_label]['max'] = max(y)
                result_dict['descriptive'][table_label]['avg'] = np.mean(y)
                result_dict['descriptive'][table_label]['std'] = np.std(y)
            result_dict['comparative']['pearson'] = SCORERS['pearson']._score_func(ytest, yhat)
            res = (None, None, result_dict, self._model.get_params(), grid_score)
        else:
            # compute the confusion matrix
            num_labels = len(self.label_list)
            conf_mat = confusion_matrix(ytest, yhat,
                                        labels=list(range(num_labels)))
            # Calculate metrics
            overall_accuracy = accuracy_score(ytest, yhat)
            result_matrix = precision_recall_fscore_support(ytest,
                                                            yhat,
                                                            labels=list(range(num_labels)),
                                                            average=None)

            # Store results
            result_dict = defaultdict(dict)
            for actual_class in sorted(self.label_list):
                c_num = self.label_dict[actual_class]
                result_dict[actual_class]["Precision"] = result_matrix[0][c_num]
                result_dict[actual_class]["Recall"] = result_matrix[1][c_num]
                result_dict[actual_class]["F-measure"] = result_matrix[2][c_num]

            res = (conf_mat.tolist(), overall_accuracy, result_dict,
                   self._model.get_params(), grid_score)
        return res

    def predict(self, examples, prediction_prefix=None, append=False,
                class_labels=False):
        '''
        Uses a given model to generate predictions on a given data set

        :param examples: The examples to predict the classes for.
        :type examples: ExamplesTuple
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
        '''
        example_ids = examples.ids

        # Need to do some transformations so the features are in the right
        # columns for the test set. Obviously a bit hacky, but storing things
        # in sparse matrices saves memory over our old list of dicts approach.
        if self.feat_vectorizer == examples.feat_vectorizer:
            xtest = examples.features
        else:
            xtest = self.feat_vectorizer.transform(examples.feat_vectorizer.inverse_transform(examples.features))

        # filter features based on those selected from training set
        xtest = self.feat_selector.transform(xtest)

        # Convert to dense if necessary
        if self._use_dense_features:
            xtest = xtest.todense()

        # Scale xtest if necessary
        if self._model_type != 'MultinomialNB':
            xtest = self.scaler.transform(xtest)

        # make the prediction on the test data
        try:
            yhat = (self._model.predict_proba(xtest)
                    if (self.probability and
                        not class_labels)
                    else self._model.predict(xtest))
        except NotImplementedError as e:
            logging.error(("Model type: {}\nModel: {}\nProbability: " +
                           "{}\n").format(self._model_type, self._model,
                                          self.probability))
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
                    if self._model_type in _REGRESSION_MODELS:
                        for example_id, pred in zip(example_ids, yhat):
                            print('{0}\t{1}'.format(example_id, pred),
                                  file=predictionfh)
                    else:
                        for example_id, pred in zip(example_ids, yhat):
                            print('{0}\t{1}'.format(example_id,
                                                    self.label_list[int(pred)]),
                                  file=predictionfh)

        if class_labels and self._model_type not in _REGRESSION_MODELS:
            yhat = np.array([self.label_list[int(pred)] for pred in yhat])

        return yhat

    def cross_validate(self, examples, stratified=True, cv_folds=10,
                       grid_search=False, grid_search_folds=5, grid_jobs=None,
                       grid_objective='f1_score_micro', prediction_prefix=None,
                       param_grid=None, shuffle=True):
        '''
        Cross-validates a given model on the training examples.

        :param examples: The data to cross-validate learner performance on.
        :type examples: ExamplesTuple
        :param stratified: Should we stratify the folds to ensure an even
                           distribution of classes for each fold?
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
                          grid search. If unspecified or None, the number of
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

        :return: The confusion matrix, overall accuracy, per-class PRFs, and
                 model parameters for each fold.
        :rtype: list of 4-tuples
        '''
        # seed the random number generator so that randomized algorithms are
        # replicable
        rand_seed = 123456789
        np.random.seed(rand_seed)

        # Shuffle so that the folds are random for CV.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if shuffle:
            ids, classes, features = sk_shuffle(examples.ids, examples.classes,
                                                examples.features,
                                                random_state=rand_seed)
            examples = ExamplesTuple(ids, classes, features,
                                     examples.feat_vectorizer)

        # call train setup
        self._train_setup(examples)

        # setup the cross-validation iterator
        if isinstance(cv_folds, int):
            stratified = (stratified and
                          not self._model_type in _REGRESSION_MODELS)
            kfold = (StratifiedKFold(examples.classes, n_folds=cv_folds) if
                     stratified else KFold(len(examples.classes),
                                           n_folds=cv_folds))
        else:
            # if we have a mapping from IDs to folds, use it for the overall
            # cross-validation as well as the grid search within each
            # training fold.  Note that this means that the grid search
            # will use K-1 folds because the Kth will be the test fold for
            # the outer cross-validation.
            labels = [cv_folds[curr_id] for curr_id in examples.ids]
            kfold = LeaveOneLabelOut(labels)
            grid_search_folds = cv_folds

        # handle each fold separately and accumulate the predictions and the
        # numbers
        results = []
        grid_search_scores = []
        append_predictions = False
        for train_index, test_index in kfold:
            # Train model
            self._model = None  # prevent feature vectorizer from being reset.
            train_tuple = ExamplesTuple(ids=examples.ids[train_index],
                                        classes=examples.classes[train_index],
                                        features=examples.features[train_index],
                                        feat_vectorizer=examples.feat_vectorizer)
            grid_search_scores.append(self.train(train_tuple,
                                                 grid_search_folds=grid_search_folds,
                                                 grid_search=grid_search,
                                                 grid_objective=grid_objective,
                                                 param_grid=param_grid,
                                                 grid_jobs=grid_jobs,
                                                 shuffle=False))
            # note: there is no need to shuffle again within each fold,
            # regardless of what the shuffle keyword argument is set to.

            # Evaluate model
            test_tuple = ExamplesTuple(ids=examples.ids[test_index],
                                       classes=examples.classes[test_index],
                                       features=examples.features[test_index],
                                       feat_vectorizer=examples.feat_vectorizer)
            results.append(self.evaluate(test_tuple,
                                         prediction_prefix=prediction_prefix,
                                         append=append_predictions,
                                         grid_objective=grid_objective))

            append_predictions = True

        # return list of results for all folds
        return results, grid_search_scores
