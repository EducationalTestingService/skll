#!/usr/bin/env python2.7
'''
Module with many functions to use for easily creating an sklearn classifier

@author: Michael Heilman (mheilman@ets.org)
@author: Nitin Madnani (nmadnani@ets.org)
@author: Dan Blanchard (dblanchard@ets.org)
'''

from __future__ import print_function, unicode_literals

import csv
import json
import os
import cPickle as pickle
import subprocess
import sys
import time
from collections import defaultdict
from itertools import islice

import numpy as np
from bs4 import UnicodeDammit
from nltk.metrics import precision, recall, f_measure
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn import metrics
from sklearn.base import is_classifier, clone
from sklearn.cross_validation import KFold, StratifiedKFold, check_cv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV, IterGrid, _has_one_grid_point
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import safe_mask, check_arrays
from sklearn.utils.validation import _num_samples
from sklearn.feature_selection import SelectKBest
from six import iteritems
import scipy.sparse as sp
from six.moves import zip, xrange
import ml_metrics


#### Globals ####
_REQUIRES_DENSE = frozenset(['naivebayes', 'rforest', 'gradient', 'dtree'])
_CORRELATION_METRICS = frozenset(['kendall_tau', 'spearman', 'pearson'])
_REGRESSION_MODELS = frozenset(['ridge', 'rescaled_ridge', 'svr_linear', 
                                'rescaled_svr_linear'])


#### METRICS ####
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
    except ValueError, e:
        print("For kappa, the labels should be integers or strings that" +
              " can be converted to ints (E.g., '4.0' or '3').",
              file=sys.stderr)
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
    except ValueError, e:
        print("For kappa, the labels should be integers or strings that" +
              " can be converted to ints (E.g., '4.0' or '3').",
              file=sys.stderr)
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
    return metrics.f1_score(y_true[y_true == least_frequent],
                            y_pred[y_true == least_frequent])


def f1_score_macro(y_true, y_pred):
    '''
    Use the macro-averaged F1 measure to select hyperparameter values during
    the cross-validation grid search during training.

    This method averages over classes (does not take imbalance into account).
    You should use this if each class is equally important.
    '''
    return metrics.f1_score(y_true, y_pred, average="macro")


def f1_score_micro(y_true, y_pred):
    '''
    Use the micro-averaged F1 measure to select hyperparameter values during
    the cross-validation grid search during training.

    This method averages over instances (takes imbalance into account). This
    implies that precision == recall == F1.
    '''
    return metrics.f1_score(y_true, y_pred, average="micro")


def accuracy(y_true, y_pred):
    '''
    Use the overall accuracy to select hyperparameter values during the cross-
    validation grid search during training.
    '''
    return metrics.accuracy_score(y_true, y_pred)


#### DATA LOADING FUNCTIONS ###
def _sanitize_line(line):
    '''
    Return copy of line with all non-ASCII characters replaced with
    <U1234> sequences where 1234 is the value of ord() for the character.
    '''
    char_list = []
    for char in line:
        char_num = ord(char)
        char_list.append('<U{}>'.format(char_num) if char_num > 127 else char)
    return ''.join(char_list)


def _megam_dict_iter(path, has_labels=True):
    '''
    Generator that yields tuples of classes and dictionaries mapping from
    features to values for each pair of lines in path

    @param path: Path to MegaM file
    @type path: C{basestring}
    '''

    line_count = 0
    print(
        "Loading {}...".format(path).encode('utf-8'), end="", file=sys.stderr)
    sys.stderr.flush()
    with open(path) as megam_file:
        curr_id = None
        for line in megam_file:
            # Process encoding
            line = _sanitize_line(UnicodeDammit(
                line, ['utf-8', 'windows-1252']).unicode_markup.strip())
            # Handle instance lines
            if line.startswith('#'):
                curr_id = line[1:].strip()
            elif line and line not in ['TRAIN', 'TEST', 'DEV']:
                split_line = line.split()
                curr_info_dict = {}

                if has_labels:
                    class_name = split_line[0]
                    field_pairs = split_line[1:]
                else:
                    class_name = None
                    field_pairs = split_line

                if len(field_pairs) > 0:
                    # Get current instances feature-value pairs
                    field_names = islice(field_pairs, 0, None, 2)
                    # Convert values to floats, because otherwise features'll
                    # be categorical
                    field_values = (float(val) for val in islice(field_pairs,
                                                                 1, None, 2))

                    # Add the feature-value pairs to dictionary
                    curr_info_dict.update(zip(field_names, field_values))
                yield curr_id, class_name, curr_info_dict
                curr_id = None
            line_count += 1
            if line_count % 100 == 0:
                print(".", end="", file=sys.stderr)
        print("done", file=sys.stderr)


def load_examples(path, has_labels=True):
    '''
    Loads examples in the TSV, JSONLINES (a json dict per line), or MegaM
    formats.

    @param path: The path to the file to load the examples from.
    @type path: C{basestring}

    @return: 2-column C{numpy.array} of examples with the "y" containing the
    class labels and "x" containing the features for each example.
    '''
    if path.endswith(".tsv"):
        out = []
        with open(path) as f:
            reader = csv.reader(f, dialect=csv.excel_tab)
            header = next(reader)
            out = [_preprocess_tsv_row(row, header, example_num, has_labels=has_labels)
                   for example_num, row in enumerate(reader)]
    elif path.endswith(".jsonlines"):
        out = []
        with open(path) as f:
            example_num = 0
            for line in f:
                example = json.loads(line.strip())
                if "id" not in example:
                    example["id"] = "EXAMPLE_{}".format(example_num)
                example_num += 1
                out.append(example)
    elif path.endswith(".megam"):
        out = [{"y": class_name, "x": feature_dict,
                "id": "EXAMPLE_{}".format(example_num) if example_id is None
                else example_id}
               for example_num, (example_id, class_name, feature_dict)
               in enumerate(_megam_dict_iter(path, has_labels=has_labels))]
    else:
        raise Exception('Example files must be in either TSV, MegaM, or the \
                         preprocessed .jsonlines format. \
                         You specified: {}'.format(path))

    return np.array(out)


def _preprocess_tsv_row(row, header, example_num, has_labels=True):
    '''
    Make a dictionary of preprocessed values (e.g., tokens, POS tags, etc.).
    This should be separate from the feature extraction code so that slow
    preprocessing steps can be saved and reused, without have to redo
    preprocessing whenever features change. This parses a TSV row and returns
    a dictionary {"y": class label, "x": dictionary of feature values} It also
    takes in an optional list of feature names to be used in the "x"
    dictionary.
    '''
    x = {}

    if has_labels:
        y = row[0]
        feature_start_col = 1
    else:
        y = None
        feature_start_col = 0

    example_id = "EXAMPLE_{}".format(example_num)
    for fname, fval in zip(islice(header, feature_start_col, None), 
                           islice(row, feature_start_col, None)):
        if fname == "id":
            example_id = fval
        else:
            x["{}".format(fname)] = float(fval)

    return {"y": y, "x": x, "id": example_id}


def _fit_grid_point(X, y, base_clf, clf_params, train, test, loss_func,
                    score_func, verbose, **fit_params):
    """
    Run fit on one set of parameters

    Returns the score and the instance of the classifier
    """
    if verbose > 1:
        start_time = time.time()
        msg = '%s' % (', '.join('%s=%s' % (k, v)
                                for k, v in iteritems(clf_params)))
        print("[GridSearchCV] %s %s" % (msg, (64 - len(msg)) * '.'))
    # update parameters of the classifier after a copy of its base structure
    clf = clone(base_clf)
    clf.set_params(**clf_params)

    if hasattr(base_clf, 'kernel') and hasattr(base_clf.kernel, '__call__'):
        # cannot compute the kernel values with custom function
        raise ValueError("Cannot use a custom kernel function. "
                         "Precompute the kernel matrix instead.")

    if not hasattr(X, "shape"):
        if getattr(base_clf, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have " +
                             "to be passed as arrays or sparse matrices.")
        X_train = [X[idx] for idx in train]
        X_test = [X[idx] for idx in test]
    else:
        if getattr(base_clf, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            X_train = X[np.ix_(train, train)]
            X_test = X[np.ix_(test, train)]
        else:
            X_train = X[safe_mask(X, train)]
            X_test = X[safe_mask(X, test)]

    if y is not None:
        y_test = y[safe_mask(y, test)]
        y_train = y[safe_mask(y, train)]
        clf.fit(X_train, y_train, **fit_params)
        if loss_func is not None:
            # Everything is the same as the original version except next line
            y_pred = clf.predict_proba(X_test)[:, 1]
            this_score = -loss_func(y_test, y_pred)
        elif score_func is not None:
            y_pred = clf.predict_proba(X_test)[:, 1]  # and this one
            this_score = score_func(y_test, y_pred)
        else:
            this_score = clf.score(X_test, y_test)
    else:
        clf.fit(X_train, **fit_params)
        this_score = clf.score(X_test)

    if verbose > 2:
        msg += ", score=%f" % this_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg,
                              logger.short_format_time(time.time() -
                                                       start_time))
        print("[GridSearchCV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))
    return this_score, clf_params, _num_samples(X)


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
            _, col_indices, _ = sp.find(X.copy())
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


# Temporary imports for _FixedStandardScaler
from sklearn.preprocessing import _mean_and_std
from sklearn.utils import warn_if_not_float
from sklearn.utils.sparsefuncs import inplace_csr_column_scale
from sklearn.utils.sparsefuncs import mean_variance_axis0


class _FixedStandardScaler(StandardScaler):

    '''
    StandardScaler has a bug in that it always scales by the standard
    deviation for sparse matrices, i.e., it ignores the value of with_std.
    This is a fixed version. This is just temporary until the bug is fixed in
    sklearn.
    '''

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        X = check_arrays(X, copy=self.copy, sparse_format="csr")[0]
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            warn_if_not_float(X, estimator=self)
            self.mean_ = None

            # we added this check for with_std
            if self.with_std:
                var = mean_variance_axis0(X)[1]
                self.std_ = np.sqrt(var)
                self.std_[var == 0.0] = 1.0
            else:
                self.std_ = None

            return self
        else:
            warn_if_not_float(X, estimator=self)
            self.mean_, self.std_ = _mean_and_std(
                X, axis=0, with_mean=self.with_mean, with_std=self.with_std)
            return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        X = check_arrays(X, copy=copy, sparse_format="csr")[0]
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            warn_if_not_float(X, estimator=self)
            if self.with_std:
                inplace_csr_column_scale(X, 1 / self.std_)
        else:
            warn_if_not_float(X, estimator=self)
            if self.with_mean:
                X -= self.mean_
            # we added this check for with_std
            if self.with_std:
                X /= self.std_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sp.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            # we added this check for with_std
            if self.with_std:
                inplace_csr_column_scale(X, self.std_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.std_
            if self.with_mean:
                X += self.mean_
        return X


class _GridSearchCVBinary(GridSearchCV):
    '''
    GridSearchCV for use with binary classification problems where you want to
    optimize the learner based on the probabilities assigned to each class,
    and not just the predicted class.
    '''

    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters

        Returns the best classifier

        Parameters
        ----------

        X: array, [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y: array-like, shape = [n_samples], optional
            Target vector relative to X for classification;
            None for unsupervised learning.

        """
        estimator = self.estimator
        cv = self.cv

        X, y = check_arrays(X, y, sparse_format="csr", allow_lists=True)
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        grid = IterGrid(self.param_grid)
        base_clf = clone(self.estimator)

        # Return early if there is only one grid point.
        if _has_one_grid_point(self.param_grid):
            params = next(iter(grid))
            base_clf.set_params(**params)
            if y is not None:
                base_clf.fit(X, y)
            else:
                base_clf.fit(X)
            self.best_estimator_ = base_clf
            self._set_methods()
            return self

        pre_dispatch = self.pre_dispatch
        out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                       pre_dispatch=pre_dispatch)(
                           delayed(_fit_grid_point)(
                               X, y, base_clf, clf_params, train, test,
                               self.loss_func, self.score_func, self.verbose,
                               **self.fit_params)
                           for clf_params in grid for train, test in cv)

        # Out is a list of triplet: score, estimator, n_test_samples
        n_grid_points = len(list(grid))
        n_fits = len(out)
        n_folds = n_fits // n_grid_points

        scores = list()
        cv_scores = list()
        for grid_start in xrange(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            these_points = list()
            for (this_score, clf_params,
                    this_n_test_samples) in out[grid_start:grid_start + n_folds]:
                these_points.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                score += this_score
                n_test_samples += this_n_test_samples
            if self.iid:
                score /= float(n_test_samples)
            scores.append((score, clf_params))
            cv_scores.append(these_points)

        cv_scores = np.asarray(cv_scores)

        # Note: we do not use max(out) to make ties deterministic even if
        # comparison on estimator instances is not deterministic
        best_score = -np.inf
        for score, params in scores:
            if score > best_score:
                best_score = score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_clf).set_params(**best_params)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
            self._set_methods()

        # Store the computed scores
        self.grid_scores_ = [(clf_params, score, all_scores) for clf_params, (
            score, _), all_scores in zip(grid, scores, cv_scores)]
        return self

    def score(self, X, y=None):
        if hasattr(self.best_estimator_, 'score'):
            return self.best_estimator_.score(X, y)
        if self.score_func is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        y_predicted = self.predict_proba(X)[:, 1]
        return self.score_func(y, y_predicted)


class RescaledRidge(Ridge):
    '''
    This is an extension of the Ridge classifier that stores a min and 
    a max for the training data.  It makes sure its predictions fall within
    that range.
    '''
    def __init__(self, rescale=True, constrain=True, alpha=1.0, 
                 fit_intercept=True, normalize=False, copy_X=True, 
                 max_iter=None, tol=1e-3, solver="auto"):
        self.constrain = constrain
        self.rescale = rescale
        self.y_min = None
        self.y_max = None
        self.yhat_mean = None
        self.yhat_sd = None
        self.y_mean = None
        self.y_sd = None
        super(RescaledRidge, self).__init__(alpha=alpha, 
                                            fit_intercept=fit_intercept,
                                            normalize=normalize, 
                                            copy_X=copy_X,
                                            max_iter=max_iter, 
                                            tol=tol, solver=solver)

    def fit(self, X, y=None):
        # fit a regular regression model
        super(RescaledRidge, self).fit(X, y=y)

        if self.constrain:
            # also record the training data min and max
            self.y_min = min(y)
            self.y_max = max(y)

        if self.rescale:
            # also record the means and SDs for the training set
            y_hat = super(RescaledRidge, self).predict(X)
            self.yhat_mean = np.mean(y_hat)
            self.yhat_sd = np.std(y_hat)
            self.y_mean = np.mean(y)
            self.y_sd = np.std(y)

    def predict(self, X):
        # get the unconstrained predictions
        res = super(RescaledRidge, self).predict(X)

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


class RescaledSVR(SVR):
    '''
    This is an extension of the SVR classifier that stores a min and 
    a max for the training data.  It makes sure its predictions fall within
    that range.
    '''
    def __init__(self, rescale=True, constrain=True, kernel='rbf', degree=3, 
        gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.10000000000000001, 
        shrinking=True, probability=False, cache_size=200, verbose=False, 
        max_iter=-1):
        self.constrain = constrain
        self.rescale = rescale
        self.y_min = None
        self.y_max = None
        self.yhat_mean = None
        self.yhat_sd = None
        self.y_mean = None
        self.y_sd = None
        super(RescaledSVR, self).__init__(kernel=kernel, degree=degree, 
              gamma=gamma, coef0=coef0, tol=tol, C=C, epsilon=epsilon, 
              shrinking=shrinking, probability=probability, 
              cache_size=cache_size, verbose=verbose, max_iter=max_iter)

    def fit(self, X, y=None):
        # fit a regular regression model
        super(RescaledSVR, self).fit(X, y=y)

        if self.constrain:
            # also record the training data min and max
            self.y_min = min(y)
            self.y_max = max(y)

        if self.rescale:
            # also record the means and SDs for the training set
            y_hat = super(RescaledSVR, self).predict(X)
            self.yhat_mean = np.mean(y_hat)
            self.yhat_sd = np.std(y_hat)
            self.y_mean = np.mean(y)
            self.y_sd = np.std(y)

    def predict(self, X):
        # get the unconstrained predictions
        res = super(RescaledSVR, self).predict(X)

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


class Classifier(object):
    """
    A simpler classifier interface around many sklearn classification
    functions.
    """

    def __init__(self, probability=False, do_scale_features=False,
                 model_type='logistic', model_kwargs=None, pos_label_str=None,
                 use_dense_features=False, min_feature_count=1):
        '''
        Initializes a classifier object with the specified settings.

        @param do_scale_features: Should we scale features with this
                                  classifier?
        @type do_scale_features: C{bool}
        @param model_type: Type of estimator to create. Options are:
                           'logistic', 'svm_linear', 'svm_radial',
                           'naivebayes', 'dtree', 'rforest', and 'gradient'
        @type model_type: C{basestring}
        @param probability: Should classifier return probabilities of all
                            classes (instead of just class with highest
                            probability)?
        @type probability: C{bool}
        @param model_kwargs: A dictionary of keyword arguments to pass to the
                             initializer for the specified model.
        @type model_kwargs: C{dict}
        @param pos_label_str: The string for the positive class in the binary
                              classification setting.  Otherwise, an arbitrary
                              class is picked.
        @type pos_label_str: C{str}
        @param use_dense_features: Whether to require conversion to dense
                                   feature matrices.
        @type use_dense_features: C{bool}
        @param min_feature_count: The minimum number of examples a feature
                                  must have a nonzero value in to be included.
        @type min_feature_count: C{int}
        '''
        super(Classifier, self).__init__()
        self.probability = probability if model_type != 'svm_linear' else False
        self.feat_vectorizer = None
        self.do_scale_features = do_scale_features
        self.scaler = None
        self.label_dict = None
        self.label_list = None
        self.pos_label_str = pos_label_str
        self._model_type = model_type
        self._model = None
        self._use_dense_features = use_dense_features
        self.feat_selector = None
        self._min_feature_count = min_feature_count
        self._model_kwargs = {}

        self._use_dense_features = (self._model_type in _REQUIRES_DENSE or
                                    self._use_dense_features)

        # Set default keyword arguments for models that we have some for.
        if self._model_type == 'svm_radial':
            self._model_kwargs['cache_size'] = 1000
            self._model_kwargs['probability'] = self.probability
        elif self._model_type == 'dtree':
            self._model_kwargs['criterion'] = 'entropy'
        elif self._model_type == 'rforest' or self._model_type == 'gradient':
            self._model_kwargs['n_estimators'] = 1000
        if self._model_type == 'rforest' or self._model_type == 'dtree':
            self._model_kwargs['compute_importances'] = True
        if model_kwargs:
            self._model_kwargs.update(model_kwargs)

    @property
    def model_type(self):
        ''' Getter for model type '''
        return self._model_type

    @property
    def model_kwargs(self):
        ''' Getter for model keyword arguments '''
        return self._model_kwargs

    @property
    def model(self):
        ''' Getter for underlying model '''
        return self._model

    def load_model(self, modelfile):
        '''
        Load a saved model.

        @param modelfile: The path to the model file to load.
        @type modelfile: C{basestring}
        '''
        with open(modelfile, "rb") as f:
            (self._model, self.probability, self.feat_vectorizer,
             self.feat_selector, self.scaler, self.label_dict,
             self.label_list) = pickle.load(f)
        if isinstance(self._model, LogisticRegression):
            self._model_type = 'logistic'
        elif isinstance(self._model, LinearSVC):
            self._model_type = 'svm_linear'
        elif isinstance(self._model, SVC):
            self._model_type = 'svm_radial'
        elif isinstance(self._model, MultinomialNB):
            self._model_type = 'naivebayes'
        elif isinstance(self._model, DecisionTreeClassifier):
            self._model_type = 'dtree'
        elif isinstance(self._model, RandomForestClassifier):
            self._model_type = 'rforest'
        elif isinstance(self._model, GradientBoostingClassifier):
            self._model_type = 'gradient'
        elif isinstance(self._model, Ridge):
            self._model_type = 'ridge'
        elif isinstance(self._model, RescaledRidge):
            self._model_type = 'rescaled_ridge'
        elif isinstance(self._model, SVR):
            self._model_type = 'svr_linear'
        elif isinstance(self._model, RescaledSVR):
            self._model_type = 'rescaled_svr_linear'

    def save_model(self, modelfile):
        '''
        Save the model to file.

        @param modelfile: The path to where you want to save the model.
        @type modelfile: C{basestring}
        '''
        # create the directory if it doesn't exist
        modeldir = os.path.dirname(modelfile)
        if not os.path.exists(modeldir):
            subprocess.call("mkdir -p {}".format(modeldir), shell=True)
        # write out the files
        with open(modelfile, "wb") as f:
            pickle.dump([self._model, self.probability,
                         self.feat_vectorizer, self.feat_selector,
                         self.scaler, self.label_dict, self.label_list],
                        f, -1)

    @staticmethod
    def _extract_features(example):
        '''
        Return a dictionary of feature values extracted from a preprocessed
        example. This base method expects all the features to be of the form
        "x1", "x2", etc.
        '''
        return example["x"]

    def _extract_label(self, example):
        '''
        Return the label for a preprocessed example.
        '''
        if self._model_type in _REGRESSION_MODELS:
            return float(example["y"])
        else:
            return self.label_dict[example["y"]]

    @staticmethod
    def _extract_id(example):
        '''
        Return the string ID for a preprocessed example.
        '''
        return example["id"]

    def _create_estimator(self):
        '''
        @return: A tuple containing an instantiation of the requested
        estimator, and a parameter grid to search.
        '''
        estimator = None
        default_param_grid = None

        if self._model_type == 'logistic':
            estimator = LogisticRegression(**self._model_kwargs)
            default_param_grid = [{'C': [1e-4, 1e-2, 1.0, 1e2, 1e4]}]
        elif self._model_type == 'svm_linear':  # No predict_proba support
            estimator = LinearSVC(**self._model_kwargs)
            default_param_grid = [{'C': [0.1, 1.0, 10, 100, 1000]}]
        elif self._model_type == 'svm_radial':
            estimator = SVC(**self._model_kwargs)
            default_param_grid = [{'C': [0.1, 1.0, 10, 100, 1000]}]
        elif self._model_type == 'naivebayes':
            estimator = MultinomialNB(**self._model_kwargs)
            default_param_grid = [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}]
        elif self._model_type == 'dtree':
            estimator = DecisionTreeClassifier(**self._model_kwargs)
            default_param_grid = [{'max_features': ["auto", None]}]
        elif self._model_type == 'rforest':
            estimator = RandomForestClassifier(**self._model_kwargs)
            default_param_grid = [{'max_depth': [1, 5, 10, None]}]
        elif self._model_type == "gradient":
            estimator = GradientBoostingClassifier(**self._model_kwargs)
            default_param_grid = [{'learning_rate': [0.01, 0.1, 0.5]}]
        elif self._model_type == 'ridge':
            estimator = Ridge(**self._model_kwargs)
            default_param_grid = [{'alpha': [0.1, 1.0, 10, 100, 1000]}]
        elif self._model_type == 'rescaled_ridge':
            estimator = RescaledRidge(**self._model_kwargs)
            default_param_grid = [{'alpha': [0.1, 1.0, 10, 100, 1000]}]
        elif self._model_type == 'svr_linear':
            estimator = SVR(kernel='linear', **self._model_kwargs)
            default_param_grid = [{'C': [0.1, 1.0, 10, 100, 1000]}]
        elif self._model_type == 'rescaled_svr_linear':
            estimator = RescaledSVR(kernel='linear', **self._model_kwargs)
            default_param_grid = [{'C': [0.2, 2.0, 20, 200, 2000]}]
        else:
            raise ValueError(
                "{} is not a valid classifier type.".format(self._model_type))

        return estimator, default_param_grid

    def train_setup(self, examples):
        '''
        Set up the feature vectorizer, the scaler and the label dict and
        return the features and the labels
        '''

        # extract the features and the labels
        features = [self._extract_features(x) for x in examples]

        # Create label_dict if we weren't passed one
        if self._model_type not in _REGRESSION_MODELS:

            # extract list of unique labels if we are doing classification
            self.label_list = np.unique(
                [example["y"] for example in examples]).tolist()

            # if one label is specified as the positive class, make sure it's
            # last
            if self.pos_label_str:
                self.label_list = sorted(
                    self.label_list,
                    key=lambda x: (x == self.pos_label_str, x))

            # Given a list of all labels in the dataset and a list of the
            # unique labels in the set, convert the first list to an array of
            # numbers.
            self.label_dict = {}
            for i, label in enumerate(self.label_list):
                self.label_dict[label] = i

        y = np.array([self._extract_label(x) for x in examples])

        # Create feature name -> value mapping
        self.feat_vectorizer = DictVectorizer(
            sparse=not self._use_dense_features)

        # initialize feature selector
        self.feat_selector = SelectByMinCount(
            min_count=self._min_feature_count)

        # Create scaler if we weren't passed one
        if self._model_type != 'naivebayes':
            if self.do_scale_features:
                self.scaler = _FixedStandardScaler(
                    copy=True, with_mean=self._use_dense_features,
                    with_std=True)
            else:
                # Doing this is to prevent any modification of feature values
                # using a dummy transformation
                self.scaler = _FixedStandardScaler(
                    copy=False, with_mean=False, with_std=False)

        return features, y

    def train(self, examples, param_grid=None,
              grid_search_folds=5, grid_search=True,
              grid_objective=f1_score_micro):
        '''
        Train a classification model and return the model, score, feature
        vectorizer, scaler, label dictionary, and inverse label dictionary.

        @param examples: The examples to train the model on.
        @type examples: C{array}
        @param param_grid: The parameter grid to search through for grid
                           search. If unspecified, a default parameter grid
                           will be used.
        @type param_grid: C{list} of C{dict}s mapping from C{basestring}s to
                          C{list}s of parameter values
        @param grid_search_folds: The number of folds to use when doing the
                                  grid search.
        @type grid_search_folds: C{int}
        @param grid_search: Should we do grid search?
        @type grid_search: C{bool}
        @param grid_objective: The objective function to use when doing the
                               grid search.
        @type grid_objective: C{function}

        @return: The best grid search objective function score, or 0 if we're
                 not doing grid search.
        @rtype: C{float}
        '''

        # seed the random number generator so that randomized algorithms are
        # replicable
        np.random.seed(9876315986142)

        # call train setup to set up the vectorizer, the labeldict, and the
        # scaler
        features, ytrain = self.train_setup(examples)

        # vectorize the features
        xtrain = self.feat_vectorizer.fit_transform(features)

        # select features
        xtrain = self.feat_selector.fit_transform(xtrain)

        # Scale features if necessary
        if self._model_type != 'naivebayes':
            xtrain = self.scaler.fit_transform(xtrain)

        # set up a grid searcher if we are asked to
        estimator, default_param_grid = self._create_estimator()

        if grid_search:
            if not param_grid:
                param_grid = default_param_grid

            if (grid_objective.__name__ in _CORRELATION_METRICS and
                    self._model_type not in _REGRESSION_MODELS):
                grid_search_class = _GridSearchCVBinary
            else:
                grid_search_class = GridSearchCV
            grid_searcher = grid_search_class(estimator, param_grid,
                                              score_func=grid_objective,
                                              cv=grid_search_folds,
                                              n_jobs=grid_search_folds)

            # run the grid search for hyperparameters
            # print('\tstarting grid search', file=sys.stderr)
            grid_searcher.fit(xtrain, ytrain)
            self._model = grid_searcher.best_estimator_
            score = grid_searcher.best_score_
        else:
            self._model = estimator.fit(xtrain, ytrain)
            score = 0.0

        return score

    def evaluate(self, examples, prediction_prefix=None, append=False,
                 grid_objective=None):
        '''
        Evaluates a given model on a given dev or test example set.

        @param examples: The examples to evaluate the performance of the model
            on.
        @type examples: C{array}
        @param prediction_prefix: If saving the predictions, this is the
                                  prefix that will be used for the filename.
                                  It will be followed by ".predictions"
        @type prediction_prefix: C{basestring}
        @param append: Should we append the current predictions to the file if
                       it exists?
        @type append: C{bool}
        @param grid_objective: The objective function that was used when doing
                               the grid search.
        @type grid_objective: C{function}

        @return: The confusion matrix, the overall accuracy, the per-class
                 PRFs, the model parameters, and the grid search objective
                 function score.
        @rtype: 3-C{tuple}
        '''
        # initialize grid score
        grid_score = None

        # make the prediction on the test data
        yhat = self.predict(examples, prediction_prefix=prediction_prefix, append=append)

        # extract actual labels
        ytest = np.array([self._extract_label(x) for x in examples])

        # if run in probability mode, convert yhat to list of classes predicted
        if self.probability:
            # if we're using a correlation grid objective, calculate it here
            if (grid_objective is not None and
                    grid_objective.__name__ in _CORRELATION_METRICS):
                grid_score = grid_objective(ytest, yhat[:, 1])
            yhat = np.array(
                [max(xrange(len(row)), key=lambda i: row[i]) for row in yhat])

        # calculate grid search objective function score, if specified
        if (grid_objective is not None and
                (grid_objective.__name__ not in _CORRELATION_METRICS or
                 not self.probability)):
            grid_score = grid_objective(ytest, yhat)

        if self._model_type in _REGRESSION_MODELS:
            res = (None, None, None, self._model.get_params(), grid_score)
        else:
            # Create prediction dicts for easier scoring
            actual_dict = defaultdict(set)
            pred_dict = defaultdict(set)
            pred_list = [self.label_list[int(pred_class)]
                         for pred_class in yhat]
            actual_list = [self.label_list[int(actual_class)]
                           for actual_class in ytest]
            for (line_num,
                 (pred_class,
                  actual_class)) in enumerate(zip(pred_list,
                                                  actual_list)):
                pred_dict[pred_class].add(line_num)
                actual_dict[actual_class].add(line_num)

            overall_accuracy = metrics.accuracy_score(ytest, yhat) * 100

            # Calculate metrics
            result_dict = defaultdict(dict)

            # Store results
            for actual_class in sorted(self.label_list):
                result_dict[actual_class]["Precision"] = precision(
                    actual_dict[actual_class], pred_dict[actual_class])
                result_dict[actual_class]["Recall"] = recall(
                    actual_dict[actual_class], pred_dict[actual_class])
                result_dict[actual_class]["F-measure"] = f_measure(
                    actual_dict[actual_class], pred_dict[actual_class])
            num_labels = len(self.label_list)
            conf_mat = metrics.confusion_matrix(ytest, yhat,
                                                labels=list(range(num_labels))
                                                ).tolist()
            res = (conf_mat, overall_accuracy, result_dict,
                   self._model.get_params(), grid_score)
        return res

    def predict(self, examples, prediction_prefix=None, append=False,
                class_labels=False):
        '''
        Uses a given model to generate predictions on a given data set

        @param examples: The examples to predict the classes for.
        @type examples: C{array}
        @param prediction_prefix: If saving the predictions, this is the
                                  prefix that will be used for the
                                  filename. It will be followed by
                                  ".predictions"
        @type prediction_prefix: C{basestring}
        @param append: Should we append the current predictions to the file if
                       it exists?
        @type append: C{bool}
        @param class_labels: For classifier, should we convert class 
                             indices to their (str) labels?
        @type class_labels: C{bool}                   

        @return: The predictions returned by the classifier.
        @rtype: C{array}
        '''
        features = [self._extract_features(x) for x in examples]
        example_ids = [self._extract_id(x) for x in examples]

        # transform the features
        xtest = self.feat_vectorizer.transform(features)

        # filter features based on those selected from training set
        xtest = self.feat_selector.transform(xtest)

        # Scale xtest
        if self._model_type != 'naivebayes':
            xtest = self.scaler.transform(xtest)

        # make the prediction on the test data
        try:
            yhat = (self._model.predict_proba(xtest)
                    if (self.probability and
                        self._model_type != 'svm_linear'
                        and not class_labels)
                    else self._model.predict(xtest))
        except NotImplementedError as e:
            print(("Model type: {}\nModel: {}\nProbability: " +
                   "{}\n").format(self._model_type, self._model,
                                  self.probability), file=sys.stderr)
            raise e

        # write out the predictions if we are asked to
        if prediction_prefix is not None:
            prediction_file = '{}.predictions'.format(prediction_prefix)
            with open(prediction_file,
                      "w" if not append else "a") as predictionfh:
                # header
                if not append:
                    if self.probability and self._model_type != 'svm_linear':
                        print('\t'.join(
                            ["id"] + self.label_list), file=predictionfh)
                    else:
                        print('\t'.join(
                            ["id", "prediction"]), file=predictionfh)

                if self.probability and self._model_type != 'svm_linear':
                    for example_id, class_probs in zip(example_ids, yhat):
                        print('\t'.join([example_id] + [str(
                            x) for x in class_probs]), file=predictionfh)
                else:
                    if self._model_type in _REGRESSION_MODELS:
                        for example_id, pred in zip(example_ids, yhat):
                            print('\t'.join(
                                [example_id, str(pred)]), file=predictionfh)
                    else:
                        for example_id, pred in zip(example_ids, yhat):
                            print('\t'.join([example_id,
                                  self.label_list[int(pred)]]),
                                  file=predictionfh)

        if class_labels and self._model_type not in _REGRESSION_MODELS:
            yhat = np.array([self.label_list[int(pred)] for pred in yhat])

        return yhat

    def cross_validate(self, examples, stratified=True,
                       cv_folds=10, grid_search=False, grid_search_folds=5,
                       grid_objective=f1_score_micro, prediction_prefix=None,
                       param_grid=None, shuffle=True):
        '''
        Cross-validates a given model on the training examples.

        @param examples: The data to cross-validate classifier performance on.
        @type examples: C{array}
        @param stratified: Should we stratify the folds to ensure an even
                           distribution of classes for each fold?
        @type stratified: C{bool}
        @param cv_folds: The number of folds to use for cross-validation.
        @type cv_folds: C{int}
        @param grid_search: Should we do grid search when training each fold?
                            Note: This will make this take *much* longer.
        @type grid_search: C{bool}
        @param grid_search_folds: The number of folds to use when doing the
                                  grid search.
        @type grid_search_folds: C{int}
        @param grid_objective: The objective function to use when doing the
                               grid search.
        @type grid_objective: C{function}
        @param param_grid: The parameter grid to search through for grid
                           search. If unspecified, a default parameter
                           grid will be used.
        @type param_grid: C{list} of C{dict}s mapping from C{basestring}s to
                          C{list}s of parameter values
        @param prediction_prefix: If saving the predictions, this is the
                                  prefix that will be used for the filename.
                                  It will be followed by ".predictions"
        @type prediction_prefix: C{basestring}
        @param shuffle: Shuffle examples before splitting into folds for CV.
        @type shuffle: C{bool}

        @return: The confusion matrix, overall accuracy, per-class PRFs, and
                 model parameters for each fold.
        @rtype: C{list} of 4-C{tuple}s
        '''
        # seed the random number generator so that randomized folds are
        # replicable
        np.random.seed(9876315986142)

        # Shuffle examples before splitting
        if shuffle:
            np.random.shuffle(examples)

        # call train setup
        _, y = self.train_setup(examples)

        # setup the cross-validation iterator
        stratified = stratified and not self._model_type in _REGRESSION_MODELS
        kfold = StratifiedKFold(y, n_folds=cv_folds) if stratified else KFold(
            len(examples), n_folds=cv_folds)

        # handle each fold separately and accumulate the predictions and the
        # numbers
        results = []
        append_predictions = False
        for train_index, test_index in kfold:
            # Train model
            self._model = None  # prevent feature vectorizer from being reset.
            self.train(
                examples[train_index], grid_search_folds=grid_search_folds,
                grid_search=grid_search, grid_objective=grid_objective,
                param_grid=param_grid)

            # Evaluate model
            results.append(
                self.evaluate(
                    examples[test_index], prediction_prefix=prediction_prefix,
                    append=append_predictions, grid_objective=grid_objective))

            append_predictions = True

        # return list of results for all folds
        return results
