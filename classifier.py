#!/usr/bin/env python2.7
'''
Module with many functions to use for easily creating an sklearn classifier

@author: Nitin Madnani (nmadnani@ets.org)
@author: Dan Blanchard (dblanchard@ets.org)
@author Michael Heilman (mheilman@ets.org)
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
from itertools import islice, izip

import numpy as np
from bs4 import UnicodeDammit
from nltk.metrics import precision, recall, f_measure
from scipy.sparse import issparse
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn import metrics
from sklearn.base import is_classifier, clone
from sklearn.cross_validation import KFold, StratifiedKFold, check_cv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV, IterGrid, _has_one_grid_point
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import safe_mask, check_arrays
from sklearn.utils.validation import _num_samples


#### Globals ####
_REQUIRES_DENSE = frozenset(['naivebayes', 'rforest', 'gradient', 'dtree'])
_CORRELATION_METRICS = frozenset(['kendall_tau', 'spearman', 'pearson'])


#### METRICS ####
def kendall_tau(y_true, y_pred):
    '''
    Optimize the hyperparameter values during the grid search based on Kendall's tau.

    This is useful in cases where you want to use the actual probabilities of the different classes after the fact, and not just the optimize based on the classification accuracy.
    '''
    ret_score = kendalltau(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def spearman(y_true, y_pred):
    '''
    Optimize the hyperparameter values during the grid search based on Spearman rank correlation.

    This is useful in cases where you want to use the actual probabilities of the different classes after the fact, and not just the optimize based on the classification accuracy.
    '''
    ret_score = spearmanr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def pearson(y_true, y_pred):
    '''
    Optimize the hyperparameter values during the grid search based on Pearson correlation.
    '''
    ret_score = pearsonr(y_true, y_pred)
    return ret_score if not np.isnan(ret_score) else 0.0


def f1_score_least_frequent(y_true, y_pred):
    '''
    Optimize the hyperparameter values during the grid search based on the F1 measure of the least frequent class.

    This is mostly intended for use when you're doing binary classification and your data is highly skewed. You should probably use f1_score_macro if your data
    is skewed and you're doing multi-class classification.
    '''
    least_frequent = np.bincount(y_true).argmin()
    return metrics.f1_score(y_true[y_true == least_frequent], y_pred[y_true == least_frequent])


def f1_score_macro(y_true, y_pred):
    '''
    Use the macro-averaged F1 measure to select hyperparameter values during the cross-validation grid search during training.

    This method averages over classes (does not take imbalance into account). You should use this if each class is equally important.
    '''
    return metrics.f1_score(y_true, y_pred, average="macro")


def f1_score_micro(y_true, y_pred):
    '''
    Use the micro-averaged F1 measure to select hyperparameter values during the cross-validation grid search during training.

    This method averages over instances (takes imbalance into account). This implies that precision == recall == F1.
    '''
    return metrics.f1_score(y_true, y_pred, average="micro")


def accuracy(y_true, y_pred):
    '''
    Use the overall accuracy to select hyperparameter values during the cross-validation grid search during training.
    '''
    return metrics.accuracy_score(y_true, y_pred)


#### DATA LOADING FUNCTIONS ###
def _sanitize_line(line):
    ''' Return copy of line with all non-ASCII characters replaced with <U1234> sequences where 1234 is the value of ord() for the character. '''
    char_list = []
    for char in line:
        char_num = ord(char)
        char_list.append('<U{}>'.format(char_num) if char_num > 127 else char)
    return ''.join(char_list)


def _megam_dict_iter(path):
    '''
    Generator that yields tuples of classes and dictionaries mapping from features to values for each pair of lines in path

    @param path: Path to MegaM file
    @type path: C{basestring}
    '''

    line_count = 0
    print("Loading {}...".format(path).encode('utf-8'), end="", file=sys.stderr)
    sys.stderr.flush()
    with open(path) as megam_file:
        for line in megam_file:
            # Process encoding
            line = _sanitize_line(UnicodeDammit(line, ['utf-8', 'windows-1252']).unicode_markup.strip())
            # Handle instance lines
            if line and not line.startswith('#') and line not in ['TRAIN', 'TEST', 'DEV']:
                split_line = line.split()
                class_name = split_line[0]
                curr_info_dict = {}
                if len(split_line) > 1:
                    # Get current instances feature-value pairs
                    field_pairs = split_line[1:]
                    field_names = islice(field_pairs, 0, None, 2)
                    field_values = islice(field_pairs, 1, None, 2)

                    # Add the feature-value pairs to dictionary
                    curr_info_dict.update(izip(field_names, field_values))
                yield class_name, curr_info_dict
            line_count += 1
            if line_count % 100 == 0:
                print(".", end="", file=sys.stderr)
        print("done", file=sys.stderr)


def load_examples(path):
    '''
    Loads examples in the TSV, JSONLINES (a json dict per line), or MegaM formats.

    @param path: The path to the file to load the examples from.
    @type path: C{basestring}

    @return: 2-column C{numpy.array} of examples with the "y" containing the class labels and "x" containing the features for each example.
    '''
    if path.endswith(".tsv"):
        out = []
        with open(path) as f:
            reader = csv.reader(f, dialect=csv.excel_tab)
            header = reader.next()
            out = [_preprocess_tsv_row(row, header, example_num) for example_num, row in enumerate(reader)]
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
        # TODO read in example ids from comments
        out = [{"y": class_name, "x": feature_dict, "id": "EXAMPLE_{}".format(example_num)} for example_num, class_name, feature_dict in enumerate(_megam_dict_iter(path))]
    else:
        raise Exception('Example files must be in either TSV, MegaM, or the preprocessed .jsonlines format. You specified: {}'.format(path))

    return np.array(out)


def _preprocess_tsv_row(row, header, example_num):
    '''
    Make a dictionary of preprocessed values (e.g., tokens, POS tags, etc.).
    This should be separate from the feature extraction code so that slow preprocessing steps
    can be saved and reused, without have to redo preprocessing whenever features change.
    This parses a TSV row and returns a dictionary {"y": class label, "x": dictionary of feature values}
    It also takes in an optional list of feature names to be used in the "x" dictionary.
    '''
    x = {}
    y = row[0]
    example_id = "EXAMPLE_{}".format(example_num)
    for fname, fval in izip(islice(header, 1, None), islice(row, 1, None)):
        if fname == "id":
            example_id = fval
        else:
            x["{}".format(fname)] = float(fval)

    return {"y": y, "x": x, "id": example_id}


def _fit_grid_point(X, y, base_clf, clf_params, train, test, loss_func, score_func, verbose, **fit_params):
    """
    Run fit on one set of parameters

    Returns the score and the instance of the classifier
    """
    if verbose > 1:
        start_time = time.time()
        msg = '%s' % (', '.join('%s=%s' % (k, v)
                                for k, v in clf_params.iteritems()))
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
            raise ValueError("Precomputed kernels or affinity matrices have "
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
            y_pred = clf.predict_proba(X_test)[:, 1]  # Everything is the same as the original version except this line...
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


class _GridSearchCVBinary(GridSearchCV):
    '''
    GridSearchCV for use with binary classification problems where you want to optimize the learner based on the probabilities assigned to each class, and not just
    the predicted class.
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
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            these_points = list()
            for this_score, clf_params, this_n_test_samples in out[grid_start:grid_start + n_folds]:
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
        self.grid_scores_ = [(clf_params, score, all_scores) for clf_params, (score, _), all_scores in zip(grid, scores, cv_scores)]
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


class Classifier(object):
    """ A simpler classifier interface around many sklearn classification functions. """

    def __init__(self, probability=False, feat_vectorizer=None, scaler=None, label_dict=None, inverse_label_dict=None, model_type='logistic', model_kwargs=None, pos_label_str=None):
        '''
        Initializes a classifier object with the specified settings.

        @param feat_vectorizer: A C{DictVectorizer} that transforms lists of feature-value mappings to vectors.
        @type feat_vectorizer: C{DictVectorizer}
        @param scaler: A pre-fit scaler for the data that this classifier will be processing.
        @type scaler: C{Scaler}
        @param label_dict: Maps from class/label names to integers.
        @type label_dict: C{dict}
        @param inverse_label_dict: Maps from integers back to label strings.
        @type inverse_label_dict: C{list} of C{basestring}
        @param model_type: Type of estimator to create.
                           Options are: 'logistic', 'svm_linear', 'svm_radial', 'naivebayes', 'dtree', 'rforest', and 'gradient'
        @type model_type: C{basestring}
        @param probability: Should classifier return probabilities of all classes (instead of just class with highest probability)?
        @type probability: C{bool}
        @param model_kwargs: A dictionary of keyword arguments to pass to the initializer for the specified model.
        @type model_kwargs: C{dict}
        '''
        super(Classifier, self).__init__()
        self.probability = probability if model_type != 'svm_linear' else False
        self.feat_vectorizer = feat_vectorizer
        self.scaler = scaler
        self.label_dict = label_dict
        self.inverse_label_dict = inverse_label_dict
        self.pos_label_str = pos_label_str
        self._model_type = model_type
        self._model = None
        self._model_kwargs = {}

        # Set default keyword arguments for models that we have some for.
        if self._model_type == 'svm_radial':
            self._model_kwargs['cache_size'] = 1000
            self._model_kwargs['probability'] = self.probability
        elif self._model_type == 'dtree':
            self._model_kwargs['criterion'] = 'entropy'
        elif self._model_type == 'rforest' or self._model_type == 'gradient':
            self._model_kwargs['n_estimators'] = 1000

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
        with open(modelfile) as f:
            self._model, self.probability = pickle.load(f)

    def load_vocab(self, vocabfile):
        '''
        Load a saved vocab (feature vectorizer, scaler, label dictionary, and inverse label dictionary).

        @param vocabfile: The path to the vocab file to load.
        @type vocabfile: C{basestring}
        '''
        with open(vocabfile) as f:
            self.feat_vectorizer, self.scaler, self.label_dict, self.inverse_label_dict = pickle.load(f)

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
        with open(modelfile, "w") as f:
            pickle.dump([self._model, self.probability], f, -1)

    def save_vocab(self, vocabfile):
        '''
        Save vocab (feature vectorizer, scaler, label dictionary, and inverse label dictionary) to file.

        @param vocabfile: The path to where you want to save the vocab.
        @type vocabfile: C{basestring}
        '''
        # create the directory if it doesn't exist
        vocabdir = os.path.dirname(vocabfile)
        if not os.path.exists(vocabdir):
            subprocess.call("mkdir -p {}".format(vocabdir), shell=True)
        with open(vocabfile, "w") as f:
            pickle.dump([self.feat_vectorizer, self.scaler, self.label_dict, self.inverse_label_dict], f, -1)

    @staticmethod
    def _extract_features(example):
        '''
        Return a dictionary of feature values extracted from a preprocessed example.
        This base method expects all the features to be of the form "x1", "x2", etc.
        '''
        return example["x"]

    @staticmethod
    def _extract_label(example):
        '''
        Return the label for a preprocessed example.
        '''
        return example["y"]

    @staticmethod
    def _extract_id(example):
        '''
        Return the string ID for a preprocessed example.
        '''
        return example["id"]

    def _create_estimator(self):
        '''
        @return: A tuple containing an instantiation of the requested estimator, and a parameter grid to search.
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
        else:
            raise ValueError("{} is not a valid classifier type.".format(self._model_type))

        return estimator, default_param_grid

    def _extract_feature_vectorizer(self, features):
        ''' Given a dict of features, create a DictVectorizer for mapping from dicts of features to arrays of features '''
        self.feat_vectorizer = DictVectorizer()
        self.feat_vectorizer.fit(features)

    @staticmethod
    def _convert_labels_to_array(labels, label_list):
        ''' Given a list of all labels in the dataset and a list of the unique labels in the set, convert the first list to an array of numbers. '''
        label_dict = {}
            
        for i, label in enumerate(label_list):
            label_dict[label] = i

        out_array = np.array([label_dict[label] for label in labels])
        return out_array, label_dict, label_list

    def train(self, examples, clear_vocab=False, param_grid=None, grid_search_folds=5, grid_search=True, grid_objective=f1_score_micro):
        '''
        Train a classification model and return the model, score, feature vectorizer, scaler, label dictionary, and inverse label dictionary.

        @param examples: The examples to train the model on.
        @type examples: C{array}
        @param clear_vocab: Wipe out the feature vectorizer, scaler, label dictionary, and inverse label dictionary. This should be done if you're retraining
                            a L{Classifier} on a completely different data set (with different features).
        @type clear_vocab: C{bool}
        @param param_grid: The parameter grid to search through for grid search. If unspecified, a default parameter grid will be used.
        @type param_grid: C{list} of C{dict}s mapping from C{basestring}s to C{list}s of parameter values
        @param grid_search_folds: The number of folds to use when doing the grid search.
        @type grid_search_folds: C{int}
        @param grid_search: Should we do grid search?
        @type grid_search: C{bool}
        @param grid_objective: The objective function to use when doing the grid search.
        @type grid_objective: C{function}

        @return: The best grid search objective function score, or 0 if we're not doing grid search.
        @rtype: C{float}
        '''

        # seed the random number generator so that randomized algorithms are replicable
        np.random.seed(9876315986142)

        # extract the features and the labels
        features = [self._extract_features(x) for x in examples]

        # Create label_dict if we weren't passed one
        if clear_vocab or self.label_dict is None:
            labels = [self._extract_label(x) for x in examples]

            # extract list of unique labels if we are doing classification
            label_list = np.unique(labels).tolist()

            # if one label is specified as the positive class, make sure it's last
            if self.pos_label_str:
                label_list = sorted(label_list, key=lambda x: (x == self.pos_label_str, x))

            # convert labels to numbers if we are doing classification
            ytrain, self.label_dict, self.inverse_label_dict = self._convert_labels_to_array(labels, label_list)
        else:
            ytrain = np.array([self.label_dict[self._extract_label(x)] for x in examples])

        # Create feat_vectorizer if we weren't passed one
        if clear_vocab or self.feat_vectorizer is None:
            self._extract_feature_vectorizer(features)  # create feature name -> value mapping

        # vectorize the features
        xtrain = self.feat_vectorizer.transform(features)

        # Convert to dense if required by model type
        if self._model_type in _REQUIRES_DENSE:
            xtrain = xtrain.todense()

        # Create scaler if we weren't passed one
        if (clear_vocab or self.scaler is None) and self._model_type != 'naivebayes':
            self.scaler = StandardScaler(copy=True, with_mean=(not issparse(xtrain)))

        # Scale features if necessary
        if self._model_type != 'naivebayes':
            xtrain = self.scaler.fit_transform(xtrain)

        # set up a grid searcher if we are asked to
        estimator, default_param_grid = self._create_estimator()

        if grid_search:
            if not param_grid:
                param_grid = default_param_grid

            # NOTE: we don't want to use multithreading for LIBLINEAR since it seems to lead to irreproducible results
            if grid_objective.__name__ in _CORRELATION_METRICS:
                grid_searcher = _GridSearchCVBinary(estimator, param_grid, score_func=grid_objective, cv=grid_search_folds,
                                                   n_jobs=(grid_search_folds if self._model_type not in {"svm_linear", "logistic"} else 1))
            else:
                grid_searcher = GridSearchCV(estimator, param_grid, score_func=grid_objective, cv=grid_search_folds,
                                             n_jobs=(grid_search_folds if self._model_type not in {"svm_linear", "logistic"} else 1))

            # run the grid search for hyperparameters
            # print('\tstarting grid search', file=sys.stderr)
            grid_searcher.fit(xtrain, ytrain)
            self._model = grid_searcher.best_estimator_
            score = grid_searcher.best_score_
        else:
            self._model = estimator.fit(xtrain, ytrain)
            score = 0.0

        return score

    def evaluate(self, examples, prediction_prefix=None, append=False, grid_objective=None):
        '''
        Evaluates a given model on a given dev or test example set.

        @param examples: The examples to evaluate the performance of the model on.
        @type examples: C{array}
        @param prediction_prefix: If saving the predictions, this is the prefix that will be used for the filename. It will be followed by ".predictions"
        @type prediction_prefix: C{basestring}
        @param append: Should we append the current predictions to the file if it exists?
        @type append: C{bool}
        @param grid_objective: The objective function that was used when doing the grid search.
        @type grid_objective: C{function}

        @return: The confusion matrix, the overall accuracy, the per-class PRFs, the model parameters, and the grid search objective function score.
        @rtype: 3-C{tuple}
        '''
        # initialize grid score
        grid_score = None

        # make the prediction on the test data
        yhat = self.predict(examples, prediction_prefix, append=append)

        # extract actual labels
        ytest = np.array([self.label_dict[self._extract_label(x)] for x in examples])

        # if run in probability mode, convert yhat to list of classes predicted
        if self.probability:
            # if we're using a correlation grid objective, calculate it here
            if grid_objective is not None and grid_objective.__name__ in _CORRELATION_METRICS:
                grid_score = grid_objective(ytest, yhat[:, 1])
            yhat = np.array([max(xrange(len(row)), key=lambda i: row[i]) for row in yhat])

        # calculate grid search objective function score, if specified
        if grid_objective is not None and (grid_objective.__name__ not in _CORRELATION_METRICS or not self.probability):
            grid_score = grid_objective(ytest, yhat)

        # Create prediction dicts for easier scoring
        actual_dict = defaultdict(set)
        pred_dict = defaultdict(set)
        pred_list = [self.inverse_label_dict[int(pred_class)] for pred_class in yhat]
        actual_list = [self.inverse_label_dict[int(actual_class)] for actual_class in ytest]
        for line_num, (pred_class, actual_class) in enumerate(izip(pred_list, actual_list)):
            pred_dict[pred_class].add(line_num)
            actual_dict[actual_class].add(line_num)

        # Calculate metrics
        result_dict = defaultdict(dict)
        overall_accuracy = metrics.accuracy_score(ytest, yhat) * 100
        # Store results
        for actual_class in sorted(actual_dict.iterkeys()):
            result_dict[actual_class]["Precision"] = precision(actual_dict[actual_class], pred_dict[actual_class])
            result_dict[actual_class]["Recall"] = recall(actual_dict[actual_class], pred_dict[actual_class])
            result_dict[actual_class]["F-measure"] = f_measure(actual_dict[actual_class], pred_dict[actual_class])

        return (metrics.confusion_matrix(ytest, yhat, labels=range(len(self.inverse_label_dict))).tolist(), overall_accuracy, result_dict, self._model.get_params(), grid_score)

    def predict(self, examples, prediction_prefix, append=False):
        '''
        Uses a given model to generate predictions on a given data set

        @param examples: The examples to predict the classes for.
        @type examples: C{array}
        @param prediction_prefix: If saving the predictions, this is the prefix that will be used for the filename. It will be followed by ".predictions"
        @type prediction_prefix: C{basestring}
        @param append: Should we append the current predictions to the file if it exists?
        @type append: C{bool}

        @return: The predictions returned by the classifier.
        @rtype: C{array}
        '''
        features = [self._extract_features(x) for x in examples]
        example_ids = [self._extract_id(x) for x in examples]

        # transform the features
        xtest = self.feat_vectorizer.transform(features)

        # Convert to dense if required by model type
        if self._model_type in _REQUIRES_DENSE:
            xtest = xtest.todense()

        # Scale xtest
        if self._model_type != 'naivebayes':
            xtest = self.scaler.transform(xtest)

        # make the prediction on the test data
        try:
            yhat = self._model.predict_proba(xtest) if self.probability and self._model_type != 'svm_linear' else self._model.predict(xtest)
        except NotImplementedError as e:
            print("Model type: {}\nModel: {}\nProbability: {}\n".format(self._model_type, self._model, self.probability), file=sys.stderr)
            raise e

        # write out the predictions if we are asked to
        if prediction_prefix is not None:
            prediction_file = '{}.predictions'.format(prediction_prefix)
            with open(prediction_file, "w" if not append else "a") as predictionfh:
                # header
                if not append:
                    if self.probability and self._model_type != 'svm_linear':
                        print('\t'.join(["id"] + self.inverse_label_dict), file=predictionfh)
                    else:
                        print('\t'.join(["id", "prediction"]), file=predictionfh)

                if self.probability and self._model_type != 'svm_linear':
                    for example_id, class_probs in zip(example_ids, yhat):
                        print('\t'.join([example_id] + [str(x) for x in class_probs]), file=predictionfh)
                else:
                    for example_id, pred in zip(example_ids, yhat):
                        print('\t'.join([example_id, self.inverse_label_dict[int(pred)]]), file=predictionfh)

        return yhat

    def cross_validate(self, examples, stratified=True, clear_vocab=False, cv_folds=10, grid_search=False, grid_search_folds=5, grid_objective=f1_score_micro,
                       prediction_prefix=None, param_grid=None):
        '''
        Cross-validates a given model on the training examples.

        @param examples: The data to cross-validate classifier performance on.
        @type examples: C{array}
        @param stratified: Should we stratify the folds to ensure an even distribution of classes for each fold?
        @type stratified: C{bool}
        @param clear_vocab: Wipe out the feature vectorizer, scaler, label dictionary, and inverse label dictionary. This should be done if you're retraining
                            a L{Classifier} on a completely different data set (with different features).
        @type clear_vocab: C{bool}
        @param cv_folds: The number of folds to use for cross-validation.
        @type cv_folds: C{int}
        @param grid_search: Should we do grid search when training each fold? Note: This will make this take *much* longer.
        @type grid_search: C{bool}
        @param grid_search_folds: The number of folds to use when doing the grid search.
        @type grid_search_folds: C{int}
        @param grid_objective: The objective function to use when doing the grid search.
        @type grid_objective: C{function}
        @param param_grid: The parameter grid to search through for grid search. If unspecified, a default parameter grid will be used.
        @type param_grid: C{list} of C{dict}s mapping from C{basestring}s to C{list}s of parameter values
        @param prediction_prefix: If saving the predictions, this is the prefix that will be used for the filename. It will be followed by ".predictions"
        @type prediction_prefix: C{basestring}

        @return: The confusion matrix, overall accuracy, per-class PRFs, and model parameters for each fold.
        @rtype: C{list} of 4-C{tuple}s
        '''
        features = [self._extract_features(x) for x in examples]

        # Create scaler if we weren't passed one
        if (clear_vocab or self.scaler is None) and self._model_type != 'naivebayes':
            self.scaler = StandardScaler(copy=True, with_mean=self._model_type in _REQUIRES_DENSE)

        # Create feat_vectorizer if we weren't passed one
        if clear_vocab or self.feat_vectorizer is None:
            self._extract_feature_vectorizer(features)  # create feature name -> value mapping

        # Create label_dict if we weren't passed one
        if clear_vocab or self.label_dict is None or self.inverse_label_dict is None:
            labels = [self._extract_label(x) for x in examples]

            # extract list of unique labels if we are doing classification
            label_list = np.unique(labels).tolist()

            # if one label is specified as the positive class, make sure it's last
            if self.pos_label_str:
                label_list = sorted(label_list, key=lambda x: (x == self.pos_label_str, x))

            # convert labels to numbers if we are doing classification
            y, self.label_dict, self.inverse_label_dict = self._convert_labels_to_array(labels, label_list)
        else:
            y = np.array([self.label_dict[self._extract_label(x)] for x in examples])

        # setup the cross-validation iterator
        kfold = StratifiedKFold(y, n_folds=cv_folds) if stratified else KFold(y, n_folds=cv_folds)

        # handle each fold separately and accumulate the predictions and the numbers
        results = []
        append_predictions = False
        for train_index, test_index in kfold:
            # Train model
            self._model = None  # Do this to prevent feature vectorizer from being reset every time.
            self.train(examples[train_index], grid_search_folds=grid_search_folds, grid_search=grid_search, grid_objective=grid_objective, param_grid=param_grid)

            # Evaluate model
            results.append(self.evaluate(examples[test_index], prediction_prefix=prediction_prefix, append=append_predictions, grid_objective=grid_objective))

            append_predictions = True

        # return list of results for all folds
        return results
