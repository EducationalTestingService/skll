#!/usr/bin/env python2.7
'''
Module with many functions to use for easily creating an sklearn classifier

@author: Dan Blanchard (based on code from Nitin Madnani and Michael Heilman)
'''

from __future__ import print_function, unicode_literals

import csv
import json
import os
import cPickle as pickle
import subprocess
import sys
from collections import defaultdict
from itertools import islice, izip

import numpy as np
from bs4 import UnicodeDammit
from nltk.metrics import precision, recall, f_measure
from scipy.sparse import issparse
from sklearn import metrics
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Scaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


#### METRICS ####
def f1_score_macro(y_true, y_pred):
    '''
    Use the macro-averaged F1 measure to select hyperparameter values during the cross-validation grid search during training.
    '''
    return metrics.f1_score(y_true, y_pred, average="macro")


def f1_score_micro(y_true, y_pred):
    '''
    Use the micro-averaged F1 measure to select hyperparameter values during the cross-validation grid search during training.
    '''
    return metrics.f1_score(y_true, y_pred, average="micro")


def accuracy(y_true, y_pred):
    '''
    Use the overall accuracy to select hyperparameter values during the cross-validation grid search during training.
    '''
    return metrics.zero_one_score(y_true, y_pred)


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
            if not line.startswith('#') and line not in ['TRAIN', 'TEST', 'DEV']:
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
            for row in reader:
                example = _preprocess_example(row, header)
                out.append(example)
    elif path.endswith(".jsonlines"):
        out = []
        with open(path) as f:
            for line in f:
                example = json.loads(line.strip())
                out.append(example)
    elif path.endswith(".megam"):
        out = [{"y": class_name, "x": feature_dict} for class_name, feature_dict in _megam_dict_iter(path)]
    else:
        raise Exception('Example files must be in either TSV, MegaM, or the preprocessed .jsonlines format. You specified: {}'.format(path))

    return np.array(out)


def _preprocess_example(example, feature_names=None):
    '''
    Make a dictionary of preprocessed values (e.g., tokens, POS tags, etc.).
    This should be separate from the feature extraction code so that slow preprocessing steps
    can be saved and reused, without have to redo preprocessing whenever features change.
    The simple classifier parses a TSV row and returns a dictionary {"y": classlabel, "x": dictionary_of_feature_values}
    It also takes in an optional list of feature names to be used in the "x" dictionary.
    '''
    x = {}
    y = example[0]
    if feature_names:
        for fname, fval in izip(islice(feature_names, 1, None), islice(example, 1, None)):
            x["{}".format(fname)] = float(fval)
    else:
        for i, fval in enumerate(example):
            x["x{}".format(i)] = float(fval)
    return {"y": y, "x": x}


class Classifier(object):
    """ A simpler wrapper around many sklearn classification functions. """

    def __init__(self, probability=False, feat_vectorizer=None, scaler=None, label_dict=None, inverse_label_dict=None, model_type='logistic'):
        '''
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
        '''
        super(Classifier, self).__init__()
        self.probability = probability
        self.feat_vectorizer = feat_vectorizer
        self.scaler = scaler
        self.label_dict = label_dict
        self.inverse_label_dict = inverse_label_dict
        self.model_type = model_type
        self.model = None

    def load_model(self, modelfile):
        '''
        Load a saved model.

        @param modelfile: The path to the model file to load.
        @type modelfile: C{basestring}
        '''
        with open(modelfile) as f:
            self.model = pickle.load(f)

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
            pickle.dump(self.model, f, -1)

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

    def _create_estimator(self):
        '''
        @param model_type: Type of estimator to create.
                           Options are: 'logistic', 'svm_linear', 'svm_radial', 'naivebayes', 'dtree', 'rforest', and 'gradient'
        @type model_type: C{basestring}

        @return: A tuple containing an instantiation of the requested estimator, and a parameter grid to search.
        '''
        estimator = None
        default_param_grid = None

        if self.model_type == 'logistic':
            estimator = LogisticRegression()
            default_param_grid = [{'C': [1e-4, 1e-2, 1.0, 1e2, 1e4]}]
        elif self.model_type == 'svm_linear':  # No predict_proba support
            estimator = LinearSVC()
            default_param_grid = [{'C': [0.1, 1.0, 10, 100, 1000]}]
        elif self.model_type == 'svm_radial':
            estimator = SVC(cache_size=1000, probability=self.probability)
            default_param_grid = [{'C': [0.1, 1.0, 10, 100, 1000]}]
        elif self.model_type == 'naivebayes':
            estimator = MultinomialNB()
            default_param_grid = [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}]
        elif self.model_type == 'dtree':
            estimator = DecisionTreeClassifier(criterion='entropy')
            default_param_grid = [{'max_features': ["auto", None]}]
        elif self.model_type == 'rforest':
            estimator = RandomForestClassifier(n_estimators=100)
            default_param_grid = [{'max_features': ["sqrt", "log2", None]}]
        elif self.model_type == "gradient":
            estimator = GradientBoostingClassifier(n_estimators=100)
            default_param_grid = [{'learn_rate': [0.01, 0.1, 0.5]}]

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

    def train(self, examples, clear_vocab=False, param_grid_file=None, grid_search_folds=5, grid_search=True, grid_objective=f1_score_micro):
        '''
        Train a classificatiion model and return the model, score, feature vectorizer, scaler, label dictionary, and inverse label dictionary.

        @param examples: The examples to train the model on.
        @type examples: C{array}
        @param clear_vocab: Wipe out the feature vectorizer, scaler, label dictionary, and inverse label dictionary. This should be done if you're retraining
                            a L{Classifier} on a completely different data set (with different features).
        @type clear_vocab: C{bool}
        @param param_grid_file: The path to a parameter grid file containing the parameters to search through for grid search.
        @type param_grid_file: C{basestring}
        @param grid_search_folds: The number of folds to use when doing the grid search.
        @type grid_search_folds: C{int}
        @param grid_search: Should we do grid search?
        @type grid_search: C{bool}
        @param grid_objective: The objective functino to use when doing the grid search.
        @type grid_objective: C{function}

        @return: The best grid search objective function score, or 0 if we're not doing grid search.
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

            # convert labels to numbers if we are doing classification
            ytrain, self.label_dict, self.inverse_label_dict = self._convert_labels_to_array(labels, label_list)
        else:
            ytrain = np.array([self.label_dict[self._extract_label(x)] for x in examples])

        # Create feat_vectorizer if we weren't passed one
        if clear_vocab or self.feat_vectorizer is None:
            self.feat_vectorizer = self._extract_feature_vectorizer(features)  # create feature name -> value mapping

        # vectorize the features
        xtrain = self.feat_vectorizer.transform(features)

        # Create scaler if we weren't passed one
        if (clear_vocab or self.scaler is None) and self.model_type != 'naivebayes':
            self.scaler = Scaler(with_mean=(not issparse(xtrain)))

        # Convert to dense if using naivebayes or rforest
        if self.model_type in ['naivebayes', 'rforest']:
            xtrain = xtrain.todense()

        # Scale features if necessary
        xtrain_scaled = xtrain if self.model_type == 'naivebayes' else self.scaler.fit_transform(xtrain)

        # set up a grid searcher if we are asked to
        estimator, param_grid = self._create_estimator()
        if grid_search:
            if param_grid_file:
                with open(param_grid_file) as f:
                    param_grid = json.load(f)

            # NOTE: we don't want to use multithreading for LIBLINEAR since it seems to lead to irreproducible results
            grid_searcher = GridSearchCV(estimator, param_grid, score_func=grid_objective, cv=grid_search_folds,
                                         n_jobs=(grid_search_folds if self.model_type not in ["svm_linear", "logistic"] else 1))

            # run the grid search for hyperparameters
            print('\tstarting grid search', file=sys.stderr)
            grid_searcher.fit(xtrain_scaled, ytrain)
            self.model = grid_searcher.best_estimator_
            score = grid_searcher.best_score_
        else:
            self.model = estimator.fit(xtrain_scaled, ytrain)
            score = 0.0

        return score

    def evaluate(self, examples, prediction_prefix=None):
        '''
        Evaluates a given model on a given dev or test example set.

        @param examples: The examples to evaluate the performance of the model on.
        @type examples: C{array}
        @param prediction_prefix: If saving the predictions, this is the prefix that will be used for the filename. It will be followed by "-{model_type}.predictions"
        @type prediction_prefix: C{basestring}

        @return: The confusion matrix, the overall accuracy, and the per-class PRFs.
        @rtype: 3-C{tuple}
        '''
        # make the prediction on the test data
        yhat = self.predict(examples, prediction_prefix)

        # if run in probability mode, convert yhat to list of classes predicted
        if self.probability:
            np.array([max(xrange(len(row)), key=lambda i: row[i]) for row in yhat])

        # extract actual labels
        ytest = np.array([self.label_dict[self._extract_label(x)] for x in examples])

        # Create prediction dicts for easier scoring
        actual_dict = defaultdict(set)
        pred_dict = defaultdict(set)
        pred_list = [self.inverse_label_dict[pred_class] for pred_class in yhat]
        actual_list = [self.inverse_label_dict[actual_class] for actual_class in ytest]
        for line_num, (pred_class, actual_class) in enumerate(izip(pred_list, actual_list)):
            pred_dict[pred_class].add(line_num)
            actual_dict[actual_class].add(line_num)

        # Calculate metrics
        result_dict = defaultdict(dict)
        overall_accuracy = metrics.zero_one_score(ytest, yhat) * 100
        # Store results
        for actual_class in sorted(actual_dict.iterkeys()):
            result_dict[actual_class]["Precision"] = precision(actual_dict[actual_class], pred_dict[actual_class])
            result_dict[actual_class]["Recall"] = recall(actual_dict[actual_class], pred_dict[actual_class])
            result_dict[actual_class]["F-measure"] = f_measure(actual_dict[actual_class], pred_dict[actual_class])

        return (metrics.confusion_matrix(ytest, yhat, labels=range(len(self.inverse_label_dict))).tolist(), overall_accuracy, result_dict)

    def predict(self, examples, prediction_prefix):
        '''
        Uses a given model to generate predictions on a given data set

        @param examples: The examples to predict the classes for.
        @type examples: C{array}
        @param prediction_prefix: If saving the predictions, this is the prefix that will be used for the filename. It will be followed by "-{model_type}.predictions"
        @type prediction_prefix: C{basestring}

        @return: The predictions returned by the classifier.
        '''
        features = [self._extract_features(x) for x in examples]

        # transform and scale the features
        xtest = self.feat_vectorizer.transform(features)
        xtest_scaled = xtest if self.model_type == 'naivebayes' else self.scaler.transform(xtest)

        # make the prediction on the test data
        yhat = self.model.predict_proba(xtest_scaled) if self.probability and self.model_type != 'svm_linear' else self.model.predict(xtest_scaled)

        # write out the predictions if we are asked to
        if prediction_prefix is not None:
            prediction_file = prediction_prefix + '-{}.predictions'.format(self.model_type)
            with open(prediction_file, "w") as predictionfh:
                if self.probability and self.model_type != 'svm_linear':
                    print('\t'.join(self.inverse_label_dict), file=predictionfh)
                    for class_probs in yhat:
                        print('\t'.join(str(x) for x in class_probs), file=predictionfh)
                else:
                    for pred in yhat:
                        print(self.inverse_label_dict[pred], file=predictionfh)
                print(file=predictionfh)

        return yhat

    def cross_validate(self, examples, stratified=True, clear_vocab=False, cv_folds=10, grid_search=False, grid_search_folds=5, grid_objective=f1_score_micro,
                       prediction_prefix=None):
        '''
        Cross-validates a given model on the training examples.

        @param examples: The data to cross-validate classifier performance on.
        @type examples: C{array}
        @param stratified: Should we stratifiy the folds to ensure an even distribution of classes for each fold?
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
        @param grid_objective: The objective functino to use when doing the grid search.
        @type grid_objective: C{function}

        @return: The confusion matrix, overall accuracy, and per-class PRFs for each fold.
        @rtype: C{list} of 3-{tuple}s
        '''
        features = [self._extract_features(x) for x in examples]

        # Create scaler if we weren't passed one
        if (clear_vocab or self.scaler is None) and self.model_type != 'naivebayes':
            self.scaler = Scaler()

        # Create feat_vectorizer if we weren't passed one
        if clear_vocab or self.feat_vectorizer is None:
            self.feat_vectorizer = self._extract_feature_vectorizer(features)  # create feature name -> value mapping

        # Create label_dict if we weren't passed one
        if clear_vocab or self.label_dict is None:
            labels = [self._extract_label(x) for x in examples]

            # extract list of unique labels if we are doing classification
            label_list = np.unique(labels).tolist()

            # convert labels to numbers if we are doing classification
            y, self.label_dict, self.inverse_label_dict = self._convert_labels_to_array(labels, label_list)
        else:
            y = np.array([self.label_dict[self._extract_label(x)] for x in examples])

        # setup the cross-validation iterator
        kfold = StratifiedKFold(y, k=cv_folds) if stratified else KFold(y, k=cv_folds)

        # handle each fold separately and accumulate the predictions and the numbers
        results = []
        for train_index, test_index in kfold:
            # Train model
            self.model = None  # Do this to prevent feature vectorizer from being reset every time.
            self.train(examples[train_index], grid_search_folds=grid_search_folds, grid_search=grid_search, grid_objective=grid_objective)

            # Evaluate model
            results.append(self.evaluate(examples[test_index], prediction_prefix=prediction_prefix))

        # return list of results for all folds
        return results
