#!/usr/bin/env python2.7
'''
Module with many functions to use for easily creating an sklearn classifier
'''

from __future__ import print_function, unicode_literals

import csv
import json
import os
import cPickle as pickle
import subprocess
import sys
from itertools import chain, islice, izip

import numpy as np
from bs4 import UnicodeDammit
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


def sanitize_line(line):
    ''' Return copy of line with all non-ASCII characters replaced with <U1234> sequences where 1234 is the value of ord() for the character. '''
    char_list = []
    for char in line:
        char_num = ord(char)
        char_list.append('<U{}>'.format(char_num) if char_num > 127 else char)
    return ''.join(char_list)


def megam_dict_iter(path):
    '''
    Generator that yields tuples of classes and dictionaries mapping from features to values for each pair of lines in path

    @param path: Path to MegaM file
    @type path: C{unicode}
    '''

    line_count = 0
    print("Loading {}...".format(path).encode('utf-8'), end="", file=sys.stderr)
    sys.stderr.flush()
    with open(path) as megam_file:
        for line in megam_file:
            # Process encoding
            line = sanitize_line(UnicodeDammit(line, ['utf-8', 'windows-1252']).unicode_markup.strip())
            # Handle instance lines
            if not line.startswith('#') and line not in ['TRAIN', 'TEST', 'DEV']:
                split_line = line.split()
                class_name = split_line[0]
                curr_info_dict = dict()
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
    '''
    if path.endswith(".tsv"):
        out = []
        with open(path) as f:
            reader = csv.reader(f, dialect=csv.excel_tab)
            header = reader.next()
            for row in reader:
                example = preprocess_example(row, header)
                out.append(example)
    elif path.endswith(".jsonlines"):
        out = []
        with open(path) as f:
            for line in f:
                example = json.loads(line.strip())
                out.append(example)
    elif path.endswith(".megam"):
        out = [{"y": class_name, "x": feature_dict} for class_name, feature_dict in megam_dict_iter(path)]
    else:
        raise Exception('Example files must be in either TSV, MegaM, or the preprocessed .jsonlines format. You specified: {}'.format(path))
    return out


def preprocess_example(example, feature_names=None):
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
        for fname, fval in izip(feature_names[1:], example[1:]):
            x["{}".format(fname)] = float(fval)
    else:
        for i, fval in enumerate(example):
            x["x{}".format(i)] = float(fval)
    return {"y": y, "x": x}


def extract_features(example):
    '''
    Return a dictionary of feature values extracted from a preprocessed example.
    This base method expects all the features to be of the form "x1", "x2", etc.
    '''
    return example["x"]


def classify(clf, feat_vectorizer, label_list, example):
    '''
    Run a list of feature values through the classification model
    to get a score (perhaps a probability).
    '''
    features = extract_features(example)
    transformed_features = feat_vectorizer.transform(features)
    return label_list[clf.predict(transformed_features)[0]]


def extract_label(example):
    '''
    Return the label for a preprocessed example.
    Note: this method should be overridden for new tasks.
    '''
    return example["y"]


def create_estimator(model_type):
    '''
    @param model_type: Type of estimator to create.
                       Options are: 'logistic', 'svm_linear', 'svm_radial', 'naivebayes', 'dtree', 'rforest', and 'gradient'
    @type model_type: C{unicode}

    @return: A tuple containing an instantiation of the requested estimator, and a parameter grid to search.
    '''
    estimator = None
    default_param_grid = None

    if model_type == 'logistic':
        estimator = LogisticRegression()
        default_param_grid = [{'C': [1e-4, 1e-2, 1.0, 1e2, 1e4]}]
    elif model_type == 'svm_linear':
        estimator = LinearSVC()
        default_param_grid = [{'C': [0.1, 1.0, 10, 100, 1000]}]
    elif model_type == 'svm_radial':
        estimator = SVC(cache_size=1000)
        default_param_grid = [{'C': [0.1, 1.0, 10, 100, 1000]}]
    elif model_type == 'naivebayes':
        estimator = MultinomialNB()
        default_param_grid = [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}]
    elif model_type == 'dtree':
        estimator = DecisionTreeClassifier(criterion='entropy')
        default_param_grid = [{'max_features': ["auto", None]}]
    elif model_type == 'rforest':
        estimator = RandomForestClassifier(n_estimators=100)
        default_param_grid = [{'max_features': ["sqrt", "log2", None]}]
    elif model_type == "gradient":
        estimator = GradientBoostingClassifier(n_estimators=100)
        default_param_grid = [{'learn_rate': [0.01, 0.1, 0.5]}]

    return estimator, default_param_grid


def extract_feature_vectorizer(features):
    ''' Given a dict of features, create a DictVectorizer for mapping from dicts of features to arrays of features '''
    vectorizer = DictVectorizer()
    vectorizer.fit(features)
    return vectorizer


def convert_labels_to_array(labels, label_list):
    ''' Given a list of all labels in the dataset and a list of the unique labels in the set, convert the first list to an array of numbers. '''
    label_dict = {}

    # we need a dictionary that stores int label to real label mapping for later prediction extraction
    inverse_label_dict = {}

    for i, label in enumerate(label_list):
        label_dict[label] = i
        inverse_label_dict[i] = label

    out_array = np.array([label_dict[label] for label in labels])
    return out_array, label_dict, inverse_label_dict


def train(examples, feat_vectorizer=None, scaler=None, label_dict=None, inverse_label_dict=None, model_type='logistic', param_grid_file=None, modelfile=None,
          vocabfile=None, cv_folds=5, grid_search=True, grid_objective=f1_score_micro):
    '''
    Train a classificatiion model and return the model, score, feature vectorizer, scaler, label dictionary, and inverse label dictionary.
    '''

    # seed the random number generator so that randomized algorithms are replicable
    np.random.seed(9876315986142)

    # extract the features and the labels
    features = [extract_features(x) for x in examples]

    # Create label_dict if we weren't passed one
    if label_dict is None:
        labels = [extract_label(x) for x in examples]

        # extract list of unique labels if we are doing classification
        label_list = np.unique(labels).tolist()

        # convert labels to numbers if we are doing classification
        ytrain, label_dict, inverse_label_dict = convert_labels_to_array(labels, label_list)
    else:
        ytrain = np.array([label_dict[extract_label(x)] for x in examples])

    # Create feat_vectorizer if we weren't passed one
    if feat_vectorizer is None:
        feat_vectorizer = extract_feature_vectorizer(features)  # create feature name -> value mapping

    # Create scaler if we weren't passed one
    if scaler is None and model_type != 'naivebayes':
        scaler = Scaler()

    # vectorize and scale the features
    xtrain = feat_vectorizer.transform(features)

    # Convert to dense if using naivebayes or rforest
    if model_type in ['naivebayes', 'rforest']:
        xtrain = xtrain.todense()

    xtrain_scaled = xtrain if model_type == 'naivebayes' else scaler.fit_transform(xtrain)

    # set up a grid searcher if we are asked to
    estimator, param_grid = create_estimator(model_type)
    if grid_search:
        if param_grid_file:
            with open(param_grid_file) as f:
                param_grid = json.load(f)

        # NOTE: we don't want to use multithreading for LIBLINEAR since it seems to lead to irreproducible results
        grid_searcher = GridSearchCV(estimator, param_grid, score_func=grid_objective, cv=cv_folds, n_jobs=(cv_folds if model_type not in ["svm_linear", "logistic"] else 1))

        # run the grid search for hyperparameters
        print('\tstarting grid search', file=sys.stderr)
        grid_searcher.fit(xtrain_scaled, ytrain)
        model = grid_searcher.best_estimator_
        score = grid_searcher.best_score_
    else:
        model = estimator.fit(xtrain_scaled, ytrain)
        score = 0.0

    # write out the model and the feature vocabulary
    if modelfile:
        # create the directory if it doesn't exist
        modeldir = os.path.dirname(modelfile)
        if not os.path.exists(modeldir):
            subprocess.call("mkdir -p {}".format(modeldir), shell=True)
        # write out the files
        with open(modelfile, "w") as f:
            pickle.dump(model, f, -1)


    if vocabfile:
        # create the directory if it doesn't exist
        vocabdir = os.path.dirname(vocabfile)
        if not os.path.exists(vocabdir):
            subprocess.call("mkdir -p {}".format(vocabdir), shell=True)
        with open(vocabfile, "w") as f:
            pickle.dump([feat_vectorizer, scaler, label_dict, inverse_label_dict], f, -1)

    # train_without_featurization function used to only return model and score, so fix references to that
    return model, score, feat_vectorizer, scaler, label_dict, inverse_label_dict


def evaluate(examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, model_type='logistic', prediction_prefix=None):
    '''
    Evaluates a given model on a given dev or test example set.
    Returns the confusion matrix, the per-class PRFs and the overall accuracy.
    '''
    features = [extract_features(x) for x in examples]

    # transform and scale the features
    xtest = feat_vectorizer.transform(features)

    # Convert to dense if using naivebayes or rforest
    if model_type in ['naivebayes', 'rforest']:
        xtest = xtest.todense()

    xtest_scaled = xtest if model_type == 'naivebayes' else scaler.fit_transform(xtest)
    ytest = np.array([label_dict[extract_label(x)] for x in examples])

    # make the prediction on the test data
    yhat = model.predict(xtest_scaled)

    # get the overall accuracy
    acc = accuracy(ytest, yhat)
    results = [round(acc, 3)]

    # get the confusion matrix and the per class accuracies
    # conf_mat = metrics.confusion_matrix(ytest, yhat)
    # if conf_mat.shape == (3, 3):
    #     per_class_accuracies = compute_class_accuracies(conf_mat)
    # else:
    #     per_class_accuracies = [round(f, 3) for f in [acc, acc]]
    # results.extend(per_class_accuracies)

    # compute the per-class PRFs
    precisions, recalls, f1_scores, _ = metrics.precision_recall_fscore_support(ytest, yhat)

    # reverse the PRF lists to get 'PNE' ordering instead of the alphabetical ordering 'ENP' (or 'SO' ordering instead of 'OS' ordering)
    precisions = [round(100 * f, 0) for f in precisions[::-1]]
    recalls = [round(100 * f, 0) for f in recalls[::-1]]
    f1_scores = [round(100 * f, 0) for f in f1_scores[::-1]]

    # append the PRF values in PNE/SO order to the results list
    for value in chain.from_iterable(izip(precisions, recalls, f1_scores)):
        results.append(value)

    # write out the predictions if we are asked to
    if prediction_prefix:
        prediction_file = prediction_prefix + '-{}.predictions'.format(model_type)
        with open(prediction_file, "w") as predictionfh:
            for pred in yhat:
                print(inverse_label_dict[pred], file=predictionfh)
            print(file=predictionfh)

    # return the results list
    return results


def predict(examples, model, feat_vectorizer, scaler, inverse_label_dict, prediction_prefix, model_type='logistic'):
    '''
    Uses a given model to generate predictions on a given data set
    '''
    features = [extract_features(x) for x in examples]

    # transform and scale the features
    xtest = feat_vectorizer.transform(features)
    xtest_scaled = xtest if model_type == 'naivebayes' else scaler.fit_transform(xtest)

    # make the prediction on the test data
    yhat = model.predict(xtest_scaled)

    # write out the predictions if we are asked to
    prediction_file = prediction_prefix + '-{}.predictions'.format(model_type)
    with open(prediction_file, "w") as predictionfh:
        for pred in yhat:
            print(inverse_label_dict[pred], file=predictionfh)
        print(file=predictionfh)


def cross_validate(examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, model_type='logistic', prediction_prefix=None, stratified=True, cv_folds=10):
    '''
    Cross-validates a given model on the training examples.
    Returns the confusion matrix, the per-class PRFs and the overall accuracy.
    '''
    features = [extract_features(x) for x in examples]

    # transform and scale the features
    X = feat_vectorizer.transform(features)

    # Convert to dense if using naivebayes or rforest
    if model_type in ['naivebayes', 'rforest']:
        X = X.todense()

    # scale the features
    X_scaled = X if model_type == 'naivebayes' else scaler.fit_transform(X)
    y = np.array([label_dict[extract_label(x)] for x in examples])

    # compute the five-fold cross-validation iterator
    kfold = StratifiedKFold(y, k=cv_folds) if stratified else KFold(y, k=cv_folds)

    # handle each fold separately and accumulate the predictions and the numbers
    yhat = -1 * np.ones(X_scaled.shape[0])
    for train_index, test_index in kfold:
        results = []
        fold_model = model.fit(X_scaled[train_index], y[train_index])
        fold_ytest = y[test_index]
        fold_yhat = fold_model.predict(X_scaled[test_index])
        yhat[test_index] = fold_yhat

        # get the fold accuracy
        fold_accuracy = metrics.zero_one_score(fold_ytest, fold_yhat)
        results = [fold_accuracy]

        # get the confusion matrix and the per class accuracies for this fold -- don't actually do the per class accuracies anymore
        results.append(metrics.confusion_matrix(fold_ytest, fold_yhat))

        # compute the per-class PRFs
        precisions, recalls, f1_scores, _ = metrics.precision_recall_fscore_support(fold_ytest, fold_yhat)

        # round the per-class PRFs (used to reverse these, but can do that after the fact)
        precisions = [round(100 * f, 0) for f in precisions]
        recalls = [round(100 * f, 0) for f in recalls]
        f1_scores = [round(100 * f, 0) for f in f1_scores]

        # append the PRF values to the results list
        results.extend(chain.from_iterable(izip(precisions, recalls, f1_scores)))

    # make sure there are no items for which we haven't generated a prediction
    missing = len(yhat[yhat == -1])
    if missing > 0:
        print('\terror: missing predictions.', file=sys.stderr)

    # write out the predictions if we are asked to
    if prediction_prefix:
        prediction_file = prediction_prefix + '-{}.predictions'.format(model_type)
        with open(prediction_file, "w") as predictionfh:
            for pred in yhat:
                print(inverse_label_dict[pred], file=predictionfh)
            print(file=predictionfh)

    # return the results list
    return results
