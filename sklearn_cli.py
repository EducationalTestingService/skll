#!/usr/bin/env python
'''
Cross-validation with scikit-learn

(Based on a bunch of different people's scripts.)

@author: Dan Blanchard, dblanchard@ets.org
@date: September 2012
'''

#### NOTE: Make pipeline version (look at Chris's code on Gitlab) that does all pre-processing, etc., but allow an optional feature file that contains precomputed features
####       Probably want to just start with the feature version now, but set things up to be future-proof

from __future__ import print_function, unicode_literals

import argparse
import sys
from collections import defaultdict
from itertools import islice, izip
from multiprocessing import Pool

import numpy as np
from bs4 import UnicodeDammit
from nltk.metrics import precision, recall, f_measure
from sklearn import linear_model, svm, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Scaler
from texttable import Texttable


class UniqueNumberDict(dict):
    """ Class for creating sequential unique numbers for each key."""

    def __getitem__(self, key):
        if key not in self:
            self[key] = len(self)
        return dict.__getitem__(self, key)


def sanitize_line(line):
    ''' Return copy of line with all non-ASCII characters replaced with <U1234> sequences where 1234 is the value of ord() for the character. '''
    char_list = []
    for char in line:
        char_num = ord(char)
        char_list.append('<U{}>'.format(char_num) if char_num > 127 else char)
    return ''.join(char_list)


def megam_dict_iter(path, classes):
    '''
    Generator that yields dictionaries mapping from features to values for each pair of lines in path

    @param path: Path to MegaM file
    @type path: C{unicode}
    @param classes: List of classes which will be appended to
    @type classes: C{list} of C{unicode}
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
                classes.append(class_name)
                curr_info_dict = dict()
                if len(split_line) > 1:
                    # Get current instances feature-value pairs
                    field_pairs = split_line[1:]
                    field_names = islice(field_pairs, 0, None, 2)
                    field_values = islice(field_pairs, 1, None, 2)

                    # Add the feature-value pairs to dictionary
                    curr_info_dict.update(izip(field_names, field_values))
                yield curr_info_dict
            line_count += 1
            if line_count % 100 == 0:
                print(".", end="", file=sys.stderr)
        print("done", file=sys.stderr)


def load_megam_file(path, class_num_dict=None, dict_vectorizer=None):
    '''
    Loads a MegaM file into a sparse NumPy matrix

    @param path: Path to MegaM file
    @type path: C{unicode}
    @param dict_vectorizer: An existing DictVectorizer to use if we've already created one
    @type dict_vectorizer: L{DictVectorizer}

    @return: A tuple of a DictVectorizer that maps from feature names to indices and a sparse matrix containing the MegaM data
    @rtype: 2-C{tuple} of L{DictVectorizer} and C{sparse matrix}
    '''
    # Initialize class list
    classes = []
    if class_num_dict is None:
        class_num_dict = UniqueNumberDict()

    # Load MegaM file into sparse array
    if dict_vectorizer is None:
        dict_vectorizer = DictVectorizer()
        data = dict_vectorizer.fit_transform(megam_dict_iter(path, classes)).tocsr()
    else:
        data = dict_vectorizer.transform(megam_dict_iter(path, classes)).tocsr()

    class_array = np.array([class_num_dict[class_name] for class_name in classes])

    print("Class num dict keys: {}".format(class_num_dict.keys()))

    return (dict_vectorizer, data, class_array, sorted(class_num_dict.keys()))


def process_fold(arg_tuple):
    ''' Function to process a given fold. Meant to be used asynchronously. '''
    fold_learner, fold_train_data, fold_test_data, fold_train_classes, fold_test_classes, k, class_names = arg_tuple
    actual_dict = defaultdict(set)
    pred_dict = defaultdict(set)

    # Scale data
    scaler = Scaler(with_mean=False)
    fold_train_data = scaler.fit_transform(fold_train_data)
    fold_test_data = scaler.transform(fold_test_data)

    # Train model
    fold_prefix = "[Fold {}]\t".format(k) if k > 0 else ""
    print(fold_prefix + "Training model...", file=sys.stderr)
    sys.stderr.flush()
    print("Train data: {}".format(fold_train_data))
    print("Train classes: {}".format(fold_train_classes))
    print("Test data: {}".format(fold_test_data))
    print("Test classes: {}".format(fold_test_classes))
    print("Num test classes: {}".format(len(fold_test_classes)))
    fold_learner.fit(fold_train_data, fold_train_classes)

    # Get predictions
    print(fold_prefix + "Testing model...".format(k), file=sys.stderr)
    sys.stderr.flush()
    raw_predictions = fold_learner.predict(fold_test_data)
    print("Learner: {}".format(fold_learner))
    print("Predictions: {}".format(raw_predictions))
    print("Num predictions: {}".format(len(raw_predictions)))
    pred_list = [class_names[pred_class] for pred_class in raw_predictions]
    actual_list = [class_names[actual_class] for actual_class in fold_test_classes]
    for line_num, (pred_class, actual_class) in enumerate(izip(pred_list, actual_list)):
        pred_dict[pred_class].add(line_num)
        actual_dict[actual_class].add(line_num)

    # Calculate metrics
    result_dict = defaultdict(dict)
    fold_score = metrics.zero_one_score(fold_test_classes, raw_predictions) * 100
    # Store results
    for actual_class in sorted(actual_dict.iterkeys()):
        result_dict[actual_class]["Precision"] = precision(actual_dict[actual_class], pred_dict[actual_class])
        result_dict[actual_class]["Recall"] = recall(actual_dict[actual_class], pred_dict[actual_class])
        result_dict[actual_class]["F-measure"] = f_measure(actual_dict[actual_class], pred_dict[actual_class])

    return (fold_score, result_dict, metrics.confusion_matrix(fold_test_classes, raw_predictions).tolist())


def print_fancy_output(result_tuples):
    ''' Function to take all of the results from all of the folds and print nice tables with the resluts. '''
    num_folds = len(result_tuples)
    score_sum = 0.0
    prec_sum_dict = defaultdict(float)
    recall_sum_dict = defaultdict(float)
    f_sum_dict = defaultdict(float)
    classes = sorted(result_tuples[0][1].iterkeys())
    for k, (fold_score, result_dict, conf_matrix) in enumerate(result_tuples, start=1):
        if num_folds > 1:
            print("\nFold: {}".format(k))
        result_table = Texttable(max_width=0)
        result_table.set_cols_align(["r"] * (len(classes) + 4))
        result_table.add_rows([[""] + classes + ["Precision", "Recall", "F-measure"]], header=True)
        for i, actual_class in enumerate(classes):
            conf_matrix[i][i] = "[{}]".format(conf_matrix[i][i])
            class_prec = (result_dict[actual_class]["Precision"] * 100) if "Precision" in result_dict[actual_class] and result_dict[actual_class]["Precision"] else 0
            class_recall = (result_dict[actual_class]["Recall"] * 100) if "Recall" in result_dict[actual_class] and result_dict[actual_class]["Recall"] else 0
            class_f = (result_dict[actual_class]["F-measure"] * 100) if "F-measure" in result_dict[actual_class] and result_dict[actual_class]["F-measure"] else 0
            prec_sum_dict[actual_class] += class_prec
            recall_sum_dict[actual_class] += class_recall
            f_sum_dict[actual_class] += class_f
            result_table.add_row([actual_class] + conf_matrix[i] + ["{:.1f}%".format(class_prec), "{:.1f}%".format(class_recall), "{:.1f}%".format(class_f)])
        print(result_table.draw())
        print("(row = reference; column = predicted)")
        print("Accuracy = {:.1f}%\n".format(fold_score))
        score_sum += fold_score

    if num_folds > 1:
        print("\nAverage:")
        result_table = Texttable(max_width=0)
        result_table.set_cols_align(["l", "r", "r", "r"])
        result_table.add_rows([["Class", "Precision", "Recall", "F-measure"]], header=True)
        for actual_class in classes:
            result_table.add_row([actual_class] + ["{:.1f}%".format(prec_sum_dict[actual_class] / num_folds),
                                                   "{:.1f}%".format(recall_sum_dict[actual_class] / num_folds),
                                                   "{:.1f}%".format(f_sum_dict[actual_class] / num_folds)])
        print(result_table.draw())
        print("Accuracy = {:.1f}%".format(score_sum / num_folds))


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes a MegaM-compatible data file and uses sklearn to perform either k-fold cross-validation or training and testing.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('train_file', help='MegaM input file representing the training data', default='-', nargs='?')
    parser.add_argument('test_file', help='Optional MegaM input file. If unspecified, script will do k-fold cross-validation', nargs='?')
    parser.add_argument('-c', '--classifier_type', help='Type of classifier to use.', choices=['logistic', 'svc'], default='logistic')
    parser.add_argument('-d', '--dev', help='Number of partitions to reserve for parameter-tuning if cross-validating.', type=int, default=1)
    parser.add_argument('-k', '--num_folds', help='Number of folds for cross-validation.', type=int, default=10)
    parser.add_argument('-l', '--loss', help='The loss function. "l1" is the hinge loss (standard SVM) while "l2" is the squared hinge loss.', choices=['l1', 'l2'], default='l2')
    parser.add_argument('-p', '--primal', help='Set %(default)s to solve the primal optimization problem (instead of the dual). Use this option when n_samples > n_features.',
                                          action='store_false', dest='dual')
    parser.add_argument('-r', '--regularization', help='The norm used in penalization/regularization. "l2" penalty is the standard used in SVC. ' +
                                                       '"l1" leads to sparse feature weight vectors.', choices=['l1', 'l2'], default='l2', dest='penalty')
    parser.add_argument('-t', '--test', help='Number of partitions to reserve for testing if cross-validating.', type=int, default=1)
    # parser.add_argument('-u', '--uniform', help='Ensure that all partitions are the same size and contain the same number of instances of each class.',
    #                     action='store_true')
    parser.add_argument('-v', '--verbose', help='Print debugging information to STDERR.', action='store_true')
    parser.add_argument('--no_parallel', help="Do not run the folds in parallel.", action='store_true')
    args = parser.parse_args()

    # Create process pool if necessary
    if not (args.no_parallel or args.test_file):
        pool = Pool(processes=args.num_folds)

    # Handle dual/primal variable setting
    if args.dual is None:
        args.dual = True

    if args.classifier_type == 'logistic':
        learner = linear_model.LogisticRegression(penalty=args.penalty, dual=args.dual, tol=0.01)
    elif args.classifier_type == 'svc':
        learner = svm.LinearSVC(penalty=args.penalty, loss=args.loss, dual=args.dual)

    # Read training file
    class_num_dict = UniqueNumberDict()
    vectorizer, train_data, train_classes, class_names = load_megam_file(args.train_file, class_num_dict=class_num_dict)

    # If given a test file, train on train_data, and test on test_data
    if args.test_file:
        vectorizer, test_data, test_classes, class_names = load_megam_file(args.test_file, class_num_dict=class_num_dict,  dict_vectorizer=vectorizer)
        print_fancy_output((process_fold((learner, train_data, test_data, train_classes, test_classes, 0, class_names)),))

    # Otherwise do cross validation
    else:
        # Pool.map needs pickle-able 1-argument function
        arg_tuple_list = [(learner, train_data[train_index], train_data[test_index], train_classes[train_index], train_classes[test_index], fold_num, class_names)
                          for fold_num, (train_index, test_index) in enumerate(StratifiedKFold(train_classes, args.num_folds, indices=True), start=1)]

        # Process each fold
        if args.no_parallel:
            results = [process_fold(a_tuple) for a_tuple in arg_tuple_list]
        else:
            results = pool.map(process_fold, arg_tuple_list)

        print_fancy_output(results)


if __name__ == '__main__':
    main()