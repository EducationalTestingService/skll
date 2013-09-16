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
Functions related to running experiments and parsing configuration files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
'''

from __future__ import absolute_import, print_function, unicode_literals

import csv
import datetime
import errno
import json
import logging
import math
import os
import sys
import tempfile
from collections import defaultdict
from io import open
from itertools import chain
from multiprocessing import Pool

import configparser  # Backported version from Python 3
import numpy as np
import scipy.sparse as sp
from prettytable import PrettyTable, ALL
from six import string_types, iterkeys, iteritems  # Python 2/3
from six.moves import zip
from sklearn.metrics import SCORERS

from skll.data import ExamplesTuple, load_examples
from skll.learner import Learner, MAX_CONCURRENT_PROCESSES

# Check if gridmap is available
try:
    from gridmap import Job, JobException, process_jobs
except ImportError:
    _HAVE_GRIDMAP = False
else:
    _HAVE_GRIDMAP = True

_VALID_TASKS = frozenset(['predict', 'train_only',
                          'evaluate', 'cross_validate'])

# Map from learner short names to full names
_SHORT_NAMES = {'logistic': 'LogisticRegression',
                'svm_linear': 'LinearSVC',
                'svm_radial': 'SVC',
                'naivebayes': 'MultinomialNB',
                'dtree': 'DecisionTreeClassifier',
                'rforest': 'RandomForestClassifier',
                'gradient': 'GradientBoostingClassifier',
                'ridge': 'Ridge',
                'rescaled_ridge': 'RescaledRidge',
                'svr_linear': 'SVR',
                'rescaled_svr_linear': 'RescaledSVR',
                'gb_regressor': 'GradientBoostingRegressor'}



def _get_stat_float(class_result_dict, stat):
    '''
    Little helper for getting output for precision, recall, and f-score
    columns in confusion matrix.

    :param class_result_dict: Dictionary containing the stat we'd like
                              to retrieve for a particular class.
    :type class_result_dict: dict
    :param stat: The statistic we're looking for in the dictionary.
    :type stat: str

    :return: The value of the stat if it's in the dictionary, and NaN
             otherwise.
    :rtype: float
    '''
    if stat in class_result_dict and class_result_dict[stat] is not None:
        return class_result_dict[stat]
    else:
        return float('nan')


def _write_summary_file(result_json_paths, output_file, ablation=False):
    '''
    Function to take a list of paths to individual result
    json files and returns a single file that summarizes
    all of them.

    :param result_json_paths: A list of paths to the
                              individual result json files.
    :type result_json_paths: list
    :return output_file: The output file to contain a summary
                        of the individual result files.
    :type output_file: file
    '''
    learner_result_dicts = []
    all_features = set()
    for json_path in result_json_paths:
        if not os.path.exists(json_path):
            raise IOError(errno.ENOENT, (('JSON file {} not found'
                                          .format(json_path))))
        else:
            with open(json_path, 'r') as json_file:
                obj = json.load(json_file)
                if ablation:
                    all_features.update(json.loads(obj[0]['featureset']))
                learner_result_dicts.extend(obj)

    header = set(learner_result_dicts[0].keys()) - {'result_table',
                                                    'descriptive',
                                                    'comparative'}
    header = (sorted(header.union(['ablated_feature'])) if ablation
              else sorted(header))
    writer = csv.DictWriter(output_file, header, extrasaction='ignore',
                            dialect=csv.excel_tab)
    writer.writeheader()

    # note that at this point each learner dict contains json dumped objects
    # which look really strange when written to a TSV. We want to convert
    # them to more readable string versions.
    for lrd in learner_result_dicts:
        if ablation:
            ablated_feature = all_features.difference(json.loads(lrd['featureset']))
            lrd['ablated_feature'] = ''
            if ablated_feature:
                lrd['ablated_feature'] = list(ablated_feature)[0]

        # write out the new learner dict with the readable fields
        writer.writerow(lrd)

    output_file.flush()


def _print_fancy_output(learner_result_dicts, output_file=sys.stdout):
    '''
    Function to take all of the results from all of the folds and print
    nice tables with the results.
    '''
    if not learner_result_dicts:
        raise ValueError('Result dictionary list is empty!')

    lrd = learner_result_dicts[0]
    print('Experiment Name: {}'.format(lrd['experiment_name']),
          file=output_file)
    print('Timestamp: {}'.format(lrd['timestamp']), file=output_file)
    print('Training Set: {}'.format(lrd['train_set_name']), file=output_file)
    print('Test Set: {}'.format(lrd['test_set_name']), file=output_file)
    print('Feature Set: {}'.format(lrd['featureset']), file=output_file)
    print('Learner: {}'.format(lrd['learner_name']), file=output_file)
    print('Task: {}'.format(lrd['task']), file=output_file)
    print('Feature Scaling: {}'.format(lrd['feature_scaling']),
          file=output_file)
    print('Grid Search: {}'.format(lrd['grid_search']), file=output_file)
    print('Grid Objective: {}'.format(lrd['grid_objective']), file=output_file)
    print('\n', file=output_file)

    for lrd in learner_result_dicts:
        print('Fold: {}'.format(lrd['fold']), file=output_file)
        print('Model Parameters: {}'.format(lrd.get('model_params', '')),
              file=output_file)
        print('Grid search score = {}'.format(lrd.get('grid_score', '')),
              file=output_file)
        if 'result_table' in lrd:
            print(lrd['result_table'], file=output_file)
            print('Accuracy = {}'.format(lrd['accuracy']),
                  file=output_file)
        if 'descriptive' in lrd:
            print('Descriptive statistics:', file=output_file)
            for desc_stat in ['min', 'max', 'avg', 'std']:
                print((' {}: {: .4f} (actual), {: .4f} ' +
                       '(predicted)').format(desc_stat.title(),
                                             lrd['descriptive']['actual'][desc_stat],
                                             lrd['descriptive']['predicted'][desc_stat]),
                      file=output_file)
            print('Pearson:{: f}'.format(lrd['comparative']['pearson']),
                  file=output_file)
        print('Objective function score = {}'.format(lrd['score']),
              file=output_file)
        print('', file=output_file)


def _parse_config_file(config_path):
    '''
    Parses a SKLL experiment configuration file with the given path.
    '''
    # initialize config parser
    config = configparser.ConfigParser({'test_location': '',
                                        'log': '',
                                        'results': '',
                                        'predictions': '',
                                        'models': '',
                                        'grid_search': 'False',
                                        'objective': "f1_score_micro",
                                        'probability': 'False',
                                        'fixed_parameters': '[]',
                                        'param_grids': '[]',
                                        'pos_label_str': '',
                                        'featureset_names': '[]',
                                        'feature_scaling': 'none',
                                        'min_feature_count': '1',
                                        'grid_search_jobs': '0',
                                        'cv_folds_location': '',
                                        'suffix': '',
                                        'classifiers': '',
                                        'tsv_label': 'y',
                                        'ids_to_floats': 'False'})

    if not os.path.exists(config_path):
        raise IOError(errno.ENOENT, "The config file doesn't exist.",
                      config_path)

    config.read(config_path)
    return config


def _load_featureset(dirpath, featureset, suffix, tsv_label='y',
                     ids_to_floats=False):
    '''
    loads a list of feature files and merges them.
    '''

    # Load a list of lists of examples, one list of examples per featureset.
    file_names = [os.path.join(dirpath, featfile + suffix) for featfile
                  in featureset]
    example_tuples = [load_examples(file_name, tsv_label=tsv_label,
                                    ids_to_floats=ids_to_floats)
                      for file_name in file_names]

    # Check that the IDs are unique within each file.
    for file_name, examples in zip(file_names, example_tuples):
        ex_ids = examples.ids
        if len(ex_ids) != len(set(ex_ids)):
            raise ValueError(('The example IDs are not unique in ' +
                              '{}.').format(file_name))

    # Check that the different feature files have the same IDs.
    # To do this, make a sorted tuple of unique IDs for each feature file,
    # and then make sure they are all the same by making sure the set has one
    # item in it.
    mismatch_num = len({tuple(sorted(examples.ids)) for examples in
                        example_tuples})
    if mismatch_num != 1:
        raise ValueError(('The sets of example IDs in {} feature files do ' +
                          'not match').format(mismatch_num))

    # Make sure there is a unique label for every example (or no label, for
    # "unseen" examples).
    # To do this, find the unique (id, y) tuples, and then make sure that all
    # those ids are unique.
    unique_tuples = set(chain(*[[(curr_id, curr_label) for curr_id, curr_label
                                 in zip(examples.ids, examples.classes)]
                                for examples in example_tuples]))
    if len({tup[0] for tup in unique_tuples}) != len(unique_tuples):
        raise ValueError('At least two feature files have different labels ' +
                         '(i.e., y values) for the same ID.')

    # Now, create the final ExamplesTuple of examples with merged features
    merged_vectorizer = None
    merged_features = None
    merged_ids = None
    merged_classes = None
    for ids, classes, features, feat_vectorizer in example_tuples:
        # Combine feature matrices and vectorizers
        if merged_features is not None:
            # Check for duplicate feature names
            if (set(merged_vectorizer.get_feature_names()) &
                    set(feat_vectorizer.get_feature_names())):
                raise ValueError('Two feature files have the same feature!')

            num_merged = merged_features.shape[1]
            merged_features = sp.hstack([merged_features, features], 'csr')

            # dictvectorizer sorts the vocabularies within each file
            for feat_name, index in sorted(feat_vectorizer.vocabulary_.items(),
                                           key=lambda x: x[1]):
                merged_vectorizer.vocabulary_[feat_name] = index + num_merged
                merged_vectorizer.feature_names_.append(feat_name)
        else:
            merged_features = features
            merged_vectorizer = feat_vectorizer

        # IDs should be the same for each ExamplesTuple, so only store once
        if merged_ids is None:
            merged_ids = ids
        # Check that IDs are in the same order
        elif not np.all(merged_ids == ids):
            raise ValueError('IDs are not in the same order in each feature ' +
                             'file!')

        # Classes should be the same for each ExamplesTuple, so only store once
        if merged_classes is None:
            merged_classes = classes
        # Check that classes don't conflict, when specified
        elif classes is not None and not np.all(merged_classes == classes):
            raise ValueError('Feature files have conflicting labels for ' +
                             'examples with the same ID!')

    # Ensure that at least one file had classes
    if merged_classes is None:
        raise ValueError('No feature files in feature set contain class' +
                         'labels!')

    return ExamplesTuple(merged_ids, merged_classes, merged_features,
                         merged_vectorizer)


def _classify_featureset(args):
    ''' Classification job to be submitted to grid '''

    # Extract all the arguments.
    # (There doesn't seem to be a better way to do this since one can't specify
    # required keyword arguments.)
    experiment_name = args.pop("experiment_name")
    task = args.pop("task")
    jobname = args.pop("jobname")
    featureset = args.pop("featureset")
    learner_name = args.pop("learner_name")
    train_path = args.pop("train_path")
    test_path = args.pop("test_path")
    train_set_name = args.pop("train_set_name")
    test_set_name = args.pop("test_set_name")
    modelpath = args.pop("modelpath")
    prediction_prefix = args.pop("prediction_prefix")
    grid_search = args.pop("grid_search")
    grid_objective = args.pop("grid_objective")
    suffix = args.pop("suffix")
    log_path = args.pop("log_path")
    probability = args.pop("probability")
    resultspath = args.pop("resultspath")
    fixed_parameters = args.pop("fixed_parameters")
    param_grid = args.pop("param_grid")
    pos_label_str = args.pop("pos_label_str")
    overwrite = args.pop("overwrite")
    feature_scaling = args.pop("feature_scaling")
    min_feature_count = args.pop("min_feature_count")
    grid_search_jobs = args.pop("grid_search_jobs")
    cv_folds = args.pop("cv_folds")
    tsv_label = args.pop("tsv_label")
    ids_to_floats = args.pop("ids_to_floats")
    if args:
        raise ValueError("Extra arguments passed: {}".format(args.keys()))

    timestamp = datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')

    with open(log_path, 'w') as log_file:

        # logging
        print("Task: {}".format(task), file=log_file)
        if task == 'cross_validate':
            print(("Cross-validating on {}, feature " +
                   "set {} ...").format(train_set_name, featureset),
                  file=log_file)
        elif task == 'evaluate':
            print(("Training on {}, Test on {}, " +
                   "feature set {} ...").format(train_set_name, test_set_name,
                                                featureset),
                  file=log_file)
        elif task == 'train_only':
            print("Training on {}, feature set {} ...".format(train_set_name,
                                                              featureset),
                  file=log_file)
        else:  # predict
            print(("Training on {}, Making predictions about {}, " +
                   "feature set {} ...").format(train_set_name, test_set_name,
                                                featureset),
                  file=log_file)

        # load the training and test examples
        train_examples = _load_featureset(train_path, featureset, suffix,
                                          tsv_label=tsv_label,
                                          ids_to_floats=ids_to_floats)
        if task == 'evaluate' or task == 'predict':
            test_examples = _load_featureset(test_path, featureset, suffix,
                                             tsv_label=tsv_label,
                                             ids_to_floats=ids_to_floats)

        # initialize a classifer object
        learner = Learner(learner_name,
                          probability=probability,
                          feature_scaling=feature_scaling,
                          model_kwargs=fixed_parameters,
                          pos_label_str=pos_label_str,
                          min_feature_count=min_feature_count)

        # check whether a trained model on the same data with the same
        # featureset already exists if so, load it (and the feature
        # vocabulary) and then use it on the test data
        modelfile = os.path.join(modelpath, '{}.model'.format(jobname))

        # create a list of dictionaries of the results information
        learner_result_dict_base = {'experiment_name': experiment_name,
                                    'train_set_name': train_set_name,
                                    'test_set_name': test_set_name,
                                    'featureset': json.dumps(featureset),
                                    'learner_name': learner_name,
                                    'task': task,
                                    'timestamp': timestamp,
                                    'feature_scaling': feature_scaling,
                                    'grid_search': grid_search,
                                    'grid_objective': grid_objective}

        # check if we're doing cross-validation, because we only load/save
        # models when we're not.
        task_results = None
        if task == 'cross_validate':
            print('\tcross-validating', file=log_file)
            task_results, grid_scores = learner.cross_validate(train_examples,
                                                               prediction_prefix=prediction_prefix,
                                                               grid_search=grid_search,
                                                               cv_folds=cv_folds,
                                                               grid_objective=grid_objective,
                                                               param_grid=param_grid,
                                                               grid_jobs=grid_search_jobs)
        else:
            # load the model if it already exists
            if os.path.exists(modelfile) and not overwrite:
                print(('\tloading pre-existing {} ' +
                       'model: {}').format(learner_name, modelfile))
                learner.load(modelfile)

            # if we have do not have a saved model, we need to train one.
            else:
                print(('\tfeaturizing and training new ' +
                       '{} model').format(learner_name),
                      file=log_file)

                grid_search_folds = 5
                if not isinstance(cv_folds, int):
                    grid_search_folds = cv_folds

                best_score = learner.train(train_examples,
                                           grid_search=grid_search,
                                           grid_search_folds=grid_search_folds,
                                           grid_objective=grid_objective,
                                           param_grid=param_grid,
                                           grid_jobs=grid_search_jobs)
                grid_scores = [best_score]

                # save model
                learner.save(modelfile)

                if grid_search:
                    print('\tbest {} grid search score: {}'
                          .format(grid_objective, round(best_score, 3)),
                          file=log_file)

            # print out the tuned parameters and best CV score
            param_out = ('{}: {}'.format(param_name, param_value)
                         for param_name, param_value in
                         iteritems(learner.model.get_params()))
            print('\thyperparameters: {}'.format(', '.join(param_out)),
                  file=log_file)

            # run on test set or cross-validate on training data,
            # depending on what was asked for

            if task == 'evaluate':
                print('\tevaluating predictions', file=log_file)
                task_results = [learner.evaluate(
                    test_examples, prediction_prefix=prediction_prefix,
                    grid_objective=grid_objective)]
            elif task == 'predict':
                print('\twriting predictions', file=log_file)
                learner.predict(test_examples,
                                prediction_prefix=prediction_prefix)
            # do nothing here for train_only

        if task == 'cross_validate' or task == 'evaluate':
            results_json_path = os.path.join(resultspath,
                                             '{}.results.json'.format(jobname))

            res = _create_learner_result_dicts(task_results, grid_scores,
                                               learner_result_dict_base)

            # write out the result dictionary to a json file
            file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
            with open(results_json_path, file_mode) as json_file:
                json.dump(res, json_file)

            with open(os.path.join(resultspath, '{}.results'.format(jobname)),
                      'w') as output_file:
                _print_fancy_output(res, output_file)
        else:
            res = [learner_result_dict_base]

    return res


def _create_learner_result_dicts(task_results, grid_scores,
                                 learner_result_dict_base):
    res = []

    num_folds = len(task_results)
    accuracy_sum = 0.0
    score_sum = None
    prec_sum_dict = defaultdict(float)
    recall_sum_dict = defaultdict(float)
    f_sum_dict = defaultdict(float)
    result_table = None

    for k, ((conf_matrix, fold_accuracy, result_dict, model_params,
            score), grid_score) \
            in enumerate(zip(task_results, grid_scores), start=1):

        # create a new dict for this fold
        learner_result_dict = {}
        learner_result_dict.update(learner_result_dict_base)

        # initialize some variables to blanks so that the
        # set of columns is fixed.
        learner_result_dict['result_table'] = ''
        learner_result_dict['accuracy'] = ''
        learner_result_dict['score'] = ''
        learner_result_dict['fold'] = ''

        if learner_result_dict_base['task'] == 'cross_validate':
            learner_result_dict['fold'] = k

        learner_result_dict['model_params'] = json.dumps(model_params)
        learner_result_dict['grid_score'] = grid_score

        if conf_matrix:
            classes = sorted(iterkeys(task_results[0][2]))
            result_table = PrettyTable([""] + classes + ["Precision",
                                                         "Recall",
                                                         "F-measure"],
                                       header=True, hrules=ALL)
            result_table.align = 'r'
            result_table.float_format = '.3'
            for i, actual_class in enumerate(classes):
                conf_matrix[i][i] = "[{}]".format(conf_matrix[i][i])
                class_prec = _get_stat_float(result_dict[actual_class],
                                             "Precision")
                class_recall = _get_stat_float(result_dict[actual_class],
                                               "Recall")
                class_f = _get_stat_float(result_dict[actual_class],
                                          "F-measure")
                if not math.isnan(class_prec):
                    prec_sum_dict[actual_class] += float(class_prec)
                if not math.isnan(class_recall):
                    recall_sum_dict[actual_class] += float(class_recall)
                if not math.isnan(class_f):
                    f_sum_dict[actual_class] += float(class_f)
                result_row = ([actual_class] + conf_matrix[i] +
                              [class_prec, class_recall, class_f])
                result_table.add_row(result_row)

            result_table_str = '{}'.format(result_table)
            result_table_str += '(row = reference; column = predicted)'
            learner_result_dict['result_table'] = result_table_str
            learner_result_dict['accuracy'] = fold_accuracy
            accuracy_sum += fold_accuracy

        # if there is no confusion matrix, then we must be dealing
        # with a regression model
        else:
            learner_result_dict.update(result_dict)

        if score is not None:
            if score_sum is None:
                score_sum = score
            else:
                score_sum += score
            learner_result_dict['score'] = score
        res.append(learner_result_dict)

    if num_folds > 1:
        learner_result_dict = {}
        learner_result_dict.update(learner_result_dict_base)

        learner_result_dict['fold'] = 'average'

        if result_table:
            result_table = PrettyTable(["Class", "Precision", "Recall",
                                        "F-measure"],
                                       header=True)
            result_table.align = "r"
            result_table.align["Class"] = "l"
            result_table.float_format = '.3'
            for actual_class in classes:
                # Convert sums to means
                prec_mean = prec_sum_dict[actual_class] / num_folds
                recall_mean = recall_sum_dict[actual_class] / num_folds
                f_mean = f_sum_dict[actual_class] / num_folds
                result_table.add_row([actual_class] +
                                     [prec_mean, recall_mean, f_mean])

            learner_result_dict['result_table'] = '{}'.format(result_table)
            learner_result_dict['accuracy'] = accuracy_sum / num_folds

        if score_sum is not None:
            learner_result_dict['score'] = score_sum / num_folds
        res.append(learner_result_dict)
    return res


def _munge_featureset_name(featureset):
    '''
    Joins features in featureset by '+' if featureset is not a string, and
    just returns featureset otherwise.
    '''
    if isinstance(featureset, string_types):
        return featureset

    res = '+'.join(featureset)
    return res


def _fix_json(json_string):
    '''
    Takes a bit of JSON that might have bad quotes or capitalized booleans
    and fixes that stuff.
    '''
    json_string = json_string.replace('True', 'true')
    json_string = json_string.replace('False', 'false')
    json_string = json_string.replace("'", '"')
    return json_string


def _load_cv_folds(cv_folds_location, ids_to_floats=False):
    '''
    Loads CV folds from a CSV file with columns for example ID and fold ID
    (and a header).
    '''
    with open(cv_folds_location, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # discard the header
        res = {}
        for row in reader:
            if ids_to_floats:
                try:
                    row[0] = float(row[0])
                except ValueError:
                    raise ValueError(('You set ids_to_floats to true, but ' +
                                      'ID {} could not be converted to ' +
                                      'float').format(row[0]))
            res[row[0]] = row[1]

    return res


def run_configuration(config_file, local=False, overwrite=True, queue='all.q',
                      hosts=None, write_summary=True):
    '''
    Takes a configuration file and runs the specified jobs on the grid.

    :param config_path: Path to the configuration file we would like to use.
    :type config_path: str
    :param local: Should this be run locally instead of on the cluster?
    :type local: bool
    :param overwrite: If the model files already exist, should we overwrite
                      them instead of re-using them?
    :type overwrite: bool
    :param queue: The DRMAA queue to use if we're running on the cluster.
    :type queue: str
    :param hosts: If running on the cluster, these are the machines we should
                  use.
    :type hosts: list of str
    :param write_summary: Write a tsv file with a summary of the results.
    :type write_summary: bool

    :return: A list of paths to .json results files for each variation in the
             experiment.
    :rtype: list of str

    '''
    # Read configuration
    config = _parse_config_file(config_file)

    if not local and not _HAVE_GRIDMAP:
        local = True
        logging.warning('gridmap 0.10.1+ not available. Forcing local ' +
                        'mode.  To run things on a DRMAA-compatible ' +
                        'cluster, install gridmap>=0.10.1 via pip.')

    ###########################
    # extract parameters from the config file

    # General
    task = config.get("General", "task")
    if task not in _VALID_TASKS:
        raise ValueError('An invalid task was specified: {}. '.format(task) +
                         'Valid tasks are: {}'.format(' '.join(_VALID_TASKS)))

    experiment_name = config.get("General", "experiment_name")

    # Input
    if config.has_option("Input", "learners"):
        learners_string = config.get("Input", "learners")
    elif config.has_option("Input", "classifiers"):
        learners_string = config.get("Input", "classifiers")  # For old files
    else:
        raise ValueError("Configuration file does not contain list of " +
                         "learners in [Input] section.")
    learners = json.loads(_fix_json(learners_string))
    learners = [(_SHORT_NAMES[learner] if learner in _SHORT_NAMES else
                       learner) for learner in learners]
    featuresets = json.loads(_fix_json(config.get("Input", "featuresets")))

    # ensure that featuresets is a list of lists
    if not isinstance(featuresets, list) or \
       not all([isinstance(fs, list) for fs in featuresets]):
        raise ValueError("The featuresets parameter should be a " +
                         "list of lists: {}".format(featuresets))

    featureset_names = json.loads(_fix_json(config.get("Input",
                                                       "featureset_names")))

    # ensure that featureset_names is a list of strings, if specified
    if featureset_names:
        if (not isinstance(featureset_names, list) or
                not all([isinstance(fs, string_types) for fs in
                         featureset_names])):
            raise ValueError("The featureset_names parameter should be a " +
                             "list of strings: {}".format(featureset_names))

    fixed_parameter_list = json.loads(_fix_json(config.get("Input",
                                                           "fixed_parameters")))
    param_grid_list = json.loads(_fix_json(config.get("Tuning", "param_grids")))
    pos_label_str = config.get("Tuning", "pos_label_str")

    # ensure that feature_scaling is specified only as one of the
    # four available choices
    feature_scaling = config.get("Tuning", "feature_scaling")
    if feature_scaling not in ['with_std', 'with_mean', 'both', 'none']:
        raise ValueError("Invalid value for feature_scaling parameter: " +
                         "{}".format(feature_scaling))

    # get all the input paths and directories (without trailing slashes)
    train_path = config.get("Input", "train_location").rstrip('/')
    test_path = config.get("Input", "test_location").rstrip('/')
    suffix = config.get("Input", "suffix")
    tsv_label = config.get("Input", "tsv_label")
    ids_to_floats = config.getboolean("Input", "ids_to_floats")

    # get the cv folds file and make a dictionary from it
    cv_folds_location = config.get("Input", "cv_folds_location")
    if cv_folds_location:
        cv_folds = _load_cv_folds(cv_folds_location,
                                  ids_to_floats=ids_to_floats)
    else:
        cv_folds = 10

    # Output
    # get all the output files and directories
    resultspath = config.get("Output", "results")
    logpath = config.get("Output", "log")
    modelpath = config.get("Output", "models")
    probability = config.getboolean("Output", "probability")

    # do we want to keep the predictions?
    prediction_dir = config.get("Output", "predictions")
    if prediction_dir and not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # make sure log path exists
    if logpath and not os.path.exists(logpath):
        os.makedirs(logpath)

    # make sure results path exists
    if resultspath and not os.path.exists(resultspath):
        os.makedirs(resultspath)

    # make sure all the specified paths exist
    if not os.path.exists(train_path):
        raise IOError(errno.ENOENT, ("The training path specified in config " +
                                     "file does not exist."), train_path)
    if test_path and not os.path.exists(test_path):
        raise IOError(errno.ENOENT, ("The test path specified in config " +
                                     "file does not exist."), train_path)

    # Tuning
    # do we need to run a grid search for the hyperparameters or are we just
    # using the defaults?
    do_grid_search = config.getboolean("Tuning", "grid_search")

    # the minimum number of examples a feature must be nonzero in to be included
    min_feature_count = config.getint("Tuning", "min_feature_count")

    # how many jobs should we run in parallel for grid search
    grid_search_jobs = config.getint("Tuning", "grid_search_jobs")
    if not grid_search_jobs:
        grid_search_jobs = None

    # what is the objective function for the grid search?
    grid_objective = config.get("Tuning", "objective")
    if grid_objective not in SCORERS:
        raise ValueError(('Invalid grid objective function: ' +
                          '{}').format(grid_objective))

    # check whether the right things are set for the given task
    if (task == 'evaluate' or task == 'predict') and not test_path:
        raise ValueError('The test set and results locations must be set ' +
                         'when task is evaluate or predict.')
    if (task == 'cross_validate' or task == 'train_only') and test_path:
        raise ValueError('The test set path should not be set ' +
                         'when task is cross_validate or train_only.')
    if (task == 'train_only' or task == 'predict') and resultspath:
        raise ValueError('The results path should not be set ' +
                         'when task is predict or train_only.')
    if task == 'train_only' and not modelpath:
        raise ValueError('The model path should be set ' +
                         'when task is train_only.')
    if task == 'train_only' and prediction_dir:
        raise ValueError('The predictions path should not be set ' +
                         'when task is train_only.')
    if task == 'cross_validate' and modelpath:
        raise ValueError('The models path should not be set ' +
                         'when task is cross_validate.')

    ###########################

    # the list of jobs submitted (if running on grid)
    if not local:
        jobs = []

    if not featureset_names:
        featureset_names = [_munge_featureset_name(x) for x in featuresets]
    assert len(featureset_names) == len(featuresets)

    # store training/test set names for later use
    train_set_name = os.path.basename(train_path)
    test_set_name = os.path.basename(test_path) if test_path else "cv"

    # the list to hold the paths to all the result json files
    result_json_paths = []

    # For each feature set
    for featureset, featureset_name in zip(featuresets,
                                           featureset_names):

        # and for each learner
        for learner_num, learner_name in enumerate(learners):

            job_name_components = [experiment_name]

            # for the individual job name, we need to add the feature set name
            # and the learner name
            job_name_components.extend([featureset_name, learner_name])
            jobname = '_'.join(job_name_components)

            # change the prediction prefix to include the feature set
            prediction_prefix = os.path.join(prediction_dir, jobname)

            # the log file that stores the actual output of this script (e.g.,
            # the tuned parameters, what kind of experiment was run, etc.)
            temp_logfile = os.path.join(logpath, '{}.log'.format(jobname))

            # create job if we're doing things on the grid
            job_args = {}
            job_args["experiment_name"] = experiment_name
            job_args["task"] = task
            job_args["jobname"] = jobname
            job_args["featureset"] = featureset
            job_args["learner_name"] = learner_name
            job_args["train_path"] = train_path
            job_args["test_path"] = test_path
            job_args["train_set_name"] = train_set_name
            job_args["test_set_name"] = test_set_name
            job_args["modelpath"] = modelpath
            job_args["prediction_prefix"] = prediction_prefix
            job_args["grid_search"] = do_grid_search
            job_args["grid_objective"] = grid_objective
            job_args["suffix"] = suffix
            job_args["log_path"] = temp_logfile
            job_args["probability"] = probability
            job_args["resultspath"] = resultspath
            job_args["fixed_parameters"] = (fixed_parameter_list[learner_num]
                                            if fixed_parameter_list
                                            else dict())
            job_args["param_grid"] = (param_grid_list[learner_num]
                                      if param_grid_list else None)
            job_args["pos_label_str"] = pos_label_str
            job_args["overwrite"] = overwrite
            job_args["feature_scaling"] = feature_scaling
            job_args["min_feature_count"] = min_feature_count
            job_args["grid_search_jobs"] = grid_search_jobs
            job_args["cv_folds"] = cv_folds
            job_args["tsv_label"] = tsv_label
            job_args["ids_to_floats"] = ids_to_floats

            if not local:
                jobs.append(Job(_classify_featureset, [job_args],
                                num_slots=(MAX_CONCURRENT_PROCESSES if
                                           do_grid_search else 1),
                                name=jobname, queue=queue))
            else:
                _classify_featureset(job_args)

            # save the path to the results json file that will be written
            result_json_paths.append(os.path.join(resultspath,
                                     '{}.results.json'.format(jobname)))

    # submit the jobs (if running on grid)
    if not local and _HAVE_GRIDMAP:
        try:
            if logpath:
                job_results = process_jobs(jobs, white_list=hosts,
                                           temp_dir=logpath)
            else:
                job_results = process_jobs(jobs, white_list=hosts)
        except JobException as e:
            logging.error('gridmap claims that one of your jobs failed, but ' +
                          'this is not always true. \n{}'.format(e))
        else:
            _check_job_results(job_results)

    # write out the summary results file
    if (task == 'cross_validate' or task == 'evaluate') and write_summary:
        summary_file_name = experiment_name + '_summary.tsv'
        file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
        with open(os.path.join(resultspath, summary_file_name),
                  file_mode) as output_file:
            _write_summary_file(result_json_paths, output_file)

    return result_json_paths


def _check_job_results(job_results):
    '''
    See if we have a complete results dictionary for every job.
    '''
    logging.info('checking job results')
    for result_dicts in job_results:
        if not result_dicts or 'task' not in result_dicts[0]:
            logging.error('There was an error running the experiment:\n' +
                          '{}'.format(result_dicts))


def _run_experiment_without_feature(arg_tuple):
    '''
    Creates a new configuration file with a given feature
    removed and runs that experiment.

    :param arg_tuple: A tuple of the actual arguments for this function:
                      feature_type, features, config, local, queue
                      cfg_path, and machines.

                      - feature_type: The name of the feature set to exclude
                      - features: A list of all features in config
                      - featureset_name: The original featureset name
                      - config: A parsed configuration file
                      - local: Are we running things locally or on the grid?
                      - queue: Grid Map queue to use for scheduling
                      - cfg_path: Path to main configuration file
                      - machines: List of machines to use for scheduling jobs
                                  with Grid Map
                      - overwrite: Should we overwrite existing models?

    :type arg_tuple: tuple
    :return: A list of paths to .json results files for each variation in the
             experiment.
    :rtype: list of str

    '''
    (feature_type, features, featureset_name, config, local, queue,
     cfg_path, machines, overwrite) = arg_tuple

    featureset = [[x for x in features if x != feature_type]]

    if feature_type:
        featureset_name = "{}_minus_{}".format(featureset_name,
                                               feature_type)
    else:
        featureset_name = "{}_all".format(featureset_name)

    config.set("Input", "featuresets", json.dumps(featureset))
    config.set("Input", "featureset_names", "['{}']".format(featureset_name))

    file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
    with tempfile.NamedTemporaryFile(mode=file_mode,
                                     suffix='.cfg',
                                     delete=False) as new_config_file:
        config.write(new_config_file)

    result_jsons = run_configuration(new_config_file.name, local=local,
                                     queue=queue, hosts=machines,
                                     overwrite=overwrite, write_summary=False)

    # remove the temporary config file that we created
    os.unlink(new_config_file.name)

    return result_jsons


def run_ablation(config_path, local=False, overwrite=True, queue='all.q',
                 hosts=None):
    '''
    Takes a configuration file and runs repeated experiments where each
    feature set has been removed from the configuration.

    :param config_path: Path to the configuration file we would like to use.
    :type config_path: str
    :param local: Should this be run locally instead of on the cluster?
    :type local: bool
    :param overwrite: If the model files already exist, should we overwrite
                      them instead of re-using them?
    :type overwrite: bool
    :param queue: The DRMAA queue to use if we're running on the cluster.
    :type queue: str
    :param hosts: If running on the cluster, these are the machines we should
                  use.
    :type hosts: list of str
    '''
    # Read configuration
    config = _parse_config_file(config_path)

    featuresets = json.loads(_fix_json(config.get("Input", "featuresets")))
    featureset_names = json.loads(_fix_json(config.get("Input",
                                                       "featureset_names")))

    # make sure there is only one list of features
    if len(featuresets) > 1 or len(featureset_names) > 1:
        raise ValueError("More than one feature set or list of names given.")

    # make a list of features rather than a list of lists
    features = featuresets[0]
    featureset_name = 'ablation'
    if featureset_names:
        featureset_name = featureset_names[0]

    # for each feature file, make a copy of the config file
    # with all but that feature, and run the jobs.
    arg_tuples = ((feature_type, features, featureset_name, config, local,
                   queue, config_path, hosts, overwrite)
                  for feature_type in features + [None])

    result_json_paths = []
    if not local and not _HAVE_GRIDMAP:
        local = True
        logging.warning('gridmap 0.10.1+ not available. Forcing local ' +
                        'mode.  To run things on a DRMAA-compatible ' +
                        'cluster, install gridmap>=0.10.1 via pip.')

    if not local:
        pool = Pool(processes=len(features) + 1)
        try:
            result_json_paths.extend(chain(*pool.map(_run_experiment_without_feature,
                                                     list(arg_tuples))))
        # If we run_ablation is run via a subprocess (like nose does),
        # this will fail, so just do things serially then.
        except AssertionError:
            del pool
            for arg_tuple in arg_tuples:
                result_json_paths.extend(_run_experiment_without_feature(arg_tuple))
    else:
        for arg_tuple in arg_tuples:
            result_json_paths.extend(_run_experiment_without_feature(arg_tuple))

    task = config.get("General", "task")
    experiment_name = config.get("General", "experiment_name")
    resultspath = config.get("Output", "results")

    if task == 'cross_validate' or task == 'evaluate':
        summary_file_name = experiment_name + '_summary.tsv'
        file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
        with open(os.path.join(resultspath, summary_file_name),
                  file_mode) as output_file:
            _write_summary_file(result_json_paths, output_file, ablation=True)
