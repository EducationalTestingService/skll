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
import errno
import itertools
import json
import logging
import math
import os
import re
import numpy as np
import sys
import datetime
from collections import defaultdict
from multiprocessing import Pool

import scipy.sparse as sp
from prettytable import PrettyTable, ALL
from six import string_types, iterkeys, iteritems  # Python 2/3
from six.moves import configparser, zip
from sklearn.metrics import SCORERS

from skll.data import ExamplesTuple, load_examples
from skll.learner import Learner, MAX_CONCURRENT_PROCESSES


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


def _write_summary_file(result_json_paths, output_file):
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
    for json_path in result_json_paths:
        if not os.path.exists(json_path):
            raise IOError(errno.ENOENT, (('JSON file {} not found'
                                          .format(json_path))))
        else:
            with open(json_path, 'r') as json_file:
                learner_result_dicts.extend(json.load(json_file))

    header = sorted(set(learner_result_dicts[0].keys()) - {'result_table'})
    writer = csv.DictWriter(output_file, header, extrasaction='ignore',
                            dialect=csv.excel_tab)
    writer.writeheader()
    for lrd in learner_result_dicts:
        writer.writerow(lrd)

    output_file.flush()


def _print_fancy_output(learner_result_dicts, output_file=sys.stdout):
    '''
    Function to take all of the results from all of the folds and print
    nice tables with the results.
    '''

    lrd = learner_result_dicts[0]
    print('Timestamp: {}'.format(lrd['timestamp']), file=output_file)
    print('Training Set: {}'.format(lrd['train_set_name']), file=output_file)
    print('Test Set: {}'.format(lrd['test_set_name']), file=output_file)
    print('Feature Set: {}'.format(lrd['featureset']), file=output_file)
    print('Learner: {}'.format(lrd['given_learner']), file=output_file)
    print('Task: {}'.format(lrd['task']), file=output_file)
    print('Scaling: {}'.format(lrd['scaling']), file=output_file)
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

        print('Objective function score = {}'.format(lrd['score']),
              file=output_file)
        print(file=output_file)


def _parse_config_file(config_file):
    '''
    Parses a given SKLL experiment configuration file.
    '''
    # initialize config parser
    config = configparser.ConfigParser({'test_location': '',
                                        'log': '',
                                        'results': '',
                                        'predictions': '',
                                        'grid_search': 'False',
                                        'objective': "f1_score_micro",
                                        'scale_features': 'False',
                                        'probability': 'False',
                                        'fixed_parameters': '[]',
                                        'param_grids': '[]',
                                        'pos_label_str': '',
                                        'featureset_names': '[]',
                                        'use_dense_features': 'False',
                                        'min_feature_count': '1',
                                        'grid_search_jobs': '0',
                                        'cv_folds_location': '',
                                        'suffix': '',
                                        'classifiers': '',
                                        'tsv_label': 'y'})
    if sys.version_info[:2] >= (3, 2):
        config.read_file(config_file)
    else:
        config.readfp(config_file)

    return config


def _load_featureset(dirpath, featureset, suffix, tsv_label='y'):
    '''
    loads a list of feature files and merges them (or loads just one if
    featureset is a string).
    '''
    if isinstance(featureset, string_types):
        featureset = [featureset]

    # Load a list of lists of examples, one list of examples per featureset.
    file_names = [os.path.join(dirpath, featfile + suffix) for featfile
                  in featureset]
    example_tuples = [load_examples(file_name, tsv_label) for file_name in
                      file_names]

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
    unique_tuples = set(itertools.chain(*[[(curr_id, curr_label) for curr_id,
                                           curr_label in zip(examples.ids,
                                                             examples.classes)]
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

            merged_features = sp.hstack([merged_features, features], 'csr')
            num_merged = merged_features.shape[0]
            for feat_name, index in feat_vectorizer.vocabulary_.items():
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

    # Sort merged_features.feature_names_, because that happens whenever the
    # list is modified internally by DictVectorizer
    merged_vectorizer.feature_names_.sort()

    return ExamplesTuple(merged_ids, merged_classes, merged_features,
                         merged_vectorizer)


def _classify_featureset(jobname, featureset, given_learner, train_path,
                         test_path, train_set_name, test_set_name, modelpath,
                         prediction_prefix, grid_search,
                         grid_objective, do_scale_features, cross_validate,
                         evaluate, suffix, log_path, probability, resultspath,
                         fixed_parameters, param_grid, pos_label_str,
                         overwrite, use_dense_features, min_feature_count,
                         grid_search_jobs, cv_folds, tsv_label):
    ''' Classification job to be submitted to grid '''

    timestamp = datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')

    with open(log_path, 'w') as log_file:
        if cross_validate:
            print("Cross-validating on {}, feature set {} ...".format(
                train_set_name, featureset), file=log_file)
        else:
            print("Training on {}, Test on {}, feature set {} ...".format(
                train_set_name, test_set_name, featureset), file=log_file)

        # load the training and test examples
        train_examples = _load_featureset(train_path, featureset, suffix,
                                          tsv_label)
        if not cross_validate:
            test_examples = _load_featureset(test_path, featureset, suffix,
                                             tsv_label)

        # initialize a classifer object
        learner = Learner(probability=probability,
                          model_type=given_learner,
                          do_scale_features=do_scale_features,
                          model_kwargs=fixed_parameters,
                          pos_label_str=pos_label_str,
                          use_dense_features=use_dense_features,
                          min_feature_count=min_feature_count)

        # check whether a trained model on the same data with the same
        # featureset already exists if so, load it (and the feature
        # vocabulary) and then use it on the test data
        modelfile = os.path.join(modelpath, '{}.model'.format(jobname))

        # check if we're doing cross-validation, because we only load/save
        # models when we're not.
        if not cross_validate:

            # load the model if it already exists
            if os.path.exists(modelfile) and not overwrite:
                print('\tloading pre-existing {} model: {}'.format(
                    given_learner, modelfile))
                learner.load(modelfile)

            # if we have do not have a saved model, we need to train one.
            else:
                print('\tfeaturizing and training new {} model'.format(
                    given_learner), file=log_file)

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
                    print('\tbest {} score: {}'.format(grid_objective,
                                                       round(best_score, 3)),
                          file=log_file)

            # print out the tuned parameters and best CV score
            param_out = ('{}: {}'.format(param_name, param_value)
                         for param_name, param_value in
                         iteritems(learner.model.get_params()))
            print('\thyperparameters: {}'.format(', '.join(param_out)),
                  file=log_file)

        # run on test set or cross-validate on training data, depending on what
        # was asked for
        if cross_validate:
            print('\tcross-validating', file=log_file)
            task_results, grid_scores = learner.cross_validate(train_examples,
                                                               prediction_prefix=prediction_prefix,
                                                               grid_search=grid_search,
                                                               cv_folds=cv_folds,
                                                               grid_objective=grid_objective,
                                                               param_grid=param_grid,
                                                               grid_jobs=grid_search_jobs)
            task = 'cross-validate'
        elif evaluate:
            print('\tevaluating predictions', file=log_file)
            task_results = [learner.evaluate(
                test_examples, prediction_prefix=prediction_prefix,
                grid_objective=grid_objective)]
            task = 'evaluate'
        else:
            print('\twriting predictions', file=log_file)
            task = 'predict'
            task_results = None
            learner.predict(test_examples, prediction_prefix=prediction_prefix)

        results_json_path = os.path.join(resultspath,
                                         '{}.results.json'.format(jobname))

        # create a list of dictionaries of the results information
        learner_result_dicts = []
        learner_result_dict_base = {'train_set_name': train_set_name,
                                    'test_set_name': test_set_name,
                                    'featureset': featureset,
                                    'given_learner': given_learner,
                                    'task': task,
                                    'timestamp': timestamp,
                                    'scaling': do_scale_features,
                                    'grid_search': grid_search,
                                    'grid_objective': grid_objective}

        res = _create_learner_result_dicts(task_results, grid_scores,
                                           learner_result_dict_base)

        if task != 'predict':
            # write out the result dictionary to a json file
            with open(results_json_path, 'w') as json_file:
                json.dump(res, json_file)

            with open(os.path.join(resultspath, '{}.results'.format(jobname)),
                      'w') as output_file:
                _print_fancy_output(res, output_file)

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

        if num_folds == 1:
            learner_result_dict['fold'] = ""
        else:
            learner_result_dict['fold'] = k
            learner_result_dict['model_params'] = model_params
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
            learner_result_dict['result_table'] = '{}'.format(result_table)
            learner_result_dict['accuracy'] = fold_accuracy
            accuracy_sum += fold_accuracy

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


def _load_cv_folds(cv_folds_location):
    '''
    Loads CV folds from a CSV file with columns for example ID and fold ID
    (and a header).
    '''
    with open(cv_folds_location, 'rb') as f:
        reader = csv.reader(f)
        reader.next()  # discard the header
        res = {}
        for row in reader:
            res[row[0]] = row[1]

    return res


def run_configuration(config_file, local=False, overwrite=True, queue='all.q',
                      hosts=None):
    '''
    Takes a configuration file and runs the specified jobs on the grid.
    '''
    # Read configuration
    config = _parse_config_file(config_file)

    if not local:
        # import gridmap if available
        try:
            from gridmap import Job, process_jobs
        except ImportError:
            local = True
            logging.warning('gridmap not available. Forcing local mode.  ' +
                            'To run things on a DRMAA-compatible cluster, ' +
                            'install gridmap via pip.')

    # extract parameters from the config file
    if config.has_option("Input", "learners"):
        learners_string = config.get("Input", "learners")
    elif config.has_option("Input", "classifiers"):
        learners_string = config.get("Input", "classifiers")  # For old files
    else:
        raise ValueError("Configuration file does not contain list of " +
                         "learners in [Input] section.")
    given_learners = json.loads(_fix_json(learners_string))
    given_learners = [(_SHORT_NAMES[learner] if learner in _SHORT_NAMES else
                       learner) for learner in given_learners]
    given_featuresets = json.loads(_fix_json(config.get("Input",
                                                        "featuresets")))
    given_featureset_names = json.loads(_fix_json(config.get("Input",
                                                             "featureset_names")))
    fixed_parameter_list = json.loads(_fix_json(config.get("Input",
                                                           "fixed_parameters")))
    param_grid_list = json.loads(_fix_json(config.get("Tuning", "param_grids")))
    pos_label_str = config.get("Tuning", "pos_label_str")
    use_dense_features = config.getboolean("Tuning", "use_dense_features")

    # get all the input paths and directories (without trailing slashes)
    train_path = config.get("Input", "train_location").rstrip('/')
    test_path = config.get("Input", "test_location").rstrip('/')
    suffix = config.get("Input", "suffix")
    tsv_label = config.get("Input", "tsv_label")

    # get the cv folds file and make a dictionary from it
    cv_folds_location = config.get("Input", "cv_folds_location")
    if cv_folds_location:
        cv_folds = _load_cv_folds(cv_folds_location)
    else:
        cv_folds = 10

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

    # do we need to run a grid search for the hyperparameters or are we just
    # using the defaults
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

    # do we need to scale the feature values?
    do_scale_features = config.getboolean("Tuning", "scale_features")

    # are we doing cross validation or actual testing or just generating
    # predictions on a new test set? If no test set was specified then assume
    # that we are doing cross validation. If the results field was not
    # specified then assume that we are just generating predictions
    evaluate = False
    cross_validate = False
    predict = False
    if test_path and resultspath:
        evaluate = True
    elif not test_path:
        cross_validate = True
    else:
        predict = True

    if cross_validate:
        task = 'cross-validate'
    elif evaluate:
        task = 'evaluate'
    else:
        task = 'predict'

    # make sure that, if we are in prediction mode, we have a prediction_dir
    if predict and not prediction_dir:
        raise ValueError('You must specify a prediction directory if you are' +
                         ' using prediction mode (no "results" option in ' +
                         'config file).')

    # the list of jobs submitted (if running on grid)
    if not local:
        jobs = []

    if not given_featureset_names:
        given_featureset_names = [_munge_featureset_name(x) for x in
                                  given_featuresets]
    assert len(given_featureset_names) == len(given_featuresets)

    # store training/test set names for later use
    train_set_name = os.path.basename(train_path)
    test_set_name = os.path.basename(test_path) if test_path else "cv"

    # the list to hold the paths to all the result json files
    result_json_paths = []

    # the components for the names of the results and summary files
    base_name_components = [train_set_name, test_set_name]

    # add scaling information to name
    if do_scale_features:
        base_name_components.append('scaled')
    else:
        base_name_components.append('unscaled')

    # add tuning information to name
    if do_grid_search:
        base_name_components.append('tuned')
        base_name_components.append(grid_objective)
    else:
        base_name_components.append('untuned')

    # add task name
    base_name_components.append(task)

    # For each feature set
    for featureset, featureset_name in zip(given_featuresets,
                                           given_featureset_names):

        # and for each learner
        for learner_num, given_learner in enumerate(given_learners):

            job_name_components = base_name_components[:]

            # for the individual job name, we need to add the feature set name and the learner name
            job_name_components.extend([featureset_name, given_learner])
            jobname = '_'.join(job_name_components)

            # change the prediction prefix to include the feature set
            prediction_prefix = os.path.join(prediction_dir, jobname)

            # the log file that stores the actual output of this script (e.g.,
            # the tuned parameters, what kind of experiment was run, etc.)
            temp_logfile = os.path.join(logpath, '{}.log'.format(jobname))

            # create job if we're doing things on the grid
            job_args = [jobname, featureset, given_learner, train_path,
                        test_path, train_set_name, test_set_name, modelpath,
                        prediction_prefix, do_grid_search,
                        grid_objective, do_scale_features, cross_validate,
                        evaluate, suffix, temp_logfile, probability,
                        resultspath, (fixed_parameter_list[learner_num]
                                      if fixed_parameter_list else dict()),
                        (param_grid_list[learner_num] if param_grid_list
                         else None),
                        pos_label_str, overwrite, use_dense_features,
                        min_feature_count, grid_search_jobs, cv_folds,
                        tsv_label]
            if not local:
                jobs.append(Job(_classify_featureset, job_args,
                                num_slots=(MAX_CONCURRENT_PROCESSES if
                                           do_grid_search else 1),
                                name=jobname, queue=queue))
            else:
                _classify_featureset(*job_args)

            # save the path to the results json file that will be written
            result_json_paths.append(os.path.join(resultspath, '{}.results.json'.format(jobname)))

    # submit the jobs (if running on grid)
    if not local:
        if logpath:
            job_results = process_jobs(jobs, white_list=hosts, temp_dir=logpath)
        else:
            job_results = process_jobs(jobs, white_list=hosts)

        # Check for errors
        for result_dict in job_results:
            if 'task' not in result_dict:
                logging.error('There was an error running the experiment:\n' +
                              '{}'.format(result_dict))

    # write out the summary results file
    summary_file_name = '_'.join(base_name_components) + '_summary.tsv'
    with open(os.path.join(resultspath, summary_file_name), 'w') as output_file:
        _write_summary_file(result_json_paths, output_file)


def _run_experiment_without_feature(arg_tuple):
    '''
    Creates a new configuration file with a given feature
    removed and runs that experiment.

    :param arg_tuple: A tuple of the actual arguments for this function:
                      feature_type, given_features, config, local, queue
                      cfg_path, and machines.

                      - feature_type: The name of the feature set to exclude
                      - given_features: A list of all features in config
                      - given_featureset_name: The original featureset name
                      - config: A parsed configuration file
                      - local: Are we running things locally or on the grid?
                      - queue: Grid Map queue to use for scheduling
                      - cfg_path: Path to main configuration file
                      - machines: List of machines to use for scheduling jobs
                                  with Grid Map
                      - overwrite: Should we overwrite existing models?

    :type arg_tuple: tuple
    '''
    (feature_type, given_features, given_featureset_name, config, local, queue,
     cfg_path, machines, overwrite) = arg_tuple

    featureset = [[x for x in given_features if x != feature_type]]

    if feature_type:
        featureset_name = "{}_minus_{}".format(given_featureset_name,
                                               feature_type)
    else:
        featureset_name = "{}_all".format(given_featureset_name)

    config.set("Input", "featuresets", json.dumps(featureset))
    config.set("Input", "featureset_names", "['{}']".format(featureset_name))

    m = re.search(r'^(.*)\.cfg$', cfg_path)
    if not m:
        raise ValueError("Configuration file should end in .cfg.")
    new_cfg_path = "{}_minus_{}.cfg".format(m.groups()[0], feature_type) \
                   if feature_type else "{}_all.cfg".format(m.groups()[0])

    with open(new_cfg_path, 'w') as new_config_file:
        config.write(new_config_file)

    with open(new_cfg_path, 'r') as new_config_file:
        run_configuration(new_config_file, local=local, queue=queue,
                          hosts=machines, overwrite=overwrite)


def run_ablation(config_file, local=False, overwrite=True, queue='all.q',
                 hosts=None):
    '''
    Takes a configuration file and runs repeated experiments where each
    feature set has been removed from the configuration.
    '''
    # Read configuration
    config = _parse_config_file(config_file)

    given_featuresets = json.loads(_fix_json(config.get("Input",
                                                        "featuresets")))
    given_featureset_names = json.loads(_fix_json(config.get("Input",
                                                             "featureset_names")))

    # make sure there is only one list of features
    if ((isinstance(given_featuresets[0], list) and len(given_featuresets) > 1)
        or (isinstance(given_featureset_names[0], list)
            and len(given_featureset_names) > 1)):
        raise ValueError("More than one feature set or list of names given.")

    # make a list of features rather than a list of lists
    given_features = given_featuresets[0]
    given_featureset_name = given_featureset_names[0]

    # for each feature file, make a copy of the config file
    # with all but that feature, and run the jobs.
    arg_tuples = ((feature_type, given_features, given_featureset_name,
                   config, local, queue, config_file.name, hosts, overwrite)
                  for feature_type in given_features + [None])

    if local:
        for arg_tuple in arg_tuples:
            _run_experiment_without_feature(arg_tuple)
    else:
        pool = Pool(processes=len(given_features) + 1)
        pool.map(_run_experiment_without_feature, list(arg_tuples))
