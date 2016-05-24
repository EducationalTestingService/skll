# License: BSD 3 clause
"""
Functions related to running experiments and parsing configuration files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Chee Wee Leong (cleong@ets.org)
"""

from __future__ import absolute_import, print_function, unicode_literals

import csv
import datetime
import json
import logging
import math
import numpy as np
import os
import sys
from collections import defaultdict
from io import open
from itertools import combinations
from os.path import basename, exists, isfile, join

import ruamel.yaml as yaml

from prettytable import PrettyTable, ALL
from six import iterkeys, iteritems  # Python 2/3
from six.moves import zip
from sklearn import __version__ as SCIKIT_VERSION

from skll.config import _munge_featureset_name, _parse_config_file
from skll.data.readers import Reader
from skll.learner import (Learner, MAX_CONCURRENT_PROCESSES,
                          _import_custom_learner)
from skll.version import __version__


# Check if gridmap is available
try:
    from gridmap import Job, process_jobs
except ImportError:
    _HAVE_GRIDMAP = False
else:
    _HAVE_GRIDMAP = True


_VALID_TASKS = frozenset(['predict', 'train', 'evaluate', 'cross_validate'])
_VALID_SAMPLERS = frozenset(['Nystroem', 'RBFSampler', 'SkewedChi2Sampler',
                             'AdditiveChi2Sampler', ''])


class NumpyTypeEncoder(json.JSONEncoder):

    '''
    This class is used when serializing results, particularly the input label
    values if the input has int-valued labels.  Numpy int64 objects can't
    be serialized by the json module, so we must convert them to int objects.

    A related issue where this was adapted from:
    http://stackoverflow.com/questions/11561932/why-does-json-dumpslistnp-arange5-fail-while-json-dumpsnp-arange5-tolis
    '''

    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def _get_stat_float(label_result_dict, stat):
    """
    Little helper for getting output for precision, recall, and f-score
    columns in confusion matrix.

    :param label_result_dict: Dictionary containing the stat we'd like
                              to retrieve for a particular label.
    :type label_result_dict: dict
    :param stat: The statistic we're looking for in the dictionary.
    :type stat: str

    :return: The value of the stat if it's in the dictionary, and NaN
             otherwise.
    :rtype: float
    """
    if stat in label_result_dict and label_result_dict[stat] is not None:
        return label_result_dict[stat]
    else:
        return float('nan')


def _write_skll_folds(skll_fold_ids, skll_fold_ids_file):
    """
    Function to take a dictionary of id:test-fold-number and
    write it to a file

    :param skll_fold_ids: the dictionary of id: test-fold-numbers
    :param skll_fold_ids_file: the open file handle to write to
    :return: None
    """

    f = csv.writer(skll_fold_ids_file)
    f.writerow(['id', 'cv_test_fold'])
    for example_id in skll_fold_ids:
        f.writerow([example_id, skll_fold_ids[example_id]])

    skll_fold_ids_file.flush()


def _write_summary_file(result_json_paths, output_file, ablation=0):
    """
    Function to take a list of paths to individual result
    json files and returns a single file that summarizes
    all of them.

    :param result_json_paths: A list of paths to the
                              individual result json files.
    :type result_json_paths: list

    :returns: The output file to contain a summary of the individual result
              files.
    :rtype: file
    """
    learner_result_dicts = []
    # Map from feature set names to all features in them
    all_features = defaultdict(set)
    logger = logging.getLogger(__name__)
    for json_path in result_json_paths:
        if not exists(json_path):
            logger.error(('JSON results file %s not found. Skipping summary '
                          'creation. You can manually create the summary file'
                          ' after the fact by using the summarize_results '
                          'script.'), json_path)
            return
        else:
            with open(json_path, 'r') as json_file:
                obj = json.load(json_file)
                featureset_name = obj[0]['featureset_name']
                if ablation != 0 and '_minus_' in featureset_name:
                    parent_set = featureset_name.split('_minus_', 1)[0]
                    all_features[parent_set].update(
                        yaml.load(obj[0]['featureset']))
                learner_result_dicts.extend(obj)

    # Build and write header
    header = set(learner_result_dicts[0].keys()) - {'result_table',
                                                    'descriptive'}
    if ablation != 0:
        header.add('ablated_features')
    header = sorted(header)
    writer = csv.DictWriter(output_file, header, extrasaction='ignore',
                            dialect=csv.excel_tab)
    writer.writeheader()

    # Build "ablated_features" list and fix some backward compatible things
    for lrd in learner_result_dicts:
        featureset_name = lrd['featureset_name']
        if ablation != 0:
            parent_set = featureset_name.split('_minus_', 1)[0]
            ablated_features = all_features[parent_set].difference(
                yaml.load(lrd['featureset']))
            lrd['ablated_features'] = ''
            if ablated_features:
                lrd['ablated_features'] = json.dumps(sorted(ablated_features))

        # write out the new learner dict with the readable fields
        writer.writerow(lrd)

    output_file.flush()


def _print_fancy_output(learner_result_dicts, output_file=sys.stdout):
    """
    Function to take all of the results from all of the folds and print
    nice tables with the results.
    """
    if not learner_result_dicts:
        raise ValueError('Result dictionary list is empty!')

    lrd = learner_result_dicts[0]
    print('Experiment Name: {}'.format(lrd['experiment_name']),
          file=output_file)
    print('SKLL Version: {}'.format(lrd['version']), file=output_file)
    print('Training Set: {}'.format(lrd['train_set_name']), file=output_file)
    print('Training Set Size: {}'.format(
        lrd['train_set_size']), file=output_file)
    print('Test Set: {}'.format(lrd['test_set_name']), file=output_file)
    print('Test Set Size: {}'.format(lrd['test_set_size']), file=output_file)
    print('Shuffle: {}'.format(lrd['shuffle']), file=output_file)
    print('Feature Set: {}'.format(lrd['featureset']), file=output_file)
    print('Learner: {}'.format(lrd['learner_name']), file=output_file)
    print('Task: {}'.format(lrd['task']), file=output_file)
    if lrd['task'] == 'cross_validate':
        print('Number of Folds: {}'.format(lrd['cv_folds']),
              file=output_file)
        print('Stratified Folds: {}'.format(lrd['stratified_folds']),
              file=output_file)
    print('Feature Scaling: {}'.format(lrd['feature_scaling']),
          file=output_file)
    print('Grid Search: {}'.format(lrd['grid_search']), file=output_file)
    print('Grid Search Folds: {}'.format(lrd['grid_search_folds']),
          file=output_file)
    print('Grid Objective Function: {}'.format(lrd['grid_objective']),
          file=output_file)
    print('Using Folds File: {}'.format(isinstance(lrd['cv_folds'], dict)),
          file=output_file)
    print('Scikit-learn Version: {}'.format(lrd['scikit_learn_version']),
          file=output_file)
    print('Start Timestamp: {}'.format(
        lrd['start_timestamp']), file=output_file)
    print('End Timestamp: {}'.format(lrd['end_timestamp']), file=output_file)
    print('Total Time: {}'.format(lrd['total_time']), file=output_file)
    print('\n', file=output_file)

    for lrd in learner_result_dicts:
        print('Fold: {}'.format(lrd['fold']), file=output_file)
        print('Model Parameters: {}'.format(lrd.get('model_params', '')),
              file=output_file)
        print('Grid Objective Score (Train) = {}'.format(lrd.get('grid_score',
                                                                 '')),
              file=output_file)
        if 'result_table' in lrd:
            print(lrd['result_table'], file=output_file)
            print('Accuracy = {}'.format(lrd['accuracy']),
                  file=output_file)
        if 'descriptive' in lrd:
            print('Descriptive statistics:', file=output_file)
            for desc_stat in ['min', 'max', 'avg', 'std']:
                actual = lrd['descriptive']['actual'][desc_stat]
                predicted = lrd['descriptive']['predicted'][desc_stat]
                print((' {} = {: .4f} (actual), {: .4f} '
                       '(predicted)').format(desc_stat.title(), actual,
                                             predicted),
                      file=output_file)
            print('Pearson = {: f}'.format(lrd['pearson']),
                  file=output_file)
        print('Objective Function Score (Test) = {}'.format(lrd['score']),
              file=output_file)
        print('', file=output_file)


def _load_featureset(dir_path, feat_files, suffix, id_col='id', label_col='y',
                     ids_to_floats=False, quiet=False, class_map=None,
                     feature_hasher=False, num_features=None):
    """
    Load a list of feature files and merge them.

    :param dir_path: Path to the directory that contains the feature files.
    :type dir_path: str
    :param feat_files: List of feature file prefixes
    :type feat_files: str
    :param suffix: Suffix to add to feature file prefixes to get full
                   filenames.
    :type suffix: str
    :param label_col: Name of the column which contains the class labels.
                      If no column with that name exists, or `None` is
                      specified, the data is considered to be unlabelled.
    :type label_col: str
    :param id_col: Name of the column which contains the instance IDs.
                   If no column with that name exists, or `None` is
                   specified, example IDs will be automatically generated.
    :type id_col: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param class_map: Mapping from original class labels to new ones. This is
                      mainly used for collapsing multiple labels into a single
                      class. Anything not in the mapping will be kept the same.
    :type class_map: dict from str to str
    :param feature_hasher: Should we use a FeatureHasher when vectorizing
                           features?
    :type feature_hasher: bool
    :param num_features: The number of features to use with the FeatureHasher.
                         This should always be set to the power of 2 greater
                         than the actual number of features you're using.
    :type num_features: int

    :returns: The labels, IDs, features, and feature vectorizer representing
              the given featureset.
    :rtype: FeatureSet
    """
    # if the training file is specified via train_file, then dir_path
    # actually contains the entire file name
    if isfile(dir_path):
        return Reader.for_path(dir_path, label_col=label_col, id_col=id_col,
                               ids_to_floats=ids_to_floats, quiet=quiet,
                               class_map=class_map,
                               feature_hasher=feature_hasher,
                               num_features=num_features).read()
    else:
        merged_set = None
        for file_name in sorted(join(dir_path, featfile + suffix) for
                                featfile in feat_files):
            fs = Reader.for_path(file_name, label_col=label_col, id_col=id_col,
                                 ids_to_floats=ids_to_floats, quiet=quiet,
                                 class_map=class_map,
                                 feature_hasher=feature_hasher,
                                 num_features=num_features).read()
            if merged_set is None:
                merged_set = fs
            else:
                merged_set += fs
        return merged_set


def _classify_featureset(args):
    """ Classification job to be submitted to grid """
    # Extract all the arguments.
    # (There doesn't seem to be a better way to do this since one can't specify
    # required keyword arguments.)
    experiment_name = args.pop("experiment_name")
    task = args.pop("task")
    sampler = args.pop("sampler")
    feature_hasher = args.pop("feature_hasher")
    hasher_features = args.pop("hasher_features")
    job_name = args.pop("job_name")
    featureset = args.pop("featureset")
    featureset_name = args.pop("featureset_name")
    learner_name = args.pop("learner_name")
    train_path = args.pop("train_path")
    test_path = args.pop("test_path")
    train_set_name = args.pop("train_set_name")
    test_set_name = args.pop("test_set_name")
    shuffle = args.pop('shuffle')
    model_path = args.pop("model_path")
    prediction_prefix = args.pop("prediction_prefix")
    grid_search = args.pop("grid_search")
    grid_objective = args.pop("grid_objective")
    suffix = args.pop("suffix")
    log_path = args.pop("log_path")
    probability = args.pop("probability")
    results_path = args.pop("results_path")
    fixed_parameters = args.pop("fixed_parameters")
    sampler_parameters = args.pop("sampler_parameters")
    param_grid = args.pop("param_grid")
    pos_label_str = args.pop("pos_label_str")
    overwrite = args.pop("overwrite")
    feature_scaling = args.pop("feature_scaling")
    min_feature_count = args.pop("min_feature_count")
    grid_search_jobs = args.pop("grid_search_jobs")
    grid_search_folds = args.pop("grid_search_folds")
    cv_folds = args.pop("cv_folds")
    save_cv_folds = args.pop("save_cv_folds")
    stratified_folds = args.pop("do_stratified_folds")
    label_col = args.pop("label_col")
    id_col = args.pop("id_col")
    ids_to_floats = args.pop("ids_to_floats")
    class_map = args.pop("class_map")
    custom_learner_path = args.pop("custom_learner_path")
    quiet = args.pop('quiet', False)

    if args:
        raise ValueError(("Extra arguments passed to _classify_featureset: "
                          "{}").format(args.keys()))
    start_timestamp = datetime.datetime.now()

    with open(log_path, 'w') as log_file:
        # logging
        print("Task:", task, file=log_file)
        if task == 'cross_validate':
            print(("Cross-validating ({} folds) on {}, feature " +
                   "set {} ...").format(cv_folds, train_set_name, featureset),
                  file=log_file)
        elif task == 'evaluate':
            print(("Training on {}, Test on {}, " +
                   "feature set {} ...").format(train_set_name, test_set_name,
                                                featureset),
                  file=log_file)
        elif task == 'train':
            print("Training on {}, feature set {} ...".format(train_set_name,
                                                              featureset),
                  file=log_file)
        else:  # predict
            print(("Training on {}, Making predictions about {}, " +
                   "feature set {} ...").format(train_set_name, test_set_name,
                                                featureset),
                  file=log_file)

        # check whether a trained model on the same data with the same
        # featureset already exists if so, load it and then use it on test data
        modelfile = join(model_path, '{}.model'.format(job_name))
        if task == 'cross_validate' or (not exists(modelfile) or
                                        overwrite):
            train_examples = _load_featureset(train_path, featureset, suffix,
                                              label_col=label_col,
                                              id_col=id_col,
                                              ids_to_floats=ids_to_floats,
                                              quiet=quiet, class_map=class_map,
                                              feature_hasher=feature_hasher,
                                              num_features=hasher_features)

            train_set_size = len(train_examples.ids)
            if not train_examples.has_labels:
                raise ValueError('Training examples do not have labels')
            # initialize a classifer object
            learner = Learner(learner_name,
                              probability=probability,
                              feature_scaling=feature_scaling,
                              model_kwargs=fixed_parameters,
                              pos_label_str=pos_label_str,
                              min_feature_count=min_feature_count,
                              sampler=sampler,
                              sampler_kwargs=sampler_parameters,
                              custom_learner_path=custom_learner_path)
        # load the model if it already exists
        else:
            # import the custom learner path here in case we are reusing a
            # saved model
            if custom_learner_path:
                _import_custom_learner(custom_learner_path, learner_name)
            train_set_size = 'unknown'
            if exists(modelfile) and not overwrite:
                print(('\tloading pre-existing %s model: %s') % (learner_name,
                                                                 modelfile))
            learner = Learner.from_file(modelfile)

        # Load test set if there is one
        if task == 'evaluate' or task == 'predict':
            test_examples = _load_featureset(test_path, featureset, suffix,
                                             label_col=label_col,
                                             id_col=id_col,
                                             ids_to_floats=ids_to_floats,
                                             quiet=quiet, class_map=class_map,
                                             feature_hasher=feature_hasher,
                                             num_features=hasher_features)
            test_set_size = len(test_examples.ids)
        else:
            test_set_size = 'n/a'

        # create a list of dictionaries of the results information
        learner_result_dict_base = {'experiment_name': experiment_name,
                                    'train_set_name': train_set_name,
                                    'train_set_size': train_set_size,
                                    'test_set_name': test_set_name,
                                    'test_set_size': test_set_size,
                                    'featureset': json.dumps(featureset),
                                    'featureset_name': featureset_name,
                                    'shuffle': shuffle,
                                    'learner_name': learner_name,
                                    'task': task,
                                    'start_timestamp':
                                    start_timestamp.strftime('%d %b %Y %H:%M:'
                                                             '%S.%f'),
                                    'version': __version__,
                                    'feature_scaling': feature_scaling,
                                    'grid_search': grid_search,
                                    'grid_objective': grid_objective,
                                    'grid_search_folds': grid_search_folds,
                                    'min_feature_count': min_feature_count,
                                    'cv_folds': cv_folds,
                                    'save_cv_folds': save_cv_folds,
                                    'stratified_folds': stratified_folds,
                                    'scikit_learn_version': SCIKIT_VERSION}

        # check if we're doing cross-validation, because we only load/save
        # models when we're not.
        task_results = None
        if task == 'cross_validate':
            print('\tcross-validating', file=log_file)
            task_results, grid_scores, skll_fold_ids = learner.cross_validate(
                train_examples, shuffle=shuffle, stratified=stratified_folds,
                prediction_prefix=prediction_prefix, grid_search=grid_search,
                grid_search_folds=grid_search_folds, cv_folds=cv_folds,
                grid_objective=grid_objective, param_grid=param_grid,
                grid_jobs=grid_search_jobs, save_cv_folds=save_cv_folds)
        else:
            # if we have do not have a saved model, we need to train one.
            if not exists(modelfile) or overwrite:
                print(('\tfeaturizing and training new ' +
                       '{} model').format(learner_name),
                      file=log_file)

                if not isinstance(cv_folds, int):
                    grid_search_folds = cv_folds

                best_score = learner.train(train_examples,
                                           shuffle=shuffle,
                                           grid_search=grid_search,
                                           grid_search_folds=grid_search_folds,
                                           grid_objective=grid_objective,
                                           param_grid=param_grid,
                                           grid_jobs=grid_search_jobs)
                grid_scores = [best_score]

                # save model
                if model_path:
                    learner.save(modelfile)

                if grid_search:
                    # note: bankers' rounding is used in python 3,
                    # so these scores may be different between runs in
                    # python 2 and 3 at the final decimal place.
                    print('\tbest {} grid search score: {}'
                          .format(grid_objective, round(best_score, 3)),
                          file=log_file)
            else:
                grid_scores = [None]

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
            # do nothing here for train

        end_timestamp = datetime.datetime.now()
        learner_result_dict_base['end_timestamp'] = end_timestamp.strftime(
            '%d %b %Y %H:%M:%S.%f')
        total_time = end_timestamp - start_timestamp
        learner_result_dict_base['total_time'] = str(total_time)

        if task == 'cross_validate' or task == 'evaluate':
            results_json_path = join(results_path,
                                     '{}.results.json'.format(job_name))

            res = _create_learner_result_dicts(task_results, grid_scores,
                                               learner_result_dict_base)

            # write out the result dictionary to a json file
            file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
            with open(results_json_path, file_mode) as json_file:
                json.dump(res, json_file, cls=NumpyTypeEncoder)

            with open(join(results_path,
                           '{}.results'.format(job_name)),
                      'w') as output_file:
                _print_fancy_output(res, output_file)
        else:
            res = [learner_result_dict_base]

        # write out the cv folds if required
        if task == 'cross_validate' and save_cv_folds:
            skll_fold_ids_file = experiment_name + '_skll_fold_ids.csv'
            file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
            with open(join(results_path, skll_fold_ids_file),
                      file_mode) as output_file:
                _write_skll_folds(skll_fold_ids, output_file)

    return res


def _create_learner_result_dicts(task_results, grid_scores,
                                 learner_result_dict_base):
    """
    Create the learner result dictionaries that are used to create JSON and
    plain-text results files.
    """
    res = []

    num_folds = len(task_results)
    accuracy_sum = 0.0
    pearson_sum = 0.0
    score_sum = None
    prec_sum_dict = defaultdict(float)
    recall_sum_dict = defaultdict(float)
    f_sum_dict = defaultdict(float)
    result_table = None

    for k, ((conf_matrix, fold_accuracy, result_dict, model_params,
             score), grid_score) in enumerate(zip(task_results, grid_scores),
                                              start=1):

        # create a new dict for this fold
        learner_result_dict = {}
        learner_result_dict.update(learner_result_dict_base)

        # initialize some variables to blanks so that the
        # set of columns is fixed.
        learner_result_dict['result_table'] = ''
        learner_result_dict['accuracy'] = ''
        learner_result_dict['pearson'] = ''
        learner_result_dict['score'] = ''
        learner_result_dict['fold'] = ''

        if learner_result_dict_base['task'] == 'cross_validate':
            learner_result_dict['fold'] = k

        learner_result_dict['model_params'] = json.dumps(model_params)
        if grid_score is not None:
            learner_result_dict['grid_score'] = grid_score

        if conf_matrix:
            labels = sorted(iterkeys(task_results[0][2]))
            result_table = PrettyTable([""] + labels + ["Precision", "Recall",
                                                        "F-measure"],
                                       header=True, hrules=ALL)
            result_table.align = 'r'
            result_table.float_format = '.3'
            for i, actual_label in enumerate(labels):
                conf_matrix[i][i] = "[{}]".format(conf_matrix[i][i])
                label_prec = _get_stat_float(result_dict[actual_label],
                                             "Precision")
                label_recall = _get_stat_float(result_dict[actual_label],
                                               "Recall")
                label_f = _get_stat_float(result_dict[actual_label],
                                          "F-measure")
                if not math.isnan(label_prec):
                    prec_sum_dict[actual_label] += float(label_prec)
                if not math.isnan(label_recall):
                    recall_sum_dict[actual_label] += float(label_recall)
                if not math.isnan(label_f):
                    f_sum_dict[actual_label] += float(label_f)
                result_row = ([actual_label] + conf_matrix[i] +
                              [label_prec, label_recall, label_f])
                result_table.add_row(result_row)

            result_table_str = '{}'.format(result_table)
            result_table_str += '\n(row = reference; column = predicted)'
            learner_result_dict['result_table'] = result_table_str
            learner_result_dict['accuracy'] = fold_accuracy
            accuracy_sum += fold_accuracy

        # if there is no confusion matrix, then we must be dealing
        # with a regression model
        else:
            learner_result_dict.update(result_dict)
            pearson_sum += float(learner_result_dict['pearson'])

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
            result_table = PrettyTable(["Label", "Precision", "Recall",
                                        "F-measure"],
                                       header=True)
            result_table.align = "r"
            result_table.align["Label"] = "l"
            result_table.float_format = '.3'
            for actual_label in labels:
                # Convert sums to means
                prec_mean = prec_sum_dict[actual_label] / num_folds
                recall_mean = recall_sum_dict[actual_label] / num_folds
                f_mean = f_sum_dict[actual_label] / num_folds
                result_table.add_row([actual_label] +
                                     [prec_mean, recall_mean, f_mean])

            learner_result_dict['result_table'] = '{}'.format(result_table)
            learner_result_dict['accuracy'] = accuracy_sum / num_folds
        else:
            learner_result_dict['pearson'] = pearson_sum / num_folds

        if score_sum is not None:
            learner_result_dict['score'] = score_sum / num_folds
        res.append(learner_result_dict)
    return res


def run_configuration(config_file, local=False, overwrite=True, queue='all.q',
                      hosts=None, write_summary=True, quiet=False,
                      ablation=0, resume=False):
    """
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
    :param quiet: Suppress printing of "Loading..." messages.
    :type quiet: bool
    :param ablation: Number of features to remove when doing an ablation
                     experiment. If positive, we will perform repeated ablation
                     runs for all combinations of features removing the
                     specified number at a time. If ``None``, we will use all
                     combinations of all lengths. If 0, the default, no
                     ablation is performed. If negative, a ``ValueError`` is
                     raised.
    :type ablation: int or None
    :param resume: If result files already exist for an experiment, do not
                   overwrite them. This is very useful when doing a large
                   ablation experiment and part of it crashes.
    :type resume: bool

    :return: A list of paths to .json results files for each variation in the
             experiment.
    :rtype: list of str

    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Read configuration
    (experiment_name, task, sampler, fixed_sampler_parameters, feature_hasher,
     hasher_features, id_col, label_col, train_set_name, test_set_name, suffix,
     featuresets, do_shuffle, model_path, do_grid_search, grid_objectives,
     probability, results_path, pos_label_str, feature_scaling,
     min_feature_count, grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     do_stratified_folds, fixed_parameter_list, param_grid_list, featureset_names,
     learners, prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path) = _parse_config_file(config_file)

    # Check if we have gridmap
    if not local and not _HAVE_GRIDMAP:
        local = True
        logger.warning('gridmap 0.10.1+ not available. Forcing local '
                       'mode.  To run things on a DRMAA-compatible '
                       'cluster, install gridmap>=0.10.1 via pip.')

    # if performing ablation, expand featuresets to include combinations of
    # features within those sets
    if ablation is None or ablation > 0:
        # Make new feature set lists so that we can iterate without issue
        expanded_fs = []
        expanded_fs_names = []
        for features, featureset_name in zip(featuresets, featureset_names):
            features = sorted(features)
            featureset = set(features)
            # Expand to all feature combinations if ablation is None
            if ablation is None:
                for i in range(1, len(features)):
                    for excluded_features in combinations(features, i):
                        expanded_fs.append(sorted(featureset -
                                                  set(excluded_features)))
                        expanded_fs_names.append(
                            featureset_name +
                            '_minus_' +
                            _munge_featureset_name(excluded_features))
            # Otherwise, just expand removing the specified number at a time
            else:
                for excluded_features in combinations(features, ablation):
                    expanded_fs.append(sorted(featureset -
                                              set(excluded_features)))
                    expanded_fs_names.append(
                        featureset_name +
                        '_minus_' +
                        _munge_featureset_name(excluded_features))
            # Also add version with nothing removed as baseline
            expanded_fs.append(features)
            expanded_fs_names.append(featureset_name + '_all')

        # Replace original feature set lists
        featuresets = expanded_fs
        featureset_names = expanded_fs_names
    elif ablation < 0:
        raise ValueError('Value for "ablation" argument must be either '
                         'positive integer or None.')

    # the list of jobs submitted (if running on grid)
    if not local:
        jobs = []

    # the list to hold the paths to all the result json files
    result_json_paths = []

    # check if the length of the featureset_name exceeds the maximum length
    # allowed
    for featureset_name in featureset_names:
        if len(featureset_name) > 210:
            raise OSError('System generated file length "{}" exceeds the '
                          'maximum length supported.  Please specify names of '
                          'your datasets with "featureset_names".  If you are '
                          'running ablation experiment, please reduce the '
                          'length of the features in "featuresets" because the'
                          ' auto-generated name would be longer than the file '
                          'system can handle'.format(featureset_name))

    # Run each featureset-learner combination
    for featureset, featureset_name in zip(featuresets, featureset_names):
        for learner_num, learner_name in enumerate(learners):
            for grid_objective in grid_objectives:

                # for the individual job name, we need to add the feature set name
                # and the learner name
                if len(grid_objectives) == 1:
                    job_name_components = [experiment_name, featureset_name,
                                           learner_name]
                else:
                    job_name_components = [experiment_name, featureset_name,
                                           learner_name, grid_objective]

                job_name = '_'.join(job_name_components)

                # change the prediction prefix to include the feature set
                prediction_prefix = join(prediction_dir, job_name)

                # the log file that stores the actual output of this script (e.g.,
                # the tuned parameters, what kind of experiment was run, etc.)
                temp_logfile = join(log_path, '{}.log'.format(job_name))

                # Figure out result json file path
                result_json_path = join(results_path,
                                        '{}.results.json'.format(job_name))

                # save the path to the results json file that will be written
                result_json_paths.append(result_json_path)

                # If result file already exists and we're resuming, move on
                if resume and (exists(result_json_path) and
                               os.path.getsize(result_json_path)):
                    logger.info('Running in resume mode and %s exists, '
                                'so skipping job.', result_json_path)
                    continue

                # create job if we're doing things on the grid
                job_args = {}
                job_args["experiment_name"] = experiment_name
                job_args["task"] = task
                job_args["sampler"] = sampler
                job_args["feature_hasher"] = feature_hasher
                job_args["hasher_features"] = hasher_features
                job_args["job_name"] = job_name
                job_args["featureset"] = featureset
                job_args["featureset_name"] = featureset_name
                job_args["learner_name"] = learner_name
                job_args["train_path"] = train_path
                job_args["test_path"] = test_path
                job_args["train_set_name"] = train_set_name
                job_args["test_set_name"] = test_set_name
                job_args["shuffle"] = do_shuffle
                job_args["model_path"] = model_path
                job_args["prediction_prefix"] = prediction_prefix
                job_args["grid_search"] = do_grid_search
                job_args["grid_objective"] = grid_objective
                job_args["suffix"] = suffix
                job_args["log_path"] = temp_logfile
                job_args["probability"] = probability
                job_args["results_path"] = results_path
                job_args["sampler_parameters"] = (fixed_sampler_parameters
                                                  if fixed_sampler_parameters
                                                  else dict())
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
                job_args["grid_search_folds"] = grid_search_folds
                job_args["cv_folds"] = cv_folds
                job_args["save_cv_folds"] = save_cv_folds
                job_args["do_stratified_folds"] = do_stratified_folds
                job_args["label_col"] = label_col
                job_args["id_col"] = id_col
                job_args["ids_to_floats"] = ids_to_floats
                job_args["quiet"] = quiet
                job_args["class_map"] = class_map
                job_args["custom_learner_path"] = custom_learner_path

                if not local:
                    jobs.append(Job(_classify_featureset, [job_args],
                                    num_slots=(MAX_CONCURRENT_PROCESSES if
                                               do_grid_search else 1),
                                    name=job_name, queue=queue))
                else:
                    _classify_featureset(job_args)
    test_set_name = basename(test_path)

    # submit the jobs (if running on grid)
    if not local and _HAVE_GRIDMAP:
        if log_path:
            job_results = process_jobs(jobs, white_list=hosts,
                                       temp_dir=log_path)
        else:
            job_results = process_jobs(jobs, white_list=hosts)
        _check_job_results(job_results)

    # write out the summary results file
    if (task == 'cross_validate' or task == 'evaluate') and write_summary:
        summary_file_name = experiment_name + '_summary.tsv'
        file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
        with open(join(results_path, summary_file_name),
                  file_mode) as output_file:
            _write_summary_file(result_json_paths, output_file,
                                ablation=ablation)

    return result_json_paths


def _check_job_results(job_results):
    """
    See if we have a complete results dictionary for every job.
    """
    logger = logging.getLogger(__name__)
    logger.info('Checking job results')
    for result_dicts in job_results:
        if not result_dicts or 'task' not in result_dicts[0]:
            logger.error('There was an error running the experiment:\n%s',
                         result_dicts)
