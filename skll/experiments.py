# License: BSD 3 clause
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
from collections import defaultdict
from io import open
from itertools import chain, combinations

import configparser  # Backported version from Python 3
import numpy as np
import scipy.sparse as sp
from prettytable import PrettyTable, ALL
from six import string_types, iterkeys, iteritems  # Python 2/3
from six.moves import zip
from sklearn.metrics import SCORERS

from skll.data import ExamplesTuple, load_examples
from skll.learner import Learner, MAX_CONCURRENT_PROCESSES
from skll.version import __version__

import yaml

# Check if gridmap is available
try:
    from gridmap import Job, JobException, process_jobs
except ImportError:
    _HAVE_GRIDMAP = False
else:
    _HAVE_GRIDMAP = True


_VALID_TASKS = frozenset(['predict', 'train', 'evaluate', 'cross_validate'])
_VALID_SAMPLERS = frozenset(['Nystroem', 'RBFSampler', 'SkewedChi2Sampler',
                             'AdditiveChi2Sampler', ''])

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


def _write_summary_file(result_json_paths, output_file, ablation=0):
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
    logger = logging.getLogger(__name__)
    for json_path in result_json_paths:
        if not os.path.exists(json_path):
            logger.error(('JSON results file %s not found. Skipping summary '
                          'creation. You can manually create the summary file'
                          ' after the fact by using the summarize_results '
                          'script.'), json_path)
            return
        else:
            with open(json_path, 'r') as json_file:
                obj = json.load(json_file)
                if ablation != 0:
                    all_features.update(yaml.load(obj[0]['featureset']))
                learner_result_dicts.extend(obj)

    # Build and write header
    header = set(learner_result_dicts[0].keys()) - {'result_table',
                                                    'descriptive'}
    if ablation != 0:
        header.add('ablated_features')
    # Backward compatibility for older JSON results files.
    if 'comparative' in header:
        header.remove('comparative')
        header.add('pearson')
    header = sorted(header)
    writer = csv.DictWriter(output_file, header, extrasaction='ignore',
                            dialect=csv.excel_tab)
    writer.writeheader()

    # Build "ablated_features" list and fix some backward compatible things
    for lrd in learner_result_dicts:
        if ablation != 0:
            ablated_features = all_features.difference(
                yaml.load(lrd['featureset']))
            lrd['ablated_features'] = ''
            if ablated_features:
                lrd['ablated_features'] = json.dumps(sorted(ablated_features))
        # Backward compatibility for older JSON results files.
        if 'comparative' in lrd:
            lrd['pearson'] = lrd['comparative']['pearson']
            del lrd['comparative']

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
    print('SKLL Version: {}'.format(lrd['version']), file=output_file)
    print('Training Set: {}'.format(lrd['train_set_name']), file=output_file)
    print('Test Set: {}'.format(lrd['test_set_name']), file=output_file)
    print('Feature Set: {}'.format(lrd['featureset']), file=output_file)
    print('Learner: {}'.format(lrd['learner_name']), file=output_file)
    print('Task: {}'.format(lrd['task']), file=output_file)
    print('Feature Scaling: {}'.format(lrd['feature_scaling']),
          file=output_file)
    print('Grid Search: {}'.format(lrd['grid_search']), file=output_file)
    print('Grid Objective Function: {}'.format(lrd['grid_objective']),
          file=output_file)
    print('Using Folds File: {}'.format(isinstance(lrd['cv_folds'], dict)),
          file=output_file)
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


def _setup_config_parser(config_path):
    '''
    Returns a config parser at a given path. Only implemented as a separate
    function to simplify testing.
    '''
    # initialize config parser
    config = configparser.ConfigParser({'test_location': '',
                                        'log': '',
                                        'results': '',
                                        'predictions': '',
                                        'models': '',
                                        'sampler': '',
                                        'feature_hasher': 'False',
                                        'grid_search': 'False',
                                        'objective': "f1_score_micro",
                                        'probability': 'False',
                                        'fixed_parameters': '[]',
                                        'sampler_parameters': '[]',
                                        'param_grids': '[]',
                                        'pos_label_str': '',
                                        'featureset_names': '[]',
                                        'feature_scaling': 'none',
                                        'min_feature_count': '1',
                                        'grid_search_jobs': '0',
                                        'cv_folds_location': '',
                                        'suffix': '',
                                        'label_col': 'y',
                                        'ids_to_floats': 'False'})
    # Read file if it exists
    if not os.path.exists(config_path):
        raise IOError(errno.ENOENT, "The config file doesn't exist",
                      config_path)
    config.read(config_path)
    return config


def _parse_config_file(config_path):
    '''
    Parses a SKLL experiment configuration file with the given path.
    '''
    logger = logging.getLogger(__name__)
    config = _setup_config_parser(config_path)

    ###########################
    # extract parameters from the config file

    # General
    task = config.get("General", "task")
    if task not in _VALID_TASKS:
        raise ValueError('An invalid task was specified: {}. '.format(task) +
                         'Valid tasks are: {}'.format(' '.join(_VALID_TASKS)))

    experiment_name = config.get("General", "experiment_name")

    # Input
    sampler = config.get("Input", "sampler")
    if sampler not in _VALID_SAMPLERS:
        raise ValueError('An invalid sample was specified: {}. '.format(sampler) +
                         'Valid samplers are: {}'.format(', '.join(_VALID_SAMPLERS)))
    hasher_features = None
    feature_hasher = config.getboolean("Input", "feature_hasher")
    if feature_hasher:
        if config.has_option("Input", "hasher_features"):
            hasher_features = config.getint("Input", "hasher_features")
        else:
            raise ValueError("Configuration file does not contain" +
                             " option hasher_features, which is " +
                             "necessary when feature_hasher is True.")

    if config.has_option("Input", "learners"):
        learners_string = config.get("Input", "learners")
    elif config.has_option("Input", "classifiers"):
        learners_string = config.get("Input", "classifiers")  # For old files
    else:
        raise ValueError("Configuration file does not contain list of " +
                         "learners in [Input] section.")
    learners = yaml.load(_fix_json(learners_string))
    if len(set(learners)) < len(learners):
        raise ValueError('Configuration file containes the same learner '
                         'multiple times, which is not currently supported.  '
                         'Please use param_grids with tuning to find the '
                         'optimal settings for the learner.')
    for i, learner in enumerate(learners):
        if learner in _SHORT_NAMES:
            logger.warning(('Using short names like {} for learners is '
                            'deprecated and they will be removed in SKLL '
                            '1.0.  Please use the full name, {}, '
                            'instead.').format(learner, _SHORT_NAMES[learner]))
            learners[i] = _SHORT_NAMES[learner]
    featuresets = yaml.load(_fix_json(config.get("Input", "featuresets")))

    # ensure that featuresets is a list of lists
    if not isinstance(featuresets, list) or not all([isinstance(fs, list) for fs
                                                     in featuresets]):
        raise ValueError("The featuresets parameter should be a " +
                         "list of lists: {}".format(featuresets))

    featureset_names = yaml.load(_fix_json(config.get("Input",
                                                       "featureset_names")))

    # ensure that featureset_names is a list of strings, if specified
    if featureset_names:
        if (not isinstance(featureset_names, list) or
                not all([isinstance(fs, string_types) for fs in
                         featureset_names])):
            raise ValueError("The featureset_names parameter should be a " +
                             "list of strings: {}".format(featureset_names))

    fixed_parameter_list = yaml.load(_fix_json(config.get("Input",
                                                           "fixed_parameters")))
    fixed_sampler_parameters = yaml.load(_fix_json(config.get("Input",
                                                               "sampler_parameters")))
    param_grid_list = yaml.load(_fix_json(config.get("Tuning", "param_grids")))
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
    # Support tsv_label for old files
    if config.has_option("Input", "tsv_label"):
        label_col = config.get("Input", "tsv_label")
    else:
        label_col = config.get("Input", "label_col")
    ids_to_floats = config.getboolean("Input", "ids_to_floats")

    # get the cv folds file and make a dictionary from it
    cv_folds_location = config.get("Input", "cv_folds_location")
    if cv_folds_location:
        cv_folds = _load_cv_folds(cv_folds_location,
                                  ids_to_floats=ids_to_floats)
    else:
        cv_folds = 10

    # Get class mapping dictionary if specified
    if config.has_option("Input", "class_map"):
        orig_class_map = yaml.load(_fix_json(config.get("Input", "class_map")))
        # Change class_map to map from originals to replacements instead of from
        # replacement to list of originals
        class_map = {}
        for replacement, original_list in iteritems(orig_class_map):
            for original in original_list:
                class_map[original] = replacement
        del orig_class_map
    else:
        class_map = None

    # Output
    # get all the output files and directories
    results_path = config.get("Output", "results")
    log_path = config.get("Output", "log")
    model_path = config.get("Output", "models")
    probability = config.getboolean("Output", "probability")

    # do we want to keep the predictions?
    prediction_dir = config.get("Output", "predictions")
    if prediction_dir and not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # make sure log path exists
    if log_path and not os.path.exists(log_path):
        os.makedirs(log_path)

    # make sure results path exists
    if results_path and not os.path.exists(results_path):
        os.makedirs(results_path)

    # make sure all the specified paths exist
    if not os.path.exists(train_path):
        raise IOError(errno.ENOENT, ("The training path specified in config "
                                     "file does not exist"), train_path)
    if test_path and not os.path.exists(test_path):
        raise IOError(errno.ENOENT, ("The test path specified in config "
                                     "file does not exist"), test_path)

    # Tuning
    # do we need to run a grid search for the hyperparameters or are we just
    # using the defaults?
    do_grid_search = config.getboolean("Tuning", "grid_search")

    # minimum number of examples a feature must be nonzero in to be included
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
        raise ValueError('The test set path must be set when task is evaluate'
                         ' or predict.')
    if (task == 'cross_validate' or task == 'train') and test_path:
        raise ValueError('The test set path should not be set ' +
                         'when task is cross_validate or train.')
    if (task == 'train' or task == 'predict') and results_path:
        raise ValueError('The results path should not be set ' +
                         'when task is predict or train.')
    if task == 'train' and not model_path:
        raise ValueError('The model path should be set ' +
                         'when task is train.')
    if task == 'train' and prediction_dir:
        raise ValueError('The predictions path should not be set ' +
                         'when task is train.')
    if task == 'cross_validate' and model_path:
        raise ValueError('The models path should not be set ' +
                         'when task is cross_validate.')

    # Create feature set names if unspecified
    if not featureset_names:
        featureset_names = [_munge_featureset_name(x) for x in featuresets]
    assert len(featureset_names) == len(featuresets)

    # store training/test set names for later use
    train_set_name = os.path.basename(train_path)
    test_set_name = os.path.basename(test_path) if test_path else "cv"

    return (experiment_name, task, sampler, fixed_sampler_parameters,
            feature_hasher, hasher_features, label_col, train_set_name,
            test_set_name, suffix, featuresets, model_path, do_grid_search,
            grid_objective, probability, results_path, pos_label_str,
            feature_scaling, min_feature_count, grid_search_jobs, cv_folds,
            fixed_parameter_list, param_grid_list, featureset_names,
            learners, prediction_dir, log_path, train_path, test_path,
            ids_to_floats, class_map)


def _load_featureset(dirpath, featureset, suffix, label_col='y',
                     ids_to_floats=False, quiet=False, class_map=None,
                     unlabelled=False, feature_hasher=False,
                     num_features=None):
    '''
    Load a list of feature files and merge them.

    :param dirpath: Path to the directory that contains the feature files.
    :type dirpath: str
    :param featureset: List of feature file prefixes
    :type featureset: str
    :param suffix: Suffix to add to feature file prefixes to get full
                   filenames.
    :type suffix: str
    :param label_col: Name of the column which contains the class labels.
                      If no column with that name exists, or `None` is
                      specified, the data is considered to be unlabelled.
    :type label_col: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param class_map: Mapping from original class labels to new ones. This is
                      mainly used for collapsing multiple classes into a single
                      class. Anything not in the mapping will be kept the same.
    :type class_map: dict from str to str
    :param unlabelled: Is this test we're loading? If so, don't raise an error
                       if there are no labels.
    :type unlabelled: bool

    :returns: The classes, IDs, features, and feature vectorizer representing
              the given featureset.
    :rtype: ExamplesTuple
    '''

    # Load a list of lists of examples, one list of examples per featureset.
    file_names = sorted(os.path.join(dirpath, featfile + suffix) for featfile
                        in featureset)
    example_tuples = [load_examples(file_name, label_col=label_col,
                                    ids_to_floats=ids_to_floats, quiet=quiet,
                                    class_map=class_map,
                                    feature_hasher=feature_hasher,
                                    num_features=num_features)
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
                                for examples in example_tuples if
                                any(x is not None for x in examples.classes)]))
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
            if not feature_hasher:
                # Check for duplicate feature names
                if (set(merged_vectorizer.get_feature_names()) &
                        set(feat_vectorizer.get_feature_names())):
                    raise ValueError('Two feature files have the same '
                                     'feature!')
            num_merged = merged_features.shape[1]
            merged_features = sp.hstack([merged_features, features], 'csr')
            if not feature_hasher:
                # dictvectorizer sorts the vocabularies within each file
                for feat_name, index in sorted(feat_vectorizer.vocabulary_.items(),
                                               key=lambda x: x[1]):
                    merged_vectorizer.vocabulary_[feat_name] = (index +
                                                                num_merged)
                    merged_vectorizer.feature_names_.append(feat_name)
        else:
            merged_features = features
            merged_vectorizer = feat_vectorizer

        # IDs should be the same for each ExamplesTuple, so only store once
        if merged_ids is None:
            merged_ids = ids
        # Check that IDs are in the same order
        elif not np.all(merged_ids == ids):
            raise ValueError('IDs are not in the same order in each feature '
                             'file!')

        # If current ExamplesTuple has labels, check that they don't conflict
        if any(x is not None for x in classes):
            # Classes should be the same for each ExamplesTuple, so store once
            if merged_classes is None:
                merged_classes = classes
            # Check that classes don't conflict, when specified
            elif not np.all(merged_classes == classes):
                raise ValueError('Feature files have conflicting labels for '
                                 'examples with the same ID!')

    # Ensure that at least one file had classes if we're expecting them
    if merged_classes is None and not unlabelled:
        raise ValueError('No feature files in feature set contain class'
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
    sampler = args.pop("sampler")
    feature_hasher = args.pop("feature_hasher")
    hasher_features = args.pop("hasher_features")
    job_name = args.pop("job_name")
    featureset = args.pop("featureset")
    learner_name = args.pop("learner_name")
    train_path = args.pop("train_path")
    test_path = args.pop("test_path")
    train_set_name = args.pop("train_set_name")
    test_set_name = args.pop("test_set_name")
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
    cv_folds = args.pop("cv_folds")
    label_col = args.pop("label_col")
    ids_to_floats = args.pop("ids_to_floats")
    class_map = args.pop("class_map")
    quiet = args.pop('quiet', False)
    if args:
        raise ValueError(("Extra arguments passed to _classify_featureset: "
                          "{}").format(args.keys()))
    timestamp = datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')

    with open(log_path, 'w') as log_file:
        # logging
        print("Task:", task, file=log_file)
        if task == 'cross_validate':
            print(("Cross-validating on {}, feature " +
                   "set {} ...").format(train_set_name, featureset),
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
        modelfile = os.path.join(model_path, '{}.model'.format(job_name))
        if task == 'cross_validate' or (not os.path.exists(modelfile) or
                                        overwrite):
            train_examples = _load_featureset(train_path, featureset, suffix,
                                              label_col=label_col,
                                              ids_to_floats=ids_to_floats,
                                              quiet=quiet, class_map=class_map,
                                              feature_hasher=feature_hasher,
                                              num_features=hasher_features)
            # initialize a classifer object
            learner = Learner(learner_name,
                              probability=probability,
                              feature_scaling=feature_scaling,
                              model_kwargs=fixed_parameters,
                              pos_label_str=pos_label_str,
                              min_feature_count=min_feature_count,
                              sampler=sampler,
                              sampler_kwargs=sampler_parameters)
        # load the model if it already exists
        else:
            if os.path.exists(modelfile) and not overwrite:
                print(('\tloading pre-existing {} ' +
                       'model: {}').format(learner_name, modelfile))
            learner = Learner.from_file(modelfile)

        # Load test set if there is one
        if task == 'evaluate' or task == 'predict':
            test_examples = _load_featureset(test_path, featureset, suffix,
                                             label_col=label_col,
                                             ids_to_floats=ids_to_floats,
                                             quiet=quiet, class_map=class_map,
                                             unlabelled=True,
                                             feature_hasher=feature_hasher,
                                             num_features=hasher_features)

        # create a list of dictionaries of the results information
        learner_result_dict_base = {'experiment_name': experiment_name,
                                    'train_set_name': train_set_name,
                                    'test_set_name': test_set_name,
                                    'featureset': json.dumps(featureset),
                                    'learner_name': learner_name,
                                    'task': task,
                                    'timestamp': timestamp,
                                    'version': __version__,
                                    'feature_scaling': feature_scaling,
                                    'grid_search': grid_search,
                                    'grid_objective': grid_objective,
                                    'min_feature_count': min_feature_count,
                                    'cv_folds': cv_folds}

        # check if we're doing cross-validation, because we only load/save
        # models when we're not.
        task_results = None
        if task == 'cross_validate':
            print('\tcross-validating', file=log_file)
            task_results, grid_scores = learner.cross_validate(
                train_examples, prediction_prefix=prediction_prefix,
                grid_search=grid_search, cv_folds=cv_folds,
                grid_objective=grid_objective, param_grid=param_grid,
                grid_jobs=grid_search_jobs, feature_hasher=feature_hasher)
        else:
            # if we have do not have a saved model, we need to train one.
            if not os.path.exists(modelfile) or overwrite:
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
                                           grid_jobs=grid_search_jobs,
                                           feature_hasher=feature_hasher)
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
                    grid_objective=grid_objective,
                    feature_hasher=feature_hasher)]
            elif task == 'predict':
                print('\twriting predictions', file=log_file)
                learner.predict(test_examples,
                                prediction_prefix=prediction_prefix,
                                feature_hasher=feature_hasher)
            # do nothing here for train

        if task == 'cross_validate' or task == 'evaluate':
            results_json_path = os.path.join(results_path, '{}.results.json'
                                                           .format(job_name))

            res = _create_learner_result_dicts(task_results, grid_scores,
                                               learner_result_dict_base)

            # write out the result dictionary to a json file
            file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
            with open(results_json_path, file_mode) as json_file:
                json.dump(res, json_file)

            with open(os.path.join(results_path,
                                   '{}.results'.format(job_name)),
                      'w') as output_file:
                _print_fancy_output(res, output_file)
        else:
            res = [learner_result_dict_base]

    return res


def _create_learner_result_dicts(task_results, grid_scores,
                                 learner_result_dict_base):
    '''
    Create the learner result dictionaries that are used to create JSON and
    plain-text results files.
    '''
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
        else:
            learner_result_dict['pearson'] = pearson_sum / num_folds

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

    res = '+'.join(sorted(featureset))
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
                      hosts=None, write_summary=True, quiet=False,
                      ablation=0, resume=False):
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

    '''
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Read configuration
    (experiment_name, task, sampler, fixed_sampler_parameters, feature_hasher,
     hasher_features, label_col, train_set_name, test_set_name, suffix,
     featuresets, model_path, do_grid_search, grid_objective, probability,
     results_path, pos_label_str, feature_scaling, min_feature_count,
     grid_search_jobs, cv_folds, fixed_parameter_list, param_grid_list,
     featureset_names, learners, prediction_dir, log_path, train_path,
     test_path, ids_to_floats, class_map) = _parse_config_file(config_file)

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
                        expanded_fs_names.append(featureset_name + '_minus_' +
                                                 _munge_featureset_name(excluded_features))
            # Otherwise, just expand removing the specified number at a time
            else:
                for excluded_features in combinations(features, ablation):
                    expanded_fs.append(sorted(featureset -
                                              set(excluded_features)))
                    expanded_fs_names.append(featureset_name + '_minus_' +
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

    # Run each featureset-learner combination
    for featureset, featureset_name in zip(featuresets, featureset_names):
        for learner_num, learner_name in enumerate(learners):

            job_name_components = [experiment_name]

            # for the individual job name, we need to add the feature set name
            # and the learner name
            job_name_components.extend([featureset_name, learner_name])
            job_name = '_'.join(job_name_components)

            # change the prediction prefix to include the feature set
            prediction_prefix = os.path.join(prediction_dir, job_name)

            # the log file that stores the actual output of this script (e.g.,
            # the tuned parameters, what kind of experiment was run, etc.)
            temp_logfile = os.path.join(log_path, '{}.log'.format(job_name))

            # Figure out result json file path
            result_json_path = os.path.join(results_path,
                                            '{}.results.json'.format(job_name))

            # save the path to the results json file that will be written
            result_json_paths.append(result_json_path)

            # If result file already exists and we're resuming, move on
            if resume and (os.path.exists(result_json_path) and
                           os.path.getsize(result_json_path)):
                logger.info('Running in resume mode and %s exists, so '
                            'skipping job.', result_json_path)
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
            job_args["learner_name"] = learner_name
            job_args["train_path"] = train_path
            job_args["test_path"] = test_path
            job_args["train_set_name"] = train_set_name
            job_args["test_set_name"] = test_set_name
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
            job_args["cv_folds"] = cv_folds
            job_args["label_col"] = label_col
            job_args["ids_to_floats"] = ids_to_floats
            job_args["quiet"] = quiet
            job_args["class_map"] = class_map

            if not local:
                jobs.append(Job(_classify_featureset, [job_args],
                                num_slots=(MAX_CONCURRENT_PROCESSES if
                                           do_grid_search else 1),
                                name=job_name, queue=queue))
            else:
                _classify_featureset(job_args)

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
        with open(os.path.join(results_path, summary_file_name),
                  file_mode) as output_file:
            _write_summary_file(result_json_paths, output_file,
                                ablation=ablation)

    return result_json_paths


def _check_job_results(job_results):
    '''
    See if we have a complete results dictionary for every job.
    '''
    logger = logging.getLogger(__name__)
    logger.info('Checking job results')
    for result_dicts in job_results:
        if not result_dicts or 'task' not in result_dicts[0]:
            logger.error('There was an error running the experiment:\n%s',
                         result_dicts)


def run_ablation(config_path, local=False, overwrite=True, queue='all.q',
                 hosts=None, quiet=False, all_combos=False):
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
    :param quiet: Suppress printing of "Loading..." messages.
    :type quiet: bool
    :param all_combos: By default we only exclude one feature from set, but if
                       `all_combos` is `True`, we do a true ablation study with
                       all feature combinations.
    :type all_combos: bool

    .. deprecated:: 0.20.0
       Use :func:`run_configuration` with the ablation argument instead.

    '''
    if all_combos:
        run_configuration(config_path, local=local, overwrite=overwrite,
                          queue=queue, hosts=hosts, write_summary=True,
                          quiet=quiet, ablation=None)
    else:
        run_configuration(config_path, local=local, overwrite=overwrite,
                          queue=queue, hosts=hosts, write_summary=True,
                          quiet=quiet, ablation=1)
