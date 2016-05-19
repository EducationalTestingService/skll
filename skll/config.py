# License: BSD 3 clause
"""
Functions related to parsing configuration files.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Chee Wee Leong (cleong@ets.org)
"""

from __future__ import absolute_import, print_function, unicode_literals

import csv
import errno
import itertools
import logging
import os
from io import open
from os.path import (basename, dirname, exists,
                     isabs, join, normpath, realpath)

import configparser  # Backported version from Python 3
import ruamel.yaml as yaml

from six import string_types, iteritems  # Python 2/3
from sklearn.metrics import SCORERS


_VALID_TASKS = frozenset(['predict', 'train', 'evaluate', 'cross_validate'])
_VALID_SAMPLERS = frozenset(['Nystroem', 'RBFSampler', 'SkewedChi2Sampler',
                             'AdditiveChi2Sampler', ''])
_VALID_FEATURE_SCALING_OPTIONS = frozenset(['with_std', 'with_mean', 'both',
                                            'none'])


class SKLLConfigParser(configparser.ConfigParser):

    """A custom configuration file parser for SKLL"""

    def __init__(self):

        # these are the three options that must be set in a config
        # file and no defaults are provided
        required = ['experiment_name', 'task', 'learners']

        # these are the optional config options for which#
        # defaults are automatically provided
        defaults = {'class_map': '{}',
                    'custom_learner_path': '',
                    'cv_folds_file': '',
                    'feature_hasher': 'False',
                    'feature_scaling': 'none',
                    'featuresets': '[]',
                    'featureset_names': '[]',
                    'fixed_parameters': '[]',
                    'grid_search': 'False',
                    'grid_search_folds': '3',
                    'grid_search_jobs': '0',
                    'hasher_features': '0',
                    'id_col': 'id',
                    'ids_to_floats': 'False',
                    'label_col': 'y',
                    'log': '',
                    'min_feature_count': '1',
                    'models': '',
                    'num_cv_folds': '10',
                    'objectives': "['f1_score_micro']",
                    'objective': "f1_score_micro",
                    'param_grids': '[]',
                    'pos_label_str': '',
                    'predictions': '',
                    'probability': 'False',
                    'random_folds': 'False',
                    'results': '',
                    'sampler': '',
                    'sampler_parameters': '[]',
                    'save_cv_folds': 'False',
                    'shuffle': 'False',
                    'suffix': '',
                    'test_directory': '',
                    'test_file': '',
                    'train_directory': '',
                    'train_file': ''}

        correct_section_mapping = {'class_map': 'Input',
                                   'custom_learner_path': 'Input',
                                   'cv_folds_file': 'Input',
                                   'feature_hasher': 'Input',
                                   'feature_scaling': 'Input',
                                   'featuresets': 'Input',
                                   'featureset_names': 'Input',
                                   'fixed_parameters': 'Input',
                                   'grid_search': 'Tuning',
                                   'grid_search_folds': 'Tuning',
                                   'grid_search_jobs': 'Tuning',
                                   'hasher_features': 'Input',
                                   'id_col': 'Input',
                                   'ids_to_floats': 'Input',
                                   'label_col': 'Input',
                                   'log': 'Output',
                                   'min_feature_count': 'Tuning',
                                   'models': 'Output',
                                   'num_cv_folds': 'Input',
                                   'objectives': 'Tuning',
                                   'objective': 'Tuning',
                                   'param_grids': 'Tuning',
                                   'pos_label_str': 'Tuning',
                                   'predictions': 'Output',
                                   'probability': 'Output',
                                   'random_folds': 'Input',
                                   'results': 'Output',
                                   'sampler': 'Input',
                                   'sampler_parameters': 'Input',
                                   'save_cv_folds': 'Output',
                                   'shuffle': 'Input',
                                   'suffix': 'Input',
                                   'test_directory': 'Input',
                                   'test_file': 'Input',
                                   'train_directory': 'Input',
                                   'train_file': 'Input'}

        # make sure that the defaults dictionary and the
        # section mapping dictionary have the same keys
        assert defaults.keys() == correct_section_mapping.keys()

        super(SKLLConfigParser, self).__init__(defaults=defaults)
        self._required_options = required
        self._section_mapping = correct_section_mapping

    def _find_invalid_options(self):
        """
        Returns the set of invalid options specified by the user
        """

        # compute a list of all the valid options
        valid_options = list(self._defaults.keys()) + self._required_options

        # get a set of all of the specified options
        specified_options = set(itertools.chain(*[self.options(section)
                                                  for section in self.sections()]))

        # find any invalid options and return
        invalid_options = set(specified_options).difference(valid_options)
        return invalid_options

    def _find_ill_specified_options(self):
        """
        Make sure that all the options are specified in the appropriate sections
        and are not specified in multiple spaces. One way to do this is to
        basically get the values for all the optional config options from each
        of the sections and then look at the section that has a non-default
        (specified) value and if that's not the right section as indicated by
        our section mapping, then we need to raise an exception.

        .. note::

            This will NOT work if the user specifies the default value for the
            option but puts it in the wrong section. However, since specifying
            the default value for the option  does not result in running an
            experiment with unexpected settings, this is not really a major
            problem.
        """
        incorrectly_specified_options = []
        multiply_specified_options = []
        for option_name, default_value in self._defaults.items():
            used_sections = [section for section in self.sections()
                             if self.get(section, option_name) != default_value]
            # skip options that are not specified in any sections
            if len(used_sections) == 0:
                pass
            # find all options that are specified in multiple sections
            elif len(used_sections) > 1:
                multiply_specified_options.append((option_name, used_sections))
            # find all options that are specified in incorrect sections
            elif used_sections[0] != self._section_mapping[option_name]:
                incorrectly_specified_options.append((option_name,
                                                      used_sections[0]))

        return (incorrectly_specified_options, multiply_specified_options)

    def validate(self):
        """Validate specified options to check:
             (a) no invalid options are specified
             (b) options are not specified in multiple sections
             (c) options are specified in the correct section
        """

        invalid_options = self._find_invalid_options()
        if invalid_options:
            raise KeyError('Configuration file contains the following '
                           'unrecognized options: {}'
                           .format(list(invalid_options)))

        incorrectly_specified_options, multiply_specified_options = self._find_ill_specified_options()
        if multiply_specified_options:
            raise KeyError('The following are defined in multiple sections: '
                           '{}'.format([t[0] for t in
                                        multiply_specified_options]))
        if incorrectly_specified_options:
            raise KeyError('The following are not defined in the appropriate '
                           'sections: {}'.format([t[0] for t in
                                                  incorrectly_specified_options]))


def _locate_file(file_path, config_dir):
    if not file_path:
        return ''
    path_to_check = file_path if isabs(file_path) else normpath(join(config_dir,
                                                                     file_path))
    ans = exists(path_to_check)
    if not ans:
        raise IOError(errno.ENOENT, "File does not exist", path_to_check)
    else:
        return path_to_check


def _setup_config_parser(config_path, validate=True):
    """
    Returns a config parser at a given path. Only implemented as a separate
    function to simplify testing.
    """
    # initialize config parser with the given defaults
    config = SKLLConfigParser()

    # Read file if it exists
    if not exists(config_path):
        raise IOError(errno.ENOENT, "Configuration file does not exist",
                      config_path)
    config.read(config_path)

    # normalize objective to objectives
    objective_value = config.get('Tuning', 'objective')
    objectives_value = config.get('Tuning', 'objectives')
    objective_default = config._defaults['objective']
    objectives_default = config._defaults['objectives']

    # if both of them are non default, raise error
    if (objectives_value != objectives_default and objective_value != objective_default):
        raise ValueError("The configuration file can specify "
                         "either 'objective' or 'objectives', "
                         "not both")
    else:
        # if objective is default value, delete it
        if objective_value == objective_default:
            config.remove_option('Tuning', 'objective')
        else:
            # else convert objective into objectives and delete objective
            objective_value = yaml.load(_fix_json(objective_value))
            if isinstance(objective_value, string_types):
                config.set(
                    'Tuning', 'objectives', "['{}']".format(objective_value))
                config.remove_option('Tuning', 'objective')
            else:
                raise TypeError("objective should be a string")

    if validate:
        config.validate()

    return config


def _parse_config_file(config_path):
    """
    Parses a SKLL experiment configuration file with the given path.
    """

    # Initialize logger
    logger = logging.getLogger(__name__)

    # check that config_path is not empty
    if config_path == "":
        raise IOError("The name of the configuration file is empty")

    # compute the absolute path for the config file
    config_path = realpath(config_path)
    config_dir = dirname(config_path)

    # set up a config parser with the above default values
    config = _setup_config_parser(config_path)

    # extract parameters from the various sections in the config file

    # 1. General
    if config.has_option("General", "experiment_name"):
        experiment_name = config.get("General", "experiment_name")
    else:
        raise ValueError("Configuration file does not contain experiment_name "
                         "in the [General] section.")

    if config.has_option("General", "task"):
        task = config.get("General", "task")
    else:
        raise ValueError("Configuration file does not contain task in the "
                         "[General] section.")
    if task not in _VALID_TASKS:
        raise ValueError('An invalid task was specified: {}.  Valid tasks are:'
                         ' {}'.format(task, ', '.join(_VALID_TASKS)))

    # 2. Input
    sampler = config.get("Input", "sampler")
    if sampler not in _VALID_SAMPLERS:
        raise ValueError('An invalid sampler was specified: {}.  Valid '
                         'samplers are: {}'.format(sampler,
                                                   ', '.join(_VALID_SAMPLERS)))

    # produce warnings if feature_hasher is set but hasher_features
    # is less than or equal to zero.
    feature_hasher = config.getboolean("Input", "feature_hasher")
    hasher_features = config.getint("Input", "hasher_features")
    if feature_hasher:
        if hasher_features <= 0:
            raise ValueError("Configuration file must specify a non-zero value "
                             "for the option hasher_features when "
                             "feature_hasher is True.")

    # produce warnings if hasher_features is set but feature_hasher
    # is not set correctly
    elif hasher_features > 0:
        logger.warning("Ignoring hasher_features since feature_hasher is either"
                       " missing or set to False.")

    if config.has_option("Input", "learners"):
        learners_string = config.get("Input", "learners")
    else:
        raise ValueError("Configuration file does not contain list of learners "
                         "in [Input] section.")
    learners = yaml.load(_fix_json(learners_string))

    if len(learners) == 0:
        raise ValueError("Configuration file contains an empty list of learners"
                         " in the [Input] section.")

    elif len(set(learners)) < len(learners):
        raise ValueError('Configuration file contains the same learner multiple'
                         ' times, which is not currently supported.  Please use'
                         ' param_grids with tuning to find the optimal settings'
                         ' for the learner.')
    custom_learner_path = _locate_file(config.get("Input", "custom_learner_path"),
                                       config_dir)

    # get the featuresets
    featuresets_string = config.get("Input", "featuresets")
    featuresets = yaml.load(_fix_json(featuresets_string))

    # ensure that featuresets is either a list of features or a list of lists
    # of features
    if not isinstance(featuresets, list) or not all(isinstance(fs, list) for fs
                                                    in featuresets):
        raise ValueError("The featuresets parameter should be a list of "
                         "features or a list of lists of features. You "
                         "specified: {}".format(featuresets))

    featureset_names = yaml.load(_fix_json(config.get("Input",
                                                      "featureset_names")))

    # ensure that featureset_names is a list of strings, if specified
    if featureset_names:
        if (not isinstance(featureset_names, list) or
                not all([isinstance(fs, string_types) for fs in
                         featureset_names])):
            raise ValueError("The featureset_names parameter should be a list "
                             "of strings. You specified: {}"
                             .format(featureset_names))

    # do we need to shuffle the training data
    do_shuffle = config.getboolean("Input", "shuffle")

    fixed_parameter_list = yaml.load(_fix_json(config.get("Input",
                                                          "fixed_parameters")))
    fixed_sampler_parameters = _fix_json(config.get("Input",
                                                    "sampler_parameters"))
    fixed_sampler_parameters = yaml.load(fixed_sampler_parameters)
    param_grid_list = yaml.load(_fix_json(config.get("Tuning", "param_grids")))
    pos_label_str = config.get("Tuning", "pos_label_str")

    # ensure that feature_scaling is specified only as one of the
    # four available choices
    feature_scaling = config.get("Input", "feature_scaling")
    if feature_scaling not in _VALID_FEATURE_SCALING_OPTIONS:
        raise ValueError("Invalid value for feature_scaling parameter: {}"
                         .format(feature_scaling))

    suffix = config.get("Input", "suffix")
    label_col = config.get("Input", "label_col")
    id_col = config.get("Input", "id_col")
    ids_to_floats = config.getboolean("Input", "ids_to_floats")

    # get the cv folds file and make a dictionary from it, if it exists
    cv_folds_file = _locate_file(config.get("Input", "cv_folds_file"),
                                 config_dir)
    num_cv_folds = config.getint("Input", "num_cv_folds")
    if cv_folds_file:
        cv_folds = _load_cv_folds(cv_folds_file,
                                  ids_to_floats=ids_to_floats)
    else:
        # set the number of folds for cross-validation
        cv_folds = num_cv_folds if num_cv_folds else 10

    # whether or not to save the cv fold ids
    save_cv_folds = config.get("Output", "save_cv_folds")

    # whether or not to do stratified cross validation
    random_folds = config.getboolean("Input", "random_folds")
    if random_folds:
        if cv_folds_file:
            logger.warning('Specifying cv_folds_file overrides random_folds')
        do_stratified_folds = False
    else:
        do_stratified_folds = True

    # get all the input paths and directories (without trailing slashes)
    train_path = config.get("Input", "train_directory").rstrip(os.sep)
    test_path = config.get("Input", "test_directory").rstrip(os.sep)
    train_file = config.get("Input", "train_file")
    test_file = config.get("Input", "test_file")

    # make sure that featuresets is not an empty list unless
    # train_file and test_file are specified
    if not train_file and not test_file and (isinstance(featuresets, list) and
                                             len(featuresets) == 0):
        raise ValueError(
            "The 'featuresets' parameters cannot be an empty list.")

    # The user must specify either train_file or train_path, not both.
    if not train_file and not train_path:
        raise ValueError('Invalid [Input] parameters: either "train_file" or '
                         '"train_directory" must be specified in the '
                         'configuration file.')

    # Either train_file or train_path must be specified.
    if train_file and train_path:
        raise ValueError('Invalid [Input] parameters: only either "train_file"'
                         ' or "train_directory" can be specified in the '
                         'configuration file, not both.')

    # Cannot specify both test_file and test_path
    if test_file and test_path:
        raise ValueError('Invalid [Input] parameters: only either "test_file" '
                         'or "test_directory" can be specified in the '
                         'configuration file, not both.')

    # if train_file is specified, then assign its value to train_path
    # this is a workaround to make this simple use case (a single train and
    # test file) compatible with the existing architecture using
    # featuresets
    if train_file:
        train_path = train_file
        featuresets = [['train_{}'.format(basename(train_file))]]
        suffix = ''

    # if test_file is specified, then assign its value to test_path to
    # enable compatibility with the pre-existing featuresets architecture
    if test_file:
        test_path = test_file
        featuresets[0][0] += '_test_{}'.format(basename(test_file))

    # make sure all the specified paths/files exist
    train_path = _locate_file(train_path, config_dir)
    test_path = _locate_file(test_path, config_dir)
    train_file = _locate_file(train_file, config_dir)
    test_file = _locate_file(test_file, config_dir)

    # Get class mapping dictionary if specified
    class_map_string = config.get("Input", "class_map")
    original_class_map = yaml.load(_fix_json(class_map_string))
    if original_class_map:
        # Change class_map to map from originals to replacements instead of
        # from replacement to list of originals
        class_map = {}
        for replacement, original_list in iteritems(original_class_map):
            for original in original_list:
                class_map[original] = replacement
        del original_class_map
    else:
        class_map = None

    # 3. Output
    probability = config.getboolean("Output", "probability")

    # do we want to keep the predictions?
    prediction_dir = _locate_file(config.get("Output", "predictions"),
                                  config_dir)
    if prediction_dir:
        if not exists(prediction_dir):
            os.makedirs(prediction_dir)

    # make sure log path exists
    log_path = _locate_file(config.get("Output", "log"), config_dir)
    if log_path:
        log_path = join(config_dir, log_path)
        if not exists(log_path):
            os.makedirs(log_path)

    # make sure model path exists
    model_path = _locate_file(config.get("Output", "models"), config_dir)
    if model_path:
        model_path = join(config_dir, model_path)
        if not exists(model_path):
            os.makedirs(model_path)

    # make sure results path exists
    results_path = _locate_file(config.get("Output", "results"), config_dir)
    if results_path:
        results_path = join(config_dir, results_path)
        if not exists(results_path):
            os.makedirs(results_path)

    # 4. Tuning
    # do we need to run a grid search for the hyperparameters or are we just
    # using the defaults?
    do_grid_search = config.getboolean("Tuning", "grid_search")

    # minimum number of examples a feature must be nonzero in to be included
    min_feature_count = config.getint("Tuning", "min_feature_count")

    # how many jobs should we run in parallel for grid search
    grid_search_jobs = config.getint("Tuning", "grid_search_jobs")
    if not grid_search_jobs:
        grid_search_jobs = None

    # how many folds should we run in parallel for grid search
    grid_search_folds = config.getint("Tuning", "grid_search_folds")

    # what are the objective functions for the grid search?
    grid_objectives = config.get("Tuning", "objectives")
    grid_objectives = yaml.load(_fix_json(grid_objectives))
    if not isinstance(grid_objectives, list):
        raise TypeError("objectives should be a "
                        "list of objectives")

    if not all([objective in SCORERS for objective in grid_objectives]):
        raise ValueError('Invalid grid objective function/s: {}'
                         .format(grid_objectives))

    # check whether the right things are set for the given task
    if (task == 'evaluate' or task == 'predict') and not test_path:
        raise ValueError('The test set must be set when task is evaluate or '
                         'predict.')
    if (task == 'cross_validate' or task == 'train') and test_path:
        raise ValueError('The test set should not be set when task is '
                         'cross_validate or train.')
    if (task == 'train' or task == 'predict') and results_path:
        raise ValueError('The results path should not be set when task is '
                         'predict or train.')
    if task == 'train' and not model_path:
        raise ValueError('The model path should be set when task is train.')
    if task == 'train' and prediction_dir:
        raise ValueError('The predictions path should not be set when task is '
                         'train.')
    if task == 'cross_validate' and model_path:
        raise ValueError('The models path should not be set when task is '
                         'cross_validate.')

    # Create feature set names if unspecified
    if not featureset_names:
        featureset_names = [_munge_featureset_name(x) for x in featuresets]
    if len(featureset_names) != len(featuresets):
        raise ValueError(('Number of feature set names (%s) does not match '
                          'number of feature sets (%s).') %
                         (len(featureset_names), len(featuresets)))

    # store training/test set names for later use
    train_set_name = basename(train_path)
    test_set_name = basename(test_path) if test_path else "cv"

    return (experiment_name, task, sampler, fixed_sampler_parameters,
            feature_hasher, hasher_features, id_col, label_col, train_set_name,
            test_set_name, suffix, featuresets, do_shuffle, model_path,
            do_grid_search, grid_objectives, probability, results_path,
            pos_label_str, feature_scaling, min_feature_count,
            grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
            do_stratified_folds, fixed_parameter_list, param_grid_list,
            featureset_names, learners, prediction_dir, log_path, train_path,
            test_path, ids_to_floats, class_map, custom_learner_path)


def _munge_featureset_name(featureset):
    """
    Joins features in featureset by '+' if featureset is not a string, and just
    returns featureset otherwise.
    """
    if isinstance(featureset, string_types):
        return featureset

    res = '+'.join(sorted(featureset))
    return res


def _fix_json(json_string):
    """
    Takes a bit of JSON that might have bad quotes or capitalized booleans and
    fixes that stuff.
    """
    json_string = json_string.replace('True', 'true')
    json_string = json_string.replace('False', 'false')
    json_string = json_string.replace("'", '"')
    return json_string


def _load_cv_folds(cv_folds_file, ids_to_floats=False):
    """
    Loads CV folds from a CSV file with columns for example ID and fold ID (and
    a header).
    """
    with open(cv_folds_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # discard the header
        res = {}
        for row in reader:
            if ids_to_floats:
                try:
                    row[0] = float(row[0])
                except ValueError:
                    raise ValueError('You set ids_to_floats to true, but ID {}'
                                     ' could not be converted to float'
                                     .format(row[0]))
            res[row[0]] = row[1]

    return res
