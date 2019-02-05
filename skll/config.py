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
import numpy as np
import ruamel.yaml as yaml

from six import string_types, iteritems  # Python 2/3
from skll import get_skll_logger
from sklearn.metrics import SCORERS

_VALID_TASKS = frozenset(['predict', 'train', 'evaluate',
                          'cross_validate', 'learning_curve'])
_VALID_SAMPLERS = frozenset(['Nystroem', 'RBFSampler', 'SkewedChi2Sampler',
                             'AdditiveChi2Sampler', ''])
_VALID_FEATURE_SCALING_OPTIONS = frozenset(['with_std', 'with_mean', 'both',
                                            'none'])


class SKLLConfigParser(configparser.ConfigParser):

    """
    A custom configuration file parser for SKLL
    """

    def __init__(self):

        # these are the three options that must be set in a config
        # file and no defaults are provided
        required = ['experiment_name', 'task', 'learners']

        # these are the optional config options for which#
        # defaults are automatically provided
        defaults = {'class_map': '{}',
                    'custom_learner_path': '',
                    'cv_folds_file': '',
                    'folds_file': '',
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
                    'learning_curve_cv_folds_list': '[]',
                    'learning_curve_train_sizes': '[]',
                    'min_feature_count': '1',
                    'models': '',
                    'num_cv_folds': '10',
                    'metrics': "[]",
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
                    'train_file': '',
                    'use_folds_file_for_grid_search': 'True'}

        correct_section_mapping = {'class_map': 'Input',
                                   'custom_learner_path': 'Input',
                                   'folds_file': 'Input',
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
                                   'learning_curve_cv_folds_list': 'Input',
                                   'learning_curve_train_sizes': 'Input',
                                   'min_feature_count': 'Tuning',
                                   'metrics': 'Output',
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
                                   'train_file': 'Input',
                                   'use_folds_file_for_grid_search': 'Tuning'}

        # make sure that the defaults dictionary and the
        # section mapping dictionary have the same keys
        assert defaults.keys() == correct_section_mapping.keys()

        super(SKLLConfigParser, self).__init__(defaults=defaults)
        self._required_options = required
        self._section_mapping = correct_section_mapping

    def _find_invalid_options(self):
        """
        Find the set of invalid options specified by the user.

        Returns
        -------
        invalid_options : set of str
            The set of invalid options specified by the user.
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

        Returns
        -------
        incorrectly_specified_options : list of str
            A list of incorrectly specified options.
        multiply_specified_options : list of str
            A list of options specified more than once.

        Notes
        -----
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
        """
        Validate specified options to check ::

            (a) no invalid options are specified
            (b) options are not specified in multiple sections
            (c) options are specified in the correct section

        Raises
        ------
        KeyError
            If configuration file contains unrecognized options.
        KeyError
            If any options are defined in multiple sections.
        KeyError
            If any options are not defined in the appropriate sections.
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
    """
    Locate a file, given a file path and configuration directory.

    Parameters
    ----------
    file_path : str
        The file to locate. Path may be absolute or relative.
    config_dir : str
        The path to the configuration file directory.

    Returns
    -------
    path_to_check : str
        The normalized absolute path, if it exists.

    Raises
    ------
    IOError
        If the file does not exist.
    """
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

    Parameters
    ----------
    config_path : str
        The path to the configuration file.
    validate : bool, optional
        Whether to validate the configuration file.
        Defaults to ``True``.

    Returns
    -------
    config : SKLLConfigParser
        A SKLL configuration object.

    Raises
    ------
    IOError
        If the configuration file does not exist.
    ValueError
        If the configuration file specifies both objective and objectives.
    TypeError
        If any objective is not a string.
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
            objective_value = yaml.safe_load(_fix_json(objective_value), )
            if isinstance(objective_value, string_types):
                config.set(
                    'Tuning', 'objectives', "['{}']".format(objective_value))
                config.remove_option('Tuning', 'objective')
            else:
                raise TypeError("objective should be a string")

    if validate:
        config.validate()

    return config


def _parse_config_file(config_path, log_level=logging.INFO):
    """
    Parses a SKLL experiment configuration file with the given path.
    Log messages with the given log level (default: INFO).

    Parameters
    ----------
    config_path : str
        The path to the configuration file.
    log_level : logging level, optional
        The logging level to use.
        Defaults to ``logging.INFO``.

    Returns
    -------
    experiment_name : str
        A string used to identify this particular experiment configuration.
        When generating result summary files, this name helps prevent
        overwriting previous summaries.
    task : str
        The types of experiment we're trying to run (e.g. 'cross_validate').
    sampler : str
        The name of a sampler to perform non-linear transformations of the input.
    fixed_sampler_parameters : dict
        A dictionary containing parameters you want to have fixed for the sampler
    feature_hasher : bool
        If True, this enables a high-speed, low-memory vectorizer that uses
        feature hashing for converting feature dictionaries into NumPy arrays
        instead of using a DictVectorizer.
    hasher_features : int
        The number of features used by the FeatureHasher if the feature_hasher
        flag is enabled.
    id_col : str
        The column with IDs.
    label_col : str
        The column with labels.
    train_set_name : str
        The name of the training set.
    test_set_name : str
        The name of the test set.
    suffix : str
        The file format the training/test files are in.
    featuresets : list of str
        A list of lists of prefixes for the files containing
        the features you would like to train/test on.
    do_shuffle : bool
        Whether to shuffle the data.
    model_path : str
        The path to the model file(s).
    do_grid_search : bool
        Whether to perform grid search.
    grid_objectives : list of str
        A list of objects functions to use for tuning.
    probability : bool
        Whether to output probabilities for each class.
    results_path : str
        Path to store result files in.
    pos_label_str : str
        The string label for the positive class in the binary
        classification setting.
    feature_scaling : str
        How to scale features (e.g. 'with_mean').
    min_feature_count : int
        The minimum number of examples for which the value of a
        feature must be nonzero to be included in the model.
    folds_file : str
        The path to the cv_folds_file, if specified.
    grid_search_jobs : int
        Number of folds to run in parallel when using grid search.
    grid_search_folds : int
        The number of folds to use for grid search.
    cv_folds : dict or int
        The specified folds mapping, or the number of folds.
    save_cv_folds : bool
        Whether to save CV Folds to file.
    use_folds_file_for_grid_search : bool
        Whether to use folds file for grid search.
    do_stratified_folds : bool
        Whether to use random folds for cross-validation.
    fixed_parameter_list : list of dict
        List of dicts containing parameters you want to have fixed for
        each classifier in learners list.
    param_grid_list : list of dict
        List of parameter grids to search for each learner.
    featureset_names : list of str
        The names of the featuresets used for each job.
    learners : list of str
        A list of learners to try using.
    prediction_dir : str
        The directories where predictions are saved.
    log_path : str
        The path to the log file.
    train_path : str
        The path to a file containing feature to train on.
    test_path : str
        The path to a file containing features to test on.
    ids_to_floats : bool
        Whether to convert IDs to floats.
    class_map : dict
        A class map collapsing several labels into one.
    custom_learner_path : str
        Path to a .py file that defines a custom learner.
    learning_curve_cv_folds_list : list of int
        A list of integers specifying the number of folds to use for CV.
    learning_curve_train_sizes : list of float or list of int
        List of floats or integers representing relative or absolute numbers
        of training examples that will be used to generate the learning
        curve respectively.
    output_metrics : list
        A list of output metrics to use.

    Raises
    ------
    IOError
        If configuration file name is empty
    ValueError
        If various configuration parameters are incorrectly specified,
        or cause conflicts.
    """

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

    # next, get the log path before anything else since we need to
    # save all logging messages to a log file in addition to displaying
    # them on the console
    try:
        log_path = _locate_file(config.get("Output", "log"), config_dir)
    except IOError as e:
        if e.errno == errno.ENOENT:
            log_path = e.filename
            os.makedirs(log_path)

    # Create a top-level log file under the log path
    main_log_file =join(log_path, '{}.log'.format(experiment_name))

    # Now create a SKLL logger that will log to this file as well
    # as to the console. Use the log level provided - note that
    # we only have to do this the first time we call `get_skll_logger()`
    # with a given name.
    logger = get_skll_logger('experiment',
                             filepath=main_log_file,
                             log_level=log_level)

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
    learners = yaml.safe_load(_fix_json(learners_string))

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
    featuresets = yaml.safe_load(_fix_json(featuresets_string))

    # ensure that featuresets is either a list of features or a list of lists
    # of features
    if not isinstance(featuresets, list) or not all(isinstance(fs, list) for fs
                                                    in featuresets):
        raise ValueError("The featuresets parameter should be a list of "
                         "features or a list of lists of features. You "
                         "specified: {}".format(featuresets))

    featureset_names = yaml.safe_load(_fix_json(config.get("Input",
                                                           "featureset_names")))

    # ensure that featureset_names is a list of strings, if specified
    if featureset_names:
        if (not isinstance(featureset_names, list) or
                not all([isinstance(fs, string_types) for fs in
                         featureset_names])):
            raise ValueError("The featureset_names parameter should be a list "
                             "of strings. You specified: {}"
                             .format(featureset_names))

    # get the value for learning_curve_cv_folds and ensure
    # that it's a list of the same length as the value of
    # learners. If it's not specified, then we just assume
    # that we are using 10 folds for each learner.
    learning_curve_cv_folds_list_string = config.get("Input",
                                                     "learning_curve_cv_folds_list")
    learning_curve_cv_folds_list = yaml.safe_load(_fix_json(learning_curve_cv_folds_list_string))
    if len(learning_curve_cv_folds_list) == 0:
        learning_curve_cv_folds_list = [10] * len(learners)
    else:
        if (not isinstance(learning_curve_cv_folds_list, list) or
            not all([isinstance(fold, int) for fold in learning_curve_cv_folds_list]) or
            not len(learning_curve_cv_folds_list) == len(learners)):
            raise ValueError("The learning_curve_cv_folds parameter should "
                             "be a list of integers of the same length as "
                             "the number of learners. You specified: {}"
                             .format(learning_curve_cv_folds_list))

    # get the value for learning_curve_train_sizes and ensure
    # that it's a list of either integers (sizes) or
    # floats (proportions). If it's not specified, then we just
    # assume that we are using np.linspace(0.1, 1.0, 5).
    learning_curve_train_sizes_string = config.get("Input", "learning_curve_train_sizes")
    learning_curve_train_sizes = yaml.safe_load(_fix_json(learning_curve_train_sizes_string))
    if len(learning_curve_train_sizes) == 0:
        learning_curve_train_sizes = np.linspace(0.1, 1.0, 5).tolist()
    else:
        if (not isinstance(learning_curve_train_sizes, list) or
            not all([isinstance(size, int) or isinstance(size, float) for size in
                         learning_curve_train_sizes])):
            raise ValueError("The learning_curve_train_sizes parameter should "
                             "be a list of integers or floats. You specified: {}"
                             .format(learning_curve_train_sizes))

    # do we need to shuffle the training data
    do_shuffle = config.getboolean("Input", "shuffle")

    fixed_parameter_list = yaml.safe_load(_fix_json(config.get("Input",
                                                               "fixed_parameters")))
    fixed_sampler_parameters = _fix_json(config.get("Input",
                                                    "sampler_parameters"))
    fixed_sampler_parameters = yaml.safe_load(fixed_sampler_parameters)
    param_grid_list = yaml.safe_load(_fix_json(config.get("Tuning", "param_grids")))
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

    # if cv_folds_file is specified, raise a deprecation warning
    cv_folds_file_value = config.get("Input", "cv_folds_file")
    if cv_folds_file_value:
        logger.warning("The parameter \"cv_folds_file\" "
                       "is deprecated and will be removed in the next "
                       "release, please use \"folds_file\" instead.")
        config.set("Input", "folds_file", cv_folds_file_value)

    # if an external folds file is specified, then read it into a dictionary
    folds_file = _locate_file(config.get("Input", "folds_file"), config_dir)
    num_cv_folds = config.getint("Input", "num_cv_folds")
    specified_folds_mapping = None
    specified_num_folds = None
    if folds_file:
        specified_folds_mapping = _load_cv_folds(folds_file, ids_to_floats=ids_to_floats)
    else:
        # if no file is specified, then set the number of folds for cross-validation
        specified_num_folds = num_cv_folds if num_cv_folds else 10

    # whether or not to save the cv fold ids
    save_cv_folds = config.get("Output", "save_cv_folds")

    # whether or not to do stratified cross validation
    random_folds = config.getboolean("Input", "random_folds")
    if random_folds:
        if folds_file:
            logger.warning('Specifying "folds_file" overrides "random_folds".')
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
    original_class_map = yaml.safe_load(_fix_json(class_map_string))
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
    # make sure the predictions path exists and if not create it
    try:
        prediction_dir = _locate_file(config.get("Output", "predictions"),
                                      config_dir)
    except IOError as e:
        if e.errno == errno.ENOENT:
            prediction_dir = e.filename
            os.makedirs(prediction_dir)

    # make sure model path exists and if not, create it
    try:
        model_path = _locate_file(config.get("Output", "models"), config_dir)
    except IOError as e:
        if e.errno == errno.ENOENT:
            model_path = e.filename
            os.makedirs(model_path)

    # make sure results path exists
    try:
        results_path = _locate_file(config.get("Output", "results"), config_dir)
    except IOError as e:
        if e.errno == errno.ENOENT:
            results_path = e.filename
            os.makedirs(results_path)

    # what are the output metrics?
    output_metrics = config.get("Output", "metrics")
    output_metrics = _parse_and_validate_metrics(output_metrics,
                                                     'metrics',
                                                     logger=logger)

    # 4. Tuning
    # do we need to run a grid search for the hyperparameters or are we just
    # using the defaults?
    do_grid_search = config.getboolean("Tuning", "grid_search")

    # Check if `param_grids` is specified, but `grid_search` is False
    if param_grid_list and not do_grid_search:
        logger.warning('Since "grid_search" is set to False, the specified'
                       ' "param_grids" will be ignored.')

    # Warn user about potential conflicts between parameter values
    # specified in `fixed_parameter_list` and values specified in
    # `param_grid_list` (or values passed in by default) if
    # `do_grid_search` is True
    if do_grid_search and fixed_parameter_list:
        logger.warning('Note that "grid_search" is set to True and '
                       '"fixed_parameters" is also specified. If there '
                       'is a conflict between the grid search parameter'
                       ' space and the fixed parameter values, the '
                       'fixed parameter values will take precedence.')

    # minimum number of examples a feature must be nonzero in to be included
    min_feature_count = config.getint("Tuning", "min_feature_count")

    # if an external folds file was specified do we use the same folds file
    # for the inner grid-search in cross-validate as well?
    use_folds_file_for_grid_search = config.getboolean("Tuning",
                                                       "use_folds_file_for_grid_search")

    # how many jobs should we run in parallel for grid search
    grid_search_jobs = config.getint("Tuning", "grid_search_jobs")
    if not grid_search_jobs:
        grid_search_jobs = None

    # how many folds should we run in parallel for grid search
    grid_search_folds = config.getint("Tuning", "grid_search_folds")

    # what are the objective functions for the grid search?
    grid_objectives = config.get("Tuning", "objectives")
    grid_objectives = _parse_and_validate_metrics(grid_objectives,
                                                  'objectives',
                                                  logger=logger)

    # check whether the right things are set for the given task
    if (task == 'evaluate' or task == 'predict') and not test_path:
        raise ValueError('The test set must be set when task is evaluate or '
                         'predict.')
    if task in ['cross_validate', 'train', 'learning_curve'] and test_path:
        raise ValueError('The test set should not be set when task is '
                         '{}.'.format(task))
    if task in ['train', 'predict'] and results_path:
        raise ValueError('The results path should not be set when task is '
                         '{}.'.format(task))
    if task == 'train' and not model_path:
        raise ValueError('The model path should be set when task is train.')
    if task in ['learning_curve', 'train'] and prediction_dir:
        raise ValueError('The predictions path should not be set when task is '
                         '{}.'.format(task))
    if task in ['cross_validate', 'learning_curve'] and model_path:
        raise ValueError('The models path should not be set when task is '
                         '{}.'.format(task))
    if task == 'learning_curve':
        if len(grid_objectives) > 0 and len(output_metrics) == 0:
            logger.warning("The \"objectives\" option "
                           "is deprecated for the learning_curve "
                           "task and will not be supported "
                           "after the next release; please "
                           "use the \"metrics\" option in the [Output] "
                           "section instead.")
            output_metrics = grid_objectives
            grid_objectives = []
        elif len(grid_objectives) == 0 and len(output_metrics) == 0:
            raise ValueError('The "metrics" option must be set when '
                             'the task is "learning_curve".')
        elif len(grid_objectives) > 0 and len(output_metrics) > 0:
            logger.warning("Ignoring \"objectives\" for the learning_curve "
                           "task since \"metrics\" is already specified.")
            grid_objectives = []
    elif task in ['evaluate', 'cross_validate']:
        # for other appropriate tasks, if metrics and objectives have
        # some overlaps - we will assume that the user meant to
        # use the metric for tuning _and_ evaluation, not just evaluation
        if (len(grid_objectives) > 0 and
            len(output_metrics) > 0):
            common_metrics_and_objectives = set(grid_objectives).intersection(output_metrics)
            if common_metrics_and_objectives:
                logger.warning('The following are specified both as '
                               'objective functions and evaluation metrics: {}. '
                               'They will be used as the '
                               'former.'.format(common_metrics_and_objectives))
                output_metrics = [metric for metric in output_metrics
                                  if metric not in common_metrics_and_objectives]

    # if the grid objectives contains `neg_log_loss`, then probability
    # must be specified as true since that's needed to compute the loss
    if 'neg_log_loss' in grid_objectives and not probability:
        raise ValueError("The 'probability' option must be true in order "
                         "to use `neg_log_loss` as the objective.")

    # set the folds appropriately based on the task:
    #  (a) if the task is `train` and if an external fold mapping is specified
    #      then use that mapping for grid search instead of the value
    #      contained in `grid_search_folds`.
    #  (b) if the task is `cross_validate` and an external fold mapping is specified
    #      then use that mapping for the outer CV loop. Depending on the value of
    #      `use_folds_file_for_grid_search`, use the fold mapping for the inner
    #       grid-search loop as well.
    cv_folds = None
    if task == 'train' and specified_folds_mapping:
        grid_search_folds = specified_folds_mapping
        # only print out the warning if the user actually wants to do grid search
        if do_grid_search:
            logger.warning("Specifying \"folds_file\" overrides both "
                           "explicit and default \"grid_search_folds\".")
    if task == 'cross_validate':
        cv_folds = specified_folds_mapping if specified_folds_mapping else specified_num_folds
        if specified_folds_mapping:
            logger.warning("Specifying \"folds_file\" overrides both "
                           "explicit and default \"num_cv_folds\".")
            if use_folds_file_for_grid_search:
                grid_search_folds = cv_folds
            else:
                # only print out the warning if the user wants to do grid search
                if do_grid_search:
                    logger.warning("The specified \"folds_file\" will "
                                   "not be used for inner grid search.")

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
            pos_label_str, feature_scaling, min_feature_count, folds_file,
            grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
            use_folds_file_for_grid_search, do_stratified_folds, fixed_parameter_list,
            param_grid_list, featureset_names, learners, prediction_dir,
            log_path, train_path, test_path, ids_to_floats, class_map,
            custom_learner_path, learning_curve_cv_folds_list,
            learning_curve_train_sizes, output_metrics)


def _munge_featureset_name(featureset):
    """
    Joins features in featureset by '+' if featureset is not a string, and just
    returns featureset otherwise.

    Parameters
    ----------
    featureset : SKLL.FeatureSet
        A SKLL feature_set object.

    Returns
    -------
    res : str
        feature_set names joined with '+', if feature_set is not a string.
    """
    if isinstance(featureset, string_types):
        return featureset

    res = '+'.join(sorted(featureset))
    return res


def _fix_json(json_string):
    """
    Fixes incorrectly formatted quotes and capitalized booleans in the given
    JSON string.

    Parameters
    ----------
    json_string : str
        A JSON-style string.

    Returns
    -------
    json_string : str
        The normalized JSON string.
    """
    json_string = json_string.replace('True', 'true')
    json_string = json_string.replace('False', 'false')
    json_string = json_string.replace("'", '"')
    return json_string


def _parse_and_validate_metrics(metrics, option_name, logger=None):
    """
    Given a string containing a list of metrics, this function
    parses that string into a list and validates the list.

    Parameters
    ----------
    metrics : str
        A string containing a list of metrics
    option_name : str
        The name of the option with which the metrics are associated.
    logger : logging.Logger, optional
        A logging object
        Defaults to ``None``.

    Returns
    -------
    metrics : list of str
        A list of metrics for the given option.

    Raises
    ------
    TypeError
        If the given string cannot be converted to a list.
    ValueError
        If there are any invalid metrics specified.
    """

    # create a logger if one was not passed in
    if not logger:
        logger = logging.getLogger(__name__)

    # what are the objective functions for the grid search?
    metrics = yaml.safe_load(_fix_json(metrics))
    if not isinstance(metrics, list):
        raise TypeError("{} should be a list".format(option_name))

    # `mean_squared_error` should be replaced with `neg_mean_squared_error`
    if 'mean_squared_error' in metrics:
        logger.warning("The metric \"mean_squared_error\" "
                       "is deprecated and will be removed in the next "
                       "release, please use the metric "
                       "\"neg_mean_squared_error\" instead.")
        metrics[metrics.index('mean_squared_error')] = 'neg_mean_squared_error'

    invalid_metrics = [metric for metric in metrics if metric not in SCORERS]
    if invalid_metrics:
        raise ValueError('Invalid metric(s) {} '
                         'specified for {}'.format(invalid_metrics, option_name))

    return metrics


def _load_cv_folds(folds_file, ids_to_floats=False):
    """
    Loads CV folds from a CSV file with columns for example ID and fold ID (and
    a header).

    Parameters
    ----------
    folds_file : str
        The path to a folds file to read.
    ids_to_floats : bool, optional
        Whether to convert IDs to floats.
        Defaults to ``False``.

    Returns
    -------
    res : dict
        A dictionary with example IDs as the keys and fold IDs as the values.

    Raises
    ------
    ValueError
        If example IDs cannot be converted to floats and `ids_to_floats` is `True`.
    """
    with open(folds_file, 'r') as f:
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
