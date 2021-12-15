# License: BSD 3 clause
"""
The main class and functions used to parse SKLL configuration files.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Chee Wee Leong (cleong@ets.org)
"""

import configparser
import errno
import itertools
import logging
import os
from os.path import basename, dirname, exists, join, realpath

import numpy as np
import ruamel.yaml as yaml

from skll.data.readers import safe_float
from skll.utils.constants import (
    PROBABILISTIC_METRICS,
    VALID_FEATURE_SCALING_OPTIONS,
    VALID_SAMPLERS,
    VALID_TASKS,
)
from skll.utils.logging import get_skll_logger

from .utils import (
    _munge_featureset_name,
    _parse_and_validate_metrics,
    fix_json,
    load_cv_folds,
    locate_file,
)

__all__ = ['SKLLConfigParser', 'fix_json', 'load_cv_folds', 'locate_file']


class SKLLConfigParser(configparser.ConfigParser):

    """
    A custom configuration file parser for SKLL.
    """

    def __init__(self):

        # these are the three options that must be set in a config
        # file and no defaults are provided
        required = ['experiment_name', 'task', 'learners']

        # these are the optional config options for which
        # defaults are automatically provided
        defaults = {'class_map': '{}',
                    'custom_learner_path': '',
                    'custom_metric_path': '',
                    'cv_seed': '123456789',
                    'folds_file': '',
                    'feature_hasher': 'False',
                    'feature_scaling': 'none',
                    'featuresets': '[]',
                    'featureset_names': '[]',
                    'fixed_parameters': '[]',
                    'grid_search': 'True',
                    'grid_search_folds': '5',
                    'grid_search_jobs': '0',
                    'hasher_features': '0',
                    'id_col': 'id',
                    'ids_to_floats': 'False',
                    'label_col': 'y',
                    'logs': '',
                    'learning_curve_cv_folds_list': '[]',
                    'learning_curve_train_sizes': '[]',
                    'min_feature_count': '1',
                    'models': '',
                    'num_cv_folds': '10',
                    'metrics': "[]",
                    'objectives': "[]",
                    'param_grids': '[]',
                    'pos_label': '',
                    'pipeline': 'False',
                    'predictions': '',
                    'probability': 'False',
                    'random_folds': 'False',
                    'results': '',
                    'sampler': '',
                    'sampler_parameters': '[]',
                    'save_cv_folds': 'True',
                    'save_cv_models': 'False',
                    'shuffle': 'False',
                    'suffix': '',
                    'test_directory': '',
                    'test_file': '',
                    'train_directory': '',
                    'train_file': '',
                    'use_folds_file_for_grid_search': 'True',
                    'save_votes': 'False'}

        correct_section_mapping = {'class_map': 'Input',
                                   'custom_learner_path': 'Input',
                                   'custom_metric_path': 'Input',
                                   'cv_seed': 'Input',
                                   'folds_file': 'Input',
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
                                   'logs': 'Output',
                                   'learning_curve_cv_folds_list': 'Input',
                                   'learning_curve_train_sizes': 'Input',
                                   'min_feature_count': 'Tuning',
                                   'metrics': 'Output',
                                   'models': 'Output',
                                   'num_cv_folds': 'Input',
                                   'objectives': 'Tuning',
                                   'param_grids': 'Tuning',
                                   'pos_label': 'Tuning',
                                   'pipeline': 'Output',
                                   'predictions': 'Output',
                                   'probability': 'Output',
                                   'random_folds': 'Input',
                                   'results': 'Output',
                                   'sampler': 'Input',
                                   'sampler_parameters': 'Input',
                                   'save_cv_folds': 'Output',
                                   'save_cv_models': 'Output',
                                   'shuffle': 'Input',
                                   'suffix': 'Input',
                                   'test_directory': 'Input',
                                   'test_file': 'Input',
                                   'train_directory': 'Input',
                                   'train_file': 'Input',
                                   'use_folds_file_for_grid_search': 'Tuning',
                                   'save_votes': 'Output'}

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
                           f'unrecognized options: {list(invalid_options)}')

        incorrectly_specified_options, multiply_specified_options = self._find_ill_specified_options()
        if multiply_specified_options:
            raise KeyError('The following are defined in multiple sections: '
                           f'{[t[0] for t in multiply_specified_options]}')
        if incorrectly_specified_options:
            raise KeyError('The following are not defined in the appropriate '
                           f'sections: {[t[0] for t in incorrectly_specified_options]}')


def parse_config_file(config_path, log_level=logging.INFO):  # noqa: C901
    """
    Parses a SKLL experiment configuration file with the given path.
    Log messages with the given log level (default: INFO).

    Parameters
    ----------
    config_path : str
        The path to the configuration file.

    log_level : int, default=logging.INFO
        Determines which messages should be logged. You can either pass
        an integer or one of the corresponding ``logging`` parameters, such
        as ``logging.INFO`` or ``logging.WARNING``.

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
        A dictionary containing parameters you want to have fixed for the sampler.

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
        A list of scoring functions to use for tuning.

    probability : bool
        Whether to output probabilities for each class.

    pipeline : bool
        Whether to include the `pipeline` attribute in the
        trained model. This will increase the size of the
        model file.

    results_path : str
        Path to store result files in.

    pos_label : str
        The string label for the positive class in the binary
        classification setting.

    feature_scaling : str
        How to scale features (e.g. 'with_mean').

    min_feature_count : int
        The minimum number of examples for which the value of a
        feature must be nonzero to be included in the model.

    folds_file : str
        The path to the folds_file, if specified.

    grid_search_jobs : int
        Number of folds to run in parallel when using grid search.

    grid_search_folds : int
        The number of folds to use for grid search.

    cv_folds : dict or int
        The specified folds mapping, or the number of folds.

    cv_seed : int
        The seed value for the random number generator used
        to create the cross-validation folds.

    save_cv_folds : bool
        Whether to save CV Folds to file.

    save_cv_models : bool
        Whether to save CV models.

    use_folds_file_for_grid_search : bool
        Whether to use folds file for grid search.

    do_stratified_folds : bool
        Whether to use random folds for cross-validation.

    fixed_parameter_list : list of dict
        List of dicts containing parameters you want to have fixed for
        each classifier in learners list.

    param_grid_list : list of dict
        List of parameter grids to search, one dict for each learner.

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

    custom_metric_path : str
        Path to a .py file that defines a custom metric.

    learning_curve_cv_folds_list : list of int
        A list of integers specifying the number of folds to use for CV.

    learning_curve_train_sizes : list of float or list of int
        List of floats or integers representing relative or absolute numbers
        of training examples that will be used to generate the learning
        curve respectively.

    output_metrics : list
        A list of output metrics to use.

    save_votes : bool
        Whether to save the individual predictions from voting learners.

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

    ######################
    # 1. General section #
    ######################
    if config.has_option("General", "experiment_name"):
        experiment_name = config.get("General", "experiment_name")
    else:
        raise ValueError("Configuration file does not contain experiment_name "
                         "in the [General] section.")

    # next, get the log path before anything else since we need to
    # save all logging messages to a log file in addition to displaying
    # them on the console
    log_value = config.get("Output", "logs")

    try:
        log_path = locate_file(log_value, config_dir)
    except IOError as e:
        if e.errno == errno.ENOENT:
            log_path = e.filename
            os.makedirs(log_path)

    # Create a top-level log file under the log path
    main_log_file = join(log_path, f'{experiment_name}.log')

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
    if task not in VALID_TASKS:
        raise ValueError(f'An invalid task was specified: {task}.  Valid '
                         f'tasks are: {", ".join(VALID_TASKS)}')

    ####################
    # 2. Input section #
    ####################
    sampler = config.get("Input", "sampler")
    if sampler not in VALID_SAMPLERS:
        raise ValueError(f'An invalid sampler was specified: {sampler}.  Valid'
                         f' samplers are: {", ".join(VALID_SAMPLERS)}')

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
    learners = yaml.safe_load(fix_json(learners_string))

    if len(learners) == 0:
        raise ValueError("Configuration file contains an empty list of learners"
                         " in the [Input] section.")

    elif len(set(learners)) < len(learners):
        raise ValueError('Configuration file contains the same learner multiple'
                         ' times, which is not currently supported.  Please use'
                         ' param_grids with tuning to find the optimal settings'
                         ' for the learner.')
    custom_learner_path = locate_file(config.get("Input", "custom_learner_path"),
                                      config_dir)

    # get the custom metric path, if specified, and locate it
    custom_metric_path = locate_file(config.get("Input", "custom_metric_path"),
                                     config_dir)

    # get the featuresets
    featuresets_string = config.get("Input", "featuresets")
    featuresets = yaml.safe_load(fix_json(featuresets_string))

    # ensure that featuresets is either a list of features or a list of lists
    # of features
    if not isinstance(featuresets, list) or not all(isinstance(fs, list) for fs
                                                    in featuresets):
        raise ValueError("The featuresets parameter should be a list of "
                         "features or a list of lists of features. You "
                         f"specified: {featuresets}")

    featureset_names = yaml.safe_load(fix_json(config.get("Input",
                                                          "featureset_names")))

    # ensure that featureset_names is a list of strings, if specified
    if featureset_names:
        if (not isinstance(featureset_names, list) or
                not all([isinstance(fs, str) for fs in
                        featureset_names])):
            raise ValueError("The featureset_names parameter should be a list "
                             f"of strings. You specified: {featureset_names}")

    # get the value for learning_curve_cv_folds and ensure
    # that it's a list of the same length as the value of
    # learners. If it's not specified, then we just assume
    # that we are using 10 folds for each learner.
    learning_curve_cv_folds_list_string = config.get("Input",
                                                     "learning_curve_cv_folds_list")
    learning_curve_cv_folds_list = yaml.safe_load(fix_json(learning_curve_cv_folds_list_string))
    if len(learning_curve_cv_folds_list) == 0:
        learning_curve_cv_folds_list = [10] * len(learners)
    else:
        if (not isinstance(learning_curve_cv_folds_list, list) or
                not all([isinstance(fold, int) for fold in learning_curve_cv_folds_list]) or
                not len(learning_curve_cv_folds_list) == len(learners)):
            raise ValueError("The learning_curve_cv_folds parameter should "
                             "be a list of integers of the same length as "
                             "the number of learners. You specified: "
                             f"{learning_curve_cv_folds_list}")

    # get the value for learning_curve_train_sizes and ensure
    # that it's a list of either integers (sizes) or
    # floats (proportions). If it's not specified, then we just
    # assume that we are using np.linspace(0.1, 1.0, 5).
    learning_curve_train_sizes_string = config.get("Input", "learning_curve_train_sizes")
    learning_curve_train_sizes = yaml.safe_load(fix_json(learning_curve_train_sizes_string))
    if len(learning_curve_train_sizes) == 0:
        learning_curve_train_sizes = np.linspace(0.1, 1.0, 5).tolist()
    else:
        if (not isinstance(learning_curve_train_sizes, list) or
            not all([isinstance(size, int) or isinstance(size, float) for size in
                     learning_curve_train_sizes])):
            raise ValueError("The learning_curve_train_sizes parameter should "
                             "be a list of integers or floats. You specified: "
                             f"{learning_curve_train_sizes}")

    # do we need to shuffle the training data
    do_shuffle = config.getboolean("Input", "shuffle")

    fixed_parameter_list = yaml.safe_load(fix_json(config.get("Input",
                                                              "fixed_parameters")))
    fixed_sampler_parameters = fix_json(config.get("Input",
                                                   "sampler_parameters"))
    fixed_sampler_parameters = yaml.safe_load(fixed_sampler_parameters)
    param_grid_list = yaml.safe_load(fix_json(config.get("Tuning", "param_grids")))

    # read and normalize the value of `pos_label`
    pos_label = safe_float(config.get("Tuning", "pos_label"))
    if pos_label == '':
        pos_label = None

    # ensure that feature_scaling is specified only as one of the
    # four available choices
    feature_scaling = config.get("Input", "feature_scaling")
    if feature_scaling not in VALID_FEATURE_SCALING_OPTIONS:
        raise ValueError("Invalid value for feature_scaling parameter: "
                         f"{feature_scaling}")

    suffix = config.get("Input", "suffix")
    label_col = config.get("Input", "label_col")
    id_col = config.get("Input", "id_col")
    ids_to_floats = config.getboolean("Input", "ids_to_floats")

    # read in the cross-validation seed
    cv_seed = config.getint("Input", "cv_seed")

    # if an external folds file is specified, then read it into a dictionary
    folds_file = locate_file(config.get("Input", "folds_file"), config_dir)
    num_cv_folds = config.getint("Input", "num_cv_folds")
    specified_folds_mapping = None
    specified_num_folds = None
    if folds_file:
        specified_folds_mapping = load_cv_folds(folds_file, ids_to_floats=ids_to_floats)
    else:
        # if no file is specified, then set the number of folds for cross-validation
        specified_num_folds = num_cv_folds if num_cv_folds else 10

    # whether or not to save the cv fold ids/models
    save_cv_folds = config.getboolean("Output", "save_cv_folds")
    save_cv_models = config.getboolean("Output", "save_cv_models")

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
        featuresets = [[f'train_{basename(train_file)}']]
        suffix = ''

    # if test_file is specified, then assign its value to test_path to
    # enable compatibility with the pre-existing featuresets architecture
    if test_file:
        test_path = test_file
        featuresets[0][0] += f'_test_{basename(test_file)}'

    # make sure all the specified paths/files exist
    train_path = locate_file(train_path, config_dir)
    test_path = locate_file(test_path, config_dir)

    # Get class mapping dictionary if specified
    class_map_string = config.get("Input", "class_map")
    original_class_map = yaml.safe_load(fix_json(class_map_string))
    if original_class_map:
        # Change class_map to map from originals to replacements instead of
        # from replacement to list of originals
        class_map = {}
        for replacement, original_list in original_class_map.items():
            for original in original_list:
                class_map[original] = replacement
        del original_class_map
    else:
        class_map = None

    #####################
    # 3. Output section #
    #####################
    probability = config.getboolean("Output", "probability")
    pipeline = config.getboolean("Output", "pipeline")

    # do we want to keep the predictions?
    # make sure the predictions path exists and if not create it
    try:
        prediction_dir = locate_file(config.get("Output", "predictions"),
                                     config_dir)
    except IOError as e:
        if e.errno == errno.ENOENT:
            prediction_dir = e.filename
            os.makedirs(prediction_dir)

    # make sure model path exists and if not, create it
    try:
        model_path = locate_file(config.get("Output", "models"), config_dir)
    except IOError as e:
        if e.errno == errno.ENOENT:
            model_path = e.filename
            os.makedirs(model_path)

    # make sure results path exists
    try:
        results_path = locate_file(config.get("Output", "results"), config_dir)
    except IOError as e:
        if e.errno == errno.ENOENT:
            results_path = e.filename
            os.makedirs(results_path)

    # what are the output metrics?
    output_metrics = config.get("Output", "metrics")
    output_metrics = _parse_and_validate_metrics(output_metrics,
                                                 'metrics',
                                                 logger=logger)

    # do we want to save the individual predictions from voting
    # learner estimators?
    save_votes = config.getboolean("Output", "save_votes")

    #####################
    # 4. Tuning section #
    #####################

    # do we need to run a grid search for the hyperparameters or are we just
    # using the defaults?
    do_grid_search = config.getboolean("Tuning", "grid_search")

    # parse any provided grid objective functions
    grid_objectives = config.get("Tuning", "objectives")
    grid_objectives = _parse_and_validate_metrics(grid_objectives,
                                                  'objectives',
                                                  logger=logger)

    # if we are doing learning curves , we don't care about
    # grid search
    if task == 'learning_curve' and do_grid_search:
        do_grid_search = False
        logger.warning("Grid search is not supported during "
                       "learning curve generation. Disabling.")

    # Check if `param_grids` is specified, but `do_grid_search` is False
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

    # check whether the right things are set for the given task
    if (task == 'evaluate' or task == 'predict') and not test_path:
        raise ValueError('The test set must be set when task is evaluate or '
                         'predict.')
    if task in ['cross_validate', 'evaluate', 'train']:
        if do_grid_search and len(grid_objectives) == 0:
            raise ValueError('Grid search is on. Either specify a list of tuning '
                             'objectives or set `grid_search` to `false` in the '
                             'Tuning section.')
        if not do_grid_search and len(grid_objectives) > 0:
            logger.warning('Since "grid_search" is set to False, any specified'
                           ' "objectives" will be ignored.')
            grid_objectives = []
    if task in ['cross_validate', 'train', 'learning_curve'] and test_path:
        raise ValueError('The test set should not be set when task is '
                         f'{task}.')
    if task in ['train', 'predict'] and results_path and not do_grid_search:
        raise ValueError('The results path should not be set when task is '
                         f'{task} and "grid_search" is set to False.')
    if task == 'train' and not model_path:
        raise ValueError('The model path should be set when task is train.')
    if task in ['learning_curve', 'train'] and prediction_dir:
        raise ValueError('The predictions path should not be set when task is '
                         f'{task}.')
    if task == 'learning_curve' and model_path:
        raise ValueError('The models path should not be set when task is '
                         'learning_curve.')
    if task == 'learning_curve':
        if len(grid_objectives) > 0:
            raise ValueError("The \"objectives\" option is no longer supported"
                             " for the \"learning_curve\" task. Please use the"
                             " \"metrics\" option in the [Output] section "
                             "instead.")
        if len(output_metrics) == 0:
            raise ValueError('The "metrics" option must be set when '
                             'the task is "learning_curve".')

    # if any of the objectives or metrics require probabilities to be output,
    # probability must be specified as true
    specified_probabilistic_metrics = PROBABILISTIC_METRICS.intersection(grid_objectives + output_metrics)
    if specified_probabilistic_metrics and not probability:
        raise ValueError("The 'probability' option must be 'true' "
                         " to compute the following: "
                         f"{list(specified_probabilistic_metrics)}.")

    # set the folds appropriately based on the task:
    #  (a) if the task is `train`/`evaluate`/`predict` and if an external
    #      fold mapping is specified then use that mapping for grid search
    #      instead of the value contained in `grid_search_folds`.
    #  (b) if the task is `cross_validate` and an external fold mapping is specified
    #      then use that mapping for the outer CV loop and for the inner grid-search
    #      loop. However, if  `use_folds_file_for_grid_search` is `False`, do not
    #      use the fold mapping for the inner loop.
    cv_folds = None
    if task in ['train', 'evaluate', 'predict'] and specified_folds_mapping:
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
        if save_cv_models is True and not model_path:
            raise ValueError("Output directory for models must be set if "
                             "\"save_cv_models\" is set to true.")

    # Create feature set names if unspecified
    if not featureset_names:
        featureset_names = [_munge_featureset_name(x) for x in featuresets]
    if len(featureset_names) != len(featuresets):
        raise ValueError('Number of feature set names '
                         f'({len(featureset_names)}) does not match number of'
                         f' feature sets ({len(featuresets)}).')

    # store training/test set names for later use
    train_set_name = basename(train_path)
    test_set_name = basename(test_path) if test_path else "cv"

    return (experiment_name, task, sampler, fixed_sampler_parameters,
            feature_hasher, hasher_features, id_col, label_col, train_set_name,
            test_set_name, suffix, featuresets, do_shuffle, model_path,
            do_grid_search, grid_objectives, probability, pipeline, results_path,
            pos_label, feature_scaling, min_feature_count, folds_file,
            grid_search_jobs, grid_search_folds, cv_folds, cv_seed, save_cv_folds,
            save_cv_models, use_folds_file_for_grid_search, do_stratified_folds,
            fixed_parameter_list, param_grid_list, featureset_names, learners,
            prediction_dir, log_path, train_path, test_path, ids_to_floats,
            class_map, custom_learner_path, custom_metric_path,
            learning_curve_cv_folds_list, learning_curve_train_sizes,
            output_metrics, save_votes)


def _setup_config_parser(config_path, validate=True):
    """
    Returns a config parser at a given path. Only implemented as a separate
    function to simplify testing.

    Parameters
    ----------
    config_path : str
        The path to the configuration file.

    validate : bool, default=True
        Whether to validate the configuration file.

    Returns
    -------
    config : SKLLConfigParser
        A SKLL configuration object.

    Raises
    ------
    IOError
        If the configuration file does not exist.
    """
    # initialize config parser with the given defaults
    config = SKLLConfigParser()

    # Read file if it exists
    if not exists(config_path):
        raise IOError(errno.ENOENT, "Configuration file does not exist",
                      config_path)
    config.read(config_path)

    if validate:
        config.validate()

    return config
