# License: BSD 3 clause
"""
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import re
import tempfile
from glob import glob
from io import open
from os.path import abspath, dirname, exists, join

from nose.tools import eq_, raises, assert_raises

from skll.experiments import (_load_featureset, _setup_config_parser,
                              _parse_config_file)
from skll.data.readers import safe_float

_my_dir = abspath(dirname(__file__))


def setup():
    """
    Create necessary directories for testing.
    """
    train_dir = join(_my_dir, 'train')
    if not exists(train_dir):
        os.makedirs(train_dir)
    test_dir = join(_my_dir, 'test')
    if not exists(test_dir):
        os.makedirs(test_dir)
    output_dir = join(_my_dir, 'output')
    if not exists(output_dir):
        os.makedirs(output_dir)


def tearDown():
    """
    Clean up after tests.
    """
    config_dir = join(_my_dir, 'configs')
    for config_file in glob(join(config_dir, 'test_config_parsing_*.cfg')):
        os.unlink(config_file)


def check_safe_float_conversion(converted_val, expected_val):
    """
    Check that value and type of converted_val and expected_val are equal.
    """
    eq_(converted_val, expected_val)
    eq_(type(converted_val), type(expected_val))


def test_safe_float_conversion():
    for input_val, expected_val in zip(['1.234', 1.234, '3.0', '3', 3, 'foo'],
                                       [1.234, 1.234, 3.0, 3, 3, 'foo']):
        yield check_safe_float_conversion, safe_float(input_val), expected_val


def fill_in_config_paths_for_parsing(config_template_path, values_to_fill_dict,
                                     sub_prefix):
    """
    Add paths to train, test, and output directories to a given config template
    file.
    """

    config = _setup_config_parser(config_template_path)

    to_fill_in = {'General': ['experiment_name', 'task'],
                  'Input': ['train_directory', 'train_file', 'test_directory',
                            'test_file', 'featuresets', 'featureset_names',
                            'feature_hasher', 'hasher_features', 'learners',
                            'sampler', 'shuffle', 'feature_scaling'],
                  'Tuning': ['grid_search', 'objective'],
                  'Output': ['probability', 'results', 'log', 'models',
                             'predictions']}

    for section in to_fill_in:
        for param_name in to_fill_in[section]:
            if param_name in values_to_fill_dict:
                config.set(section, param_name,
                           values_to_fill_dict[param_name])

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}_{}.cfg'.format(config_prefix, sub_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


@raises(ValueError)
def test_input_checking1():
    """
    Test merging featuresets with different number of examples
    """
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_2examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix, quiet=True)


@raises(ValueError)
def test_input_checking2():
    """
    Test joining featuresets that contain the same features for each instance
    """
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix, quiet=True)


def test_input_checking3():
    """
    Test to ensure that we correctly merge featuresets
    """
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_2']
    examples_tuple = _load_featureset(dirpath, featureset, suffix, quiet=True)
    eq_(examples_tuple.features.shape[0], 3)


def test_one_file_load_featureset():
    """
    Test loading a single file with _load_featureset
    """
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_2examples_1']
    single_file_fs = _load_featureset(join(dirpath,
                                           'test_input_2examples_1.jsonlines'),
                                      '', '', quiet=True)
    single_fs = _load_featureset(dirpath, featureset, suffix, quiet=True)
    eq_(single_file_fs, single_fs)


@raises(ValueError)
def test_config_parsing_no_name():
    """
    Test to ensure config file parsing raises an error missing experiment name
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'train_directory': train_dir,
                           'test_directory': test_dir,
                           'task': 'evaluate',
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                   values_to_fill_dict,
                                                   'no_name')

    _parse_config_file(config_path)


def test_config_parsing_bad_task():
    # Test to ensure config file parsing raises an error with invalid or
    # missing task
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir}

    for task_value, sub_prefix in zip([None, '', 'procrastinate'],
                                      ['no_task', 'missing_task', 'bad_task']):
        if task_value is not None:
            values_to_fill_dict['task'] = task_value
        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                       values_to_fill_dict,
                                                       sub_prefix)

        yield check_config_parsing_value_error, config_path


@raises(ValueError)
def check_config_parsing_value_error(config_path):
    """
    Assert that calling _parse_config_file on config_path raises ValueError
    """
    _parse_config_file(config_path)


def test_config_parsing_bad_learner():
    # Test to ensure config file parsing raises an error with missing, bad and
    # duplicate learners

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'log': output_dir,
                           'results': output_dir}

    for learners_list, sub_prefix in zip([None, '[]', 'LogisticRegression',
                                          "['LogisticRegression', "
                                          "'LogisticRegression']"],
                                         ['no_learner', 'empty_learner',
                                          'not_list_learner',
                                          'duplicate_learner']):
        if learners_list is not None:
            values_to_fill_dict['learners'] = learners_list

        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                       values_to_fill_dict,
                                                       sub_prefix)
        yield check_config_parsing_value_error, config_path


@raises(ValueError)
def test_config_parsing_bad_sampler():
    """
    Test to ensure config file parsing raises an error with an invalid sampler
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'sampler': 'RFBSampler'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                   values_to_fill_dict,
                                                   'bad_sampler')

    _parse_config_file(config_path)


@raises(ValueError)
def test_config_parsing_bad_hashing():
    """
    Test to ensure config file parsing raises an error when feature_hasher is specified but not hasher_features
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'feature_hasher': 'True'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                   values_to_fill_dict,
                                                   'bad_hashing')

    _parse_config_file(config_path)


def test_config_parsing_bad_featuresets():
    # Test to ensure config file parsing raises an error with badly specified
    # featuresets

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir}

    for featuresets, sub_prefix in zip([None, '[]', "{'f1', 'f2', 'f3'}",
                                        "[['f1', 'f2'], 'f3', 'f4']"],
                                       ['no_feats', 'empty_feats',
                                        'non_list_feats1', 'non_list_feats2']):
        if featuresets is not None:
            values_to_fill_dict['featuresets'] = featuresets

        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                       values_to_fill_dict,
                                                       sub_prefix)
        yield check_config_parsing_value_error, config_path


def test_config_parsing_bad_featurenames():
    # Test to ensure config file parsing raises an error with badly specified
    # featureset names

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'learners': "['LogisticRegression']",
                           'featuresets': ("[['f1', 'f2', 'f3'], ['f4', 'f5', "
                                           "'f6']]"),
                           'log': output_dir,
                           'results': output_dir}

    for fname, sub_prefix in zip(["['set_a']", "['1', 2]", "set_a", "1"],
                                 ['wrong_num_names', 'wrong_type_names',
                                  'wrong_num_and_type1',
                                  'wrong_num_and_type2']):
        if fname is not None:
            values_to_fill_dict['featureset_names'] = fname

        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                       values_to_fill_dict,
                                                       sub_prefix)

        yield check_config_parsing_value_error, config_path


def test_config_parsing_bad_scaling():
    # Test to ensure config file parsing raises an error with invalid scaling
    # type

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'learners': "['LogisticRegression']",
                           'featuresets': ("[['f1', 'f2', 'f3'], ['f4', 'f5', "
                                           "'f6']]"),
                           'log': output_dir,
                           'results': output_dir}

    for scaling_type, sub_prefix in zip(["foo", "True", "False"],
                                        ['bad_scaling1', 'bad_scaling2',
                                         'bad_scaling3']):

        values_to_fill_dict['feature_scaling'] = scaling_type

        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                       values_to_fill_dict,
                                                       sub_prefix)

        yield check_config_parsing_value_error, config_path


def test_config_parsing_bad_train():
    # Test to ensure config file parsing raises an error with invalid train
    # path specifications

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'test_directory': test_dir,
                           'learners': "['LogisticRegression']",
                           'featuresets': ("[['f1', 'f2', 'f3'], ['f4', 'f5', "
                                           "'f6']]"),
                           'log': output_dir,
                           'results': output_dir}

    for sub_prefix in ['no_train_path_or_file',
                       'both_train_path_and_file',
                       'nonexistent_train_path',
                       'nonexistent_test_file']:

        if sub_prefix == 'both_train_path_and_file':
            train_fh = tempfile.NamedTemporaryFile(suffix='jsonlines',
                                                   prefix=join(_my_dir,
                                                               'other',
                                                               ('test_config_'
                                                                'parsing_')))
            values_to_fill_dict['train_file'] = train_fh.name
            values_to_fill_dict['train_directory'] = train_dir

        elif sub_prefix == 'nonexistent_train_path':
            values_to_fill_dict['train_directory'] = join(train_dir, 'foo')

        elif sub_prefix == 'nonexistent_test_file':
            values_to_fill_dict['train_file'] = 'foo.jsonlines'

        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                       values_to_fill_dict,
                                                       sub_prefix)

        yield check_config_parsing_value_error, config_path

        if sub_prefix == 'both_train_path_and_file':
            train_fh.close()


def test_config_parsing_bad_test():
    # Test to ensure config file parsing raises an error with invalid test path
    # specifications

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'learners': "['LogisticRegression']",
                           'featuresets': ("[['f1', 'f2', 'f3'], ['f4', 'f5', "
                                           "'f6']]"),
                           'log': output_dir,
                           'results': output_dir}

    for sub_prefix in ['both_test_path_and_file',
                       'nonexistent_test_path',
                       'nonexistent_test_file']:

        if sub_prefix == 'both_test_path_and_file':
            test_fh = tempfile.NamedTemporaryFile(suffix='jsonlines',
                                                  prefix=join(_my_dir,
                                                              'other',
                                                              ('test_config_'
                                                               'parsing_')))
            values_to_fill_dict['test_file'] = test_fh.name
            values_to_fill_dict['test_directory'] = test_dir

        elif sub_prefix == 'nonexistent_test_path':
            values_to_fill_dict['test_directory'] = join(test_dir, 'foo')

        elif sub_prefix == 'nonexistent_test_file':
            values_to_fill_dict['test_file'] = 'foo.jsonlines'

        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                       values_to_fill_dict,
                                                       sub_prefix)

        yield check_config_parsing_value_error, config_path

        if sub_prefix == 'both_test_path_and_file':
            test_fh.close()


@raises(ValueError)
def test_config_parsing_bad_objective():
    """
    Test to ensure config file parsing raises an error with an invalid grid objective
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'foobar'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                   values_to_fill_dict,
                                                   'bad_objective')

    _parse_config_file(config_path)


def test_config_parsing_bad_task_paths():
    # Test to ensure config file parsing raises an error with various
    # incorrectly set path

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'train_directory': train_dir,
                           'learners': "['LogisticRegression']",
                           'featuresets': ("[['f1', 'f2', 'f3'], ['f4', 'f5', "
                                           "'f6']]"),
                           'log': output_dir}

    for sub_prefix in ['predict_no_test', 'evaluate_no_test',
                       'xv_with_test_path', 'train_with_test_path',
                       'xv_with_test_file', 'train_with_test_file',
                       'train_with_results', 'predict_with_results',
                       'train_no_model', 'train_with_predictions',
                       'xv_with_model']:

        if sub_prefix == 'predict_no_test':
            values_to_fill_dict['task'] = 'predict'
            values_to_fill_dict['predictions'] = output_dir

        elif sub_prefix == 'evaluate_no_test':
            values_to_fill_dict['task'] = 'evaluate'
            values_to_fill_dict['results'] = output_dir

        elif sub_prefix == 'xv_with_test_path':
            values_to_fill_dict['task'] = 'cross_validate'
            values_to_fill_dict['results'] = output_dir
            values_to_fill_dict['test_directory'] = test_dir

        elif sub_prefix == 'train_with_test_path':
            values_to_fill_dict['task'] = 'train'
            values_to_fill_dict['models'] = output_dir
            values_to_fill_dict['test_directory'] = test_dir

        elif sub_prefix == 'xv_with_test_file':
            values_to_fill_dict['task'] = 'cross_validate'
            values_to_fill_dict['results'] = output_dir
            test_fh1 = tempfile.NamedTemporaryFile(suffix='jsonlines',
                                                   prefix=join(_my_dir,
                                                               'other',
                                                               ('test_config_'
                                                                'parsing_')))
            values_to_fill_dict['test_file'] = test_fh1.name

        elif sub_prefix == 'train_with_test_file':
            values_to_fill_dict['task'] = 'train'
            values_to_fill_dict['models'] = output_dir
            test_fh2 = tempfile.NamedTemporaryFile(suffix='jsonlines',
                                                   prefix=join(_my_dir,
                                                               'other',
                                                               ('test_config_'
                                                                'parsing_')))

            values_to_fill_dict['test_file'] = test_fh2.name

        elif sub_prefix == 'train_with_results':
            values_to_fill_dict['task'] = 'train'
            values_to_fill_dict['models'] = output_dir
            values_to_fill_dict['results'] = output_dir

        elif sub_prefix == 'predict_with_results':
            values_to_fill_dict['task'] = 'predict'
            values_to_fill_dict['test_directory'] = test_dir
            values_to_fill_dict['predictions'] = output_dir
            values_to_fill_dict['results'] = output_dir

        elif sub_prefix == 'train_no_model':
            values_to_fill_dict['task'] = 'train'

        elif sub_prefix == 'train_with_predictions':
            values_to_fill_dict['task'] = 'train'
            values_to_fill_dict['models'] = output_dir
            values_to_fill_dict['predictions'] = output_dir

        elif sub_prefix == 'xv_with_model':
            values_to_fill_dict['task'] = 'cross_validate'
            values_to_fill_dict['results'] = output_dir
            values_to_fill_dict['models'] = output_dir

        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths_for_parsing(config_template_path,
                                                       values_to_fill_dict,
                                                       sub_prefix)

        yield check_config_parsing_value_error, config_path

        if sub_prefix == 'xv_with_test_file':
            test_fh1.close()

        elif sub_prefix == 'train_with_test_file':
            test_fh2.close()
