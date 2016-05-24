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
import tempfile
from glob import glob
from os.path import abspath, dirname, exists, join, normpath, realpath

from nose.tools import eq_, raises

from skll.config import _parse_config_file, _locate_file
from skll.data.readers import safe_float
from skll.experiments import _load_featureset

from utils import fill_in_config_options

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


def test_locate_file_valid_paths1():
    """
    Test that `config.locate_file` works with absolute paths.
    """

    config_abs_path = join(_my_dir, 'configs',
                           'test_config_parsing_relative_path1.cfg')
    open(config_abs_path, 'w').close()
    eq_(_locate_file(config_abs_path, _my_dir),
        join(_my_dir, 'configs', 'test_config_parsing_relative_path1.cfg'))


def test_locate_file_valid_paths2():
    """
    Test that `config.locate_file` works with relative paths.
    """

    config_abs_path = join(_my_dir, 'configs',
                           'test_config_parsing_relative_path2.cfg')
    config_rel_path = 'configs/test_config_parsing_relative_path2.cfg'
    open(config_abs_path, 'w').close()
    eq_(_locate_file(config_rel_path, _my_dir), config_abs_path)


def test_locate_file_valid_paths3():
    """
    Test that `config.locate_file` works with relative/absolute paths.
    """

    config_abs_path = join(_my_dir, 'configs',
                           'test_config_parsing_relative_path3.cfg')
    config_rel_path = 'configs/test_config_parsing_relative_path3.cfg'
    open(config_abs_path, 'w').close()
    eq_(_locate_file(config_abs_path, _my_dir),
        _locate_file(config_rel_path, _my_dir))


@raises(IOError)
def test_locate_file_invalid_path():
    """
    Test that `config.locate_file` raises an error for paths that do not
    exist.
    """

    _locate_file('test/does_not_exist.cfg', _my_dir)


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
def check_config_parsing_value_error(config_path):
    """
    Assert that calling `_parse_config_file` on `config_path` raises ValueError
    """
    _parse_config_file(config_path)


@raises(TypeError)
def check_config_parsing_type_error(config_path):
    """
    Assert that calling `_parse_config_file` on `config_path` raises TypeError
    """
    _parse_config_file(config_path)


@raises(KeyError)
def check_config_parsing_key_error(config_path):
    """
    Assert that calling `_parse_config_file` on `config_path` raises KeyError
    """
    _parse_config_file(config_path)


@raises(IOError)
def check_config_parsing_file_not_found_error(config_path):
    """
    Assert that calling `_parse_config_file` on `config_path` raises FileNotFoundError
    """
    _parse_config_file(config_path)


@raises(IOError)
def test_empty_config_name_raises_file_not_found_error():
    """
    Assert that calling _parse_config_file on an empty string raises IOError
    """
    _parse_config_file("")


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
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'no_name')

    yield check_config_parsing_value_error, config_path


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
        config_path = fill_in_config_options(config_template_path,
                                             values_to_fill_dict,
                                             sub_prefix)

        yield check_config_parsing_value_error, config_path


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
        config_path = fill_in_config_options(config_template_path,
                                             values_to_fill_dict,
                                             sub_prefix)
        yield check_config_parsing_value_error, config_path


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
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'bad_sampler')

    yield check_config_parsing_value_error, config_path


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
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'bad_hashing')

    yield check_config_parsing_value_error, config_path


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
        config_path = fill_in_config_options(config_template_path,
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
        config_path = fill_in_config_options(config_template_path,
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
        config_path = fill_in_config_options(config_template_path,
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
        config_path = fill_in_config_options(config_template_path,
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
        config_path = fill_in_config_options(config_template_path,
                                             values_to_fill_dict,
                                             sub_prefix)

        yield check_config_parsing_value_error, config_path

        if sub_prefix == 'both_test_path_and_file':
            test_fh.close()


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
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'bad_objective')

    yield check_config_parsing_value_error, config_path


def test_config_parsing_bad_objective_2():
    """
    Test to ensure config file parsing raises an error with a grid objective given as a list
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
                           'objective': "['accuracy']"}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'bad_objective')

    yield check_config_parsing_type_error, config_path


def test_config_parsing_bad_objectives():
    """
    Test to ensure config file parsing raises an error with a grid objectives given as a string
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objectives': "accuracy"}
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'bad_objectives')
    yield check_config_parsing_type_error, config_path



def test_config_parsing_bad_objective_and_objectives():
    """
    Test to ensure config file parsing raises an error with
    a grid objectives and objective both given non default values
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    values_to_fill_dict = {'train_directory': train_dir,
                           'log': output_dir,
                           'results': output_dir,
                           'objectives': "['accuracy']",
                           'objective': "accuracy"}
    config_path = fill_in_config_options(config_template_path,
                           values_to_fill_dict,
                           'bad_objective_and_objectives')
    yield check_config_parsing_value_error, config_path


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
        config_path = fill_in_config_options(config_template_path,
                                             values_to_fill_dict,
                                             sub_prefix)

        yield check_config_parsing_value_error, config_path

        if sub_prefix == 'xv_with_test_file':
            test_fh1.close()

        elif sub_prefix == 'train_with_test_file':
            test_fh2.close()


def test_config_parsing_bad_cv_folds():
    """
    Test to ensure config file parsing raises an error with an invalid cv_folds
    """

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad value for cv_folds
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'cross_validate',
                           'train_directory': train_dir,
                           'num_cv_folds': 'random',
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'bad_cv_folds')

    yield check_config_parsing_value_error, config_path


def test_config_parsing_invalid_option():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has an invalid option
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'cross_validate',
                           'train_directory': train_dir,
                           'bad_option': 'whatever',
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'invalid_option')

    yield check_config_parsing_key_error, config_path


def test_config_parsing_duplicate_option():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a duplicate option
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'cross_validate',
                           'train_directory': train_dir,
                           'duplicate_option': 'value',
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'duplicate_option')

    yield check_config_parsing_key_error, config_path


def test_config_parsing_option_in_wrong_section():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has an option in the wrong section
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'cross_validate',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'probability': 'true',
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'option_in_wrong_section')

    yield check_config_parsing_key_error, config_path


def test_config_parsing_mislocated_input_path():

    train_dir = 'train'
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has an invalid option
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'cross_validate',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'mislocated_input_file')

    yield check_config_parsing_file_not_found_error, config_path


def test_config_parsing_relative_input_path():

    train_dir = join('..', 'train')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has an invalid option
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'cross_validate',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'mislocated_input_file')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path) = _parse_config_file(config_path)

    eq_(normpath(train_path), (join(_my_dir, 'train')))


def test_config_parsing_relative_input_paths():

    train_dir = '../train'
    train_file = join(train_dir, 'f0.jsonlines')
    test_file = join(train_dir, 'f1.jsonlines')
    output_dir = '../output'
    custom_learner_path_input = join('other', 'majority_class_learner.py')

    # make a simple config file that has relative paths
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_file': train_file,
                           'test_file': test_file,
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_micro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_relative_paths.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'relative_paths')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path) = _parse_config_file(config_path)


def test_default_number_of_cv_folds():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that does not set cv_folds

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'cross_validate',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'default_cv_folds')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path) = _parse_config_file(config_path)

    eq_(cv_folds, 10)


def test_setting_number_of_cv_folds():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')
    # make a simple config file that does not set cv_folds

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'cross_validate',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'num_cv_folds': "5",
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'default_cv_folds')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path) = _parse_config_file(config_path)

    eq_(cv_folds, 5)


def test_setting_param_grids():

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that does not set cv_folds

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LinearSVC']",
                           'log': output_dir,
                           'results': output_dir,
                           'param_grids': "[{'C': [1e-6, 0.001, 1, 10, 100, 1e5]}]",
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'param_grids')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path) = _parse_config_file(config_path)

    eq_(param_grid_list[0]['C'][0], 1e-6)
    eq_(param_grid_list[0]['C'][1], 1e-3)
    eq_(param_grid_list[0]['C'][2], 1)
    eq_(param_grid_list[0]['C'][3], 10)
    eq_(param_grid_list[0]['C'][4], 100)
    eq_(param_grid_list[0]['C'][5], 1e5)


def test_setting_fixed_parameters():

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that does not set cv_folds

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_directory': train_dir,
                           'test_directory': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LinearSVC']",
                           'log': output_dir,
                           'results': output_dir,
                           'fixed_parameters': "[{'C': [1e-6, 0.001, 1, 10, 100, 1e5]}]",
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'fixed_parameters')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path) = _parse_config_file(config_path)

    eq_(fixed_parameter_list[0]['C'][0], 1e-6)
    eq_(fixed_parameter_list[0]['C'][1], 1e-3)
    eq_(fixed_parameter_list[0]['C'][2], 1)
    eq_(fixed_parameter_list[0]['C'][3], 10)
    eq_(fixed_parameter_list[0]['C'][4], 100)
    eq_(fixed_parameter_list[0]['C'][5], 1e5)
