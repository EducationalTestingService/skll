# License: BSD 3 clause
"""
Tests for SKLL inputs, mainly configuration files.

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
from itertools import product
from os.path import abspath, dirname, exists, join, normpath
import numpy as np

from nose.tools import eq_, ok_, raises
from sklearn.utils.testing import assert_equal

from six import string_types, PY2

from skll.config import (_parse_config_file,
                         _load_cv_folds,
                         _locate_file)
from skll.data.readers import safe_float
from skll.experiments import _load_featureset

from utils import (create_jsonlines_feature_files,
                   fill_in_config_options)

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

    # create jsonlines feature files
    train_path = join(_my_dir, 'train')
    create_jsonlines_feature_files(train_path)


def tearDown():
    """
    Clean up after tests.
    """
    config_dir = join(_my_dir, 'configs')
    for config_file in glob(join(config_dir, 'test_config_parsing_*.cfg')):
        os.unlink(config_file)
    for auto_dir in glob(join(_my_dir, 'auto*')):
        for auto_dir_file in os.listdir(auto_dir):
            os.unlink(join(auto_dir, auto_dir_file))
        os.rmdir(auto_dir)


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
    Test to ensure config file parsing raises an error with grid objectives given as a string
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


def test_config_parsing_bad_metric():
    """
    Test to ensure config file parsing raises an error with an invalid evaluation metric
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
                           'metrics': "['foobar', 'accuracy']"}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'bad_metric')

    yield check_config_parsing_value_error, config_path


def test_config_parsing_bad_metric_2():
    """
    Test to ensure config file parsing raises an error with metrics given as a string
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
                           'metrics': "accuracy"}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'bad_metric_as_string')

    yield check_config_parsing_type_error, config_path


def test_config_parsing_log_loss_no_probability():
    """
    Test that config parsing raises an error if log loss is used without probability
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
                           'objective': 'neg_log_loss'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'log_loss_no_probability')

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


def test_config_parsing_mse_to_neg_mse():

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
                           'objective': 'mean_squared_error'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'mse_to_neg_mse')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(grid_objective, ['neg_mean_squared_error'])


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
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(normpath(train_path), (join(_my_dir, 'train')))


def test_config_parsing_relative_input_paths():

    train_dir = '../train'
    train_file = join(train_dir, 'f0.jsonlines')
    test_file = join(train_dir, 'f1.jsonlines')
    output_dir = '../output'

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
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)


def test_config_parsing_automatic_output_directory_creation():

    train_dir = '../train'
    train_file = join(train_dir, 'f0.jsonlines')
    test_file = join(train_dir, 'f1.jsonlines')
    output_dir = '../output'

    # make a simple config file that has new directories that should
    # be automatically created
    new_log_path = join(_my_dir, 'autolog')
    new_results_path = join(_my_dir, 'autoresults')
    new_models_path = join(_my_dir, 'automodels')
    new_predictions_path = join(_my_dir, 'autopredictions')

    ok_(not(exists(new_log_path)))
    ok_(not(exists(new_results_path)))
    ok_(not(exists(new_models_path)))
    ok_(not(exists(new_predictions_path)))

    values_to_fill_dict = {'experiment_name': 'auto_dir_creation',
                           'task': 'evaluate',
                           'train_file': train_file,
                           'test_file': test_file,
                           'learners': "['LogisticRegression']",
                           'log': new_log_path,
                           'results': new_results_path,
                           'models': new_models_path,
                           'predictions': new_predictions_path,
                           'objective': 'f1_score_micro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_relative_paths.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'auto_dir_creation')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    ok_(exists(new_log_path))
    ok_(exists(new_results_path))
    ok_(exists(new_models_path))
    ok_(exists(new_predictions_path))


def check_config_parsing_metrics_and_objectives_overlap(task,
                                                        metrics,
                                                        objectives):

    test_dir = join('..', 'test')
    train_dir = join('..', 'train')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has an invalid option
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': task,
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'metrics': str(metrics)}

    if task == 'evaluate':
        values_to_fill_dict['test_directory'] = test_dir

    if objectives:
        values_to_fill_dict['objectives'] = str(objectives)

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'metrics_and_objectives_overlap')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, parsed_objectives, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, parsed_metrics) = _parse_config_file(config_path)

    if not objectives:
        objectives = ["f1_score_micro"]
    common_metrics = set(objectives).intersection(metrics)
    pruned_metrics = [metric for metric in metrics if metric not in common_metrics]
    eq_(parsed_objectives, objectives)
    eq_(parsed_metrics, pruned_metrics)


def test_config_parsing_metrics_and_objectives_overlap():

    for task, metrics, objectives in product(["evaluate", "cross_validate"],
                                             [["f1_score_micro", "unweighted_kappa"],
                                              ["accuracy", "unweighted_kappa"]],
                                             [[], ["accuracy"]]):
        metrics = [str(m) for m in metrics] if PY2 else metrics
        objectives = [str(o) for o in objectives] if PY2 else objectives
        yield check_config_parsing_metrics_and_objectives_overlap, \
                task, metrics, objectives


def test_cv_folds_and_grid_search_folds():

    # we want to test all possible combinations of the following variables:
    #  task = train, cross_validate
    #  cv_folds/folds_file = not specified, number, csv file
    #  grid_search_folds = not specified, number
    #  use_folds_file_for_grid_search = not specified, True, False

    # below is a table of what we expect for each of the combinations
    # note: `fold_mapping` refers to the dictionary version of the folds file

    # task, cv_folds/folds_file, grid_search_folds, use_folds_file_for_grid_search -> cv_folds, grid_search_folds
    # ('train', None, None, None) ->  (None, 3)
    # ('train', None, None, True) ->  (None, 3)
    # ('train', None, None, False) ->  (None, 3)
    # ('train', None, 7, None) ->  (None, 7)
    # ('train', None, 7, True) ->  (None, 7)
    # ('train', None, 7, False) ->  (None, 7)
    # ('train', 5, None, None) ->  (None, 3)
    # ('train', 5, None, True) ->  (None, 3)
    # ('train', 5, None, False) ->   (None, 3)
    # ('train', 5, 7, None) ->  (None, 7)
    # ('train', 5, 7, True) ->  (None, 7)
    # ('train', 5, 7, False) ->  (None, 7)
    # ('train', 'train/folds_file_test.csv', None, None) ->  (None, fold_mapping)
    # ('train', 'train/folds_file_test.csv', None, True) ->  (None, fold_mapping)
    # ('train', 'train/folds_file_test.csv', None, False) ->  (None, fold_mapping)
    # ('train', 'train/folds_file_test.csv', 7, None) ->  (None, fold_mapping)
    # ('train', 'train/folds_file_test.csv', 7, True) ->  (None, fold_mapping)
    # ('train', 'train/folds_file_test.csv', 7, False) ->  (None, fold_mapping)
    # ('cross_validate', None, None, None) ->  (10, 3)
    # ('cross_validate', None, None, True) ->  (10, 3)
    # ('cross_validate', None, None, False) ->  (10, 3)
    # ('cross_validate', None, 7, None) ->  (10, 7)
    # ('cross_validate', None, 7, True) ->  (10, 7)
    # ('cross_validate', None, 7, False) ->  (10, 7)
    # ('cross_validate', 5, None, None) ->  (5, 3)
    # ('cross_validate', 5, None, True) ->  (5, 3)
    # ('cross_validate', 5, None, False) ->  (5, 3)
    # ('cross_validate', 5, 7, None) ->  (5, 7)
    # ('cross_validate', 5, 7, True) ->  (5, 7)
    # ('cross_validate', 5, 7, False) ->  (5, 7)
    # ('cross_validate', 'train/folds_file_test.csv', None, None) ->  (fold_mapping, fold_mapping)
    # ('cross_validate', 'train/folds_file_test.csv', None, True) ->  (fold_mapping, fold_mapping)
    # ('cross_validate', 'train/folds_file_test.csv', None, False) ->  (fold_mapping, 3)
    # ('cross_validate', 'train/folds_file_test.csv', 7, None) ->  (fold_mapping, fold_mapping)
    # ('cross_validate', 'train/folds_file_test.csv', 7, True) ->  (fold_mapping, fold_mapping)
    # ('cross_validate', 'train/folds_file_test.csv', 7, False) ->  (fold_mapping, 7)

    # note that we are passing the string 'fold_mapping' instead of passing in the
    # actual fold mapping dictionary since we don't want it printed in the test log

    for ((task,
          cv_folds_or_file,
          grid_search_folds,
          use_folds_file_for_grid_search),
         (chosen_cv_folds,
          chosen_grid_search_folds)) in zip(product(['train', 'cross_validate'],
                                                  [None, 5, join(_my_dir, 'train/folds_file_test.csv')],
                                                  [None, 7],
                                                  [None, True, False]),
                                            [(None, 3),  (None, 3), (None, 3),
                                             (None, 7), (None, 7), (None, 7),
                                             (None, 3), (None, 3), (None, 3),
                                             (None, 7), (None, 7), (None, 7),
                                             (None, 'fold_mapping'), (None, 'fold_mapping'),
                                             (None, 'fold_mapping'), (None, 'fold_mapping'),
                                             (None, 'fold_mapping'), (None, 'fold_mapping'),
                                             (10, 3), (10, 3), (10, 3), (10, 7),
                                             (10, 7), (10, 7), (5, 3), (5, 3),
                                             (5, 3), (5, 7), (5, 7), (5, 7),
                                             ('fold_mapping', 'fold_mapping'),
                                             ('fold_mapping', 'fold_mapping'),
                                             ('fold_mapping', 3),
                                             ('fold_mapping', 'fold_mapping'),
                                             ('fold_mapping', 'fold_mapping'),
                                             ('fold_mapping', 7)]):

         yield check_cv_folds_and_grid_search_folds, task, cv_folds_or_file, \
                    grid_search_folds, use_folds_file_for_grid_search, \
                    chosen_cv_folds, chosen_grid_search_folds


def check_cv_folds_and_grid_search_folds(task,
                                         cv_folds_or_file,
                                         grid_search_folds,
                                         use_folds_file_for_grid_search,
                                         chosen_cv_folds,
                                         chosen_grid_search_folds):

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')


    # read in the folds file into a dictionary and replace the string
    # 'fold_mapping' with this dictionary.
    fold_mapping = _load_cv_folds(join(_my_dir, 'train/folds_file_test.csv'), ids_to_floats=False)
    if chosen_grid_search_folds == 'fold_mapping':
        chosen_grid_search_folds = fold_mapping
    if chosen_cv_folds == 'fold_mapping':
        chosen_cv_folds = fold_mapping

    # make a simple config file
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': task,
                           'grid_search': 'true',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'objective': 'f1_score_macro'}

    # we need the models field when training but the results field
    # when cross-validating
    if task == 'train':
        values_to_fill_dict['models'] = output_dir
    elif task == 'cross_validate':
        values_to_fill_dict['results'] = output_dir

    # now add the various fields that are passed in
    if isinstance(cv_folds_or_file, int):
        values_to_fill_dict['num_cv_folds'] = str(cv_folds_or_file)
    elif isinstance(cv_folds_or_file, string_types):
        values_to_fill_dict['folds_file'] = cv_folds_or_file

    if isinstance(grid_search_folds, int):
        values_to_fill_dict['grid_search_folds'] = str(grid_search_folds)

    if isinstance(use_folds_file_for_grid_search, bool):
        values_to_fill_dict['use_folds_file_for_grid_search'] = \
                                str(use_folds_file_for_grid_search).lower()

    config_template_path = join(_my_dir,
                                'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'test_cv_and_grid_search_folds')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(cv_folds, chosen_cv_folds)
    eq_(grid_search_folds, chosen_grid_search_folds)


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
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

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
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

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
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

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
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(fixed_parameter_list[0]['C'][0], 1e-6)
    eq_(fixed_parameter_list[0]['C'][1], 1e-3)
    eq_(fixed_parameter_list[0]['C'][2], 1)
    eq_(fixed_parameter_list[0]['C'][3], 10)
    eq_(fixed_parameter_list[0]['C'][4], 100)
    eq_(fixed_parameter_list[0]['C'][5], 1e5)


def test_default_learning_curve_options():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'learning_curve',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression', 'MultinomialNB']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'default_learning_curve')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(learning_curve_cv_folds_list, [10, 10])
    ok_(np.all(learning_curve_train_sizes == np.linspace(0.1, 1.0, 5)))


def test_setting_learning_curve_options():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'learning_curve',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression', 'MultinomialNB']",
                           'log': output_dir,
                           'results': output_dir,
                           'learning_curve_cv_folds_list': "[100, 10]",
                           'learning_curve_train_sizes': "[10, 50, 100, 200, 500]",
                           'objective': 'f1_score_macro'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'setting_learning_curve')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(learning_curve_cv_folds_list, [100, 10])
    eq_(learning_curve_train_sizes, [10, 50, 100, 200, 500])


def test_learning_curve_metrics_and_objectives():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'learning_curve',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression', 'MultinomialNB']",
                           'log': output_dir,
                           'results': output_dir,
                           'objective': 'f1_score_macro',
                           'metrics': '["accuracy", "f1_score_micro"]'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'learning_curve_metrics_and_objectives')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(output_metrics, ["accuracy", "f1_score_micro"])
    eq_(grid_objective, [])


def test_learning_curve_metrics_and_no_objectives():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'learning_curve',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression', 'MultinomialNB']",
                           'log': output_dir,
                           'results': output_dir,
                           'metrics': '["accuracy", "unweighted_kappa"]'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'learning_curve_metrics_and_no_objectives')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(output_metrics, ["accuracy", "unweighted_kappa"])
    eq_(grid_objective, [])


def test_learning_curve_objectives_and_no_metrics():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'learning_curve',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression', 'MultinomialNB']",
                           'log': output_dir,
                           'results': output_dir,
                           'objectives': '["accuracy"]'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'learning_curve_objectives_and_no_metrics')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(output_metrics, ["accuracy"])
    eq_(grid_objective, [])


def test_learning_curve_default_objectives_and_no_metrics():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'learning_curve',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression', 'MultinomialNB']",
                           'log': output_dir,
                           'results': output_dir}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'learning_curve_default_objectives_and_no_metrics')

    (experiment_name, task, sampler, fixed_sampler_parameters,
     feature_hasher, hasher_features, id_col, label_col, train_set_name,
     test_set_name, suffix, featuresets, do_shuffle, model_path,
     do_grid_search, grid_objective, probability, results_path,
     pos_label_str, feature_scaling, min_feature_count, folds_file,
     grid_search_jobs, grid_search_folds, cv_folds, save_cv_folds,
     use_folds_file_for_grid_search, do_stratified_folds,
     fixed_parameter_list, param_grid_list, featureset_names, learners,
     prediction_dir, log_path, train_path, test_path, ids_to_floats,
     class_map, custom_learner_path, learning_curve_cv_folds_list,
     learning_curve_train_sizes, output_metrics) = _parse_config_file(config_path)

    eq_(output_metrics, ["f1_score_micro"])
    eq_(grid_objective, [])


def test_learning_curve_no_metrics_and_no_objectives():

    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'learning_curve',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression', 'MultinomialNB']",
                           'log': output_dir,
                           'results': output_dir,
                           'objectives': '[]'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'learning_curve_no_metrics_and_no_objectives')

    yield check_config_parsing_value_error, config_path


def test_config_parsing_param_grids_no_grid_search():
    """
    Test to check whether a warning is logged if parameter grids are
    provided but `grid_search` is off.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name':
                               'config_parsing_param_grids_no_grid_search',
                           'task': 'train',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LinearSVC']",
                           'log': output_dir,
                           'models': output_dir,
                           'param_grids': "[{'C': [1e-6, 0.001, 1, 10, 100, 1e5]}]"}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'param_grids_no_grid_search')

    _parse_config_file(config_path)
    log_path = join(output_dir, "config_parsing_param_grids_no_grid_search.log")
    with open(log_path) as f:
        warning_pattern = re.compile('Since "grid_search" is set to False, '
                                     'the specified "param_grids" will be '
                                     'ignored.')
        matches = re.findall(warning_pattern, f.read())
        assert_equal(len(matches), 1)


def test_config_parsing_param_grids_fixed_parameters_conflict():
    """
    Test to check whether a warning is logged if parameter grids are
    provided in addition to fixed parameters.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name':
                               'config_parsing_param_grids_fixed_parameters_conflict',
                           'task': 'train',
                           'train_directory': train_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LinearSVC']",
                           'log': output_dir,
                           'models': output_dir,
                           'grid_search': 'true',
                           'fixed_parameters': "[{'C': 0.001}]",
                           'param_grids': "[{'C': [1e-6, 0.001, 1, 10, 100, 1e5]}]"}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'param_grids_no_grid_search')

    _parse_config_file(config_path)
    log_path = join(output_dir,
                    "config_parsing_param_grids_fixed_parameters_conflict.log")
    with open(log_path) as f:
        warning_pattern = \
            re.compile('Note that "grid_search" is set to True and '
                       '"fixed_parameters" is also specified. If there '
                       'is a conflict between the grid search parameter '
                       'space and the fixed parameter values, the fixed '
                       'parameter values will take precedence.')
        matches = re.findall(warning_pattern, f.read())
        assert_equal(len(matches), 1)
