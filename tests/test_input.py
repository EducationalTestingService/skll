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

import glob
import os
import re
import tempfile

from io import open
from os.path import abspath, dirname, exists, join

from nose.tools import eq_, raises, assert_raises
from skll.experiments import _load_featureset, _setup_config_parser, _parse_config_file

_my_dir = abspath(dirname(__file__))


def setup():
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
    config_dir = join(_my_dir, 'configs')
    for config_file in glob.glob(join(config_dir, 'test_config_parsing_*.cfg')):
        os.unlink(config_file)


def fill_in_config_paths(config_template_path, values_to_fill_dict, sub_prefix):
    """
    Add paths to train, test, and output directories to a given config template
    file.
    """

    config = _setup_config_parser(config_template_path)

    to_fill_in = {'General': ['experiment_name', 'task'],
                  'Input': ['train_location', 'train_file',
                            'test_location', 'test_file', 'featuresets',
                            'featureset_names', 'feature_hasher', 'hasher_features',
                            'learners', 'sampler', 'shuffle'],
                  'Tuning': ['grid_search', 'objective', 'feature_scaling'],
                  'Output': ['probability', 'results', 'log', 'models',
                             'predictions']}

    for section in to_fill_in:
        for param_name in to_fill_in[section]:
            if param_name in values_to_fill_dict:
                config.set(section, param_name, values_to_fill_dict[param_name])

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
    values_to_fill_dict = {'train_location': train_dir,
                           'test_location': test_dir,
                           'task': 'evaluate',
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'grid_search': 'False',
                           'feature_scaling': 'none'}

    config_template_path = join(_my_dir, 'configs',
                                'test_config_parsing.template.cfg')
    config_path = fill_in_config_paths(config_template_path,
                                       values_to_fill_dict, 'no_name')

    _parse_config_file(config_path)


def test_config_parsing_bad_task():
    """
    Test to ensure config file parsing raises an error with invalid or missing tasks
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'train_location': train_dir,
                           'test_location': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'results': output_dir,
                           'grid_search': 'False',
                           'feature_scaling': 'none'}

    for task_value, sub_prefix in zip([None, '', 'procrastinate'],
                                      ['no_task', 'missing_task', 'bad_task']):
        if task_value != None:
            values_to_fill_dict['task'] = task_value
        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths(config_template_path,
                                           values_to_fill_dict, sub_prefix)

        assert_raises(ValueError, _parse_config_file, config_path)


def test_config_parsing_bad_learner():
    """
    Test to ensure config file parsing raises an error with missing, bad and duplicate learners
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    # make a simple config file that has a bad task
    # but everything else is correct
    values_to_fill_dict = {'experiment_name': 'config_parsing',
                           'task': 'evaluate',
                           'train_location': train_dir,
                           'test_location': test_dir,
                           'featuresets': "[['f1', 'f2', 'f3']]",
                           'log': output_dir,
                           'results': output_dir,
                           'grid_search': 'False',
                           'feature_scaling': 'none'}

    for learners_list, sub_prefix in zip([None, '[]', 'LogisticRegression',
                                          "['LogisticRegression', 'LogisticRegression']"],
                                         ['no_learner', 'empty_learner',
                                          'not_list_learner', 'duplicate_learner']):
        if learners_list != None:
            values_to_fill_dict['learners'] = learners_list

        config_template_path = join(_my_dir, 'configs',
                                    'test_config_parsing.template.cfg')
        config_path = fill_in_config_paths(config_template_path,
                                           values_to_fill_dict, sub_prefix)

        # _parse_config_file(config_path)
        assert_raises(ValueError, _parse_config_file, config_path)

