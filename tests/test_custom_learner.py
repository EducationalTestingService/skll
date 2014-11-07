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

import csv
import os
import re
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
from numpy.testing import assert_array_equal
from skll.data import FeatureSet, NDJWriter
from skll.experiments import run_configuration, _setup_config_parser
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import make_classification_data

_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
SCORE_OUTPUT_RE = re.compile(r'Objective Function Score \(Test\) = '
                             r'([\-\d\.]+)')
GRID_RE = re.compile(r'Grid Objective Score \(Train\) = ([\-\d\.]+)')
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


def fill_in_config_paths(config_template_path):
    """
    Add paths to train, test, and output directories to a given config template
    file.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path)

    task = config.get("General", "task")
    # experiment_name = config.get("General", "experiment_name")

    config.set("Input", "train_location", train_dir)

    to_fill_in = ['log', 'predictions']

    if task != 'cross_validate':
        to_fill_in.append('models')

    if task == 'evaluate' or task == 'cross_validate':
        to_fill_in.append('results')

    for d in to_fill_in:
        config.set("Output", d, join(output_dir))

    if task == 'cross_validate':
        cv_folds_location = config.get("Input", "cv_folds_location")
        if cv_folds_location:
            config.set("Input", "cv_folds_location",
                       join(train_dir, cv_folds_location))

    if task == 'predict' or task == 'evaluate':
        config.set("Input", "test_location", test_dir)

    # set up custom learner path, if relevant
    custom_learner_path = config.get("Input", "custom_learner_path")
    custom_learner_abs_path = join(_my_dir, custom_learner_path)
    config.set("Input", "custom_learner_path", custom_learner_abs_path)

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}.cfg'.format(config_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def read_predictions(path):
    with open(path) as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        res = np.array([float(x[1]) for x in reader])
    return res


def test_majority_class_custom_learner():
    num_labels = 10

    # This will make data where the last class happens about 50% of the time.
    class_weights = [(0.5 / (num_labels - 1))
                             for x in range(num_labels - 1)] + [0.5]
    train_fs, test_fs = make_classification_data(num_examples=600,
                                                 train_test_ratio=0.8,
                                                 num_labels=num_labels,
                                                 num_features=5,
                                                 non_negative=True,
                                                 class_weights=class_weights)

    # Write training feature set to a file
    train_path = join(_my_dir, 'train',
                      'test_majority_class_custom_learner.jsonlines')
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = join(_my_dir, 'test',
                     'test_majority_class_custom_learner.jsonlines')
    writer = NDJWriter(test_path, test_fs)
    writer.write()

    cfgfile = 'test_majority_class_custom_learner.template.cfg'
    config_template_path = join(_my_dir, 'configs', cfgfile)
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    outprefix = 'test_majority_class_custom_learner'

    preds = read_predictions(
        join(_my_dir, 'output', ('{}_{}_MajorityClassLearner.predictions'
                                 .format(outprefix, outprefix))))
    expected = np.array([float(num_labels - 1) for x in preds])
    assert_array_equal(preds, expected)


def test_logistic_custom_learner():
    num_labels = 10

    class_weights = [(0.5 / (num_labels - 1))
                     for x in range(num_labels - 1)] + [0.5]
    train_fs, test_fs = make_classification_data(num_examples=600,
                                                 train_test_ratio=0.8,
                                                 num_labels=num_labels,
                                                 num_features=5,
                                                 non_negative=True,
                                                 class_weights=class_weights)

    # Write training feature set to a file
    train_path = join(_my_dir, 'train',
                      'test_logistic_custom_learner.jsonlines')
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = join(_my_dir, 'test',
                     'test_logistic_custom_learner.jsonlines')
    writer = NDJWriter(test_path, test_fs)
    writer.write()

    cfgfile = 'test_logistic_custom_learner.template.cfg'
    config_template_path = join(_my_dir, 'configs', cfgfile)
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    outprefix = 'test_logistic_custom_learner'
    preds = read_predictions(join(_my_dir, 'output',
                                  ('{}_{}_CustomLogisticRegressionWrapper.predictions'
                                   .format(outprefix, outprefix))))

    expected = read_predictions(join(_my_dir, 'output',
                                     ('{}_{}_LogisticRegression.predictions'
                                      .format(outprefix, outprefix))))

    assert_array_equal(preds, expected)
