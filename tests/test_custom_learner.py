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
from glob import glob
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
from numpy.testing import assert_array_equal
from skll.data import NDJWriter
from skll.experiments import run_configuration
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import fill_in_config_paths, make_classification_data

_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
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
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')

    input_files = ['test_logistic_custom_learner.jsonlines',
                   'test_majority_class_custom_learner.jsonlines',
                   'test_model_custom_learner.jsonlines']
    for inf in input_files:
        if exists(join(train_dir, inf)):
            os.unlink(join(train_dir, inf))
        if exists(join(test_dir, inf)):
            os.unlink(join(test_dir, inf))

    for cfg_file in glob(join(config_dir, '*custom_learner.cfg')):
        os.unlink(cfg_file)

    for output_file in (glob(join(output_dir,
                                  'test_logistic_custom_learner_*')) +
                        glob(join(output_dir,
                                  'test_majority_class_custom_learner_*')) +
                        glob(join(output_dir, 'test_model_custom_learner_*'))):
        os.unlink(output_file)


def read_predictions(path):
    """
    Read in prediction file as a numpy array.
    """
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

    preds = read_predictions(join(_my_dir, 'output',
                                  ('{}_{}_MajorityClassLearner.predictions'
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
                                  ('{}_{}_CustomLogisticRegressionWrapper'
                                   '.predictions'.format(outprefix,
                                                         outprefix))))

    expected = read_predictions(join(_my_dir, 'output',
                                     ('{}_{}_LogisticRegression.predictions'
                                      .format(outprefix, outprefix))))

    assert_array_equal(preds, expected)


def test_custom_learner_model_loading():
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
                      'test_model_custom_learner.jsonlines')
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = join(_my_dir, 'test',
                     'test_model_custom_learner.jsonlines')
    writer = NDJWriter(test_path, test_fs)
    writer.write()

    # run the configuration that trains the custom model and saves it
    cfgfile = 'test_model_save_custom_learner.template.cfg'
    config_template_path = join(_my_dir, 'configs', cfgfile)
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    # save the predictions from disk into memory
    # and delete the predictions file
    outprefix = 'test_model_custom_learner'
    pred_file = join(_my_dir, 'output',
                     '{}_{}_CustomLogisticRegressionWrapper'
                     '.predictions'.format(outprefix,
                                           outprefix))
    preds1 = read_predictions(pred_file)
    os.unlink(pred_file)

    # run the configuration that loads the saved model
    # and generates the predictions again
    cfgfile = 'test_model_load_custom_learner.template.cfg'
    config_template_path = join(_my_dir, 'configs', cfgfile)
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, overwrite=False, quiet=True)

    # load the newly generated predictions
    preds2 = read_predictions(pred_file)

    # make sure that they are the same as before
    assert_array_equal(preds1, preds2)
