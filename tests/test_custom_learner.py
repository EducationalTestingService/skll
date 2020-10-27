# License: BSD 3 clause
"""
Module for running tests with custom learners.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

import csv
import os
import re
from glob import glob
from os.path import abspath, dirname, exists, join
from tempfile import NamedTemporaryFile

import numpy as np
from nose.tools import raises
from numpy.testing import assert_array_equal

from skll.data import NDJWriter
from skll.experiments import run_configuration
from skll.learner import Learner
from skll.utils.constants import KNOWN_DEFAULT_PARAM_GRIDS
from skll.utils.logging import (get_skll_logger)
from tests.utils import fill_in_config_paths, make_classification_data

_ALL_MODELS = list(KNOWN_DEFAULT_PARAM_GRIDS.keys())
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
    log_dir = join(_my_dir, 'log')
    if not exists(log_dir):
        os.makedirs(log_dir)


def tearDown():
    """
    Clean up after tests.
    """
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')
    log_dir = join(_my_dir, 'log')

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

    for log_file in glob(join(log_dir, '*')):
        os.unlink(log_file)


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
                                  ('{}_{}_MajorityClassLearner_predictions.tsv'
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
                                   '_predictions.tsv'.format(outprefix,
                                                             outprefix))))

    expected = read_predictions(join(_my_dir, 'output',
                                     ('{}_{}_LogisticRegression_predictions.tsv'
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
                     '_predictions.tsv'.format(outprefix, outprefix))
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


@raises(ValueError)
def test_custom_learner_api_missing_file():
    _ = Learner('CustomType')


@raises(ValueError)
def test_custom_learner_api_bad_extension():
    other_dir = join(_my_dir, 'other')
    _ = Learner('_CustomLogisticRegressionWrapper', custom_learner_path=join(other_dir, 'custom_learner.txt'))


@raises(ValueError)
def test_custom_learner_learning_curve_min_examples():
    """
    Test to check learning curve raises error with less than 500 examples
    """
    # generates a training split with less than 500 examples
    train_fs_less_than_500, _ = make_classification_data(num_examples=499,
                                                         train_test_ratio=1.0,
                                                         num_labels=3)

    # creating an example learner
    learner = Learner('LogisticRegression')

    # this must throw an error because `examples` has less than 500 items
    _ = learner.learning_curve(examples=train_fs_less_than_500, metric='accuracy')


def test_custom_learner_learning_curve_min_examples_override():
    """
    Test to check learning curve displays warning with less than 500 examples
    """

    # creates a logger which writes to a temporary log file
    log_dir = join(_my_dir, 'log')
    log_file = NamedTemporaryFile("w", delete=False, dir=log_dir)
    logger = get_skll_logger("test_custom_learner_learning_curve_min_examples_override",
                             filepath=log_file.name)

    # generates a training split with less than 500 examples
    train_fs_less_than_500, _ = make_classification_data(num_examples=499,
                                                         train_test_ratio=1.0,
                                                         num_labels=3)

    # creating an example learner
    learner = Learner('LogisticRegression', logger=logger)

    # this must throw an error because `examples` has less than 500 items
    _ = learner.learning_curve(examples=train_fs_less_than_500, metric='accuracy',
                               override_minimum=True)

    # checks that the learning_curve warning message is contained in the log file
    with open(log_file.name) as tf:
        log_text = tf.read()
        learning_curve_warning_re = \
            re.compile(r'Because the number of training examples provided - '
                       r'\d+ - is less than the ideal minimum - \d+ - '
                       r'learning curve generation is unreliable'
                       r' and might break')
        assert learning_curve_warning_re.search(log_text)
