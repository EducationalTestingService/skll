# License: BSD 3 clause
"""
Module containing tests for custom metrics.

:author: Nitin Madnani (nmadnani@ets.org)
"""

import json
import os
from glob import glob
from os.path import abspath, dirname, join

from nose.tools import assert_almost_equal, eq_, ok_, raises

from sklearn.metrics import fbeta_score, SCORERS
from skll import Learner, run_configuration
from skll.data import NDJReader
from skll.metrics import _CUSTOM_METRICS, register_custom_metric, use_score_func
from tests.utils import fill_in_config_paths_for_single_file

_my_dir = abspath(dirname(__file__))


def setup():
    """
    Create necessary directories for testing.
    """
    pass


def tearDown():
    """
    Clean up after tests.
    """
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')

    for cfg_file in glob(join(config_dir, '*custom_metrics.cfg')):
        os.unlink(cfg_file)

    for output_file in glob(join(output_dir, 'test_custom_metrics*')):
        os.unlink(output_file)


def test_register_custom_metric_load_one():
    """Test loading a single custom metric"""

    # load a single metric from a custom metric file
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")

    # make sure that this metric is now registered with SKLL
    assert "f075_macro" in _CUSTOM_METRICS
    assert "f075_macro" in SCORERS

    # make sure that the other metric in that same file
    # is _not_ registered with SKLL since we didn't ask for it
    assert "ratio_of_ones" not in _CUSTOM_METRICS
    assert "ratio_of_ones" not in SCORERS


def test_register_custom_metric_load_both():
    """Test loading two custom metrics from one file"""

    # load both metrics in the custom file
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")
    register_custom_metric(custom_metrics_file, "ratio_of_ones")

    # now make sure that both are registered
    assert "f075_macro" in _CUSTOM_METRICS
    assert "f075_macro" in SCORERS
    assert "ratio_of_ones" in _CUSTOM_METRICS
    assert "ratio_of_ones" in SCORERS


def test_register_custom_metric_load_different_files():
    """Test loading two custom metrics from two files"""

    # load two custom metrics from two different files
    metric_dir = join(_my_dir, "other")
    custom_metrics_file1 = join(metric_dir, "custom_metrics.py")
    custom_metrics_file2 = join(metric_dir, "custom_metrics2.py")
    register_custom_metric(custom_metrics_file1, "f075_macro")
    register_custom_metric(custom_metrics_file2, "f06_micro")

    # make sure both are registered
    assert "f075_macro" in _CUSTOM_METRICS
    assert "f075_macro" in SCORERS
    assert "f06_micro" in _CUSTOM_METRICS
    assert "f06_micro" in SCORERS


@raises(ValueError)
def test_register_custom_metric_bad_extension():
    """Test loading custom metric from non-py file"""

    # try to load a metric from a txt file, not a py file
    metric_dir = join(_my_dir, "other")
    bad_custom_metrics_file = join(metric_dir, "custom_metrics.txt")
    register_custom_metric(bad_custom_metrics_file, "f075_macro")


@raises(ValueError)
def test_register_custom_metric_missing_file():
    """Test loading custom metric from missing file"""

    # try to load a metric from a py file that does not exist
    metric_dir = join(_my_dir, "other")
    missing_custom_metrics_file = join(metric_dir, "missing_metrics.py")
    register_custom_metric(missing_custom_metrics_file, "f075_macro")


@raises(AttributeError)
def test_register_custom_metric_wrong_name():
    """Test loading custom metric with wrong name"""

    # try to load a metric that does not exist in a file
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "blah")


def test_register_custom_metric_values():
    """Test to check values of custom metrics"""

    # register two metrics in the same file
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")
    register_custom_metric(custom_metrics_file, "ratio_of_ones")

    # check that the values that SKLL would compute matches what we expect
    y_true = [1, 1, 1, 0, 2, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 1, 2]
    skll_value = use_score_func("f075_macro", y_true, y_pred)
    sklearn_value = fbeta_score(y_true, y_pred, 0.75, average='macro')
    eq_(skll_value, sklearn_value)

    y_true = [1, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    skll_value = use_score_func("ratio_of_ones", y_true, y_pred)
    true_ones = len([true for true in y_true if true == 1])
    pred_ones = len([pred for pred in y_pred if pred == 1])
    expected_value = pred_ones / (true_ones + pred_ones)
    eq_(skll_value, expected_value)


def test_custom_metric_api_experiment():
    """Test API SKLL experiment with custom metrics"""

    # register two different metrics from two files
    input_dir = join(_my_dir, "other")
    custom_metrics_file1 = join(input_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file1, "f075_macro")
    custom_metrics_file2 = join(input_dir, "custom_metrics2.py")
    register_custom_metric(custom_metrics_file2, "f06_micro")

    # read in some train/test data
    train_file = join(input_dir, "examples_train.jsonlines")
    test_file = join(input_dir, "examples_test.jsonlines")

    train_fs = NDJReader.for_path(train_file).read()
    test_fs = NDJReader.for_path(test_file).read()

    # set up a learner to tune using one of the custom metrics
    # and evaluate it using the other one
    learner = Learner("LogisticRegression")
    _ = learner.train(train_fs, grid_objective="f075_macro")
    results = learner.evaluate(test_fs,
                               grid_objective="f075_macro",
                               output_metrics=["balanced_accuracy", "f06_micro"])
    test_objective_value = results[-2]
    test_output_metrics_dict = results[-1]
    test_accuracy_value = test_output_metrics_dict["balanced_accuracy"]
    test_f06_micro_value = test_output_metrics_dict["f06_micro"]

    # check that the values are as expected
    assert_almost_equal(test_objective_value, 0.9785, places=4)
    assert_almost_equal(test_accuracy_value, 0.9792, places=4)
    assert_almost_equal(test_f06_micro_value, 0.98, places=4)


def test_custom_metric_config_experiment():
    """Test config SKLL experiment with custom metrics"""

    # Run experiment
    input_dir = join(_my_dir, "other")
    train_file = join(input_dir, "examples_train.jsonlines")
    test_file = join(input_dir, "examples_test.jsonlines")
    config_path = fill_in_config_paths_for_single_file(join(_my_dir, "configs",
                                                            "test_custom_metrics"
                                                            ".template.cfg"),
                                                       train_file,
                                                       test_file)
    run_configuration(config_path, quiet=True)

    # Check results for objective functions and output metrics

    # objective function f075_macro
    with open(join(_my_dir, 'output', ('test_custom_metrics_train_'
                                       'examples_train.jsonlines_test_'
                                       'examples_test.jsonlines_'
                                       'LogisticRegression.results.json'))) as f:
        result_dict = json.load(f)[0]

    test_objective_value = result_dict['score']
    test_output_metrics_dict = result_dict['additional_scores']
    test_accuracy_value = test_output_metrics_dict["balanced_accuracy"]
    test_f06_micro_value = test_output_metrics_dict["f06_micro"]

    # check that the values are as expected
    assert_almost_equal(test_objective_value, 0.9785, places=4)
    assert_almost_equal(test_accuracy_value, 0.9792, places=4)
    assert_almost_equal(test_f06_micro_value, 0.98, places=4)
