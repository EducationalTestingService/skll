# License: BSD 3 clause
"""
Module for running tests for custom metrics.

:author: Nitin Madnani (nmadnani@ets.org)
"""

from os.path import abspath, dirname, join

from nose.tools import assert_almost_equal, eq_, ok_, raises

from sklearn.metrics import fbeta_score, SCORERS
from skll import Learner
from skll.data import NDJReader
from skll.metrics import _CUSTOM_METRICS, register_custom_metric, use_score_func

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
    pass


def test_register_custom_metric_load_one():
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")
    assert "f075_macro" in _CUSTOM_METRICS
    assert "f075_macro" in SCORERS

    assert "ratio_of_ones" not in _CUSTOM_METRICS
    assert "ratio_of_ones" not in SCORERS


def test_register_custom_metric_load_both():
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")
    register_custom_metric(custom_metrics_file, "ratio_of_ones")
    assert "f075_macro" in _CUSTOM_METRICS
    assert "f075_macro" in SCORERS
    assert "ratio_of_ones" in _CUSTOM_METRICS
    assert "ratio_of_ones" in SCORERS


@raises(ValueError)
def test_register_custom_metric_bad_extension():
    metric_dir = join(_my_dir, "other")
    bad_custom_metrics_file = join(metric_dir, "custom_metrics.txt")
    register_custom_metric(bad_custom_metrics_file, "f075_macro")


@raises(ValueError)
def test_register_custom_metric_missing_file():
    metric_dir = join(_my_dir, "other")
    missing_custom_metrics_file = join(metric_dir, "missing_metrics.py")
    register_custom_metric(missing_custom_metrics_file, "f075_macro")


@raises(AttributeError)
def test_register_custom_metric_wrong_name():
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "blah")


def test_register_custom_metric_values():
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")
    register_custom_metric(custom_metrics_file, "ratio_of_ones")

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
    input_dir = join(_my_dir, "other")
    custom_metrics_file = join(input_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")

    train_file = join(input_dir, "examples_train.jsonlines")
    test_file = join(input_dir, "examples_test.jsonlines")

    train_fs = NDJReader.for_path(train_file).read()
    test_fs = NDJReader.for_path(test_file).read()

    learner = Learner("LogisticRegression")
    _ = learner.train(train_fs, grid_objective="f075_macro")
    results = learner.evaluate(test_fs,
                               grid_objective="f075_macro",
                               output_metrics=["balanced_accuracy"])
    test_objective_value = results[-2]
    test_output_metrics_dict = results[-1]
    test_metric_value = test_output_metrics_dict['balanced_accuracy']
    assert_almost_equal(test_objective_value, 0.9785, places=4)
    assert_almost_equal(test_metric_value, 0.9792, places=4)


def test_custom_metric_api_regression():
    pass
