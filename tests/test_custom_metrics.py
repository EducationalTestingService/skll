# License: BSD 3 clause
"""
Module for running tests for custom metrics.

:author: Nitin Madnani (nmadnani@ets.org)
"""

from os.path import abspath, dirname, join

from nose.tools import eq_, ok_, raises

from sklearn.metrics import fbeta_score, SCORERS
from skll.metrics import (_CUSTOM_METRICS,
                          register_custom_metric,
                          use_score_func)

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
    register_custom_metric(custom_metrics_file, "f075")
    assert "f075" in _CUSTOM_METRICS
    assert "f075" in SCORERS

    assert "ratio_of_ones" not in _CUSTOM_METRICS
    assert "ratio_of_ones" not in SCORERS


def test_register_custom_metric_load_both():
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075")
    register_custom_metric(custom_metrics_file, "ratio_of_ones")
    assert "f075" in _CUSTOM_METRICS
    assert "f075" in SCORERS
    assert "ratio_of_ones" in _CUSTOM_METRICS
    assert "ratio_of_ones" in SCORERS


@raises(ValueError)
def test_register_custom_metric_bad_extension():
    metric_dir = join(_my_dir, "other")
    bad_custom_metrics_file = join(metric_dir, "custom_metrics.txt")
    register_custom_metric(bad_custom_metrics_file, "f075")


@raises(ValueError)
def test_register_custom_metric_missing_file():
    metric_dir = join(_my_dir, "other")
    missing_custom_metrics_file = join(metric_dir, "missing_metrics.py")
    register_custom_metric(missing_custom_metrics_file, "f075")


def test_register_custom_metric_values():
    metric_dir = join(_my_dir, "other")
    custom_metrics_file = join(metric_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075")
    register_custom_metric(custom_metrics_file, "ratio_of_ones")

    y_true = [1, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    skll_value = use_score_func("f075", y_true, y_pred)
    sklearn_value = fbeta_score(y_true, y_pred, 0.75)
    eq_(skll_value, sklearn_value)

    skll_value = use_score_func("ratio_of_ones", y_true, y_pred)
    true_ones = len([true for true in y_true if true == 1])
    pred_ones = len([pred for pred in y_pred if pred == 1])
    expected_value = pred_ones / (true_ones + pred_ones)
    eq_(skll_value, expected_value)
