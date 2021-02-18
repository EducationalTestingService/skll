# License: BSD 3 clause
"""
Module containing tests for custom metrics.

:author: Nitin Madnani (nmadnani@ets.org)
"""

import json
import sys
from glob import glob
from os.path import join

import numpy as np
from nose.tools import assert_almost_equal, eq_, ok_, raises, with_setup
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises_regex,
)
from sklearn.metrics import SCORERS, fbeta_score

import skll.metrics
from skll.data import NDJReader
from skll.experiments import run_configuration
from skll.learner import Learner
from skll.metrics import _CUSTOM_METRICS, register_custom_metric, use_score_func
from tests import config_dir, other_dir, output_dir
from tests.utils import (
    fill_in_config_paths_for_single_file,
    make_classification_data,
    unlink,
)


def setup_func():
    """
    Run before each test function.
    """
    # clean up any already registered metrics to simulate
    # as if we are starting a new Python session
    _cleanup_custom_metrics()


def tearDown():
    """
    Clean up after all tests are run.
    """

    for glob_pattern in [join(config_dir, "*custom_metrics.cfg"),
                         join(config_dir, "*custom_metrics_kappa.cfg"),
                         join(config_dir, "*custom_metrics_bad.cfg"),
                         join(config_dir, "*custom_metrics_kwargs1.cfg"),
                         join(config_dir, "*custom_metrics_kwargs2.cfg"),
                         join(config_dir, "*custom_metrics_kwargs3.cfg"),
                         join(config_dir, "*custom_metrics_kwargs4.cfg"),
                         join(output_dir, "test_custom_metrics*")]:
        for f in glob(glob_pattern):
            unlink(f)


def _cleanup_custom_metrics():
    """A helper function to clean up any custom metrics"""

    for metric_name in ['f06_micro',
                        'f075_macro',
                        'fake_prob_metric',
                        'fake_prob_metric_multiclass',
                        'one_minus_f1_macro',
                        'one_minus_precision',
                        'ratio_of_ones']:

        try:
            delattr(skll.experiments, metric_name)
        except AttributeError:
            pass

        try:
            delattr(skll.metrics, metric_name)
        except AttributeError:
            pass

        try:
            del SCORERS[metric_name]
        except KeyError:
            pass

    for module_name in ['custom_metrics', 'custom_metrics2']:
        try:
            del sys.modules[module_name]
        except KeyError:
            pass

    # remove any metric functions from _CUSTOM_METRICS
    _CUSTOM_METRICS.difference_update(["f075_macro",
                                       "ratio_of_ones",
                                       "f06_micro",
                                       "one_minus_precision",
                                       "one_minus_f1_macro",
                                       "fake_prob_meltric",
                                       "fake_prob_metric_multiclass"])


@with_setup(setup_func)
def test_register_custom_metric_load_one():
    """Test loading a single custom metric"""

    # load a single metric from a custom metric file
    custom_metrics_file = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")

    # make sure that this metric is now registered with SKLL
    assert "f075_macro" in _CUSTOM_METRICS
    assert "f075_macro" in SCORERS

    # make sure that the other metric in that same file
    # is _not_ registered with SKLL since we didn't ask for it
    assert "ratio_of_ones" not in _CUSTOM_METRICS
    assert "ratio_of_ones" not in SCORERS


@with_setup(setup_func)
def test_register_custom_metric_load_both():
    """Test loading two custom metrics from one file"""

    # load both metrics in the custom file
    custom_metrics_file = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")
    register_custom_metric(custom_metrics_file, "ratio_of_ones")

    # now make sure that both are registered
    assert "f075_macro" in _CUSTOM_METRICS
    assert "f075_macro" in SCORERS
    assert "ratio_of_ones" in _CUSTOM_METRICS
    assert "ratio_of_ones" in SCORERS


@with_setup(setup_func)
def test_register_custom_metric_load_different_files():
    """Test loading two custom metrics from two files"""

    # load two custom metrics from two different files
    custom_metrics_file1 = join(other_dir, "custom_metrics.py")
    custom_metrics_file2 = join(other_dir, "custom_metrics2.py")
    register_custom_metric(custom_metrics_file1, "f075_macro")
    register_custom_metric(custom_metrics_file2, "f06_micro")

    # make sure both are registered
    assert "f075_macro" in _CUSTOM_METRICS
    assert "f075_macro" in SCORERS
    assert "f06_micro" in _CUSTOM_METRICS
    assert "f06_micro" in SCORERS


@with_setup(setup_func)
@raises(NameError)
def test_reregister_same_metric_same_session():
    """Test loading custom metric again in same session"""

    # try to load a metric from a txt file, not a py file
    custom_metrics_file = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")

    # re-registering should raise an error
    register_custom_metric(custom_metrics_file, "f075_macro")


@with_setup(setup_func)
def test_reregister_same_metric_different_session():
    """Test loading custom metric again in different session"""

    # try to load a metric from a txt file, not a py file
    custom_metrics_file = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")

    # clean up any already registered metrics to simulate
    # as if we are starting a new Python session
    _cleanup_custom_metrics()

    # now re-registering should work just fine
    register_custom_metric(custom_metrics_file, "f075_macro")


@with_setup(setup_func)
@raises(ValueError)
def test_register_custom_metric_bad_extension():
    """Test loading custom metric from non-py file"""

    # try to load a metric from a txt file, not a py file
    bad_custom_metrics_file = join(other_dir, "custom_metrics.txt")
    register_custom_metric(bad_custom_metrics_file, "f075_macro")


@with_setup(setup_func)
@raises(ValueError)
def test_register_custom_metric_missing_name():
    """Test loading custom metric from empty string"""

    # try to load a metric from a missing file name
    # which can happen via a bad configuration file
    register_custom_metric("", "f075_macro")


@with_setup(setup_func)
@raises(ValueError)
def test_register_custom_metric_missing_file():
    """Test loading custom metric from missing file"""

    # try to load a metric from a py file that does not exist
    missing_custom_metrics_file = join(other_dir, "missing_metrics.py")
    register_custom_metric(missing_custom_metrics_file, "f075_macro")


@with_setup(setup_func)
@raises(AttributeError)
def test_register_custom_metric_wrong_name():
    """Test loading custom metric with wrong name"""

    # try to load a metric that does not exist in a file
    custom_metrics_file = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "blah")


@with_setup(setup_func)
@raises(NameError)
def test_register_custom_metric_conflicting_metric_name():
    """Test loading custom metric with conflicting name"""

    # try to load a metric that does not exist in a file
    custom_metrics_file = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "r2")


@with_setup(setup_func)
def test_register_custom_metric_values():
    """Test to check values of custom metrics"""

    # register two metrics in the same file
    custom_metrics_file = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "f075_macro")
    register_custom_metric(custom_metrics_file, "ratio_of_ones")

    # check that the values that SKLL would compute matches what we expect
    y_true = [1, 1, 1, 0, 2, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 1, 2]
    skll_value = use_score_func("f075_macro", y_true, y_pred)
    sklearn_value = fbeta_score(y_true, y_pred, beta=0.75, average='macro')
    eq_(skll_value, sklearn_value)

    y_true = [1, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    skll_value = use_score_func("ratio_of_ones", y_true, y_pred)
    true_ones = len([true for true in y_true if true == 1])
    pred_ones = len([pred for pred in y_pred if pred == 1])
    expected_value = pred_ones / (true_ones + pred_ones)
    eq_(skll_value, expected_value)


@with_setup(setup_func)
def test_custom_metric_api_experiment():
    """Test API with custom metrics"""

    # register two different metrics from two files
    custom_metrics_file1 = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file1, "f075_macro")
    custom_metrics_file2 = join(other_dir, "custom_metrics2.py")
    register_custom_metric(custom_metrics_file2, "f06_micro")

    # read in some train/test data
    train_file = join(other_dir, "examples_train.jsonlines")
    test_file = join(other_dir, "examples_test.jsonlines")

    train_fs = NDJReader.for_path(train_file).read()
    test_fs = NDJReader.for_path(test_file).read()

    # set up a learner to tune using one of the custom metrics
    # and evaluate it using the other one
    learner = Learner("LogisticRegression")
    _ = learner.train(train_fs,
                      grid_objective="f075_macro",
                      grid_search_folds=3)
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


@with_setup(setup_func)
def test_custom_metric_config_experiment():
    """Test config with custom metrics"""

    # Run experiment
    train_file = join(other_dir, "examples_train.jsonlines")
    test_file = join(other_dir, "examples_test.jsonlines")
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_custom_metrics.template.cfg"),
        train_file,
        test_file
    )
    run_configuration(config_path, local=True, quiet=True)

    # Check results for objective functions and output metrics

    # objective function f075_macro
    with open(join(output_dir,
                   'test_custom_metrics_train_examples_train.jsonlines_test_'
                   'examples_test.jsonlines_LogisticRegression.results.json')) as f:
        result_dict = json.load(f)[0]

    test_objective_value = result_dict['score']
    test_output_metrics_dict = result_dict['additional_scores']
    test_accuracy_value = test_output_metrics_dict["balanced_accuracy"]
    test_ratio_of_ones_value = test_output_metrics_dict["ratio_of_ones"]

    # check that the values are as expected
    assert_almost_equal(test_objective_value, 0.9785, places=4)
    assert_almost_equal(test_accuracy_value, 0.9792, places=4)
    assert_almost_equal(test_ratio_of_ones_value, 0.5161, places=4)


@with_setup(setup_func)
def test_custom_metric_api_experiment_with_kappa_filename():
    """Test API with metric defined in a file named kappa"""

    # register a dummy metric that just returns 1 from
    # a file called 'kappa.py'
    custom_metrics_file = join(other_dir, "kappa.py")
    register_custom_metric(custom_metrics_file, "dummy_metric")

    # read in some train/test data
    train_file = join(other_dir, "examples_train.jsonlines")
    test_file = join(other_dir, "examples_test.jsonlines")

    train_fs = NDJReader.for_path(train_file).read()
    test_fs = NDJReader.for_path(test_file).read()

    # set up a learner to tune using our usual kappa metric
    # and evaluate it using the dummy metric we loaded
    # this should work as there should be no confict between
    # the two "kappa" names
    learner = Learner("LogisticRegression")
    _ = learner.train(train_fs,
                      grid_objective="unweighted_kappa",
                      grid_search_folds=3)
    results = learner.evaluate(test_fs,
                               grid_objective="unweighted_kappa",
                               output_metrics=["balanced_accuracy",
                                               "dummy_metric"])
    test_objective_value = results[-2]
    test_output_metrics_dict = results[-1]
    test_accuracy_value = test_output_metrics_dict["balanced_accuracy"]
    test_dummy_metric_value = test_output_metrics_dict["dummy_metric"]

    # check that the values are as expected
    assert_almost_equal(test_objective_value, 0.9699, places=4)
    assert_almost_equal(test_accuracy_value, 0.9792, places=4)
    eq_(test_dummy_metric_value, 1.0)


@with_setup(setup_func)
def test_custom_metric_config_experiment_with_kappa_filename():
    """Test config with metric defined in a file named kappa"""

    # Run experiment
    train_file = join(other_dir, "examples_train.jsonlines")
    test_file = join(other_dir, "examples_test.jsonlines")
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_custom_metrics_kappa.template.cfg"),
        train_file,
        test_file
    )
    run_configuration(config_path, local=True, quiet=True)

    # Check results for objective functions and output metrics

    # objective function f075_macro
    with open(join(output_dir,
                   'test_custom_metrics_kappa_train_examples_train.jsonlines'
                   '_test_examples_test.jsonlines_LogisticRegression.results'
                   '.json')) as f:
        result_dict = json.load(f)[0]

    test_objective_value = result_dict['score']
    test_output_metrics_dict = result_dict['additional_scores']
    test_accuracy_value = test_output_metrics_dict["balanced_accuracy"]
    test_dummy_metric_value = test_output_metrics_dict["dummy_metric"]

    # check that the values are as expected
    assert_almost_equal(test_objective_value, 0.9699, places=4)
    assert_almost_equal(test_accuracy_value, 0.9792, places=4)
    eq_(test_dummy_metric_value, 1.0)


@with_setup(setup_func)
def test_custom_metric_config_with_invalid_custom_metric():
    """Test config with a valid and an invalid custom metric"""

    # Run experiment
    train_file = join(other_dir, "examples_train.jsonlines")
    test_file = join(other_dir, "examples_test.jsonlines")
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_custom_metrics_bad.template.cfg"),
        train_file,
        test_file
    )
    # since this configuration file consists of an invalid
    # metric, this should raise an error
    assert_raises_regex(ValueError,
                        r"invalid metrics specified:.*missing_metric",
                        run_configuration, config_path, local=True, quiet=True)


@with_setup(setup_func)
def test_api_with_inverted_custom_metric():
    """Test API with a lower-is-better custom metric"""

    # register a lower-is-better custom metrics from our file
    # which is simply 1 minus the precision score
    custom_metrics_file1 = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file1, "one_minus_precision")

    # create some classification data
    train_fs, _ = make_classification_data(num_examples=1000,
                                           num_features=10,
                                           num_labels=2)

    # set up a learner to tune using the lower-is-better custom metric
    learner1 = Learner("LogisticRegression")
    (grid_score1,
     grid_results_dict1) = learner1.train(train_fs,
                                          grid_objective="one_minus_precision",
                                          grid_search_folds=3)

    # now setup another learner that uses the complementary version
    # of our custom metric (regular precision) for grid search
    learner2 = Learner("LogisticRegression")
    (grid_score2,
     grid_results_dict2) = learner2.train(train_fs,
                                          grid_objective="precision",
                                          grid_search_folds=3)

    # for both learners the ranking of the C hyperparameter should be
    # should be the identical since when we defined one_minus_precision
    # we set the `greater_is_better` keyword argument to `False`
    assert_array_equal(grid_results_dict1['rank_test_score'],
                       grid_results_dict2['rank_test_score'])

    # furthermore, the final grid score and the mean scores for each
    # C hyperparameter value should follow the same 1-X relationship
    # except that our custom metric should be negated due to the
    # keyword argument that we set when we defined it
    assert_almost_equal(1 - grid_score2, -1 * grid_score1, places=6)
    assert_array_almost_equal(1 - grid_results_dict2['mean_test_score'],
                              -1 * grid_results_dict1['mean_test_score'],
                              decimal=6)


@with_setup(setup_func)
def test_config_with_inverted_custom_metric():
    """Test config with a lower-is-better custom metric"""

    # run the first experiment that uses a lower-is-better custom metric
    # for grid saerch defined as simply 1 minus the macro-averaged F1 score
    train_file = join(other_dir, "examples_train.jsonlines")
    test_file = join(other_dir, "examples_test.jsonlines")
    config_path1 = fill_in_config_paths_for_single_file(
        join(config_dir, "test_custom_metrics_kwargs1.template.cfg"),
        train_file,
        test_file
    )
    run_configuration(config_path1, local=True, quiet=True)

    # laod the results
    with open(join(output_dir,
                   'test_custom_metrics_kwargs1_train_examples_train.jsonlines'
                   '_LogisticRegression.results.json')) as f:
        result_dict1 = json.load(f)
        grid_score1 = result_dict1['grid_score']
        grid_results_dict1 = result_dict1['grid_search_cv_results']

    # now run the second experiment that is identical except that
    # that it uses the regular macro-averaged F1 score for grid search
    config_path2 = fill_in_config_paths_for_single_file(
        join(config_dir, "test_custom_metrics_kwargs2.template.cfg"),
        train_file,
        test_file
    )
    run_configuration(config_path2, local=True, quiet=True)

    # laod the results
    with open(join(output_dir,
                   'test_custom_metrics_kwargs2_train_examples_train.jsonlines'
                   '_LogisticRegression.results.json')) as f:
        result_dict2 = json.load(f)
        grid_score2 = result_dict2['grid_score']
        grid_results_dict2 = result_dict2['grid_search_cv_results']

    # for both experiments the ranking of the C hyperparameter should be
    # should be the identical since when we defined one_minus_precision
    # we set the `greater_is_better` keyword argument to `False`
    assert_array_equal(grid_results_dict1['rank_test_score'],
                       grid_results_dict2['rank_test_score'])

    # furthermore, the final grid score and the mean scores for each
    # C hyperparameter value should follow the same 1-X relationship
    # except that our custom metric should be negated due to the
    # keyword argument that we set when we defined it
    assert_almost_equal(1 - grid_score2, -1 * grid_score1, places=6)
    assert_array_almost_equal(1 - np.array(grid_results_dict2['mean_test_score']),
                              -1 * np.array(grid_results_dict1['mean_test_score']),
                              decimal=6)


@with_setup(setup_func)
def test_api_with_custom_prob_metric():
    """Test API with custom probabilistic metric"""

    # register a custom metric from our file that requires probabilities
    custom_metrics_file = join(other_dir, "custom_metrics.py")
    register_custom_metric(custom_metrics_file, "fake_prob_metric")

    # create some classification data
    train_fs, _ = make_classification_data(num_examples=1000,
                                           num_features=10,
                                           num_labels=2)

    # set up a learner to tune using this probabilistic metric
    # this should fail since LinearSVC doesn't support probabilities
    learner1 = Learner("LinearSVC")
    assert_raises_regex(AttributeError,
                        r"has no attribute 'predict_proba'",
                        learner1.train,
                        train_fs,
                        grid_objective="fake_prob_metric",
                        grid_search_folds=3)

    # set up another learner with explicit probability support
    # this should work just fine with our custom metric
    learner2 = Learner("SVC", probability=True)
    grid_score, _ = learner2.train(train_fs,
                                   grid_objective="fake_prob_metric",
                                   grid_search_folds=3)
    ok_(grid_score > 0.95)


@with_setup(setup_func)
def test_config_with_custom_prob_metric():
    """Test config with custom probabilistic metric"""

    # run the first experiment that uses a custom probabilistic metric
    # for grid search but with a learner that does not produce probabilities
    train_file = join(other_dir, "examples_train.jsonlines")
    test_file = join(other_dir, "examples_test.jsonlines")
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_custom_metrics_kwargs3.template.cfg"),
        train_file,
        test_file
    )

    # this should fail as expected
    assert_raises_regex(AttributeError,
                        r"has no attribute 'predict_proba'",
                        run_configuration, config_path, local=True, quiet=True)

    # now run the second experiment that is identical except that
    # the learner now produces probabilities
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_custom_metrics_kwargs4.template.cfg"),
        train_file,
        test_file
    )
    # this should succeed and produce results
    run_configuration(config_path, local=True, quiet=True)

    # laod the results and verify them
    with open(join(output_dir,
                   'test_custom_metrics_kwargs4_train_examples_train.jsonlines'
                   '_SVC.results.json')) as f:
        result_dict = json.load(f)
        grid_score = result_dict['grid_score']

    ok_(grid_score > 0.95)
