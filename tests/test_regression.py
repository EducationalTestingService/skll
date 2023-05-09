# License: BSD 3 clause
"""
Run tests with regression learners.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

import math
import re
import warnings
from itertools import chain, product

import numpy as np
from nose.tools import (
    assert_almost_equal,
    assert_false,
    assert_greater,
    assert_less,
    assert_true,
    eq_,
    ok_,
    raises,
)
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression

from skll.config import _setup_config_parser
from skll.data import FeatureSet, NDJReader, NDJWriter
from skll.experiments import run_configuration
from skll.learner import Learner
from skll.learner.utils import rescaled
from skll.utils.constants import CLASSIFICATION_ONLY_METRICS
from tests import config_dir, other_dir, output_dir, test_dir, train_dir
from tests.utils import (
    fill_in_config_paths_for_fancy_output,
    make_regression_data,
    unlink,
)


def setup():
    """Create necessary directories for testing."""
    for dir_path in [train_dir, test_dir, output_dir]:
        dir_path.mkdir(exist_ok=True)


def tearDown():
    """Clean up after tests."""
    for dir_path in [train_dir, test_dir]:
        unlink(dir_path / "fancy_train.jsonlines")

    for output_file in chain(
        output_dir.glob("regression_fancy_output*"),
        output_dir.glob("test_int_labels_cv*"),
        [test_dir / "fancy_test.jsonlines"],
    ):
        unlink(output_file)

    for file_name in ["test_regression_fancy_output.cfg", "test_int_labels_cv.cfg"]:
        unlink(config_dir / file_name)


# a utility function to check rescaling for linear models
def check_rescaling(name, grid_search=False):
    train_fs, test_fs, _ = make_regression_data(num_examples=2000, sd_noise=4, num_features=3)

    # instantiate the given learner and its rescaled counterpart
    learner = Learner(name)
    rescaled_learner = Learner(f"Rescaled{name}")

    # train both the regular regressor and the rescaled regressor
    # with and without using grid search
    if grid_search:
        learner.train(train_fs, grid_search=True, grid_objective="pearson")
        rescaled_learner.train(train_fs, grid_search=True, grid_objective="pearson")
    else:
        learner.train(train_fs, grid_search=False)
        rescaled_learner.train(train_fs, grid_search=False)

    # now generate both sets of predictions on the test feature set
    predictions = learner.predict(test_fs)
    rescaled_predictions = rescaled_learner.predict(test_fs)

    # ... and on the training feature set
    train_predictions = learner.predict(train_fs)
    rescaled_train_predictions = rescaled_learner.predict(train_fs)

    # make sure that both sets of correlations are close to perfectly
    # correlated, since the only thing different is that one set has been
    # rescaled
    assert_almost_equal(pearsonr(predictions, rescaled_predictions)[0], 1.0, places=3)

    # make sure that the standard deviation of the rescaled test set
    # predictions is higher than the standard deviation of the regular test set
    # predictions
    p_std = np.std(predictions)
    rescaled_p_std = np.std(rescaled_predictions)
    assert_greater(rescaled_p_std, p_std)

    # make sure that the standard deviation of the rescaled predictions
    # on the TRAINING set (not the TEST) is closer to the standard
    # deviation of the training set labels than the standard deviation
    # of the regular predictions.
    train_y_std = np.std(train_fs.labels)
    train_p_std = np.std(train_predictions)
    rescaled_train_p_std = np.std(rescaled_train_predictions)
    assert_less(abs(rescaled_train_p_std - train_y_std), abs(train_p_std - train_y_std))


def test_rescaling():
    for regressor_name in [
        "BayesianRidge",
        "ElasticNet",
        "HuberRegressor",
        "Lars",
        "Lasso",
        "LinearRegression",
        "LinearSVR",
        "RANSACRegressor",
        "Ridge",
        "SGDRegressor",
        "SVR",
        "TheilSenRegressor",
    ]:
        for do_grid_search in [True, False]:
            yield check_rescaling, regressor_name, do_grid_search


# the utility function to run the linear regession tests
def check_linear_models(name, use_feature_hashing=False, use_rescaling=False):
    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        (train_fs, test_fs, weightdict) = make_regression_data(
            num_examples=5000, num_features=10, use_feature_hashing=True, feature_bins=5
        )
    else:
        train_fs, test_fs, weightdict = make_regression_data(num_examples=2000, num_features=3)

    # create the learner
    model_kwargs = (
        {} if name in ["BayesianRidge", "Lars", "LinearRegression"] else {"max_iter": 500}
    )
    name = f"Rescaled{name}" if use_rescaling else name
    learner = Learner(name, model_kwargs=model_kwargs)

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_search=True, grid_objective="pearson")

    # make sure that the weights are close to the weights
    # that we got from make_regression_data. Take the
    # ceiling before comparing since just comparing
    # the ceilings should be enough to make sure nothing
    # catastrophic happened. However, sometimes with
    # feature hashing, the ceiling is not exactly identical
    # so when that fails we want to check that the rounded
    # feature values are the same. One of those two equalities
    # _must_ be satisified.

    # get the weights for this trained model
    learned_weights = learner.model_params[0]

    for feature_name in learned_weights:
        learned_w_ceil = math.ceil(learned_weights[feature_name])
        given_w_ceil = math.ceil(weightdict[feature_name])
        learned_w_round = round(learned_weights[feature_name], 0)
        given_w_round = round(weightdict[feature_name], 0)
        ceil_equal = learned_w_ceil == given_w_ceil
        round_equal = learned_w_round == given_w_round
        either_equal = ceil_equal or round_equal
        assert either_equal

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, 0.95)


# the runner function for linear regression models
def test_linear_models():
    for regressor_name, use_feature_hashing, use_rescaling in product(
        [
            "BayesianRidge",
            "ElasticNet",
            "HuberRegressor",
            "Lars",
            "Lasso",
            "LinearRegression",
            "Ridge",
            "LinearSVR",
            "SGDRegressor",
            "TheilSenRegressor",
        ],
        [False, True],
        [False, True],
    ):
        yield (check_linear_models, regressor_name, use_feature_hashing, use_rescaling)


# the utility function to run the non-linear tests
def check_non_linear_models(name, use_feature_hashing=False, use_rescaling=False):
    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        train_fs, test_fs, _ = make_regression_data(
            num_examples=5000, num_features=10, use_feature_hashing=True, feature_bins=5
        )
    else:
        train_fs, test_fs, _ = make_regression_data(num_examples=2000, num_features=3)

    # create the learner
    if use_rescaling:
        name = f"Rescaled{name}"
    learner = Learner(name)

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_search=True, grid_objective="pearson")

    # Note that we cannot check the feature weights here
    # since `model_params()` is not defined for non-linear
    # kernels.

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, 0.95)


# the runner function for non-linear regression models
def test_non_linear_models():
    for regressor_name, use_feature_hashing, use_rescaling in product(
        ["SVR"], [False, True], [False, True]
    ):
        yield (check_non_linear_models, regressor_name, use_feature_hashing, use_rescaling)


# the utility function to run the tree-based regression tests


def check_tree_models(name, use_feature_hashing=False, use_rescaling=False):
    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        train_fs, test_fs, _ = make_regression_data(
            num_examples=5000, num_features=10, use_feature_hashing=True, feature_bins=5
        )
    else:
        train_fs, test_fs, _ = make_regression_data(num_examples=2000, num_features=3)

    # create the learner
    if use_rescaling:
        name = f"Rescaled{name}"
    learner = Learner(name)

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_search=True, grid_objective="pearson")

    # make sure that the feature importances are as expected.
    if name.endswith("DecisionTreeRegressor"):
        expected_feature_importances = (
            [0.730811, 0.001834, 0.247603, 0.015241, 0.004511]
            if use_feature_hashing
            else [0.08926899, 0.15585068, 0.75488033]
        )
    else:
        expected_feature_importances = (
            [0.733654, 0.002528, 0.245527, 0.013664, 0.004627]
            if use_feature_hashing
            else [0.07974267, 0.16121895, 0.75903838]
        )

    feature_importances = learner.model.feature_importances_
    assert_allclose(feature_importances, expected_feature_importances, atol=1e-2, rtol=0)

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, 0.95)


# the runner function for tree-based regression models
def test_tree_models():
    for regressor_name, use_feature_hashing, use_rescaling in product(
        ["DecisionTreeRegressor", "RandomForestRegressor"], [False, True], [False, True]
    ):
        yield (check_tree_models, regressor_name, use_feature_hashing, use_rescaling)


# the utility function to run the ensemble-based regression tests
def check_ensemble_models(name, use_feature_hashing=False, use_rescaling=False):
    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        train_fs, test_fs, _ = make_regression_data(
            num_examples=5000, num_features=10, use_feature_hashing=True, feature_bins=5
        )
    else:
        train_fs, test_fs, _ = make_regression_data(num_examples=2000, num_features=3)

    # create the learner
    if use_rescaling:
        name = f"Rescaled{name}"
    learner = Learner(name)

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_search=True, grid_objective="pearson")

    # make sure that the feature importances are as expected.
    if name.endswith("AdaBoostRegressor"):
        if use_feature_hashing:
            expected_feature_importances = [0.749811, 0.001373, 0.23357, 0.011691, 0.003554]
        else:
            expected_feature_importances = [0.10266744, 0.18681777, 0.71051479]
    else:
        expected_feature_importances = (
            [0.735756, 0.001034, 0.242734, 0.015836, 0.00464]
            if use_feature_hashing
            else [0.082621, 0.166652, 0.750726]
        )

    feature_importances = learner.model.feature_importances_
    assert_allclose(feature_importances, expected_feature_importances, atol=1e-2, rtol=0)

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, 0.95)


# the runner function for ensemble regression models
def test_ensemble_models():
    for regressor_name, use_feature_hashing, use_rescaling in product(
        ["AdaBoostRegressor", "GradientBoostingRegressor"], [False, True], [False, True]
    ):
        yield (check_ensemble_models, regressor_name, use_feature_hashing, use_rescaling)


def test_int_labels():
    """
    Test that SKLL can take integer input.

    This is just to test that SKLL can take int labels in the input
    (rather than floats or strings).  For v1.0.0, it could not because the
    json package doesn't know how to serialize numpy.int64 objects.
    """
    config_template_path = config_dir / "test_int_labels_cv.template.cfg"
    config_path = config_dir / "test_int_labels_cv.cfg"

    config = _setup_config_parser(config_template_path, validate=False)
    config.set("Input", "train_file", str(other_dir / "test_int_labels_cv.jsonlines"))
    config.set("Output", "results", str(output_dir))
    config.set("Output", "logs", str(output_dir))
    config.set("Output", "predictions", str(output_dir))

    with open(config_path, "w") as new_config_file:
        config.write(new_config_file)

    run_configuration(config_path, quiet=True, local=True)


def test_additional_metrics():
    """Test additional metrics in the results file for a regressor."""
    train_fs, test_fs, _ = make_regression_data(num_examples=2000, num_features=3)

    # train a regression model using the train feature set
    learner = Learner("LinearRegression")
    learner.train(train_fs, grid_search=True, grid_objective="pearson")

    # evaluate the trained model using the test feature set
    results = learner.evaluate(test_fs, output_metrics=["spearman", "kendall_tau"])

    # check that the values for the additional metrics are as expected
    additional_scores_dict = results[-1]
    assert_almost_equal(additional_scores_dict["spearman"], 0.9996, places=4)
    assert_almost_equal(additional_scores_dict["kendall_tau"], 0.9847, places=4)


def test_fancy_output():
    """Test the descriptive statistics output in the results file for a regressor."""
    train_fs, test_fs, _ = make_regression_data(num_examples=2000, num_features=3)

    # train a regression model using the train feature set
    learner = Learner("LinearRegression")
    learner.train(train_fs, grid_search=True, grid_objective="pearson")

    # evaluate the trained model using the test feature set
    resultdict = learner.evaluate(test_fs)
    actual_stats_from_api = dict(resultdict[2]["descriptive"]["actual"])
    pred_stats_from_api = dict(resultdict[2]["descriptive"]["predicted"])

    # write out the training and test feature set
    train_file = train_dir / "fancy_train.jsonlines"
    train_writer = NDJWriter(train_file, train_fs)
    train_writer.write()
    test_file = test_dir / "fancy_test.jsonlines"
    test_writer = NDJWriter(test_file, test_fs)
    test_writer.write()

    # now get the config file template, fill it in and run it
    # so that we can get a results file
    config_template_path = config_dir / "test_regression_fancy_output.template.cfg"
    config_path = fill_in_config_paths_for_fancy_output(config_template_path)

    run_configuration(config_path, quiet=True, local=True)

    # read in the results file and get the descriptive statistics
    actual_stats_from_file = {}
    pred_stats_from_file = {}
    with open(
        output_dir / "regression_fancy_output_train_fancy_train.jsonlines_test_"
        "fancy_test.jsonlines_LinearRegression.results"
    ) as resultf:
        result_output = resultf.read().strip().split("\n")
        for desc_stat_line in result_output[26:30]:
            desc_stat_line = desc_stat_line.strip()
            if not desc_stat_line:
                continue
            else:
                m = re.search(
                    r"([A-Za-z]+)\s+=\s+(-?[0-9]+.?[0-9]*)\s+\((actual)\),\s+"
                    r"(-?[0-9]+.?[0-9]*)\s+\((predicted)\)",
                    desc_stat_line,
                )
                stat_type, actual_value, _, pred_value, _ = m.groups()
                actual_stats_from_file[stat_type.lower()] = float(actual_value)
                pred_stats_from_file[stat_type.lower()] = float(pred_value)

    for stat_type in actual_stats_from_api:
        assert_almost_equal(
            actual_stats_from_file[stat_type], actual_stats_from_api[stat_type], places=4
        )

        assert_almost_equal(
            pred_stats_from_file[stat_type], pred_stats_from_api[stat_type], places=4
        )


def check_adaboost_regression(base_estimator):
    train_fs, test_fs, _ = make_regression_data(num_examples=2000, sd_noise=4, num_features=3)

    # train an AdaBoostRegressor on the training data and evalute on the
    # testing data
    learner = Learner("AdaBoostRegressor", model_kwargs={"base_estimator": base_estimator})
    learner.train(train_fs, grid_search=False)

    # now generate the predictions on the test set
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, 0.95)


def test_adaboost_regression():
    for base_estimator_name in ["DecisionTreeRegressor", "SGDRegressor", "SVR"]:
        yield check_adaboost_regression, base_estimator_name


def check_ransac_regression(base_estimator, pearson_value):
    train_fs, test_fs, _ = make_regression_data(num_examples=2000, sd_noise=4, num_features=3)

    # train a RANSACRegressor on the training data and evalute on the
    # testing data
    model_kwargs = {"estimator": base_estimator, "min_samples": 4} if base_estimator else {}
    learner = Learner("RANSACRegressor", model_kwargs=model_kwargs)
    learner.train(train_fs, grid_search=False)

    # now generate the predictions on the test set
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated and the value
    # of the correlation is as expected
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, pearson_value)


def test_ransac_regression():
    for base_estimator_name, pearson_value in zip(
        [None, "SGDRegressor", "DecisionTreeRegressor", "SVR"], [0.95, 0.45, 0.75, 0.64]
    ):
        yield check_ransac_regression, base_estimator_name, pearson_value


def check_mlp_regression(use_rescaling=False):
    train_fs, test_fs, _ = make_regression_data(num_examples=500, sd_noise=4, num_features=5)

    # train an MLPRegressor on the training data and evalute on the
    # testing data
    name = "MLPRegressor" if use_rescaling else "RescaledMLPRegressor"
    learner = Learner(name)
    # we don't want to see any convergence warnings during the grid search
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        learner.train(train_fs, grid_search=False)

    # now generate the predictions on the test set
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated
    cor, _ = pearsonr(predictions, test_fs.labels)
    assert_greater(cor, 0.98)


def test_mlp_regression():
    yield check_mlp_regression, False
    yield check_mlp_regression, True


def check_dummy_regressor_predict(model_args, train_labels, expected_output):
    # create hard-coded featuresets with the given labels
    train_fs = FeatureSet(
        "regression_train",
        [f"TrainExample{i}" for i in range(20)],
        labels=train_labels,
        features=[{"feature": i} for i in range(20)],
    )

    test_fs = FeatureSet(
        "regression_test",
        [f"TestExample{i}" for i in range(10)],
        features=[{"feature": i} for i in range(20, 30)],
    )

    # Ensure predictions are as expectedfor the given strategy
    learner = Learner("DummyRegressor", model_kwargs=model_args)
    learner.train(train_fs, grid_search=False)
    predictions = learner.predict(test_fs)
    eq_(np.array_equal(expected_output, predictions), True)


def test_dummy_regressor_predict():
    # create a hard-coded set of labels
    prng = np.random.RandomState(123456789)
    train_labels = prng.random_sample(20)

    for model_args, expected_output in zip(
        [
            {"strategy": "mean"},
            {"strategy": "median"},
            {"strategy": "quantile", "quantile": 0.5},
            {"strategy": "quantile", "quantile": 0.0},
            {"strategy": "quantile", "quantile": 1.0},
            {"strategy": "constant", "constant": 1},
        ],
        [
            np.ones(10) * np.mean(train_labels),
            np.ones(10) * np.median(train_labels),
            np.ones(10) * np.median(train_labels),
            np.ones(10) * np.min(train_labels),
            np.ones(10) * np.max(train_labels),
            np.ones(10),
        ],
    ):
        yield check_dummy_regressor_predict, model_args, train_labels, expected_output


@raises(ValueError)
def test_learner_api_rescaling_classifier():
    """Check that rescaling fails for classifiers."""
    _ = rescaled(LogisticRegression)


@raises(ValueError)
def check_invalid_regression_grid_objective(learner, grid_objective):
    """Check whether the grid objective function is valid for this regressor."""
    (train_fs, _, _) = make_regression_data()
    clf = Learner(learner)
    clf.train(train_fs, grid_objective=grid_objective)


def test_invalid_regression_grid_objective():
    for learner in [
        "AdaBoostRegressor",
        "BayesianRidge",
        "DecisionTreeRegressor",
        "ElasticNet",
        "GradientBoostingRegressor",
        "HuberRegressor",
        "KNeighborsRegressor",
        "Lars",
        "Lasso",
        "LinearRegression",
        "MLPRegressor",
        "RandomForestRegressor",
        "RANSACRegressor",
        "Ridge",
        "LinearSVR",
        "SVR",
        "SGDRegressor",
        "TheilSenRegressor",
    ]:
        for metric in CLASSIFICATION_ONLY_METRICS:
            yield check_invalid_regression_grid_objective, learner, metric


@raises(ValueError)
def check_invalid_regression_metric(learner, metric, by_itself=False):
    """Check that invalid metrics raise exceptions."""
    (train_fs, test_fs, _) = make_regression_data()
    clf = Learner(learner)
    clf.train(train_fs, grid_search=False)
    output_metrics = [metric] if by_itself else ["pearson", metric]
    clf.evaluate(test_fs, output_metrics=output_metrics)


def test_invalid_regression_metric():
    for learner in [
        "AdaBoostRegressor",
        "BayesianRidge",
        "DecisionTreeRegressor",
        "ElasticNet",
        "GradientBoostingRegressor",
        "HuberRegressor",
        "KNeighborsRegressor",
        "Lars",
        "Lasso",
        "LinearRegression",
        "MLPRegressor",
        "RandomForestRegressor",
        "RANSACRegressor",
        "Ridge",
        "LinearSVR",
        "SVR",
        "SGDRegressor",
        "TheilSenRegressor",
    ]:
        for metric in CLASSIFICATION_ONLY_METRICS:
            yield check_invalid_regression_metric, learner, metric, True
            yield check_invalid_regression_metric, learner, metric, False


def test_train_non_sparse_featureset():
    """Test that we can train a regressor on a non-sparse featureset."""
    train_file = other_dir / "test_int_labels_cv.jsonlines"
    train_fs = NDJReader.for_path(train_file, sparse=False).read()
    learner = Learner("LinearRegression")
    learner.train(train_fs, grid_search=False)
    ok_(hasattr(learner.model, "coef_"))


@raises(TypeError)
def test_train_string_labels():
    """Test that regression on string labels raises TypeError."""
    train_file = other_dir / "test_int_labels_cv.jsonlines"
    train_fs = NDJReader.for_path(train_file).read()
    train_fs.labels = train_fs.labels.astype("str")
    learner = Learner("LinearRegression")
    learner.train(train_fs, grid_search=False)


def test_non_negative_regression():
    """Test that non-negative regression works as expected."""
    # read in the example training data into a featureset
    train_path = train_dir / "test_non_negative.jsonlines"
    train_fs = NDJReader.for_path(train_path).read()

    # train a regular SKLL linear regerssion learner first
    # and check that it gets some negative weights with this data
    skll_estimator1 = Learner("LinearRegression", min_feature_count=0)
    skll_estimator1.train(train_fs, grid_search=False)
    assert_true(np.any(np.where(skll_estimator1.model.coef_ < 0, True, False)))

    # now train a non-negative linear regression learner and check
    # that _none_ of its weights are negative with the same data
    skll_estimator2 = Learner(
        "LinearRegression", model_kwargs={"positive": True}, min_feature_count=0
    )
    skll_estimator2.train(train_fs, grid_search=False)
    assert_false(np.any(np.where(skll_estimator2.model.coef_ < 0, True, False)))

    # just for good measure, train a non-negative learner directly
    # in sklearn space and check that it has the same weights
    sklearn_estimator = LinearRegression(positive=True)
    X, y = train_fs.features.toarray(), train_fs.labels
    _ = sklearn_estimator.fit(X, y)
    assert_array_equal(skll_estimator2.model.coef_, sklearn_estimator.coef_)
