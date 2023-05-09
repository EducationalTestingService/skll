# License: BSD 3 clause
"""
Prediction tests for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from itertools import product
from pathlib import Path

import numpy as np
from nose.tools import ok_
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.ensemble import VotingClassifier, VotingRegressor

from skll.learner.voting import VotingLearner
from tests import other_dir, output_dir
from tests.utils import make_california_housing_data, make_digits_data

# define some constants needed for testing
TRAIN_FS_DIGITS, TEST_FS_DIGITS = make_digits_data(use_digit_names=True)
FS_DIGITS, _ = make_digits_data(test_size=0, use_digit_names=True)
TRAIN_FS_HOUSING, TEST_FS_HOUSING = make_california_housing_data(num_examples=2000)
FS_HOUSING, _ = make_california_housing_data(num_examples=2000, test_size=0)
FS_HOUSING.ids = np.arange(2000)
CUSTOM_LEARNER_PATH = other_dir / "custom_logistic_wrapper.py"


def setup():
    """Set up the tests."""
    for dir_path in [other_dir, output_dir]:
        dir_path.mkdir(exist_ok=True)


def tearDown():
    """Clean up after tests."""
    for output_file_path in output_dir.glob("test_predict_voting*"):
        output_file_path.unlink()


def check_predict(
    learner_type,
    with_grid_search,
    with_soft_voting,
    with_class_labels,
    with_file_output,
    with_individual_predictions,
):
    # to test the predict() method, we instantiate the SKLL voting learner,
    # train it on either the digits (classification) or housing (regression)
    # data set, and generate predictions on the corresponding test set; then
    # we do the same in scikit-learn space and compare the objective and
    # compare the SKLL and scikit-learn predictions

    # set the prediction prefix in case we need to write out the predictions
    prediction_prefix = (
        output_dir / f"test_predict_voting_"
        f"{learner_type}_"
        f"{with_grid_search}_"
        f"{with_class_labels}"
        if with_file_output
        else None
    )
    prediction_prefix = str(prediction_prefix) if prediction_prefix else None

    # set various parameters based on whether we are using
    # a classifier or a regressor
    if learner_type == "classifier":
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        voting_type = "soft" if with_soft_voting else "hard"
        train_fs, test_fs = TRAIN_FS_DIGITS, TEST_FS_DIGITS
        objective = "accuracy"
    else:
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        voting_type = "hard"
        train_fs, test_fs = TRAIN_FS_HOUSING, TEST_FS_HOUSING
        objective = "pearson"

    # instantiate and train the SKLL voting learner on the digits dataset
    skll_vl = VotingLearner(
        learner_names, feature_scaling="none", min_feature_count=0, voting=voting_type
    )
    skll_vl.train(
        train_fs, grid_objective=objective, grid_search=with_grid_search, grid_search_folds=3
    )

    # get the overall and individual predictions from SKLL
    (skll_predictions, skll_individual_dict) = skll_vl.predict(
        test_fs,
        class_labels=with_class_labels,
        prediction_prefix=prediction_prefix,
        individual_predictions=with_individual_predictions,
    )

    # get the underlying scikit-learn estimators from SKLL
    named_estimators = skll_vl.model.named_estimators_
    clf1 = named_estimators[learner_names[0]]["estimator"]
    clf2 = named_estimators[learner_names[1]]["estimator"]
    clf3 = named_estimators[learner_names[2]]["estimator"]

    # instantiate and train the scikit-learn voting classifer
    sklearn_model_type = VotingClassifier if learner_type == "classifier" else VotingRegressor
    sklearn_model_kwargs = {
        "estimators": [(learner_names[0], clf1), (learner_names[1], clf2), (learner_names[2], clf3)]
    }
    if learner_type == "classifier":
        sklearn_model_kwargs["voting"] = voting_type
    sklearn_vl = sklearn_model_type(**sklearn_model_kwargs)
    sklearn_vl.fit(train_fs.features, train_fs.labels)

    # get the overall predictions from scikit-learn
    sklearn_predictions = sklearn_vl.predict(test_fs.features)

    # if we are doing classification and not asked to output class
    # labels get either the scikit-learn probabilities or the class
    # indices depending on the voting type (soft vs. hard)
    if learner_type == "classifier" and not with_class_labels:
        if voting_type == "soft":
            sklearn_predictions = sklearn_vl.predict_proba(test_fs.features)
        else:
            sklearn_predictions = np.array(
                [skll_vl.label_dict[class_] for class_ in sklearn_predictions]
            )

    # get the individual scikit-learn predictions, if necessary
    sklearn_individual_dict = {}
    if with_individual_predictions:
        for name, estimator in sklearn_vl.named_estimators_.items():
            estimator_predictions = estimator.predict(test_fs.features)
            # scikit-learn individual predictions are indices not class labels
            # so we need to convert them to labels if required
            if with_class_labels:
                estimator_predictions = [
                    sklearn_vl.classes_[index] for index in estimator_predictions
                ]
            # if no class labels, then get the probabilities with soft
            # voting; note that since the individual predictions from
            # scikit-learn are already indices, we do not need to do
            # anything for the hard voting case
            else:
                if voting_type == "soft":
                    estimator_predictions = estimator.predict_proba(test_fs.features)

            sklearn_individual_dict[name] = estimator_predictions

    # now we start the actual tests

    # if individual predictions were not asked for, then SKLL
    # should have returned None for those
    if not with_individual_predictions:
        ok_(skll_individual_dict is None)

    # if we are doing soft voting over classifiers and not returning
    # the class labels, then we need to compare SKLL and scikit-learn
    # probabilities; for the digits dataset, the numbers only match
    # exactly for 2 decimal places because of the way that SVC computes
    # probabilities; we also check that the index of the highest probability
    # is the same for both
    if learner_type == "classifier":
        if voting_type == "soft" and not with_class_labels:
            assert_array_almost_equal(skll_predictions, sklearn_predictions, decimal=2)
            skll_max_prob_indices = np.argmax(skll_predictions, axis=1)
            sklearn_max_prob_indices = np.argmax(sklearn_predictions, axis=1)
            assert_array_equal(skll_max_prob_indices, sklearn_max_prob_indices)
            # check individual probabilities but only for non-SVC estimators
            if with_individual_predictions:
                assert_array_almost_equal(
                    skll_individual_dict["LogisticRegression"],
                    sklearn_individual_dict["LogisticRegression"],
                    decimal=2,
                )
                assert_array_almost_equal(
                    skll_individual_dict["MultinomialNB"],
                    sklearn_individual_dict["MultinomialNB"],
                    decimal=2,
                )
        # in all other cases, we expect the actual class labels or class indices
        # to be identical between SKLL and scikit-learn
        else:
            assert_array_equal(skll_predictions, sklearn_predictions)

    # for regression, we expect the overall predictions to match exactly
    # but individual predictions only up to 2 decimal places
    else:
        assert_array_equal(skll_predictions, sklearn_predictions)
        if with_individual_predictions:
            assert_array_almost_equal(
                skll_individual_dict[learner_names[0]],
                sklearn_individual_dict[learner_names[0]],
                decimal=2,
            )
            assert_array_almost_equal(
                skll_individual_dict[learner_names[1]],
                sklearn_individual_dict[learner_names[1]],
                decimal=2,
            )
            assert_array_almost_equal(
                skll_individual_dict[learner_names[2]],
                sklearn_individual_dict[learner_names[2]],
                decimal=2,
            )

    # if we were asked to write output to disk, then check that
    # the files actually exist
    if with_file_output:
        ok_(Path(f"{prediction_prefix}_predictions.tsv").exists())
        if with_individual_predictions:
            ok_(Path(f"{prediction_prefix}_{learner_names[0]}_predictions.tsv").exists())
            ok_(Path(f"{prediction_prefix}_{learner_names[1]}_predictions.tsv").exists())
            ok_(Path(f"{prediction_prefix}_{learner_names[2]}_predictions.tsv").exists())


def test_predict():
    for (
        learner_type,
        with_grid_search,
        with_soft_voting,
        with_class_labels,
        with_file_output,
        with_individual_predictions,
    ) in product(
        ["classifier", "regressor"],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
    ):
        # regressors do not support soft voting or class labels
        if learner_type == "regressor" and (with_soft_voting or with_class_labels):
            continue
        else:
            yield (
                check_predict,
                learner_type,
                with_grid_search,
                with_soft_voting,
                with_class_labels,
                with_file_output,
                with_individual_predictions,
            )
