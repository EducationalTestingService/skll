# License: BSD 3 clause
"""
Evaluation and learning curve tests for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import re
from itertools import product

import numpy as np
from nose.tools import assert_almost_equal, eq_, ok_, raises
from numpy.testing import assert_raises_regex
from scipy.stats import pearsonr
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.pipeline import Pipeline

from skll.learner import Learner
from skll.learner.voting import VotingLearner
from skll.utils.logging import close_and_remove_logger_handlers, get_skll_logger
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


def check_evaluate(learner_type, with_grid_search, with_soft_voting):
    """Run checks when evaluating voting learners."""
    # to test the evaluate() method, we instantiate the SKLL voting learner,
    # train it on either the digits (classification) or housing (regression)
    # data set, and evaluate on the corresponding test set; then we do the
    # same in scikit-learn space and compare the objective and value for
    # on additional output metric

    # set various parameters based on whether we are using
    # a classifier or a regressor
    if learner_type == "classifier":
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        voting_type = "soft" if with_soft_voting else "hard"
        train_fs, test_fs = TRAIN_FS_DIGITS, TEST_FS_DIGITS
        objective = "accuracy"
        extra_metric = "f1_score_macro"
        expected_voting_type = voting_type
    else:
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        voting_type = "hard"
        train_fs, test_fs = TRAIN_FS_HOUSING, TEST_FS_HOUSING
        objective = "pearson"
        extra_metric = "neg_mean_squared_error"
        expected_voting_type = None

    # instantiate and train a SKLL voting learner
    skll_vl = VotingLearner(
        learner_names, voting=voting_type, feature_scaling="none", min_feature_count=0
    )
    skll_vl.train(
        train_fs, grid_objective=objective, grid_search=with_grid_search, grid_search_folds=3
    )

    # evaluate on the test set
    res = skll_vl.evaluate(test_fs, grid_objective=objective, output_metrics=[extra_metric])

    # make sure all the parts of the results tuple
    # have the expected types
    ok_(len(res), 6)
    if learner_type == "classifier":
        ok_(isinstance(res[0], list))  # confusion matrix
        ok_(isinstance(res[1], float))  # accuracy
    else:
        eq_(res[0], None)  # no confusion matrix
        eq_(res[1], None)  # no accuracy
    ok_(isinstance(res[2], dict))  # result dict
    ok_(isinstance(res[3], dict))  # model params
    ok_(isinstance(res[4], float))  # objective
    ok_(isinstance(res[5], dict))  # metric scores

    # make sure the model params in the results match what we passed in
    estimators_from_params = res[3]["estimators"]
    for idx, (name, estimator) in enumerate(estimators_from_params):
        eq_(name, learner_names[idx])
        ok_(isinstance(estimator, Pipeline))
    if learner_type == "classifier":
        eq_(res[3]["voting"], expected_voting_type)

    # get the values for the objective and the additional metric
    skll_objective = res[4]
    skll_extra_metric = res[5][extra_metric]

    # now get the estimators that underlie the SKLL voting classifier
    # and use them to train a voting learner directly in scikit-learn
    named_estimators = skll_vl.model.named_estimators_
    clf1 = named_estimators[learner_names[0]]["estimator"]
    clf2 = named_estimators[learner_names[1]]["estimator"]
    clf3 = named_estimators[learner_names[2]]["estimator"]
    sklearn_model_type = VotingClassifier if learner_type == "classifier" else VotingRegressor
    sklearn_model_kwargs = {"estimators": [("clf1", clf1), ("clf2", clf2), ("clf3", clf3)]}
    if learner_type == "classifier":
        sklearn_model_kwargs["voting"] = voting_type
    sklearn_vl = sklearn_model_type(**sklearn_model_kwargs)
    sklearn_vl.fit(train_fs.features, train_fs.labels)

    # get the predictions from this voting classifier on the test set
    sklearn_predictions = sklearn_vl.predict(test_fs.features)

    # compute the values of the objective and the extra metric
    # on the scikit-learn side
    if learner_type == "classifier":
        sklearn_objective = accuracy_score(test_fs.labels, sklearn_predictions)
        sklearn_extra_metric = f1_score(test_fs.labels, sklearn_predictions, average="macro")
    else:
        sklearn_objective = pearsonr(test_fs.labels, sklearn_predictions)[0]
        sklearn_extra_metric = -1 * mean_squared_error(test_fs.labels, sklearn_predictions)

    # check that the values match between SKLL and scikit-learn
    assert_almost_equal(skll_objective, sklearn_objective)
    assert_almost_equal(skll_extra_metric, sklearn_extra_metric)


def test_evaluate():
    for learner_type, with_grid_search, with_soft_voting in product(
        ["classifier", "regressor"], [False, True], [False, True]
    ):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (check_evaluate, learner_type, with_grid_search, with_soft_voting)


def test_evaluate_bad_output_metric():
    vl = VotingLearner(["SVC", "LogisticRegression", "MultinomialNB"])
    vl.train(TRAIN_FS_DIGITS[:100], grid_search=False)
    assert_raises_regex(
        ValueError,
        r"metrics are not valid",
        vl.evaluate,
        TEST_FS_DIGITS[:100],
        output_metrics=["f05", "pearson"],
    )


def check_learning_curve(learner_type, with_soft_voting):
    # to test the learning_curve() method, we instantiate the SKLL voting
    # learner, get the SKLL learning curve output; then we do the
    # same in scikit-learn space and compare the outputs

    # instantiate some needed variables
    cv_folds = 10
    random_state = 123456789
    cv = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=random_state)
    train_sizes = np.linspace(0.1, 1.0, 5)

    # set various parameters based on whether we are using
    # a classifier or a regressor
    if learner_type == "classifier":
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        voting_type = "soft" if with_soft_voting else "hard"
        featureset = FS_DIGITS
        scoring_function = "accuracy"
    else:
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        voting_type = "hard"
        featureset = FS_HOUSING
        scoring_function = "neg_mean_squared_error"

    skll_vl = VotingLearner(
        learner_names, feature_scaling="none", min_feature_count=0, voting=voting_type
    )
    (train_scores1, test_scores1, _, train_sizes1) = skll_vl.learning_curve(
        featureset, cv_folds=cv_folds, train_sizes=train_sizes, metric=scoring_function
    )

    # now instantiate the scikit-learn version with the exact
    # same classifiers;
    # NOTE: here we need to do a bit of hackery
    # to get the same underlying scikit-learn estimators that
    # SKLL would have used since `learning_curve()` doesn't
    # save the underlying estimators like `train()` does
    learner_kwargs = {"probability": True} if with_soft_voting else {}
    learner1 = Learner(learner_names[0], **learner_kwargs)
    learner2 = Learner(learner_names[1], **learner_kwargs)
    learner3 = Learner(learner_names[2], **learner_kwargs)
    learner1.train(featureset[:100], grid_search=False)
    learner2.train(featureset[:100], grid_search=False)
    learner3.train(featureset[:100], grid_search=False)
    clf1, clf2, clf3 = learner1.model, learner2.model, learner3.model
    sklearn_model_type = VotingClassifier if learner_type == "classifier" else VotingRegressor
    sklearn_model_kwargs = {
        "estimators": [(learner_names[0], clf1), (learner_names[1], clf2), (learner_names[2], clf3)]
    }
    if learner_type == "classifier":
        sklearn_model_kwargs["voting"] = voting_type
    sklearn_vl = sklearn_model_type(**sklearn_model_kwargs)

    # now call `learning_curve()` directly from scikit-learn
    # and get its output
    (train_sizes2, train_scores2, test_scores2) = learning_curve(
        sklearn_vl,
        featureset.features,
        featureset.labels,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring_function,
    )

    # now check that SKLL and scikit-learn outputs match
    assert np.all(train_sizes1 == train_sizes2)

    # NOTE: because the digits dataset is quite easy and because
    # we are using SVC, numbers only match up to two significant digits;
    # for regression, we can match to a larger precision
    if learner_type == "classifier":
        assert np.allclose(train_scores1, train_scores2, rtol=1e-2)
        assert np.allclose(test_scores1, test_scores2, rtol=1e-2)
    else:
        assert np.allclose(train_scores1, train_scores2)
        assert np.allclose(test_scores1, test_scores2)


def test_learning_curve():
    for learner_type, with_soft_voting in product(["classifier", "regressor"], [False, True]):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (check_learning_curve, learner_type, with_soft_voting)


@raises(ValueError)
def test_learning_curve_min_examples_check():
    # generates a training split with less than 500 examples
    fs_less_than_500 = FS_DIGITS[:499]

    # create a simple voting classifier
    voting_learner = VotingLearner(["LogisticRegression", "SVC", "MultinomialNB"], voting="hard")

    # this must throw an error because `examples` has less than 500 items
    _ = voting_learner.learning_curve(examples=fs_less_than_500, metric="accuracy")


def test_learning_curve_min_examples_check_override():
    # creates a logger which writes to a temporary log file
    log_file_path = (
        output_dir / "test_check_override_voting_learner_" "learning_curve_min_examples.log"
    )

    logger = get_skll_logger(
        "test_voting_learner_learning_curve_min_examples", filepath=log_file_path
    )

    # generates a training split with less than 500 examples
    fs_less_than_500 = FS_DIGITS[:499]

    # create a simple voting classifier
    voting_learner = VotingLearner(
        ["LogisticRegression", "SVC", "MultinomialNB"], voting="hard", logger=logger
    )

    # this must throw an error because `examples` has less than 500 items
    _ = voting_learner.learning_curve(
        examples=fs_less_than_500, metric="accuracy", override_minimum=True
    )

    # checks that the learning_curve warning message is contained in the log file
    with open(log_file_path) as tf:
        log_text = tf.read()
        learning_curve_warning_re = re.compile(
            r"Learning curves can be unreliable for examples fewer than " r"500. You provided \d+\."
        )
        assert learning_curve_warning_re.search(log_text)

    close_and_remove_logger_handlers(logger)
