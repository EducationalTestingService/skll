# License: BSD 3 clause
"""
Cross-validation tests without grid search for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from nose.tools import assert_almost_equal, eq_, ok_
from numpy.testing import assert_raises_regex
from scipy.stats import pearsonr
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import PredefinedSplit, cross_val_predict

from skll.learner.voting import VotingLearner
from tests import other_dir, output_dir
from tests.utils import (
    make_california_housing_data,
    make_classification_data,
    make_digits_data,
)

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
    for output_file_path in output_dir.glob("test_xval_voting_no_gs*"):
        output_file_path.unlink()


def test_cross_validate_with_continuous_labels():
    """Test that voting learner cross validation fails with continuous labels."""
    fs, _ = make_classification_data(
        num_examples=500, train_test_ratio=0.8, num_labels=3, non_negative=True
    )
    fs.labels = fs.labels.astype(float) + 0.5
    voting_learner = VotingLearner(["LogisticRegression", "SVC", "MultinomialNB"])
    assert_raises_regex(
        ValueError,
        "must be encoded as strings",
        voting_learner.cross_validate,
        fs,
        grid_search=False,
    )


def test_cross_validate_grid_search_but_no_objective():
    """Test that voting learner cross validation fails with continuous labels."""
    fs, _ = make_classification_data(
        num_examples=500, train_test_ratio=0.8, num_labels=3, non_negative=True
    )
    voting_learner = VotingLearner(["LogisticRegression", "SVC", "MultinomialNB"])
    assert_raises_regex(
        ValueError, "must either specify a grid objective", voting_learner.cross_validate, fs
    )


def check_cross_validate_without_grid_search(
    learner_type, with_soft_voting, with_individual_predictions
):
    # to test the cross_validate() method without grid search, we
    # instantiate the SKLL voting learner, call `cross_validate()` on it
    # while writing out the predictions and also asking it to return
    # the actual folds it used as well as the models. Then we use these
    # exact folds with `cross_val_predict()` from scikit-learn as applied
    # to a voting learner instantiated in scikit-learn space. Then we compute
    # metrics over both sets of cross-validated predictions on the
    # test set and compare their values.

    # set the prediction prefix in case we need to write out the predictions
    prediction_prefix = (
        output_dir / f"test_xval_voting_no_gs_" f"{learner_type}_" f"{with_soft_voting}"
    )
    prediction_prefix = str(prediction_prefix)

    # set various parameters based on whether we are using
    # a classifier or a regressor
    if learner_type == "classifier":
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        voting_type = "soft" if with_soft_voting else "hard"
        featureset = FS_DIGITS
        extra_metric = "f1_score_macro"
    else:
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        voting_type = "hard"
        featureset = FS_HOUSING
        extra_metric = "neg_mean_squared_error"

    # instantiate and cross-validate the SKLL voting learner
    # on the full digits dataset
    skll_vl = VotingLearner(
        learner_names, feature_scaling="none", min_feature_count=0, voting=voting_type
    )
    (xval_results, used_fold_ids, used_models) = skll_vl.cross_validate(
        featureset,
        grid_search=False,
        prediction_prefix=prediction_prefix,
        output_metrics=[extra_metric],
        save_cv_folds=True,
        save_cv_models=True,
        individual_predictions=with_individual_predictions,
    )

    # check that the results are as expected
    ok_(len(xval_results), 10)  # number of folds
    for i in range(10):
        if learner_type == "classifier":
            ok_(isinstance(xval_results[i][0], list))  # confusion matrix
            ok_(isinstance(xval_results[i][1], float))  # accuracy
        else:
            eq_(xval_results[i][0], None)  # no confusion matrix
            eq_(xval_results[i][1], None)  # no accuracy
        ok_(isinstance(xval_results[i][2], dict))  # result dict
        ok_(isinstance(xval_results[i][3], dict))  # model params
        eq_(xval_results[i][4], None)  # No objective
        ok_(isinstance(xval_results[i][5], dict))  # metric scores

    # create a pandas dataframe with the returned fold IDs
    # and create a scikit-learn CV splitter with the exact folds
    df_folds = pd.DataFrame(used_fold_ids.items(), columns=["id", "fold"])
    df_folds = df_folds.sort_values(by="id").reset_index(drop=True)
    splitter = PredefinedSplit(df_folds["fold"].astype(int).to_numpy())
    eq_(splitter.get_n_splits(), 10)

    # now read in the SKLL xval predictions from the file written to disk
    df_preds = pd.read_csv(f"{prediction_prefix}_predictions.tsv", sep="\t")

    # sort the columns so that consecutive IDs are actually next to
    # each other in the frame; this is not always guaranteed because
    # consecutive IDs may be in different folds
    df_preds = df_preds.sort_values(by="id").reset_index(drop=True)

    # if we are doing soft voting, then save the argmax-ed prediction
    # as a separate column along with the probabilities themselves
    if with_soft_voting:
        non_id_columns = [c for c in df_preds.columns if c != "id"]

        # write a simple function to get the argmax
        def get_argmax(row):
            return row.index[row.argmax()]

        # apply the function to each row of the predictions frame
        df_preds["skll"] = df_preds[non_id_columns].apply(get_argmax, axis=1)
    else:
        df_preds.rename(columns={"prediction": "skll"}, inplace=True)

    # now create a voting learner directly in scikit-learn using
    # any of the returned learners - since there is grid search,
    # all the underlying estimators have the same (default)
    # hyper-parameters
    used_estimators = used_models[0].model.named_estimators_
    clf1 = used_estimators[learner_names[0]]["estimator"]
    clf2 = used_estimators[learner_names[1]]["estimator"]
    clf3 = used_estimators[learner_names[2]]["estimator"]

    # instantiate the scikit-learn voting classifier
    sklearn_model_type = VotingClassifier if learner_type == "classifier" else VotingRegressor
    sklearn_model_kwargs = {
        "estimators": [(learner_names[0], clf1), (learner_names[1], clf2), (learner_names[2], clf3)]
    }
    if learner_type == "classifier":
        sklearn_model_kwargs["voting"] = voting_type
    sklearn_vl = sklearn_model_type(**sklearn_model_kwargs)

    # now call `cross_val_predict()` with this learner on the
    # digits data set using the same folds as we did in SKLL;
    # also set the prediction method to be `predict_proba` if
    # we are doing soft voting so that we get probabiities back
    sklearn_predict_method = "predict_proba" if with_soft_voting else "predict"
    sklearn_preds = cross_val_predict(
        sklearn_vl,
        featureset.features,
        featureset.labels,
        cv=splitter,
        method=sklearn_predict_method,
    )

    # save the (argmax-ed) sklearn predictions into our data frame
    if with_soft_voting:
        argmax_label_indices = np.argmax(sklearn_preds, axis=1)
        labels = skll_vl.learners[0].label_list
        sklearn_argmax_preds = np.array([labels[x] for x in argmax_label_indices])
        df_preds["sklearn"] = sklearn_argmax_preds
    else:
        df_preds["sklearn"] = sklearn_preds

    # now check that metrics computed over SKLL and scikit-learn predictions
    # are close enough; we only expect them to match up to 2 decimal places
    # due to various differences between SKLL and scikit-learn
    if learner_type == "classifier":
        skll_metrics = [
            accuracy_score(featureset.labels, df_preds["skll"]),
            f1_score(featureset.labels, df_preds["skll"], average="macro"),
        ]
        sklearn_metrics = [
            accuracy_score(featureset.labels, df_preds["sklearn"]),
            f1_score(featureset.labels, df_preds["sklearn"], average="macro"),
        ]
    else:
        skll_metrics = [
            pearsonr(featureset.labels, df_preds["skll"])[0],
            mean_squared_error(featureset.labels, df_preds["skll"]),
        ]
        sklearn_metrics = [
            pearsonr(featureset.labels, df_preds["sklearn"])[0],
            mean_squared_error(featureset.labels, df_preds["sklearn"]),
        ]

    assert_almost_equal(skll_metrics[0], sklearn_metrics[0], places=2)
    assert_almost_equal(skll_metrics[1], sklearn_metrics[1], places=2)

    # if we asked for individual predictions, make sure that they exist
    # note that we do not need to check that the individual predictions
    # match because we already tested that with `predict()` in
    # `test_voting_learners_api_3.py` and `cross_validate()` calls
    # `predict()` anyway
    if with_individual_predictions:
        for learner_name in learner_names:
            prediction_path = Path(f"{prediction_prefix}_{learner_name}_predictions.tsv")
            ok_(prediction_path.exists())


def test_cross_validate_without_grid_search():
    for learner_type, with_soft_voting, with_individual_predictions in product(
        ["classifier", "regressor"], [False, True], [False, True]
    ):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (
                check_cross_validate_without_grid_search,
                learner_type,
                with_soft_voting,
                with_individual_predictions,
            )
