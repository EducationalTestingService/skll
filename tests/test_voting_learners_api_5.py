# License: BSD 3 clause
"""
Cross-validation tests with grid search for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from itertools import product
from os.path import exists
from pathlib import Path

import numpy as np
import pandas as pd
from nose.tools import assert_almost_equal, eq_, ok_
from scipy.stats import pearsonr
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import PredefinedSplit

from skll.data import FeatureSet
from skll.learner.voting import VotingLearner
from tests import other_dir, output_dir
from tests.utils import make_california_housing_data, make_digits_data

# define some constants needed for testing
TRAIN_FS_DIGITS, TEST_FS_DIGITS = make_digits_data(use_digit_names=True)
FS_DIGITS, _ = make_digits_data(test_size=0, use_digit_names=True)
TRAIN_FS_HOUSING, TEST_FS_HOUSING = make_california_housing_data(num_examples=2000)
FS_HOUSING, _ = make_california_housing_data(num_examples=2000, test_size=0)
FS_HOUSING.ids = np.arange(2000)
CUSTOM_LEARNER_PATH = Path(other_dir) / "custom_logistic_wrapper.py"


def setup():
    """Set up the tests"""
    for dir_path in [other_dir, output_dir]:
        Path(dir_path).mkdir(exist_ok=True)


def tearDown():
    """Clean up after tests"""
    for output_file_path in Path(output_dir).glob("test_xval_voting_gs*"):
        output_file_path.unlink()


def check_cross_validate_with_grid_search(learner_type,
                                          with_soft_voting,
                                          with_individual_predictions):

    # to test the cross_validate() method with grid search, we
    # instantiate the SKLL voting learner, call `cross_validate()` with
    # 3 folds on it while writing out the predictions and also asking it
    # to return the actual folds it used as well as the models. Then, we take
    # each of the 3 models, take its underlying estimators, use them
    # to train a scikit-learn voting learner directly on the corresponding
    # training fold and make predictions on the test fold. Then we compute
    # metrics over both sets of cross-validated predictions on the
    # test set and compare their values.

    # set the prediction prefix in case we need to write out the predictions
    prediction_prefix = (Path(output_dir) / f"test_xval_voting_gs_"
                                            f"{learner_type}_"
                                            f"{with_soft_voting}")
    prediction_prefix = str(prediction_prefix)

    # set various parameters based on whether we are using
    # a classifier or a regressor
    if learner_type == "classifier":
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        voting_type = "soft" if with_soft_voting else "hard"
        featureset = FS_DIGITS
        objective = "accuracy"
        extra_metric = "f1_score_macro"
    else:
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        voting_type = "hard"
        featureset = FS_HOUSING
        objective = "pearson"
        extra_metric = "neg_mean_squared_error"

    # instantiate and cross-validate the SKLL voting learner
    # on the full digits dataset
    skll_vl = VotingLearner(learner_names,
                            feature_scaling="none",
                            min_feature_count=0,
                            voting=voting_type)
    (xval_results,
     used_fold_ids,
     used_models) = skll_vl.cross_validate(featureset,
                                           grid_search=True,
                                           grid_objective=objective,
                                           grid_search_folds=3,
                                           cv_folds=3,
                                           prediction_prefix=prediction_prefix,
                                           output_metrics=[extra_metric],
                                           save_cv_folds=True,
                                           save_cv_models=True,
                                           individual_predictions=with_individual_predictions)

    # check that the results are as expected
    ok_(len(xval_results), 3)               # number of folds
    for i in range(3):
        if learner_type == "classifier":
            ok_(isinstance(xval_results[i][0], list))  # confusion matrix
            ok_(isinstance(xval_results[i][1], float))  # accuracy
        else:
            eq_(xval_results[i][0], None)  # no confusion matrix
            eq_(xval_results[i][1], None)  # no accuracy
        ok_(isinstance(xval_results[i][2], dict))   # result dict
        ok_(isinstance(xval_results[i][3], dict))   # model params
        ok_(isinstance(xval_results[i][4], float))  # objective
        ok_(isinstance(xval_results[i][5], dict))   # metric scores

    # create a pandas dataframe with the returned fold IDs
    # and create a scikit-learn CV splitter with the exact folds
    df_folds = pd.DataFrame(used_fold_ids.items(), columns=["id", "fold"])
    df_folds = df_folds.sort_values(by="id").reset_index(drop=True)
    splitter = PredefinedSplit(df_folds["fold"].astype(int).to_numpy())
    eq_(splitter.get_n_splits(), 3)

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

    # now iterate over each fold and each model together;
    # create a voting learner directly in scikit-learn using
    # the estimators underlying the model, fit it on the training
    # partition of the fold and then predict on the test partition
    cv_splits = splitter.split()
    for ((train, test), fold_model) in zip(cv_splits, used_models):
        used_estimators = used_models[0].model.named_estimators_
        clf1 = used_estimators[learner_names[0]]["estimator"]
        clf2 = used_estimators[learner_names[1]]["estimator"]
        clf3 = used_estimators[learner_names[2]]["estimator"]

        # instantiate the scikit-learn voting classifier
        sklearn_model_type = (VotingClassifier if learner_type == "classifier"
                              else VotingRegressor)
        sklearn_model_kwargs = {"estimators": [(learner_names[0], clf1),
                                               (learner_names[1], clf2),
                                               (learner_names[2], clf3)]}
        if learner_type == "classifier":
            sklearn_model_kwargs["voting"] = voting_type
        sklearn_vl = sklearn_model_type(**sklearn_model_kwargs)

        train_fs_fold, test_fs_fold = FeatureSet.split_by_ids(featureset, train, test)
        sklearn_vl.fit(train_fs_fold.features, train_fs_fold.labels)

        # save the (argmax-ed) sklearn predictions into our data frame
        # for the test instances in this fold
        if with_soft_voting:
            sklearn_preds_fold = sklearn_vl.predict_proba(test_fs_fold.features)
            argmax_label_indices = np.argmax(sklearn_preds_fold, axis=1)
            sklearn_preds_fold = np.array([sklearn_vl.classes_[x] for x in argmax_label_indices])
        else:
            sklearn_preds_fold = sklearn_vl.predict(test_fs_fold.features)

        df_preds.loc[test, "sklearn"] = sklearn_preds_fold

    # at this point, no sklearn predictions should be NaN
    eq_(len(df_preds[df_preds["sklearn"].isnull()]), 0)

    # now check that metrics computed over SKLL and scikit-learn predictions
    # are close enough; we only expect them to match up to 2 decimal places
    # due to various differences between SKLL and scikit-learn
    if learner_type == "classifier":
        skll_metrics = [accuracy_score(featureset.labels, df_preds["skll"]),
                        f1_score(featureset.labels, df_preds["skll"], average="macro")]
        sklearn_metrics = [accuracy_score(featureset.labels, df_preds["sklearn"]),
                           f1_score(featureset.labels, df_preds["sklearn"], average="macro")]
    else:
        skll_metrics = [pearsonr(featureset.labels, df_preds["skll"])[0],
                        mean_squared_error(featureset.labels, df_preds["skll"])]
        sklearn_metrics = [pearsonr(featureset.labels, df_preds["sklearn"])[0],
                           mean_squared_error(featureset.labels, df_preds["sklearn"])]

    assert_almost_equal(skll_metrics[0], sklearn_metrics[0], places=2)
    assert_almost_equal(skll_metrics[1], sklearn_metrics[1], places=2)

    # if we asked for individual predictions, make sure that they exist
    # note that we do not need to check that the individual predictions
    # match because we already tested that with `predict()` in
    # `test_voting_learners_api_3.py` and `cross_validate()` calls
    # `predict()` anyway
    if with_individual_predictions:
        for learner_name in learner_names:
            ok_(exists(f"{prediction_prefix}_{learner_name}_predictions.tsv"))


def test_cross_validate_with_grid_search():
    for (learner_type,
         with_soft_voting,
         with_individual_predictions) in product(["classifier", "regressor"],
                                                 [False, True],
                                                 [False, True]):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (check_cross_validate_with_grid_search,
                   learner_type,
                   with_soft_voting,
                   with_individual_predictions)
