# License: BSD 3 clause
"""
Module containing tests for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import re
from itertools import product
from pathlib import Path
from skll.data.featureset import FeatureSet
from skll import learner

import numpy as np
from nose.tools import (assert_almost_equal,
                        assert_is_not_none,
                        eq_,
                        ok_,
                        raises)
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises_regex)
from numpy.testing._private.utils import assert_allclose
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import (RandomForestRegressor,
                              VotingClassifier,
                              VotingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import (cross_val_predict,
                                     learning_curve,
                                     PredefinedSplit,
                                     ShuffleSplit)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from skll import Learner, run_configuration
from skll.data import NDJReader
from skll.learner.voting import VotingLearner
from skll.metrics import f1_score
from skll.utils.logging import get_skll_logger, close_and_remove_logger_handlers
from tests.other.custom_logistic_wrapper import CustomLogisticRegressionWrapper
from tests.utils import (fill_in_config_paths_for_single_file,
                         make_classification_data,
                         make_california_housing_data,
                         make_digits_data)

MY_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = MY_DIR / "output"
TRAIN_FS_DIGITS, TEST_FS_DIGITS = make_digits_data(use_digit_names=True)
FS_DIGITS, _ = make_digits_data(test_size=0, use_digit_names=True)
TRAIN_FS_HOUSING, TEST_FS_HOUSING = make_california_housing_data(num_examples=2000)
FS_HOUSING, _ = make_california_housing_data(num_examples=2000, test_size=0)
FS_HOUSING.ids = np.arange(2000)
CUSTOM_LEARNER_PATH = MY_DIR / "other" / "custom_logistic_wrapper.py"


def check_initialize(learner_type,
                     voting_type,
                     feature_scaling,
                     pos_label_str,
                     min_feature_count,
                     model_kwargs_list,
                     sampler_list):
    """Run checks for voting learner initialization."""
    # instantiate the keyword arguments for the initialization
    kwargs = {}
    if voting_type:
        kwargs["voting"] = voting_type
    if feature_scaling:
        kwargs["feature_scaling"] = feature_scaling
    if pos_label_str:
        kwargs["pos_label_str"] = pos_label_str
    if min_feature_count:
        kwargs["min_feature_count"] = min_feature_count
    if sampler_list is not None:
        sampler_list = ["RBFSampler", "Nystroem"]
        kwargs["sampler_list"] = sampler_list

    # if the voting learner is a classifier
    if learner_type == "classifier":

        # we are using 2 learners
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]

        # add the model parameters for each of the learners
        if model_kwargs_list is not None:
            given_model_kwargs_list = [{"C": 0.01},
                                       {"C": 10.0, "kernel": "poly"},
                                       {"alpha": 0.75}]
            kwargs["model_kwargs_list"] = given_model_kwargs_list
    else:

        # we are using 2 learners
        learner_names = ["LinearRegression", "SVR", "RandomForestRegressor"]

        # add the model parameters for each of the learners
        if model_kwargs_list is not None:
            given_model_kwargs_list = [{},
                                       {"C": 0.01, "kernel": "linear"},
                                       {"n_estimators": 1000}]
            kwargs["model_kwargs_list"] = given_model_kwargs_list

    # initialize the voting classifier
    vl = VotingLearner(learner_names, **kwargs)

    # check that we have the right number and type of learners
    eq_(len(vl.learners), len(learner_names))
    eq_(vl.learners[0].model_type.__name__, learner_names[0])
    eq_(vl.learners[1].model_type.__name__, learner_names[1])
    eq_(vl.learners[2].model_type.__name__, learner_names[2])

    # check that the probability attribute is properly set
    if learner_type == "classifier":
        eq_(vl.learners[0].probability, voting_type == "soft")
        eq_(vl.learners[1].probability, voting_type == "soft")
        eq_(vl.learners[2].probability, voting_type == "soft")

    # check that we have the right attribute values
    eq_(vl.learner_type, learner_type)
    eq_(vl.label_dict, None)

    # check that voting type is properly set
    if learner_type == "classifier":
        expected_voting_type = "hard" if voting_type is None else voting_type
    else:
        expected_voting_type = None
    eq_(vl.voting, expected_voting_type)

    # check that feature scaling is properly set
    expected_feature_scaling = 'none' if feature_scaling is None else feature_scaling
    eq_(vl.learners[0]._feature_scaling, expected_feature_scaling)
    eq_(vl.learners[1]._feature_scaling, expected_feature_scaling)
    eq_(vl.learners[2]._feature_scaling, expected_feature_scaling)

    # check that any given model kwargs are reflected
    if model_kwargs_list:
        eq_(vl.model_kwargs_list, given_model_kwargs_list)
        if learner_type == "classifier":
            eq_(vl.learners[0].model_kwargs["C"],
                given_model_kwargs_list[0]["C"])
            eq_(vl.learners[1].model_kwargs["C"],
                given_model_kwargs_list[1]["C"])
            eq_(vl.learners[1].model_kwargs["kernel"],
                given_model_kwargs_list[1]["kernel"])
            eq_(vl.learners[2].model_kwargs["alpha"],
                given_model_kwargs_list[2]["alpha"])
        else:
            eq_(vl.learners[1].model_kwargs["C"],
                given_model_kwargs_list[1]["C"])
            eq_(vl.learners[1].model_kwargs["kernel"],
                given_model_kwargs_list[1]["kernel"])
            eq_(vl.learners[2].model_kwargs["n_estimators"],
                given_model_kwargs_list[2]["n_estimators"])
    else:
        eq_(vl.model_kwargs_list, [])

    # check that any given samplers are actually used
    if sampler_list:
        eq_(vl.sampler_list, sampler_list)
        eq_(vl.learners[0].sampler.__class__.__name__, "RBFSampler")
        eq_(vl.learners[1].sampler.__class__.__name__, "Nystroem")
        eq_(vl.learners[2].sampler, None)
    else:
        eq_(vl.sampler_list, [])

    # check that sampler kwargs is reflected
    eq_(vl.sampler_kwargs_list, [])


def test_initialize():
    for (learner_type,
         voting_type,
         feature_scaling,
         pos_label_str,
         min_feature_count,
         model_kwargs_list,
         sampler_list) in product(["classifier", "regressor"],
                                  [None, "hard", "soft"],
                                  [None, "none", "both", "with_mean", "with_std"],
                                  [None, "a"],
                                  [None, 5],
                                  [None, True],
                                  [None, True]):
        yield (check_initialize,
               learner_type,
               voting_type,
               feature_scaling,
               pos_label_str,
               min_feature_count,
               model_kwargs_list,
               sampler_list)


def check_train(learner_type, with_grid_search):
    """Run checks when training voting learners."""

    # if the voting learner is a classifier
    if learner_type == "classifier":
        # use 3 classifiers, the digits training set, and accuracy
        # as the grid search objective
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        estimator_classes = [LogisticRegression, SVC, MultinomialNB]
        featureset = TRAIN_FS_DIGITS
        objective = "accuracy"
    else:
        # otherwise use 3 regressors, the housing training set
        # and pearson as the grid search objective
        learner_names = ["LinearRegression", "SVR", "RandomForestRegressor"]
        estimator_classes = [LinearRegression, SVR, RandomForestRegressor]
        featureset = TRAIN_FS_HOUSING
        objective = "pearson"

    # instantiate and train a voting learner
    vl = VotingLearner(learner_names)
    vl.train(featureset,
             grid_objective=objective,
             grid_search=with_grid_search)

    # check that the training worked
    assert_is_not_none(vl.model)
    model_type = VotingClassifier if learner_type == "classifier" else VotingRegressor
    assert(isinstance(vl.model, model_type))

    # check the underlying learners
    eq_(len(vl.learners), len(learner_names))
    assert(isinstance(vl.learners[0].model, estimator_classes[0]))
    assert(isinstance(vl.learners[1].model, estimator_classes[1]))
    assert(isinstance(vl.learners[2].model, estimator_classes[2]))

    eq_(len(vl.model.named_estimators_), 3)
    pipeline1 = vl.model.named_estimators_[learner_names[0]]
    pipeline2 = vl.model.named_estimators_[learner_names[1]]
    pipeline3 = vl.model.named_estimators_[learner_names[2]]

    assert(isinstance(pipeline1, Pipeline))
    assert(isinstance(pipeline2, Pipeline))
    assert(isinstance(pipeline3, Pipeline))

    assert(isinstance(pipeline1['estimator'], estimator_classes[0]))
    assert(isinstance(pipeline2['estimator'], estimator_classes[1]))
    assert(isinstance(pipeline3['estimator'], estimator_classes[2]))


def test_train():
    for (learner_type,
         with_grid_search) in product(["classifier", "regressor"],
                                      [False, True]):
        yield check_train, learner_type, with_grid_search


def test_train_with_custom_path():
    """Test voting classifier with custom learner path."""

    # instantiate and train a voting classifier on the digits training set
    learner_names = ["CustomLogisticRegressionWrapper", "SVC"]
    vl = VotingLearner(learner_names,
                       custom_learner_path=str(CUSTOM_LEARNER_PATH))
    vl.train(TRAIN_FS_DIGITS,
             grid_objective="accuracy",
             grid_search=False)

    # check that we have a trained model
    assert_is_not_none(vl.model)
    assert(isinstance(vl.model, VotingClassifier))

    # check the underlying learners
    eq_(len(vl.learners), 2)
    eq_(vl.learners[0].model.__class__.__name__, "CustomLogisticRegressionWrapper")
    assert(isinstance(vl.learners[1].model, SVC))
    eq_(len(vl.model.named_estimators_), 2)
    pipeline1 = vl.model.named_estimators_["CustomLogisticRegressionWrapper"]
    pipeline2 = vl.model.named_estimators_["SVC"]
    assert(isinstance(pipeline1, Pipeline))
    assert(isinstance(pipeline2, Pipeline))
    eq_(pipeline1['estimator'].__class__.__name__, "CustomLogisticRegressionWrapper")
    assert(isinstance(pipeline2['estimator'], SVC))


def check_evaluate(learner_type,
                   with_grid_search,
                   with_soft_voting):
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
    skll_vl = VotingLearner(learner_names,
                            voting=voting_type,
                            feature_scaling="none",
                            min_feature_count=0)
    skll_vl.train(train_fs,
                  grid_objective=objective,
                  grid_search=with_grid_search)

    # evaluate on the test set
    res = skll_vl.evaluate(test_fs,
                           grid_objective=objective,
                           output_metrics=[extra_metric])

    # make sure all the parts of the results tuple
    # have the expected types
    ok_(len(res), 6)
    if learner_type == "classifier":
        ok_(isinstance(res[0], list))  # confusion matrix
        ok_(isinstance(res[1], float))  # accuracy
    else:
        eq_(res[0], None)  # no confusion matrix
        eq_(res[1], None)  # no accuracy
    ok_(isinstance(res[2], dict))   # result dict
    ok_(isinstance(res[3], dict))   # model params
    ok_(isinstance(res[4], float))  # objective
    ok_(isinstance(res[5], dict))   # metric scores

    # make sure the model params in the results match what we passed in
    estimators_from_params = res[3]['estimators']
    for idx, (name, estimator) in enumerate(estimators_from_params):
        eq_(name, learner_names[idx])
        ok_(isinstance(estimator, Pipeline))
    if learner_type == "classifier":
        eq_(res[3]['voting'], expected_voting_type)

    # get the values for the objective and the additional metric
    skll_objective = res[4]
    skll_extra_metric = res[5][extra_metric]

    # now get the estimators that underlie the SKLL voting classifier
    # and use them to train a voting classifier directly in scikit-learn
    named_estimators = skll_vl.model.named_estimators_
    clf1 = named_estimators[learner_names[0]]["estimator"]
    clf2 = named_estimators[learner_names[1]]["estimator"]
    clf3 = named_estimators[learner_names[2]]["estimator"]
    sklearn_model_type = (VotingClassifier if learner_type == "classifier"
                          else VotingRegressor)
    sklearn_model_kwargs = {"estimators": [('clf1', clf1),
                                           ('clf2', clf2),
                                           ('clf3', clf3)]}
    if learner_type == "classifier":
        sklearn_model_kwargs["voting"] = voting_type
    sklearn_vl = sklearn_model_type(**sklearn_model_kwargs)
    sklearn_vl.fit(train_fs.features, train_fs.labels)

    # get the predictions from this voting classifier on the test ste
    sklearn_predictions = sklearn_vl.predict(test_fs.features)

    # compute the values of the objective and the extra metric
    # on the scikit-learn side
    if learner_type == "classifier":
        sklearn_objective = accuracy_score(test_fs.labels, sklearn_predictions)
        sklearn_extra_metric = f1_score(test_fs.labels,
                                        sklearn_predictions,
                                        average="macro")
    else:
        sklearn_objective = pearsonr(test_fs.labels, sklearn_predictions)[0]
        sklearn_extra_metric = -1 * mean_squared_error(test_fs.labels,
                                                       sklearn_predictions)

    # check that the values match between SKLL and scikit-learn
    assert_almost_equal(skll_objective, sklearn_objective)
    assert_almost_equal(skll_extra_metric, sklearn_extra_metric)


def test_evaluate():
    for (learner_type,
         with_grid_search,
         with_soft_voting) in product(["classifier", "regressor"],
                                      [False, True],
                                      [False, True]):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (check_evaluate,
                   learner_type,
                   with_grid_search,
                   with_soft_voting)


def check_predict(learner_type,
                  with_grid_search,
                  with_soft_voting,
                  with_class_labels,
                  with_file_output,
                  with_individual_predictions):

    # to test the predict() method, we instantiate the SKLL voting learner,
    # train it on either the digits (classification) or housing (regression)
    # data set, and generate predictions on the corresponding test set; then
    # we do the same in scikit-learn space and compare the objective and
    # compare the SKLL and scikit-learn predictions

    # set the prediction prefix in case we need to write out the predictions
    prediction_prefix = (OUTPUT_DIR / f"test_predict_voting_"
                                      f"{learner_type}_"
                                      f"{with_grid_search}_"
                                      f"{with_class_labels}" if with_file_output else None)
    prediction_prefix = str(prediction_prefix)

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
    skll_vl = VotingLearner(learner_names,
                            feature_scaling="none",
                            min_feature_count=0,
                            voting=voting_type)
    skll_vl.train(train_fs,
                  grid_objective=objective,
                  grid_search=with_grid_search)

    # get the overall and individual predictions from SKLL
    (skll_predictions,
     skll_individual_dict) = skll_vl.predict(test_fs,
                                             class_labels=with_class_labels,
                                             prediction_prefix=prediction_prefix,
                                             individual_predictions=with_individual_predictions)

    # get the underlying scikit-learn estimators from SKLL
    named_estimators = skll_vl.model.named_estimators_
    clf1 = named_estimators[learner_names[0]]["estimator"]
    clf2 = named_estimators[learner_names[1]]["estimator"]
    clf3 = named_estimators[learner_names[2]]["estimator"]

    # instantiate and train the scikit-learn voting classifer
    sklearn_model_type = (VotingClassifier if learner_type == "classifier"
                          else VotingRegressor)
    sklearn_model_kwargs = {"estimators": [(learner_names[0], clf1),
                                           (learner_names[1], clf2),
                                           (learner_names[2], clf3)]}
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
            sklearn_predictions = np.array([skll_vl.label_dict[class_]
                                            for class_ in sklearn_predictions])

    # get the individual scikit-learn predictions, if necessary
    sklearn_individual_dict = {}
    if with_individual_predictions:
        for name, estimator in sklearn_vl.named_estimators_.items():
            estimator_predictions = estimator.predict(test_fs.features)
            # scikit-learn individual predictions are indices not class labels
            # so we need to convert them to labels if required
            if with_class_labels:
                estimator_predictions = [sklearn_vl.classes_[index]
                                         for index in estimator_predictions]
            # if no class labels, then get the probabilities with soft
            # voting; note that sinec the individual predictions from
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
                assert_array_almost_equal(skll_individual_dict["LogisticRegression"],
                                          sklearn_individual_dict["LogisticRegression"],
                                          decimal=2)
                assert_array_almost_equal(skll_individual_dict["MultinomialNB"],
                                          sklearn_individual_dict["MultinomialNB"],
                                          decimal=2)
        # in all other cases, we expect the actual class lables or class indices
        # to be identical between SKLL and scikit-learn
        else:
            assert_array_equal(skll_predictions, sklearn_predictions)

    # for regression, we expect the overall predictions to match exactly
    # but individual predictions only up to 2 decimal places
    else:
        assert_array_equal(skll_predictions, sklearn_predictions)
        if with_individual_predictions:
            assert_array_almost_equal(skll_individual_dict[learner_names[0]],
                                      sklearn_individual_dict[learner_names[0]],
                                      decimal=2)
            assert_array_almost_equal(skll_individual_dict[learner_names[1]],
                                      sklearn_individual_dict[learner_names[1]],
                                      decimal=2)
            assert_array_almost_equal(skll_individual_dict[learner_names[2]],
                                      sklearn_individual_dict[learner_names[2]],
                                      decimal=2)

    # if we were asked to write output to disk, then check that
    # the files actually exist
    if with_file_output:
        ok_(Path(f"{prediction_prefix}_predictions.tsv").exists())
        if with_individual_predictions:
            ok_(Path(f"{prediction_prefix}_{learner_names[0]}_predictions.tsv").exists())
            ok_(Path(f"{prediction_prefix}_{learner_names[1]}_predictions.tsv").exists())
            ok_(Path(f"{prediction_prefix}_{learner_names[2]}_predictions.tsv").exists())


def test_predict():
    for (learner_type,
         with_grid_search,
         with_soft_voting,
         with_class_labels,
         with_file_output,
         with_individual_predictions) in product(["classifier", "regressor"],
                                                 [False, True],
                                                 [False, True],
                                                 [False, True],
                                                 [False, True],
                                                 [False, True]):
        # regressors do not support soft voting or class labels
        if (learner_type == "regressor" and
                (with_soft_voting or with_class_labels)):
            continue
        else:
            yield (check_predict,
                   learner_type,
                   with_grid_search,
                   with_soft_voting,
                   with_class_labels,
                   with_file_output,
                   with_individual_predictions)


def check_learning_curve(learner_type, with_soft_voting):

    # to test the learning_curve() method, we instantiate the SKLL voting
    # learner, get the SKLL learning curve output; then we do the
    # same in scikit-learn space and compare the outputs

    # instantiate some needed variables
    cv_folds = 10
    random_state = 123456789
    cv = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=random_state)
    train_sizes = np.linspace(.1, 1.0, 5)

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

    skll_vl = VotingLearner(learner_names,
                            feature_scaling="none",
                            min_feature_count=0,
                            voting=voting_type)
    (train_scores1,
     test_scores1,
     train_sizes1) = skll_vl.learning_curve(featureset,
                                            cv_folds=cv_folds,
                                            train_sizes=train_sizes,
                                            metric=scoring_function)

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
    sklearn_model_type = (VotingClassifier if learner_type == "classifier"
                          else VotingRegressor)
    sklearn_model_kwargs = {"estimators": [(learner_names[0], clf1),
                                           (learner_names[1], clf2),
                                           (learner_names[2], clf3)]}
    if learner_type == "classifier":
        sklearn_model_kwargs["voting"] = voting_type
    sklearn_vl = sklearn_model_type(**sklearn_model_kwargs)

    # now call `learning_curve()` directly from scikit-learn
    # and get its output
    (train_sizes2,
     train_scores2,
     test_scores2) = learning_curve(sklearn_vl,
                                    featureset.features,
                                    featureset.labels,
                                    cv=cv,
                                    train_sizes=train_sizes,
                                    scoring=scoring_function)

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
    for (learner_type,
         with_soft_voting) in product(["classifier", "regressor"],
                                      [False, True]):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (check_learning_curve,
                   learner_type,
                   with_soft_voting)


@raises(ValueError)
def test_learning_curve_min_examples_check():
    # generates a training split with less than 500 examples
    fs_less_than_500 = FS_DIGITS[:499]

    # create a simple voting classifier
    voting_learner = VotingLearner(["LogisticRegression", "SVC", "MultinomialNB"],
                                   voting="hard")

    # this must throw an error because `examples` has less than 500 items
    _ = voting_learner.learning_curve(examples=fs_less_than_500,
                                      metric="accuracy")


def test_learning_curve_min_examples_check_override():

    # creates a logger which writes to a temporary log file
    log_file_path = (OUTPUT_DIR / "test_check_override_voting_learner_"
                                  "learning_curve_min_examples.log")

    logger = get_skll_logger("test_voting_learner_learning_curve_min_examples",
                             filepath=log_file_path)

    # generates a training split with less than 500 examples
    fs_less_than_500 = FS_DIGITS[:499]

    # create a simple voting classifier
    voting_learner = VotingLearner(["LogisticRegression", "SVC", "MultinomialNB"],
                                   voting="hard",
                                   logger=logger)

    # this must throw an error because `examples` has less than 500 items
    _ = voting_learner.learning_curve(examples=fs_less_than_500,
                                      metric="accuracy",
                                      override_minimum=True)

    # checks that the learning_curve warning message is contained in the log file
    with open(log_file_path) as tf:
        log_text = tf.read()
        learning_curve_warning_re = re.compile(
            r"Learning curves can be unreliable for examples fewer than "
            r"500. You provided \d+\."
        )
        assert learning_curve_warning_re.search(log_text)

    close_and_remove_logger_handlers(logger)


def check_cross_validate_without_grid_search(learner_type, with_soft_voting):

    # to test the cross_validate() method without grid search, we
    # instantiate the SKLL voting learner, call `cross_validate()` on it
    # while writing out the predictions and also asking it to return
    # the actual folds it used as well as the models. Then we use these
    # exact folds with `cross_val_predict()` from scikit-learn as applied
    # to a voting learner instantiated in scikit-learn space. Then we compare
    # the SKLL and scikit-learn cross-validated predictions. Note that since
    # we are not using grid search, the models for all the folds will
    # have the same hyper-parameters. This is why we can use this

    # set the prediction prefix in case we need to write out the predictions
    # TODO: delete these files in tear down
    prediction_prefix = (OUTPUT_DIR / f"test_xval_voting_"
                                      f"{learner_type}_"
                                      f"{with_soft_voting}")
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
    skll_vl = VotingLearner(learner_names,
                            feature_scaling="none",
                            min_feature_count=0,
                            voting=voting_type)
    (xval_results,
     used_fold_ids,
     used_models) = skll_vl.cross_validate(featureset,
                                           grid_search=False,
                                           prediction_prefix=prediction_prefix,
                                           output_metrics=[extra_metric],
                                           save_cv_folds=True,
                                           save_cv_models=True)

    # check that the results are as expected
    ok_(len(xval_results), 10)               # number of folds
    for i in range(10):
        if learner_type == "classifier":
            ok_(isinstance(xval_results[i][0], list))  # confusion matrix
            ok_(isinstance(xval_results[i][1], float))  # accuracy
        else:
            eq_(xval_results[i][0], None)  # no confusion matrix
            eq_(xval_results[i][1], None)  # no accuracy
        ok_(isinstance(xval_results[i][2], dict))   # result dict
        ok_(isinstance(xval_results[i][3], dict))   # model params
        eq_(xval_results[i][4], None)               # No objective
        ok_(isinstance(xval_results[i][5], dict))   # metric scores

    # create a pandas dataframe with the returned fold IDs
    # and create a scikit-learn CV splitter with the exact folds
    df_folds = pd.DataFrame(used_fold_ids.items(), columns=["id", "fold"])
    df_folds = df_folds.sort_values(by="id").reset_index(drop=True)
    splitter = PredefinedSplit(df_folds["fold"].to_numpy())
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

    # now create a voting classifier directly in scikit-learn using
    # any of the returned learners - since there is grid search,
    # all the underlying estimators have the same (default)
    # hyper-parameters
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

    # now call `cross_val_predict()` with this learner on the
    # digits data set using the same folds as we did in SKLL;
    # also set the prediction method to be `predict_proba` if
    # we are doing soft voting so that we get probabiities back
    sklearn_predict_method = "predict_proba" if with_soft_voting else "predict"
    sklearn_preds = cross_val_predict(sklearn_vl,
                                      featureset.features,
                                      featureset.labels,
                                      cv=splitter,
                                      method=sklearn_predict_method)

    # save the (argmax-ed) sklearn predictions into our data frame
    if with_soft_voting:
        argmax_label_indices = np.argmax(sklearn_preds, axis=1)
        labels = skll_vl.learners[0].label_list
        sklearn_argmax_preds = np.array([labels[x] for x in argmax_label_indices])
        df_preds["sklearn"] = sklearn_argmax_preds
    else:
        df_preds["sklearn"] = sklearn_preds

    # now check that the SKLL and scikit-learn predictions match
    # NOTE: unfortunately, likely because of the SVC, we know that the
    # hard-voting classifier predictions will not match for one instance
    # in the set (with id=275)
    if learner_type == "classifier":
        if with_soft_voting:
            assert_array_equal(df_preds["skll"], df_preds["sklearn"])
        else:
            df_preds_filtered = df_preds.drop(275, axis=0)
            assert_array_equal(df_preds_filtered["skll"], df_preds_filtered["sklearn"])
    else:
        # now confirm that the SKLL and scikit-learn predictions match
        assert_allclose(df_preds["skll"], df_preds["sklearn"])


def test_cross_validate_without_grid_search():
    for (learner_type,
         with_soft_voting) in product(["classifier", "regressor"],
                                      [False, True]):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (check_cross_validate_without_grid_search,
                   learner_type,
                   with_soft_voting)
