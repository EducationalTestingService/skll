# License: BSD 3 clause
"""
Module containing tests for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
"""

from itertools import product
from pathlib import Path

import numpy as np
from nose.tools import (assert_almost_equal,
                        assert_is_not_none,
                        eq_,
                        ok_,
                        raises)
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises_regex)
from scipy.stats import pearsonr
from sklearn.ensemble import (RandomForestRegressor,
                              VotingClassifier,
                              VotingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from skll import Learner, run_configuration
from skll.data import NDJReader
from skll.learner.voting import VotingLearner
from skll.metrics import f1_score
from tests.other.custom_logistic_wrapper import CustomLogisticRegressionWrapper
from tests.utils import (fill_in_config_paths_for_single_file,
                         make_classification_data,
                         make_california_housing_data,
                         make_digits_data,)

MY_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = MY_DIR / "output"
TRAIN_FS_DIGITS, TEST_FS_DIGITS = make_digits_data(use_digit_names=True)
TRAIN_FS_HOUSING, TEST_FS_HOUSING = make_california_housing_data(num_examples=2000)
CUSTOM_LEARNER_PATH = MY_DIR / "other" / "custom_logistic_wrapper.py"


def check_init_voting_learner(learner_type,
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

        # add the model parameters for each of the learners
        if model_kwargs_list is not None:
            model_kwargs_list = [{"C": 0.01}, {"C": 10.0, "kernel": "poly"}]
            kwargs["model_kwargs_list"] = model_kwargs_list

        # initialize the voting classifier
        vl = VotingLearner(["LogisticRegression", "SVC"], **kwargs)

        # check that we have the right number and type of learners
        eq_(len(vl.learners), 2)
        eq_(vl.learners[0].model_type.__name__, "LogisticRegression")
        eq_(vl.learners[1].model_type.__name__, "SVC")

        # check that the probability attribute is properly set
        eq_(vl.learners[0].probability, voting_type == "soft")
        eq_(vl.learners[1].probability, voting_type == "soft")

        # check that we have the right attribute values
        eq_(vl.learner_type, learner_type)
        eq_(vl.label_dict, None)
        expected_voting_type = "hard" if voting_type is None else voting_type
        eq_(vl.voting, expected_voting_type)
        expected_feature_scaling = 'none' if feature_scaling is None else feature_scaling
        eq_(vl.learners[0]._feature_scaling, expected_feature_scaling)
        eq_(vl.learners[1]._feature_scaling, expected_feature_scaling)
        if model_kwargs_list:
            eq_(vl.model_kwargs_list, model_kwargs_list)
            eq_(vl.learners[0].model_kwargs["C"], 0.01)
            eq_(vl.learners[1].model_kwargs["C"], 10.0)
            eq_(vl.learners[1].model_kwargs["kernel"], "poly")
        else:
            eq_(vl.model_kwargs_list, [])
        if sampler_list:
            eq_(vl.sampler_list, sampler_list)
            eq_(vl.learners[0].sampler.__class__.__name__, "RBFSampler")
            eq_(vl.learners[1].sampler.__class__.__name__, "Nystroem")
        else:
            eq_(vl.sampler_list, [])
        eq_(vl.sampler_kwargs_list, [])

    # else if it's a voting regressor
    else:

        # add the model parameters for each of the learners
        if model_kwargs_list is not None:
            model_kwargs_list = [{},
                                 {"C": 0.01, "kernel": "linear"},
                                 {"n_estimators": 1000}]
            kwargs["model_kwargs_list"] = model_kwargs_list

        # initialize the voting regressor
        vl = VotingLearner(["LinearRegression",
                            "SVR",
                            "RandomForestRegressor"], **kwargs)

        # check that we have the right number and type of learners
        eq_(len(vl.learners), 3)
        eq_(vl.learners[0].model_type.__name__, "LinearRegression")
        eq_(vl.learners[1].model_type.__name__, "SVR")
        eq_(vl.learners[2].model_type.__name__, "RandomForestRegressor")

        # check that we have the right attribute values
        eq_(vl.learner_type, learner_type)
        eq_(vl.label_dict, None)
        eq_(vl.voting, None)
        expected_feature_scaling = 'none' if feature_scaling is None else feature_scaling
        eq_(vl.learners[0]._feature_scaling, expected_feature_scaling)
        eq_(vl.learners[1]._feature_scaling, expected_feature_scaling)
        if model_kwargs_list:
            eq_(vl.model_kwargs_list, model_kwargs_list)
            eq_(vl.learners[1].model_kwargs["C"], 0.01)
            eq_(vl.learners[1].model_kwargs["kernel"], "linear")
            eq_(vl.learners[2].model_kwargs["n_estimators"], 1000)
        else:
            eq_(vl.model_kwargs_list, [])
        if sampler_list:
            eq_(vl.sampler_list, sampler_list)
            eq_(vl.learners[0].sampler.__class__.__name__, "RBFSampler")
            eq_(vl.learners[1].sampler.__class__.__name__, "Nystroem")
        else:
            eq_(vl.sampler_list, [])
        eq_(vl.sampler_kwargs_list, [])


def test_init_voting_learner():
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
        yield (check_init_voting_learner,
               learner_type,
               voting_type,
               feature_scaling,
               pos_label_str,
               min_feature_count,
               model_kwargs_list,
               sampler_list)


def check_train_voting_learner(learner_type, with_grid_search):
    """Run checks when training voting learners."""

    # if the voting learner is a classifier
    if learner_type == "classifier":

        # instantiate and train on iris example training set
        learner_names = ["LogisticRegression", "SVC"]
        vl = VotingLearner(learner_names)
        vl.train(TRAIN_FS_DIGITS,
                 grid_objective="accuracy",
                 grid_search=with_grid_search)

        # check that the training sets the proper attributes
        assert_is_not_none(vl.model)
        assert(isinstance(vl.model, VotingClassifier))
        eq_(len(vl.learners), 2)
        assert(isinstance(vl.learners[0].model, LogisticRegression))
        assert(isinstance(vl.learners[1].model, SVC))
        eq_(len(vl.model.named_estimators_), 2)
        pipeline1 = vl.model.named_estimators_["LogisticRegression"]
        pipeline2 = vl.model.named_estimators_["SVC"]
        assert(isinstance(pipeline1, Pipeline))
        assert(isinstance(pipeline2, Pipeline))
        assert(isinstance(pipeline1['estimator'], LogisticRegression))
        assert(isinstance(pipeline2['estimator'], SVC))
    # else if it's a regressor
    else:

        # instantiate and train on boston example training set
        vl = VotingLearner(["LinearRegression", "SVR", "RandomForestRegressor"])
        vl.train(TRAIN_FS_HOUSING,
                 grid_objective="pearson",
                 grid_search=with_grid_search)

        # check that the training sets the proper attributes
        assert_is_not_none(vl.model)
        assert(isinstance(vl.model, VotingRegressor))
        eq_(len(vl.learners), 3)
        assert(isinstance(vl.learners[0].model, LinearRegression))
        assert(isinstance(vl.learners[1].model, SVR))
        assert(isinstance(vl.learners[2].model, RandomForestRegressor))
        eq_(len(vl.model.named_estimators_), 3)
        pipeline1 = vl.model.named_estimators_["LinearRegression"]
        pipeline2 = vl.model.named_estimators_["SVR"]
        pipeline3 = vl.model.named_estimators_["RandomForestRegressor"]
        assert(isinstance(pipeline1, Pipeline))
        assert(isinstance(pipeline2, Pipeline))
        assert(isinstance(pipeline3, Pipeline))
        assert(isinstance(pipeline1['estimator'], LinearRegression))
        assert(isinstance(pipeline2['estimator'], SVR))
        assert(isinstance(pipeline3['estimator'], RandomForestRegressor))


def test_train_voting_learner():
    for (learner_type,
         with_grid_search) in product(["classifier", "regressor"],
                                      [False, True]):
        yield check_train_voting_learner, learner_type, with_grid_search


def test_train_voting_learner_with_custom_path():
    """Test voting classifier with custom learner path."""
    # instantiate and train on iris example training set
    learner_names = ["CustomLogisticRegressionWrapper", "SVC"]
    vl = VotingLearner(learner_names,
                       custom_learner_path=str(CUSTOM_LEARNER_PATH))
    vl.train(TRAIN_FS_DIGITS,
             grid_objective="accuracy",
             grid_search=False)
    assert_is_not_none(vl.model)
    assert(isinstance(vl.model, VotingClassifier))
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


def check_evaluate_voting_learner(learner_type,
                                  with_grid_search,
                                  with_soft_voting):
    # instantiate and train, evaluate on test set, compare to sklearn
    # if the voting learner is a classifier
    if learner_type == "classifier":

        # instantiate and train on iris example training set
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        voting_type = "soft" if with_soft_voting else "hard"
        skll_vl = VotingLearner(learner_names,
                                voting=voting_type,
                                feature_scaling="none",
                                min_feature_count=0)
        skll_vl.train(TRAIN_FS_DIGITS,
                      grid_objective="accuracy",
                      grid_search=with_grid_search)
        res = skll_vl.evaluate(TEST_FS_DIGITS,
                               grid_objective="accuracy",
                               output_metrics=["f1_score_macro"])
        skll_accuracy = res[1]
        skll_f1_score = res[-1]["f1_score_macro"]

        named_estimators = skll_vl.model.named_estimators_
        clf1 = named_estimators["LogisticRegression"]["estimator"]
        clf2 = named_estimators["SVC"]["estimator"]
        clf3 = named_estimators["MultinomialNB"]["estimator"]

        sklearn_vl = VotingClassifier(estimators=[('lr', clf1),
                                                  ('svc', clf2),
                                                  ('nb', clf3)],
                                      voting=voting_type)
        sklearn_vl.fit(TRAIN_FS_DIGITS.features, TRAIN_FS_DIGITS.labels)
        sklearn_predictions = sklearn_vl.predict(TEST_FS_DIGITS.features)
        sklearn_accuracy = accuracy_score(TEST_FS_DIGITS.labels, sklearn_predictions)
        sklearn_f1_score = f1_score(TEST_FS_DIGITS.labels,
                                    sklearn_predictions,
                                    average="macro")
        assert_almost_equal(skll_accuracy, sklearn_accuracy)
        assert_almost_equal(skll_f1_score, sklearn_f1_score)
    else:

        # instantiate and train on boston example training set
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        skll_vl = VotingLearner(learner_names,
                                feature_scaling="none",
                                min_feature_count=0)
        skll_vl.train(TRAIN_FS_HOUSING,
                      grid_objective="pearson",
                      grid_search=with_grid_search)
        res = skll_vl.evaluate(TEST_FS_HOUSING,
                               grid_objective="pearson",
                               output_metrics=["neg_mean_squared_error"])
        skll_pearson = res[-2]
        skll_mse = -1 * res[-1]["neg_mean_squared_error"]

        named_estimators = skll_vl.model.named_estimators_
        clf1 = named_estimators["LinearRegression"]["estimator"]
        clf2 = named_estimators["SVR"]["estimator"]
        clf3 = named_estimators["Ridge"]["estimator"]

        sklearn_vl = VotingRegressor(estimators=[('lr', clf1),
                                                 ('svr', clf2),
                                                 ('rdg', clf3)])
        sklearn_vl.fit(TRAIN_FS_HOUSING.features, TRAIN_FS_HOUSING.labels)
        sklearn_predictions = sklearn_vl.predict(TEST_FS_HOUSING.features)
        sklearn_pearson = pearsonr(TEST_FS_HOUSING.labels, sklearn_predictions)[0]
        sklearn_mse = mean_squared_error(TEST_FS_HOUSING.labels, sklearn_predictions)
        assert_almost_equal(skll_pearson, sklearn_pearson)
        assert_almost_equal(skll_mse, sklearn_mse)


def test_evaluate_voting_learner():
    for (learner_type,
         with_grid_search,
         with_soft_voting) in product(["classifier", "regressor"],
                                      [False, True],
                                      [False, True]):
        yield (check_evaluate_voting_learner,
               learner_type,
               with_grid_search,
               with_soft_voting)


def check_predict_voting_learner(learner_type,
                                 with_grid_search,
                                 with_class_labels,
                                 with_file_output):

    prediction_prefix = str(OUTPUT_DIR / f"test_predict_voting_{learner_type}_{with_grid_search}_{with_class_labels}") if with_file_output else None

    # instantiate and train, predict on test set, compare predictions to sklearn
    # if the voting learner is a classifier
    if learner_type == "classifier":

        # instantiate and train on iris example training set
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        skll_vl = VotingLearner(learner_names,
                                feature_scaling="none",
                                min_feature_count=0)
        skll_vl.train(TRAIN_FS_DIGITS,
                      grid_objective="accuracy",
                      grid_search=with_grid_search)

        skll_predictions, _ = skll_vl.predict(TEST_FS_DIGITS,
                                              class_labels=with_class_labels,
                                              prediction_prefix=prediction_prefix)

        named_estimators = skll_vl.model.named_estimators_
        clf1 = named_estimators["LogisticRegression"]["estimator"]
        clf2 = named_estimators["SVC"]["estimator"]
        clf3 = named_estimators["MultinomialNB"]["estimator"]

        sklearn_vl = VotingClassifier(estimators=[('lr', clf1),
                                                  ('svc', clf2),
                                                  ('nb', clf3)])
        sklearn_vl.fit(TRAIN_FS_DIGITS.features, TRAIN_FS_DIGITS.labels)
        sklearn_predictions = sklearn_vl.predict(TEST_FS_DIGITS.features)
        if not with_class_labels:
            sklearn_predictions = np.array([skll_vl.label_dict[class_] for class_ in sklearn_predictions])
    else:
        # instantiate and train on boston example training set
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        skll_vl = VotingLearner(learner_names,
                                feature_scaling="none",
                                min_feature_count=0)
        skll_vl.train(TRAIN_FS_HOUSING,
                      grid_objective="pearson",
                      grid_search=with_grid_search)

        skll_predictions, _ = skll_vl.predict(TEST_FS_HOUSING,
                                              class_labels=with_class_labels,
                                              prediction_prefix=prediction_prefix)

        named_estimators = skll_vl.model.named_estimators_
        clf1 = named_estimators["LinearRegression"]["estimator"]
        clf2 = named_estimators["SVR"]["estimator"]
        clf3 = named_estimators["Ridge"]["estimator"]

        sklearn_vl = VotingRegressor(estimators=[('lr', clf1),
                                                 ('svr', clf2),
                                                 ('rdg', clf3)])
        sklearn_vl.fit(TRAIN_FS_HOUSING.features, TRAIN_FS_HOUSING.labels)
        sklearn_predictions = sklearn_vl.predict(TEST_FS_HOUSING.features)

    assert_array_equal(skll_predictions, sklearn_predictions)
    if with_file_output:
        ok_(Path(f"{prediction_prefix}_predictions.tsv").exists())


def test_predict_voting_learner():
    for (learner_type,
         with_grid_search,
         with_class_labels,
         with_file_output) in product(["classifier", "regressor"],
                                      [False, True],
                                      [False, True],
                                      [False, True]):
        yield (check_predict_voting_learner,
               learner_type,
               with_grid_search,
               with_class_labels,
               with_file_output)
