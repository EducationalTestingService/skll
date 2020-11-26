# License: BSD 3 clause
"""
Module containing tests for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
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
from sklearn.model_selection import learning_curve, ShuffleSplit
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
FS_DIGITS, _ = make_digits_data(test_size=0)
TRAIN_FS_HOUSING, TEST_FS_HOUSING = make_california_housing_data(num_examples=2000)
FS_HOUSING, _ = make_california_housing_data(num_examples=2000, test_size=0)
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

        # instantiate and train on the digits training set
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

        # instantiate and train on the housing training set
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
    """Run checks when evaluating voting learners."""

    # to test the evaluate() method, we instantiate the SKLL voting learner,
    # train it on either the digits (classification) or housing (regression)
    # data set, and evaluate on the corresponding test set; then we do the
    # same in scikit-learn space and compare the objective and value for
    # on additional output metric

    if learner_type == "classifier":

        # use three classifiers for voting
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]

        # use soft voting type if given
        voting_type = "soft" if with_soft_voting else "hard"

        # instantiate and train a SKLL voting classifier on digits training set
        skll_vl = VotingLearner(learner_names,
                                voting=voting_type,
                                feature_scaling="none",
                                min_feature_count=0)
        skll_vl.train(TRAIN_FS_DIGITS,
                      grid_objective="accuracy",
                      grid_search=with_grid_search)

        # evaluate on the digits test set
        res = skll_vl.evaluate(TEST_FS_DIGITS,
                               grid_objective="accuracy",
                               output_metrics=["f1_score_macro"])

        # make sure all the parts of the results tuple
        # have the expected types
        ok_(len(res), 6)
        ok_(isinstance(res[0], list))   # confusion matrix
        ok_(isinstance(res[1], float))  # accuracy
        ok_(isinstance(res[2], dict))   # result dict
        ok_(isinstance(res[3], dict))   # model params
        ok_(isinstance(res[4], float))  # objective
        ok_(isinstance(res[5], dict))   # metric scores

        # make sure the model params in the results match what we passed in
        estimators_from_params = res[3]['estimators']
        for idx, (name, estimator) in enumerate(estimators_from_params):
            eq_(name, learner_names[idx])
            ok_(isinstance(estimator, Pipeline))
        eq_(res[3]['voting'], voting_type)

        # get the values for the objective and the additional metric
        skll_accuracy = res[1]
        skll_f1_score = res[5]["f1_score_macro"]

        # now get the estimators that underlie the SKLL voting classifier
        # and use them to train a voting classifier directly in scikit-learn
        named_estimators = skll_vl.model.named_estimators_
        clf1 = named_estimators["LogisticRegression"]["estimator"]
        clf2 = named_estimators["SVC"]["estimator"]
        clf3 = named_estimators["MultinomialNB"]["estimator"]
        sklearn_vl = VotingClassifier(estimators=[('lr', clf1),
                                                  ('svc', clf2),
                                                  ('nb', clf3)],
                                      voting=voting_type)
        sklearn_vl.fit(TRAIN_FS_DIGITS.features, TRAIN_FS_DIGITS.labels)

        # get the predictions from this voting classifier on the test ste
        sklearn_predictions = sklearn_vl.predict(TEST_FS_DIGITS.features)

        # compute the accuracy and f1-score values directly
        sklearn_accuracy = accuracy_score(TEST_FS_DIGITS.labels, sklearn_predictions)
        sklearn_f1_score = f1_score(TEST_FS_DIGITS.labels,
                                    sklearn_predictions,
                                    average="macro")

        # check that the values match between SKLL and scikit-learn
        assert_almost_equal(skll_accuracy, sklearn_accuracy)
        assert_almost_equal(skll_f1_score, sklearn_f1_score)
    else:
        # instantiate and train a SKLL voting regressor on housing training set
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        skll_vl = VotingLearner(learner_names,
                                feature_scaling="none",
                                min_feature_count=0)
        skll_vl.train(TRAIN_FS_HOUSING,
                      grid_objective="pearson",
                      grid_search=with_grid_search)

        # evaluate on the housing test set
        res = skll_vl.evaluate(TEST_FS_HOUSING,
                               grid_objective="pearson",
                               output_metrics=["neg_mean_squared_error"])

        # make sure all the parts of the results tuple
        # have the expected types; note that the confusion
        # matrix and accuracy fields are `None` since we
        # are doing regression
        ok_(len(res), 6)
        eq_(res[0], None)               # None
        eq_(res[0], None)               # None
        ok_(isinstance(res[2], dict))   # result dict
        ok_(isinstance(res[3], dict))   # model params
        ok_(isinstance(res[4], float))  # objective
        ok_(isinstance(res[5], dict))   # metric scores

        # make sure the model params match what we passed in
        estimators_from_params = res[3]['estimators']
        for idx, (name, estimator) in enumerate(estimators_from_params):
            eq_(name, learner_names[idx])
            ok_(isinstance(estimator, Pipeline))
        eq_(res[3]['voting'], None)

        # get the values for the objective and the additional metric
        skll_pearson = res[4]
        skll_mse = -1 * res[5]["neg_mean_squared_error"]

        # now get the estimators that underlie the SKLL voting regressor
        # and use them to train a voting regressor directly in scikit-learn
        named_estimators = skll_vl.model.named_estimators_
        clf1 = named_estimators["LinearRegression"]["estimator"]
        clf2 = named_estimators["SVR"]["estimator"]
        clf3 = named_estimators["Ridge"]["estimator"]
        sklearn_vl = VotingRegressor(estimators=[('lr', clf1),
                                                 ('svr', clf2),
                                                 ('rdg', clf3)])
        sklearn_vl.fit(TRAIN_FS_HOUSING.features, TRAIN_FS_HOUSING.labels)

        # get the predictions from this voting regressor on the test set
        sklearn_predictions = sklearn_vl.predict(TEST_FS_HOUSING.features)

        # compute the pearson and MSE values directly
        sklearn_pearson = pearsonr(TEST_FS_HOUSING.labels, sklearn_predictions)[0]
        sklearn_mse = mean_squared_error(TEST_FS_HOUSING.labels, sklearn_predictions)

        # check that the values match between SKLL and scikit-learn
        assert_almost_equal(skll_pearson, sklearn_pearson)
        assert_almost_equal(skll_mse, sklearn_mse)


def test_evaluate_voting_learner():
    for (learner_type,
         with_grid_search,
         with_soft_voting) in product(["classifier", "regressor"],
                                      [False, True],
                                      [False, True]):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (check_evaluate_voting_learner,
                   learner_type,
                   with_grid_search,
                   with_soft_voting)


def check_predict_voting_learner(learner_type,
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

    if learner_type == "classifier":

        # use three classifiers for voting
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]

        # use soft voting type if given
        voting_type = "soft" if with_soft_voting else "hard"

        # instantiate and train the SKLL voting learner on the digits dataset
        skll_vl = VotingLearner(learner_names,
                                feature_scaling="none",
                                min_feature_count=0,
                                voting=voting_type)
        skll_vl.train(TRAIN_FS_DIGITS,
                      grid_objective="accuracy",
                      grid_search=with_grid_search)

        # get the overall and individual predictions from SKLL
        (skll_predictions,
         skll_individual_dict) = skll_vl.predict(TEST_FS_DIGITS,
                                                 class_labels=with_class_labels,
                                                 prediction_prefix=prediction_prefix,
                                                 individual_predictions=with_individual_predictions)

        # get the underlying scikit-learn estimators from SKLL
        named_estimators = skll_vl.model.named_estimators_
        clf1 = named_estimators[learner_names[0]]["estimator"]
        clf2 = named_estimators[learner_names[1]]["estimator"]
        clf3 = named_estimators[learner_names[2]]["estimator"]

        # instantiate and train the scikit-learn voting classifer
        sklearn_vl = VotingClassifier(estimators=[(learner_names[0], clf1),
                                                  (learner_names[1], clf2),
                                                  (learner_names[2], clf3)],
                                      voting=voting_type)
        sklearn_vl.fit(TRAIN_FS_DIGITS.features, TRAIN_FS_DIGITS.labels)

        # get the overall predictions from scikit-learn
        sklearn_predictions = sklearn_vl.predict(TEST_FS_DIGITS.features)

        # get the individual scikit-learn predictions, if necessary
        # note that scikit-learn individual predictions are indices
        # and not class labels so we need to convert them to labels
        # if required
        if with_individual_predictions:
            sklearn_individual_dict = {}
            for name, estimator in sklearn_vl.named_estimators_.items():
                estimator_predictions = estimator.predict(TEST_FS_DIGITS.features)
                if with_class_labels:
                    estimator_predictions = [sklearn_vl.classes_[index] for index in estimator_predictions]
                sklearn_individual_dict[name] = estimator_predictions

        # if we are asked to _not_ output the class labels, then get the
        # either the scikit-learn probabilities or the class indices depending
        # on the voting type (soft and hard respectively)
        if not with_class_labels:
            if voting_type == "soft":
                sklearn_predictions = sklearn_vl.predict_proba(TEST_FS_DIGITS.features)
                if with_individual_predictions:
                    for name, estimator in sklearn_vl.named_estimators_.items():
                        sklearn_individual_dict[name] = estimator.predict_proba(TEST_FS_DIGITS.features)
            else:
                sklearn_predictions = np.array([skll_vl.label_dict[class_] for class_ in sklearn_predictions])
                # note that the individual predictions from scikit-learn are
                # already indices so we do not need to do anything there
    else:
        # use three regressors for voting
        learner_names = ["LinearRegression", "SVR", "Ridge"]

        # instantiate and train a SKLL voting learner on the housing dataset
        skll_vl = VotingLearner(learner_names,
                                feature_scaling="none",
                                min_feature_count=0)
        skll_vl.train(TRAIN_FS_HOUSING,
                      grid_objective="pearson",
                      grid_search=with_grid_search)

        # get the overall and individual predictions from SKLL
        (skll_predictions,
         skll_individual_dict) = skll_vl.predict(TEST_FS_HOUSING,
                                                 class_labels=with_class_labels,
                                                 prediction_prefix=prediction_prefix,
                                                 individual_predictions=with_individual_predictions)

        # get the underlying scikit-learn estimators from SKLL
        named_estimators = skll_vl.model.named_estimators_
        clf1 = named_estimators[learner_names[0]]["estimator"]
        clf2 = named_estimators[learner_names[1]]["estimator"]
        clf3 = named_estimators[learner_names[2]]["estimator"]

        # instantiate and train the scikit-learn voting classifer
        sklearn_vl = VotingRegressor(estimators=[(learner_names[0], clf1),
                                                 (learner_names[1], clf2),
                                                 (learner_names[2], clf3)])
        sklearn_vl.fit(TRAIN_FS_HOUSING.features, TRAIN_FS_HOUSING.labels)

        # get the overall predictions from scikit-learn
        sklearn_predictions = sklearn_vl.predict(TEST_FS_HOUSING.features)

        # get the individual scikit-learn predictions, if necessary
        if with_individual_predictions:
            sklearn_individual_dict = {}
            for name, estimator in sklearn_vl.named_estimators_.items():
                sklearn_individual_dict[name] = estimator.predict(TEST_FS_HOUSING.features)

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
    if (learner_type == "classifier" and
            voting_type == "soft" and
            not with_class_labels):
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
    # to be identical between SKLL and scikit-learn; note that this also
    # includes all regression scenarios
    else:
        assert_array_equal(skll_predictions, sklearn_predictions)
        # for individual predictions, we still only guarantee equality
        # up to two decimal places
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


def test_predict_voting_learner():
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
            yield (check_predict_voting_learner,
                   learner_type,
                   with_grid_search,
                   with_soft_voting,
                   with_class_labels,
                   with_file_output,
                   with_individual_predictions)


def check_learning_curve_implementation(learner_type, with_soft_voting):

    # to test the leanring_curve() method, we instantiate the SKLL voting
    # learner, get the SKLL learning curve output; then we do the
    # same in scikit-learn space and compare the outputs

    # instantiate some needed variables
    cv_folds = 10
    random_state = 123456789
    cv = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=random_state)
    train_sizes = np.linspace(.1, 1.0, 5)

    if learner_type == "classifier":

        # instantiate a SKLL voting classifier and compute
        # the learning curve on the digits data
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        voting_type = "soft" if with_soft_voting else "hard"
        skll_vl = VotingLearner(learner_names,
                                feature_scaling="none",
                                min_feature_count=0,
                                voting=voting_type)
        (train_scores1,
         test_scores1,
         train_sizes1) = skll_vl.learning_curve(FS_DIGITS,
                                                cv_folds=cv_folds,
                                                train_sizes=train_sizes,
                                                metric='accuracy')

        # now instantiate the scikit-learn version with the exact
        # same classifiers;
        # NOTE: here we need to do a bit of hackery
        # to get the same underlying scikit-learn estimators that
        # SKLL would have used since `learning_curve()` doesn't
        # save the underlying estimators like `train()` does
        learner1 = Learner("LogisticRegression", probability=with_soft_voting)
        learner2 = Learner("SVC", probability=with_soft_voting)
        learner3 = Learner("MultinomialNB", probability=with_soft_voting)
        learner1.train(FS_DIGITS[:100], grid_search=False)
        learner2.train(FS_DIGITS[:100], grid_search=False)
        learner3.train(FS_DIGITS[:100], grid_search=False)
        clf1, clf2, clf3 = learner1.model, learner2.model, learner3.model
        sklearn_vl = VotingClassifier(estimators=[('lr', clf1),
                                                  ('svc', clf2),
                                                  ('nb', clf3)],
                                      voting=voting_type)

        # now call `learning_curve()` directly from scikit-learn
        # and get its output
        (train_sizes2,
         train_scores2,
         test_scores2) = learning_curve(sklearn_vl,
                                        FS_DIGITS.features,
                                        FS_DIGITS.labels,
                                        cv=cv,
                                        train_sizes=train_sizes,
                                        scoring='accuracy')

        # now check that SKLL and scikit-learn outputs match
        assert np.all(train_sizes1 == train_sizes2)
        # NOTE: because the digits dataset is quite easy and because
        # we are using SVC, numbers only match up to two significant digits
        assert np.allclose(train_scores1, train_scores2, rtol=1e-2)
        assert np.allclose(test_scores1, test_scores2, rtol=1e-2)

    else:
        # instantiate a SKLL voting regressor and compute the learning
        # curve on the housing data
        learner_names = ["LinearRegression", "SVR", "Ridge"]
        skll_vl = VotingLearner(learner_names,
                                feature_scaling="none",
                                min_feature_count=0)
        (train_scores1,
         test_scores1,
         train_sizes1) = skll_vl.learning_curve(FS_HOUSING,
                                                cv_folds=cv_folds,
                                                train_sizes=train_sizes,
                                                metric='neg_mean_squared_error')

        # now instantiate the scikit-learn version with the exact
        # same regressors;
        # NOTE: here we need to do a bit of hackery
        # to get the same underlying scikit-learn estimators that
        # SKLL would have used since `learning_curve()` doesn't
        # save the underlying estimators like `train()` does
        learner1 = Learner("LinearRegression")
        learner2 = Learner("SVR")
        learner3 = Learner("Ridge")
        learner1.train(FS_HOUSING[:100], grid_search=False)
        learner2.train(FS_HOUSING[:100], grid_search=False)
        learner3.train(FS_HOUSING[:100], grid_search=False)
        clf1, clf2, clf3 = learner1.model, learner2.model, learner3.model
        sklearn_vl = VotingRegressor(estimators=[('lr', clf1),
                                                 ('svr', clf2),
                                                 ('rdg', clf3)])

        # now call `learning_curve()` directly from scikit-learn
        # and get its output
        (train_sizes2,
         train_scores2,
         test_scores2) = learning_curve(sklearn_vl,
                                        FS_HOUSING.features,
                                        FS_HOUSING.labels,
                                        cv=cv,
                                        train_sizes=train_sizes,
                                        scoring='neg_mean_squared_error')

        # now check that the SKLL and scikit-learn outputs match
        assert np.all(train_sizes1 == train_sizes2)
        assert np.allclose(train_scores1, train_scores2)
        assert np.allclose(test_scores1, test_scores2)


def test_learning_curve_implementation_voting_learner():
    for (learner_type,
         with_soft_voting) in product(["classifier", "regressor"],
                                      [False, True]):
        # regressors do not support soft voting
        if learner_type == "regressor" and with_soft_voting:
            continue
        else:
            yield (check_learning_curve_implementation,
                   learner_type,
                   with_soft_voting)
