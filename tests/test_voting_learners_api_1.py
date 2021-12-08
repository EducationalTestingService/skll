# License: BSD 3 clause
"""
Initialization, training, saving, and loading tests for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from itertools import product
from os.path import exists, join
from pathlib import Path

import numpy as np
from nose.tools import assert_is_not_none, eq_, ok_
from numpy.testing import assert_raises_regex
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

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
    for model_path in [Path("test_current_directory.model"),
                       Path(output_dir) / "test_other_directory.model"]:
        if model_path.exists():
            model_path.unlink()


def check_initialize(learner_type,
                     voting_type,
                     feature_scaling,
                     pos_label,
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
    if pos_label:
        kwargs["pos_label"] = pos_label
    if min_feature_count:
        kwargs["min_feature_count"] = min_feature_count
    if sampler_list is not None:
        sampler_list = ["RBFSampler", "Nystroem", "AdditiveChi2Sampler"]
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
        eq_(vl.learners[2].sampler.__class__.__name__, "AdditiveChi2Sampler")
    else:
        eq_(vl.sampler_list, [])

    # check that sampler kwargs is reflected
    eq_(vl.sampler_kwargs_list, [])


def test_initialize():
    for (learner_type,
         voting_type,
         feature_scaling,
         pos_label,
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
               pos_label,
               min_feature_count,
               model_kwargs_list,
               sampler_list)


def test_initialize_bad_model_kwargs_list():
    assert_raises_regex(ValueError,
                        r"should be a list",
                        VotingLearner,
                        ["SVC", "LogisticRegression", "MultinomialNB"],
                        model_kwargs_list={"C": 0.01})


def test_initialize_bad_sampler_list():
    assert_raises_regex(ValueError,
                        r"should be a list",
                        VotingLearner,
                        ["SVC", "LogisticRegression", "MultinomialNB"],
                        sampler_list="Nystroem")


def test_initialize_bad_sampler_kwargs_list():
    assert_raises_regex(ValueError,
                        r"should be a list",
                        VotingLearner,
                        ["SVC", "LogisticRegression", "MultinomialNB"],
                        sampler_kwargs_list=0.01)


def test_initialize_incorrect_model_kwargs_list():
    assert_raises_regex(ValueError,
                        r"must have 3 entries",
                        VotingLearner,
                        ["SVC", "LogisticRegression", "MultinomialNB"],
                        model_kwargs_list=[{"C": 0.01}, {"C": 0.1}])


def test_initialize_incorrect_sampler_list():
    assert_raises_regex(ValueError,
                        r"must have 3 entries",
                        VotingLearner,
                        ["SVC", "LogisticRegression", "MultinomialNB"],
                        sampler_list=["RBFSampler"])


def test_initialize_incorrect_sampler_kwargs_list():
    assert_raises_regex(ValueError,
                        r"must have 3 entries",
                        VotingLearner,
                        ["SVC", "LogisticRegression", "MultinomialNB"],
                        sampler_kwargs_list=[{"gamma": 1.0}])


def test_intialize_bad_learner_types():
    assert_raises_regex(ValueError,
                        r"cannot mix classifiers and regressors",
                        VotingLearner,
                        ["SVC", "LinearRegression", "MultinomialNB"])


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
             grid_search=with_grid_search,
             grid_search_folds=3)

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


def test_train_bad_param_grid_list():
    vl = VotingLearner(["SVC", "LogisticRegression", "MultinomialNB"])
    assert_raises_regex(ValueError,
                        r"should be a list",
                        vl.train,
                        TRAIN_FS_DIGITS[:100],
                        grid_objective="accuracy",
                        param_grid_list={"C": [0.01, 0.1, 1.0, 10.0]})


def check_save_and_load(learner_type, use_current_directory):
    """Check that saving models works as expected"""
    # if the voting learner is a classifier
    if learner_type == "classifier":
        # use 3 classifiers, the digits training set, and accuracy
        # as the grid search objective
        learner_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        featureset = TRAIN_FS_DIGITS[:100]
    else:
        # otherwise use 3 regressors, the housing training set
        # and pearson as the grid search objective
        learner_names = ["LinearRegression", "SVR", "RandomForestRegressor"]
        featureset = TRAIN_FS_HOUSING[:100]

    # instantiate and train a voting learner without grid search
    vl = VotingLearner(learner_names)
    vl.train(featureset, grid_search=False)

    # save this trained model into the current directory
    if use_current_directory:
        model_name = "test_current_directory.model"
    else:
        model_name = join(output_dir, "test_other_directory.model")

    vl.save(model_name)

    # make sure that the model saved and that it's the same model
    ok_(exists(model_name))
    vl2 = VotingLearner.from_file(model_name)
    eq_(vl._learner_names, vl2._learner_names)
    eq_(vl.model_type, vl2.model_type)
    eq_(vl.voting, vl2.voting)
    eq_(vl.learner_type, vl2.learner_type)


def test_save_and_load():
    for (learner_type,
         use_current_directory) in product(["classifier", "regressor"],
                                           [False, True]):
        yield check_save_and_load, learner_type, use_current_directory
