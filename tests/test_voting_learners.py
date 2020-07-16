# License: BSD 3 clause
"""
Module containing tests for voting learners.

:author: Nitin Madnani (nmadnani@ets.org)
"""

from itertools import product
from os.path import abspath, dirname

from nose.tools import assert_almost_equal, eq_, ok_, raises, with_setup
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises_regex)

from skll import Learner, run_configuration
from skll.data import NDJReader
from skll.learner.voting import VotingLearner
from tests.utils import (fill_in_config_paths_for_single_file,
                         make_classification_data)

_my_dir = abspath(dirname(__file__))


def check_init_voting_learner(learner_type,
                              voting_type,
                              feature_scaling,
                              pos_label_str,
                              min_feature_count,
                              model_kwargs_list,
                              sampler_list):
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

    if learner_type == "classifier":
        if model_kwargs_list is not None:
            model_kwargs_list = [{"C": 0.01}, {"C": 10.0, "kernel": "poly"}]
            kwargs["model_kwargs_list"] = model_kwargs_list
        vl = VotingLearner(["LogisticRegression", "SVC"], **kwargs)
        eq_(len(vl.learners), 2)
        eq_(vl.learners[0].model_type.__name__, "LogisticRegression")
        eq_(vl.learners[1].model_type.__name__, "SVC")
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
    else:
        if model_kwargs_list is not None:
            model_kwargs_list = [{},
                                 {"C": 0.01, "kernel": "linear"},
                                 {"n_estimators": 1000}]
            kwargs["model_kwargs_list"] = model_kwargs_list
        vl = VotingLearner(["LinearRegression",
                            "SVR",
                            "RandomForestRegressor"], **kwargs)
        eq_(len(vl.learners), 3)
        eq_(vl.learners[0].model_type.__name__, "LinearRegression")
        eq_(vl.learners[1].model_type.__name__, "SVR")
        eq_(vl.learners[2].model_type.__name__, "RandomForestRegressor")
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
