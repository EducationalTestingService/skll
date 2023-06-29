"""
Training task experiment tests for voting learners.

The main idea behind these tests is to not run every single possible
experiment but rather to simply confirm that the various options specified
in the configuration file call ``__init__()`` and ``train()`` with the right
arguments. This is reasonable because those two methods are already
tested comprehensively in ``test_voting_learners_api_1.py``.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import unittest
from itertools import product
from pathlib import Path
from unittest.mock import DEFAULT, patch

from skll.data import FeatureSet
from skll.experiments import run_configuration
from skll.learner.voting import VotingLearner
from tests import config_dir, output_dir, train_dir
from tests.utils import (
    BoolDict,
    create_jsonlines_feature_files,
    fill_in_config_options_for_voting_learners,
    remove_jsonlines_feature_files,
)


class TestVotingLearnersExptsOne(unittest.TestCase):
    """Test class for first set of voting learner experiment tests."""

    @classmethod
    def setUpClass(cls):
        """Set up the tests."""
        for dir_path in [train_dir, output_dir]:
            dir_path.mkdir(exist_ok=True)

        # create the training and test data files that we will use
        create_jsonlines_feature_files(train_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        for output_file_path in output_dir.glob("test_voting_learner_train*"):
            output_file_path.unlink()

        for output_file_path in Path(".").glob("test_voting_learner_train*"):
            output_file_path.unlink()

        config_file_path = config_dir / "test_voting_learner_train.cfg"
        config_file_path.unlink()

        remove_jsonlines_feature_files(train_dir)

    def check_train_task(self, learner_type, options_dict):
        """Check given combination of training configuration options."""
        # create a configuration file with the given options
        (
            config_path,
            estimator_names,
            job_name,
            custom_learner,
            objectives,
            _,
            model_kwargs_list,
            param_grid_list,
            sampler_list,
            _,
            _,
            _,
            _,
        ) = fill_in_config_options_for_voting_learners(learner_type, "train", options_dict)

        #  mock the `__init__()` method for the `VotingLearner` class so
        # that we can check that the voting learner was instantiated with
        # the right arguments; note that we are doing this mock separately
        # from all the others below since any `__init__`` patch needs to
        # return a `None` value which the other ones do not
        patcher = patch.object(VotingLearner, "__init__", return_value=None)
        mock_vl_init = patcher.start()

        # run the configuration file but with various methods/attributes for
        # `VotingLearner` mocked so that we can check that things were called
        # as expected without actually needing to train any models
        with patch.multiple(
            VotingLearner, train=DEFAULT, save=DEFAULT, model=DEFAULT, create=True
        ) as mocks:
            run_configuration(config_path, quiet=True, local=True)

            # check that init was called the expected number of times;
            # if we are not doing grid search, everything is only called once
            # otherwise they are called as many times as there are objectives
            # (number of featuresets is 1)
            num_expected_calls = len(objectives) if options_dict["with_grid_search"] else 1
            self.assertEqual(mock_vl_init.call_count, num_expected_calls)

            # note that the init arguments are the same no matter the call
            expected_init_args = (estimator_names,)
            expected_init_kwargs = {
                "voting": "soft" if options_dict["with_soft_voting"] else "hard",
                "custom_learner_path": custom_learner,
                "feature_scaling": "with_mean" if options_dict["with_centering"] else "none",
                "pos_label": "dog" if options_dict["with_pos_label"] else None,
                "min_feature_count": 2 if options_dict["with_min_feature_count"] else 1,
                "model_kwargs_list": model_kwargs_list,
                "sampler_list": sampler_list,
                "sampler_kwargs_list": None,
            }

            # check that each init call had the expected arguments
            for actual_call in mock_vl_init.call_args_list:
                self.assertEqual(actual_call[0], expected_init_args)
                for key, expected_value in expected_init_kwargs.items():
                    actual_value = actual_call[1][key]
                    self.assertEqual(actual_value, expected_value)

            # check that train was called the expected number of times
            self.assertEqual(mocks["train"].call_count, num_expected_calls)

            # check that each train call had the expected arguments
            expected_train_kwargs = {
                "param_grid_list": param_grid_list,
                "grid_search_folds": 4 if options_dict["with_gs_folds"] else 5,
                "grid_search": options_dict["with_grid_search"],
                "grid_jobs": None,
                "shuffle": options_dict["with_shuffle"],
            }

            for idx, actual_call in enumerate(mocks["train"].call_args_list):
                actual_arg = actual_call[0][0]
                self.assertTrue(isinstance(actual_arg, FeatureSet))
                self.assertEqual(set(actual_arg.labels), {"cat", "dog"})
                for key, expected_value in expected_train_kwargs.items():
                    actual_value = actual_call[1][key]
                    self.assertEqual(actual_value, expected_value)

                # if we aren't doing grid search, then the objective should be `None`
                self.assertEqual(
                    actual_call[1]["grid_objective"],
                    objectives[idx] if options_dict["with_grid_search"] else None,
                )

            # check that save was called the expected number of times
            self.assertEqual(mocks["save"].call_count, num_expected_calls)

            # check that each save call had the expected arguments
            for idx, actual_call in enumerate(mocks["save"].call_args_list):
                if not options_dict["with_grid_search"] or (
                    options_dict["with_grid_search"]
                    and not options_dict["with_multiple_objectives"]
                ):
                    expected_save_args = (output_dir / f"{job_name}.model",)
                else:
                    expected_save_args = (output_dir / f"{job_name}_{objectives[idx]}.model",)
                self.assertEqual(actual_call[0], expected_save_args)

        # stop all the manual patchers
        _ = patcher.stop()

    def test_train_task(self):
        # test various combinations of experiment configuration options
        option_names = [
            "with_soft_voting",
            "with_centering",
            "with_min_feature_count",
            "with_custom_learner_path",
            "with_pos_label",
            "with_model_kwargs_list",
            "with_sampler_list",
            "with_grid_search",
            "with_param_grid_list",
            "with_shuffle",
            "with_multiple_objectives",
            "with_gs_folds",
        ]

        for option_values in product(
            ["classifier", "regressor"],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
        ):
            # assign the learner type separately
            learner_type = option_values[0]

            # create a dictionary for all the other options; we are using
            # a dictionary class that returns `False` for non-existent keys
            options_dict = BoolDict(zip(option_names, option_values[1:]))

            # voting regressors do not support soft voting or `pos_label`
            if learner_type == "regressor" and (
                options_dict["with_soft_voting"] or options_dict["with_pos_label"]
            ):
                continue

            yield self.check_train_task, learner_type, options_dict
