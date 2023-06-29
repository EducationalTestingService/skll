"""
Learning curve task experiment tests for voting learners.

The main idea behind these tests is to not run every single possible
experiment but rather to simply confirm that the various options specified
in the configuration file call ``__init__()`` and ``learning_curve()`` with
the right arguments. This is reasonable because those two methods are already
tested comprehensively in ``test_voting_learners_api_2.py``.

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
from tests import config_dir, output_dir, test_dir, train_dir
from tests.utils import (
    BoolDict,
    create_jsonlines_feature_files,
    fill_in_config_options_for_voting_learners,
    remove_jsonlines_feature_files,
)


class TestVotingLearnersExptsFive(unittest.TestCase):
    """Test class for fifth set of voting learner experiment tests."""

    @classmethod
    def setUpClass(cls):
        for dir_path in [train_dir, test_dir, output_dir]:
            dir_path.mkdir(exist_ok=True)

        # create the training and test data files that we will use
        create_jsonlines_feature_files(train_dir)
        create_jsonlines_feature_files(test_dir)

    @classmethod
    def tearDownClass(cls):
        for output_file_path in output_dir.glob("test_voting_learner_learning_curve*"):
            output_file_path.unlink()

        for output_file_path in Path(".").glob("test_voting_learner_learning_curve*"):
            output_file_path.unlink()

        config_file_path = config_dir / "test_voting_learner_learning_curve.cfg"
        config_file_path.unlink()

        remove_jsonlines_feature_files(train_dir)
        remove_jsonlines_feature_files(test_dir)

    def check_learning_curve_task(self, learner_type, options_dict):
        """Check given combination of prediction configuration options."""
        # create a configuration file with the given options
        (
            config_path,
            estimator_names,
            _,
            custom_learner,
            _,
            output_metrics,
            model_kwargs_list,
            _,
            sampler_list,
            _,
            _,
            learning_curve_cv_folds,
            learning_curve_train_sizes,
        ) = fill_in_config_options_for_voting_learners(learner_type, "learning_curve", options_dict)

        #  mock the `__init__()` method for the `VotingLearner` class so
        # that we can check that the voting learner was instantiated with
        # the right arguments; note that we are doing this mock separately
        # from all the others below since any `__init__`` patch needs to
        # return a `None` value which the other ones do not
        init_patcher = patch.object(VotingLearner, "__init__", return_value=None)

        # mock the `learning_curve()` method for the `VotingLearner` class;
        # this method needs to return a tuple of 4 values
        learning_curve_return_value = ([[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]])
        learning_curve_patcher = patch.object(
            VotingLearner, "learning_curve", return_value=learning_curve_return_value
        )

        # we also need to patch the `_create_learner_result_dicts()` function
        # since there are no actual results
        clrd_patcher = patch("skll.experiments._create_learner_result_dicts", return_value={})

        # we also need to patch some other output functions that write
        # various results to disk since we are not actually producing
        # any results from `evaluate()`
        output_patchers = patch.multiple(
            "skll.experiments",
            _print_fancy_output=DEFAULT,
            _write_summary_file=DEFAULT,
            generate_learning_curve_plots=DEFAULT,
        )

        mock_vl_init = init_patcher.start()
        mock_vl_learning_curve = learning_curve_patcher.start()
        clrd_patcher.start()
        output_patchers.start()

        # run the configuration file but with various methods/attributes for
        # `VotingLearner` mocked so that we can check that things were called
        # as expected without actually needing to evaluate any models
        with patch.multiple(VotingLearner, model=DEFAULT, create=True):
            run_configuration(config_path, quiet=True, local=True)

            # check that init was called the expected number of times which
            # is the number of metrics we wanted the learning curve for
            self.assertEqual(mock_vl_init.call_count, len(output_metrics))

            # check that the init call had the expected arguments
            expected_init_args = (estimator_names,)
            expected_init_kwargs = {
                "voting": "soft" if options_dict["with_soft_voting"] else "hard",
                "custom_learner_path": custom_learner,
                "feature_scaling": "none",
                "pos_label": None,
                "min_feature_count": 1,
                "model_kwargs_list": model_kwargs_list,
                "sampler_list": sampler_list,
                "sampler_kwargs_list": None,
            }

            actual_call = mock_vl_init.call_args
            self.assertEqual(actual_call[0], expected_init_args)
            for key, expected_value in expected_init_kwargs.items():
                actual_value = actual_call[1][key]
                self.assertEqual(actual_value, expected_value)

            # check that predict was called the expected number of times
            self.assertEqual(mock_vl_learning_curve.call_count, len(output_metrics))

            # check that each predict call had the expected arguments
            expected_learning_curve_kwargs = {
                "cv_folds": learning_curve_cv_folds,
                "train_sizes": learning_curve_train_sizes,
            }

            for idx, actual_call in enumerate(mock_vl_learning_curve.call_args_list):
                actual_args = actual_call[0]
                self.assertTrue(isinstance(actual_args[0], FeatureSet))
                self.assertEqual(set(actual_args[0].labels), {"cat", "dog"})
                self.assertEqual(actual_args[1], output_metrics[idx])
                for key, expected_value in expected_learning_curve_kwargs.items():
                    actual_value = actual_call[1][key]
                    self.assertEqual(actual_value, expected_value)

        # stop all the manual patchers
        _ = output_patchers.stop()
        _ = clrd_patcher.stop()
        _ = learning_curve_patcher.stop()
        _ = init_patcher.stop()

    def test_learning_curve_task(self):
        # test various combinations of experiment configuration options
        option_names = [
            "with_soft_voting",
            "with_model_kwargs_list",
            "with_learning_curve_cv_folds",
            "with_train_sizes",
            "with_output_metrics",
        ]

        # metrics _always_ need to be specified for learning curve tasks
        for option_values in product(
            ["classifier", "regressor"],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [True],
        ):
            # assign the learner type separately
            learner_type = option_values[0]

            # create a dictionary for all the other options; we are using
            # a dictionary class that returns `False` for non-existent keys
            options_dict = BoolDict(zip(option_names, option_values[1:]))

            # voting regressors do not support soft voting
            if learner_type == "regressor" and options_dict["with_soft_voting"]:
                continue

            yield self.check_learning_curve_task, learner_type, options_dict
