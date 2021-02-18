"""
Evaluation task experiment tests for voting learners.

The main idea behind these tests is to not run every single possible
experiment but rather to simply confirm that the various options specified
in the configuration file call ``__init__()`` and ``evaluate()`` with the right
arguments. This is reasonable because those two methods are already
tested comprehensively in ``test_voting_learners_api_2.py``.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from itertools import product
from os.path import join
from pathlib import Path
from unittest.mock import DEFAULT, patch

from nose.tools import eq_, ok_

from skll.data import FeatureSet
from skll.experiments import run_configuration
from skll.learner.voting import VotingLearner
from tests import config_dir, other_dir, output_dir, test_dir, train_dir
from tests.utils import (
    BoolDict,
    create_jsonlines_feature_files,
    fill_in_config_options_for_voting_learners,
    remove_jsonlines_feature_files,
)


def setup():
    """Set up the tests"""
    for dir_path in [train_dir, test_dir, output_dir]:
        Path(dir_path).mkdir(exist_ok=True)

    # create the training and test data files that we will use
    create_jsonlines_feature_files(train_dir)
    create_jsonlines_feature_files(test_dir)


def tearDown():
    """ Clean up after tests"""
    for output_file_path in Path(output_dir).glob("test_voting_learner_evaluate*"):
        output_file_path.unlink()

    for output_file_path in Path(".").glob("test_voting_learner_evaluate*"):
        output_file_path.unlink()

    config_file_path = Path(config_dir) / "test_voting_learner_evaluate.cfg"
    config_file_path.unlink()

    remove_jsonlines_feature_files(train_dir)
    remove_jsonlines_feature_files(test_dir)


def check_evaluate_task(learner_type, options_dict):
    """Check given combination of training configuration options"""

    # create a configuration file with the given options
    (config_path,
     estimator_names,
     job_name,
     custom_learner,
     objectives,
     output_metrics,
     model_kwargs_list,
     param_grid_list,
     sampler_list,
     _,
     _,
     _) = fill_in_config_options_for_voting_learners(learner_type,
                                                     "evaluate",
                                                     options_dict)

    #  mock the `__init__()` method for the `VotingLearner` class so
    # that we can check that the voting learner was instantiated with
    # the right arguments; note that we are doing this mock separately
    # from all the others below since any `__init__`` patch needs to
    # return a `None` value which the other ones do not
    init_patcher = patch.object(VotingLearner, "__init__", return_value=None)

    # mock the `from_file()` method for the `VotingLearner` class;
    # this method needs to return an instance of VotingLearner
    from_file_patcher = patch.object(VotingLearner,
                                     "from_file",
                                     return_value=VotingLearner(["SVC"]))

    # we also need to patch the `_create_learner_result_dicts()` function
    # since there are no actual results
    clrd_patcher = patch("skll.experiments._create_learner_result_dicts",
                         return_value={})

    # we also need to patch some other output functions that write
    # various results to disk since we are not actually producing
    # any results from `evaluate()`
    output_patchers = patch.multiple("skll.experiments",
                                     _print_fancy_output=DEFAULT,
                                     _write_summary_file=DEFAULT)

    mock_vl_init = init_patcher.start()
    mock_vl_from_file = from_file_patcher.start()
    clrd_patcher.start()
    output_patchers.start()

    # run the configuration file but with various methods/attributes for
    # `VotingLearner` mocked so that we can check that things were called
    # as expected without actually needing to evaluate any models
    with patch.multiple(VotingLearner,
                        evaluate=DEFAULT,
                        train=DEFAULT,
                        save=DEFAULT,
                        model=DEFAULT,
                        create=True) as mocks:

        run_configuration(config_path,
                          overwrite=not options_dict["with_existing_model"],
                          quiet=True)

        # check that init was called the expected number of times;
        # if we are loading an existing model from disk, it should
        # never be called, otherwise is called as many times as
        # there are objectives (number of featuresets is 1)
        if options_dict["with_existing_model"]:
            num_expected_init_calls = 0
        else:
            num_expected_init_calls = len(objectives) if options_dict["with_grid_search"] else 1
        eq_(mock_vl_init.call_count, num_expected_init_calls)

        # note that the init arguments are the same no matter the call
        expected_init_args = (estimator_names,)
        expected_init_kwargs = {"voting": "soft" if options_dict["with_soft_voting"] else "hard",
                                "custom_learner_path": custom_learner,
                                "feature_scaling": "none",
                                "pos_label_str": None,
                                "min_feature_count": 1,
                                "model_kwargs_list": model_kwargs_list,
                                "sampler_list": sampler_list,
                                "sampler_kwargs_list": None}

        # check that each init call had the expected arguments
        for actual_call in mock_vl_init.call_args_list:
            eq_(actual_call[0], expected_init_args)
            for key, expected_value in expected_init_kwargs.items():
                actual_value = actual_call[1][key]
                eq_(actual_value, expected_value)

        # we either trained a model via `train()` or used an existing
        # model via `from_file()`; check that they were called with
        # the expected arguments
        if not options_dict["with_existing_model"]:
            eq_(mocks['train'].call_count,
                len(objectives) if options_dict["with_grid_search"] else 1)
            expected_train_kwargs = {"param_grid_list": param_grid_list,
                                     "grid_search_folds": 5,
                                     "grid_search": options_dict["with_grid_search"],
                                     "grid_jobs": None,
                                     "shuffle": False}
            for idx, actual_call in enumerate(mocks['train'].call_args_list):
                actual_arg = actual_call[0][0]
                ok_(isinstance(actual_arg, FeatureSet))
                eq_(set(actual_arg.labels), {"cat", "dog"})
                for key, expected_value in expected_train_kwargs.items():
                    actual_value = actual_call[1][key]
                    eq_(actual_value, expected_value)

                # if we aren't doing grid search, then the objective should be `None`
                eq_(actual_call[1]["grid_objective"],
                    objectives[idx] if options_dict["with_grid_search"] else None)

            # if we trained a model, we also saved it
            eq_(mocks['save'].call_count, len(objectives) if options_dict["with_grid_search"] else 1)
            eq_(mocks['save'].call_args[0][0], join(output_dir, f"{job_name}.model"))
        else:
            eq_(mock_vl_from_file.call_count, 1)
            eq_(mock_vl_from_file.call_args[0][0], join(other_dir, f"{job_name}.model"))

        # check that evaluate was called the expected number of times
        eq_(mocks['evaluate'].call_count, len(objectives) if options_dict["with_grid_search"] else 1)

        # check that each evaluate call had the expected arguments
        expected_eval_kwargs = {"grid_objective": objectives[idx] if options_dict["with_grid_search"] else None,
                                "prediction_prefix": join(output_dir, job_name) if options_dict["with_prediction_prefix"] else job_name,
                                "individual_predictions": options_dict["with_individual_predictions"],
                                "output_metrics": output_metrics}

        for idx, actual_call in enumerate(mocks['evaluate'].call_args_list):
            actual_arg = actual_call[0][0]
            ok_(isinstance(actual_arg, FeatureSet))
            eq_(set(actual_arg.labels), {"cat", "dog"})
            for key, expected_value in expected_eval_kwargs.items():
                actual_value = actual_call[1][key]
                eq_(actual_value, expected_value)

    # stop all the manual patchers
    _ = output_patchers.stop()
    _ = clrd_patcher.stop()
    _ = from_file_patcher.stop()
    _ = init_patcher.stop()


def test_evaluate_task():

    # test various combinations of experiment configuration options
    option_names = ["with_soft_voting",
                    "with_model_kwargs_list",
                    "with_grid_search",
                    "with_existing_model",
                    "with_prediction_prefix",
                    "with_individual_predictions",
                    "with_output_metrics"]

    for option_values in product(["classifier", "regressor"],
                                 [False, True],
                                 [False, True],
                                 [False, True],
                                 [False, True],
                                 [False, True],
                                 [False, True],
                                 [False, True]):

        # assign the learner type separately
        learner_type = option_values[0]

        # create a dictionary for all the other options; we are using
        # a dictionary class that returns `False` for non-existent keys
        options_dict = BoolDict(zip(option_names, option_values[1:]))

        # voting regressors do not support soft voting
        if learner_type == "regressor" and options_dict["with_soft_voting"]:
            continue

        # if we are using an existing model, there is no reason for grid search
        if options_dict["with_existing_model"] and options_dict["with_grid_search"]:
            continue

        yield check_evaluate_task, learner_type, options_dict
