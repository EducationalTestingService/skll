"""
Cross-validation task experiment tests for voting learners.

The main idea behind these tests is to not run every single possible
experiment but rather to simply confirm that the various options specified
in the configuration file call ``__init__()`` and ``cross_validate()`` with
the right arguments. This is reasonable because those two methods are already
tested comprehensively in ``test_voting_learners_api_4.py``.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from itertools import product
from pathlib import Path
from unittest.mock import DEFAULT, patch

from nose.tools import eq_, ok_

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


def setup():
    """Set up the tests."""
    for dir_path in [train_dir, output_dir]:
        dir_path.mkdir(exist_ok=True)

    # create the training and test data files that we will use
    create_jsonlines_feature_files(train_dir)


def tearDown():
    """Clean up after tests."""
    for output_file_path in output_dir.glob("test_voting_learner_cross_validate*"):
        output_file_path.unlink()

    for output_file_path in Path(".").glob("test_voting_learner_cross_validate*"):
        output_file_path.unlink()

    config_file_path = config_dir / "test_voting_learner_cross_validate.cfg"
    config_file_path.unlink()

    remove_jsonlines_feature_files(train_dir)


def check_xval_task(learner_type, options_dict):
    """Check given combination of cross-validation configuration options."""
    # create a configuration file with the given options
    (
        config_path,
        estimator_names,
        job_name,
        _,
        objectives,
        output_metrics,
        model_kwargs_list,
        param_grid_list,
        sampler_list,
        num_cv_folds,
        cv_seed,
        _,
        _,
    ) = fill_in_config_options_for_voting_learners(learner_type, "cross_validate", options_dict)

    #  mock the `__init__()` method for the `VotingLearner` class so
    # that we can check that the voting learner was instantiated with
    # the right arguments; note that we are doing this mock separately
    # from all the others below since any `__init__`` patch needs to
    # return a `None` value which the other ones do not
    init_patcher = patch.object(VotingLearner, "__init__", return_value=None)

    # mock the `cross_validate()` method for the `VotingLearner` class;
    # this method needs to return a tuple of 3 values where the last item
    # in the tuple is the list of models if we want to save them; since we
    # want to capture the `save()` calls further below, we need to ensure
    # that we return dummy `VotingLearner` model instances as part of the
    # patched return value
    if options_dict["with_save_cv_models"]:
        xval_return_value = (None, None, [VotingLearner(["SVC"])] * num_cv_folds)
    else:
        xval_return_value = (None, None, [])
    xval_patcher = patch.object(VotingLearner, "cross_validate", return_value=xval_return_value)

    # we also need to patch the `_create_learner_result_dicts()` function
    # since there are no actual results
    clrd_patcher = patch("skll.experiments._create_learner_result_dicts", return_value={})

    # we also need to patch some other output functions that write
    # various results to disk since we are not actually producing
    # any results from `cross_validate()`
    output_patchers = patch.multiple(
        "skll.experiments",
        _print_fancy_output=DEFAULT,
        _write_skll_folds=DEFAULT,
        _write_summary_file=DEFAULT,
    )

    # start all the patchers
    mock_vl_init = init_patcher.start()
    mock_vl_xval = xval_patcher.start()
    clrd_patcher.start()
    output_patchers.start()

    # run the configuration file but with various methods/attributes for
    # `VotingLearner` mocked so that we can check that things were called
    # as expected without actually needing to train any models
    with patch.multiple(VotingLearner, save=DEFAULT, model=DEFAULT, create=True) as mocks:
        run_configuration(config_path, quiet=True, local=True)

        # check that init was called the expected number of times;
        # if we are not doing grid search, everything is only called once
        # otherwise they are called as many times as there are objectives
        # (number of featuresets is 1)
        num_expected_init_calls = len(objectives) if options_dict["with_grid_search"] else 1
        eq_(mock_vl_init.call_count, num_expected_init_calls)

        # note that the init arguments are the same no matter the call
        expected_init_args = (estimator_names,)
        expected_init_kwargs = {
            "voting": "soft" if options_dict["with_soft_voting"] else "hard",
            "feature_scaling": "none",
            "pos_label": "dog" if options_dict["with_pos_label"] else None,
            "model_kwargs_list": model_kwargs_list,
            "sampler_list": sampler_list,
            "sampler_kwargs_list": None,
        }

        # check that each init call had the expected arguments
        for actual_call in mock_vl_init.call_args_list:
            eq_(actual_call[0], expected_init_args)
            for key, expected_value in expected_init_kwargs.items():
                actual_value = actual_call[1][key]
                eq_(actual_value, expected_value)

        # check that cross_validate was called the expected number of times
        eq_(mock_vl_xval.call_count, num_expected_init_calls)

        # check that each cross_validate call had the expected arguments
        expected_xval_kwargs = {
            "param_grid_list": param_grid_list,
            "grid_search_folds": 4 if options_dict["with_gs_folds"] else 5,
            "grid_search": options_dict["with_grid_search"],
            "grid_jobs": None,
            "stratified": True,
            "cv_folds": num_cv_folds,
            "cv_seed": cv_seed if options_dict["with_custom_cv_seed"] else 123456789,
            "output_metrics": output_metrics,
            "save_cv_folds": not options_dict["without_save_cv_folds"],
            "save_cv_models": options_dict["with_save_cv_models"],
            "individual_predictions": options_dict["with_individual_predictions"],
        }

        for idx, actual_call in enumerate(mock_vl_xval.call_args_list):
            actual_arg = actual_call[0][0]
            ok_(isinstance(actual_arg, FeatureSet))
            eq_(set(actual_arg.labels), {"cat", "dog"})
            for key, expected_value in expected_xval_kwargs.items():
                actual_value = actual_call[1][key]
                eq_(actual_value, expected_value)

            # check the grid objective value
            eq_(
                actual_call[1]["grid_objective"],
                objectives[idx] if options_dict["with_grid_search"] else None,
            )

            # check the prediction prefix value
            if not options_dict["with_grid_search"] or (
                options_dict["with_grid_search"] and not options_dict["with_multiple_objectives"]
            ):
                expected_prediction_prefix = job_name
            else:
                expected_prediction_prefix = f"{job_name}_{objectives[idx]}"
            eq_(actual_call[1]["prediction_prefix"], expected_prediction_prefix)

        # if we were asked to save the CV models, check that
        # save was called the expected number of times
        # which will be number of folds x number of objectives
        if options_dict["with_save_cv_models"]:
            num_expected_save_calls = (
                num_cv_folds * len(objectives) if options_dict["with_grid_search"] else num_cv_folds
            )
            eq_(mocks["save"].call_count, num_expected_save_calls)

            # check that each save call had the expected arguments
            for idx, actual_call in enumerate(mocks["save"].call_args_list):
                objective_idx, fold_idx = divmod(idx, num_cv_folds)
                if not options_dict["with_grid_search"] or (
                    options_dict["with_grid_search"]
                    and not options_dict["with_multiple_objectives"]
                ):
                    expected_save_args = (output_dir / f"{job_name}_fold{fold_idx+1}.model",)
                else:
                    expected_save_args = (
                        output_dir
                        / f"{job_name}_{objectives[objective_idx]}_fold{fold_idx+1}.model",
                    )

                eq_(actual_call[0], expected_save_args)

    # stop all the patchers
    _ = output_patchers.stop()
    _ = clrd_patcher.stop()
    _ = xval_patcher.stop()
    _ = init_patcher.stop()


def test_xval_task():
    # test various combinations of experiment configuration options
    # NOTE: this is not the full set of combinations since that that would
    # have been too slow; we exclude some options because they are either
    # not very common or because they are already tested in other voting
    # learner experiment tests
    option_names = [
        "with_soft_voting",
        "with_pos_label",
        "with_model_kwargs_list",
        "with_grid_search",
        "with_param_grid_list",
        "with_multiple_objectives",
        "with_gs_folds",
        "with_cv_folds",
        "with_custom_cv_seed",
        "with_output_metrics",
        "without_save_cv_folds",
        "with_save_cv_models",
        "with_individual_predictions",
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

        yield check_xval_task, learner_type, options_dict
