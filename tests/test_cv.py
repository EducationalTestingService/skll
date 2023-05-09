# License: BSD 3 clause
"""
Run cross-validation tests.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

import csv
import itertools
import json
import re

import numpy as np
from nose.tools import assert_greater, assert_less, eq_, raises
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_classification
from sklearn.feature_extraction import FeatureHasher

from skll.config import load_cv_folds
from skll.data import FeatureSet
from skll.experiments import load_featureset, run_configuration
from skll.learner import Learner
from tests import config_dir, other_dir, output_dir, train_dir
from tests.utils import (
    compute_expected_folds_for_cv_testing,
    create_jsonlines_feature_files,
    fill_in_config_paths_for_single_file,
    remove_jsonlines_feature_files,
    unlink,
)


def setup():
    """Create necessary directories for testing."""
    for dir_path in [train_dir, output_dir]:
        dir_path.mkdir(exist_ok=True)

    # create jsonlines feature files
    create_jsonlines_feature_files(train_dir)


def tearDown():
    """Clean up after tests."""
    fold_file_path = other_dir / "custom_folds.csv"
    unlink(fold_file_path)

    config_dir_obj = config_dir
    cfg_files = [
        config_dir_obj / "test_save_cv_folds.cfg",
        config_dir_obj / "test_save_cv_models.cfg",
        config_dir_obj / "test_custom_cv_seed_classifier.cfg",
        config_dir_obj / "test_custom_cv_seed_regressor.cfg",
        config_dir_obj / "test_folds_file.cfg",
        config_dir_obj / "test_folds_file_grid.cfg",
    ]
    for cfg_file in cfg_files:
        unlink(cfg_file)

    for output_file in itertools.chain(
        output_dir.glob("test_save_cv_folds*"),
        output_dir.glob("test_int_labels_cv_*"),
        output_dir.glob("test_save_cv_models*"),
        output_dir.glob("test_custom_cv_seed*"),
        output_dir.glob("test_folds_file*"),
    ):
        unlink(output_file)

    remove_jsonlines_feature_files(train_dir)


def make_cv_folds_data(num_examples_per_fold=100, num_folds=3, use_feature_hashing=False):
    """Create data for pre-specified CV folds tests with or without feature hashing."""
    num_total_examples = num_examples_per_fold * num_folds

    # create the numeric features and the binary labels
    X, _ = make_classification(
        n_samples=num_total_examples,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=1234567890,
    )
    y = np.array([0, 1] * int(num_total_examples / 2))

    # the folds mapping: the first num_examples_per_fold examples
    # are in fold 1 the second num_examples_per_fold are in
    # fold 2 and so on
    foldgen = ([str(i)] * num_examples_per_fold for i in range(num_folds))
    folds = list(itertools.chain(*foldgen))

    # now create the list of feature dictionaries
    # and add the binary features that depend on
    # the class and fold number
    feature_names = [f"f{i}" for i in range(1, 4)]
    features = []
    for row, classid, foldnum in zip(X, y, folds):
        string_feature_name = f"is_{classid}_{foldnum}"
        string_feature_value = 1
        feat_dict = dict(zip(feature_names, row))
        feat_dict.update({string_feature_name: string_feature_value})
        features.append(feat_dict)

    # create the example IDs
    ids = [
        f"EXAMPLE_{num_examples_per_fold * k + i}"
        for k in range(num_folds)
        for i in range(num_examples_per_fold)
    ]

    # create the cross-validation feature set with or without feature hashing
    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    cv_fs = FeatureSet("cv_folds", ids, features=features, labels=y, vectorizer=vectorizer)

    # make the custom cv folds dictionary
    custom_cv_folds = dict(zip(ids, folds))

    return cv_fs, custom_cv_folds


def test_specified_cv_folds():
    """Check cross-validation results with specified folds, feature hashing, and RBFSampler."""
    # This runs four tests.

    # The first does not use feature hashing with 9 features (3 numeric, 6
    # binary) has pre-specified folds and has less than 60% accuracy for each
    # of the 3 folds.

    # The second uses feature hashing with 4 features, uses 10 folds (not pre-
    # specified) and has more than 70% accuracy accuracy for each of the 10
    # folds.

    # The third is the same as the first but uses an RBFSampler.

    # The fourth is the same as the second but uses an RBFSampler.

    for test_value, assert_func, expected_folds, use_hashing, use_sampler in [
        (0.58, assert_less, 3, False, False),
        (0.1, assert_greater, 10, True, False),
        (0.57, assert_less, 3, False, True),
        (0.69, assert_greater, 10, True, True),
    ]:
        sampler = "RBFSampler" if use_sampler else None
        learner = Learner("LogisticRegression", sampler=sampler)
        cv_fs, custom_cv_folds = make_cv_folds_data(use_feature_hashing=use_hashing)
        folds = custom_cv_folds if not use_hashing else 10
        (grid_scores, _, _, _, _) = learner.cross_validate(
            cv_fs,
            cv_folds=folds,
            grid_search=True,
            grid_objective="f1_score_micro",
            save_cv_folds=False,
        )
        fold_test_scores = [t[-2] for t in grid_scores]

        overall_score = np.mean(fold_test_scores)

        assert_func(overall_score, test_value)

        eq_(len(fold_test_scores), expected_folds)
        for fold_score in fold_test_scores:
            assert_func(fold_score, test_value)


def test_load_cv_folds():
    """Test to check that cross-validation folds are correctly loaded from a CSV file."""
    # create custom CV folds
    custom_cv_folds = make_cv_folds_data()[1]

    # write the generated CV folds to a CSV file
    fold_file_path = other_dir / "custom_folds.csv"
    with open(fold_file_path, "w", newline="") as foldf:
        w = csv.writer(foldf)
        w.writerow(["id", "fold"])
        for example_id, fold_label in custom_cv_folds.items():
            w.writerow([example_id, fold_label])

    # now read the CSV file using _load_cv_folds
    custom_cv_folds_loaded = load_cv_folds(fold_file_path)

    eq_(custom_cv_folds_loaded, custom_cv_folds)


@raises(ValueError)
def test_load_cv_folds_non_float_ids():
    """Test to check that CV folds with non-float IDs raise error when converted to floats."""
    # create custom CV folds
    custom_cv_folds = make_cv_folds_data()[1]

    # write the generated CV folds to a CSV file
    fold_file_path = other_dir / "custom_folds.csv"
    with open(fold_file_path, "w", newline="") as foldf:
        w = csv.writer(foldf)
        w.writerow(["id", "fold"])
        for example_id, fold_label in custom_cv_folds.items():
            w.writerow([example_id, fold_label])

    # now read the CSV file using _load_cv_folds, which should raise ValueError
    load_cv_folds(fold_file_path, ids_to_floats=True)


def test_retrieve_cv_folds():
    """Test to make sure that the fold ids get returned correctly after cross-validation."""
    # Setup
    learner = Learner("LogisticRegression")
    num_folds = 5
    cv_fs, custom_cv_folds = make_cv_folds_data(num_examples_per_fold=2, num_folds=num_folds)

    # Test 1: learner.cross_validate() makes the folds itself.
    expected_fold_ids = {
        "EXAMPLE_0": "0",
        "EXAMPLE_1": "4",
        "EXAMPLE_2": "3",
        "EXAMPLE_3": "1",
        "EXAMPLE_4": "2",
        "EXAMPLE_5": "2",
        "EXAMPLE_6": "1",
        "EXAMPLE_7": "0",
        "EXAMPLE_8": "4",
        "EXAMPLE_9": "3",
    }
    _, _, _, skll_fold_ids, _ = learner.cross_validate(
        cv_fs,
        stratified=True,
        cv_folds=num_folds,
        grid_search=True,
        grid_objective="f1_score_micro",
        shuffle=False,
    )
    eq_(skll_fold_ids, expected_fold_ids)

    # Test 2: if we pass in custom fold ids, those are also preserved.
    _, _, _, skll_fold_ids, _ = learner.cross_validate(
        cv_fs,
        stratified=True,
        cv_folds=custom_cv_folds,
        grid_search=True,
        grid_objective="f1_score_micro",
        shuffle=False,
    )
    eq_(skll_fold_ids, custom_cv_folds)

    # Test 3: when learner.cross_validate() makes the folds but stratified=False
    # and grid_search=False, so that KFold is used.
    expected_fold_ids = {
        "EXAMPLE_0": "0",
        "EXAMPLE_1": "0",
        "EXAMPLE_2": "1",
        "EXAMPLE_3": "1",
        "EXAMPLE_4": "2",
        "EXAMPLE_5": "2",
        "EXAMPLE_6": "3",
        "EXAMPLE_7": "3",
        "EXAMPLE_8": "4",
        "EXAMPLE_9": "4",
    }
    _, _, _, skll_fold_ids, _ = learner.cross_validate(
        cv_fs, stratified=False, cv_folds=num_folds, grid_search=False, shuffle=False
    )
    eq_(skll_fold_ids, custom_cv_folds)


def test_folds_file_logging_num_folds():
    """Test when using `folds_file`, log shows number of folds and appropriate warning."""
    # Run experiment
    suffix = ".jsonlines"
    train_path = train_dir / f"f0{suffix}"

    template_path = config_dir / "test_folds_file.template.cfg"
    config_path = fill_in_config_paths_for_single_file(template_path, train_path, None)
    run_configuration(config_path, quiet=True, local=True)

    # Check experiment log output
    with open(output_dir / "test_folds_file_logging.log") as f:
        cv_file_pattern = re.compile(
            r'Specifying "folds_file" overrides both explicit and default ' r'"num_cv_folds".'
        )
        matches = re.findall(cv_file_pattern, f.read())
        eq_(len(matches), 1)

    # Check job log output
    with open(
        output_dir / "test_folds_file_logging_train_f0." "jsonlines_LogisticRegression.log"
    ) as f:
        cv_folds_pattern = re.compile(
            r"(Task: cross_validate\n)(.+)(Cross-validating \([0-9]+ folds, seed=[0-9]+\))"
        )
        matches = re.findall(cv_folds_pattern, f.read())
        eq_(len(matches), 1)


def test_folds_file_with_fewer_ids_than_featureset():
    """Test when using `folds_file`, log shows warning for extra IDs in featureset."""
    # Run experiment with a special featureset that has extra IDs
    suffix = ".jsonlines"
    train_path = train_dir / f"f5{suffix}"

    template_path = config_dir / "test_folds_file.template.cfg"
    config_path = fill_in_config_paths_for_single_file(template_path, train_path, None)
    run_configuration(config_path, quiet=True, local=True)

    # Check job log output
    with open(
        output_dir / "test_folds_file_logging_train_f5." "jsonlines_LogisticRegression.log"
    ) as f:
        cv_file_pattern = re.compile(
            r"Feature set contains IDs that are not in folds dictionary. " r"Skipping those IDs."
        )
        matches = re.findall(cv_file_pattern, f.read())
        eq_(len(matches), 1)


def test_folds_file_logging_grid_search():
    """
    Test logging with `folds_file`.

    When `folds_file` is used but `use_folds_file` for grid search
    is specified that we get an appropriate message in the log.
    """
    # Run experiment
    suffix = ".jsonlines"
    train_path = train_dir / f"f0{suffix}"

    template_path = config_dir / "test_folds_file_grid.template.cfg"
    config_path = fill_in_config_paths_for_single_file(template_path, train_path, None)
    run_configuration(config_path, quiet=True, local=True)

    # Check experiment log output
    with open(output_dir / "test_folds_file_logging.log") as f:
        cv_file_pattern = re.compile(
            r'Specifying "folds_file" overrides both explicit and default '
            r'"num_cv_folds".\n(.+)The specified "folds_file" will not be '
            r"used for inner grid search."
        )
        matches = re.findall(cv_file_pattern, f.read())
        eq_(len(matches), 1)


def test_cross_validate_task():
    """Test that 10-fold cross_validate experiments work and fold ids get saved."""
    # Run experiment
    suffix = ".jsonlines"
    train_path = train_dir / f"f0{suffix}"

    template_path = config_dir / "test_save_cv_folds.template.cfg"
    config_path = fill_in_config_paths_for_single_file(template_path, train_path, None)
    run_configuration(config_path, quiet=True, local=True)

    # Check final average results
    with open(
        output_dir / "test_save_cv_folds_train_f0.jsonlines_LogisticRegression.results.json"
    ) as f:
        result_dict = json.load(f)[10]

    assert_almost_equal(result_dict["accuracy"], 0.517)

    # Check that the fold ids were saved correctly
    # First compute the expected fold IDs
    examples = load_featureset(train_path, "", suffix, quiet=True)
    expected_skll_ids = compute_expected_folds_for_cv_testing(examples, num_folds=10)
    # read in the computed fold IDs
    skll_fold_ids = {}
    with open(output_dir / "test_save_cv_folds_skll_fold_ids.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skll_fold_ids[row["id"]] = row["cv_test_fold"]

    # convert the dictionary to strings (sorted by key) for quick comparison
    skll_fold_ids_str = "".join(f"{key}{val}" for key, val in sorted(skll_fold_ids.items()))
    expected_skll_ids_str = "".join(f"{key}{val}" for key, val in sorted(expected_skll_ids.items()))

    eq_(skll_fold_ids_str, expected_skll_ids_str)


def test_cross_validate_task_save_cv_models():
    """Test 10-fold cross_validate experiments and model saving."""
    suffix = ".jsonlines"
    train_path = train_dir / f"f0{suffix}"

    template_path = config_dir / "test_save_cv_models.template.cfg"
    config_path = fill_in_config_paths_for_single_file(template_path, train_path, None)
    run_configuration(config_path, quiet=True, local=True)

    cv_model_prefix = "test_save_cv_models_train_f0.jsonlines_LogisticRegression_fold"
    for i in range(1, 11):
        model_path = output_dir / f"{cv_model_prefix}{i}.model"
        assert model_path.exists()


def check_cross_validate_task_with_custom_seed(learner_type, use_config):
    """
    Test cross-validation with custom specified seed.

    Test that cross-validation for either a classifier or a regressor
    with an custom seed and with grid search enabled generates the
    same folds as expected. We run the test using a configuration file
    as well as the API.
    """
    # load in the featureset
    suffix = ".jsonlines"
    train_path = train_dir / f"f0{suffix}"
    if learner_type == "classifier":
        examples = load_featureset(train_path, "", suffix, quiet=True)
    else:
        examples = load_featureset(
            train_path, "", suffix, class_map={"dog": 0, "cat": 1}, quiet=True
        )

    # use a configuration file or the API, as specified
    if use_config:
        template_name = (
            "test_custom_cv_seed_classifier.template.cfg"
            if learner_type == "classifier"
            else "test_custom_cv_seed_regressor.template.cfg"
        )
        template_path = config_dir / template_name
        config_path = fill_in_config_paths_for_single_file(template_path, train_path, None)
        run_configuration(config_path, quiet=True, local=True)

        # read in the folds file produced by SKLL
        computed_fold_ids = {}
        cv_folds_file_name = (
            "test_custom_cv_seed_clf_skll_fold_ids.csv"
            if learner_type == "classifier"
            else "test_custom_cv_seed_reg_skll_fold_ids.csv"
        )
        with open(output_dir / cv_folds_file_name) as f:
            reader = csv.DictReader(f)
            for row in reader:
                computed_fold_ids[row["id"]] = row["cv_test_fold"]

    else:
        learner_name = "LogisticRegression" if learner_type == "classifier" else "Ridge"
        learner = Learner(learner_name)
        objective = "accuracy" if learner_type == "classifier" else "pearson"
        (_, _, _, computed_fold_ids, _) = learner.cross_validate(
            examples, cv_folds=10, cv_seed=54321, grid_search=True, grid_objective=objective
        )

    # compute the fold IDs we expect from SKLL using the custom seed directly;
    # obviously, we only use stratified fold splitting with classifiers
    expected_fold_ids = compute_expected_folds_for_cv_testing(
        examples, stratified=learner_type == "classifier", seed=54321
    )

    # convert the dictionary to strings (sorted by key) for quick comparison
    computed_fold_ids_str = "".join(f"{key}{val}" for key, val in sorted(computed_fold_ids.items()))
    expected_fold_ids_str = "".join(f"{key}{val}" for key, val in sorted(expected_fold_ids.items()))

    # the two sets of folds must be equal
    eq_(computed_fold_ids_str, expected_fold_ids_str)


def test_cross_validate_task_with_custom_seed():
    yield check_cross_validate_task_with_custom_seed, "classifier", False
    yield check_cross_validate_task_with_custom_seed, "classifier", True
    yield check_cross_validate_task_with_custom_seed, "regressor", False
    yield check_cross_validate_task_with_custom_seed, "regressor", True
