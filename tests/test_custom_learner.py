# License: BSD 3 clause
"""
Module for running tests with custom learners.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

import csv
from itertools import chain

import numpy as np
from nose.tools import raises
from numpy.testing import assert_array_equal

from skll.data import NDJWriter
from skll.experiments import run_configuration
from skll.learner import Learner
from tests import config_dir, other_dir, output_dir, test_dir, train_dir
from tests.utils import fill_in_config_paths, make_classification_data, unlink


def setup():
    """Create necessary directories for testing."""
    for dir_path in [train_dir, test_dir, output_dir]:
        dir_path.mkdir(exist_ok=True)


def tearDown():
    """Clean up after tests."""
    input_files = [
        "test_logistic_custom_learner.jsonlines",
        "test_majority_class_custom_learner.jsonlines",
        "test_model_custom_learner.jsonlines",
    ]
    for inf in input_files:
        for dir_path in [train_dir, test_dir]:
            unlink(dir_path / inf)

    for cfg_file in config_dir.glob("*custom_learner.cfg"):
        unlink(cfg_file)

    for output_file in chain(
        output_dir.glob("test_logistic_custom_learner*"),
        output_dir.glob("test_majority_class_custom_learner*"),
        output_dir.glob("test_model_custom_learner*"),
    ):
        unlink(output_file)


def read_predictions(path):
    """Read in prediction file as a numpy array."""
    with open(path) as f:
        reader = csv.reader(f, dialect="excel-tab")
        next(reader)
        res = np.array([float(x[1]) for x in reader])
    return res


def test_majority_class_custom_learner():
    num_labels = 10

    # This will make data where the last class happens about 50% of the time.
    class_weights = [(0.5 / (num_labels - 1)) for _ in range(num_labels - 1)] + [0.5]
    train_fs, test_fs = make_classification_data(
        num_examples=600,
        train_test_ratio=0.8,
        num_labels=num_labels,
        num_features=5,
        non_negative=True,
        class_weights=class_weights,
    )

    # Write training feature set to a file
    train_path = train_dir / "test_majority_class_custom_learner.jsonlines"
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = test_dir / "test_majority_class_custom_learner.jsonlines"
    writer = NDJWriter(test_path, test_fs)
    writer.write()

    cfgfile = "test_majority_class_custom_learner.template.cfg"
    config_template_path = config_dir / cfgfile
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, local=True)

    outprefix = "test_majority_class_custom_learner"

    prediction_path = output_dir / f"{outprefix}_{outprefix}_MajorityClassLearner_predictions.tsv"
    preds = read_predictions(prediction_path)

    expected = np.array([float(num_labels - 1) for _ in preds])
    assert_array_equal(preds, expected)


def test_logistic_custom_learner():
    num_labels = 10

    class_weights = [(0.5 / (num_labels - 1)) for x in range(num_labels - 1)] + [0.5]
    train_fs, test_fs = make_classification_data(
        num_examples=600,
        train_test_ratio=0.8,
        num_labels=num_labels,
        num_features=5,
        non_negative=True,
        class_weights=class_weights,
    )

    # Write training feature set to a file
    train_path = train_dir / "test_logistic_custom_learner.jsonlines"
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = test_dir / "test_logistic_custom_learner.jsonlines"
    writer = NDJWriter(test_path, test_fs)
    writer.write()

    cfgfile = "test_logistic_custom_learner.template.cfg"
    config_template_path = config_dir / cfgfile
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, local=True)

    outprefix = "test_logistic_custom_learner"
    computed_predictions_path = (
        output_dir / f"{outprefix}_{outprefix}_CustomLogisticRegressionWrapper_predictions.tsv"
    )
    computed_preds = read_predictions(computed_predictions_path)

    expected_predictions_path = (
        output_dir / f"{outprefix}_{outprefix}_LogisticRegression_predictions.tsv"
    )
    expected_preds = read_predictions(expected_predictions_path)

    assert_array_equal(computed_preds, expected_preds)


def test_custom_learner_model_loading():
    num_labels = 10

    class_weights = [(0.5 / (num_labels - 1)) for x in range(num_labels - 1)] + [0.5]
    train_fs, test_fs = make_classification_data(
        num_examples=600,
        train_test_ratio=0.8,
        num_labels=num_labels,
        num_features=5,
        non_negative=True,
        class_weights=class_weights,
    )

    # Write training feature set to a file
    train_path = train_dir / "test_model_custom_learner.jsonlines"
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = test_dir / "test_model_custom_learner.jsonlines"
    writer = NDJWriter(test_path, test_fs)
    writer.write()

    # run the configuration that trains the custom model and saves it
    cfgfile = "test_model_save_custom_learner.template.cfg"
    config_template_path = config_dir / cfgfile
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, local=True)

    # save the predictions from disk into memory
    # and delete the predictions file
    outprefix = "test_model_custom_learner"
    pred_file = (
        output_dir / f"{outprefix}_{outprefix}_CustomLogisticRegressionWrapper_predictions.tsv"
    )
    preds1 = read_predictions(pred_file)
    unlink(pred_file)

    # run the configuration that loads the saved model
    # and generates the predictions again
    cfgfile = "test_model_load_custom_learner.template.cfg"
    config_template_path = config_dir / cfgfile
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, overwrite=False, quiet=True, local=True)

    # load the newly generated predictions
    preds2 = read_predictions(pred_file)

    # make sure that they are the same as before
    assert_array_equal(preds1, preds2)


@raises(ValueError)
def test_custom_learner_api_missing_file():
    _ = Learner("CustomType")


@raises(ValueError)
def test_custom_learner_api_bad_extension():
    _ = Learner(
        "_CustomLogisticRegressionWrapper",
        custom_learner_path=str(other_dir / "custom_learner.txt"),
    )
