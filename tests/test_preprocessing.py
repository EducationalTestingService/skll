# License: BSD 3 clause
"""
Tests related to data preprocessing options with run_experiment.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
"""

import glob
import json
import os
import re
from os.path import join
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.datasets import make_classification
from sklearn.feature_extraction import FeatureHasher

from skll.data import FeatureSet, NDJWriter
from skll.experiments import run_configuration
from skll.learner import Learner
from skll.learner.utils import SelectByMinCount
from skll.utils.constants import KNOWN_DEFAULT_PARAM_GRIDS
from tests import config_dir, output_dir, test_dir, train_dir
from tests.utils import fill_in_config_paths, unlink

_ALL_MODELS = list(KNOWN_DEFAULT_PARAM_GRIDS.keys())
SCORE_OUTPUT_RE = re.compile(r"Objective Function Score \(Test\) = ([\-\d\.]+)")


def setup():
    """
    Create necessary directories for testing.
    """
    for dir_path in [train_dir, test_dir, output_dir]:
        Path(dir_path).mkdir(exist_ok=True)


def tearDown():
    """
    Clean up after tests.
    """

    for output_file in glob.glob(join(output_dir, "test_class_map*")):
        os.unlink(output_file)

    for dir_path in [train_dir, test_dir]:
        unlink(Path(dir_path) / "test_class_map.jsonlines")

    config_files = ["test_class_map.cfg", "test_class_map_feature_hasher.cfg"]
    for cf in config_files:
        unlink(Path(config_dir) / cf)


def test_SelectByMinCount():
    """ Test SelectByMinCount feature selector """
    m2 = [
        [0.001, 0.0, 0.0, 0.0],
        [0.00001, -2.0, 0.0, 0.0],
        [0.001, 0.0, 0.0, 4.0],
        [0.0101, -200.0, 0.0, 0.0],
    ]

    # default should keep all nonzero features (i.e. ones that appear 1+ times)
    feat_selector = SelectByMinCount()
    expected = np.array(
        [
            [0.001, 0.0, 0.0],
            [0.00001, -2.0, 0.0],
            [0.001, 0.0, 4.0],
            [0.0101, -200.0, 0.0],
        ]
    )
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(
        feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected
    )

    # keep features that happen 2+ times
    feat_selector = SelectByMinCount(min_count=2)
    expected = np.array([[0.001, 0.0], [0.00001, -2.0], [0.001, 0.0], [0.0101, -200.0]])
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(
        feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected
    )

    # keep features that happen 3+ times
    feat_selector = SelectByMinCount(min_count=3)
    expected = np.array([[0.001], [0.00001], [0.001], [0.0101]])
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(
        feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected
    )


def make_class_map_data():
    # Create training file
    train_path = join(train_dir, "test_class_map.jsonlines")
    ids = []
    labels = []
    features = []
    class_names = ["beagle", "cat", "dachsund", "cat"]
    for i in range(1, 101):
        y = class_names[i % 4]
        ex_id = f"{y}{i}"
        # note that f1 and f5 are missing in all instances but f4 is not
        x = {"f2": i + 1, "f3": i + 2, "f4": i + 5}
        ids.append(ex_id)
        labels.append(y)
        features.append(x)
    train_fs = FeatureSet("train_class_map", ids, features=features, labels=labels)
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Create test file
    test_path = join(test_dir, "test_class_map.jsonlines")
    ids = []
    labels = []
    features = []
    for i in range(1, 51):
        y = class_names[i % 4]
        ex_id = f"{y}{i}"
        # f1 and f5 are not missing in any instances here but f4 is
        x = {"f1": i, "f2": i + 2, "f3": i % 10, "f5": i * 2}
        ids.append(ex_id)
        labels.append(y)
        features.append(x)
    test_fs = FeatureSet("test_class_map", ids, features=features, labels=labels)
    writer = NDJWriter(test_path, test_fs)
    writer.write()


def test_class_map():
    """
    Test class maps
    """

    make_class_map_data()

    config_template_path = join(config_dir, "test_class_map.template.cfg")
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(
        join(
            output_dir,
            "test_class_map_test_class_map_LogisticRegression.results.json",
        )
    ) as f:
        outd = json.loads(f.read())
        logistic_result_score = outd[0]["accuracy"]

    assert_almost_equal(logistic_result_score, 0.5)


def test_class_map_feature_hasher():
    """
    Test class maps with feature hashing
    """

    make_class_map_data()

    config_template_path = join(
        config_dir, "test_class_map_feature_hasher.template.cfg"
    )
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(
        join(
            output_dir,
            "test_class_map_test_class_map_LogisticRegression.results.json",
        )
    ) as f:
        outd = json.loads(f.read())
        logistic_result_score = outd[0]["accuracy"]

    assert_almost_equal(logistic_result_score, 0.5)


def make_scaling_data(use_feature_hashing=False):

    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=1234567890,
    )

    # we want to arbitrary scale the various features to test the scaling
    scalers = np.array([1, 10, 100, 1000, 10000])
    X = X * scalers

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = [f"EXAMPLE_{n}" for n in range(1, 1001)]

    # create a list of dictionaries as the features
    feature_names = [f"f{n}" for n in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # split everything into training and testing portions
    train_features, test_features = features[:800], features[800:]
    train_y, test_y = y[:800], y[800:]
    train_ids, test_ids = ids[:800], ids[800:]

    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    train_fs = FeatureSet(
        "train_scaling",
        train_ids,
        features=train_features,
        labels=train_y,
        vectorizer=vectorizer,
    )
    test_fs = FeatureSet(
        "test_scaling",
        test_ids,
        features=test_features,
        labels=test_y,
        vectorizer=vectorizer,
    )

    return (train_fs, test_fs)


def check_scaling_features(use_feature_hashing=False, use_scaling=False):
    train_fs, test_fs = make_scaling_data(use_feature_hashing=use_feature_hashing)

    # create a Linear SVM with the value of scaling as specified
    feature_scaling = "both" if use_scaling else "none"
    learner = Learner("SGDClassifier", feature_scaling=feature_scaling, pos_label_str=1)

    # train the learner on the training set and test on the testing set
    learner.train(train_fs, grid_search=True, grid_objective="f1_score_micro")
    test_output = learner.evaluate(test_fs)
    fmeasures = [test_output[2][0]["F-measure"], test_output[2][1]["F-measure"]]

    # these are the expected values of the f-measures, sorted
    if not use_feature_hashing:
        expected_fmeasures = (
            [0.5333333333333333, 0.4842105263157895]
            if not use_scaling
            else [0.7219512195121951, 0.7076923076923077]
        )
    else:
        expected_fmeasures = (
            [0.5288461538461539, 0.4895833333333333]
            if not use_scaling
            else [0.663157894736842, 0.6952380952380952]
        )

    assert_almost_equal(expected_fmeasures, fmeasures)


def test_scaling():
    yield check_scaling_features, False, False
    yield check_scaling_features, False, True
    yield check_scaling_features, True, False
    yield check_scaling_features, True, True
