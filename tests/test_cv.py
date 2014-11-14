# License: BSD 3 clause
"""
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import itertools
from io import open
import os
from os.path import abspath, dirname, join, exists

import numpy as np
from nose.tools import eq_, raises
from six import PY2

from sklearn.feature_extraction import FeatureHasher
from sklearn.datasets.samples_generator import make_classification
from sklearn.utils.testing import assert_greater, assert_less
from skll.data import FeatureSet
from skll.experiments import _load_cv_folds
from skll.learner import Learner
from skll.learner import _DEFAULT_PARAM_GRIDS


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
_my_dir = abspath(dirname(__file__))


def tearDown():
    """
    Clean up after tests.
    """
    fold_file_path = join(_my_dir, 'other', 'custom_folds.csv')
    if exists(fold_file_path):
        os.unlink(fold_file_path)


def make_cv_folds_data(num_examples_per_fold=100,
                       num_folds=3,
                       use_feature_hashing=False):
    """
    Create data for pre-specified CV folds tests
    with or without feature hashing
    """

    num_total_examples = num_examples_per_fold * num_folds

    # create the numeric features and the binary labels
    X, _ = make_classification(n_samples=num_total_examples,
                               n_features=3, n_informative=3, n_redundant=0,
                               n_classes=2, random_state=1234567890)
    y = np.array([0, 1] * int(num_total_examples / 2))

    # the folds mapping: the first num_examples_per_fold examples
    # are in fold 1 the second num_examples_per_fold are in
    # fold 2 and so on
    foldgen = ([str(i)] * num_examples_per_fold for i in range(num_folds))
    folds = list(itertools.chain(*foldgen))

    # now create the list of feature dictionaries
    # and add the binary features that depend on
    # the class and fold number
    feature_names = ['f{}'.format(i) for i in range(1, 4)]
    features = []
    for row, classid, foldnum in zip(X, y, folds):
        string_feature_name = 'is_{}_{}'.format(classid, foldnum)
        string_feature_value = 1
        feat_dict = dict(zip(feature_names, row))
        feat_dict.update({string_feature_name: string_feature_value})
        features.append(feat_dict)

    # create the example IDs
    ids = ['EXAMPLE_{}'.format(num_examples_per_fold * k + i)
           for k in range(num_folds) for i in range(num_examples_per_fold)]

    # create the cross-validation feature set with or without feature hashing
    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    cv_fs = FeatureSet('cv_folds', ids, features=features, labels=y,
                       vectorizer=vectorizer)

    # make the custom cv folds dictionary
    custom_cv_folds = dict(zip(ids, folds))

    return (cv_fs, custom_cv_folds)


def test_specified_cv_folds():
    """
    Test to check cross-validation results with specified folds, feature hashing, and RBFSampler
    """
    # This runs four tests.

    # The first does not use feature hashing with 9 features (3 numeric, 6
    # binary) has pre-specified folds and has less than 60% accuracy for each
    # of the 3 folds.

    # The second uses feature hashing with 4 features, uses 10 folds (not pre-
    # specified) and has more than 70% accuracy accuracy for each of the 10
    # folds.

    # The third is the same as the first but uses an RBFSampler.

    # The fourth is the same as the second but uses an RBFSampler.

    for test_value, assert_func, grid_size, use_hashing, use_sampler in \
            [(0.55, assert_less, 3, False, False),
             (0.1, assert_greater, 10, True, False),
             (0.53, assert_less, 3, False, True),
             (0.7, assert_greater, 10, True, True)]:

        sampler = 'RBFSampler' if use_sampler else None
        learner = Learner('LogisticRegression', sampler=sampler)
        cv_fs, custom_cv_folds = make_cv_folds_data(
            use_feature_hashing=use_hashing)
        folds = custom_cv_folds if not use_hashing else 10
        cv_output = learner.cross_validate(cv_fs,
                                           cv_folds=folds,
                                           grid_search=True)
        fold_test_scores = [t[-1] for t in cv_output[0]]

        overall_score = np.mean(fold_test_scores)

        assert_func(overall_score, test_value)

        eq_(len(fold_test_scores), grid_size)
        for fold_score in fold_test_scores:
            assert_func(fold_score, test_value)


def test_load_cv_folds():
    """
    Test to check that cross-validation folds are correctly loaded from a CSV file
    """

    # create custom CV folds
    custom_cv_folds = make_cv_folds_data()[1]

    # write the generated CV folds to a CSV file
    fold_file_path = join(_my_dir, 'other', 'custom_folds.csv')
    with open(fold_file_path, 'wb' if PY2 else 'w') as foldf:
        w = csv.writer(foldf)
        w.writerow(['id', 'fold'])
        for example_id, fold_label in custom_cv_folds.items():
            w.writerow([example_id, fold_label])

    # now read the CSV file using _load_cv_folds
    custom_cv_folds_loaded = _load_cv_folds(fold_file_path)

    eq_(custom_cv_folds_loaded, custom_cv_folds)


@raises(ValueError)
def test_load_cv_folds_non_float_ids():
    """
    Test to check that CV folds with non-float IDs raise error when converted to floats
    """

    # create custom CV folds
    custom_cv_folds = make_cv_folds_data()[1]

    # write the generated CV folds to a CSV file
    fold_file_path = join(_my_dir, 'other', 'custom_folds.csv')
    with open(fold_file_path, 'wb' if PY2 else 'w') as foldf:
        w = csv.writer(foldf)
        w.writerow(['id', 'fold'])
        for example_id, fold_label in custom_cv_folds.items():
            w.writerow([example_id, fold_label])

    # now read the CSV file using _load_cv_folds, which should raise ValueError
    _load_cv_folds(fold_file_path, ids_to_floats=True)
