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

import itertools
import os
from collections import OrderedDict
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
from nose.tools import eq_, raises, assert_not_equal
from nose.plugins.attrib import attr
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.datasets.samples_generator import make_classification

from skll.data import FeatureSet, Writer, Reader
from skll.data.readers import DictListReader
from skll.experiments import _load_featureset
from skll.learner import _DEFAULT_PARAM_GRIDS
from skll.utilities import skll_convert

from utils import make_classification_data, make_regression_data


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
_my_dir = abspath(dirname(__file__))


def setup():
    """
    Create necessary directories for testing.
    """
    train_dir = join(_my_dir, 'train')
    if not exists(train_dir):
        os.makedirs(train_dir)
    test_dir = join(_my_dir, 'test')
    if not exists(test_dir):
        os.makedirs(test_dir)
    output_dir = join(_my_dir, 'output')
    if not exists(output_dir):
        os.makedirs(output_dir)


@raises(ValueError)
def test_empty_ids():
    """
    Test to ensure that an error is raised if ids is None
    """

    # get a 100 instances with 4 features each
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=4, n_redundant=0,
                               n_classes=3, random_state=1234567890)

    # convert the features into a list of dictionaries
    feature_names = ['f{}'.format(n) for n in range(1, 5)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # create a feature set with ids set to None and raise ValueError
    FeatureSet('test', None, features=features, labels=y)


def test_empty_labels():
    """
    Test to check behaviour when labels is None
    """

    # create a feature set with empty labels
    fs, _ = make_classification_data(num_examples=100,
                                     num_features=4,
                                     num_labels=3,
                                     empty_labels=True,
                                     train_test_ratio=1.0)
    assert np.isnan(fs.labels).all()


def test_length():
    """
    Test to whether len() returns the number of instances
    """

    # create a featureset
    fs, _ = make_classification_data(num_examples=100,
                                     num_features=4,
                                     num_labels=3,
                                     train_test_ratio=1.0)

    eq_(len(fs), 100)


def test_string_feature():
    """
    Test to make sure that string-valued features are properly
    encoded as binary features
    """
    # create a featureset that is derived from an original
    # set of features containing 3 numeric features and
    # one string-valued feature that can take six possible
    # values between 'a' to 'f'. This means that the
    # featureset will have 3 numeric + 6 binary features.
    fs, _ = make_classification_data(num_examples=100,
                                     num_features=4,
                                     num_labels=3,
                                     one_string_feature=True,
                                     num_string_values=6,
                                     train_test_ratio=1.0)

    # confirm that the number of features are as expected
    eq_(fs.features.shape, (100, 9))

    # confirm the feature names
    eq_(fs.vectorizer.feature_names_, ['f01', 'f02', 'f03',
                                       'f04=a', 'f04=b', 'f04=c',
                                       'f04=d', 'f04=e', 'f04=f'])

    # confirm that the final six features are binary
    assert_array_equal(fs.features[:, [3, 4, 5, 6, 7, 8]].data, 1)


def test_equality():
    """
    Test featureset equality
    """

    # create a featureset
    fs1, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    # create a featureset with a different set but same number
    # of features and everything else the same
    fs2, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    fs2.features *= 2

    # create a featureset with different feature names
    # and everything else the same
    fs3, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      feature_prefix='g',
                                      train_test_ratio=1.0)

    # create a featureset with a different set of labels
    # and everything else the same
    fs4, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=2,
                                      train_test_ratio=1.0)

    # create a featureset with a different set but same number
    # of IDs and everything else the same
    fs5, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0)
    fs5.ids = np.array(['A' + i for i in fs2.ids])

    # create a featureset with a different vectorizer
    # and everything else the same
    fs6, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0,
                                      use_feature_hashing=True,
                                      feature_bins=2)

    # create a featureset with a different number of features
    # and everything else the same
    fs7, _ = make_classification_data(num_examples=100,
                                      num_features=5,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    # create a featureset with a different number of examples
    # and everything else the same
    fs8, _ = make_classification_data(num_examples=200,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    # create a featureset with a different vectorizer instance
    # and everything else the same
    fs9, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    # now check for the expected equalities
    assert_not_equal(fs1, fs2)
    assert_not_equal(fs1, fs3)
    assert_not_equal(fs1, fs4)
    assert_not_equal(fs1, fs5)
    assert_not_equal(fs1, fs6)
    assert_not_equal(fs1, fs7)
    assert_not_equal(fs1, fs8)
    assert_not_equal(id(fs1.vectorizer), id(fs9.vectorizer))
    eq_(fs1, fs9)


@raises(ValueError)
def test_merge_different_vectorizers():
    """
    Test to ensure rejection of merging featuresets with different vectorizers
    """

    # create a featureset each with a DictVectorizer
    fs1, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    # create another featureset using hashing
    fs2, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      feature_prefix='g',
                                      num_labels=3,
                                      train_test_ratio=1.0,
                                      use_feature_hashing=True)
    # This should raise a ValueError
    fs1 + fs2


@raises(ValueError)
def test_merge_different_hashers():
    """
    Test to ensure rejection of merging featuresets with different FeatureHashers
    """

    # create a feature set with 4 feature hashing bins
    fs1, _ = make_classification_data(num_examples=100,
                                      num_features=10,
                                      num_labels=3,
                                      train_test_ratio=1.0,
                                      use_feature_hashing=True,
                                      feature_bins=4)

    # create a second feature set with 3 feature hashing bins
    fs2, _ = make_classification_data(num_examples=100,
                                      num_features=10,
                                      num_labels=3,
                                      feature_prefix='g',
                                      train_test_ratio=1.0,
                                      use_feature_hashing=True,
                                      feature_bins=3)
    # This should raise a ValueError
    fs1 + fs2


@raises(ValueError)
def test_merge_different_labels_same_ids():
    """
    Test to ensure rejection of merging featuresets that have conflicting labels
    """

    # create a feature set
    fs1, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    # create a different feature set that has everything
    # the same but has different labels for the same IDs
    fs2, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      feature_prefix='g',
                                      train_test_ratio=1.0)

    # artificially modify the class labels
    fs2.labels = fs2.labels + 1

    # This should raise a ValueError
    fs1 + fs2


def test_merge_missing_labels():
    """
    Test to ensure that labels are sucessfully copied when merging
    """

    # create a feature set
    fs1, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    # create a different feature set with no labels specified
    fs2, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      feature_prefix='g',
                                      empty_labels=True,
                                      num_labels=3,
                                      train_test_ratio=1.0)

    # merge the two featuresets in different orders
    fs12 = fs1 + fs2
    fs21 = fs2 + fs1

    # make sure that the labels are the same after merging
    assert_array_equal(fs12.labels, fs1.labels)
    assert_array_equal(fs21.labels, fs1.labels)


def test_subtract():
    """
    Test to ensure that subtraction works
    """

    # create a feature set
    fs1, _ = make_classification_data(num_examples=100,
                                      num_features=4,
                                      num_labels=2,
                                      train_test_ratio=1.0,
                                      random_state=1234)

    # create a different feature set with the same feature names
    # but different feature values
    fs2, _ = make_classification_data(num_examples=100,
                                      num_features=2,
                                      num_labels=2,
                                      train_test_ratio=1.0,
                                      random_state=5678)

    # subtract fs1 from fs2, i.e., the features in fs2
    # should be removed from fs1 but nothing else should change
    fs = fs1 - fs2

    # ensure that the labels are the same in fs and fs1
    assert_array_equal(fs.labels, fs1.labels)

    # ensure that there are only two features left
    eq_(fs.features.shape[1], 2)

    # and that they are f3 and f4
    assert_array_equal(np.array(fs.vectorizer.feature_names_), ['f03', 'f04'])


@raises(ValueError)
def test_mismatch_ids_features():
    """
    Test to catch mistmatch between the shape of the ids vector and the feature matrix
    """

    # get a 100 instances with 4 features each
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=4, n_redundant=0,
                               n_classes=3, random_state=1234567890)

    # convert the features into a list of dictionaries
    feature_names = ['f{}'.format(n) for n in range(1, 5)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # get 200 ids since we don't want to match the number of feature rows
    ids = ['EXAMPLE_{}'.format(i) for i in range(200)]

    # This should raise a ValueError
    FeatureSet('test', ids, features=features, labels=y)


@raises(ValueError)
def test_mismatch_labels_features():
    """
    Test to catch mistmatch between the shape of the labels vector and the feature matrix
    """

    # get a 100 instances with 4 features but ignore the labels we
    # get from here
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=4, n_redundant=0,
                               n_classes=3, random_state=1234567890)

    # double-stack y to ensure we don't match the number of feature rows
    y2 = np.hstack([y, y])

    # convert the features into a list of dictionaries
    feature_names = ['f{}'.format(n) for n in range(1, 5)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # get 100 ids
    ids = ['EXAMPLE_{}'.format(i) for i in range(100)]

    # This should raise a ValueError
    FeatureSet('test', ids, features=features, labels=y2)


@raises(ValueError)
def test_iteration_without_dictvectorizer():
    """
    Test to allow iteration only if the vectorizer is a DictVectorizer
    """

    # create a feature set
    fs, _ = make_classification_data(num_examples=100,
                                     num_features=4,
                                     num_labels=3,
                                     train_test_ratio=1.0,
                                     use_feature_hashing=True,
                                     feature_bins=2)
    # This should raise a ValueError
    for _ in fs:
        pass


def check_filter_ids(inverse=False):

    # create a feature set
    fs, _ = make_classification_data(num_examples=100,
                                     num_features=4,
                                     num_labels=3,
                                     train_test_ratio=1.0)

    # keep just the IDs after Example_50 or do the inverse
    ids_to_filter = ['EXAMPLE_{}'.format(i) for i in range(51, 101)]
    if inverse:
        ids_kept = ['EXAMPLE_{}'.format(i) for i in range(1, 51)]
    else:
        ids_kept = ids_to_filter
    fs.filter(ids=ids_to_filter, inverse=inverse)

    # make sure that we removed the right things
    assert_array_equal(fs.ids, np.array(ids_kept))

    # make sure that number of ids, labels and features are the same
    eq_(fs.ids.shape[0], fs.labels.shape[0])
    eq_(fs.labels.shape[0], fs.features.shape[0])


def test_filter_ids():
    """
    Test filtering with specified IDs, with and without inverting
    """

    yield check_filter_ids
    yield check_filter_ids, True


def check_filter_labels(inverse=False):

    # create a feature set
    fs, _ = make_classification_data(num_examples=1000,
                                     num_features=4,
                                     num_labels=5,
                                     train_test_ratio=1.0)

    # keep just the instaces with 0, 1 and 2 labels
    labels_to_filter = [0, 1, 2]

    # do the actual filtering
    fs.filter(labels=labels_to_filter, inverse=inverse)

    # make sure that we removed the right things
    if inverse:
        ids_kept = fs.ids[np.where(np.logical_not(np.in1d(fs.labels,
                                                          labels_to_filter)))]
    else:
        ids_kept = fs.ids[np.where(np.in1d(fs.labels, labels_to_filter))]

    assert_array_equal(fs.ids, np.array(ids_kept))

    # make sure that number of ids, labels and features are the same
    eq_(fs.ids.shape[0], fs.labels.shape[0])
    eq_(fs.labels.shape[0], fs.features.shape[0])


def test_filter_labels():
    """
    Test filtering with specified labels, with and without inverting
    """

    yield check_filter_labels
    yield check_filter_labels, True


def check_filter_features(inverse=False):

    # create a feature set
    fs, _ = make_classification_data(num_examples=100,
                                     num_features=5,
                                     num_labels=3,
                                     train_test_ratio=1.0)

    # store the features in a separate matrix before filtering
    X = fs.features.todense()

    # filter features f1 and f4 or their inverse
    fs.filter(features=['f01', 'f04'], inverse=inverse)

    # make sure that we have the right number of feature columns
    # depending on whether we are inverting
    feature_shape = (100, 3) if inverse else (100, 2)
    eq_(fs.features.shape, feature_shape)

    # and that they are the first and fourth columns
    # of X that we generated, if not inverting and
    # the second, third and fifth, if inverting
    if inverse:
        feature_columns = X[:, [1, 2, 4]]
    else:
        feature_columns = X[:, [0, 3]]

    assert (fs.features.todense() == feature_columns).all()

    # make sure that the feature names that we kept are also correct
    feature_names = ['f02', 'f03', 'f05'] if inverse else ['f01', 'f04']
    assert_array_equal(np.array(fs.vectorizer.feature_names_),
                       feature_names)

    # make sure that number of ids, labels and features are the same
    eq_(fs.ids.shape[0], fs.labels.shape[0])
    eq_(fs.labels.shape[0], fs.features.shape[0])


def test_filter_features():
    """
    Test filtering with specified features, with and without inverting
    """

    yield check_filter_features
    yield check_filter_features, True


@raises(ValueError)
def test_filter_with_hashing():
    """
    Test to ensure rejection of filtering by features when using hashing
    """

    # create a feature set
    fs, _ = make_classification_data(num_examples=100,
                                     num_features=5,
                                     num_labels=3,
                                     train_test_ratio=1.0,
                                     use_feature_hashing=True,
                                     feature_bins=2)

    # filter features f1 and f4 or their inverse
    fs.filter(features=['f1', 'f4'])


def test_feature_merging_order_invariance():
    """
    Test whether featuresets with different orders of IDs can be merged
    """

    # First, randomly generate two feature sets and then make sure they have
    # the same labels.
    train_fs1, _, _ = make_regression_data()
    train_fs2, _, _ = make_regression_data(start_feature_num=3,
                                           random_state=87654321)
    train_fs2.labels = train_fs1.labels.copy()

    # make a reversed copy of feature set 2
    shuffled_indices = list(range(len(train_fs2.ids)))
    np.random.seed(123456789)
    np.random.shuffle(shuffled_indices)
    train_fs2_ids_shuf = train_fs2.ids[shuffled_indices]
    train_fs2_labels_shuf = train_fs2.labels[shuffled_indices]
    train_fs2_features_shuf = train_fs2.features[shuffled_indices]
    train_fs2_shuf = FeatureSet("f2_shuf",
                                train_fs2_ids_shuf,
                                labels=train_fs2_labels_shuf,
                                features=train_fs2_features_shuf,
                                vectorizer=train_fs2.vectorizer)

    # merge feature set 1 with feature set 2 and its reversed version
    merged_fs = train_fs1 + train_fs2
    merged_fs_shuf = train_fs1 + train_fs2_shuf

    # check that the two merged versions are the same
    feature_names = (train_fs1.vectorizer.get_feature_names()
                     + train_fs2.vectorizer.get_feature_names())
    assert_array_equal(merged_fs.vectorizer.get_feature_names(), feature_names)
    assert_array_equal(merged_fs_shuf.vectorizer.get_feature_names(),
                       feature_names)

    assert_array_equal(merged_fs.labels, train_fs1.labels)
    assert_array_equal(merged_fs.labels, train_fs2.labels)
    assert_array_equal(merged_fs.labels, merged_fs_shuf.labels)

    assert_array_equal(merged_fs.ids, train_fs1.ids)
    assert_array_equal(merged_fs.ids, train_fs2.ids)
    assert_array_equal(merged_fs.ids, merged_fs_shuf.ids)

    assert_array_equal(merged_fs.features[:, 0:2].todense(),
                       train_fs1.features.todense())
    assert_array_equal(merged_fs.features[:, 2:4].todense(),
                       train_fs2.features.todense())
    assert_array_equal(merged_fs.features.todense(),
                       merged_fs_shuf.features.todense())

    assert not np.all(merged_fs.features[:, 0:2].todense()
                      == merged_fs.features[:, 2:4].todense())


# Tests related to loading featuresets and merging them
def make_merging_data(num_feat_files, suffix, numeric_ids):
    num_examples = 500
    num_feats_per_file = 17

    np.random.seed(1234567890)

    merge_dir = join(_my_dir, 'train', 'test_merging')
    if not exists(merge_dir):
        os.makedirs(merge_dir)

    # Create lists we will write files from
    ids = []
    features = []
    labels = []
    for j in range(num_examples):
        y = "dog" if j % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, j) if not numeric_ids else j
        x = {"f{:03d}".format(feat_num): np.random.randint(0, 4) for feat_num
             in range(num_feat_files * num_feats_per_file)}
        x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
        ids.append(ex_id)
        labels.append(y)
        features.append(x)

    # Unmerged
    subset_dict = {}
    for i in range(num_feat_files):
        feat_num = i * num_feats_per_file
        subset_dict['{}'.format(i)] = ["f{:03d}".format(feat_num + j) for j in
                                       range(num_feats_per_file)]
    train_path = join(merge_dir, suffix)
    train_fs = FeatureSet('train', ids, labels=labels, features=features)
    Writer.for_path(train_path, train_fs, subsets=subset_dict).write()

    # Merged
    train_path = join(merge_dir, 'all{}'.format(suffix))
    Writer.for_path(train_path, train_fs).write()


def check_load_featureset(suffix, numeric_ids):
    num_feat_files = 5

    # Create test data
    make_merging_data(num_feat_files, suffix, numeric_ids)

    # Load unmerged data and merge it
    dirpath = join(_my_dir, 'train', 'test_merging')
    featureset = ['{}'.format(i) for i in range(num_feat_files)]
    merged_exs = _load_featureset(dirpath, featureset, suffix, quiet=True)

    # Load pre-merged data
    featureset = ['all']
    premerged_exs = _load_featureset(dirpath, featureset, suffix,
                                     quiet=True)

    assert_array_equal(merged_exs.ids, premerged_exs.ids)
    assert_array_equal(merged_exs.labels, premerged_exs.labels)
    for (_, _, merged_feats), (_, _, premerged_feats) in zip(merged_exs,
                                                             premerged_exs):
        eq_(merged_feats, premerged_feats)
    eq_(sorted(merged_exs.vectorizer.feature_names_),
        sorted(premerged_exs.vectorizer.feature_names_))


def test_load_featureset():
    # Test merging with numeric IDs
    for suffix in ['.jsonlines', '.ndj', '.megam', '.tsv', '.csv', '.arff']:
        yield check_load_featureset, suffix, True

    for suffix in ['.jsonlines', '.ndj', '.megam', '.tsv', '.csv', '.arff']:
        yield check_load_featureset, suffix, False


def test_ids_to_floats():
    path = join(_my_dir, 'train', 'test_input_2examples_1.jsonlines')

    examples = Reader.for_path(path, ids_to_floats=True, quiet=True).read()
    assert isinstance(examples.ids[0], float)

    examples = Reader.for_path(path, quiet=True).read()
    assert not isinstance(examples.ids[0], float)
    assert isinstance(examples.ids[0], str)


def test_dict_list_reader():
    examples = [{"id": "example0", "y": 1.0, "x": {"f1": 1.0}},
                {"id": "example1", "y": 2.0, "x": {"f1": 1.0, "f2": 1.0}},
                {"id": "example2", "y": 3.0, "x": {"f2": 1.0, "f3": 3.0}}]
    converted = DictListReader(examples).read()

    eq_(converted.ids[0], "example0")
    eq_(converted.ids[1], "example1")
    eq_(converted.ids[2], "example2")

    eq_(converted.labels[0], 1.0)
    eq_(converted.labels[1], 2.0)
    eq_(converted.labels[2], 3.0)

    eq_(converted.features[0, 0], 1.0)
    eq_(converted.features[0, 1], 0.0)
    eq_(converted.features[1, 0], 1.0)
    eq_(converted.features[1, 1], 1.0)
    eq_(converted.features[2, 2], 3.0)
    eq_(converted.features[2, 0], 0.0)

    eq_(converted.vectorizer.get_feature_names(), ['f1', 'f2', 'f3'])


# Tests related to converting featuresets
def make_conversion_data(num_feat_files, from_suffix, to_suffix):
    num_examples = 500
    num_feats_per_file = 7

    np.random.seed(1234567890)

    convert_dir = join(_my_dir, 'train', 'test_conversion')
    if not exists(convert_dir):
        os.makedirs(convert_dir)

    # Create lists we will write files from
    ids = []
    features = []
    labels = []
    for j in range(num_examples):
        y = "dog" if j % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, j)
        x = {"f{:03d}".format(feat_num): np.random.randint(0, 4) for feat_num
             in range(num_feat_files * num_feats_per_file)}
        x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
        ids.append(ex_id)
        labels.append(y)
        features.append(x)
    # Create vectorizers/maps for libsvm subset writing
    feat_vectorizer = DictVectorizer()
    feat_vectorizer.fit(features)
    label_map = {label: num for num, label in
                 enumerate(sorted({label for label in labels if
                                   not isinstance(label, (int, float))}))}
    # Add fake item to vectorizer for None
    label_map[None] = '00000'

    # get the feature name prefix
    feature_name_prefix = '{}_to_{}'.format(from_suffix.lstrip('.'),
                                            to_suffix.lstrip('.'))

    # Write out unmerged features in the `from_suffix` file format
    for i in range(num_feat_files):
        train_path = join(convert_dir, '{}_{}{}'.format(feature_name_prefix,
                                                        i, from_suffix))
        sub_features = []
        for example_num in range(num_examples):
            feat_num = i * num_feats_per_file
            x = {"f{:03d}".format(feat_num + j):
                 features[example_num]["f{:03d}".format(feat_num + j)] for j in
                 range(num_feats_per_file)}
            sub_features.append(x)
        train_fs = FeatureSet('sub_train', ids, labels=labels,
                              features=sub_features,
                              vectorizer=feat_vectorizer)
        if from_suffix == '.libsvm':
            Writer.for_path(train_path, train_fs,
                            label_map=label_map).write()
        else:
            Writer.for_path(train_path, train_fs).write()

    # Write out the merged features in the `to_suffix` file format
    train_path = join(convert_dir, '{}_all{}'.format(feature_name_prefix,
                                                     to_suffix))
    train_fs = FeatureSet('train', ids, labels=labels, features=features,
                          vectorizer=feat_vectorizer)
    if to_suffix == '.libsvm':
        Writer.for_path(train_path, train_fs,
                        label_map=label_map).write()
    else:
        Writer.for_path(train_path, train_fs).write()


def check_convert_featureset(from_suffix, to_suffix):
    num_feat_files = 5

    # Create test data
    make_conversion_data(num_feat_files, from_suffix, to_suffix)

    # the path to the unmerged feature files
    dirpath = join(_my_dir, 'train', 'test_conversion')

    # get the feature name prefix
    feature_name_prefix = '{}_to_{}'.format(from_suffix.lstrip('.'),
                                            to_suffix.lstrip('.'))

    # Load each unmerged feature file in the `from_suffix` format and convert
    # it to the `to_suffix` format
    for feature in range(num_feat_files):
        input_file_path = join(dirpath, '{}_{}{}'.format(feature_name_prefix,
                                                         feature,
                                                         from_suffix))
        output_file_path = join(dirpath, '{}_{}{}'.format(feature_name_prefix,
                                                          feature, to_suffix))
        skll_convert.main(['--quiet', input_file_path, output_file_path])

    # now load and merge all unmerged, converted features in the `to_suffix`
    # format
    featureset = ['{}_{}'.format(feature_name_prefix, i) for i in
                  range(num_feat_files)]
    merged_exs = _load_featureset(dirpath, featureset, to_suffix,
                                  quiet=True)

    # Load pre-merged data in the `to_suffix` format
    featureset = ['{}_all'.format(feature_name_prefix)]
    premerged_exs = _load_featureset(dirpath, featureset, to_suffix,
                                     quiet=True)

    # make sure that the pre-generated merged data in the to_suffix format
    # is the same as the converted, merged data in the to_suffix format
    assert_array_equal(merged_exs.ids, premerged_exs.ids)
    assert_array_equal(merged_exs.labels, premerged_exs.labels)
    for (_, _, merged_feats), (_, _, premerged_feats) in zip(merged_exs,
                                                             premerged_exs):
        eq_(merged_feats, premerged_feats)
    eq_(sorted(merged_exs.vectorizer.feature_names_),
        sorted(premerged_exs.vectorizer.feature_names_))


def test_convert_featureset():
    # Test the conversion from every format to every other format
    for from_suffix, to_suffix in itertools.permutations(['.jsonlines', '.ndj',
                                                          '.megam', '.tsv',
                                                          '.csv', '.arff',
                                                          '.libsvm'], 2):
        yield check_convert_featureset, from_suffix, to_suffix


def featureset_creation_from_dataframe_helper(with_labels, use_feature_hasher):
    """
    Helper function for the two unit tests for FeatureSet.from_data_frame().
    Since labels are optional, run two tests, one with, one without.
    """
    import pandas

    # First, setup the test data.
    # get a 100 instances with 4 features each
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=4, n_redundant=0,
                               n_classes=3, random_state=1234567890)

    # Not using 0 - 100 here because that would be pandas' default index names anyway.
    # So let's make sure pandas is using the ids we supply.
    ids = list(range(100, 200))

    featureset_name = 'test'

    # if use_feature_hashing, run these tests with a vectorizer
    feature_bins = 4
    vectorizer = (FeatureHasher(n_features=feature_bins)
                  if use_feature_hasher else None)
    
    # convert the features into a list of dictionaries
    feature_names = ['f{}'.format(n) for n in range(1, 5)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # Now, create a FeatureSet object.
    if with_labels:
        expected = FeatureSet(featureset_name, ids, features=features, labels=y,
                              vectorizer=vectorizer)
    else:
        expected = FeatureSet(featureset_name, ids, features=features,
                              vectorizer=vectorizer)

    # Also create a DataFrame and then create a FeatureSet from it.
    df = pandas.DataFrame(features, index=ids)
    if with_labels:
        df['y'] = y
        current = FeatureSet.from_data_frame(df, featureset_name, labels_column='y',
                                             vectorizer=vectorizer)
    else:
        current = FeatureSet.from_data_frame(df, featureset_name, vectorizer=vectorizer)

    return (expected, current)


@attr('have_pandas')
def test_featureset_creation_from_dataframe_with_labels():
    (expected, current) = featureset_creation_from_dataframe_helper(True, False)
    assert expected == current


@attr('have_pandas')
def test_featureset_creation_from_dataframe_without_labels():
    (expected, current) = featureset_creation_from_dataframe_helper(False, False)
    # Directly comparing FeatureSet objects fails here because both sets
    # of labels are all nan when labels isn't specified, and arrays of
    # all nan are not equal to each other.
    # Based off of FeatureSet.__eq__()
    assert (expected.name == current.name and
            (expected.ids == current.ids).all() and
            expected.vectorizer == current.vectorizer and
            np.allclose(expected.features.data,
                        current.features.data,
                        rtol=1e-6) and
            np.all(np.isnan(expected.labels)) and
            np.all(np.isnan(current.labels)))


@attr('have_pandas')
def test_featureset_creation_from_dataframe_with_labels_and_vectorizer():
    (expected, current) = featureset_creation_from_dataframe_helper(True, True)
    assert expected == current


@attr('have_pandas')
def test_featureset_creation_from_dataframe_without_labels_with_vectorizer():
    (expected, current) = featureset_creation_from_dataframe_helper(False, True)
    # Directly comparing FeatureSet objects fails here because both sets
    # of labels are all nan when labels isn't specified, and arrays of
    # all nan are not equal to each other.
    # Based off of FeatureSet.__eq__()
    assert (expected.name == current.name and
            (expected.ids == current.ids).all() and
            expected.vectorizer == current.vectorizer and
            np.allclose(expected.features.data,
                        current.features.data,
                        rtol=1e-6) and
            np.all(np.isnan(expected.labels)) and
            np.all(np.isnan(current.labels)))
