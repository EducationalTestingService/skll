# License: BSD 3 clause
"""
Run tests related to FeatureSets.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

import itertools
import unittest
from collections import OrderedDict
from shutil import rmtree

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

import skll
from skll.data import (
    CSVReader,
    FeatureSet,
    NDJReader,
    NDJWriter,
    Reader,
    TSVReader,
    Writer,
)
from skll.data.readers import DictListReader
from skll.experiments import load_featureset
from skll.utils.commandline import skll_convert
from tests import other_dir, output_dir, test_dir, train_dir
from tests.utils import make_classification_data, make_regression_data, unlink


class TestFeatureset(unittest.TestCase):
    """Test class for featureset tests."""

    @classmethod
    def setUpClass(cls):
        """Create necessary directories for testing."""
        for dir_path in [train_dir, test_dir, output_dir]:
            dir_path.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up files created during testing."""
        for filetype in ["csv", "jsonlines", "libsvm", "tsv"]:
            unlink(other_dir / f"empty.{filetype}")

        file_names = [
            f"{x}.jsonlines"
            for x in ["test_string_ids", "test_string_ids_df", "test_string_labels_df"]
        ] + [
            "test_read_csv_tsv_drop_blanks.csv",
            "test_read_csv_tsv_drop_blanks.tsv",
            "test_read_csv_tsv_fill_blanks.csv",
            "test_read_csv_tsv_fill_blanks.tsv",
            "test_read_csv_tsv_fill_blanks_dict.csv",
            "test_read_csv_tsv_fill_blanks_dict.tsv",
            "test_drop_blanks_error.csv",
        ]

        for file_name in file_names:
            unlink(other_dir / file_name)

        for dir_name in ["test_conversion", "test_merging"]:
            path = train_dir / dir_name
            if path.exists():
                rmtree(train_dir / dir_name)

    def _create_empty_file(self, filetype):
        filepath = other_dir / f"empty.{filetype}"
        open(filepath, "w").close()
        return filepath

    def _create_test_file(self, filepath, contents):
        with open(filepath, "w") as filefh:
            filefh.write(contents)

    def test_empty_ids(self):
        """Test to ensure that an error is raised if ids is None."""
        # get a 100 instances with 4 features each
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            random_state=1234567890,
        )

        # convert the features into a list of dictionaries
        feature_names = [f"f{n}" for n in range(1, 5)]
        features = []
        for row in X:
            features.append(dict(zip(feature_names, row)))

        # create a feature set with ids set to None and raise ValueError
        with self.assertRaises(ValueError):
            FeatureSet("test", None, features=features, labels=y)

    def check_empty_file_read(self, filetype, reader_type):
        empty_filepath = self._create_empty_file(filetype)
        reader = getattr(skll.data, reader_type).for_path(empty_filepath)
        with self.assertRaises(ValueError):
            _ = reader.read()

    def test_empty_file_read(self):
        for filetype, reader_type in zip(
            ["csv", "jsonlines", "libsvm", "tsv"],
            ["CSVReader", "NDJReader", "LibSVMReader", "TSVReader"],
        ):
            yield self.check_empty_file_read, filetype, reader_type

    def test_empty_labels(self):
        """Test to check behaviour when labels is None."""
        # create a feature set with empty labels
        fs, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, empty_labels=True, train_test_ratio=1.0
        )
        assert np.isnan(fs.labels).all()

    def test_length(self):
        """Test to whether len() returns the number of instances."""
        # create a featureset
        fs, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        self.assertEqual(len(fs), 100)

    def test_string_feature(self):
        """Test that string-valued features are properly encoded as binary features."""
        # create a featureset that is derived from an original
        # set of features containing 3 numeric features and
        # one string-valued feature that can take six possible
        # values between 'a' to 'f'. This means that the
        # featureset will have 3 numeric + 6 binary features.
        fs, _ = make_classification_data(
            num_examples=100,
            num_features=4,
            num_labels=3,
            one_string_feature=True,
            num_string_values=6,
            train_test_ratio=1.0,
        )

        # confirm that the number of features are as expected
        self.assertEqual(fs.features.shape, (100, 9))

        # confirm the feature names
        self.assertEqual(
            fs.vectorizer.feature_names_,
            ["f01", "f02", "f03", "f04=a", "f04=b", "f04=c", "f04=d", "f04=e", "f04=f"],
        )

        # confirm that the final six features are binary
        assert_array_equal(fs.features[:, [3, 4, 5, 6, 7, 8]].data, 1)

    def test_equality(self):
        """Test featureset equality."""
        # create a featureset
        fs1, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        # create a featureset with a different set but same number
        # of features and everything else the same
        fs2, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        fs2.features *= 2

        # create a featureset with different feature names
        # and everything else the same
        fs3, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, feature_prefix="g", train_test_ratio=1.0
        )

        # create a featureset with a different set of labels
        # and everything else the same
        fs4, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=2, train_test_ratio=1.0
        )

        # create a featureset with a different set but same number
        # of IDs and everything else the same
        fs5, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )
        fs5.ids = np.array(["A" + i for i in fs2.ids])

        # create a featureset with a different vectorizer
        # and everything else the same
        fs6, _ = make_classification_data(
            num_examples=100,
            num_features=4,
            num_labels=3,
            train_test_ratio=1.0,
            use_feature_hashing=True,
            feature_bins=2,
        )

        # create a featureset with a different number of features
        # and everything else the same
        fs7, _ = make_classification_data(
            num_examples=100, num_features=5, num_labels=3, train_test_ratio=1.0
        )

        # create a featureset with a different number of examples
        # and everything else the same
        fs8, _ = make_classification_data(
            num_examples=200, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        # create a featureset with a different vectorizer instance
        # and everything else the same
        fs9, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        # now check for the expected equalities
        self.assertNotEqual(fs1, fs2)
        self.assertNotEqual(fs1, fs3)
        self.assertNotEqual(fs1, fs4)
        self.assertNotEqual(fs1, fs5)
        self.assertNotEqual(fs1, fs6)
        self.assertNotEqual(fs1, fs7)
        self.assertNotEqual(fs1, fs8)
        self.assertNotEqual(id(fs1.vectorizer), id(fs9.vectorizer))
        self.assertEqual(fs1, fs9)

    def test_vectorizer_inequality(self):
        """Test to make sure that vectorizer equality fails properly."""
        v = DictVectorizer()
        self.assertNotEqual(v, 1)
        self.assertNotEqual(v, "passthrough")
        self.assertNotEqual(v, [1.0, 2.0, 3.0])

    def test_merge_no_vectorizers(self):
        """Test to ensure rejection of merging featuresets with no labels."""
        # create a featureset each with a DictVectorizer
        fs1, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        # create another featureset using hashing
        fs2, _ = make_classification_data(
            num_examples=100,
            num_features=4,
            feature_prefix="g",
            num_labels=3,
            train_test_ratio=1.0,
            use_feature_hashing=True,
        )
        fs2.vectorizer = None

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            fs1 + fs2

    def test_merge_no_features(self):
        """Test to ensure rejection of merging featuresets with no labels."""
        # create a featureset each with a DictVectorizer
        fs1, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )
        fs1.features = None

        # create another featureset using hashing
        fs2, _ = make_classification_data(
            num_examples=100,
            num_features=4,
            feature_prefix="g",
            num_labels=3,
            train_test_ratio=1.0,
            use_feature_hashing=True,
        )
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            fs1 + fs2

    def test_merge_different_vectorizers(self):
        """Test to ensure rejection of merging featuresets with different vectorizers."""
        # create a featureset each with a DictVectorizer
        fs1, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        # create another featureset using hashing
        fs2, _ = make_classification_data(
            num_examples=100,
            num_features=4,
            feature_prefix="g",
            num_labels=3,
            train_test_ratio=1.0,
            use_feature_hashing=True,
        )
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            fs1 + fs2

    def test_merge_different_hashers(self):
        """Test to ensure rejection of merging featuresets with different FeatureHashers."""
        # create a feature set with 4 feature hashing bins
        fs1, _ = make_classification_data(
            num_examples=100,
            num_features=10,
            num_labels=3,
            train_test_ratio=1.0,
            use_feature_hashing=True,
            feature_bins=4,
        )

        # create a second feature set with 3 feature hashing bins
        fs2, _ = make_classification_data(
            num_examples=100,
            num_features=10,
            num_labels=3,
            feature_prefix="g",
            train_test_ratio=1.0,
            use_feature_hashing=True,
            feature_bins=3,
        )
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            fs1 + fs2

    def test_merge_different_labels_same_ids(self):
        """Test to ensure rejection of merging featuresets that have conflicting labels."""
        # create a feature set
        fs1, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        # create a different feature set that has everything
        # the same but has different labels for the same IDs
        fs2, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, feature_prefix="g", train_test_ratio=1.0
        )

        # artificially modify the class labels
        fs2.labels = fs2.labels + 1

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            fs1 + fs2

    def test_merge_missing_labels(self):
        """Test to ensure that labels are sucessfully copied when merging."""
        # create a feature set
        fs1, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        # create a different feature set with no labels specified
        fs2, _ = make_classification_data(
            num_examples=100,
            num_features=4,
            feature_prefix="g",
            empty_labels=True,
            num_labels=3,
            train_test_ratio=1.0,
        )

        # merge the two featuresets in different orders
        fs12 = fs1 + fs2
        fs21 = fs2 + fs1

        # make sure that the labels are the same after merging
        assert_array_equal(fs12.labels, fs1.labels)
        assert_array_equal(fs21.labels, fs1.labels)

    def test_write_hashed_featureset(self):
        """Test to check that hashed featuresets cannot be written out."""
        fs, _ = make_classification_data(
            num_examples=100,
            num_features=4,
            use_feature_hashing=True,
            feature_bins=2,
            random_state=1234,
        )
        writer = NDJWriter(output_dir / "foo.jsonlines", fs)
        with self.assertRaises(ValueError):
            writer.write()

    def test_subtract(self):
        """Test to ensure that subtraction works."""
        # create a feature set
        fs1, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=2, train_test_ratio=1.0, random_state=1234
        )

        # create a different feature set with the same feature names
        # but different feature values
        fs2, _ = make_classification_data(
            num_examples=100, num_features=2, num_labels=2, train_test_ratio=1.0, random_state=5678
        )

        # subtract fs1 from fs2, i.e., the features in fs2
        # should be removed from fs1 but nothing else should change
        fs = fs1 - fs2

        # ensure that the labels are the same in fs and fs1
        assert_array_equal(fs.labels, fs1.labels)

        # ensure that there are only two features left
        self.assertEqual(fs.features.shape[1], 2)

        # and that they are f3 and f4
        assert_array_equal(np.array(fs.vectorizer.feature_names_), ["f03", "f04"])

    def test_mismatch_ids_features(self):
        """Test to catch mistmatch between the shape of the ids vector and the feature matrix."""
        # get a 100 instances with 4 features each
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            random_state=1234567890,
        )

        # convert the features into a list of dictionaries
        feature_names = [f"f{n}" for n in range(1, 5)]
        features = []
        for row in X:
            features.append(dict(zip(feature_names, row)))

        # get 200 ids since we don't want to match the number of feature rows
        ids = [f"EXAMPLE_{i}" for i in range(200)]

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            FeatureSet("test", ids, features=features, labels=y)

    def test_mismatch_labels_features(self):
        """Test to catch mistmatch between the shape of the labels vector and the feature matrix."""
        # get a 100 instances with 4 features but ignore the labels we
        # get from here
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            random_state=1234567890,
        )

        # double-stack y to ensure we don't match the number of feature rows
        y2 = np.hstack([y, y])

        # convert the features into a list of dictionaries
        feature_names = [f"f{n}" for n in range(1, 5)]
        features = []
        for row in X:
            features.append(dict(zip(feature_names, row)))

        # get 100 ids
        ids = [f"EXAMPLE_{i}" for i in range(100)]

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            FeatureSet("test", ids, features=features, labels=y2)

    def test_iteration_without_dictvectorizer(self):
        """Test to allow iteration only if the vectorizer is a DictVectorizer."""
        # create a feature set
        fs, _ = make_classification_data(
            num_examples=100,
            num_features=4,
            num_labels=3,
            train_test_ratio=1.0,
            use_feature_hashing=True,
            feature_bins=2,
        )
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            for _ in fs:
                pass

    def check_filter_ids(self, inverse):
        # create a feature set
        fs, _ = make_classification_data(
            num_examples=100, num_features=4, num_labels=3, train_test_ratio=1.0
        )

        # keep just the IDs after Example_50 or do the inverse
        ids_to_filter = [f"EXAMPLE_{i}" for i in range(51, 101)]
        if inverse:
            ids_kept = [f"EXAMPLE_{i}" for i in range(1, 51)]
        else:
            ids_kept = ids_to_filter
        fs.filter(ids=ids_to_filter, inverse=inverse)

        # make sure that we removed the right things
        assert_array_equal(fs.ids, np.array(ids_kept))

        # make sure that number of ids, labels and features are the same
        self.assertEqual(fs.ids.shape[0], fs.labels.shape[0])
        self.assertEqual(fs.labels.shape[0], fs.features.shape[0])

    def test_filter_ids(self):
        """Test filtering with specified IDs, with and without inverting."""
        yield self.check_filter_ids, False
        yield self.check_filter_ids, True

    def check_filter_labels(self, inverse):
        # create a feature set
        fs, _ = make_classification_data(
            num_examples=1000, num_features=4, num_labels=5, train_test_ratio=1.0
        )

        # keep just the instaces with 0, 1 and 2 labels
        labels_to_filter = [0, 1, 2]

        # do the actual filtering
        fs.filter(labels=labels_to_filter, inverse=inverse)

        # make sure that we removed the right things
        if inverse:
            ids_kept = fs.ids[np.where(np.logical_not(np.in1d(fs.labels, labels_to_filter)))]
        else:
            ids_kept = fs.ids[np.where(np.in1d(fs.labels, labels_to_filter))]

        assert_array_equal(fs.ids, np.array(ids_kept))

        # make sure that number of ids, labels and features are the same
        self.assertEqual(fs.ids.shape[0], fs.labels.shape[0])
        self.assertEqual(fs.labels.shape[0], fs.features.shape[0])

    def test_filter_labels(self):
        """Test filtering with specified labels, with and without inverting."""
        yield self.check_filter_labels, False
        yield self.check_filter_labels, True

    def check_filter_features(self, inverse):
        # create a feature set
        fs, _ = make_classification_data(
            num_examples=100, num_features=5, num_labels=3, train_test_ratio=1.0
        )

        # store the features in a separate matrix before filtering
        X = fs.features.toarray()

        # filter features f1 and f4 or their inverse
        fs.filter(features=["f01", "f04"], inverse=inverse)

        # make sure that we have the right number of feature columns
        # depending on whether we are inverting
        feature_shape = (100, 3) if inverse else (100, 2)
        self.assertEqual(fs.features.shape, feature_shape)

        # and that they are the first and fourth columns
        # of X that we generated, if not inverting and
        # the second, third and fifth, if inverting
        if inverse:
            feature_columns = X[:, [1, 2, 4]]
        else:
            feature_columns = X[:, [0, 3]]

        assert (fs.features.toarray() == feature_columns).all()

        # make sure that the feature names that we kept are also correct
        feature_names = ["f02", "f03", "f05"] if inverse else ["f01", "f04"]
        assert_array_equal(np.array(fs.vectorizer.feature_names_), feature_names)

        # make sure that number of ids, labels and features are the same
        self.assertEqual(fs.ids.shape[0], fs.labels.shape[0])
        self.assertEqual(fs.labels.shape[0], fs.features.shape[0])

    def test_filter_features(self):
        """Test filtering with specified features, with and without inverting."""
        yield self.check_filter_features, False
        yield self.check_filter_features, True

    def test_filter_with_hashing(self):
        """Test to ensure rejection of filtering by features when using hashing."""
        # create a feature set
        fs, _ = make_classification_data(
            num_examples=100,
            num_features=5,
            num_labels=3,
            train_test_ratio=1.0,
            use_feature_hashing=True,
            feature_bins=2,
        )

        # filter features f1 and f4 or their inverse
        with self.assertRaises(ValueError):
            fs.filter(features=["f1", "f4"])

    def test_feature_merging_order_invariance(self):
        """Test whether featuresets with different orders of IDs can be merged."""
        # First, randomly generate two feature sets and then make sure they have
        # the same labels.
        train_fs1, _, _ = make_regression_data()
        train_fs2, _, _ = make_regression_data(start_feature_num=3, random_state=87654321)
        train_fs2.labels = train_fs1.labels.copy()

        # make a reversed copy of feature set 2
        shuffled_indices = list(range(len(train_fs2.ids)))
        np.random.seed(123456789)
        np.random.shuffle(shuffled_indices)
        train_fs2_ids_shuf = train_fs2.ids[shuffled_indices]
        train_fs2_labels_shuf = train_fs2.labels[shuffled_indices]
        train_fs2_features_shuf = train_fs2.features[shuffled_indices]
        train_fs2_shuf = FeatureSet(
            "f2_shuf",
            train_fs2_ids_shuf,
            labels=train_fs2_labels_shuf,
            features=train_fs2_features_shuf,
            vectorizer=train_fs2.vectorizer,
        )

        # merge feature set 1 with feature set 2 and its reversed version
        merged_fs = train_fs1 + train_fs2
        merged_fs_shuf = train_fs1 + train_fs2_shuf

        # check that the two merged versions are the same
        feature_names = (
            train_fs1.vectorizer.get_feature_names_out().tolist()
            + train_fs2.vectorizer.get_feature_names_out().tolist()
        )
        assert_array_equal(merged_fs.vectorizer.get_feature_names_out().tolist(), feature_names)
        assert_array_equal(
            merged_fs_shuf.vectorizer.get_feature_names_out().tolist(), feature_names
        )

        assert_array_equal(merged_fs.labels, train_fs1.labels)
        assert_array_equal(merged_fs.labels, train_fs2.labels)
        assert_array_equal(merged_fs.labels, merged_fs_shuf.labels)

        assert_array_equal(merged_fs.ids, train_fs1.ids)
        assert_array_equal(merged_fs.ids, train_fs2.ids)
        assert_array_equal(merged_fs.ids, merged_fs_shuf.ids)

        assert_array_equal(merged_fs.features[:, 0:2].toarray(), train_fs1.features.toarray())
        assert_array_equal(merged_fs.features[:, 2:4].toarray(), train_fs2.features.toarray())
        assert_array_equal(merged_fs.features.toarray(), merged_fs_shuf.features.toarray())

        assert not np.all(
            merged_fs.features[:, 0:2].toarray() == merged_fs.features[:, 2:4].toarray()
        )

    # Tests related to loading featuresets and merging them
    def make_merging_data(self, num_feat_files, suffix, numeric_ids):
        num_examples = 500
        num_feats_per_file = 17

        np.random.seed(1234567890)

        merge_dir = train_dir / "test_merging"
        if not merge_dir.exists():
            merge_dir.mkdir(parents=True)

        # Create lists we will write files from
        ids = []
        features = []
        labels = []
        for j in range(num_examples):
            y = "dog" if j % 2 == 0 else "cat"
            ex_id = f"{y}{j}" if not numeric_ids else j
            x = {
                f"f{feat_num:03d}": np.random.randint(0, 4)
                for feat_num in range(num_feat_files * num_feats_per_file)
            }
            x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
            ids.append(ex_id)
            labels.append(y)
            features.append(x)

        # Unmerged
        subset_dict = {}
        for i in range(num_feat_files):
            feat_num = i * num_feats_per_file
            subset_dict[str(i)] = [f"f{feat_num + j:03d}" for j in range(num_feats_per_file)]
        train_path = merge_dir / suffix
        train_fs = FeatureSet("train", ids, labels=labels, features=features)
        Writer.for_path(train_path, train_fs, subsets=subset_dict).write()

        # Merged
        train_path = merge_dir / f"all{suffix}"
        Writer.for_path(train_path, train_fs).write()

    def check_load_featureset(self, suffix, numeric_ids):
        num_feat_files = 5

        # Create test data
        self.make_merging_data(num_feat_files, suffix, numeric_ids)

        # Load unmerged data and merge it
        dirpath = train_dir / "test_merging"
        featureset = [str(i) for i in range(num_feat_files)]
        merged_exs = load_featureset(dirpath, featureset, suffix, quiet=True)

        # Load pre-merged data
        featureset = ["all"]
        premerged_exs = load_featureset(dirpath, featureset, suffix, quiet=True)

        assert_array_equal(merged_exs.ids, premerged_exs.ids)
        assert_array_equal(merged_exs.labels, premerged_exs.labels)
        for (_, _, merged_feats), (_, _, premerged_feats) in zip(merged_exs, premerged_exs):
            self.assertEqual(merged_feats, premerged_feats)
        self.assertEqual(
            sorted(merged_exs.vectorizer.feature_names_),
            sorted(premerged_exs.vectorizer.feature_names_),
        )

    def test_load_featureset(self):
        # Test merging with numeric IDs
        for suffix in [".jsonlines", ".ndj", ".tsv", ".csv", ".arff"]:
            yield self.check_load_featureset, suffix, True

        for suffix in [".jsonlines", ".ndj", ".tsv", ".csv", ".arff"]:
            yield self.check_load_featureset, suffix, False

    def test_ids_to_floats(self):
        path = train_dir / "test_input_2examples_1.jsonlines"

        examples = Reader.for_path(path, ids_to_floats=True, quiet=True).read()
        assert isinstance(examples.ids[0], float)

        examples = Reader.for_path(path, quiet=True).read()
        assert not isinstance(examples.ids[0], float)
        assert isinstance(examples.ids[0], str)

    def test_dict_list_reader(self):
        examples = [
            {"id": "example0", "y": 1.0, "x": {"f1": 1.0}},
            {"id": "example1", "y": 2.0, "x": {"f1": 1.0, "f2": 1.0}},
            {"id": "example2", "y": 3.0, "x": {"f2": 1.0, "f3": 3.0}},
        ]
        converted = DictListReader(examples).read()

        self.assertEqual(converted.ids[0], "example0")
        self.assertEqual(converted.ids[1], "example1")
        self.assertEqual(converted.ids[2], "example2")

        self.assertEqual(converted.labels[0], 1.0)
        self.assertEqual(converted.labels[1], 2.0)
        self.assertEqual(converted.labels[2], 3.0)

        self.assertEqual(converted.features[0, 0], 1.0)
        self.assertEqual(converted.features[0, 1], 0.0)
        self.assertEqual(converted.features[1, 0], 1.0)
        self.assertEqual(converted.features[1, 1], 1.0)
        self.assertEqual(converted.features[2, 2], 3.0)
        self.assertEqual(converted.features[2, 0], 0.0)

        self.assertEqual(converted.vectorizer.get_feature_names_out().tolist(), ["f1", "f2", "f3"])

    # Tests related to converting featuresets
    def make_conversion_data(self, num_feat_files, from_suffix, to_suffix, with_labels=True):
        num_examples = 500
        num_feats_per_file = 7

        np.random.seed(1234567890)

        convert_dir = train_dir / "test_conversion"
        if not convert_dir.exists():
            convert_dir.mkdir(parents=True)

        # Create lists we will write files from
        ids = []
        features = []
        labels = [] if with_labels else None
        for j in range(num_examples):
            y = "dog" if j % 2 == 0 else "cat"
            ex_id = f"{y}{j}"
            x = {
                f"f{feat_num:03d}": np.random.randint(4)
                for feat_num in range(num_feat_files * num_feats_per_file)
            }
            x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
            ids.append(ex_id)
            if with_labels:
                labels.append(y)
            features.append(x)

        # Create vectorizers/maps for libsvm subset writing
        feat_vectorizer = DictVectorizer()
        feat_vectorizer.fit(features)
        if with_labels:
            label_map = {
                label: num
                for num, label in enumerate(
                    sorted({label for label in labels if not isinstance(label, (int, float))})
                )
            }
            # Add fake item to vectorizer for None
            label_map[None] = "00000"
        else:
            label_map = None

        # get the feature name prefix
        feature_name_prefix = f"{from_suffix.lstrip('.')}_to_{to_suffix.lstrip('.')}"

        # use '_unlabeled' as part of any file names when not using labels
        with_labels_part = "" if with_labels else "_unlabeled"

        # Write out unmerged features in the `from_suffix` file format
        for i in range(num_feat_files):
            train_path = convert_dir / f"{feature_name_prefix}_{i}{with_labels_part}{from_suffix}"
            sub_features = []
            for example_num in range(num_examples):
                feat_num = i * num_feats_per_file
                x = {
                    f"f{feat_num + j:03d}": features[example_num][f"f{feat_num + j:03d}"]
                    for j in range(num_feats_per_file)
                }
                sub_features.append(x)
            train_fs = FeatureSet(
                "sub_train", ids, labels=labels, features=sub_features, vectorizer=feat_vectorizer
            )
            if from_suffix == ".libsvm":
                Writer.for_path(train_path, train_fs, label_map=label_map).write()
            elif from_suffix in [".arff", ".csv", ".tsv"]:
                label_col = "y" if with_labels else None
                Writer.for_path(train_path, train_fs, label_col=label_col).write()
            else:
                Writer.for_path(train_path, train_fs).write()

        # Write out the merged features in the `to_suffix` file format
        train_path = convert_dir / f"{feature_name_prefix}{with_labels_part}_all{to_suffix}"
        train_fs = FeatureSet(
            "train", ids, labels=labels, features=features, vectorizer=feat_vectorizer
        )

        # we need to do this to get around the FeatureSet using NaNs
        # instead of None when there are no labels which causes problems
        # later when comparing featuresets
        if not with_labels:
            train_fs.labels = [None] * len(train_fs.labels)

        if to_suffix == ".libsvm":
            Writer.for_path(train_path, train_fs, label_map=label_map).write()
        elif to_suffix in [".arff", ".csv", ".tsv"]:
            label_col = "y" if with_labels else None
            Writer.for_path(train_path, train_fs, label_col=label_col).write()
        else:
            Writer.for_path(train_path, train_fs).write()

    def check_convert_featureset(self, from_suffix, to_suffix, with_labels=True):
        num_feat_files = 5

        # Create test data
        self.make_conversion_data(num_feat_files, from_suffix, to_suffix, with_labels=with_labels)

        # the path to the unmerged feature files
        dirpath = train_dir / "test_conversion"

        # get the feature name prefix
        feature_name_prefix = f"{from_suffix.lstrip('.')}_to_{to_suffix.lstrip('.')}"

        # use '_unlabeled' as part of any file names when not using labels
        with_labels_part = "" if with_labels else "_unlabeled"

        # Load each unmerged feature file in the `from_suffix` format and convert
        # it to the `to_suffix` format
        for feature in range(num_feat_files):
            input_file_path = (
                dirpath / f"{feature_name_prefix}_{feature}{with_labels_part}{from_suffix}"
            )
            output_file_path = (
                dirpath / f"{feature_name_prefix}_{feature}{with_labels_part}{to_suffix}"
            )
            skll_convert_args = ["--quiet", str(input_file_path), str(output_file_path)]
            if not with_labels:
                skll_convert_args.append("--no_labels")
            skll_convert.main(skll_convert_args)

        # now load and merge all unmerged, converted features in the `to_suffix`
        # format
        featureset = [f"{feature_name_prefix}_{i}{with_labels_part}" for i in range(num_feat_files)]
        label_col = "y" if with_labels else None
        merged_exs = load_featureset(
            dirpath, featureset, to_suffix, label_col=label_col, quiet=True
        )

        # Load pre-merged data in the `to_suffix` format
        featureset = [f"{feature_name_prefix}{with_labels_part}_all"]
        premerged_exs = load_featureset(
            dirpath, featureset, to_suffix, label_col=label_col, quiet=True
        )

        # make sure that the pre-generated merged data in the to_suffix format
        # is the same as the converted, merged data in the to_suffix format

        # first check the IDs
        assert_array_equal(merged_exs.ids, premerged_exs.ids)
        assert_array_equal(merged_exs.labels, premerged_exs.labels)
        for (_, _, merged_feats), (_, _, premerged_feats) in zip(merged_exs, premerged_exs):
            self.assertEqual(merged_feats, premerged_feats)
        self.assertEqual(
            sorted(merged_exs.vectorizer.feature_names_),
            sorted(premerged_exs.vectorizer.feature_names_),
        )

    def test_convert_featureset(self):
        # Test the conversion from every format to every other format
        # with and without labels
        for from_suffix, to_suffix in itertools.permutations(
            [".jsonlines", ".tsv", ".csv", ".arff", ".libsvm"], 2
        ):
            yield self.check_convert_featureset, from_suffix, to_suffix, True
            yield self.check_convert_featureset, from_suffix, to_suffix, False

    def featureset_creation_from_dataframe_helper(self, with_labels, use_feature_hasher):
        """
        Create featureset from dataframes for tests.

        Helper function for the two unit tests for FeatureSet.from_data_frame().
        Since labels are optional, run two tests, one with, one without.
        """
        # First, setup the test data.
        # get a 100 instances with 4 features each
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            random_state=1234567890,
        )

        # Not using 0 - 100 here because that would be pandas' default index names anyway.
        # So let's make sure pandas is using the ids we supply.
        ids = list(range(100, 200))

        featureset_name = "test"

        # if use_feature_hashing, run these tests with a vectorizer
        feature_bins = 4
        vectorizer = FeatureHasher(n_features=feature_bins) if use_feature_hasher else None

        # convert the features into a list of dictionaries
        feature_names = [f"f{n}" for n in range(1, 5)]
        features = []
        for row in X:
            features.append(dict(zip(feature_names, row)))

        # Now, create a FeatureSet object.
        if with_labels:
            expected = FeatureSet(
                featureset_name, ids, features=features, labels=y, vectorizer=vectorizer
            )
        else:
            expected = FeatureSet(featureset_name, ids, features=features, vectorizer=vectorizer)

        # Also create a DataFrame and then create a FeatureSet from it.
        df = pd.DataFrame(features, index=ids)
        if with_labels:
            df["y"] = y
            current = FeatureSet.from_data_frame(
                df, featureset_name, labels_column="y", vectorizer=vectorizer
            )
        else:
            current = FeatureSet.from_data_frame(df, featureset_name, vectorizer=vectorizer)

        return expected, current

    def test_featureset_creation_from_dataframe_with_labels(self):
        (expected, current) = self.featureset_creation_from_dataframe_helper(True, False)
        assert expected == current

    def test_featureset_creation_from_dataframe_without_labels(self):
        (expected, current) = self.featureset_creation_from_dataframe_helper(False, False)
        # Directly comparing FeatureSet objects fails here because both sets
        # of labels are all nan when labels isn't specified, and arrays of
        # all nan are not equal to each other.
        # Based off of FeatureSet.__self.assertEqual_()
        assert (
            expected.name == current.name
            and (expected.ids == current.ids).all()
            and expected.vectorizer == current.vectorizer
            and np.allclose(expected.features.data, current.features.data, rtol=1e-6)
            and np.all(np.isnan(expected.labels))
            and np.all(np.isnan(current.labels))
        )

    def test_featureset_creation_from_dataframe_with_labels_and_vectorizer(self):
        (expected, current) = self.featureset_creation_from_dataframe_helper(True, True)
        assert expected == current

    def test_featureset_creation_from_dataframe_without_labels_with_vectorizer(self):
        (expected, current) = self.featureset_creation_from_dataframe_helper(False, True)
        # Directly comparing FeatureSet objects fails here because both sets
        # of labels are all nan when labels isn't specified, and arrays of
        # all nan are not equal to each other.
        # Based off of FeatureSet.__self.assertEqual_()
        assert (
            expected.name == current.name
            and (expected.ids == current.ids).all()
            and expected.vectorizer == current.vectorizer
            and np.allclose(expected.features.data, current.features.data, rtol=1e-6)
            and np.all(np.isnan(expected.labels))
            and np.all(np.isnan(current.labels))
        )

    def test_writing_ndj_featureset_with_string_ids(self):
        test_dict_vectorizer = DictVectorizer()
        test_feat_dict_list = [{"a": 1.0, "b": 1.0}, {"b": 1.0, "c": 1.0}]
        Xtest = test_dict_vectorizer.fit_transform(test_feat_dict_list)
        fs_test = FeatureSet(
            "test", ids=["1", "2"], labels=[1, 2], features=Xtest, vectorizer=test_dict_vectorizer
        )
        output_path = other_dir / "test_string_ids.jsonlines"
        test_writer = NDJWriter(output_path, fs_test)
        test_writer.write()

        # read in the written file into a featureset and confirm that the
        # two featuresets are equal
        fs_test2 = NDJReader.for_path(output_path).read()

        assert fs_test == fs_test2

    def test_featureset_creation_from_dataframe_with_string_ids(self):
        dftest = pd.DataFrame({"id": ["1", "2"], "score": [1, 2], "text": ["a b", "b c"]})
        dftest.set_index("id", inplace=True)
        test_feat_dict_list = [{"a": 1.0, "b": 1.0}, {"b": 1.0, "c": 1.0}]
        test_dict_vectorizer = DictVectorizer()
        Xtest = test_dict_vectorizer.fit_transform(test_feat_dict_list)
        fs_test = FeatureSet(
            "test",
            ids=dftest.index.values,
            labels=dftest["score"].values,
            features=Xtest,
            vectorizer=test_dict_vectorizer,
        )
        output_path = other_dir / "test_string_ids_df.jsonlines"
        test_writer = NDJWriter(output_path, fs_test)
        test_writer.write()

        # read in the written file into a featureset and confirm that the
        # two featuresets are equal
        fs_test2 = NDJReader.for_path(output_path).read()

        assert fs_test == fs_test2

    def test_featureset_creation_from_dataframe_with_string_labels(self):
        dftest = pd.DataFrame({"id": [1, 2], "score": ["yes", "no"], "text": ["a b", "b c"]})
        dftest.set_index("id", inplace=True)
        test_feat_dict_list = [{"a": 1.0, "b": 1.0}, {"b": 1.0, "c": 1.0}]
        test_dict_vectorizer = DictVectorizer()
        Xtest = test_dict_vectorizer.fit_transform(test_feat_dict_list)
        fs_test = FeatureSet(
            "test",
            ids=dftest.index.values,
            labels=dftest["score"].values,
            features=Xtest,
            vectorizer=test_dict_vectorizer,
        )

        output_path = other_dir / "test_string_labels_df.jsonlines"
        test_writer = NDJWriter(output_path, fs_test)
        test_writer.write()

        # read in the written file into a featureset and confirm that the
        # two featuresets are equal
        fs_test2 = NDJReader.for_path(output_path, ids_to_floats=True).read()

        assert fs_test == fs_test2

    def test_reading_csv_and_tsv_with_drop_blanks(self):
        # create CSV and TSV strings with blanks
        test_csv = "1,1,6\n2,,2\n3,9,3\n,,\n,5,\n,,\n2,7,7"
        test_tsv = test_csv.replace(",", "\t")

        # specify pandas_kwargs for CSV and TSV readers
        kwargs = {"header": None, "names": ["A", "B", "C"]}

        expected = pd.DataFrame(
            {"A": [1, 3, 2], "B": [1, 9, 7], "C": [6, 3, 7], "L": [None, None, None]},
            index=["EXAMPLE_0", "EXAMPLE_1", "EXAMPLE_2"],
        )

        fs_expected = FeatureSet.from_data_frame(expected, "test", labels_column="L")

        # write out the test data
        csv_path = other_dir / "test_read_csv_tsv_drop_blanks.csv"
        self._create_test_file(csv_path, test_csv)

        tsv_path = other_dir / "test_read_csv_tsv_drop_blanks.tsv"
        self._create_test_file(tsv_path, test_tsv)

        fs_csv = CSVReader(csv_path, drop_blanks=True, pandas_kwargs=kwargs).read()
        fs_csv.name = "test"

        fs_tsv = TSVReader(tsv_path, drop_blanks=True, pandas_kwargs=kwargs).read()
        fs_tsv.name = "test"

        self.assertEqual(fs_csv, fs_expected)
        self.assertEqual(fs_tsv, fs_expected)

    def test_reading_csv_and_tsv_with_fill_blanks(self):
        # create CSV and TSV strings with blanks
        test_csv = "1,1,6\n2,,2\n3,9,3\n,,\n,5,\n,,\n2,7,7"
        test_tsv = test_csv.replace(",", "\t")

        # specify pandas_kwargs for CSV and TSV readers
        kwargs = {"header": None, "names": ["A", "B", "C"]}

        expected = pd.DataFrame(
            {
                "A": [1, 2, 3, 4.5, 4.5, 4.5, 2],
                "B": [1, 4.5, 9, 4.5, 5, 4.5, 7],
                "C": [6, 2, 3, 4.5, 4.5, 4.5, 7],
                "L": [None, None, None, None, None, None, None],
            },
            index=[
                "EXAMPLE_0",
                "EXAMPLE_1",
                "EXAMPLE_2",
                "EXAMPLE_3",
                "EXAMPLE_4",
                "EXAMPLE_5",
                "EXAMPLE_6",
            ],
        )

        fs_expected = FeatureSet.from_data_frame(expected, "test", labels_column="L")

        # write out the test data
        csv_path = other_dir / "test_read_csv_tsv_fill_blanks.csv"
        self._create_test_file(csv_path, test_csv)

        tsv_path = other_dir / "test_read_csv_tsv_fill_blanks.tsv"
        self._create_test_file(tsv_path, test_tsv)

        fs_csv = CSVReader(csv_path, replace_blanks_with=4.5, pandas_kwargs=kwargs).read()
        fs_csv.name = "test"

        fs_tsv = TSVReader(tsv_path, replace_blanks_with=4.5, pandas_kwargs=kwargs).read()
        fs_tsv.name = "test"

        self.assertEqual(fs_csv, fs_expected)
        self.assertEqual(fs_tsv, fs_expected)

    def test_reading_csv_and_tsv_with_fill_blanks_with_dictionary(self):
        # create CSV and TSV strings with blanks
        test_csv = "1,1,6\n2,,2\n3,9,3\n,,\n,5,\n,,\n2,7,7"
        test_tsv = test_csv.replace(",", "\t")

        # specify pandas_kwargs for CSV and TSV readers
        kwargs = {"header": None, "names": ["A", "B", "C"]}

        expected = pd.DataFrame(
            {
                "A": [1, 2, 3, 4.5, 4.5, 4.5, 2],
                "B": [1, 2.5, 9, 2.5, 5, 2.5, 7],
                "C": [6, 2, 3, 1, 1, 1, 7],
                "L": [None, None, None, None, None, None, None],
            },
            index=[
                "EXAMPLE_0",
                "EXAMPLE_1",
                "EXAMPLE_2",
                "EXAMPLE_3",
                "EXAMPLE_4",
                "EXAMPLE_5",
                "EXAMPLE_6",
            ],
        )

        fs_expected = FeatureSet.from_data_frame(expected, "test", labels_column="L")

        # write out the test data
        csv_path = other_dir / "test_read_csv_tsv_fill_blanks_dict.csv"
        self._create_test_file(csv_path, test_csv)

        tsv_path = other_dir / "test_read_csv_tsv_fill_blanks_dict.tsv"
        self._create_test_file(tsv_path, test_tsv)

        replacement_dict = {"A": 4.5, "B": 2.5, "C": 1}
        fs_csv = CSVReader(
            csv_path, replace_blanks_with=replacement_dict, pandas_kwargs=kwargs
        ).read()
        fs_csv.name = "test"

        fs_tsv = TSVReader(
            tsv_path, replace_blanks_with=replacement_dict, pandas_kwargs=kwargs
        ).read()
        fs_tsv.name = "test"

        self.assertEqual(fs_csv, fs_expected)
        self.assertEqual(fs_tsv, fs_expected)

    def test_drop_blanks_and_replace_blanks_with_raises_error(self):
        test_csv = "1,1,6\n2,,2\n3,9,3\n,,\n,5,\n,,\n2,7,7"
        csv_path = other_dir / "test_drop_blanks_error.csv"
        self._create_test_file(csv_path, test_csv)
        with self.assertRaises(ValueError):
            CSVReader(csv_path, replace_blanks_with=4.5, drop_blanks=True).read()

    def test_split_two_id_sets(self):
        """Test split() with two input id sets."""
        fs, _ = make_classification_data(
            num_examples=10, num_features=4, num_labels=2, train_test_ratio=1.0
        )

        # split by two id sets
        ids_split1 = range(5)
        ids_split2 = range(5, 10)

        fs1, fs2 = FeatureSet.split(fs, ids_split1, ids_split2)

        # verify that the ids, labels and features are split as expected
        assert_array_equal(fs.ids[ids_split1], fs1.ids)
        assert_array_equal(fs.ids[ids_split2], fs2.ids)
        assert_array_equal(fs.labels[ids_split1], fs1.labels)
        assert_array_equal(fs.labels[ids_split2], fs2.labels)
        assert_array_equal(fs.features[ids_split1, :].data, fs1.features.data)
        assert_array_equal(fs.features[ids_split2, :].data, fs2.features.data)

    def test_split_one_id_set(self):
        """Test split() with one input id sets."""
        fs, _ = make_classification_data(
            num_examples=10, num_features=4, num_labels=2, train_test_ratio=1.0
        )
        # split by one id set
        ids1_idx = [2, 3, 5, 6, 7]
        ids2_idx = [0, 1, 4, 8, 9]  # these are the expected ids for the second set

        fs1, fs2 = FeatureSet.split(fs, ids1_idx)

        # verify that the ids, labels and features are split as expected
        assert_array_equal(fs.ids[ids1_idx], fs1.ids)
        assert_array_equal(fs.ids[ids2_idx], fs2.ids)
        assert_array_equal(fs.labels[ids1_idx], fs1.labels)
        assert_array_equal(fs.labels[ids2_idx], fs2.labels)
        assert_array_equal(fs.features[ids1_idx, :].data, fs1.features.data)
        assert_array_equal(fs.features[ids2_idx, :].data, fs2.features.data)

    def test_split_int_ids(self):
        """Test split() when ids are integers."""
        fs, _ = make_classification_data(
            num_examples=10, num_features=4, num_labels=2, train_test_ratio=1.0, id_type="integer"
        )
        ids1_idx = [1, 3, 5, 7, 9]
        ids2_idx = [0, 2, 4, 6, 8]

        fs1, fs2 = FeatureSet.split(fs, ids1_idx, ids2_idx)

        # verify that the ids, labels and features are split as expected
        assert_array_equal(fs.ids[ids1_idx], fs1.ids)
        assert_array_equal(fs.ids[ids2_idx], fs2.ids)
        assert_array_equal(fs.labels[ids1_idx], fs1.labels)
        assert_array_equal(fs.labels[ids2_idx], fs2.labels)
        assert_array_equal(fs.features[ids1_idx, :].data, fs1.features.data)
        assert_array_equal(fs.features[ids2_idx, :].data, fs2.features.data)
