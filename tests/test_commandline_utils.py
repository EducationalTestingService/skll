# License: BSD 3 clause
"""
Module for running unit tests related to command line utilities.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
"""

import ast
import copy
import itertools
import logging
import sys
import unittest
from collections import defaultdict
from io import StringIO
from itertools import chain, combinations, product
from unittest.mock import create_autospec, patch

import numpy as np
import pandas as pd
import scipy as sp
from numpy import concatenate
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier, SGDRegressor

import skll
import skll.utils.commandline.compute_eval_from_predictions as cefp
import skll.utils.commandline.filter_features as ff
import skll.utils.commandline.generate_predictions as gp
import skll.utils.commandline.join_features as jf
import skll.utils.commandline.plot_learning_curves as plc
import skll.utils.commandline.print_model_weights as pmw
import skll.utils.commandline.run_experiment as rex
import skll.utils.commandline.skll_convert as sk
import skll.utils.commandline.summarize_results as sr
from skll.data import FeatureSet, LibSVMReader, LibSVMWriter, NDJWriter, safe_float
from skll.data.readers import EXT_TO_READER
from skll.data.writers import EXT_TO_WRITER
from skll.experiments import generate_learning_curve_plots, run_configuration
from skll.experiments.output import _write_summary_file
from skll.learner import Learner
from skll.utils.commandline.compute_eval_from_predictions import (
    get_prediction_from_probabilities,
)
from tests import other_dir, output_dir, test_dir, train_dir
from tests.utils import make_classification_data, make_regression_data, unlink


class TestCommandlineUtils(unittest.TestCase):
    """Test class for commandline utility tests."""

    @classmethod
    def setUpClass(cls):
        """Create necessary directories for testing."""
        for dir_path in [train_dir, test_dir, output_dir, other_dir / "features"]:
            dir_path.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        for path in [
            test_dir / "test_generate_predictions.jsonlines",
            test_dir / "test_single_file_subset.jsonlines",
            other_dir / "summary_file",
            other_dir / "test_filter_features_input.arff",
            output_dir / "test_generate_predictions.tsv",
            output_dir / "train_test_single_file.log",
        ]:
            unlink(path)

        for filepath in chain(
            output_dir.glob("test_print_model_weights.model*"),
            output_dir.glob("test_generate_predictions.model*"),
            output_dir.glob("pos_label_predict*"),
            output_dir.glob("test_generate_predictions_console.model*"),
            other_dir.glob("test_skll_convert*"),
            other_dir.glob("test_join_features*"),
            other_dir.glob("test_filter_features_labels*"),
            other_dir.glob("features/features*"),
        ):
            unlink(filepath)

    def test_compute_eval_from_predictions(self):
        """Test compute_eval_from_predictions function console script."""
        pred_path = other_dir / "test_compute_eval_from_predictions_predictions.tsv"
        input_path = other_dir / "test_compute_eval_from_predictions.jsonlines"

        # we need to capture stdout since that's what main() writes to
        compute_eval_from_predictions_cmd = [
            str(input_path),
            str(pred_path),
            "pearson",
            "unweighted_kappa",
        ]
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = mystderr = StringIO()
        try:
            cefp.main(compute_eval_from_predictions_cmd)
            score_rows = mystdout.getvalue().strip().split("\n")
            print(mystderr.getvalue())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        scores = {}
        for score_row in score_rows:
            score, metric_name, pred_path = score_row.split("\t")
            scores[metric_name] = float(score)

        self.assertAlmostEqual(scores["pearson"], 0.6197797868009122)
        self.assertAlmostEqual(scores["unweighted_kappa"], 0.2)

    def test_warning_when_prediction_method_and_no_probabilities(self):
        """
        Test for presence of warning.

        Test compute_eval_from_predictions logs for a warning if a prediction method
        is provided but the predictions file doesn't contain probabilities.
        """
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        logger = logging.getLogger("skll.utils.commandline.compute_eval_from_predictions")
        logger.addHandler(ch)

        pred_path = other_dir / "test_compute_eval_from_predictions_predictions.tsv"
        input_path = other_dir / "test_compute_eval_from_predictions.jsonlines"

        # we need to capture stdout since that's what main() writes to
        compute_eval_from_predictions_cmd = [
            str(input_path),
            str(pred_path),
            "pearson",
            "unweighted_kappa",
            "--method",
            "highest",
        ]

        with patch("sys.stdout", new=StringIO()) as fake_out, patch(
            "sys.stderr", new=StringIO()
        ) as fake_err:
            cefp.main(compute_eval_from_predictions_cmd)
            _ = fake_out.getvalue().strip().split("\n")
            err = fake_err.getvalue()
            print(err)

        expected_log_msg = (
            "A prediction method was provided, but the predictions file doesn't "
            "contain probabilities. Ignoring prediction method 'highest'."
        )

        log_output = log_capture_string.getvalue().strip()
        self.assertTrue(expected_log_msg in log_output)

    def test_compute_eval_from_predictions_with_probs(self):
        """
        Test `compute_eval_from_predictions` function console script.

        This test uses probabilities in the predictions file.
        """
        pred_path = other_dir / "test_compute_eval_from_predictions_probs_predictions.tsv"
        input_path = other_dir / "test_compute_eval_from_predictions_probs.jsonlines"

        # we need to capture stdout since that's what main() writes to
        compute_eval_from_predictions_cmd = [
            str(input_path),
            str(pred_path),
            "pearson",
            "unweighted_kappa",
        ]
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = mystderr = StringIO()
        try:
            cefp.main(compute_eval_from_predictions_cmd)
            score_rows = mystdout.getvalue().strip().split("\n")
            print(mystderr.getvalue())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        scores = {}
        for score_row in score_rows:
            score, metric_name, pred_path = score_row.split("\t")
            scores[metric_name] = float(score)

        self.assertAlmostEqual(scores["pearson"], 0.6197797868009122)
        self.assertAlmostEqual(scores["unweighted_kappa"], 0.2)

        #
        # Test expected value predictions method
        #
        compute_eval_from_predictions_cmd = [
            str(input_path),
            str(pred_path),
            "explained_variance",
            "r2",
            "--method",
            "expected_value",
        ]
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = mystderr = StringIO()
        try:
            cefp.main(compute_eval_from_predictions_cmd)
            score_rows = mystdout.getvalue().strip().split("\n")
            print(mystderr.getvalue())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        scores = {}
        for score_row in score_rows:
            score, metric_name, pred_path = score_row.split("\t")
            scores[metric_name] = float(score)

        self.assertAlmostEqual(scores["r2"], 0.19999999999999996)
        self.assertAlmostEqual(scores["explained_variance"], 0.23809523809523792)

    def test_compute_eval_from_predictions_breaks_with_expval_and_nonnumeric_classes(self):
        """
        Make sure `compute_eval_from_predictions` raises ValueError for non-numeric classes.

        This is when predictions are calculated via expected_value and the classes
        are non numeric.
        """
        pred_path = (
            other_dir / "test_compute_eval_from_predictions_nonnumeric_classes_predictions.tsv"
        )
        input_path = other_dir / "test_compute_eval_from_predictions_nonnumeric_classes.jsonlines"

        compute_eval_from_predictions_cmd = [
            str(input_path),
            str(pred_path),
            "explained_variance",
            "r2",
            "--method",
            "expected_value",
        ]
        with self.assertRaises(ValueError):
            cefp.main(compute_eval_from_predictions_cmd)

    def test_conflicting_prediction_and_example_ids(self):
        """
        Make sure `compute_eval_from_predictions` raises ValueError for mismatched IDs.

        This is when predictions and examples don't have the same id set in
        `compute_eval_from_predictions`.
        """
        pred_path = other_dir / "test_compute_eval_from_predictions_probs_predictions.tsv"
        input_path = other_dir / "test_compute_eval_from_predictions_different_ids.jsonlines"

        compute_eval_from_predictions_cmd = [str(input_path), str(pred_path), "pearson"]
        with self.assertRaises(ValueError):
            cefp.main(compute_eval_from_predictions_cmd)

    def test_compute_eval_from_predictions_random_choice(self):
        """Test that random selection of classes with the same probabilities works."""
        classes = ["A", "B", "C", "D"]
        probs = ["0.25", "0.25", "0.25", "0.25"]
        prediction_method = "highest"
        pred = get_prediction_from_probabilities(classes, probs, prediction_method)
        self.assertEqual(pred, "C")

    def _run_generate_predictions_and_capture_output(self, generate_cmd, output_file):
        if output_file == "stdout":
            # we need to capture stdout since that's what main() writes to
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            try:
                gp.main(generate_cmd)
                out = mystdout.getvalue()
                output_lines = out.strip().split("\n")
            finally:
                sys.stdout = old_stdout
        else:
            unlink(output_file)
            gp.main(generate_cmd)
            with open(output_file) as outputfh:
                output_lines = [line.strip() for line in outputfh.readlines()]

        return output_lines

    def check_generate_predictions(
        self,
        use_regression=False,
        string_labels=False,
        num_labels=2,
        use_probability=False,
        use_pos_label=False,
        test_on_subset=False,
        use_threshold=False,
        predict_labels=False,
        use_stdout=False,
        multiple_input_files=False,
    ):
        # create some simple classification feature sets for training and testing
        if string_labels:
            string_label_list = ["a", "b"] if num_labels == 2 else ["a", "b", "c", "d"]
        else:
            string_label_list = None

        # generate the train and test featuresets
        if use_regression:
            pos_label = None
            train_fs, test_fs, _ = make_regression_data(num_examples=100, num_features=5)
        else:
            train_fs, test_fs = make_classification_data(
                num_examples=100,
                num_features=5,
                num_labels=num_labels,
                string_label_list=string_label_list,
            )

            # get the sorted list of unique class labels
            class_labels = np.unique(train_fs.labels).tolist()

            # if we are using `pos_label`, then randomly pick a label
            # as its value even for multi-class since we want to check that
            # it properly gets ignored; we also instantiate variables in the
            # binary case that contain the positive and negative class labels
            # since we need those to process our expected predictions we
            # are matching against
            prng = np.random.RandomState(123456789)
            if use_pos_label:
                pos_label = prng.choice(class_labels, size=1)[0]
                if num_labels == 2:
                    positive_class_label = pos_label
                    negative_class_label = [
                        label for label in class_labels if label != positive_class_label
                    ][0]
                    class_labels = [negative_class_label, positive_class_label]
            else:
                pos_label = None
                if num_labels == 2:
                    positive_class_label = class_labels[-1]
                    negative_class_label = [
                        label for label in class_labels if label != positive_class_label
                    ][0]

        # if we are asked to use only a subset, then filter out
        # one of the features if we are not using feature hashing,
        # do nothing if we are using feature hashing
        if test_on_subset:
            if use_regression:
                test_fs.filter(features=["f1", "f2", "f3", "f4"])
            else:
                test_fs.filter(features=["f01", "f02", "f03", "f04"])

        # write out the test set to disk so we can use it as a file
        test_file = test_dir / "test_generate_predictions.jsonlines"
        NDJWriter.for_path(test_file, test_fs).write()

        # we need probabilities if we are using thresholds or label inference
        enable_probability = any([use_probability, use_threshold, predict_labels])

        # create a SKLL learner that is an SGDClassifier or an SGDRegressor,
        # train it, and then save it to disk for use as a file
        learner_name = "SGDRegressor" if use_regression else "SGDClassifier"
        learner = Learner(learner_name, probability=enable_probability, pos_label=pos_label)
        learner.train(train_fs, grid_search=False)
        model_file = output_dir / "test_generate_predictions.model"
        learner.save(model_file)

        # now train equivalent sklearn estimators that we will use
        # to get the expected predictions
        if use_regression:
            model = SGDRegressor(max_iter=1000, random_state=123456789, tol=1e-3)
        else:
            model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=123456789, tol=1e-3)
        model.fit(train_fs.features, train_fs.labels)

        # get the predictions from this sklearn model on the test set
        if test_on_subset:
            xtest = learner.feat_vectorizer.transform(
                test_fs.vectorizer.inverse_transform(test_fs.features)
            )
        else:
            xtest = test_fs.features

        if not (use_regression or predict_labels) and (use_threshold or use_probability):
            predictions = model.predict_proba(xtest)

            # since we are directly passing in the string labels to
            # sklearn, it would sort the labels internally which
            # may not match our expectation of having the positive
            # class label probability last when doing binary
            # classification so let's re-sort the sklearn predictions
            sklearn_classes = model.classes_.tolist()
            if num_labels == 2 and not np.all(model.classes_ == class_labels):
                positive_class_index = sklearn_classes.index(positive_class_label)
                negative_class_index = 1 - positive_class_index
                predictions[:, [0, 1]] = predictions[
                    :, [negative_class_index, positive_class_index]
                ]
        else:
            predictions = model.predict(xtest)

        # now start constructing the `generate_predictions` command
        generate_cmd = ["-q"]

        # if we asked for thresholded predictions
        if use_threshold and not use_regression:
            # append the threshold to the command
            threshold = 0.6
            generate_cmd.append("-t 0.6")

            # threshold the expected predictions; note that
            # since we have already reordered sklearn predictions
            # we can just look at the second index to get the
            # positive class probability
            predictions = np.where(
                predictions[:, 1] >= threshold, positive_class_label, negative_class_label
            )

        # if we asked to predict most likely labels
        elif predict_labels and not use_regression:
            # append the option to the command
            generate_cmd.append("-p")

        # are we using an output file or the console?
        if not use_stdout:
            output_file = output_dir / "test_generate_predictions.tsv"
            generate_cmd.extend(["--output", str(output_file)])
        else:
            output_file = "stdout"

        # append the model file to the command
        generate_cmd.append(str(model_file))

        # if we are using multiple input files, repeat the input file
        # and, correspondingly, concatenate the expected predictions
        if multiple_input_files:
            predictions = concatenate([predictions, predictions])
            generate_cmd.extend([str(test_file), str(test_file)])
        else:
            generate_cmd.append(str(test_file))

        # run the constructed command and capture its output
        gp_output_lines = self._run_generate_predictions_and_capture_output(
            generate_cmd, output_file
        )

        # check the header first
        generated_header = gp_output_lines[0].strip().split("\t")
        if use_regression or not enable_probability or use_threshold or predict_labels:
            expected_header = ["id", "prediction"]
        else:
            if num_labels == 2:
                expected_header = ["id"] + [str(negative_class_label), str(positive_class_label)]
            else:
                expected_header = ["id"] + [str(x) for x in class_labels]
        self.assertEqual(generated_header, expected_header)

        # now check the ids, and predictions
        for gp_line, expected_id, expected_prediction in zip(
            gp_output_lines[1:], test_fs.ids, predictions
        ):
            generated_fields = gp_line.strip().split("\t")
            # comparing most likely labels
            if len(generated_fields) == 2:
                generated_id, generated_prediction = generated_fields[0], generated_fields[1]
                self.assertEqual(generated_id, expected_id)
                self.assertEqual(safe_float(generated_prediction), expected_prediction)
            # comparing probabilities
            else:
                generated_id = generated_fields[0]
                generated_prediction = list(map(safe_float, generated_fields[1:]))
                self.assertEqual(generated_id, expected_id)
                assert_array_almost_equal(generated_prediction, expected_prediction)

    def test_generate_predictions(self):
        for (
            use_regression,
            string_labels,
            num_labels,
            use_probability,
            use_pos_label,
            test_on_subset,
            use_threshold,
            predict_labels,
            use_stdout,
            multiple_input_files,
        ) in product(
            [True, False],
            [True, False],
            [2, 4],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        ):
            # skip testing conditions that will raise exceptions
            # in `generate_predictions`
            if (
                use_threshold
                and num_labels != 2
                or use_threshold
                and predict_labels
                or use_regression
                and string_labels
            ):
                continue
            yield (
                self.check_generate_predictions,
                use_regression,
                string_labels,
                num_labels,
                use_probability,
                use_pos_label,
                test_on_subset,
                use_threshold,
                predict_labels,
                use_stdout,
                multiple_input_files,
            )

    def test_generate_predictions_console_bad_input_ext(self):
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.ERROR)

        logger = logging.getLogger("skll.utils.commandline.generate_predictions")
        logger.addHandler(ch)

        # create some simple classification data without feature hashing
        train_fs, test_fs = make_classification_data(num_examples=1000, num_features=5)

        # create a learner that uses an SGD classifier
        learner = Learner("SGDClassifier")

        # train the learner with grid search
        learner.train(train_fs, grid_search=False)

        # get the predictions on the test featureset
        _ = learner.predict(test_fs)

        # save the learner to a file
        model_file = output_dir / "test_generate_predictions_console.model"
        learner.save(model_file)

        # now call main() from generate_predictions.py
        generate_cmd = [str(model_file), "fake_input_file.txt"]

        _ = self._run_generate_predictions_and_capture_output(generate_cmd, "stdout")

        expected_log_msg = (
            "Input file must be in either .arff, .csv, .jsonlines, .libsvm, .ndj, "
            "or .tsv format.  Skipping file fake_input_file.txt"
        )

        log_output = log_capture_string.getvalue().strip()
        self.assertTrue(expected_log_msg in log_output)

    def test_generate_predictions_threshold_not_trained_with_probability(self):
        # create some simple classification data without feature hashing
        train_fs, test_fs = make_classification_data(
            num_examples=1000, num_features=5, num_labels=2
        )

        # save the test feature set to an NDJ file
        input_file = test_dir / "test_generate_predictions.jsonlines"
        writer = NDJWriter(input_file, test_fs)
        writer.write()

        # create a learner that uses an SGD classifier
        learner = Learner("SGDClassifier")

        # train the learner with grid search
        learner.train(train_fs, grid_search=False)

        # save the learner to a file
        model_file = output_dir / "test_generate_predictions_console.model"
        learner.save(model_file)

        # now call main() from generate_predictions.py
        generate_cmd = [str(model_file), str(input_file)]
        generate_cmd.append("-t 0.6")
        with self.assertRaises(ValueError):
            gp.main(generate_cmd)

    def test_generate_predictions_threshold_multi_class(self):
        # create some simple classification data without feature hashing
        train_fs, test_fs = make_classification_data(
            num_examples=1000, num_features=5, num_labels=4
        )

        # save the test feature set to an NDJ file
        input_file = test_dir / "test_generate_predictions.jsonlines"
        writer = NDJWriter(input_file, test_fs)
        writer.write()

        # create a learner that uses an SGD classifier
        learner = Learner("LogisticRegression", probability=True)

        # train the learner with grid search
        learner.train(train_fs, grid_search=False)

        # save the learner to a file
        model_file = output_dir / "test_generate_predictions_console.model"
        learner.save(model_file)

        # now call main() from generate_predictions.py
        generate_cmd = [str(model_file), str(input_file)]
        generate_cmd.append("-t 0.6")
        with self.assertRaises(ValueError):
            gp.main(generate_cmd)

    def test_generate_predictions_threshold_non_probabilistic(self):
        # create some simple classification data without feature hashing
        train_fs, test_fs = make_classification_data(
            num_examples=1000, num_features=5, num_labels=2
        )

        # save the test feature set to an NDJ file
        input_file = test_dir / "test_generate_predictions.jsonlines"
        writer = NDJWriter(input_file, test_fs)
        writer.write()

        # create a learner that uses an SGD classifier
        learner = Learner("LinearSVC", probability=True)

        # train the learner with grid search
        learner.train(train_fs, grid_search=False)

        # save the learner to a file
        model_file = output_dir / "test_generate_predictions_console.model"
        learner.save(model_file)

        # now call main() from generate_predictions.py
        generate_cmd = [str(model_file), str(input_file)]
        generate_cmd.append("-t 0.6")
        with self.assertRaises(ValueError):
            gp.main(generate_cmd)

    def test_generate_predictions_predict_labels_not_trained_with_probability(self):
        # create some simple classification data without feature hashing
        train_fs, test_fs = make_classification_data(
            num_examples=1000, num_features=5, num_labels=2
        )

        # save the test feature set to an NDJ file
        input_file = test_dir / "test_generate_predictions.jsonlines"
        writer = NDJWriter(input_file, test_fs)
        writer.write()

        # create a learner that uses an SGD classifier
        learner = Learner("SGDClassifier")

        # train the learner with grid search
        learner.train(train_fs, grid_search=False)

        # save the learner to a file
        model_file = output_dir / "test_generate_predictions_console.model"
        learner.save(model_file)

        # now call main() from generate_predictions.py
        generate_cmd = [str(model_file), str(input_file)]
        generate_cmd.append("-p")
        with self.assertRaises(ValueError):
            gp.main(generate_cmd)

    def test_generate_predictions_predict_labels_non_probabilistic(self):
        # create some simple classification data without feature hashing
        train_fs, test_fs = make_classification_data(
            num_examples=1000, num_features=5, num_labels=4
        )

        # save the test feature set to an NDJ file
        input_file = test_dir / "test_generate_predictions.jsonlines"
        writer = NDJWriter(input_file, test_fs)
        writer.write()

        # create a learner that uses an SGD classifier
        learner = Learner("LinearSVC", probability=True)

        # train the learner with grid search
        learner.train(train_fs, grid_search=False)

        # save the learner to a file
        model_file = output_dir / "test_generate_predictions_console.model"
        learner.save(model_file)

        # now call main() from generate_predictions.py
        generate_cmd = [str(model_file), str(input_file)]
        generate_cmd.append("-p")
        with self.assertRaises(ValueError):
            gp.main(generate_cmd)

    def test_mutually_exclusive_generate_predictions_args(self):
        # create some simple classification data without feature hashing
        train_fs, test_fs = make_classification_data(num_examples=1000, num_features=5)
        threshold = 0.6

        # save the test feature set to an NDJ file
        input_file = test_dir / "test_generate_predictions.jsonlines"
        writer = NDJWriter(input_file, test_fs)
        writer.write()

        # create a learner that uses an SGD classifier
        learner = Learner("SGDClassifier")

        # train the learner with grid search
        learner.train(train_fs, grid_search=False)

        # save the learner to a file
        model_file = output_dir / "test_generate_predictions_console.model"
        learner.save(model_file)

        # now call main() from generate_predictions.py
        generate_cmd = [f"-t {threshold}", "-p"]
        generate_cmd.extend([str(model_file), str(input_file)])
        with self.assertRaises(SystemExit):
            gp.main(generate_cmd)

    def check_skll_convert(self, from_suffix, to_suffix, id_type):
        # create some simple classification data
        orig_fs, _ = make_classification_data(
            train_test_ratio=1.0, one_string_feature=True, id_type=id_type
        )

        # now write out this feature set in the given suffix
        from_suffix_file = other_dir / f"test_skll_convert_{id_type}_ids_in{from_suffix}"
        to_suffix_file = other_dir / f"test_skll_convert_{id_type}_ids_out{to_suffix}"

        writer = EXT_TO_WRITER[from_suffix](from_suffix_file, orig_fs, quiet=True)
        writer.write()

        # now run skll convert to convert the featureset into the other format
        skll_convert_cmd = [str(from_suffix_file), str(to_suffix_file), "--quiet"]

        # we need to capture stderr to make sure we don't miss any errors
        old_stderr = sys.stderr
        sys.stderr = mystderr = StringIO()
        try:
            sk.main(skll_convert_cmd)
            print(mystderr.getvalue())
        finally:
            sys.stderr = old_stderr

        # now read the converted file and appropriately set `ids_to_floats`
        # depending on the ID types that we generated earlier
        ids_to_floats = True if id_type in ["float", "integer"] else False
        reader = EXT_TO_READER[to_suffix](to_suffix_file, ids_to_floats=ids_to_floats, quiet=True)
        converted_fs = reader.read()

        # TODO : For now, we are converting feature arrays to dense, and then back to sparse.
        # The reason for this is that scikit-learn DictVectorizers now retain any
        # explicit zeros that are in files (e.g., in CSVs and TSVs). There's an issue
        # open on scikit-learn: https://github.com/scikit-learn/scikit-learn/issues/14718

        orig_fs.features = sp.sparse.csr_matrix(orig_fs.features.toarray())
        converted_fs.features = sp.sparse.csr_matrix(converted_fs.features.toarray())

        self.assertEqual(orig_fs, converted_fs)

    def test_skll_convert(self):
        for from_suffix, to_suffix in itertools.permutations(
            [".arff", ".csv", ".jsonlines", ".libsvm", ".tsv"], 2
        ):
            yield self.check_skll_convert, from_suffix, to_suffix, "string"
            yield self.check_skll_convert, from_suffix, to_suffix, "integer_string"
            yield self.check_skll_convert, from_suffix, to_suffix, "float"
            yield self.check_skll_convert, from_suffix, to_suffix, "integer"

    def test_skll_convert_libsvm_map(self):
        """Test to check whether the --reuse_libsvm_map option works for skll_convert."""
        # create some simple classification data
        orig_fs, _ = make_classification_data(train_test_ratio=1.0, one_string_feature=True)

        # now write out this feature set as a libsvm file
        orig_libsvm_file = other_dir / "test_skll_convert_libsvm_map.libsvm"
        writer = LibSVMWriter(orig_libsvm_file, orig_fs, quiet=True)
        writer.write()

        # now make a copy of the dataset
        swapped_fs = copy.deepcopy(orig_fs)

        # now modify this new featureset to swap the first two columns
        del swapped_fs.vectorizer.vocabulary_["f01"]
        del swapped_fs.vectorizer.vocabulary_["f02"]
        swapped_fs.vectorizer.vocabulary_["f01"] = 1
        swapped_fs.vectorizer.vocabulary_["f02"] = 0
        tmp = swapped_fs.features[:, 0]
        swapped_fs.features[:, 0] = swapped_fs.features[:, 1]
        swapped_fs.features[:, 1] = tmp

        # now run skll_convert to convert this into a libsvm file
        # but using the mapping specified in the first libsvm file
        converted_libsvm_file = other_dir / "test_skll_convert_libsvm_map2.libsvm"

        # now call skll convert's main function
        skll_convert_cmd = [
            "--reuse_libsvm_map",
            str(orig_libsvm_file),
            "--quiet",
            str(orig_libsvm_file),
            str(converted_libsvm_file),
        ]
        old_stderr = sys.stderr
        sys.stderr = mystderr = StringIO()
        try:
            sk.main(skll_convert_cmd)
            print(mystderr.getvalue())
        finally:
            sys.stderr = old_stderr

        # now read the converted libsvm file into a featureset
        reader = LibSVMReader(converted_libsvm_file, quiet=True)
        converted_fs = reader.read()

        # now ensure that this new featureset and the original
        # featureset are the same
        self.assertEqual(orig_fs, converted_fs)

    def test_skll_convert_no_labels_with_label_col(self):
        """Check that --no_labels/--label_col cannot both be specified for skll_convert."""
        skll_convert_cmd = ["--no_labels", "--label_col", "t", "foo.tsv", "foo.libsvm"]
        with self.assertRaises(SystemExit):
            sk.main(argv=skll_convert_cmd)

    def check_print_model_weights(self, task="classification", sort_by_labels=False):  # noqa: C901
        # create some simple classification or regression data
        if task in ["classification", "classification_no_intercept"]:
            train_fs, _ = make_classification_data(train_test_ratio=0.8)
        elif task == "classification_with_hashing":
            train_fs, _ = make_classification_data(
                train_test_ratio=0.8, use_feature_hashing=True, feature_bins=10
            )
        elif task in ["multiclass_classification", "multiclass_classification_svc"]:
            train_fs, _ = make_classification_data(train_test_ratio=0.8, num_labels=3)
        elif task in [
            "multiclass_classification_with_hashing",
            "multiclass_classification_svc_with_hashing",
        ]:
            train_fs, _ = make_classification_data(
                train_test_ratio=0.8, num_labels=3, use_feature_hashing=True, feature_bins=10
            )

        elif task in [
            "regression_with_hashing",
            "regression_linearsvr_with_hashing",
            "regression_svr_linear_with_hashing",
            "regression_svr_linear_with_scaling_and_hashing",
        ]:
            train_fs, _, _ = make_regression_data(
                num_features=4, train_test_ratio=0.8, use_feature_hashing=True, feature_bins=2
            )
        else:
            train_fs, _, _ = make_regression_data(num_features=4, train_test_ratio=0.8)

        # now train the appropriate model
        if task in [
            "classification",
            "classification_with_hashing",
            "multiclass_classification",
            "multiclass_classification_with_hashing",
        ]:
            learner = Learner("LogisticRegression")
            learner.train(train_fs, grid_search=True, grid_objective="f1_score_micro")
        elif task in [
            "multiclass_classification_svc",
            "multiclass_classification_svc_with_hashing",
        ]:
            learner = Learner("SVC", model_kwargs={"kernel": "linear"})
            learner.train(train_fs, grid_search=True, grid_objective="f1_score_micro")
        elif task == "classification_no_intercept":
            learner = Learner("LogisticRegression")
            learner.train(
                train_fs,
                grid_search=True,
                grid_objective="f1_score_micro",
                param_grid=[{"fit_intercept": [False]}],
            )
        elif task in ["regression", "regression_with_hashing"]:
            learner = Learner("LinearRegression")
            learner.train(train_fs, grid_search=True, grid_objective="pearson")
        elif task in ["regression_linearsvr", "regression_linearsvr_with_hashing"]:
            learner = Learner("LinearSVR")
            learner.train(train_fs, grid_search=True, grid_objective="pearson")
        elif task in ["regression_svr_linear", "regression_svr_linear_with_hashing"]:
            learner = Learner("SVR", model_kwargs={"kernel": "linear"})
            learner.train(train_fs, grid_search=True, grid_objective="pearson")
        else:
            learner = Learner("SVR", model_kwargs={"kernel": "linear"}, feature_scaling="both")
            learner.train(train_fs, grid_search=True, grid_objective="pearson")

        # now save the model to disk
        model_file = output_dir / "test_print_model_weights.model"
        learner.save(model_file)

        # now call print_model_weights main() and capture the output
        if sort_by_labels:
            print_model_weights_cmd = [str(model_file), "--sort_by_labels"]
        else:
            print_model_weights_cmd = [str(model_file)]
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = mystderr = StringIO()
        sys.stdout = mystdout = StringIO()
        try:
            pmw.main(print_model_weights_cmd)
            out = mystdout.getvalue()
            print(mystderr.getvalue())
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout

        # now parse the output of the print_model_weight command
        # and get the intercept and the feature values
        if task in ["classification", "classification_with_hashing"]:
            lines_to_parse = [l for l in out.split("\n")[1:] if l]  # noqa: E741
            intercept = safe_float(lines_to_parse[0].split("\t")[0])
            feature_values = []
            for ltp in lines_to_parse[1:]:
                weight, _, feature_name = ltp.split("\t")
                feature_values.append((feature_name, safe_float(weight)))
            feature_values = [t[1] for t in sorted(feature_values)]
            self.assertAlmostEqual(intercept, learner.model.intercept_[0])
            assert_allclose(learner.model.coef_[0], feature_values)
        elif task in ["multiclass_classification", "multiclass_classification_with_hashing"]:
            # for multiple classes we get an intercept for each class
            # as well as a list of weights for each class

            lines_to_parse = [l for l in out.split("\n")[1:] if l]  # noqa: E741
            intercept = []
            for intercept_string in lines_to_parse[0:3]:
                intercept.append(safe_float(intercept_string.split("\t")[0]))

            feature_values = [[], [], []]
            for ltp in lines_to_parse[3:]:
                weight, label, feature_name = ltp.split("\t")
                feature_values[int(label)].append((feature_name, safe_float(weight)))

            if sort_by_labels:
                # making sure that the weights are sorted by label

                # get the labels
                labels_list = [line.split("\t")[1] for line in lines_to_parse[3:]]

                # first test that the labels are sorted
                assert labels_list == sorted(labels_list)

                # then test that weights are sorted descending by absolute value
                # for each label
                for features_and_weights in feature_values:
                    feature_weights = [t[1] for t in features_and_weights]
                    assert feature_weights == sorted(feature_weights, key=lambda x: -abs(x))

            for index, weights in enumerate(feature_values):
                feature_values[index] = [t[1] for t in sorted(weights)]

            for index, weights in enumerate(learner.model.coef_):
                assert_array_almost_equal(weights, feature_values[index])

            assert_array_almost_equal(intercept, learner.model.intercept_)
        elif task in [
            "multiclass_classification_svc",
            "multiclass_classification_svc_with_hashing",
        ]:
            # for multiple classes with the SVC with a linear kernel,
            # we get an intercept for each class pair combination
            # as well as a list of weights for each class pair
            # combination

            # save the computed intercept values in a dictionary
            # with the class oair label as the key
            lines_to_parse = [l for l in out.split("\n")[1:] if l]  # noqa: E741
            parsed_intercepts_dict = {}
            for intercept_string in lines_to_parse[0:3]:
                fields = intercept_string.split("\t")
                parsed_intercepts_dict[fields[1]] = safe_float(fields[0])

            # save the computed feature weights in a dictionary
            # with the class pair label as the key and the value
            # being a list; each feature weight for this class pair
            # is stored at the index of the feature name as given
            # by the feature vectorizer vocabulary dictionary
            parsed_weights_dict = {}
            for ltp in lines_to_parse[3:]:
                (weight, class_pair, feature) = ltp.split("\t")
                if class_pair not in parsed_weights_dict:
                    parsed_weights_dict[class_pair] = [0] * 10
                if isinstance(learner.feat_vectorizer, FeatureHasher):
                    feature_index = int(feature.split("_")[-1]) - 1
                else:
                    feature_index = learner.feat_vectorizer.vocabulary_[feature]
                parsed_weights_dict[class_pair][feature_index] = safe_float(weight)

            if sort_by_labels:
                # making sure that the weights are sorted by label

                # get the feature weights and class pairs
                temp_weights_dict = defaultdict(list)
                class_pair_list = []
                for ltp in lines_to_parse[3:]:
                    (weight, class_pair, feature) = ltp.split("\t")
                    class_pair_list.append(class_pair)
                    if class_pair not in parsed_weights_dict:
                        parsed_weights_dict[class_pair] = [0] * 10
                    temp_weights_dict[class_pair].append(safe_float(weight))

                # first test that the class pairs are sorted
                assert class_pair_list == sorted(class_pair_list)

                # then test that weifghts are sorted descending by absolute value
                # for each label
                for class_pair, feature_weights in temp_weights_dict.items():
                    assert feature_weights == sorted(feature_weights, key=lambda x: -abs(x))

            # to validate that our coefficients are correct, we will
            # get the coefficient array (for all features) from `coef_`
            # for a particular class pair and then check that this array
            # is equal to the list that we computed above. We will do
            # the same for intercepts which are even easier to validate
            # since they _only_ depend on the class pair
            for idx, (class1, class2) in enumerate(itertools.combinations([0, 1, 2], 2)):
                class_pair_label = f"{class1}-vs-{class2}"
                computed_coefficients = np.array(parsed_weights_dict[class_pair_label])
                # we want to remove any extra zeros here for features that ended up
                # with zero weights since those are never printed out
                computed_coefficients = computed_coefficients[computed_coefficients.nonzero()]
                expected_coefficients = learner.model.coef_[idx].toarray()[0]
                assert_array_almost_equal(computed_coefficients, expected_coefficients)

                computed_intercept = parsed_intercepts_dict[class_pair_label]
                expected_intercept = learner.model.intercept_[idx]
                self.assertAlmostEqual(computed_intercept, expected_intercept)

        elif task == "classification_no_intercept":
            lines_to_parse = [l for l in out.split("\n")[0:] if l]  # noqa: E741
            intercept = safe_float(lines_to_parse[0].split("=")[1])
            computed_coefficients = []
            for ltp in lines_to_parse[1:]:
                fields = ltp.split("\t")
                computed_coefficients.append((fields[2], safe_float(fields[0])))
            computed_coefficients = [t[1] for t in sorted(computed_coefficients)]
            self.assertAlmostEqual(intercept, learner.model.intercept_)
            expected_coefficients = learner.model.coef_[0]
            assert_allclose(expected_coefficients, computed_coefficients)
        elif task in ["regression", "regression_with_hashing"]:
            lines_to_parse = [l for l in out.split("\n") if l]  # noqa: E741
            intercept = safe_float(lines_to_parse[0].split("=")[1])
            computed_coefficients = []
            for ltp in lines_to_parse[1:]:
                weight, feature_name = ltp.split("\t")
                computed_coefficients.append((feature_name, safe_float(weight)))
            computed_coefficients = [t[1] for t in sorted(computed_coefficients)]
            self.assertAlmostEqual(intercept, learner.model.intercept_)
            assert_allclose(learner.model.coef_, computed_coefficients)
        else:
            lines_to_parse = [l for l in out.split("\n") if l]  # noqa: E741

            intercept_list = ast.literal_eval(lines_to_parse[0].split("=")[1].strip())
            intercept = safe_float(intercept_list)

            feature_values = []
            for ltp in lines_to_parse[1:]:
                fields = ltp.split("\t")
                feature_values.append((fields[1], safe_float(fields[0])))
            feature_values = [t[1] for t in sorted(feature_values)]

            assert_array_almost_equal(intercept, learner.model.intercept_)
            if task in ["regression_svr_linear", "regression_svr_linear_with_hashing"]:
                coef = learner.model.coef_.toarray()[0]
                assert_allclose(coef, feature_values)
            elif task in [
                "regression_svr_linear_with_scaling",
                "regression_svr_linear_with_scaling_and_hashing",
            ]:
                coef = learner.model.coef_[0]
                assert_allclose(coef, feature_values)
            else:
                assert_allclose(learner.model.coef_, feature_values)

    def test_print_model_weights(self):
        yield self.check_print_model_weights, "classification"
        yield self.check_print_model_weights, "classification_with_hashing"
        yield self.check_print_model_weights, "multiclass_classification"
        yield self.check_print_model_weights, "multiclass_classification", True
        yield self.check_print_model_weights, "multiclass_classification_with_hashing"
        yield self.check_print_model_weights, "multiclass_classification_with_hashing", True
        yield self.check_print_model_weights, "multiclass_classification_svc"
        yield self.check_print_model_weights, "multiclass_classification_svc", True
        yield self.check_print_model_weights, "multiclass_classification_svc_with_hashing"
        yield self.check_print_model_weights, "multiclass_classification_svc_with_hashing", True
        yield self.check_print_model_weights, "classification_no_intercept"
        yield self.check_print_model_weights, "regression"
        yield self.check_print_model_weights, "regression_with_hashing"
        yield self.check_print_model_weights, "regression_linearsvr"
        yield self.check_print_model_weights, "regression_linearsvr_with_hashing"
        yield self.check_print_model_weights, "regression_svr_linear"
        yield self.check_print_model_weights, "regression_svr_linear_with_hashing"
        yield self.check_print_model_weights, "regression_svr_linear_with_scaling"
        yield self.check_print_model_weights, "regression_svr_linear_with_scaling_and_hashing"

    def check_summarize_results_argparse(self, use_ablation=False):
        """
        Check that we are setting up argument parsing correctly for `summarize_results`.

        We are not checking whether the summaries produced are accurate because we
        have separate tests for that.
        """
        # replace the _write_summary_file function that's called
        # by the main() in summarize_results with a mocked up version
        write_summary_file_mock = create_autospec(_write_summary_file)
        sr._write_summary_file = write_summary_file_mock

        # now call main with some arguments
        summary_file_name = other_dir / "summary_file"
        list_of_input_files = ["infile1", "infile2", "infile3"]
        sr_cmd_args = [str(summary_file_name)]
        sr_cmd_args.extend(list_of_input_files)
        if use_ablation:
            sr_cmd_args.append("--ablation")
        sr.main(argv=sr_cmd_args)

        # now check to make sure that _write_summary_file (or our mocked up version
        # of it) got the arguments that we passed
        positional_arguments, keyword_arguments = write_summary_file_mock.call_args
        self.assertEqual(positional_arguments[0], list_of_input_files)
        self.assertEqual(positional_arguments[1].name, str(summary_file_name))
        self.assertEqual(keyword_arguments["ablation"], int(use_ablation))

    def test_summarize_results_argparse(self):
        yield self.check_summarize_results_argparse, False
        yield self.check_summarize_results_argparse, True

    def test_plot_learning_curves_argparse(self):
        # A utility function to check that we are setting up argument parsing
        # correctly for plot_learning_curves. We are not checking whether the learning
        # curves produced are accurate because we have separate tests for that.

        # replace the _generate_learning_curve_plots function that's called
        # by the main() in plot_learning_curves with a mocked up version
        generate_learning_curve_plots_mock = create_autospec(generate_learning_curve_plots)
        plc.generate_learning_curve_plots = generate_learning_curve_plots_mock

        # now call main with some arguments
        summary_file_name = other_dir / "sample_learning_curve_summary.tsv"
        experiment_name = "sample_learning_curve"
        plc_cmd_args = [str(summary_file_name), str(other_dir)]
        plc.main(argv=plc_cmd_args)

        # now check to make sure that _generate_learning_curve_plots (or our mocked up version
        # of it) got the arguments that we passed
        positional_arguments, _ = generate_learning_curve_plots_mock.call_args
        self.assertEqual(positional_arguments[0], experiment_name)
        self.assertEqual(positional_arguments[1], str(other_dir))
        self.assertEqual(positional_arguments[2], str(summary_file_name))

    def test_plot_learning_curves_missing_file(self):
        summary_file_name = other_dir / "non_existent_summary.tsv"
        plc_cmd_args = [str(summary_file_name), str(other_dir)]
        with self.assertRaises(SystemExit):
            plc.main(argv=plc_cmd_args)

    def test_plot_learning_curves_create_output_directory(self):
        summary_file_name = other_dir / "sample_learning_curve_summary.tsv"
        output_dir_name = other_dir / "foobar"
        plc_cmd_args = [str(summary_file_name), str(output_dir_name)]
        plc.main(argv=plc_cmd_args)
        output_dir_name.exists()

    def check_run_experiments_argparse(
        self,
        multiple_config_files=False,
        n_ablated_features="1",
        keep_models=False,
        local=False,
        resume=False,
    ):
        """
        Check that we are setting up argument parsing correcly for `run_experiment`.

        We are not checking whether the results are correct because we have
        separate tests for that.
        """
        # replace the run_configuration function that's called
        # by the main() in run_experiment with a mocked up version
        run_configuration_mock = create_autospec(run_configuration)
        rex.run_configuration = run_configuration_mock

        # now call main with some arguments
        config_file1_name = other_dir / "config_file1"
        config_files = [str(config_file1_name)]
        rex_cmd_args = [str(config_file1_name)]
        if multiple_config_files:
            config_file2_name = other_dir / "config_file2"
            rex_cmd_args.extend([str(config_file2_name)])
            config_files.extend([str(config_file2_name)])

        if n_ablated_features != "all":
            rex_cmd_args.extend(["-a", str(n_ablated_features)])
        else:
            rex_cmd_args.append("-A")

        if keep_models:
            rex_cmd_args.append("-k")

        if resume:
            rex_cmd_args.append("-r")

        if local:
            rex_cmd_args.append("-l")
        else:
            machine_list = ['"foo.1.org"', '"x.test.com"', '"z.a.b.d"']
            rex_cmd_args.extend(["-m", ",".join(machine_list)])

        rex_cmd_args.extend(["-q", "foobar.q"])

        rex.main(argv=rex_cmd_args)

        # now check to make sure that run_configuration (or our mocked up version
        # of it) got the arguments that we passed
        positional_arguments, keyword_arguments = run_configuration_mock.call_args

        if multiple_config_files:
            self.assertEqual(positional_arguments[0], config_files[1])
        else:
            self.assertEqual(positional_arguments[0], str(config_file1_name))

        if n_ablated_features != "all":
            self.assertEqual(keyword_arguments["ablation"], int(n_ablated_features))
        else:
            self.assertEqual(keyword_arguments["ablation"], None)

        if local:
            self.assertEqual(keyword_arguments["local"], local)
        else:
            self.assertEqual(keyword_arguments["hosts"], machine_list)

        self.assertEqual(keyword_arguments["overwrite"], not keep_models)
        self.assertEqual(keyword_arguments["queue"], "foobar.q")
        self.assertEqual(keyword_arguments["resume"], resume)

    def test_run_experiment_argparse(self):
        for multiple_config_files, n_ablated_features, keep_models, local, resume in product(
            [True, False], ["2", "all"], [True, False], [True, False], [True, False]
        ):
            yield (
                self.check_run_experiments_argparse,
                multiple_config_files,
                n_ablated_features,
                keep_models,
                local,
                resume,
            )

    def check_filter_features_no_arff_argparse(
        self, extension, filter_type, label_col="y", id_col="id", inverse=False, quiet=False
    ):
        """
        Check that we are setting up argument parsing correctly for `filter_features`.

        We are checking for ALL file types except ARFF. We are not checking whether
        the results are correct because we have separate tests for that.
        """
        # replace the run_configuration function that's called
        # by the main() in filter_feature with a mocked up version
        reader_class = EXT_TO_READER[extension]
        writer_class = EXT_TO_WRITER[extension]

        # create some dummy input and output filenames
        infile = f"foo{extension}"
        outfile = f"bar{extension}"

        # create a simple featureset with actual ids, labels and features
        fs, _ = make_classification_data(num_labels=3, train_test_ratio=1.0)

        ff_cmd_args = ["-i", infile, "-o", outfile]

        if filter_type == "feature":
            if inverse:
                features_to_keep = ["f01", "f04", "f07", "f10"]
            else:
                features_to_keep = ["f02", "f03", "f05", "f06", "f08", "f09"]

            ff_cmd_args.append("-f")

            for f in features_to_keep:
                ff_cmd_args.append(f)

        elif filter_type == "id":
            if inverse:
                ids_to_keep = [f"EXAMPLE_{x}" for x in range(1, 100, 2)]
            else:
                ids_to_keep = [f"EXAMPLE_{x}" for x in range(2, 102, 2)]

            ff_cmd_args.append("-I")

            for idee in ids_to_keep:
                ff_cmd_args.append(idee)

        elif filter_type == "label":
            # any numeric labels will get converted to integers via
            # `safe_float` before they get passed to `FeatureSet.filter()`
            if inverse:
                label_values = ["0", "1"]
                labels_to_keep = [0, 1]
            else:
                label_values = ["2"]
                labels_to_keep = [2]

            ff_cmd_args.append("-L")

            for lbl in label_values:
                ff_cmd_args.append(lbl)

        ff_cmd_args.extend(["-l", label_col])
        ff_cmd_args.extend(["--id_col", id_col])

        if inverse:
            ff_cmd_args.append("--inverse")

        if quiet:
            ff_cmd_args.append("-q")

        # substitute mock methods for the three main methods that get called by
        # filter_features: the __init__() method of the appropriate reader,
        # FeatureSet.filter() and the __init__() method of the appropriate writer.
        # We also need to mock the read() and write() methods to prevent actual
        # reading and writing.
        with patch.object(
            reader_class, "__init__", autospec=True, return_value=None
        ) as read_init_mock, patch.object(
            reader_class, "read", autospec=True, return_value=fs
        ), patch.object(
            FeatureSet, "filter", autospec=True
        ) as filter_mock, patch.object(
            writer_class, "__init__", autospec=True, return_value=None
        ) as write_init_mock, patch.object(
            writer_class, "write", autospec=True
        ):
            ff.main(argv=ff_cmd_args)

            # get the various arguments from the three mocked up methods
            read_pos_arguments, read_kw_arguments = read_init_mock.call_args
            filter_pos_arguments, filter_kw_arguments = filter_mock.call_args
            write_pos_arguments, write_kw_arguments = write_init_mock.call_args

            # make sure that the arguments they got were the ones we specified
            self.assertEqual(read_pos_arguments[1], infile)
            self.assertEqual(read_kw_arguments["quiet"], quiet)
            self.assertEqual(read_kw_arguments["label_col"], label_col)
            self.assertEqual(read_kw_arguments["id_col"], id_col)

            self.assertEqual(write_pos_arguments[1], outfile)
            self.assertEqual(write_kw_arguments["quiet"], quiet)

            # Note that we cannot test the label_col column for the writer.
            # The reason is that it is set conditionally and those conditions
            # do not execute with mocking.
            self.assertEqual(filter_pos_arguments[0], fs)
            self.assertEqual(filter_kw_arguments["inverse"], inverse)

            if filter_type == "feature":
                self.assertEqual(filter_kw_arguments["features"], features_to_keep)
            elif filter_type == "id":
                self.assertEqual(filter_kw_arguments["ids"], ids_to_keep)
            elif filter_type == "label":
                self.assertEqual(filter_kw_arguments["labels"], labels_to_keep)

    def test_filter_features_no_arff_argparse(self):
        for extension, filter_type, id_col, label_col, inverse, quiet in product(
            [
                ".jsonlines",
                ".ndj",
                ".tsv",
                ".csv",
            ],
            ["feature", "id", "label"],
            ["id", "id_foo"],
            ["y", "foo"],
            [True, False],
            [True, False],
        ):
            yield (
                self.check_filter_features_no_arff_argparse,
                extension,
                filter_type,
                label_col,
                id_col,
                inverse,
                quiet,
            )

    def check_filter_features_arff_argparse(
        self, filter_type, label_col="y", id_col="id", inverse=False, quiet=False
    ):
        """
        Check that we are setting up argument parsing correctly for `filter_features`.

        We are only checking ARFF file types. We are not checking whether the results
        are correct because we have separate tests for that.
        """
        # replace the run_configuration function that's called
        # by the main() in filter_feature with a mocked up version
        writer_class = skll.data.writers.ARFFWriter

        # create some dummy input and output filenames
        infile = other_dir / "test_filter_features_input.arff"
        outfile = "bar.arff"

        # create a simple featureset with actual ids, labels and features
        fs, _ = make_classification_data(num_labels=3, train_test_ratio=1.0)

        writer = writer_class(infile, fs, label_col=label_col, id_col=id_col)
        writer.write()

        ff_cmd_args = ["-i", str(infile), "-o", outfile]

        if filter_type == "feature":
            if inverse:
                features_to_keep = ["f01", "f04", "f07", "f10"]
            else:
                features_to_keep = ["f02", "f03", "f05", "f06", "f08", "f09"]

            ff_cmd_args.append("-f")

            for f in features_to_keep:
                ff_cmd_args.append(f)

        elif filter_type == "id":
            if inverse:
                ids_to_keep = [f"EXAMPLE_{x}" for x in range(1, 100, 2)]
            else:
                ids_to_keep = [f"EXAMPLE_{x}" for x in range(2, 102, 2)]

            ff_cmd_args.append("-I")

            for idee in ids_to_keep:
                ff_cmd_args.append(idee)

        elif filter_type == "label":
            # any numeric labels will get converted to integers via
            # `safe_float` before they get passed to `FeatureSet.filter()`
            if inverse:
                label_values = ["0", "1"]
                labels_to_keep = [0, 1]
            else:
                label_values = ["2"]
                labels_to_keep = [2]

            ff_cmd_args.append("-L")

            for lbl in label_values:
                ff_cmd_args.append(lbl)

        ff_cmd_args.extend(["-l", label_col])
        ff_cmd_args.extend(["--id_col", id_col])

        if inverse:
            ff_cmd_args.append("--inverse")

        if quiet:
            ff_cmd_args.append("-q")

        # Substitute mock methods for the main methods that get called by
        # filter_features for arff files: FeatureSet.filter() and the __init__()
        # method of the appropriate writer.  We also need to mock the write()
        # method to prevent actual writing.
        with patch.object(FeatureSet, "filter", autospec=True) as filter_mock, patch.object(
            writer_class, "__init__", autospec=True, return_value=None
        ) as write_init_mock, patch.object(writer_class, "write", autospec=True):
            ff.main(argv=ff_cmd_args)

            # get the various arguments from the three mocked up methods
            filter_pos_arguments, filter_kw_arguments = filter_mock.call_args
            write_pos_arguments, write_kw_arguments = write_init_mock.call_args

            # make sure that the arguments they got were the ones we specified
            self.assertEqual(write_pos_arguments[1], outfile)
            self.assertEqual(write_kw_arguments["quiet"], quiet)

            # note that we cannot test the label_col column for the writer
            # the reason is that is set conditionally and those conditions
            # do not execute with mocking

            self.assertEqual(filter_pos_arguments[0], fs)
            self.assertEqual(filter_kw_arguments["inverse"], inverse)

            if filter_type == "feature":
                self.assertEqual(filter_kw_arguments["features"], features_to_keep)
            elif filter_type == "id":
                self.assertEqual(filter_kw_arguments["ids"], ids_to_keep)
            elif filter_type == "label":
                self.assertEqual(filter_kw_arguments["labels"], labels_to_keep)

    def test_filter_features_arff_argparse(self):
        for filter_type, label_col, id_col, inverse, quiet in product(
            ["feature", "id", "label"], ["y", "foo"], ["id", "id_foo"], [True, False], [True, False]
        ):
            yield (
                self.check_filter_features_arff_argparse,
                filter_type,
                label_col,
                id_col,
                inverse,
                quiet,
            )

    def check_filter_features_labels(self, extension, label_type):
        """Make sure that labels are correctly converted before filtering."""
        reader_class = EXT_TO_READER[extension]
        writer_class = EXT_TO_WRITER[extension]

        # create some dummy input and output filenames
        infile = other_dir / f"test_filter_features_labels_input_{label_type}{extension}"
        outfile = other_dir / f"test_filter_features_labels_output_{label_type}{extension}"

        # create a simple featureset with 4 labels, either string or integer
        if label_type == "integer":
            fs, _ = make_classification_data(num_labels=4, train_test_ratio=1.0)
        else:
            fs, _ = make_classification_data(
                num_labels=4, train_test_ratio=1.0, string_label_list=["a", "b", "c", "d"]
            )

        # write out this featuerset to disk
        writer = writer_class(infile, fs, quiet=True)
        writer.write()

        # set up the arguments for the `filter_features` call and
        # compute the IDs that we expect to be in the filtered output
        if label_type == "integer":
            ff_cmd_args = ["-i", str(infile), "-o", str(outfile), "-L", "1", "3", "-q"]
            expected_ids = fs.ids[np.logical_or(fs.labels == 1, fs.labels == 3)]
        else:
            ff_cmd_args = ["-i", str(infile), "-o", str(outfile), "-L", "c", "d", "-q"]
            expected_ids = fs.ids[np.logical_or(fs.labels == "c", fs.labels == "d")]

        # call `filter_features`
        ff.main(argv=ff_cmd_args)

        # read in the output file and check that the filtered IDs match
        filtered_fs = reader_class.for_path(outfile).read()
        assert_array_equal(filtered_fs.ids, expected_ids)

    def test_filter_features_labels(self):
        for extension, label_type in product(
            [".jsonlines", ".ndj", ".tsv", ".csv"], ["integer", "string"]
        ):
            yield self.check_filter_features_labels, extension, label_type

    def test_filter_features_libsvm_input_argparse(self):
        """Make sure filter_features exits when passing in input libsvm files."""
        ff_cmd_args = ["-f", "a", "b", "c", "-i", "foo.libsvm", "-o", "bar.csv"]
        with self.assertRaises(SystemExit):
            ff.main(argv=ff_cmd_args)

    def test_filter_features_libsvm_output_argparse(self):
        """Make sure filter_features exits when passing in output libsvm files."""
        ff_cmd_args = ["-f", "a", "b", "c", "i", "foo.csv", "-o", "bar.libsvm"]
        with self.assertRaises(SystemExit):
            ff.main(argv=ff_cmd_args)

    def test_filter_features_unknown_input_format(self):
        """Make sure that filter_features exits when passing in an unknown input file format."""
        ff_cmd_args = ["-f", "a", "b", "c", "-i", "foo.xxx", "-o", "bar.csv"]
        with self.assertRaises(SystemExit):
            ff.main(argv=ff_cmd_args)

    def test_filter_features_unknown_output_format(self):
        """Make sure that filter_features exits when passing in an unknown input file format."""
        ff_cmd_args = ["-f", "a", "b", "c", "-i", "foo.csv", "-o", "bar.xxx"]
        with self.assertRaises(SystemExit):
            ff.main(argv=ff_cmd_args)

    def check_filter_features_raises_system_exit(self, cmd_args):
        """
        Clean test output.

        Make test output cleaner for tests that check that `filter_features` exits
        with the specified arguments.
        """
        with self.assertRaises(SystemExit):
            ff.main(cmd_args)

    def test_filter_features_unmatched_formats(self):
        # Make sure filter_feature exits when the output file is in a different format
        for inext, outext in combinations([".arff", ".ndj", ".tsv", ".jsonlines", ".csv"], 2):
            ff_cmd_args = ["-i", f"foo{inext}", "-o", f"bar{outext}", "-f", "a", "b", "c"]
            yield self.check_filter_features_raises_system_exit, ff_cmd_args

    def check_join_features_argparse(self, extension, label_col="y", id_col="id", quiet=False):
        """
        Check that we are setting up argument parsing correctly for `join_features`.

        We are not checking whether the results are correct because we have separate
        tests for that.
        """
        # replace the run_configuration function that's called
        # by the main() in filter_feature with a mocked up version
        writer_class = EXT_TO_WRITER[extension]

        # create some dummy input and output filenames
        infile1 = other_dir / f"test_join_features1{extension}"
        infile2 = other_dir / f"test_join_features2{extension}"
        outfile = f"bar{extension}"

        # create a simple featureset with actual ids, labels and features
        fs1, _ = make_classification_data(num_labels=3, train_test_ratio=1.0, random_state=1234)
        fs2, _ = make_classification_data(
            num_labels=3, train_test_ratio=1.0, feature_prefix="g", random_state=5678
        )

        jf_cmd_args = [str(infile1), str(infile2), outfile]

        if extension in [".tsv", ".csv", ".arff"]:
            writer1 = writer_class(infile1, fs1, label_col=label_col, id_col=id_col)
            writer2 = writer_class(infile2, fs2, label_col=label_col, id_col=id_col)
            jf_cmd_args.extend(["-l", label_col])
            jf_cmd_args.extend(["--id_col", id_col])
        else:
            writer1 = writer_class(infile1, fs1)
            writer2 = writer_class(infile2, fs2)

        writer1.write()
        writer2.write()

        if quiet:
            jf_cmd_args.append("-q")

        # Substitute mock methods for the main methods that get called by
        # filter_features: FeatureSet.filter() and the __init__() method
        # of the appropriate writer. We also need to mock the write()
        # method to prevent actual writing.
        with patch.object(FeatureSet, "__add__", autospec=True) as add_mock, patch.object(
            writer_class, "__init__", autospec=True, return_value=None
        ) as write_init_mock, patch.object(writer_class, "write", autospec=True):
            jf.main(argv=jf_cmd_args)

            # get the various arguments from the three mocked up methods
            add_pos_arguments, _ = add_mock.call_args
            write_pos_arguments, write_kw_arguments = write_init_mock.call_args

            # make sure that the arguments they got were the ones we specified
            self.assertEqual(write_pos_arguments[1], outfile)
            self.assertEqual(write_kw_arguments["quiet"], quiet)

            # note that we cannot test the label_col column for the writer
            # the reason is that is set conditionally and those conditions
            # do not execute with mocking

            self.assertEqual(add_pos_arguments[0], fs1)
            self.assertEqual(add_pos_arguments[1], fs2)

    def test_join_features_argparse(self):
        for extension, label_col, id_col, quiet in product(
            [".jsonlines", ".ndj", ".tsv", ".csv", ".arff"],
            ["y", "foo"],
            ["id", "id_foo"],
            [True, False],
        ):
            yield self.check_join_features_argparse, extension, label_col, id_col, quiet

    def test_join_features_libsvm_input_argparse(self):
        """Make sure that join_features exits when passing in input libsvm files."""
        jf_cmd_args = ["foo.libsvm", "bar.libsvm", "baz.csv"]
        with self.assertRaises(SystemExit):
            jf.main(argv=jf_cmd_args)

    def test_join_features_libsvm_output_argparse(self):
        """Make sure that join_features exits when passing in output libsvm files."""
        jf_cmd_args = ["foo.csv", "bar.csv", "baz.libsvm"]
        with self.assertRaises(SystemExit):
            jf.main(argv=jf_cmd_args)

    def test_join_features_unknown_input_format(self):
        """Make that join_features exits when passing in an unknown input file format."""
        jf_cmd_args = ["foo.xxx", "bar.tsv", "baz.csv"]
        with self.assertRaises(SystemExit):
            jf.main(argv=jf_cmd_args)

    def test_join_features_unknown_output_format(self):
        """Make sure that join_features exits when passing in an unknown output file format."""
        jf_cmd_args = ["foo.csv", "bar.csv", "baz.xxx"]
        with self.assertRaises(SystemExit):
            jf.main(argv=jf_cmd_args)

    def check_join_features_raises_system_exit(self, cmd_args):
        """
        Clean test output for `join_features`.

        Make test output cleaner for tests that check that `join_features` exits
        with the specified arguments.
        """
        with self.assertRaises(SystemExit):
            jf.main(cmd_args)

    def test_join_features_unmatched_input_formats(self):
        # Make sure that join_feature exits when the input files are in different formats
        for ext1, ext2 in combinations([".arff", ".ndj", ".tsv", ".jsonlines", ".csv"], 2):
            jf_cmd_args = [f"foo{ext1}", f"bar{ext2}", f"baz{ext1}"]
            yield self.check_join_features_raises_system_exit, jf_cmd_args

    def test_join_features_unmatched_output_format(self):
        # Make sure join_features exits when the output file is in a different format
        for ext1, ext2 in combinations([".arff", ".ndj", ".tsv", ".jsonlines", ".csv"], 2):
            jf_cmd_args = [f"foo{ext1}", f"bar{ext1}", f"baz{ext2}"]
            yield self.check_join_features_raises_system_exit, jf_cmd_args

    def test_filter_features_with_drop_blanks(self):
        """Test `filter_features` with CSV & TSV readers using `drop_blanks` option."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, np.nan, 5, 6],
                "B": [5, 9, np.nan, 2, 9, 1],
                "C": [1.0, 1.0, 1.0, 1.0, 1.0, 1.1],
                "L": [1, 2, 1, 2, 1, 2],
            }
        )

        # create the expected results
        df_expected = df.copy().dropna().reset_index(drop=True)
        df_expected["id"] = [f"EXAMPLE_{i}" for i in range(4)]
        df_expected = df_expected[["A", "B", "C", "id", "L"]]

        csv_infile = other_dir / "features" / "features_drop_blanks.csv"
        tsv_infile = other_dir / "features" / "features_drop_blanks.tsv"

        csv_outfile = other_dir / "features" / "features_drop_blanks_out.csv"
        tsv_outfile = other_dir / "features" / "features_drop_blanks_out.tsv"

        df.to_csv(csv_infile, index=False)
        df.to_csv(tsv_infile, index=False, sep="\t")

        filter_features_csv_cmd = [
            "-i",
            str(csv_infile),
            "-o",
            str(csv_outfile),
            "-l",
            "L",
            "--drop_blanks",
        ]
        filter_features_tsv_cmd = [
            "-i",
            str(tsv_infile),
            "-o",
            str(tsv_outfile),
            "-l",
            "L",
            "--drop_blanks",
        ]

        ff.main(filter_features_csv_cmd)
        ff.main(filter_features_tsv_cmd)

        df_csv_output = pd.read_csv(csv_outfile)
        df_tsv_output = pd.read_csv(tsv_outfile, sep="\t")

        assert_frame_equal(df_csv_output, df_expected)
        assert_frame_equal(df_tsv_output, df_expected)

    def test_filter_features_with_drop_blanks_all_blanks_csv(self):
        """Test `filter_features` with CSV readers using `drop_blanks` and blanks in each row."""
        df = pd.DataFrame(
            {
                "A": [1, 2, np.nan, 4, 5, 6],
                "B": [5, 9, 7, 2, np.nan, 1],
                "C": [np.nan, 1.0, 1.0, np.nan, 1.0, 1.1],
                "L": [1, np.nan, 1, 2, 1, np.nan],
            }
        )

        csv_infile = other_dir / "features" / "features_drop_blanks_all_blanks.csv"

        df.to_csv(csv_infile, index=False)

        filter_features_csv_cmd = [
            "-i",
            str(csv_infile),
            "-o",
            "blah.csv",
            "-l",
            "L",
            "--drop_blanks",
        ]

        with self.assertRaises(ValueError):
            ff.main(filter_features_csv_cmd)

    def test_filter_features_with_drop_blanks_all_blanks_tsv(self):
        """Test `filter_features` with TSV readers using `drop_blanks` and blanks in each row."""
        df = pd.DataFrame(
            {
                "A": [1, 2, np.nan, 4, 5, 6],
                "B": [5, 9, 7, 2, np.nan, 1],
                "C": [np.nan, 1.0, 1.0, np.nan, 1.0, 1.1],
                "L": [1, np.nan, 1, 2, 1, np.nan],
            }
        )

        tsv_infile = other_dir / "features" / "features_drop_blanks_all_blanks.tsv"

        df.to_csv(tsv_infile, index=False, sep="\t")

        filter_features_tsv_cmd = [
            "-i",
            str(tsv_infile),
            "-o",
            "blah.tsv",
            "-l",
            "L",
            "--drop_blanks",
        ]

        with self.assertRaises(ValueError):
            ff.main(filter_features_tsv_cmd)

    def test_filter_features_with_replace_blanks_with(self):
        """Test `filter_features` with CSV & TSV readers using the `replace_blanks_with` option."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, np.nan, 5, 6],
                "B": [5, 9, np.nan, 2, 9, 1],
                "C": [1.0, 1.0, 1.0, 1.0, 1.0, 1.1],
                "L": [1, 2, 1, 2, 1, 2],
            }
        )

        # create the expected results
        df_expected = df.fillna(4.5).copy()
        df_expected["id"] = [f"EXAMPLE_{i}" for i in range(6)]
        df_expected = df_expected[["A", "B", "C", "id", "L"]]

        csv_infile = other_dir / "features" / "features_replace_blanks_with.csv"
        tsv_infile = other_dir / "features" / "features_replace_blanks_with.tsv"

        csv_outfile = other_dir / "features" / "features_replace_blanks_with_out.csv"
        tsv_outfile = other_dir / "features" / "features_replace_blanks_with_out.tsv"

        df.to_csv(csv_infile, index=False)
        df.to_csv(tsv_infile, index=False, sep="\t")

        filter_features_csv_cmd = [
            "-i",
            str(csv_infile),
            "-o",
            str(csv_outfile),
            "-l",
            "L",
            "--replace_blanks_with",
            "4.5",
        ]
        filter_features_tsv_cmd = [
            "-i",
            str(tsv_infile),
            "-o",
            str(tsv_outfile),
            "-l",
            "L",
            "--replace_blanks_with",
            "4.5",
        ]

        ff.main(filter_features_csv_cmd)
        ff.main(filter_features_tsv_cmd)

        df_csv_output = pd.read_csv(csv_outfile)
        df_tsv_output = pd.read_csv(tsv_outfile, sep="\t")

        assert_frame_equal(df_csv_output, df_expected)
        assert_frame_equal(df_tsv_output, df_expected)

    def test_filter_features_with_replace_blanks_with_and_drop_blanks_raises_error(self):
        df = pd.DataFrame(np.random.randn(5, 10))

        csv_infile = other_dir / "features" / "features_drop_and_replace_error.csv"
        csv_outfile = other_dir / "features" / "features_drop_and_replace_error_out.csv"

        df.to_csv(csv_infile, index=False)

        filter_features_csv_cmd = [
            "-i",
            str(csv_infile),
            "-o",
            str(csv_outfile),
            "--drop_blanks",
            "--replace_blanks_with",
            "4.5",
        ]

        with self.assertRaises(ValueError):
            ff.main(filter_features_csv_cmd)
