# License: BSD 3 clause
"""
Tests related to classification experiments.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

import csv
import itertools
import json
import re
import warnings
from glob import glob
from itertools import product
from os.path import join
from pathlib import Path

import numpy as np
from nose.tools import assert_almost_equal, assert_raises, eq_, ok_, raises
from numpy.testing import assert_array_equal, assert_raises_regex
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.utils import shuffle as sk_shuffle

from skll.config import parse_config_file
from skll.data import FeatureSet, NDJReader, NDJWriter
from skll.experiments import run_configuration
from skll.learner import Learner
from skll.learner.utils import (
    FilteredLeaveOneGroupOut,
    contiguous_ints_or_floats,
    train_and_score,
)
from skll.metrics import use_score_func
from skll.utils.constants import (
    CORRELATION_METRICS,
    KNOWN_DEFAULT_PARAM_GRIDS,
    PROBABILISTIC_METRICS,
    REGRESSION_ONLY_METRICS,
    UNWEIGHTED_KAPPA_METRICS,
    WEIGHTED_KAPPA_METRICS,
)
from tests import config_dir, other_dir, output_dir, test_dir, train_dir
from tests.utils import (
    fill_in_config_options,
    fill_in_config_paths_for_single_file,
    make_classification_data,
    make_regression_data,
    make_sparse_data,
    unlink,
)

_ALL_MODELS = list(KNOWN_DEFAULT_PARAM_GRIDS.keys())


def setup():
    for dir_path in [train_dir, test_dir, output_dir]:
        Path(dir_path).mkdir(exist_ok=True)


def tearDown():

    for path in [join(train_dir, 'train_single_file.jsonlines'),
                 join(train_dir, 'pos_label_train.jsonlines'),
                 join(test_dir, 'test_single_file.jsonlines'),
                 join(output_dir, 'rare_class_predictions.tsv'),
                 join(output_dir, 'float_class_predictions.tsv')]:
        unlink(path)

    for glob_pattern in [join(output_dir, 'train_test_single_file_*'),
                         join(output_dir, 'clf_metric_value_*'),
                         join(output_dir, 'pos_label_via_config*'),
                         join(output_dir, 'clf_metrics_objective_overlap*'),
                         join(output_dir, 'test_multinomialnb_loading*'),
                         join(output_dir, 'test_predict_return_and_write_predictions*')]:
        for path in glob(glob_pattern):
            unlink(path)

    config_files = [join(config_dir,
                         cfgname) for cfgname in ['test_single_file.cfg',
                                                  'test_single_file_saved_subset.cfg']]
    config_files.extend(glob(join(config_dir,
                                  'test_metric_values_for_classification_*.cfg')))
    config_files.extend(glob(join(config_dir,
                                  'test_fancy_pos_label*.cfg')))
    config_files.extend(glob(join(config_dir,
                                  'test_fancy_metrics_objective_overlap*.cfg')))

    for config_file in config_files:
        unlink(config_file)

    for file_name in ["metric_values_test.jsonlines",
                      "metric_values_train.jsonlines"]:
        unlink(Path(train_dir) / file_name)


def test_contiguous_int_or_float_labels():
    """
    Test that we can accurately detect contiguous int/float labels
    """
    eq_(contiguous_ints_or_floats([1, 2, 3, 4]), True)
    eq_(contiguous_ints_or_floats([0, 1]), True)
    eq_(contiguous_ints_or_floats([1.0, 2.0]), True)
    eq_(contiguous_ints_or_floats([0, 1.0]), True)
    eq_(contiguous_ints_or_floats([-2, -1, 0, 1, 2]), True)
    eq_(contiguous_ints_or_floats([1.0, 2.0, 3.0, 4.0]), True)
    eq_(contiguous_ints_or_floats([4, 5, 6]), True)
    eq_(contiguous_ints_or_floats([1, 2, 3, 4.0]), True)
    eq_(contiguous_ints_or_floats([-1, 1]), False)
    eq_(contiguous_ints_or_floats([2, 4, 6]), False)
    eq_(contiguous_ints_or_floats([3, 6, 11]), False)
    eq_(contiguous_ints_or_floats([-2.0, -1.0, 1.0, 2.0]), False)
    eq_(contiguous_ints_or_floats([1.0, 1.1, 1.2]), False)
    assert_raises(TypeError, contiguous_ints_or_floats, ['a', 'b', 'c'])
    assert_raises(TypeError, contiguous_ints_or_floats, np.array([1, 2, 3, 'a']))
    assert_raises(ValueError, contiguous_ints_or_floats, [])
    assert_raises(ValueError, contiguous_ints_or_floats, np.array([]))


def test_label_index_order():
    """
    Test that labels are sorted before creating the indices
    """
    train_fs, _ = make_classification_data()
    prng = np.random.RandomState(123456789)
    for (unique_label_list,
         correct_order) in zip([['B', 'C', 'A'],
                                [3, 1, 2],
                                [1.0, 2.5, 1.4],
                                [4, 5, 6],
                                [1.0, 3.0, 2.0, 4.0],
                                [1, 2, 3, 4.0]],
                               [['A', 'B', 'C'],
                                [1, 2, 3],
                                [1.0, 1.4, 2.5],
                                [4, 5, 6],
                                [1.0, 2.0, 3.0, 4.0],
                                [1.0, 2.0, 3.0, 4.0]]):
        labels = prng.choice(unique_label_list, size=len(train_fs))
        train_fs.labels = labels

        clf = Learner('LogisticRegression')
        clf.train(train_fs, grid_search=False)
        reverse_label_dict = {y: x for x, y in clf.label_dict.items()}
        label_order_from_dict = [reverse_label_dict[x] for x in range(len(unique_label_list))]

        eq_(clf.label_list, correct_order)
        eq_(label_order_from_dict, correct_order)


def check_label_index_order_with_pos_label(use_api=True):
    """
    Test that only binary labels are sorted to satisfy `pos_label`
    and that the sorting happens when `pos_label` is specified
    both via the API as well as the configuration file.
    """
    train_fs, _ = make_classification_data()
    prng = np.random.RandomState(123456789)
    for (unique_label_list,
         pos_label,
         correct_order) in zip([['B', 'C'],
                                ['B', 'A', 'C'],
                                [1, 2],
                                [3, 1, 2],
                                [1.0, 3.0, 2.0, 4.0],
                                [1.0, 2.0],
                                ['FRAG', 'NONE']],
                               ['B', 'A', 2, 1, 3.0, 1.0, 'FRAG'],
                               [['C', 'B'],
                                ['A', 'B', 'C'],
                                [1, 2],
                                [1, 2, 3],
                                [1.0, 2.0, 3.0, 4.0],
                                [2.0, 1.0],
                                ['NONE', 'FRAG']]):
        labels = prng.choice(unique_label_list, size=len(train_fs))
        train_fs.labels = labels

        # if we are using the API, create a learner instance directly
        # otherwise run a configuration file to train a learner
        if use_api:
            clf = Learner('LogisticRegression', pos_label=pos_label)
            clf.train(train_fs, grid_search=False)
        else:

            train_file = join(train_dir, 'pos_label_train.jsonlines')

            # write out the feature set we made to a file for use in config
            writer = NDJWriter.for_path(train_file, train_fs)
            writer.write()

            # get the config template
            config_template_path = join(config_dir, 'test_fancy.template.cfg')

            values_to_fill_dict = {'experiment_name': 'pos_label_via_config',
                                   'task': 'train',
                                   'learners': '["LogisticRegression"]',
                                   'train_file': train_file,
                                   'grid_search': 'false',
                                   'pos_label': str(pos_label),
                                   'logs': output_dir,
                                   'models': output_dir}

            config_path = fill_in_config_options(config_template_path,
                                                 values_to_fill_dict,
                                                 'pos_label')
            _ = run_configuration(config_path, local=True, quiet=True)
            clf = Learner.from_file(
                join(output_dir,
                     "pos_label_via_config_train_pos_label_train."
                     "jsonlines_LogisticRegression.model")
            )

        reverse_label_dict = {y: x for x, y in clf.label_dict.items()}
        label_order_from_dict = [reverse_label_dict[x] for x in range(len(unique_label_list))]

        eq_(clf.label_list, correct_order)
        eq_(label_order_from_dict, correct_order)
        if len(unique_label_list) != 2:
            eq_(clf.pos_label, None)
        else:
            eq_(clf.pos_label, pos_label)


def test_label_index_order_with_pos_label():
    yield check_label_index_order_with_pos_label, True
    yield check_label_index_order_with_pos_label, False


def check_binary_predictions_for_pos_label(label_list,
                                           use_pos_label=True,
                                           probability=True,
                                           use_api=True):
    """
    This tests whether binary predictions and probabilities
    obtained from SKLL (via the API as well as the config)
    match the predictions from scikit-learn on the same data
    for when `pos_label` is set and also when it is not set.
    """

    if label_list == ['B', 'C']:
        pos_label, other_label = 'B', 'C'
    elif label_list == [1, 2]:
        pos_label, other_label = 2, 1
    else:
        pos_label, other_label = 'FRAG', 'NONE'

    # create the mapping from class indices to class labels
    # manually so that they match our expectations for what
    # is supposed to happen when `pos_label` is specified
    # and when it is not
    if use_pos_label:
        index_label_dict = {0: other_label, 1: pos_label}
    else:
        sorted_label_list = sorted([pos_label, other_label])
        index_label_dict = dict(enumerate(sorted_label_list))

    # initialize a random number generator
    prng = np.random.RandomState(123456789)

    # create training and test featuresets whose labels we
    # generate manually using random sampling from [0, 1]
    # we convert [0, 1] to the appropriate labels so that
    # SKLL gets trained on those labels
    train_fs, test_fs = make_classification_data(num_examples=100)
    train_class_indices = prng.choice([0, 1], size=len(train_fs))
    train_fs.labels = np.array([index_label_dict[index] for index in train_class_indices])
    test_class_indices = prng.choice([0, 1], size=len(test_fs))
    test_fs.labels = np.array([index_label_dict[index] for index in test_class_indices])

    # train a scikit-learn LogisticRegression estimator
    # with the same fixed parameters as SKLL and using
    # the class indices we generated randomly above;
    # predict class indices/probabilities for the test set
    clf_sklearn = LogisticRegression(max_iter=1000,
                                     multi_class='auto',
                                     solver='liblinear',
                                     random_state=123456789)
    sklearn_features = train_fs.features
    clf_sklearn.fit(sklearn_features, train_class_indices)
    if probability:
        sklearn_predictions = clf_sklearn.predict_proba(test_fs.features)
    else:
        sklearn_predictions = clf_sklearn.predict(test_fs.features)

    # train a SKLL LogisticRegression learner on the same data;
    # predict the class indices/probabilities for the test set
    # we train+predict either using the API or using a config
    pos_label = pos_label if use_pos_label else None
    if use_api:
        clf_skll = Learner('LogisticRegression',
                           pos_label=pos_label,
                           probability=probability)
        clf_skll.train(train_fs, grid_search=False)
    else:
        train_file = join(train_dir, 'pos_label_train.jsonlines')

        # write out the feature set we made to a file for use in config
        writer = NDJWriter.for_path(train_file, train_fs)
        writer.write()

        # get the config template
        config_template_path = join(config_dir, 'test_fancy.template.cfg')

        values_to_fill_dict = {'experiment_name': 'pos_label_predict',
                               'task': 'train',
                               'learners': '["LogisticRegression"]',
                               'train_file': train_file,
                               'grid_search': 'false',
                               'pos_label': str(pos_label),
                               'logs': output_dir,
                               'models': output_dir,
                               'probability': 'true' if probability else 'false'}

        config_path = fill_in_config_options(config_template_path,
                                             values_to_fill_dict,
                                             'pos_label_predict',
                                             good_probability_option=True)

        _ = run_configuration(config_path, local=True, quiet=True)
        clf_skll = Learner.from_file(
            join(output_dir,
                 "pos_label_predict_train_pos_label_train.jsonlines"
                 "_LogisticRegression.model")
        )

    skll_predictions = clf_skll.predict(test_fs, class_labels=False)

    # the two sets of predicted class indices should be identical
    assert_array_equal(sklearn_predictions, skll_predictions)


def test_binary_predictions_for_pos_label():

    for (unique_label_list,
         use_pos_label,
         probability,
         use_api) in product([['B', 'C'],
                              [1, 2],
                              ['FRAG', 'NONE']],
                             [True, False],
                             [True, False],
                             [True, False]):

        yield (check_binary_predictions_for_pos_label,
               unique_label_list,
               use_pos_label,
               probability,
               use_api)


def check_predict(model, use_feature_hashing=False):
    """
    This tests whether predict task runs and generates the same
    number of predictions as samples in the test set. The specified
    model indicates whether to generate random regression
    or classification data.
    """

    # create the random data for the given model
    if model._estimator_type == 'regressor':
        train_fs, test_fs, _ = \
            make_regression_data(use_feature_hashing=use_feature_hashing,
                                 feature_bins=5)
    # feature hashing will not work for Naive Bayes since it requires
    # non-negative feature values
    elif model.__name__ == 'MultinomialNB':
        train_fs, test_fs = \
            make_classification_data(use_feature_hashing=False,
                                     non_negative=True)
    else:
        train_fs, test_fs = \
            make_classification_data(use_feature_hashing=use_feature_hashing,
                                     feature_bins=25)

    # create the learner with the specified model
    learner = Learner(model.__name__)

    # now train the learner on the training data and use feature hashing when
    # specified and when we are not using a Naive Bayes model
    learner.train(train_fs, grid_search=False)

    # now make predictions on the test set
    predictions = learner.predict(test_fs)

    # make sure we have the same number of outputs as the
    # number of test set samples
    eq_(len(predictions), test_fs.features.shape[0])


def test_default_param_grids_no_duplicates():
    """
    Verify that the default parameter grids don't contain duplicate values.
    """
    for learner, param_dict in KNOWN_DEFAULT_PARAM_GRIDS.items():
        for param_name, values in param_dict.items():
            assert(len(set(values)) == len(values))


# the runner function for the prediction tests
def test_predict():
    for model, use_feature_hashing in \
            itertools.product(_ALL_MODELS, [True, False]):
        yield check_predict, model, use_feature_hashing


# test predictions when both the model and the data use DictVectorizers
def test_predict_dict_dict():
    train_file = join(other_dir, 'examples_train.jsonlines')
    test_file = join(other_dir, 'examples_test.jsonlines')
    train_fs = NDJReader.for_path(train_file).read()
    test_fs = NDJReader.for_path(test_file).read()
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False)
    predictions = learner.predict(test_fs)
    eq_(len(predictions), test_fs.features.shape[0])


# test predictions when both the model and the data use FeatureHashers
# and the same number of bins
def test_predict_hasher_hasher_same_bins():
    train_file = join(other_dir, 'examples_train.jsonlines')
    test_file = join(other_dir, 'examples_test.jsonlines')
    train_fs = NDJReader.for_path(train_file, feature_hasher=True, num_features=3).read()
    test_fs = NDJReader.for_path(test_file, feature_hasher=True, num_features=3).read()
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False)
    predictions = learner.predict(test_fs)
    eq_(len(predictions), test_fs.features.shape[0])


# test predictions when both the model and the data use FeatureHashers
# but different number of bins
@raises(RuntimeError)
def test_predict_hasher_hasher_different_bins():
    train_file = join(other_dir, 'examples_train.jsonlines')
    test_file = join(other_dir, 'examples_test.jsonlines')
    train_fs = NDJReader.for_path(train_file, feature_hasher=True, num_features=3).read()
    test_fs = NDJReader.for_path(test_file, feature_hasher=True, num_features=2).read()
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False)
    _ = learner.predict(test_fs)


# test predictions when model uses a FeatureHasher and data
# uses a DictVectorizer
def test_predict_hasher_dict():
    train_file = join(other_dir, 'examples_train.jsonlines')
    test_file = join(other_dir, 'examples_test.jsonlines')
    train_fs = NDJReader.for_path(train_file, feature_hasher=True, num_features=3).read()
    test_fs = NDJReader.for_path(test_file).read()
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False)
    predictions = learner.predict(test_fs)
    eq_(len(predictions), test_fs.features.shape[0])


# test predictions when model uses a DictVectorizer and data
# uses a FeatureHasher
@raises(RuntimeError)
def test_predict_dict_hasher():
    train_file = join(other_dir, 'examples_train.jsonlines')
    test_file = join(other_dir, 'examples_test.jsonlines')
    train_fs = NDJReader.for_path(train_file).read()
    test_fs = NDJReader.for_path(test_file, feature_hasher=True, num_features=3).read()
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False)
    _ = learner.predict(test_fs)


# the function to create data with rare labels for cross-validation
def make_rare_class_data():
    """
    We want to create data that has five instances per class, for three labels
    and for each instance within the group of 5, there's only a single feature
    firing
    """

    ids = [f'EXAMPLE_{n}' for n in range(1, 16)]
    y = [0] * 5 + [1] * 5 + [2] * 5
    X = np.vstack([np.identity(5), np.identity(5), np.identity(5)])
    feature_names = [f'f{i}' for i in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    return FeatureSet('rare-class', ids, features=features, labels=y)


def test_rare_class():
    """
    Test cross-validation when some labels are very rare
    """

    rare_class_fs = make_rare_class_data()
    prediction_prefix = join(output_dir, 'rare_class')
    learner = Learner('LogisticRegression')
    learner.cross_validate(rare_class_fs,
                           grid_objective='unweighted_kappa',
                           prediction_prefix=prediction_prefix)

    with open(f'{prediction_prefix}_predictions.tsv') as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        pred = [row[1] for row in reader]

        eq_(len(pred), 15)


def check_sparse_predict(learner_name, expected_score, use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(
        use_feature_hashing=use_feature_hashing)

    # train the given classifier on the training
    # data and evalute on the testing data
    learner = Learner(learner_name)
    learner.train(train_fs, grid_search=False)
    test_score = learner.evaluate(test_fs)[1]
    assert_almost_equal(test_score, expected_score)


def test_sparse_predict():
    for learner_name, expected_scores in zip(['LogisticRegression',
                                              'DecisionTreeClassifier',
                                              'RandomForestClassifier',
                                              'AdaBoostClassifier',
                                              'MultinomialNB',
                                              'KNeighborsClassifier',
                                              'RidgeClassifier',
                                              'MLPClassifier'],
                                             [(0.45, 0.52), (0.52, 0.5),
                                              (0.48, 0.5), (0.49, 0.5),
                                              (0.43, 0), (0.53, 0.57),
                                              (0.45, 0.52), (0.5, 0.49)]):
        yield check_sparse_predict, learner_name, expected_scores[0], False
        if learner_name != 'MultinomialNB':
            yield check_sparse_predict, learner_name, expected_scores[1], True


def test_mlp_classification():
    train_fs, test_fs = make_classification_data(num_examples=600,
                                                 train_test_ratio=0.8,
                                                 num_labels=3,
                                                 num_features=5)

    # train an MLPCLassifier on the training data and evalute on the
    # testing data
    learner = Learner('MLPClassifier')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        learner.train(train_fs, grid_search=False)

    # now generate the predictions on the test set
    predictions = learner.predict(test_fs)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated
    accuracy = accuracy_score(predictions, test_fs.labels)
    assert_almost_equal(accuracy, 0.858, places=3)


def check_sparse_predict_sampler(use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(
        use_feature_hashing=use_feature_hashing)

    if use_feature_hashing:
        sampler = 'RBFSampler'
        sampler_parameters = {"gamma": 1.0, "n_components": 50}
    else:
        sampler = 'Nystroem'
        sampler_parameters = {"gamma": 1.0, "n_components": 50,
                              "kernel": 'rbf'}

    learner = Learner('LogisticRegression',
                      sampler=sampler,
                      sampler_kwargs=sampler_parameters)

    learner.train(train_fs, grid_search=False)
    test_score = learner.evaluate(test_fs)[1]

    expected_score = 0.48 if use_feature_hashing else 0.53
    assert_almost_equal(test_score, expected_score)


def check_dummy_classifier_predict(model_args, train_labels, expected_output):

    # create hard-coded featuresets based with known labels
    train_fs = FeatureSet('classification_train',
                          [f'TrainExample{i}' for i in range(20)],
                          labels=train_labels,
                          features=[{"feature": i} for i in range(20)])

    test_fs = FeatureSet('classification_test',
                         [f'TestExample{i}' for i in range(10)],
                         features=[{"feature": i} for i in range(20, 30)])

    # Ensure predictions are as expectedfor the given strategy
    learner = Learner('DummyClassifier', model_kwargs=model_args)
    learner.train(train_fs, grid_search=False)
    predictions = learner.predict(test_fs)
    eq_(np.array_equal(expected_output, predictions), True)


def test_dummy_classifier_predict():

    # create a known set of labels
    train_labels = ([0] * 14) + ([1] * 6)
    for (model_args, expected_output) in zip([{"strategy": "stratified"},
                                              {"strategy": "most_frequent"},
                                              {"strategy": "constant", "constant": 1}],
                                             [np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 0]),
                                              np.zeros(10),
                                              np.ones(10) * 1]):
        yield check_dummy_classifier_predict, model_args, train_labels, expected_output


def test_sparse_predict_sampler():
    yield check_sparse_predict_sampler, False
    yield check_sparse_predict_sampler, True


def make_single_file_featureset_data():
    """
    Write a training file and a test file for tests that check whether
    specifying train_file and test_file actually works.
    """
    train_fs, test_fs = make_classification_data(num_examples=600,
                                                 train_test_ratio=0.8,
                                                 num_labels=2,
                                                 num_features=3,
                                                 non_negative=False)

    # Write training feature set to a file
    train_path = join(train_dir, 'train_single_file.jsonlines')
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = join(test_dir, 'test_single_file.jsonlines')
    writer = NDJWriter(test_path, test_fs)
    writer.write()

    # Also write another test feature set that has fewer features than the training set
    test_fs.filter(features=['f01', 'f02'])
    test_path = join(test_dir, 'test_single_file_subset.jsonlines')
    writer = NDJWriter(test_path, test_fs)
    writer.write()


def test_train_file_test_file():
    """
    Test that train_file and test_file experiments work
    """
    # Create data files
    make_single_file_featureset_data()

    # Run experiment
    config_path = fill_in_config_paths_for_single_file(join(config_dir,
                                                            "test_single_file"
                                                            ".template.cfg"),
                                                       join(train_dir,
                                                            'train_single_file'
                                                            '.jsonlines'),
                                                       join(test_dir,
                                                            'test_single_file.'
                                                            'jsonlines'))
    run_configuration(config_path, quiet=True)

    # Check results for objective functions ["accuracy", "f1", "f05"]

    # objective function accuracy
    with open(join(output_dir,
                   'train_test_single_file_train_train_single_file.jsonlines'
                   '_test_test_single_file.jsonlines_RandomForestClassifier'
                   '_accuracy.results.json')) as f:
        result_dict = json.load(f)[0]
    assert_almost_equal(result_dict['score'], 0.95)

    # objective function f1
    with open(join(output_dir,
                   'train_test_single_file_train_train_single_file.jsonlines'
                   '_test_test_single_file.jsonlines_RandomForestClassifier'
                   '_f1.results.json')) as f:
        result_dict = json.load(f)[0]
    assert_almost_equal(result_dict['score'], 0.9491525423728813)

    # objective function f05
    with open(join(output_dir,
                   'train_test_single_file_train_train_single_file.jsonlines'
                   '_test_test_single_file.jsonlines_RandomForestClassifier'
                   '_f05.results.json')) as f:
        result_dict = json.load(f)[0]
    assert_almost_equal(result_dict['score'], 0.9302325581395348)


def test_predict_on_subset_with_existing_model():
    """
    Test generating predictions on subset with existing model
    """
    # Create data files
    make_single_file_featureset_data()

    # train and save a model on the training file
    train_fs = NDJReader.for_path(join(train_dir, 'train_single_file.jsonlines')).read()
    learner = Learner('RandomForestClassifier')
    learner.train(train_fs, grid_search=True, grid_objective="accuracy")
    model_filename = join(output_dir,
                          'train_test_single_file_train_train_single_file.'
                          'jsonlines_test_test_single_file_subset.jsonlines'
                          '_RandomForestClassifier.model')

    learner.save(model_filename)

    # Run experiment
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_single_file_saved_subset.template.cfg"),
        join(train_dir, 'train_single_file.jsonlines'),
        join(test_dir, 'test_single_file_subset.jsonlines')
    )
    run_configuration(config_path, quiet=True, overwrite=False)

    # Check results
    with open(join(output_dir,
                   'train_test_single_file_train_train_single_file.jsonlines'
                   '_test_test_single_file_subset.jsonlines_'
                   'RandomForestClassifier.results.json')) as f:
        result_dict = json.load(f)[0]
    assert_almost_equal(result_dict['accuracy'], 0.7333333)


def test_train_file_test_file_ablation():
    """
    Test that specifying ablation with train and test file is ignored
    """
    # Create data files
    make_single_file_featureset_data()

    # Run experiment
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_single_file.template.cfg"),
        join(train_dir, 'train_single_file.jsonlines'),
        join(test_dir, 'test_single_file.jsonlines')
    )
    run_configuration(config_path, quiet=True, ablation=None)

    # check that we see the message that ablation was ignored in the experiment log
    # Check experiment log output
    with open(join(output_dir, 'train_test_single_file.log')) as f:
        cv_file_pattern = re.compile(r'Not enough featuresets for ablation. '
                                     r'Ignoring.')
        matches = re.findall(cv_file_pattern, f.read())
        eq_(len(matches), 1)


@raises(ValueError)
def test_train_file_and_train_directory():
    """
    Test that train_file + train_directory = ValueError
    """
    # Run experiment
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_single_file.template.cfg"),
        join(train_dir, 'train_single_file.jsonlines'),
        join(test_dir, 'test_single_file.jsonlines'),
        train_directory='foo'
    )
    parse_config_file(config_path)


@raises(ValueError)
def test_test_file_and_test_directory():
    """
    Test that test_file + test_directory = ValueError
    """
    # Run experiment
    config_path = fill_in_config_paths_for_single_file(
        join(config_dir, "test_single_file.template.cfg"),
        join(train_dir, 'train_single_file.jsonlines'),
        join(test_dir, 'test_single_file.jsonlines'),
        test_directory='foo'
    )
    parse_config_file(config_path)


def check_adaboost_predict(base_estimator, algorithm, expected_score):
    train_fs, test_fs = make_sparse_data()

    # train an AdaBoostClassifier on the training data and evalute on the
    # testing data
    learner = Learner('AdaBoostClassifier', model_kwargs={'base_estimator': base_estimator,
                                                          'algorithm': algorithm})
    learner.train(train_fs, grid_search=False)
    test_score = learner.evaluate(test_fs)[1]
    assert_almost_equal(test_score, expected_score)


def test_adaboost_predict():
    for base_estimator_name, algorithm, expected_score in zip(['MultinomialNB',
                                                               'DecisionTreeClassifier',
                                                               'SGDClassifier',
                                                               'SVC'],
                                                              ['SAMME.R', 'SAMME.R',
                                                               'SAMME', 'SAMME'],
                                                              [0.46, 0.52, 0.46, 0.5]):
        yield check_adaboost_predict, base_estimator_name, algorithm, expected_score


def check_results_with_unseen_labels(res, n_labels, new_label_list):
    (confusion_matrix,
     score,
     result_dict,
     model_params,
     grid_score,
     additional_scores) = res

    # check that the new label is included into the results
    for output in [confusion_matrix, result_dict]:
        eq_(len(output), n_labels)

    # check that any additional metrics are zero
    eq_(additional_scores, {})

    # check that all metrics for new label are 0
    for label in new_label_list:
        for metric in ['Precision', 'Recall', 'F-measure']:
            eq_(result_dict[label][metric], 0)


def test_new_labels_in_test_set():
    """
    Test classification experiment with an unseen label in the test set.
    """
    train_fs, test_fs = make_classification_data(num_labels=3,
                                                 train_test_ratio=0.8)
    # add new labels to the test set
    test_fs.labels[-3:] = 3

    learner = Learner('SVC')
    learner.train(train_fs, grid_search=False)
    res = learner.evaluate(test_fs)
    yield check_results_with_unseen_labels, res, 4, [3]
    yield assert_almost_equal, res[1], 0.7


def test_new_labels_in_test_set_change_order():
    """
    Test classification with an unseen label in the test set when the
    new label falls between the existing labels.
    """
    train_fs, test_fs = make_classification_data(num_labels=3,
                                                 train_test_ratio=0.8)
    # change train labels to create a gap
    train_fs.labels = train_fs.labels * 10
    # add new test labels
    test_fs.labels = test_fs.labels * 10
    test_fs.labels[-3:] = 15

    learner = Learner('SVC')
    learner.train(train_fs, grid_search=False)
    res = learner.evaluate(test_fs)
    yield check_results_with_unseen_labels, res, 4, [15]
    yield assert_almost_equal, res[1], 0.7


def test_all_new_labels_in_test():
    """
    Test classification with all labels in test set unseen
    """
    train_fs, test_fs = make_classification_data(num_labels=3,
                                                 train_test_ratio=0.8)
    # change all test labels
    test_fs.labels = test_fs.labels + 3

    learner = Learner('SVC')
    learner.train(train_fs, grid_search=False)
    res = learner.evaluate(test_fs)
    yield check_results_with_unseen_labels, res, 6, [3, 4, 5]
    yield assert_almost_equal, res[1], 0


# the function to create data with labels that look like floats
# that are either encoded as strings or not depending on the
# keyword argument
def make_float_class_data(labels_as_strings=False):
    """
    We want to create data that has labels that look like
    floats to make sure they are preserved correctly
    """

    ids = [f'EXAMPLE_{n}' for n in range(1, 76)]
    y = [1.2] * 25 + [1.5] * 25 + [1.8] * 25
    if labels_as_strings:
        y = list(map(str, y))
    X = np.vstack([np.identity(25), np.identity(25), np.identity(25)])
    feature_names = [f'f{i}' for i in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    return FeatureSet('float-classes', ids, features=features, labels=y)


def test_xval_float_classes_as_strings():
    """
    Test that classification with float labels encoded as strings works
    """

    float_class_fs = make_float_class_data(labels_as_strings=True)
    prediction_prefix = join(output_dir, 'float_class')
    learner = Learner('LogisticRegression')
    learner.cross_validate(float_class_fs,
                           grid_search=True,
                           grid_objective='accuracy',
                           prediction_prefix=prediction_prefix)

    with open(f'{prediction_prefix}_predictions.tsv') as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        pred = [row[1] for row in reader]
        for p in pred:
            assert p in ['1.2', '1.5', '1.8']


@raises(ValueError)
def check_bad_xval_float_classes(do_stratified_xval):

    float_class_fs = make_float_class_data()
    prediction_prefix = join(output_dir, 'float_class')
    learner = Learner('LogisticRegression')
    learner.cross_validate(float_class_fs,
                           stratified=do_stratified_xval,
                           grid_search=True,
                           grid_objective='accuracy',
                           prediction_prefix=prediction_prefix)


def test_bad_xval_float_classes():

    yield check_bad_xval_float_classes, True
    yield check_bad_xval_float_classes, False


def check_train_and_score_function(model_type):
    """
    Check that the _train_and_score() function works as expected
    """

    # create train and test data
    (train_fs,
     test_fs) = make_classification_data(num_examples=500,
                                         train_test_ratio=0.7,
                                         num_features=5,
                                         use_feature_hashing=False,
                                         non_negative=True)

    # call _train_and_score() on this data
    estimator_name = 'LogisticRegression' if model_type == 'classifier' else 'Ridge'
    metric = 'accuracy' if model_type == 'classifier' else 'pearson'
    learner1 = Learner(estimator_name)
    train_score1, test_score1 = train_and_score(learner1, train_fs, test_fs, metric)

    # this should yield identical results when training another instance
    # of the same learner without grid search and shuffling and evaluating
    # that instance on the train and the test set
    learner2 = Learner(estimator_name)
    learner2.train(train_fs, grid_search=False, shuffle=False)
    train_score2 = learner2.evaluate(train_fs, output_metrics=[metric])[-1][metric]
    test_score2 = learner2.evaluate(test_fs, output_metrics=[metric])[-1][metric]

    eq_(train_score1, train_score2)
    eq_(test_score1, test_score2)


def test_train_and_score_function():
    yield check_train_and_score_function, 'classifier'
    yield check_train_and_score_function, 'regressor'


@raises(ValueError)
def check_learner_api_grid_search_no_objective(task='train'):

    (train_fs,
     test_fs) = make_classification_data(num_examples=500,
                                         train_test_ratio=0.7,
                                         num_features=5,
                                         use_feature_hashing=False,
                                         non_negative=True)
    learner = Learner('LogisticRegression')
    if task == 'train':
        _ = learner.train(train_fs)
    else:
        _ = learner.cross_validate(train_fs)


def test_learner_api_grid_search_no_objective():
    yield check_learner_api_grid_search_no_objective, 'train'
    yield check_learner_api_grid_search_no_objective, 'cross_validate'


def test_learner_api_load_into_existing_instance():
    """
    Check that `Learner.load()` works as expected
    """

    # create a LinearSVC instance and train it on some data
    learner1 = Learner('LinearSVC')
    (train_fs,
     test_fs) = make_classification_data(num_examples=200,
                                         num_features=5,
                                         use_feature_hashing=False,
                                         non_negative=True)
    learner1.train(train_fs, grid_search=False)

    # now use `load()` to replace the existing instance with a
    # different saved learner
    other_model_file = join(other_dir, 'test_load_saved_model.model')
    learner1.load(other_model_file)

    # now load the saved model into another instance using the class method
    # `from_file()`
    learner2 = Learner.from_file(other_model_file)

    # check that the two instances are now basically the same
    eq_(learner1.model_type, learner2.model_type)
    eq_(learner1.model_params, learner2.model_params)
    eq_(learner1.model_kwargs, learner2.model_kwargs)


@raises(ValueError)
def test_hashing_for_multinomialNB():
    (train_fs, _) = make_classification_data(num_examples=200,
                                             use_feature_hashing=True)
    learner = Learner('MultinomialNB', sampler='RBFSampler')
    learner.train(train_fs, grid_search=False)


@raises(ValueError)
def test_sampling_for_multinomialNB():
    (train_fs, _) = make_classification_data(num_examples=200)
    learner = Learner('MultinomialNB', sampler='RBFSampler')
    learner.train(train_fs, grid_search=False)


@raises(ValueError)
def check_invalid_classification_grid_objective(learner, grid_objective, label_array):
    """
    Checks that an invalid classification objective raises an exception
    """

    # initialize a random number generator
    prng = np.random.RandomState(123456789)

    # make a feature set
    train_fs, _ = make_classification_data()

    # generate the labels by randomly sampling repeatedly from
    # the given label array
    train_fs.labels = prng.choice(label_array, size=len(train_fs))

    clf = Learner(learner)
    clf.train(train_fs, grid_objective=grid_objective)


def test_invalid_classification_grid_objective():

    for (learner,
         (label_array,
          bad_objectives)) in product(['AdaBoostClassifier', 'DecisionTreeClassifier',
                                       'GradientBoostingClassifier', 'KNeighborsClassifier',
                                       'MLPClassifier', 'MultinomialNB',
                                       'RandomForestClassifier', 'LogisticRegression',
                                       'LinearSVC', 'SVC', 'SGDClassifier'],
                                      zip([np.array(['A', 'B', 'C']),
                                           np.array([2, 4, 6]),
                                           np.array(['yes', 'no']),
                                           np.array([1, 2, 4.0]),
                                           np.array(['A', 'B', 1, 2])],
                                          [CORRELATION_METRICS | REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS,
                                           REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS,
                                           CORRELATION_METRICS | REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS,
                                           REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS,
                                           CORRELATION_METRICS | REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS])):

        # check each bad objective
        for metric in bad_objectives:
            yield check_invalid_classification_grid_objective, learner, metric, label_array


@raises(ValueError)
def check_invalid_classification_metric(learner,
                                        metric,
                                        label_array,
                                        by_itself=True):
    """
    Checks that an invalid classification metric raises an exception
    """
    # initialize a random number generator
    prng = np.random.RandomState(123456789)

    # make a feature set
    train_fs, test_fs = make_classification_data()

    # generate the labels for the two sets by randomly sampling
    # repeatedly from the given label array
    train_fs.labels = prng.choice(label_array, size=len(train_fs))
    test_fs.labels = prng.choice(label_array, size=len(test_fs))

    # instantiate a learner
    clf = Learner(learner)
    clf.train(train_fs, grid_search=False)
    output_metrics = [metric] if by_itself else ['accuracy', metric]
    clf.evaluate(test_fs, output_metrics=output_metrics)


def test_invalid_classification_metric():
    for (learner,
         (label_array,
          bad_objectives)) in product(['AdaBoostClassifier', 'DecisionTreeClassifier',
                                       'GradientBoostingClassifier', 'KNeighborsClassifier',
                                       'MLPClassifier', 'MultinomialNB',
                                       'RandomForestClassifier', 'LogisticRegression',
                                       'LinearSVC', 'SVC', 'SGDClassifier'],
                                      zip([np.array(['A', 'B', 'C']),
                                           np.array([2, 4, 6]),
                                           np.array(['yes', 'no']),
                                           np.array([1, 2, 4.0]),
                                           np.array(['A', 'B', 1, 2])],
                                          [CORRELATION_METRICS | REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS,
                                           REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS,
                                           CORRELATION_METRICS | REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS,
                                           REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS,
                                           CORRELATION_METRICS | REGRESSION_ONLY_METRICS | WEIGHTED_KAPPA_METRICS])):

        # check each bad objective
        for metric in bad_objectives:
            yield check_invalid_classification_metric, learner, metric, label_array, True
            yield check_invalid_classification_metric, learner, metric, label_array, False


def check_objective_values_for_classification(metric_name,  # noqa: C901
                                              label_array,
                                              use_probabilities):

    # instantiate a random number generator
    prng = np.random.RandomState(123456789)

    # create our training set
    train_fs, _ = make_classification_data(num_examples=200)

    # create our labels by repeatedly sampling from the given label array
    train_fs.labels = prng.choice(label_array, size=len(train_fs))

    # get the label type
    label_type = train_fs.labels.dtype.type

    # create a dictionary of folds that assign half the IDs to one fold
    # and the other half to the second fold
    folds_dict = dict(zip(train_fs.ids, itertools.cycle([0, 1])))

    # instantiate our logistic regression learner and run grid search
    # using the folds dictionary
    clf = Learner('LogisticRegression', probability=use_probabilities)
    _, grid_search_results = clf.train(train_fs,
                                       grid_objective=metric_name,
                                       grid_search_folds=folds_dict,
                                       grid_jobs=1,
                                       param_grid=[{'C': [1.0, 10.0]}])

    # load in the featureset to get the feature matrix (X) and the
    # labels array (y) we need to pass to scikit-learn; we also need
    # to shuffle after we load in the matrix since that's what SKLL does
    ids, labels, features = sk_shuffle(train_fs.ids,
                                       train_fs.labels,
                                       train_fs.features,
                                       random_state=123456789)
    shuffled_fs = FeatureSet(train_fs.name,
                             ids,
                             labels=labels,
                             features=features,
                             vectorizer=train_fs.vectorizer)
    X = shuffled_fs.features
    y = shuffled_fs.labels

    # instantiate and save two different sklearn LogisticRegression
    # models for each value of C that was in the SKLL grid and with the
    # same other fixed parameters that we used for SKLL
    models_with_C_values = {}
    for param_value in grid_search_results['params']:
        model_kwargs = param_value
        model_kwargs.update({'max_iter': 1000,
                             'solver': 'liblinear',
                             'multi_class': 'auto',
                             'random_state': 123456789})
        sklearn_learner = LogisticRegression(**model_kwargs)
        models_with_C_values[param_value['C']] = sklearn_learner

    # now let's split the featureset the same way SKLL would have
    # done using the folds file
    dummy_label = next(iter(folds_dict.values()))
    fold_groups = [folds_dict.get(curr_id, dummy_label) for curr_id in shuffled_fs.ids]
    kfold = FilteredLeaveOneGroupOut(folds_dict, shuffled_fs.ids)
    fold_train_test_ids = list(kfold.split(shuffled_fs.features, shuffled_fs.labels, fold_groups))

    # generate predictions on the test split of the each fold, and
    # then compute the objectives appropriately (using either the
    # labels or the probabilities) and store them for comparison
    metric_values_dict = {}

    # compute values for each fold
    for fold_id in [0, 1]:

        metric_values_dict[fold_id] = []

        fold_train_ids = fold_train_test_ids[fold_id][0]
        fold_test_ids = fold_train_test_ids[fold_id][1]

        X_fold_train = X[fold_train_ids, :]
        X_fold_test = X[fold_test_ids, :]

        y_fold_train = y[fold_train_ids, ]
        y_fold_test = y[fold_test_ids, ]

        # let's also compute the class/label indices
        # since we will need them later
        y_fold_train_indices = [clf.label_dict[label] for label in y_fold_train]
        y_fold_test_indices = [clf.label_dict[label] for label in y_fold_test]

        # iterate over the two trained sklearn models;
        # one for each point on the C grid
        for C_value in models_with_C_values:

            # for training, we use the class indices
            # rather than the class labels like SKLL does
            # so that our predictions are in the index
            # space rather than label space
            sklearn_learner = models_with_C_values[C_value]
            sklearn_learner.fit(X_fold_train, y_fold_train_indices)

            # compute both labels and probabilities via sklearn
            sklearn_fold_test_labels = sklearn_learner.predict(X_fold_test)
            sklearn_fold_test_probs = sklearn_learner.predict_proba(X_fold_test)

            # now let's proceed on a metric by metric basis
            # and note that we are again using the class
            # indices rather than the class labels themselves

            # 1. Correlation metrics are only computed for integer
            #    or float labels; they use probability values if
            #    available only in the binary case.
            if metric_name in ['pearson', 'spearman', 'kendall_tau']:
                if metric_name == 'pearson':
                    corr_metric_func = pearsonr
                elif metric_name == 'spearman':
                    corr_metric_func = spearmanr
                elif metric_name == 'kendall_tau':
                    corr_metric_func = kendalltau

                if issubclass(label_type, (np.int32, np.int64, np.float64)):
                    if len(label_array) == 2 and use_probabilities:
                        metric_value = corr_metric_func(y_fold_test_indices,
                                                        sklearn_fold_test_probs[:, 1])[0]
                    else:
                        metric_value = corr_metric_func(y_fold_test_indices,
                                                        sklearn_fold_test_labels)[0]
            # 2. `neg_log_loss` requires probability values irrespective
            #     of label types and number of labels
            elif metric_name == 'neg_log_loss':
                if use_probabilities:
                    metric_value = -1 * log_loss(y_fold_test_indices,
                                                 sklearn_fold_test_probs)
            # 3. The other probabilistic metrics `average_precision`
            #    and `roc_auc` only work with positive class probabilities
            #    and only for the binary case
            elif metric_name == 'average_precision':
                if len(label_array) == 2 and use_probabilities:
                    metric_value = average_precision_score(y_fold_test_indices,
                                                           sklearn_fold_test_probs[:, 1])
            elif metric_name == 'roc_auc':
                if len(label_array) == 2 and use_probabilities:
                    metric_value = roc_auc_score(y_fold_test_indices,
                                                 sklearn_fold_test_probs[:, 1])

            # 4. Accuracy and unweighted kappas should work no matter what
            #    and use the labels; kappas are not in scikit-learn so
            #    we have to use the SKLL implementation
            elif metric_name == 'accuracy':
                metric_value = accuracy_score(y_fold_test_indices,
                                              sklearn_fold_test_labels)
            elif metric_name in UNWEIGHTED_KAPPA_METRICS:
                metric_value = use_score_func(metric_name,
                                              y_fold_test_indices,
                                              sklearn_fold_test_labels)

            # 5. The only ones left are the weighted kapps;
            #    these require contiguous ints or floats
            elif metric_name in WEIGHTED_KAPPA_METRICS:
                if contiguous_ints_or_floats(label_array):
                    metric_value = use_score_func(metric_name,
                                                  y_fold_test_indices,
                                                  sklearn_fold_test_labels)

            # save computed metric value for the fold, if any
            metric_values_dict[fold_id].append(metric_value)

    # compare SKLL grid-search values with sklearn values for each fold
    skll_fold_values = (grid_search_results['split0_test_score'],
                        grid_search_results['split1_test_score'])
    sklearn_fold_values = (metric_values_dict[0], metric_values_dict[1])
    assert_array_equal(skll_fold_values[0], sklearn_fold_values[0])
    assert_array_equal(skll_fold_values[1], sklearn_fold_values[1])


def test_objective_values_for_classification():

    # Test that the objectives for classifications yield expected results.
    # We need a special test here since in some cases we use the probabilities
    # and in others we use the labels. Specific cases to test:
    # 1. Probabilistic metrics: neg_log_loss, average_precision, roc_auc. The
    #    last two only work for the binary case. Work for both string and integer
    #    labels.
    # 2. Correlation metrics: pearson, spearman, and kendall_tau. We want to test
    #    that they work as expected for {binary, multi-class} x {probabilities, no
    #    probabilities}. Only work for integer/float labels.
    # 3. Weighted kappa metrics. Work for only int labels.
    # 4. Accuracy and unweighted kappa metrics. Want to test for {probabilities, no
    #    probabilities}. Works for both int and string labels.

    # Here's how the test works:
    # 1. We train a LogisticRegression classifier via the SKLL API on
    #    an artificial training set with:
    #    (a) each metric as the tuning objective
    #    (b) only two points on the grid for a single hyperparameter
    #    (b) 2 externally specified grid search folds via a dictionary
    #    (c) ``probability`` either True or False
    #
    # 2. We get the specific values of the objective for each point
    #    on the grid and for each of the grid search test splits, by
    #    using the results JSON file.
    #
    # 3. We run a separate experiment in scikit-learn space where:
    #    (a) for of the two points on the grid, we explicitly train
    #        the same model on the train split
    #    (b) compute the two trained model's predictions on the respective
    #        test splits
    #    (c) compute the two values of the objective using these predictions
    #
    # 4. We then compare values in 2 and 3 to verify that they are equal.

    metrics_to_test = {'accuracy'}
    metrics_to_test.update(CORRELATION_METRICS,
                           PROBABILISTIC_METRICS,
                           UNWEIGHTED_KAPPA_METRICS,
                           WEIGHTED_KAPPA_METRICS)
    metrics_to_test = sorted(metrics_to_test)

    for (metric,
         label_array,
         use_probabilities) in product(metrics_to_test,
                                       [np.array([1, 2, 3]),
                                        np.array(['A', 'B', 'C']),
                                        np.array([-2, -1, 0, 1, 2]),
                                        np.array([2, 4, 6]),
                                        np.array([0, 1]),
                                        np.array([-1, 1]),
                                        np.array(['yes', 'no']),
                                        np.array([1.0, 2.0]),
                                        np.array([-1.0, 1]),
                                        np.array([1.0, 1.1, 1.2]),
                                        np.array([3, 5, 10]),
                                        np.array([1.0, 2.0, 3.0]),
                                        np.array([1, 2, 3, 4.0]),
                                        np.array([4, 5, 6]),
                                        np.array(['A', 'B', 'C', 1, 2, 3])],
                                       [True, False]):

        # skip following configurations that would raise an exception
        # during grid search either from SKLL or from sklearn:
        # (a) average_precision/roc_auc and non-binary classification
        # (b) correlation/weighted kappa metrics and non-integer labels
        # (c) weighted kappa metrics and non-contiguous integer/float labels
        # (d) probabilistic metrics and no probabilities
        skipped_conditions = ((metric in ['average_precision', 'roc_auc'] and
                               len(label_array) != 2) or
                              ((metric in WEIGHTED_KAPPA_METRICS or
                                metric in CORRELATION_METRICS) and
                               issubclass(label_array.dtype.type, str)) or
                              (metric in WEIGHTED_KAPPA_METRICS and
                               not contiguous_ints_or_floats(label_array)) or
                              (metric in PROBABILISTIC_METRICS and
                               not use_probabilities))
        if skipped_conditions:
            continue
        else:
            yield (check_objective_values_for_classification,
                   metric,
                   label_array,
                   use_probabilities)


def check_metric_values_for_classification(metric_name,
                                           label_array,
                                           use_probabilities):

    # get the config template
    config_template_path = join(
        config_dir,
        'test_metric_values_for_classification.template.cfg'
    )

    # instantiate a random number generator
    prng = np.random.RandomState(123456789)

    # create our training and test sets
    train_fs, test_fs = make_classification_data(num_examples=200)

    # create the labels by repeatedly sampling from the given label array
    train_fs.labels = prng.choice(label_array, size=len(train_fs))
    test_fs.labels = prng.choice(label_array, size=len(test_fs))

    # get the label type of the test set
    label_type = test_fs.labels.dtype.type

    # write out the train and test sets to disk so that we can use them
    # when we run the configuration file
    train_file = join(train_dir, 'metric_values_train.jsonlines')
    test_file = join(train_dir, 'metric_values_test.jsonlines')
    NDJWriter.for_path(train_file, train_fs).write()
    NDJWriter.for_path(test_file, test_fs).write()

    experiment_name = (
        f'clf_metric_value_{use_probabilities}_{len(label_array)}_'
        f'{label_type.__name__}_{metric_name}'
    )

    values_to_fill_dict = {'experiment_name': experiment_name,
                           'train_file': train_file,
                           'test_file': test_file,
                           'logs': output_dir,
                           'models': output_dir,
                           'results': output_dir,
                           'predictions': output_dir,
                           'probability': 'true' if use_probabilities else 'false',
                           'metrics': f"['{metric_name}']"}

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         f'{metric_name}_{use_probabilities}',
                                         good_probability_option=True)

    # run this experiment and load the results_json and the SKLL model
    results_json_path = run_configuration(config_path, local=True, quiet=True)[0]
    with open(results_json_path) as results_json_file:
        results_obj = json.load(results_json_file)[0]
    model_file_path = join(
        output_dir,
        f'{experiment_name}_metric_{results_obj["learner_name"]}.model')
    clf = Learner.from_file(model_file_path)

    # get the value of the metric from SKLL
    skll_metric_value = results_obj['additional_scores'][metric_name]

    # get the feature matrix (X) and the labels array (y) for both
    # the training and test set to pass to scikit-learn; note that
    # we want y to be class indices since those are what we need
    # when computing the metrics to match what SKLL does
    X_train = train_fs.features
    y_train = [clf.label_dict[label] for label in train_fs.labels]
    X_test = test_fs.features
    y_test = [clf.label_dict[label] for label in test_fs.labels]

    # instantiate a LogisticRegression models with the default
    # parameters that are used in SKLL
    model_kwargs = {'max_iter': 1000,
                    'solver': 'liblinear',
                    'multi_class': 'auto',
                    'random_state': 123456789}

    sklearn_learner = LogisticRegression(**model_kwargs)

    # now generate predictions on the test set from sklearn and
    # then compute the metrics appropriately (using either the
    # labels or the probabilities) and store them for comparison
    # for training, we use the class indices rather than the class
    # labels like SKLL does so that our predictions are in the index
    # space rather than label space
    sklearn_learner.fit(X_train, y_train)

    # compute both labels and probabilities via sklearn
    sklearn_test_labels = sklearn_learner.predict(X_test)
    sklearn_test_probs = sklearn_learner.predict_proba(X_test)

    # now let's proceed on a metric by metric basis
    # and note that we are again using the class
    # indices rather than the class labels themselves

    # 1. Correlation metrics are only computed for integer
    #    or float labels; they use probability values if
    #    available only in the binary case.
    if metric_name in ['pearson', 'spearman', 'kendall_tau']:
        if metric_name == 'pearson':
            corr_metric_func = pearsonr
        elif metric_name == 'spearman':
            corr_metric_func = spearmanr
        elif metric_name == 'kendall_tau':
            corr_metric_func = kendalltau

        if issubclass(label_type, (np.int32, np.int64, np.float64)):
            if len(label_array) == 2 and use_probabilities:
                sklearn_metric_value = corr_metric_func(y_test,
                                                        sklearn_test_probs[:, 1])[0]
            else:
                sklearn_metric_value = corr_metric_func(y_test,
                                                        sklearn_test_labels)[0]

    # 2. `neg_log_loss` requires probability values irrespective
    #     of label types and number of labels
    elif metric_name == 'neg_log_loss':
        if use_probabilities:
            sklearn_metric_value = -1 * log_loss(y_test,
                                                 sklearn_test_probs)
    # 3. The other probabilistic metrics `average_precision`
    #    and `roc_auc` only work with positive class probabilities
    #    and only for the binary case
    elif metric_name == 'average_precision':
        if len(label_array) == 2 and use_probabilities:
            sklearn_metric_value = average_precision_score(y_test,
                                                           sklearn_test_probs[:, 1])
    elif metric_name == 'roc_auc':
        if len(label_array) == 2 and use_probabilities:
            sklearn_metric_value = roc_auc_score(y_test,
                                                 sklearn_test_probs[:, 1])

    # 4. Accuracy and unweighted kappas should work no matter what
    #    and use the labels; kappas are not in scikit-learn so
    #    we have to use the SKLL implementation
    elif metric_name == 'accuracy':
        sklearn_metric_value = accuracy_score(y_test,
                                              sklearn_test_labels)
    elif metric_name in UNWEIGHTED_KAPPA_METRICS:
        sklearn_metric_value = use_score_func(metric_name,
                                              y_test,
                                              sklearn_test_labels)

    # 5. The only ones left are the weighted kappas and they are not in sklearn
    #    so we are forced to use the SKLL implementations; both types
    #    require integer labels
    elif metric_name in WEIGHTED_KAPPA_METRICS:
        if contiguous_ints_or_floats(label_array):
            sklearn_metric_value = use_score_func(metric_name,
                                                  y_test,
                                                  sklearn_test_labels)

    eq_(skll_metric_value, sklearn_metric_value)


def test_metric_values_for_classification():

    # Test that the metrics for classifications yield expected results.
    # We need a special test here since in some cases we use the probabilities
    # and in others we use the labels. Specific cases to test:
    # 1. Probabilistic metrics: neg_log_loss, average_precision, roc_auc. The
    #    last two only work for the binary case. Work for both string and integer
    #    labels.
    # 2. Correlation metrics: pearson, spearman, and kendall_tau. We want to test
    #    that they work as expected for {binary, multi-class} x {probabilities, no
    #    probabilities}. Only works for integer labels.
    # 3. Unweighted kappa metrics. Work for both int and string labels
    # 4. Weighted kappa metrics. Works for only int labels.
    # 5. Regular metrics: accuracy. Want to test for {probabilities, no
    #    probabilities}. Works for both int and string labels.

    # Here's how the test works:
    # 1. We first run SKLL "evaluate" experiments by training
    #    a LogisticRegression classifier on an artificial training set
    #    with:
    #    (a) no grid search
    #    (b) each metric specified as an output metric
    #    (c) probability either true or false
    #
    # 2. We get the values of the metric(s) on the test set from
    #    the results JSON file.
    #
    # 3. We run a separate experiment in scikit-learn space where:
    #    (a) we explicitly train a LogisticRegression model on the
    #        training set
    #    (b) we compute the trained model's predictions on the test set
    #    (c) we compute the metric value using these predictions
    #
    # 4. We then compare values in 2 and 3 to verify that they are equal.

    metrics_to_test = {'accuracy'}
    metrics_to_test.update(CORRELATION_METRICS,
                           PROBABILISTIC_METRICS,
                           UNWEIGHTED_KAPPA_METRICS,
                           WEIGHTED_KAPPA_METRICS)
    metrics_to_test = sorted(metrics_to_test)

    for (metric,
         label_array,
         use_probabilities) in product(metrics_to_test,
                                       [np.array([1, 2, 3]),
                                        np.array(['A', 'B', 'C']),
                                        np.array([-2, -1, 0, 1, 2]),
                                        np.array([2, 4, 6]),
                                        np.array([0, 1]),
                                        np.array([-1, 1]),
                                        np.array(['yes', 'no']),
                                        np.array([1.0, 2.0]),
                                        np.array([-1.0, 1]),
                                        np.array([1.0, 1.1, 1.2]),
                                        np.array([3, 5, 10]),
                                        np.array([1.0, 2.0, 3.0]),
                                        np.array([1, 2, 3, 4.0]),
                                        np.array([4, 5, 6]),
                                        np.array(['A', 'B', 'C', 1, 2, 3])],
                                       [True, False]):

        # skip following configurations that would raise an exception
        # during grid search either from SKLL or from sklearn:
        # (a) average_precision/roc_auc and non-binary classification
        # (b) correlation/weighted kappa metrics and non-integer labels
        # (c) weighted kappa metrics and non-contiguous integer/float labels
        # (d) probabilistic metrics and no probabilities
        skipped_conditions = ((metric in ['average_precision', 'roc_auc'] and
                               len(label_array) != 2) or
                              ((metric in WEIGHTED_KAPPA_METRICS or
                                metric in CORRELATION_METRICS) and
                               issubclass(label_array.dtype.type, str)) or
                              (metric in WEIGHTED_KAPPA_METRICS and
                               not contiguous_ints_or_floats(label_array)) or
                              (metric in PROBABILISTIC_METRICS and
                               not use_probabilities))
        if skipped_conditions:
            continue
        else:
            yield (check_metric_values_for_classification,
                   metric,
                   label_array,
                   use_probabilities)


def check_metrics_and_objectives_overlap(task, metrics, objectives):

    train_file = join(train_dir, 'metric_values_train.jsonlines')
    test_file = join(train_dir, 'metric_values_test.jsonlines')

    # make a simple config file
    values_to_fill_dict = {'experiment_name': 'clf_metrics_objective_overlap',
                           'task': task,
                           'train_file': train_file,
                           'learners': "['LogisticRegression']",
                           'logs': output_dir,
                           'probability': 'false',
                           'results': output_dir,
                           'predictions': output_dir,
                           'metrics': str(metrics)}

    if task == 'evaluate':
        values_to_fill_dict['test_file'] = test_file

    if objectives:
        values_to_fill_dict['objectives'] = str(objectives)
    else:
        values_to_fill_dict['grid_search'] = 'false'

    config_template_path = join(config_dir, 'test_fancy.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'metrics_objective_overlap',
                                         good_probability_option=True)

    # run this configuration file and get the output JSON
    # files for each experiment
    results_json_paths = run_configuration(config_path, local=True, quiet=True)

    # make a dummy objective if we do not have any for the
    # purposes of iteration
    if not objectives:
        objectives = [None]

    # iterate over each objective in the list and each results JSON file
    # there should be a 1:1 correspondence since there is a single learner
    # and a single featureset
    for objective, results_json_path in zip(objectives, results_json_paths):
        if objective is None:
            pruned_metrics = metrics
        else:
            pruned_metrics = [metric for metric in metrics if metric != objective]
        with open(results_json_path) as results_json_file:
            results_obj = json.load(results_json_file)[0]
        eq_(results_obj['grid_objective'], objective)
        eq_(set(results_obj['additional_scores'].keys()), set(pruned_metrics))
        ok_(objective not in results_obj['additional_scores'])


def test_metrics_and_objectives_overlap():

    for task, metrics, objectives in product(["evaluate", "cross_validate"],
                                             [["f1_score_weighted", "unweighted_kappa", "accuracy"]],
                                             [[], ["accuracy"], ["accuracy", "unweighted_kappa"]]):
        yield check_metrics_and_objectives_overlap, task, metrics, objectives


def test_multinomialnb_loading():
    """
    Make sure we can load MultnomialNB models from disk
    """

    learner = Learner('MultinomialNB')
    train_fs, test_fs = make_classification_data(num_examples=100, non_negative=True)
    learner.train(train_fs, grid_search=False)
    model_file = join(output_dir, 'test_multinomialnb_loading.model')
    learner.save(model_file)
    predictions1 = learner.predict(test_fs)
    del learner

    learner2 = Learner.from_file(model_file)
    predictions2 = learner2.predict(test_fs)

    assert_array_equal(predictions1, predictions2)


def test_load_old_skll_model():
    """
    Make sure loading an older SKLL model raises an error
    """

    # get the path to the old model
    model_path = join(other_dir, "v2.1_SVC.model")

    # suppress warnings and check that error is raised
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_raises_regex(
            ValueError,
            r"created with v2.1 of SKLL, which is incompatible with the "
            r"current",
            Learner.from_file,
            model_path
        )


def check_predict_return_and_write(learner, test_fs, class_labels):

    returned_predictions = learner.predict(test_fs,
                                           class_labels=class_labels,
                                           prediction_prefix=f"{output_dir}/test_predict_return_and_write")

    # open and read the predictions file
    with open(join(output_dir,
                   "test_predict_return_and_write_predictions.tsv")) as predictfh:
        reader = csv.DictReader(predictfh, dialect=csv.excel_tab)
        written_predictions = list(reader)

    # if `class_labels` is `True`, then we both print out
    # and return class labels
    if class_labels:
        eq_(returned_predictions.shape, (250,))
        assert(set(returned_predictions).issubset({'a', 'b', 'c'}))
        header = written_predictions[0]
        eq_(sorted(header.keys()), ['id', 'prediction'])
        for row in written_predictions[1:]:
            assert(row['prediction'] in ['a', 'b', 'c'])
    else:
        # if `class_labels` is `False` but the learner is probabilistic,
        # then we both print out and return class probabilities
        if learner.probability:
            eq_(returned_predictions.shape, (250, 3))
            header = written_predictions[0]
            eq_(sorted(header.keys()), ['a', 'b', 'c', 'id'])
            for row in written_predictions[1:]:
                assert('a' in row)
                assert('b' in row)
                assert('c' in row)
                assert(0 <= float(row['a']) <= 1)
                assert(0 <= float(row['b']) <= 1)
                assert(0 <= float(row['c']) <= 1)
        # if `class_labels` is `False` and the learner is non-probabilistic,
        # then we print out class labels but return class indices
        else:
            eq_(returned_predictions.shape, (250,))
            assert(set(returned_predictions).issubset({0, 1, 2}))
            header = written_predictions[0]
            eq_(sorted(header.keys()), ['id', 'prediction'])
            for row in written_predictions[1:]:
                assert(row['prediction'] in ['a', 'b', 'c'])


def test_predict_return_and_write():
    """
    Test the predictions are returned and written out correctly
    """
    train_fs, test_fs = make_classification_data(num_examples=500,
                                                 non_negative=True,
                                                 num_labels=3,
                                                 string_label_list=['a', 'b', 'c'])

    # create a non-probabilistic and a probabilistic learner
    learner1 = Learner('SGDClassifier', probability=False)
    learner2 = Learner('SGDClassifier', probability=True)

    # train both the learners
    learner1.train(train_fs, grid_search=False)
    learner2.train(train_fs, grid_search=False)

    # run the various tests
    for (learner, class_labels) in product([learner1, learner2],
                                           [False, True]):
        yield check_predict_return_and_write, learner, test_fs, class_labels


def test_train_non_sparse_featureset():
    """Test that we can train a classifier on a non-sparse featureset"""
    train_file = join(other_dir, 'examples_train.jsonlines')
    train_fs = NDJReader.for_path(train_file, sparse=False).read()
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False)
    ok_(hasattr(learner.model, "coef_"))
