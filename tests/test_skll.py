# License: BSD 3 clause
'''
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import glob
import itertools
import json
import math
import os
import re
import sys
from collections import OrderedDict
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
import scipy.sparse as sp
from nose.tools import eq_, raises, assert_almost_equal, assert_not_equal
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.datasets.samples_generator import (make_classification,
                                                make_regression)
from sklearn.utils.testing import assert_greater, assert_less
from skll.data import write_feature_file, load_examples, convert_examples
from skll.data import FeatureSet, NDJWriter
from skll.experiments import (_load_featureset, run_configuration,
                              _load_cv_folds, _setup_config_parser,
                              run_ablation)
from skll.learner import Learner, SelectByMinCount
from skll.learner import _REGRESSION_MODELS, _DEFAULT_PARAM_GRIDS
from skll.metrics import kappa
from skll.utilities import skll_convert
from skll.utilities.compute_eval_from_predictions \
    import compute_eval_from_predictions
from scipy.stats import pearsonr


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
SCORE_OUTPUT_RE = re.compile(r'Objective Function Score \(Test\) = '
                             r'([\-\d\.]+)')
GRID_RE = re.compile(r'Grid Objective Score \(Train\) = ([\-\d\.]+)')
_my_dir = abspath(dirname(__file__))


def setup():
    train_dir = join(_my_dir, 'train')
    if not exists(train_dir):
        os.makedirs(train_dir)
    test_dir = join(_my_dir, 'test')
    if not exists(test_dir):
        os.makedirs(test_dir)
    output_dir = join(_my_dir, 'output')
    if not exists(output_dir):
        os.makedirs(output_dir)

def test_SelectByMinCount():
    ''' Test SelectByMinCount feature selector '''
    m2 = [[0.001, 0.0, 0.0, 0.0],
          [0.00001, -2.0, 0.0, 0.0],
          [0.001, 0.0, 0.0, 4.0],
          [0.0101, -200.0, 0.0, 0.0]]

    # default should keep all nonzero features (i.e. ones that appear 1+ times)
    feat_selector = SelectByMinCount()
    expected = np.array([[0.001, 0.0, 0.0],
                         [0.00001, -2.0, 0.0],
                         [0.001, 0.0, 4.0],
                         [0.0101, -200.0, 0.0]])
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(feat_selector.fit_transform(
        sp.csr_matrix(m2)).todense(),
        expected)

    # keep features that happen 2+ times
    feat_selector = SelectByMinCount(min_count=2)
    expected = np.array([[0.001, 0.0],
                         [0.00001, -2.0],
                         [0.001, 0.0],
                         [0.0101, -200.0]])
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(
        feat_selector.fit_transform(sp.csr_matrix(m2)).todense(),
        expected)

    # keep features that happen 3+ times
    feat_selector = SelectByMinCount(min_count=3)
    expected = np.array([[0.001], [0.00001], [0.001], [0.0101]])
    assert_array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert_array_equal(
        feat_selector.fit_transform(sp.csr_matrix(m2)).todense(),
        expected)


@raises(ValueError)
def test_input_checking1():
    '''
    Test merging featuresets with different number of examples
    '''
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_2examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix, quiet=True)


@raises(ValueError)
def test_input_checking2():
    '''
    Test joining featuresets that contain the same features for each instance
    '''
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix, quiet=True)


def test_input_checking3():
    '''
    Test to ensure that we correctly merge featuresets
    '''
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_2']
    examples_tuple = _load_featureset(dirpath, featureset, suffix, quiet=True)
    eq_(examples_tuple.features.shape[0], 3)

def fill_in_config_paths(config_template_path):
    '''
    Add paths to train, test, and output directories to a given config template
    file.
    '''

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path)

    task = config.get("General", "task")
    # experiment_name = config.get("General", "experiment_name")

    config.set("Input", "train_location", train_dir)

    to_fill_in = ['log', 'vocabs', 'predictions']

    if task != 'cross_validate':
        to_fill_in.append('models')

    if task == 'evaluate' or task == 'cross_validate':
        to_fill_in.append('results')

    for d in to_fill_in:
        config.set("Output", d, join(output_dir))

    if task == 'cross_validate':
        cv_folds_location = config.get("Input", "cv_folds_location")
        if cv_folds_location:
            config.set("Input", "cv_folds_location",
                       join(train_dir, cv_folds_location))

    if task == 'predict' or task == 'evaluate':
        config.set("Input", "test_location", test_dir)

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}.cfg'.format(config_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def make_cv_folds_data(num_examples_per_fold=100,
                       num_folds=3,
                       use_feature_hashing=False):
    '''
    Create data for pre-specified CV folds tests
    with or without feature hashing
    '''

    num_total_examples = num_examples_per_fold * num_folds

    # create the numeric features and the binary classes
    X, _ = make_classification(n_samples=num_total_examples,
                               n_features=3, n_informative=3, n_redundant=0,
                               n_classes=2, random_state=1234567890)
    y = np.array([0, 1] * int(num_total_examples/2))

    # the folds mapping: the first num_examples_per_fold examples
    # are in fold 1 the second num_examples_per_fold are in
    # fold 2 and so on
    foldgen = ([i]*num_examples_per_fold for i in range(num_folds))
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
    ids = ['EXAMPLE_{}'.format(num_examples_per_fold * k + i) \
            for k in range(num_folds) for i in range(num_examples_per_fold)]

    # create the cross-validation feature set with or without feature hashing
    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    cv_fs = FeatureSet('cv_folds', ids=ids, features=features,
                       classes=y, vectorizer=vectorizer)

    # make the custom cv folds dictionary
    custom_cv_folds = dict(zip(ids, folds))

    return (cv_fs, custom_cv_folds)


def test_specified_cv_folds():
    '''
    Test to check cross-validation results with specified folds, feature hashing, and RBFSampler
    '''

    # this runs four tests
    # the first does not use feature hashing with 9 features (3 numeric, 6 binary)
    # has pre-specified folds and has less than 60% accuracy for each of the 3 folds

    # the second uses feature hashing with 4 features, uses 10 folds (not pre-specified)
    # and has more than 70% accuracy accuracy for each of the 10 folds.

    # the third is the same as the first but uses an RBFSampler

    # the fourth is the same as the second but uses an RBFSampler

    for test_value, assert_func, grid_size, use_hashing, use_sampler in \
        [(0.55, assert_less, 3, False, False),
         (0.1, assert_greater, 10, True, False),
         (0.53, assert_less, 3, False, True),
         (0.7, assert_greater, 10, True, True)]:

        sampler = 'RBFSampler' if use_sampler else None
        learner = Learner('LogisticRegression', sampler=sampler)
        cv_fs, custom_cv_folds = make_cv_folds_data(use_feature_hashing=use_hashing)
        folds = custom_cv_folds if not use_hashing else 10
        cv_output = learner.cross_validate(cv_fs,
                                           cv_folds=folds,
                                           grid_search=True,
                                           feature_hasher=use_hashing)
        fold_test_scores = [t[-1] for t in cv_output[0]]

        overall_score = np.mean(fold_test_scores)

        assert_func(overall_score, test_value)

        eq_(len(fold_test_scores), grid_size)
        for fold_score in fold_test_scores:
            assert_func(fold_score, test_value)

def test_feature_merging_order_invariance():
    '''
    Test whether featuresets with different orders of IDs can be merged
    '''

    # First, randomly generate two feature sets and then make sure they have
    # the same labels.
    train_fs1, _, _ = make_regression_data()
    train_fs2, _, _ = make_regression_data(start_feature_num=3,
                                           random_state=87654321)
    train_fs2.classes = train_fs1.classes.copy()

    # make a reversed copy of feature set 2
    shuffled_indices = list(range(len(train_fs2.ids)))
    np.random.seed(123456789)
    np.random.shuffle(shuffled_indices)
    train_fs2_ids_shuf = train_fs2.ids[shuffled_indices]
    train_fs2_classes_shuf = train_fs2.classes[shuffled_indices]
    train_fs2_features_shuf = train_fs2.features[shuffled_indices]
    train_fs2_shuf = FeatureSet("f2_shuf",
                                train_fs2_ids_shuf,
                                classes=train_fs2_classes_shuf,
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

    assert_array_equal(merged_fs.classes, train_fs1.classes)
    assert_array_equal(merged_fs.classes, train_fs2.classes)
    assert_array_equal(merged_fs.classes, merged_fs_shuf.classes)

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


def make_regression_data(num_examples=100, train_test_ratio=0.5,
                         num_features=2, sd_noise=1.0,
                         use_feature_hashing=False,
                         start_feature_num=1,
                         random_state=1234567890):

    # use sklearn's make_regression to generate the data for us
    X, y, weights = make_regression(n_samples=num_examples,
                                    n_features=num_features,
                                    noise=sd_noise, random_state=random_state,
                                    coef=True)

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, num_examples + 1)]

    # create a list of dictionaries as the features
    feature_names = ['f{}'.format(n) for n
                     in range(start_feature_num,
                              start_feature_num + num_features)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # convert the weights array into a dictionary for convenience
    weightdict = dict(zip(feature_names, weights))

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples],
                                     features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # create a FeatureHasher if we are asked to use feature hashing
    # and use 2.5 times the number of features to be on the safe side
    vectorizer = (FeatureHasher(n_features=int(round(2.5 * num_features))) if
                  use_feature_hashing else None)
    train_fs = FeatureSet('regression_train', ids=train_ids,
                          classes=train_y, features=train_features,
                          vectorizer=vectorizer)
    test_fs = FeatureSet('regression_test', test_ids,
                         classes=test_y, features=test_features,
                         vectorizer=vectorizer)

    return (train_fs, test_fs, weightdict)

# the utility function to run the basic regression tests
# with or without feature hashing
def check_regression(use_feature_hashing=False):
    # This is a bit of a contrived test, but it should fail if anything drastic
    # happens to the regression code.

    # create a FeatureSet object with the data we want to use
    if use_feature_hashing:
        train_fs, test_fs, weightdict = make_regression_data(num_examples=5000,
                                                             num_features=10,
                                                             use_feature_hashing=True)
    else:
        train_fs, test_fs, weightdict = make_regression_data(num_examples=2000,
                                                             num_features=3)

    # create a LinearRegression learner
    learner = Learner('LinearRegression')

    # train it with the training feature set we created
    # make sure to set the grid objective to pearson
    learner.train(train_fs, grid_objective='pearson', feature_hasher=use_feature_hashing)

    # make sure that the weights are close to the weights
    # that we got from make_regression_data. Take the
    # ceiling before  comparing since just comparing
    # the ceilings should be enough to make sure nothing
    # catastrophic happened. Note though that we cannot
    # test feature weights if we are using feature hashing
    # since model_params is not defined with a featurehasher.
    if not use_feature_hashing:

        # get the weights for this trained model
        learned_weights = learner.model_params[0]

        for feature_name in learned_weights:
            learned_w = math.ceil(learned_weights[feature_name])
            given_w = math.ceil(weightdict[feature_name])
            eq_(learned_w, given_w)

    # now generate the predictions on the test FeatureSet
    predictions = learner.predict(test_fs, feature_hasher=use_feature_hashing)

    # now make sure that the predictions are close to
    # the actual test FeatureSet labels that we generated
    # using make_regression_data. To do this, we just
    # make sure that they are correlated with pearson > 0.95
    cor, _ = pearsonr(predictions, test_fs.classes)
    assert_greater(cor, 0.95)

# the runner function for the regression tests
def test_regression():
    # without feature hashing
    yield check_regression

    # with feature hashing
    yield check_regression, True


def make_classification_data(num_examples=100, train_test_ratio=0.5,
                             num_features=10, use_feature_hashing=False,
                             num_redundant=0, num_classes=2,
                             class_weights=None, non_negative=False,
                             random_state=1234567890):

    # use sklearn's make_classification to generate the data for us
    num_informative = num_features - num_redundant
    X, y = make_classification(n_samples=num_examples, n_features=num_features,
                               n_informative=num_informative, n_redundant=num_redundant,
                               n_classes=num_classes, weights=class_weights,
                               random_state=random_state)

    # if we were told to only generate non-negative features, then
    # we can simply take the absolute values of the generated features
    if non_negative:
        X = abs(X)

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, num_examples + 1)]

    # create a list of dictionaries as the features
    feature_names = ['f{}'.format(n) for n in range(1, num_features + 1)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples],
                                     features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # create a FeatureHasher if we are asked to use feature hashing
    # and use 2.5 times the number of features to be on the safe side
    vectorizer = (FeatureHasher(n_features=int(round(2.5 * num_features)))
                  if use_feature_hashing else None)
    train_fs = FeatureSet('classification_train', ids=train_ids,
                          classes=train_y, features=train_features,
                          vectorizer=vectorizer)
    test_fs = FeatureSet('classification_test', ids=test_ids,
                         classes=test_y, features=test_features,
                         vectorizer=vectorizer)

    return (train_fs, test_fs)


def check_predict(model='LogisticRegression', use_feature_hashing=False):
    '''
    This tests whether predict task runs and generates the same
    number of predictions as samples in the test set. The specified
    model indicates whether to generate random regression
    or classification data.
    '''

    # create the random data for the given model
    if model in _REGRESSION_MODELS:
        train_fs, test_fs, _ = make_regression_data(use_feature_hashing=use_feature_hashing)
    # feature hashing will not work for Naive Bayes since it requires non-negative feature values
    elif model == 'MultinomialNB':
        train_fs, test_fs = make_classification_data(use_feature_hashing=False, non_negative=True)
    else:
        train_fs, test_fs = make_classification_data(use_feature_hashing=use_feature_hashing)

    # create the learner with the specified model
    learner = Learner(model)

    # now train the learner on the training data and use feature hashing when specified
    # and when we are not using a Naive Bayes model
    learner.train(train_fs, grid_search=False, feature_hasher=use_feature_hashing and model != 'MultinomialNB')

    # now make predictions on the test set
    predictions = learner.predict(test_fs, feature_hasher=use_feature_hashing and model != 'MultinomialNB')

    # make sure we have the same number of outputs as the
    # number of test set samples
    eq_(len(predictions), test_fs.features.shape[0])


# the runner function for the prediction tests
def test_predict():
    for model, use_feature_hashing in itertools.product(_ALL_MODELS, [True, False]):
        yield check_predict, model, use_feature_hashing


# the function to create data with rare classes for cross-validation
def make_rare_class_data():
    '''
    We want to create data that has five instances per class, for three classes
    and for each instance within the group of 5, there's only a single feature firing
    '''

    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 16)]
    y = [0]*5 + [1]*5 + [2]*5
    X = np.vstack([np.identity(5), np.identity(5), np.identity(5)])
    feature_names = ['f{}'.format(i) for i in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    return FeatureSet('rare-class', ids=ids, features=features, classes=y)

def test_rare_class():
    '''
    Test cross-validation when some classes are very rare
    '''

    rare_class_fs = make_rare_class_data()
    prediction_prefix = join(_my_dir, 'output', 'rare_class')
    learner = Learner('LogisticRegression')
    learner.cross_validate(rare_class_fs,
                           grid_objective='unweighted_kappa',
                           prediction_prefix=prediction_prefix)

    with open(prediction_prefix + '.predictions', 'r') as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        pred = [row[1] for row in reader]

        eq_(len(pred), 15)

# Generate and write out data for the test that checks summary scores
def make_summary_data():
    train_fs, test_fs = make_classification_data(num_examples=600,
                                                 train_test_ratio=0.8,
                                                 num_classes=2,
                                                 num_features=3,
                                                 non_negative=True)

    # Write training feature set to a file
    train_path = join(_my_dir, 'train', 'test_summary.jsonlines')
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Write test feature set to a file
    test_path = join(_my_dir, 'test', 'test_summary.jsonlines')
    writer = NDJWriter(test_path, test_fs)
    writer.write()


# Function that checks to make sure that the summary files
# contain the right results
def check_summary_score(use_feature_hashing=False):

    # Test to validate summary file scores
    make_summary_data()

    cfgfile = 'test_summary_feature_hasher.template.cfg' if use_feature_hashing else 'test_summary.template.cfg'
    config_template_path = join(_my_dir, 'configs', cfgfile)
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    outprefix = 'test_summary_feature_hasher_test_summary' if use_feature_hashing else 'test_summary_test_summary'
    summprefix = 'test_summary_feature_hasher' if use_feature_hashing else 'test_summary'

    with open(join(_my_dir, 'output', ('{}_'
                                       'LogisticRegression.results'.format(outprefix)))) as f:
        outstr = f.read()
        logistic_result_score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])

    with open(join(_my_dir, 'output', '{}_SVC.results'.format(outprefix))) as f:
        outstr = f.read()
        svm_result_score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])

    # note that Naive Bayes doesn't work with feature hashing
    if not use_feature_hashing:
        with open(join(_my_dir, 'output', ('{}_'
                                           'MultinomialNB.results'.format(outprefix)))) as f:
            outstr = f.read()
            naivebayes_score_str = SCORE_OUTPUT_RE.search(outstr).groups()[0]
            naivebayes_result_score = float(naivebayes_score_str)

    with open(join(_my_dir, 'output', '{}_summary.tsv'.format(summprefix)), 'r') as f:
        reader = csv.DictReader(f, dialect='excel-tab')

        for row in reader:
            # the learner results dictionaries should have 26 rows,
            # and all of these except results_table
            # should be printed (though some columns will be blank).
            eq_(len(row), 26)
            assert row['model_params']
            assert row['grid_score']
            assert row['score']

            if row['learner_name'] == 'LogisticRegression':
                logistic_summary_score = float(row['score'])
            elif row['learner_name'] == 'MultinomialNB':
                naivebayes_summary_score = float(row['score'])
            elif row['learner_name'] == 'SVC':
                svm_summary_score = float(row['score'])

    test_tuples = [(logistic_result_score,
                    logistic_summary_score,
                    'LogisticRegression'),
                   (svm_result_score,
                    svm_summary_score,
                    'SVC')]

    if not use_feature_hashing:
        test_tuples.append((naivebayes_result_score,
                            naivebayes_summary_score,
                            'MultinomialNB'))

    for result_score, summary_score, learner_name in test_tuples:
        assert_almost_equal(result_score, summary_score,
                            msg=('mismatched scores for {} '
                                 '(result:{}, summary:'
                                 '{})').format(learner_name, result_score,
                                               summary_score))


def test_summary():
    # test summary score without feature hashing
    yield check_summary_score

    # test summary score with feature hashing
    yield check_summary_score, True


# Verify v0.9.17 model can still be loaded and generate the same predictions.
def test_backward_compatibility():
    '''
    Test to validate backward compatibility
    '''
    predict_path = join(_my_dir, 'backward_compatibility',
                        ('v0.9.17_test_summary_test_summary_'
                         'LogisticRegression.predictions'))
    model_path = join(_my_dir, 'backward_compatibility',
                      ('v0.9.17_test_summary_test_summary_LogisticRegression.'
                       '{}.model').format(sys.version_info[0]))
    test_path = join(_my_dir, 'backward_compatibility',
                     'v0.9.17_test_summary.jsonlines')

    learner = Learner.from_file(model_path)
    examples = load_examples(test_path, quiet=True)
    new_predictions = learner.predict(examples)[:, 1]

    with open(predict_path) as predict_file:
        for line, new_val in zip(predict_file, new_predictions):
            assert_almost_equal(float(line.strip()), new_val)


def make_sparse_data(use_feature_hashing=False):
    '''
    Function to create sparse data with two features always zero
    in the training set and a different one always zero in the
    test set
    '''
    # Create training data
    X, y = make_classification(n_samples=500, n_features=3,
                               n_informative=3, n_redundant=0,
                               n_classes=2, random_state=1234567890)

    # we need features to be non-negative since we will be
    # using naive bayes laster
    X = np.abs(X)

    # make sure that none of the features are zero
    X[np.where(X == 0)] += 1

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 501)]

    # create a list of dictionaries as the features
    # with f1 and f5 always 0
    feature_names = ['f{}'.format(n) for n in range(1, 6)]
    features = []
    for row in X:
        row = [0] + row.tolist() + [0]
        features.append(dict(zip(feature_names, row)))

    # use a FeatureHasher if we are asked to do feature hashing
    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    train_fs = FeatureSet('train_sparse', ids=ids,
                          features=features, classes=y,
                          vectorizer=vectorizer)

    # now create the test set with f4 always 0 but nothing else
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=4, n_redundant=0,
                               n_classes=2, random_state=1234567890)
    X = np.abs(X)
    X[np.where(X == 0)] += 1
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 101)]

    # create a list of dictionaries as the features
    # with f4 always 0
    feature_names = ['f{}'.format(n) for n in range(1, 6)]
    features = []
    for row in X:
        row = row.tolist()
        row = row[:3] + [0] + row[3:]
        features.append(dict(zip(feature_names, row)))

    test_fs = FeatureSet('test_sparse', ids=ids,
                         features=features, classes=y,
                         vectorizer=vectorizer)

    return train_fs, test_fs


def check_sparse_predict(use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(use_feature_hashing=use_feature_hashing)

    # train a linear SVM on the training data and evalute on the testing data
    learner = Learner('LogisticRegression')
    learner.train(train_fs, grid_search=False, feature_hasher=use_feature_hashing)
    test_score = learner.evaluate(test_fs, feature_hasher=use_feature_hashing)[1]

    expected_score = 0.51 if use_feature_hashing else 0.45
    assert_almost_equal(test_score, expected_score)


def test_sparse_predict():
    yield check_sparse_predict, False
    yield check_sparse_predict, True


def check_sparse_predict_sampler(use_feature_hashing=False):
    train_fs, test_fs = make_sparse_data(use_feature_hashing=use_feature_hashing)

    if use_feature_hashing:
        sampler = 'RBFSampler'
        sampler_parameters = {"gamma": 1.0, "n_components":50}
    else:
        sampler = 'Nystroem'
        sampler_parameters = {"gamma": 1.0, "n_components":50, "kernel":'rbf'}

    learner = Learner('LogisticRegression',
                      sampler=sampler,
                      sampler_kwargs=sampler_parameters)

    learner.train(train_fs, grid_search=False, feature_hasher=use_feature_hashing)
    test_score = learner.evaluate(test_fs, feature_hasher=use_feature_hashing)[1]

    expected_score = 0.4 if use_feature_hashing else 0.48
    assert_almost_equal(test_score, expected_score)

def test_sparse_predict_sampler():
    yield check_sparse_predict_sampler, False
    yield check_sparse_predict_sampler, True


def make_class_map_data():
    # Create training file
    train_path = join(_my_dir, 'train', 'test_class_map.jsonlines')
    ids = []
    classes = []
    features = []
    class_names = ['beagle', 'cat', 'dachsund', 'cat']
    for i in range(1, 101):
        y = class_names[i % 4]
        ex_id = "{}{}".format(y, i)
        # note that f1 and f5 are missing in all instances but f4 is not
        x = {"f2": i + 1, "f3": i + 2, "f4": i + 5}
        ids.append(ex_id)
        classes.append(y)
        features.append(x)
    train_fs = FeatureSet('train_class_map', ids=ids, features=features, classes=classes)
    writer = NDJWriter(train_path, train_fs)
    writer.write()

    # Create test file
    test_path = join(_my_dir, 'test', 'test_class_map.jsonlines')
    ids = []
    classes = []
    features = []
    for i in range(1, 51):
        y = class_names[i % 4]
        ex_id = "{}{}".format(y, i)
        # f1 and f5 are not missing in any instances here but f4 is
        x = {"f1": i, "f2": i + 2, "f3": i % 10, "f5": i * 2}
        ids.append(ex_id)
        classes.append(y)
        features.append(x)
    test_fs = FeatureSet('test_class_map', ids=ids, features=features, classes=classes)
    writer = NDJWriter(test_path, test_fs)
    writer.write()


def test_class_map():
    '''
    Test class maps
    '''

    make_class_map_data()

    config_template_path = join(
        _my_dir,
        'configs',
        'test_class_map.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output', ('test_class_map_test_class_map_Logistic'
                                       'Regression.results'))) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

    assert_almost_equal(logistic_result_score, 0.5)


def test_class_map_feature_hasher():
    '''
    Test class maps with feature hashing
    '''

    make_class_map_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_class_map_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output', ('test_class_map_test_class_map_'
                                       'LogisticRegression.results'))) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

    assert_almost_equal(logistic_result_score, 0.5)


def make_ablation_data():
    # Remove old CV data
    for old_file in glob.glob(join(_my_dir, 'output',
                                   'ablation_cv_*.results')):
        os.remove(old_file)

    num_examples = 1000

    np.random.seed(1234567890)

    # Create lists we will write files from
    ids = []
    features = []
    classes = []
    for j in range(num_examples):
        y = "dog" if j % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, j)
        x = {"f{}".format(feat_num): np.random.randint(0, 4) for feat_num in
             range(5)}
        x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
        ids.append(ex_id)
        classes.append(y)
        features.append(x)

    for i in range(5):
        train_path = join(_my_dir, 'train', 'f{}.jsonlines'.format(i))
        sub_features = []
        for example_num in range(num_examples):
            feat_num = i
            x = {"f{}".format(feat_num):
                 features[example_num]["f{}".format(feat_num)]}
            sub_features.append(x)
        train_fs = FeatureSet('ablation_cv', ids=ids, features=sub_features, classes=classes)
        writer = NDJWriter(train_path, train_fs)
        writer.write()

def check_ablation_rows(reader):
    '''
    Helper function to ensure that all ablated_features and featureset values
    are correct for each row in results summary file.

    :returns: Number of items in reader
    '''
    row_num = 0
    for row_num, row in enumerate(reader, 1):
        if row['ablated_features']:
            fs_str, ablated_str = row['featureset_name'].split('_minus_')
            actual_ablated = json.loads(row['ablated_features'])
        else:
            fs_str, ablated_str = row['featureset_name'].split('_all')
            actual_ablated = []
        expected_fs = set(fs_str.split('+'))
        expected_ablated = ablated_str.split('+') if ablated_str else []
        expected_fs = sorted(expected_fs - set(expected_ablated))
        actual_fs = json.loads(row['featureset'])
        eq_(expected_ablated, actual_ablated)
        eq_(expected_fs, actual_fs)
    return row_num


def test_ablation_cv():
    '''
    Test if ablation works with cross-validate
    '''

    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          'ablation_cv_*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_all_combos():
    '''
    Test to validate whether ablation all-combos works with cross-validate
    '''

    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True, all_combos=True)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          'ablation_cv_*results')))
    eq_(num_result_files, 20)


def test_ablation_cv_feature_hasher():
    '''
    Test if ablation works with cross-validate and feature_hasher
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(_my_dir, 'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          ('ablation_cv_feature_hasher_'
                                           '*.results'))))
    eq_(num_result_files, 14)


def test_ablation_cv_feature_hasher_all_combos():
    '''
    Test if ablation all-combos works with cross-validate and feature_hasher
    '''

    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True, all_combos=True)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(_my_dir, 'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          ('ablation_cv_feature_hasher_'
                                           '*results'))))
    eq_(num_result_files, 20)


def make_scaling_data(use_feature_hashing=False):

    X, y = make_classification(n_samples=1000, n_classes=2,
                               n_features=5, n_informative=5,
                               n_redundant=0, random_state=1234567890)

    # we want to arbitrary scale the various features to test the scaling
    scalers = np.array([1, 10, 100, 1000, 10000])
    X = X * scalers

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 1001)]

    # create a list of dictionaries as the features
    feature_names = ['f{}'.format(n) for n in range(1, 6)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # split everything into training and testing portions
    train_features, test_features = features[:800], features[800:]
    train_y, test_y = y[:800], y[800:]
    train_ids, test_ids = ids[:800], ids[800:]

    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    train_fs = FeatureSet('train_scaling', ids=train_ids,
                          features=train_features, classes=train_y,
                          vectorizer=vectorizer)
    test_fs = FeatureSet('test_scaling', ids=test_ids,
                         features=test_features, classes=test_y,
                         vectorizer=vectorizer)

    return (train_fs, test_fs)


def check_scaling_features(use_feature_hashing=False, use_scaling=False):
    train_fs, test_fs = make_scaling_data(use_feature_hashing=use_feature_hashing)

    # create a Linear SVM with the value of scaling as specified
    feature_scaling = 'both' if use_scaling else 'none'
    learner = Learner('SGDClassifier', feature_scaling=feature_scaling, pos_label_str=1)

    # train the learner on the training set and test on the testing set
    learner.train(train_fs, feature_hasher=use_feature_hashing)
    test_output = learner.evaluate(test_fs, feature_hasher=use_feature_hashing)
    fmeasures = [test_output[2][0]['F-measure'], test_output[2][1]['F-measure']]

    # these are the expected values of the f-measures, sorted
    if not use_feature_hashing:
        expected_fmeasures = [0.7979797979797979, 0.80198019801980192] if not use_scaling else [0.94883720930232551, 0.94054054054054048]
    else:
        expected_fmeasures = [0.83962264150943389, 0.81914893617021278] if not use_scaling else [0.88038277511961716, 0.86910994764397898]

    for expected, actual in zip(expected_fmeasures, fmeasures):
        assert_almost_equal(expected, actual)


def test_scaling():
    yield check_scaling_features, False, False
    yield check_scaling_features, False, True
    yield check_scaling_features, True, False
    yield check_scaling_features, True, True


# Test our kappa implementation based on Ben Hamner's unit tests.
kappa_inputs = [([1, 2, 3], [1, 2, 3]),
                ([1, 2, 1], [1, 2, 2]),
                ([1, 2, 3, 1, 2, 2, 3], [1, 2, 3, 1, 2, 3, 2]),
                ([1, 2, 3, 3, 2, 1], [1, 1, 1, 2, 2, 2]),
                ([-1, 0, 1, 2], [-1, 0, 0, 2]),
                ([5, 6, 7, 8], [5, 6, 6, 8]),
                ([1, 1, 2, 2], [3, 3, 4, 4]),
                ([1, 1, 3, 3], [2, 2, 4, 4]),
                ([1, 1, 4, 4], [2, 2, 3, 3]),
                ([1, 2, 4], [1, 2, 4]),
                ([1, 2, 4], [1, 2, 2])]


def check_kappa(y_true, y_pred, weights, allow_off_by_one, expected):
    assert_almost_equal(kappa(y_true, y_pred, weights=weights,
                              allow_off_by_one=allow_off_by_one), expected)


def test_quadratic_weighted_kappa():
    outputs = [1.0, 0.4, 0.75, 0.0, 0.9, 0.9, 0.11111111, 0.6666666666667, 0.6,
               1.0, 0.4]

    for (y_true, y_pred), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', False, expected

    # Swap y_true and y_pred and test again
    for (y_pred, y_true), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', False, expected


def test_allow_off_by_one_qwk():
    outputs = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.3333333333333333, 1.0, 1.0,
               1.0, 0.5]

    for (y_true, y_pred), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', True, expected

    # Swap y_true and y_pred and test again
    for (y_pred, y_true), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', True, expected


def test_linear_weighted_kappa():
    outputs = [1.0, 0.4, 0.65, 0.0, 0.8, 0.8, 0.0, 0.3333333, 0.3333333, 1.0,
               0.4]

    for (y_true, y_pred), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, 'linear', False, expected

    # Swap y_true and y_pred and test again
    for (y_pred, y_true), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, 'linear', False, expected


def test_unweighted_kappa():
    outputs = [1.0, 0.4, 0.5625, 0.0, 0.6666666666667, 0.6666666666667,
               0.0, 0.0, 0.0, 1.0, 0.5]

    for (y_true, y_pred), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, None, False, expected

    # Swap y_true and y_pred and test again
    for (y_pred, y_true), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, None, False, expected


@raises(ValueError)
def test_invalid_weighted_kappa():
    kappa([1, 2, 1], [1, 2, 1], weights='invalid', allow_off_by_one=False)
    kappa([1, 2, 1], [1, 2, 1], weights='invalid', allow_off_by_one=True)


@raises(ValueError)
def test_invalid_lists_kappa():
    kappa(['a', 'b', 'c'], ['a', 'b', 'c'])


@raises(ValueError)
def check_invalid_regr_grid_obj_func(learner_name, grid_objective_function):
    '''
    Checks whether the grid objective function is
    valid for this regression learner
    '''
    (train_fs, _, _) = make_regression_data()
    clf = Learner(learner_name)
    grid_search_score = clf.train(train_fs, grid_objective=grid_objective_function)


def test_invalid_grid_obj_func():
    for model in ['AdaBoostRegressor', 'DecisionTreeRegressor',
                  'ElasticNet', 'GradientBoostingRegressor',
                  'KNeighborsRegressor', 'Lasso',
                  'LinearRegression', 'RandomForestRegressor',
                  'Ridge', 'SVR', 'SGDRegressor']:
        for metric in ['accuracy',
                       'precision',
                       'recall',
                       'f1',
                       'f1_score_micro',
                       'f1_score_macro',
                       'f1_score_weighted',
                       'f1_score_least_frequent',
                       'average_precision',
                       'roc_auc']:
            yield check_invalid_regr_grid_obj_func, model, metric


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
    classes = []
    for j in range(num_examples):
        y = "dog" if j % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, j) if not numeric_ids else j
        x = {"f{:03d}".format(feat_num): np.random.randint(0, 4) for feat_num
             in range(num_feat_files * num_feats_per_file)}
        x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
        ids.append(ex_id)
        classes.append(y)
        features.append(x)

    # Unmerged
    subset_dict = {}
    for i in range(num_feat_files):
        feat_num = i * num_feats_per_file
        subset_dict['{}'.format(i)] = ["f{:03d}".format(feat_num + j) for j in
                                       range(num_feats_per_file)]
    train_path = join(merge_dir, suffix)
    write_feature_file(train_path, ids, classes, features, subsets=subset_dict)

    # Merged
    train_path = join(merge_dir, 'all{}'.format(suffix))
    write_feature_file(train_path, ids, classes, features)


def check_load_featureset(suffix, numeric_ids):
    num_feat_files = 5

    # Create test data
    make_merging_data(num_feat_files, suffix, numeric_ids)

    # Load unmerged data and merge it
    dirpath = join(_my_dir, 'train', 'test_merging')
    featureset = ['{}'.format(i) for i in range(num_feat_files)]
    merged_examples = _load_featureset(dirpath, featureset, suffix, quiet=True)

    # Load pre-merged data
    featureset = ['all']
    premerged_examples = _load_featureset(dirpath, featureset, suffix,
                                          quiet=True)

    assert_array_equal(merged_examples.ids, premerged_examples.ids)
    assert_array_equal(merged_examples.classes, premerged_examples.classes)
    for (_, _, merged_feats), (_, _, premerged_feats) in zip(merged_examples,
                                                             premerged_examples):
        eq_(merged_feats, premerged_feats)
    eq_(sorted(merged_examples.vectorizer.feature_names_),
        sorted(premerged_examples.vectorizer.feature_names_))

def test_load_featureset():
    # Test merging with numeric IDs
    for suffix in ['.jsonlines', '.ndj', '.megam', '.tsv', '.csv', '.arff']:
        yield check_load_featureset, suffix, True

    for suffix in ['.jsonlines', '.ndj', '.megam', '.tsv', '.csv', '.arff']:
        yield check_load_featureset, suffix, False


def test_ids_to_floats():
    path = join(_my_dir, 'train', 'test_input_2examples_1.jsonlines')

    examples = load_examples(path, ids_to_floats=True, quiet=True)
    assert isinstance(examples.ids[0], float)

    examples = load_examples(path, quiet=True)
    assert not isinstance(examples.ids[0], float)
    assert isinstance(examples.ids[0], str)


def test_convert_examples():
    examples = [{"id": "example0", "y": 1.0, "x": {"f1": 1.0}},
                {"id": "example1", "y": 2.0, "x": {"f1": 1.0, "f2": 1.0}},
                {"id": "example2", "y": 3.0, "x": {"f2": 1.0, "f3": 3.0}}]
    converted = convert_examples(examples)

    eq_(converted.ids[0], "example0")
    eq_(converted.ids[1], "example1")
    eq_(converted.ids[2], "example2")

    eq_(converted.classes[0], 1.0)
    eq_(converted.classes[1], 2.0)
    eq_(converted.classes[2], 3.0)

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
    classes = []
    for j in range(num_examples):
        y = "dog" if j % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, j)
        x = {"f{:03d}".format(feat_num): np.random.randint(0, 4) for feat_num
             in range(num_feat_files * num_feats_per_file)}
        x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
        ids.append(ex_id)
        classes.append(y)
        features.append(x)
    # Create vectorizers/maps for libsvm subset writing
    feat_vectorizer = DictVectorizer()
    feat_vectorizer.fit(features)
    label_map = {label: num for num, label in
                 enumerate(sorted({label for label in classes if
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
        write_feature_file(train_path, ids, classes, sub_features,
                           feat_vectorizer=feat_vectorizer,
                           label_map=label_map)

    # Write out the merged features in the `to_suffix` file format
    train_path = join(convert_dir, '{}_all{}'.format(feature_name_prefix,
                                                     to_suffix))
    write_feature_file(train_path, ids, classes, features,
                       feat_vectorizer=feat_vectorizer, label_map=label_map)


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
    merged_examples = _load_featureset(dirpath, featureset, to_suffix,
                                       quiet=True)

    # Load pre-merged data in the `to_suffix` format
    featureset = ['{}_all'.format(feature_name_prefix)]
    premerged_examples = _load_featureset(dirpath, featureset, to_suffix,
                                          quiet=True)

    # make sure that the pre-generated merged data in the to_suffix format
    # is the same as the converted, merged data in the to_suffix format
    assert_array_equal(merged_examples.ids, premerged_examples.ids)
    assert_array_equal(merged_examples.classes, premerged_examples.classes)
    for (_, _, merged_feats), (_, _, premerged_feats) in zip(merged_examples,
                                                             premerged_examples):
        eq_(merged_feats, premerged_feats)
    eq_(sorted(merged_examples.vectorizer.feature_names_),
        sorted(premerged_examples.vectorizer.feature_names_))


def test_convert_featureset():
    # Test the conversion from every format to every other format
    for from_suffix, to_suffix in itertools.permutations(['.jsonlines', '.ndj',
                                                          '.megam', '.tsv',
                                                          '.csv', '.arff',
                                                          '.libsvm'], 2):
        yield check_convert_featureset, from_suffix, to_suffix


def test_compute_eval_from_predictions():
    pred_path = join(_my_dir, 'other',
                     'test_compute_eval_from_predictions.predictions')
    input_path = join(_my_dir, 'other',
                      'test_compute_eval_from_predictions.jsonlines')

    scores = compute_eval_from_predictions(input_path, pred_path,
                                           ['pearson', 'unweighted_kappa'])

    assert_almost_equal(scores['pearson'], 0.6197797868009122)
    assert_almost_equal(scores['unweighted_kappa'], 0.2)


def test_ablation_cv_sampler():
    '''
    Test to validate whether ablation works with cross-validate and samplers
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation_sampler.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          'ablation_cv_*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_all_combos_sampler():
    '''
    Test to validate whether ablation works with cross-validate
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation_sampler.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True, all_combos=True)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          'ablation_cv_*results')))
    eq_(num_result_files, 20)


def test_ablation_cv_feature_hasher_sampler():
    '''
    Test to validate whether ablation works with cross-validate
    and feature_hasher
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs', ('test_ablation_feature_'
                                                     'hasher_sampler.template'
                                                     '.cfg'))
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(_my_dir, 'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          ('ablation_cv_feature_hasher_'
                                           '*.results'))))
    eq_(num_result_files, 14)


def test_ablation_cv_feature_hasher_all_combos_sampler():
    '''
    Test to validate whether ablation works with cross-validate
    and feature_hasher
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs', ('test_ablation_feature_'
                                                     'hasher_sampler.template'
                                                     '.cfg'))
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True, all_combos=True)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(_my_dir, 'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          ('ablation_cv_feature_hasher_'
                                           '*results'))))
    eq_(num_result_files, 20)

