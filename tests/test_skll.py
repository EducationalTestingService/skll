# Copyright (C) 2012-2013 Educational Testing Service

# This file is part of SciKit-Learn Laboratory.

# SciKit-Learn Laboratory is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

# SciKit-Learn Laboratory is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# SciKit-Learn Laboratory.  If not, see <http://www.gnu.org/licenses/>.

'''
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import json
import os
import re
from collections import OrderedDict
from io import open

import numpy as np
import scipy.sparse as sp
from nose.tools import *

from skll.data import write_feature_file
from skll.experiments import (_load_featureset, run_configuration,
                              _load_cv_folds, _parse_config_file)
from skll.learner import Learner, SelectByMinCount
from skll.metrics import kappa


SCORE_OUTPUT_RE = re.compile(r'Objective function score = ([\-\d\.]+)')
GRID_RE = re.compile(r'Grid search score = ([\-\d\.]+)')
_my_dir = os.path.abspath(os.path.dirname(__file__))


def test_SelectByMinCount():
    m2 = [[0.001,   0.0,    0.0,    0.0],
          [0.00001, -2.0,   0.0,    0.0],
          [0.001,   0.0,    0.0,    4.0],
          [0.0101,  -200.0, 0.0,    0.0]]

    # default should keep all nonzero features (i.e., ones that appear 1+ times)
    feat_selector = SelectByMinCount()
    expected = np.array([[0.001,    0.0,   0.0],
                         [0.00001, -2.0,   0.0],
                         [0.001,   0.0,    4.0],
                         [0.0101,  -200.0, 0.0]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)

    # keep features that happen 2+ times
    feat_selector = SelectByMinCount(min_count=2)
    expected = np.array([[0.001,   0.0],
                         [0.00001, -2.0],
                         [0.001,   0.0],
                         [0.0101,  -200.0]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)

    # keep features that happen 3+ times
    feat_selector = SelectByMinCount(min_count=3)
    expected = np.array([[0.001], [0.00001], [0.001], [0.0101]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)


@raises(ValueError)
def test_input_checking1():
    dirpath = os.path.join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_2examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix)


@raises(ValueError)
def test_input_checking2():
    dirpath = os.path.join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix)


def test_input_checking3():
    dirpath = os.path.join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_2']
    examples_tuple = _load_featureset(dirpath, featureset, suffix)
    assert examples_tuple.features.shape[0] == 3


def make_cv_folds_data():
    train_dir = os.path.join(_my_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    num_examples_per_fold = 100
    num_folds = 3

    with open(os.path.join(train_dir, 'test_cv_folds1.jsonlines'), 'w') as json_out, open(os.path.join(train_dir, 'test_cv_folds1.csv'), 'w') as csv_out:
        csv_out.write('id,fold\n')
        for k in range(num_folds):
            for i in range(num_examples_per_fold):
                y = "dog" if i % 2 == 0 else "cat"
                ex_id = "{}{}".format(y, num_examples_per_fold * k + i)
                x = {"f1": 1.0, "f2": -1.0, "f3": 1.0, "is_{}{}".format(y, k): 1.0}
                json_out.write(json.dumps({"y": y, "id": ex_id, "x": x}) + '\n')
                csv_out.write('{},{}\n'.format(ex_id, k))


def fill_in_config_paths(config_template_path, task='cross-validate'):
    train_dir = os.path.join(_my_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    test_dir = os.path.join(_my_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    output_dir = os.path.join(_my_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = _parse_config_file(config_template_path)

    config.set("Input", "train_location", train_dir)

    to_fill_in = ['log', 'models', 'vocabs', 'predictions']

    if task == 'evaluate' or task == 'cross-validate':
        to_fill_in.append('results')

    for d in to_fill_in:
        config.set("Output", d, os.path.join(output_dir))

    if task == 'cross-validate':
        cv_folds_location = config.get("Input", "cv_folds_location")
        if cv_folds_location:
            config.set("Input", "cv_folds_location", os.path.join(train_dir, cv_folds_location))

    if task == 'predict' or task == 'evaluate':
        config.set("Input", "test_location", test_dir)

    new_config_path = '{}.cfg'.format(re.search(r'^(.*)\.template\.cfg', config_template_path).groups()[0])

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def test_specified_cv_folds():
    make_cv_folds_data()

    # test_cv_folds1.cfg is with prespecified folds and should have about 50% performance
    # test_cv_folds2.cfg is without prespecified folds and should have very high performance
    for config_template_file, test_func, grid_size in [('test_cv_folds1.template.cfg', lambda x: x < 0.6, 3), ('test_cv_folds2.template.cfg', lambda x: x > 0.95, 10)]:
        config_path = fill_in_config_paths(os.path.join(_my_dir, 'configs', config_template_file))

        run_configuration(os.path.join(_my_dir, config_path), local=True)

        with open(os.path.join(_my_dir, 'output', 'train_cv_unscaled_tuned_accuracy_cross-validate_test_cv_folds1_LogisticRegression.results')) as f:
            # check held out scores
            outstr = f.read()
            score = float(SCORE_OUTPUT_RE.search(outstr).groups()[-1])
            assert test_func(score)

            grid_score_matches = GRID_RE.findall(outstr)
            assert len(grid_score_matches) == grid_size
            for match_str in grid_score_matches:
                assert test_func(float(match_str))

    # try the same tests for just training (and specifying the folds for the grid search)
    dirpath = os.path.join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_cv_folds1']
    examples = _load_featureset(dirpath, featureset, suffix)
    clf = Learner(probability=True)
    cv_folds = _load_cv_folds(os.path.join(_my_dir, 'train', 'test_cv_folds1.csv'))
    grid_search_score = clf.train(examples, grid_search_folds=cv_folds, grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score < 0.6
    grid_search_score = clf.train(examples, grid_search_folds=5, grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score > 0.95


def make_regression_data():
    num_examples = 2000
    num_train_examples = int(num_examples / 2)

    np.random.seed(1234567890)
    f1 = np.random.rand(num_examples)
    f2 = np.random.rand(num_examples)
    f3 = np.random.rand(num_examples)
    err = np.random.randn(num_examples) / 2.0
    y = 1.0 * f1 + 1.0 * f2 - 2.0 * f3 + err
    y = y.tolist()

    # Write training file
    train_dir = os.path.join(_my_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    train_path = os.path.join(train_dir, 'test_regression1.jsonlines')
    features = [{"f1": f1[i], "f2": f2[i], "f3": f3[i]} for i in
                range(num_train_examples)]
    write_feature_file(train_path, None, y[:num_train_examples], features)

    # Write test file
    test_dir = os.path.join(_my_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_path = os.path.join(test_dir, 'test_regression1.jsonlines')
    features = [{"f1": f1[i], "f2": f2[i], "f3": f3[i]} for i in
                range(num_train_examples, num_examples)]
    write_feature_file(test_path, None, y[num_train_examples: num_examples],
                       features)

    return y


def test_regression1():
    '''
    This is a bit of a contrived test, but it should fail
    if anything drastic happens to the regression code.
    '''

    y = make_regression_data()

    config_template_path = os.path.join(_my_dir, 'configs', 'test_regression1.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    config_template_path = "test_regression1.cfg"

    run_configuration(os.path.join(_my_dir, config_path), local=True)

    with open(os.path.join(_my_dir, 'output', 'train_cv_unscaled_tuned_pearson_cross-validate_test_regression1_RescaledRidge.results')) as f:
        # check held out scores
        outstr = f.read()
        score = float(SCORE_OUTPUT_RE.search(outstr).groups()[-1])
        assert score > 0.7

    with open(os.path.join(_my_dir, 'output', 'train_cv_unscaled_tuned_pearson_cross-validate_test_regression1_RescaledRidge.predictions'), 'r') as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        pred = [float(row[1]) for row in reader]

        assert np.min(pred) >= np.min(y)
        assert np.max(pred) <= np.max(y)

        assert abs(np.mean(pred) - np.mean(y)) < 0.1
        assert abs(np.std(pred) - np.std(y)) < 0.1


def test_predict():
    '''
    This tests whether predict task runs.
    '''

    y = make_regression_data()

    config_template_path = os.path.join(_my_dir, 'configs', 'test_predict.template.cfg')
    config_path = fill_in_config_paths(config_template_path, task='predict')

    run_configuration(os.path.join(_my_dir, config_path), local=True)

    with open(os.path.join(_my_dir, 'test', 'test_regression1.jsonlines')) as test_file:
        inputs = [x for x in test_file]
        assert len(inputs) == 1000

    with open(os.path.join(_my_dir, 'output', 'train_test_unscaled_tuned_pearson_predict_test_regression1_RescaledRidge.predictions')) as outfile:
        reader = csv.DictReader(outfile, dialect=csv.excel_tab)
        predictions = [x['prediction'] for x in reader]
        assert len(predictions) == len(inputs)


def make_summary_data():
    num_train_examples = 500
    num_test_examples = 100

    np.random.seed(1234567890)

    # Write training file
    train_path = os.path.join(_my_dir, 'train', 'test_summary.jsonlines')
    classes = []
    ids = []
    features = []
    for i in range(num_train_examples):
        y = "dog" if i % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, i)
        x = {"f1": np.random.randint(1, 4), "f2": np.random.randint(1, 4),
             "f3": np.random.randint(1, 4)}
        classes.append(y)
        ids.append(ex_id)
        features.append(x)
    write_feature_file(train_path, ids, classes, features)

    # Write test file
    test_path = os.path.join(_my_dir, 'test', 'test_summary.jsonlines')
    classes = []
    ids = []
    features = []
    for i in range(num_test_examples):
        y = "dog" if i % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, i)
        x = {"f1": np.random.randint(1, 4), "f2": np.random.randint(1, 4),
             "f3": np.random.randint(1, 4)}
        classes.append(y)
        ids.append(ex_id)
        features.append(x)
    write_feature_file(test_path, ids, classes, features)


def check_summary_score(result_score, summary_score, learner_name):
    eq_(result_score, summary_score, msg='mismatched scores for {} (result:{}, summary:{})'.format(learner_name, result_score, summary_score))


def test_summary():
    '''
    Test to validate summary file scores
    '''
    make_summary_data()

    config_template_path = os.path.join(_my_dir, 'configs', 'test_summary.template.cfg')
    config_path = fill_in_config_paths(config_template_path, task='evaluate')

    run_configuration(config_path, local=True)

    with open(os.path.join(_my_dir, 'output', 'train_test_unscaled_tuned_accuracy_evaluate_test_summary_LogisticRegression.results')) as f:
        outstr = f.read()
        logistic_result_score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])

    with open(os.path.join(_my_dir, 'output', 'train_test_unscaled_tuned_accuracy_evaluate_test_summary_MultinomialNB.results')) as f:
        outstr = f.read()
        naivebayes_result_score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])

    with open(os.path.join(_my_dir, 'output', 'train_test_unscaled_tuned_accuracy_evaluate_test_summary_SVC.results')) as f:
        outstr = f.read()
        svm_result_score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])

    with open(os.path.join(_my_dir, 'output', 'train_test_unscaled_tuned_accuracy_evaluate_summary.tsv'), 'r') as f:
        reader = csv.DictReader(f, dialect='excel-tab')

        for row in reader:
            if row['given_learner'] == 'LogisticRegression':
                logistic_summary_score = float(row['score'])
            elif row['given_learner'] == 'MultinomialNB':
                naivebayes_summary_score = float(row['score'])
            elif row['given_learner'] == 'SVC':
                svm_summary_score = float(row['score'])

    for result_score, summary_score, learner_name in [(logistic_result_score, logistic_summary_score, 'LogisticRegression'), (naivebayes_result_score, naivebayes_summary_score, 'MultinomialNB'), (svm_result_score, svm_summary_score, 'SVC')]:
        yield check_summary_score, result_score, summary_score, learner_name


def make_sparse_data():
    # Create training file
    train_path = os.path.join(_my_dir, 'train', 'test_sparse.jsonlines')
    ids = []
    classes = []
    features = []
    for i in range(1, 101):
        y = "dog" if i % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, i)
        # note that f1 and f5 are missing in all instances but f4 is not
        x = {"f2": i+1, "f3": i+2, "f4": i+5}
        ids.append(ex_id)
        classes.append(y)
        features.append(x)
    write_feature_file(train_path, ids, classes, features)

    # Create test file
    test_path = os.path.join(_my_dir, 'test', 'test_sparse.jsonlines')
    ids = []
    classes = []
    features = []
    for i in range(1, 51):
        y = "dog" if i % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, i)
        # f1 and f5 are not missing in any instances here but f4 is
        x = {"f1": i, "f2": i+2, "f3": i % 10, "f5": i * 2}
        ids.append(ex_id)
        classes.append(y)
        features.append(x)
    write_feature_file(test_path, ids, classes, features)


def test_sparse_predict():
    '''
    Test to validate whether predict works with sparse data
    '''
    make_sparse_data()

    config_template_path = os.path.join(_my_dir, 'configs', 'test_sparse.template.cfg')
    config_path = fill_in_config_paths(config_template_path, task='evaluate')

    run_configuration(config_path, local=True)

    with open(os.path.join(_my_dir, 'output', 'train_test_unscaled_untuned_evaluate_test_sparse_LogisticRegression.results')) as f:
        outstr = f.read()
        logistic_result_score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])

    assert_almost_equal(logistic_result_score, 0.5)


# Test our kappa implementation based on Ben Hamner's unit tests.
kappa_inputs = [([1, 2, 3], [1, 2, 3]),
                ([1, 2, 1], [1, 2, 2]),
                ([1, 2, 3, 1, 2, 2, 3], [1, 2, 3, 1, 2, 3, 2]),
                ([1, 2, 3, 3, 2, 1], [1, 1, 1, 2, 2, 2]),
                ([-1, 0, 1, 2], [-1, 0, 0, 2]),
                ([5, 6, 7, 8], [5, 6, 6, 8])]


def check_kappa(y_true, y_pred, weights, expected):
    assert_almost_equal(kappa(y_true, y_pred, weights), expected)


def test_quadratic_weighted_kappa():
    outputs = [1.0, 0.4, 0.75, 0.0, 0.9, 0.9]

    for (y_true, y_pred), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, 'quadratic', expected


def test_linear_weighted_kappa():
    outputs = [1.0, 0.4, 0.65, 0.0, 0.8, 0.8]

    for (y_true, y_pred), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, 'linear', expected


def test_unweighted_kappa():
    outputs = [1.0, 0.4, 0.5625, 0.0, 0.6666666666667, 0.6666666666667]

    for (y_true, y_pred), expected in zip(kappa_inputs, outputs):
        yield check_kappa, y_true, y_pred, None, expected


# Tests related to loading featuresets and merging them
def make_merging_data(num_feat_files, suffix):
    num_examples = 500
    num_feats_per_file = 17

    np.random.seed(1234567890)

    merge_dir = os.path.join(_my_dir, 'train', 'test_merging')
    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir)

    # Create lists we will write files from
    ids = []
    features = []
    classes = []
    for j in range(num_examples):
        y = "dog" if j % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, j)
        x = {"f{:03d}".format(feat_num): np.random.randint(0, 4) for feat_num in
             range(num_feat_files * num_feats_per_file)}
        x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
        ids.append(ex_id)
        classes.append(y)
        features.append(x)

    # Unmerged
    for i in range(num_feat_files):
        train_path = os.path.join(merge_dir, '{}{}'.format(i, suffix))
        sub_features = []
        for example_num in range(num_examples):
            feat_num = i * num_feats_per_file
            x = {"f{:03d}".format(feat_num + j): features[example_num]["f{:03d}".format(feat_num + j)] for j in range(num_feats_per_file)}
            sub_features.append(x)
        write_feature_file(train_path, ids, classes, sub_features)

    # Merged
    train_path = os.path.join(merge_dir, 'all{}'.format(suffix))
    write_feature_file(train_path, ids, classes, features)


def check_load_featureset(suffix):
    num_feat_files = 5

    # Create test data
    make_merging_data(num_feat_files, suffix)

    # Load unmerged data and merge it
    dirpath = os.path.join(_my_dir, 'train', 'test_merging')
    featureset = [str(i) for i in range(num_feat_files)]
    merged_examples = _load_featureset(dirpath, featureset, suffix)

    # Load pre-merged data
    featureset = ['all']
    premerged_examples = _load_featureset(dirpath, featureset, suffix)

    assert np.all(merged_examples.ids == premerged_examples.ids)
    assert np.all(merged_examples.classes == premerged_examples.classes)
    assert np.all(merged_examples.features.todense() ==
                  premerged_examples.features.todense())
    eq_(merged_examples.feat_vectorizer.feature_names_,
        premerged_examples.feat_vectorizer.feature_names_)
    eq_(merged_examples.feat_vectorizer.vocabulary_,
        premerged_examples.feat_vectorizer.vocabulary_)


def test_load_featureset():
    for suffix in ['.jsonlines', '.megam', '.tsv']:
        yield check_load_featureset, suffix
