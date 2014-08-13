# License: BSD 3 clause
'''
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import itertools
import json
import glob
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
from sklearn.feature_extraction import DictVectorizer

from skll.data import write_feature_file, load_examples, convert_examples
from skll.experiments import (_load_featureset, run_configuration,
                              _load_cv_folds, _setup_config_parser,
                              run_ablation)
from skll.learner import Learner, SelectByMinCount
from skll.metrics import kappa
from skll.utilities import skll_convert
from skll.utilities.compute_eval_from_predictions import compute_eval_from_predictions


SCORE_OUTPUT_RE = re.compile(r'Objective Function Score \(Test\) = '
                             r'([\-\d\.]+)')
GRID_RE = re.compile(r'Grid Objective Score \(Train\) = ([\-\d\.]+)')
_my_dir = abspath(dirname(__file__))


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
    Ensure that we raise ValueError when trying to join featuresets with
    different number of examples.
    '''
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_2examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix, quiet=True)


@raises(ValueError)
def test_input_checking2():
    '''
    Ensure that we raise ValueError when trying to join featuresets
    that contain the same features for each instance.
    '''
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix, quiet=True)


def test_input_checking3():
    '''
    Small test to ensure that we correctly merge featuresets.
    '''
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_2']
    examples_tuple = _load_featureset(dirpath, featureset, suffix, quiet=True)
    assert examples_tuple.features.shape[0] == 3


def make_cv_folds_data(numeric_ids):
    ''' Create input files for pre-specified CV folds tests '''
    train_dir = join(_my_dir, 'train')
    if not exists(train_dir):
        os.makedirs(train_dir)

    num_examples_per_fold = 100
    num_folds = 3

    json_path = join(train_dir, 'test_cv_folds.jsonlines')
    csv_path = join(train_dir, 'test_cv_folds.csv')

    with open(json_path, 'w') as json_out, open(csv_path, 'w') as csv_out:
        csv_out.write('id,fold\n')
        for k in range(num_folds):
            for i in range(num_examples_per_fold):
                y = "dog" if i % 2 == 0 else "cat"
                if numeric_ids:
                    ex_id = num_examples_per_fold * k + i
                else:
                    ex_id = "{}{}".format(y, num_examples_per_fold * k + i)
                x = {"f1": 1.0, "f2": -1.0, "f3": 1.0,
                     "is_{}{}".format(y, k): 1.0}
                json_out.write(json.dumps({"y": y, "id": ex_id, "x": x}) +
                               '\n')
                csv_out.write('{},{}\n'.format(ex_id, k))


def fill_in_config_paths(config_template_path):
    '''
    Add paths to train, test, and output directories to a given config template
    file.
    '''
    train_dir = join(_my_dir, 'train')
    if not exists(train_dir):
        os.makedirs(train_dir)
    test_dir = join(_my_dir, 'test')
    if not exists(test_dir):
        os.makedirs(test_dir)
    output_dir = join(_my_dir, 'output')
    if not exists(output_dir):
        os.makedirs(output_dir)

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


def check_specified_cv_folds_feature_hasher(numeric_ids):
    make_cv_folds_data(numeric_ids)

    # test_cv_folds1.cfg has prespecified folds and should have ~50% accuracy
    # test_cv_folds2.cfg doesn't have prespecified folds and >95% accuracy
    for experiment_name, test_func, grid_size in [('test_cv_folds1',
                                                   lambda x: x < 0.6,
                                                   3),
                                                  ('test_cv_folds2_feature_hasher',
                                                   lambda x: x > 0.95,
                                                   10)]:
        config_template_file = '{}.template.cfg'.format(experiment_name)
        config_template_path = join(_my_dir, 'configs', config_template_file)
        config_path = join(_my_dir, fill_in_config_paths(config_template_path))

        # Modify config file to change ids_to_floats depending on numeric_ids
        # setting
        with open(config_path, 'r+') as config_template_file:
            lines = config_template_file.readlines()
            config_template_file.seek(0)
            config_template_file.truncate()
            for line in lines:
                if line.startswith('ids_to_floats='):
                    if numeric_ids:
                        line = 'ids_to_floats=true\n'
                    else:
                        line = 'ids_to_floats=false\n'
                config_template_file.write(line)

        run_configuration(config_path, quiet=True)
        result_filename = ('{}_test_cv_folds_LogisticRegression.' +
                           'results').format(experiment_name)
        with open(join(_my_dir, 'output', result_filename)) as f:
            # check held out scores
            outstr = f.read()
            score = float(SCORE_OUTPUT_RE.search(outstr).groups()[-1])
            assert test_func(score)

            grid_score_matches = GRID_RE.findall(outstr)
            assert len(grid_score_matches) == grid_size
            for match_str in grid_score_matches:
                assert test_func(float(match_str))

    # try the same tests for just training (and specifying the folds for the
    # grid search)
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_cv_folds']
    examples = _load_featureset(dirpath, featureset, suffix, quiet=True)
    clf = Learner('LogisticRegression', probability=True)
    cv_folds = _load_cv_folds(join(_my_dir, 'train', 'test_cv_folds.csv'))
    grid_search_score = clf.train(examples, grid_search_folds=cv_folds,
                                  grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score < 0.6
    grid_search_score = clf.train(examples, grid_search_folds=5,
                                  grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score > 0.95


def check_specified_cv_folds(numeric_ids):
    make_cv_folds_data(numeric_ids)

    # test_cv_folds1.cfg has prespecified folds and should have ~50% accuracy
    # test_cv_folds2.cfg doesn't have prespecified folds and >95% accuracy
    for experiment_name, test_func, grid_size in [('test_cv_folds1',
                                                   lambda x: x < 0.6,
                                                   3),
                                                  ('test_cv_folds2',
                                                   lambda x: x > 0.95,
                                                   10)]:
        config_template_file = '{}.template.cfg'.format(experiment_name)
        config_template_path = join(_my_dir, 'configs', config_template_file)
        config_path = join(_my_dir, fill_in_config_paths(config_template_path))

        # Modify config file to change ids_to_floats depending on numeric_ids
        # setting
        with open(config_path, 'r+') as config_template_file:
            lines = config_template_file.readlines()
            config_template_file.seek(0)
            config_template_file.truncate()
            for line in lines:
                if line.startswith('ids_to_floats='):
                    if numeric_ids:
                        line = 'ids_to_floats=true\n'
                    else:
                        line = 'ids_to_floats=false\n'
                config_template_file.write(line)

        run_configuration(config_path, quiet=True)
        result_filename = ('{}_test_cv_folds_LogisticRegression.' +
                           'results').format(experiment_name)
        with open(join(_my_dir, 'output', result_filename)) as f:
            # check held out scores
            outstr = f.read()
            score = float(SCORE_OUTPUT_RE.search(outstr).groups()[-1])
            assert test_func(score)

            grid_score_matches = GRID_RE.findall(outstr)
            assert len(grid_score_matches) == grid_size
            for match_str in grid_score_matches:
                assert test_func(float(match_str))

    # try the same tests for just training (and specifying the folds for the
    # grid search)
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_cv_folds']
    examples = _load_featureset(dirpath, featureset, suffix, quiet=True)
    clf = Learner('LogisticRegression', probability=True)
    cv_folds = _load_cv_folds(join(_my_dir, 'train', 'test_cv_folds.csv'))
    grid_search_score = clf.train(examples, grid_search_folds=cv_folds,
                                  grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score < 0.6
    grid_search_score = clf.train(examples, grid_search_folds=5,
                                  grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score > 0.95


def test_specified_cv_folds_feature_hasher():
    yield check_specified_cv_folds_feature_hasher, False
    yield check_specified_cv_folds_feature_hasher, True


def test_specified_cv_folds():
    yield check_specified_cv_folds, False
    yield check_specified_cv_folds, True


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
    train_dir = join(_my_dir, 'train')
    if not exists(train_dir):
        os.makedirs(train_dir)
    train_path = join(train_dir, 'test_regression1.jsonlines')
    features = [{"f1": f1[i], "f2": f2[i], "f3": f3[i]} for i in
                range(num_train_examples)]
    write_feature_file(train_path, None, y[:num_train_examples], features)

    # Write test file
    test_dir = join(_my_dir, 'test')
    if not exists(test_dir):
        os.makedirs(test_dir)
    test_path = join(test_dir, 'test_regression1.jsonlines')
    features = [{"f1": f1[i], "f2": f2[i], "f3": f3[i]} for i in
                range(num_train_examples, num_examples)]
    write_feature_file(test_path, None, y[num_train_examples: num_examples],
                       features)
    return y


def test_regression1_feature_hasher():
    '''
    This is a bit of a contrived test, but it should fail
    if anything drastic happens to the regression code.
    '''

    y = make_regression_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_regression1_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    config_template_path = "test_regression1_feature_hasher.cfg"

    run_configuration(join(_my_dir, config_path), quiet=True)

    with open(join(_my_dir, 'output',
                   ('test_regression1_feature_hasher_test_regression1_'
                    'RescaledRidge.results'))) as f:
        # check held out scores
        outstr = f.read()
        score = float(SCORE_OUTPUT_RE.search(outstr).groups()[-1])
        assert score > 0.7

    with open(join(_my_dir, 'output', ('test_regression1_feature_hasher_'
                                       'test_regression1_RescaledRidge'
                                       '.predictions')), 'r') as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        pred = [float(row[1]) for row in reader]

        assert np.min(pred) >= np.min(y)
        assert np.max(pred) <= np.max(y)

        assert abs(np.mean(pred) - np.mean(y)) < 0.1
        assert abs(np.std(pred) - np.std(y)) < 0.1


def test_regression1():
    '''
    This is a bit of a contrived test, but it should fail
    if anything drastic happens to the regression code.
    '''

    y = make_regression_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_regression1.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    config_template_path = "test_regression1.cfg"

    run_configuration(join(_my_dir, config_path), quiet=True)

    with open(join(_my_dir, 'output', ('test_regression1_test_regression1'
                                       '_RescaledRidge.results'))) as f:
        # check held out scores
        outstr = f.read()
        score = float(SCORE_OUTPUT_RE.search(outstr).groups()[-1])
        assert score > 0.7

    with open(join(_my_dir, 'output', ('test_regression1_test_regression1_'
                                       'RescaledRidge.predictions')),
              'r') as f:
        reader = csv.reader(f, dialect='excel-tab')
        next(reader)
        pred = [float(row[1]) for row in reader]

        assert np.min(pred) >= np.min(y)
        assert np.max(pred) <= np.max(y)

        assert abs(np.mean(pred) - np.mean(y)) < 0.1
        assert abs(np.std(pred) - np.std(y)) < 0.1


def test_predict_feature_hasher():
    '''
    This tests whether predict task runs for feature_hasher.
    '''

    make_regression_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_predict_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(join(_my_dir, config_path), quiet=True)

    with open(join(_my_dir, 'test', 'test_regression1.jsonlines')) as test_file:
        inputs = [x for x in test_file]
        assert len(inputs) == 1000

    with open(join(_my_dir, 'output', ('test_predict_feature_hasher_test'
                                       '_regression1_RescaledRidge'
                                       '.predictions'))) as outfile:
        reader = csv.DictReader(outfile, dialect=csv.excel_tab)
        predictions = [x['prediction'] for x in reader]
        assert len(predictions) == len(inputs)


def test_predict():
    '''
    This tests whether predict task runs.
    '''

    make_regression_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_predict.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(join(_my_dir, config_path), quiet=True)

    with open(join(_my_dir, 'test',
                   'test_regression1.jsonlines')) as test_file:
        inputs = [x for x in test_file]
        assert len(inputs) == 1000

    with open(join(_my_dir, 'output', ('test_predict_test_regression1_'
                                       'RescaledRidge'
                                       '.predictions'))) as outfile:
        reader = csv.DictReader(outfile, dialect=csv.excel_tab)
        predictions = [x['prediction'] for x in reader]
        assert len(predictions) == len(inputs)


def make_summary_data():
    num_train_examples = 500
    num_test_examples = 100

    np.random.seed(1234567890)

    # Write training file
    train_path = join(_my_dir, 'train', 'test_summary.jsonlines')
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
    test_path = join(_my_dir, 'test', 'test_summary.jsonlines')
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
    eq_(result_score, summary_score, msg=('mismatched scores for {} '
                                          '(result:{}, summary:'
                                          '{})').format(learner_name,
                                                        result_score,
                                                        summary_score))


def test_summary_feature_hasher():
    '''
    Test to validate summary file scores with feature_hasher
    '''
    make_summary_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_summary_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output', ('test_summary_feature_hasher_test_'
                                       'summary_LogisticRegression'
                                       '.results'))) as f:
        outstr = f.read()
        logistic_result_score_str = SCORE_OUTPUT_RE.search(outstr).groups()[0]
        logistic_result_score = float(logistic_result_score_str)

    with open(join(_my_dir, 'output', ('test_summary_feature_hasher_test_'
                                       'summary_SVC.results'))) as f:
        outstr = f.read()
        svm_result_score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])

    with open(join(_my_dir, 'output',
                   'test_summary_feature_hasher_summary.tsv'), 'r') as f:
        reader = csv.DictReader(f, dialect='excel-tab')

        for row in reader:
            # the learner results dictionaries should have 19 rows,
            # and all of these except results_table
            # should be printed (though some columns will be blank).
            eq_(len(row), 19)
            assert row['model_params']
            assert row['grid_score']
            assert row['score']

            if row['learner_name'] == 'LogisticRegression':
                logistic_summary_score = float(row['score'])
            elif row['learner_name'] == 'SVC':
                svm_summary_score = float(row['score'])

    for result_score, summary_score, learner_name in [(logistic_result_score,
                                                       logistic_summary_score,
                                                       'LogisticRegression'),
                                                      (svm_result_score,
                                                       svm_summary_score,
                                                       'SVC')]:
        yield check_summary_score, result_score, summary_score, learner_name


def test_summary():
    '''
    Test to validate summary file scores
    '''
    make_summary_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_summary.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output', ('test_summary_test_summary_'
                                       'LogisticRegression.results'))) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

    with open(join(_my_dir, 'output', ('test_summary_test_summary_'
                                       'MultinomialNB.results'))) as f:
        outstr = f.read()
        naivebayes_score_str = SCORE_OUTPUT_RE.search(outstr).groups()[0]
        naivebayes_result_score = float(naivebayes_score_str)

    with open(join(_my_dir, 'output',
                   'test_summary_test_summary_SVC.results')) as f:
        outstr = f.read()
        svm_result_score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])

    with open(join(_my_dir, 'output', 'test_summary_summary.tsv'), 'r') as f:
        reader = csv.DictReader(f, dialect='excel-tab')

        for row in reader:
            # the learner results dictionaries should have 19 rows,
            # and all of these except results_table
            # should be printed (though some columns will be blank).
            eq_(len(row), 19)
            assert row['model_params']
            assert row['grid_score']
            assert row['score']

            if row['learner_name'] == 'LogisticRegression':
                logistic_summary_score = float(row['score'])
            elif row['learner_name'] == 'MultinomialNB':
                naivebayes_summary_score = float(row['score'])
            elif row['learner_name'] == 'SVC':
                svm_summary_score = float(row['score'])

    for result_score, summary_score, learner_name in [(logistic_result_score,
                                                       logistic_summary_score,
                                                       'LogisticRegression'),
                                                      (naivebayes_result_score,
                                                       naivebayes_summary_score,
                                                       'MultinomialNB'),
                                                      (svm_result_score,
                                                       svm_summary_score,
                                                       'SVC')]:
        yield check_summary_score, result_score, summary_score, learner_name


def test_backward_compatibility():
    '''
    Verify that a model from v0.9.17 can still be loaded and generate the same predictions.
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


def make_sparse_data():
    # Create training file
    train_path = join(_my_dir, 'train', 'test_sparse.jsonlines')
    ids = []
    classes = []
    features = []
    for i in range(1, 101):
        y = "dog" if i % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, i)
        # note that f1 and f5 are missing in all instances but f4 is not
        x = {"f2": i + 1, "f3": i + 2, "f4": i + 5}
        ids.append(ex_id)
        classes.append(y)
        features.append(x)
    write_feature_file(train_path, ids, classes, features)

    # Create test file
    test_path = join(_my_dir, 'test', 'test_sparse.jsonlines')
    ids = []
    classes = []
    features = []
    for i in range(1, 51):
        y = "dog" if i % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, i)
        # f1 and f5 are not missing in any instances here but f4 is
        x = {"f1": i, "f2": i + 2, "f3": i % 10, "f5": i * 2}
        ids.append(ex_id)
        classes.append(y)
        features.append(x)
    write_feature_file(test_path, ids, classes, features)


def test_sparse_feature_hasher_predict():
    '''
    Test to validate whether predict works with sparse data
    and feature_hasher
    '''
    make_sparse_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_sparse_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output', ('test_sparse_test_sparse_Logistic'
                                       'Regression.results'))) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

    assert_almost_equal(logistic_result_score, 0.5)


def test_sparse_predict():
    '''
    Test to validate whether predict works with sparse data
    '''
    make_sparse_data()

    config_template_path = join(_my_dir, 'configs', 'test_sparse.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output', ('test_sparse_test_sparse_Logistic'
                                       'Regression.results'))) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

    assert_almost_equal(logistic_result_score, 0.5)


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
    write_feature_file(train_path, ids, classes, features)

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
    write_feature_file(test_path, ids, classes, features)


def test_class_map():
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
        write_feature_file(train_path, ids, classes, sub_features)


def test_ablation_cv():
    '''
    Test to validate whether ablation works with cross-validate
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True)

    # read in the summary file and make sure it has
    # 6 ablated featuresets * (10 folds + 1 average line) * 2 learners = 132
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        all_rows = list(reader)
        eq_(len(all_rows), 132)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          'ablation_cv_*.results')))
    eq_(num_result_files, 12)


def test_ablation_cv_all_combos():
    '''
    Test to validate whether ablation works with cross-validate
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True, all_combos=True)

    # read in the summary file and make sure it has
    # 31 ablated featuresets * (10 folds + 1 average line) * 2 learners = 682
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        all_rows = list(reader)
        eq_(len(all_rows), 682)

    # make sure there are 31 ablated featuresets * 2 learners = 62 results
    # files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          'ablation_cv_*results')))
    eq_(num_result_files, 62)


def test_ablation_cv_feature_hasher():
    '''
    Test to validate whether ablation works with cross-validate
    and feature_hasher
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True)

    # read in the summary file and make sure it has
    # 6 ablated featuresets * (10 folds + 1 average line) * 2 learners = 132
    # lines
    with open(join(_my_dir, 'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        all_rows = list(reader)
        eq_(len(all_rows), 132)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          ('ablation_cv_feature_hasher_'
                                           '*.results'))))
    eq_(num_result_files, 12)


def test_ablation_cv_feature_hasher_all_combos():
    '''
    Test to validate whether ablation works with cross-validate
    and feature_hasher
    '''
    make_ablation_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_ablation_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_ablation(config_path, quiet=True, all_combos=True)

    # read in the summary file and make sure it has
    # 31 ablated featuresets * (10 folds + 1 average line) * 2 learners = 682
    # lines
    with open(join(_my_dir, 'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        all_rows = list(reader)
        eq_(len(all_rows), 682)

    # make sure there are 31 ablated featuresets * 2 learners = 62 results
    # files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          ('ablation_cv_feature_hasher_'
                                           '*results'))))
    eq_(num_result_files, 62)


def make_scaling_data():
    num_train_examples = 1000
    num_test_examples = 100

    np.random.seed(1234567890)

    # create training data
    ids = []
    features = []
    classes = []
    for j in range(num_train_examples):
        y = "dog" if j % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, j)
        x = {"g{}".format(feat_num): np.random.randint(0, 4) for feat_num in
             range(5)}
        x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
        ids.append(ex_id)
        classes.append(y)
        features.append(x)

    for i in range(5):
        train_path = join(_my_dir, 'train', 'g{}.jsonlines'.format(i))
        sub_features = []
        for example_num in range(num_train_examples):
            feat_num = i
            x = {"g{}".format(feat_num):
                 features[example_num]["g{}".format(feat_num)]}
            sub_features.append(x)
        write_feature_file(train_path, ids, classes, sub_features)

    # create the test data
    for j in range(num_test_examples):
        y = "dog" if j % 2 == 0 else "cat"
        ex_id = "{}{}".format(y, j)
        x = {"g{}".format(feat_num): np.random.randint(0, 4) for feat_num in
             range(5)}
        x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
        ids.append(ex_id)
        classes.append(y)
        features.append(x)

    for i in range(5):
        train_path = join(_my_dir, 'test', 'g{}.jsonlines'.format(i))
        sub_features = []
        for example_num in range(num_test_examples):
            feat_num = i
            x = {"g{}".format(feat_num):
                 features[example_num]["g{}".format(feat_num)]}
            sub_features.append(x)
        write_feature_file(train_path, ids, classes, sub_features)


def test_scaling_feature_hasher():
    '''
    Test to validate whether feature scaling works
    using the feature_hasher option
    '''
    make_scaling_data()

    # run the experiment without scaling
    config_template_path = join(_my_dir, 'configs', ('test_scaling_without_'
                                                     'feature_hasher.template'
                                                     '.cfg'))
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    # now run the version with scaling
    config_template_path = join(_my_dir, 'configs', ('test_scaling_with_'
                                                     'feature_hasher.template'
                                                     '.cfg'))
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    # make sure that the result with and without scaling aren't the same
    with open(join(_my_dir, 'output',
                   'without_scaling_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        row = list(reader)[0]
        without_scaling_score = row['score']
        without_scaling_scaling_value = row['feature_scaling']

    with open(join(_my_dir, 'output',
                   'with_scaling_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        row = list(reader)[0]
        with_scaling_score = row['score']
        with_scaling_scaling_value = row['feature_scaling']

    assert_not_equal(without_scaling_score, with_scaling_score)
    eq_(without_scaling_scaling_value, 'none')
    eq_(with_scaling_scaling_value, 'both')


def test_scaling():
    '''
    Test to validate whether feature scaling works
    '''
    make_scaling_data()

    # run the experiment without scaling
    config_template_path = join(_my_dir, 'configs',
                                'test_scaling_without.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    # now run the version with scaling
    config_template_path = join(_my_dir, 'configs',
                                'test_scaling_with.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    # make sure that the result with and without scaling aren't the same
    with open(join(_my_dir, 'output', 'without_scaling_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        row = list(reader)[0]
        without_scaling_score = row['score']
        without_scaling_scaling_value = row['feature_scaling']

    with open(join(_my_dir, 'output', 'with_scaling_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        row = list(reader)[0]
        with_scaling_score = row['score']
        with_scaling_scaling_value = row['feature_scaling']

    assert_not_equal(without_scaling_score, with_scaling_score)
    eq_(without_scaling_scaling_value, 'none')
    eq_(with_scaling_scaling_value, 'both')


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

    assert np.all(merged_examples.ids == premerged_examples.ids)
    assert np.all(merged_examples.classes == premerged_examples.classes)
    assert np.all(merged_examples.features.todense() ==
                  premerged_examples.features.todense())
    eq_(merged_examples.feat_vectorizer.feature_names_,
        premerged_examples.feat_vectorizer.feature_names_)
    eq_(merged_examples.feat_vectorizer.vocabulary_,
        premerged_examples.feat_vectorizer.vocabulary_)


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

    eq_(converted.feat_vectorizer.get_feature_names(), ['f1', 'f2', 'f3'])


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
    assert_array_equal(merged_examples.features.todense(),
                       premerged_examples.features.todense())
    eq_(merged_examples.feat_vectorizer.feature_names_,
        premerged_examples.feat_vectorizer.feature_names_)
    eq_(merged_examples.feat_vectorizer.vocabulary_,
        premerged_examples.feat_vectorizer.vocabulary_)


def test_convert_featureset():
    ''' Test the conversion from every format to every other format '''
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
    # 6 ablated featuresets * (10 folds + 1 average line) * 2 learners = 132
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        all_rows = list(reader)
        eq_(len(all_rows), 132)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          'ablation_cv_*.results')))
    eq_(num_result_files, 12)


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
    # 31 ablated featuresets * (10 folds + 1 average line) * 2 learners = 682
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        all_rows = list(reader)
        eq_(len(all_rows), 682)

    # make sure there are 31 ablated featuresets * 2 learners = 62 results
    # files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          'ablation_cv_*results')))
    eq_(num_result_files, 62)


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
    # 6 ablated featuresets * (10 folds + 1 average line) * 2 learners = 132
    # lines
    with open(join(_my_dir, 'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        all_rows = list(reader)
        eq_(len(all_rows), 132)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          ('ablation_cv_feature_hasher_'
                                           '*.results'))))
    eq_(num_result_files, 12)


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
    # 31 ablated featuresets * (10 folds + 1 average line) * 2 learners = 682
    # lines
    with open(join(_my_dir, 'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        all_rows = list(reader)
        eq_(len(all_rows), 682)

    # make sure there are 31 ablated featuresets * 2 learners = 62 results
    # files
    num_result_files = len(glob.glob(join(_my_dir, 'output',
                                          ('ablation_cv_feature_hasher_'
                                           '*results'))))
    eq_(num_result_files, 62)


def test_sparse_feature_hasher_predict_sampler():
    '''
    Test to validate whether predict works with sparse data
    and feature_hasher
    '''
    make_sparse_data()

    config_template_path = join(_my_dir, 'configs', ('test_sparse_feature_'
                                                     'hasher_sampler.template'
                                                     '.cfg'))
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output',
                   'test_sparse_test_sparse_LogisticRegression.results')) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

    assert_almost_equal(logistic_result_score, 0.5)


def test_sparse_predict_sampler():
    '''
    Test to validate whether predict works with sparse data
    '''
    make_sparse_data()

    config_template_path = join(_my_dir, 'configs',
                                'test_sparse_sampler.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output',
                   'test_sparse_test_sparse_LogisticRegression.results')) as f:
        outstr = f.read()
        logistic_result_score = float(
            SCORE_OUTPUT_RE.search(outstr).groups()[0])

    assert_almost_equal(logistic_result_score, 0.5, delta=0.025)


def check_specified_cv_folds_feature_hasher_sampler(numeric_ids):
    make_cv_folds_data(numeric_ids)

    # test_cv_folds1.cfg has prespecified folds and should have ~50% accuracy
    # test_cv_folds2.cfg doesn't have prespecified folds and >95% accuracy
    for experiment_name, test_func, grid_size in [(('test_cv_folds1_feature_'
                                                    'hasher_sampler'),
                                                   lambda x: x < 0.6,
                                                   3),
                                                  (('test_cv_folds2_feature_'
                                                    'hasher_sampler'),
                                                   lambda x: x > 0.95,
                                                   10)]:
        config_template_file = '{}.template.cfg'.format(experiment_name)
        config_template_path = join(_my_dir, 'configs', config_template_file)
        config_path = join(_my_dir, fill_in_config_paths(config_template_path))

        # Modify config file to change ids_to_floats depending on numeric_ids
        # setting
        with open(config_path, 'r+') as config_template_file:
            lines = config_template_file.readlines()
            config_template_file.seek(0)
            config_template_file.truncate()
            for line in lines:
                if line.startswith('ids_to_floats='):
                    if numeric_ids:
                        line = 'ids_to_floats=true\n'
                    else:
                        line = 'ids_to_floats=false\n'
                config_template_file.write(line)

        run_configuration(config_path, quiet=True)
        result_filename = ('{}_test_cv_folds_LogisticRegression.'
                           'results').format(experiment_name)
        with open(join(_my_dir, 'output', result_filename)) as f:
            # check held out scores
            outstr = f.read()
            score = float(SCORE_OUTPUT_RE.search(outstr).groups()[-1])
            assert test_func(score)

            grid_score_matches = GRID_RE.findall(outstr)
            assert len(grid_score_matches) == grid_size
            for match_str in grid_score_matches:
                assert test_func(float(match_str))

    # try the same tests for just training (and specifying the folds for the
    # grid search)
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_cv_folds']
    examples = _load_featureset(dirpath, featureset, suffix, quiet=True)
    clf = Learner('LogisticRegression', probability=True)
    cv_folds = _load_cv_folds(join(_my_dir, 'train', 'test_cv_folds.csv'))
    grid_search_score = clf.train(examples, grid_search_folds=cv_folds,
                                  grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score < 0.6
    grid_search_score = clf.train(examples, grid_search_folds=5,
                                  grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score > 0.95


def check_specified_cv_folds_sampler(numeric_ids):
    make_cv_folds_data(numeric_ids)

    # test_cv_folds1.cfg has prespecified folds and should have ~50% accuracy
    # test_cv_folds2.cfg doesn't have prespecified folds and >95% accuracy
    for experiment_name, test_func, grid_size in [('test_cv_folds1_sampler',
                                                   lambda x: x < 0.8,
                                                   3),
                                                  ('test_cv_folds2_sampler',
                                                   lambda x: x > 0.95,
                                                   10)]:
        config_template_file = '{}.template.cfg'.format(experiment_name)
        config_template_path = join(_my_dir, 'configs', config_template_file)
        config_path = join(_my_dir, fill_in_config_paths(config_template_path))

        # Modify config file to change ids_to_floats depending on numeric_ids
        # setting
        with open(config_path, 'r+') as config_template_file:
            lines = config_template_file.readlines()
            config_template_file.seek(0)
            config_template_file.truncate()
            for line in lines:
                if line.startswith('ids_to_floats='):
                    if numeric_ids:
                        line = 'ids_to_floats=true\n'
                    else:
                        line = 'ids_to_floats=false\n'
                config_template_file.write(line)

        run_configuration(config_path, quiet=True)
        result_filename = ('{}_test_cv_folds_LogisticRegression.'
                           'results').format(experiment_name)
        with open(join(_my_dir, 'output', result_filename)) as f:
            # check held out scores
            outstr = f.read()
            score = float(SCORE_OUTPUT_RE.search(outstr).groups()[-1])
            assert test_func(score)

            grid_score_matches = GRID_RE.findall(outstr)
            assert len(grid_score_matches) == grid_size
            for match_str in grid_score_matches:
                assert test_func(float(match_str))

    # try the same tests for just training (and specifying the folds for the
    # grid search)
    dirpath = join(_my_dir, 'train')
    suffix = '.jsonlines'
    featureset = ['test_cv_folds']
    examples = _load_featureset(dirpath, featureset, suffix, quiet=True)
    clf = Learner('LogisticRegression', probability=True)
    cv_folds = _load_cv_folds(join(_my_dir, 'train', 'test_cv_folds.csv'))
    grid_search_score = clf.train(examples, grid_search_folds=cv_folds,
                                  grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score < 0.6
    grid_search_score = clf.train(examples, grid_search_folds=5,
                                  grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score > 0.95


def test_specified_cv_folds_feature_hasher_sampler():
    yield check_specified_cv_folds_feature_hasher_sampler, False
    yield check_specified_cv_folds_feature_hasher_sampler, True


def test_specified_cv_folds_sampler():
    yield check_specified_cv_folds_sampler, False
    yield check_specified_cv_folds_sampler, True
