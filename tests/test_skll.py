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
'''


import csv
import json
import os
import re

import numpy as np
import scipy.sparse as sp
from nose.tools import *

from skll.experiments import (_load_featureset, run_configuration,
                              _load_cv_folds, _parse_config_file)
from skll.learner import Learner, SelectByMinCount
from skll.metrics import kappa


SCORE_OUTPUT_RE = re.compile((r'Average:.+Objective function score = ' +
                              r'([\-\d\.]+)'), re.DOTALL)
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
    dirpath = os.path.join(_my_dir)
    suffix = '.jsonlines'
    featureset = ['test_input_2examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix)


@raises(ValueError)
def test_input_checking2():
    dirpath = os.path.join(_my_dir)
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_1']
    _load_featureset(dirpath, featureset, suffix)


def test_input_checking3():
    dirpath = os.path.join(_my_dir)
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_2']
    examples_tuple = _load_featureset(dirpath, featureset, suffix)
    assert examples_tuple.features.shape[0] == 3


def make_cv_folds_data():
    num_examples_per_fold = 100
    num_folds = 3

    with open(os.path.join(_my_dir, 'test_cv_folds1.jsonlines'), 'w') as json_out, open(os.path.join(_my_dir, 'test_cv_folds1.csv'), 'w') as csv_out:
        csv_out.write('id,fold\n')
        for k in range(num_folds):
            for i in range(num_examples_per_fold):
                y = "dog" if i % 2 == 0 else "cat"
                ex_id = "{}{}".format(y, num_examples_per_fold * k + i)
                x = {"f1": 1.0, "f2": -1.0, "f3": 1.0, "is_{}{}".format(y, k): 1.0}
                json_out.write(json.dumps({"y": y, "id": ex_id, "x": x}) + '\n')
                csv_out.write('{},{}\n'.format(ex_id, k))


def fill_in_config_paths(config_template_path):
    with open(config_template_path, 'r') as config_template:
        config = _parse_config_file(config_template)

    config.set("Input", "train_location", _my_dir)
    for d in ['results', 'log', 'models', 'vocabs', 'predictions']:
        config.set("Output", d, _my_dir)

    cv_folds_location = config.get("Input", "cv_folds_location")
    if cv_folds_location:
        config.set("Input", "cv_folds_location", os.path.join(_my_dir, cv_folds_location))

    new_config_path = '{}.cfg'.format(re.search(r'^(.*)\.template\.cfg', config_template_path).groups()[0])

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def test_specified_cv_folds():
    make_cv_folds_data()

    # test_cv_folds1.cfg is with prespecified folds and should have about 50% performance
    # test_cv_folds2.cfg is without prespecified folds and should have very high performance
    for config_template_file, test_func, grid_size in [('test_cv_folds1.template.cfg', lambda x: x < 0.6, 3), ('test_cv_folds2.template.cfg', lambda x: x > 0.95, 10)]:
        config_path = fill_in_config_paths(os.path.join(_my_dir, config_template_file))

        with open(os.path.join(_my_dir, config_path)) as config:
            run_configuration(config, local=True)
        with open(os.path.join(_my_dir, 'tests_cv_test_cv_folds1_LogisticRegression_unscaled_tuned_accuracy_cross-validate.results')) as f:
            # check held out scores
            outstr = f.read()
            score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])
            assert test_func(score)

            grid_score_matches = GRID_RE.findall(outstr)
            assert len(grid_score_matches) == grid_size
            for match_str in grid_score_matches:
                assert test_func(float(match_str))

    # try the same tests for just training (and specifying the folds for the grid search)
    dirpath = os.path.join(_my_dir)
    suffix = '.jsonlines'
    featureset = ['test_cv_folds1']
    examples = _load_featureset(dirpath, featureset, suffix)
    clf = Learner(probability=True)
    cv_folds = _load_cv_folds(os.path.join(_my_dir, 'test_cv_folds1.csv'))
    grid_search_score = clf.train(examples, grid_search_folds=cv_folds, grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score < 0.6
    grid_search_score = clf.train(examples, grid_search_folds=5, grid_objective='accuracy', grid_jobs=1)
    assert grid_search_score > 0.95


def make_regression_data():
    num_examples = 1000

    np.random.seed(1234567890)
    f1 = np.random.rand(num_examples)
    f2 = np.random.rand(num_examples)
    f3 = np.random.rand(num_examples)
    err = np.random.randn(num_examples) / 2.0
    y = 1.0 * f1 + 1.0 * f2 - 2.0 * f3 + err

    with open(os.path.join(_my_dir, 'test_regression1.jsonlines'), 'w') as f:
        for i in range(num_examples):
            ex_id = "EXAMPLE{}".format(i)
            x = {"f1": f1[i], "f2": f2[i], "f3": f3[i]}
            f.write(json.dumps({"y": y[i], "id": ex_id, "x": x}) + '\n')

    return x, y


def test_regression1():
    '''
    This is a bit of a contrived test, but it should fail
    if anything drastic happens to the regression code.
    '''

    _, y = make_regression_data()

    config_template_path = os.path.join(_my_dir, "test_regression1.template.cfg")
    config_path = fill_in_config_paths(config_template_path)

    config_template_path = "test_regression1.cfg"
    test_func = lambda x: x > 0.7

    with open(os.path.join(_my_dir, config_path)) as cfg:
        run_configuration(cfg, local=True)
    with open(os.path.join(_my_dir, 'tests_cv_test_regression1_RescaledRidge_unscaled_tuned_pearson_cross-validate.results')) as f:
        # check held out scores
        outstr = f.read()
        score = float(SCORE_OUTPUT_RE.search(outstr).groups()[0])
        assert test_func(score)

    with open(os.path.join(_my_dir, 'tests_cv_test_regression1_RescaledRidge_unscaled_tuned_pearson_cross-validate.predictions'), 'rb') as f:
        reader = csv.reader(f, dialect='excel-tab')
        reader.next()
        pred = [float(row[1]) for row in reader]

        assert np.min(pred) >= np.min(y)
        assert np.max(pred) <= np.max(y)

        assert abs(np.mean(pred) - np.mean(y)) < 0.1
        assert abs(np.std(pred) - np.std(y)) < 0.1


# Test our kappa implementation based on Ben Hamner's unit tests.
def test_quadratic_weighted_kappa():
    k = kappa([1, 2, 3], [1, 2, 3], weights='quadratic')
    assert_almost_equal(k, 1.0)

    k = kappa([1, 2, 1], [1, 2, 2], weights='quadratic')
    assert_almost_equal(k, 0.4)

    k = kappa([1, 2, 3, 1, 2, 2, 3], [1, 2, 3, 1, 2, 3, 2], weights='quadratic')
    assert_almost_equal(k, 0.75)


def test_linear_weighted_kappa():
    k = kappa([1, 2, 3], [1, 2, 3], weights='linear')
    assert_almost_equal(k, 1.0)

    k = kappa([1, 2, 1], [1, 2, 2], weights='linear')
    assert_almost_equal(k, 0.4)

    k = kappa([1, 2, 3, 1, 2, 2, 3], [1, 2, 3, 1, 2, 3, 2], weights='linear')
    assert_almost_equal(k, 0.65)


def test_unweighted_kappa():
    k = kappa([1, 2, 3], [1, 2, 3])
    assert_almost_equal(k, 1.0)

    k = kappa([1, 2, 1], [1, 2, 2])
    assert_almost_equal(k, 0.4)

    k = kappa([1, 2, 3, 1, 2, 2, 3], [1, 2, 3, 1, 2, 3, 2])
    assert_almost_equal(k, 0.5625)
