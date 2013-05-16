
import scipy.sparse as sp
import sys
import os
import numpy as np
import classifier
import json
import re

_my_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(_my_path)

from run_experiment import load_featureset, run_configuration, load_cv_folds
from classifier import Classifier, accuracy
from nose.tools import *


def test_SelectByMinCount():
    m2 = [[0.001,   0.0,    0.0,    0.0],
          [0.00001, -2.0,   0.0,    0.0],
          [0.001,   0.0,    0.0,    4.0], 
          [0.0101,  -200.0, 0.0,    0.0]]

    # default should keep all nonzero features (i.e., ones that appear 1+ times)
    feat_selector = classifier.SelectByMinCount()
    expected = np.array([[0.001, 0.0, 0.0], [0.00001, -2.0, 0.0], [0.001, 0.0, 4.0], [0.0101, -200.0, 0.0]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)

    # keep features that happen 2+ times
    feat_selector = classifier.SelectByMinCount(min_count=2)
    expected = np.array([[0.001, 0.0], [0.00001, -2.0], [0.001, 0.0], [0.0101, -200.0]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)

    # keep features that happen 3+ times
    feat_selector = classifier.SelectByMinCount(min_count=3)
    expected = np.array([[0.001], [0.00001], [0.001], [0.0101]])
    assert np.array_equal(feat_selector.fit_transform(np.array(m2)), expected)
    assert np.array_equal(feat_selector.fit_transform(sp.csr_matrix(m2)).todense(), expected)


@raises(ValueError)
def test_input_checking1():
    dirpath = os.path.join(_my_path, 'tests')
    suffix = '.jsonlines'
    featureset = ['test_input_2examples_1', 'test_input_3examples_1']
    load_featureset(dirpath, featureset, suffix)


@raises(ValueError)
def test_input_checking2():
    dirpath = os.path.join(_my_path, 'tests')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_1']
    load_featureset(dirpath, featureset, suffix)


def test_input_checking3():
    dirpath = os.path.join(_my_path, 'tests')
    suffix = '.jsonlines'
    featureset = ['test_input_3examples_1', 'test_input_3examples_2']
    feats = load_featureset(dirpath, featureset, suffix)
    assert len(feats) == 3


def make_cv_folds_data():
    num_examples_per_fold = 100
    num_folds = 3

    with open(os.path.join(_my_path, 'tests', 'test_cv_folds1.jsonlines'), 'w') as f, open(os.path.join(_my_path, 'tests', 'test_cv_folds1.csv'), 'w') as csv_out:
        csv_out.write('id,fold\n')
        for k in range(num_folds):
            for i in range(num_examples_per_fold):
                y = "dog" if i % 2 == 0 else "cat"
                ex_id = "{}{}".format(y, num_examples_per_fold * k + i)
                x = {"f1": 1.0, "f2": -1.0, "f3": 1.0, "is_{}{}".format(y, k): 1.0}
                f.write(json.dumps({"y": y, "id": ex_id, "x":x}) + '\n')
                csv_out.write('{},{}\n'.format(ex_id, k))


def test_specified_cv_folds():
    make_cv_folds_data()
    
    output_re = re.compile(r'Objective function score = ([\-\d\.]+)')
    grid_re = re.compile(r'Grid search score = ([\-\d\.]+)', re.M)

    # test_cv_folds1.cfg is with prespecified folds and should have about 50% performance
    # test_cv_folds2.cfg is without prespecified folds and should have very high performance
    for cfg_filename, test_func, grid_size in [('test_cv_folds1.cfg', lambda x: x < 0.6, 3), ('test_cv_folds2.cfg', lambda x: x > 0.95, 10)]:
        with open(os.path.join(_my_path, 'tests', cfg_filename)) as cfg:
            run_configuration(cfg, local=True)
        with open(os.path.join(_my_path, 'tests', 'tests_cv_test_cv_folds1_logistic_scaled_tuned_accuracy_cross-validate.results')) as f:
            # check held out scores
            outstr = f.read()
            score = float(output_re.search(outstr).groups()[0])
            assert test_func(score)

            grid_score_matches = grid_re.findall(outstr)
            assert len(grid_score_matches) == grid_size
            for match_str in grid_score_matches:
                assert test_func(float(match_str))


    # try the same tests for just training (and specifying the folds for the grid search)
    dirpath = os.path.join(_my_path, 'tests')
    suffix = '.jsonlines'
    featureset = ['test_cv_folds1']
    examples = load_featureset(dirpath, featureset, suffix)
    clf = Classifier(probability=True)
    cv_folds = load_cv_folds(os.path.join(_my_path, 'tests', 'test_cv_folds1.csv'))
    grid_search_score = clf.train(examples, grid_search_folds=cv_folds, grid_objective=accuracy, grid_jobs=1)
    assert grid_search_score < 0.6
    grid_search_score = clf.train(examples, grid_search_folds=5, grid_objective=accuracy, grid_jobs=1)
    assert grid_search_score > 0.95
