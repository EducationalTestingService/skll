# License: BSD 3 clause
"""
Tests related to output from run_experiment

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import glob
import os
import re
import sys
from io import open
from os.path import abspath, dirname, exists, join

from nose.tools import eq_, assert_almost_equal
from skll.data import FeatureSet, NDJWriter, Reader
from skll.experiments import run_configuration, _setup_config_parser
from skll.learner import Learner
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import make_classification_data


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


def tearDown():
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')

    if exists(join(train_dir, 'test_summary.jsonlines')):
        os.unlink(join(train_dir, 'test_summary.jsonlines'))

    if exists(join(test_dir, 'test_summary.jsonlines')):
        os.unlink(join(test_dir, 'test_summary.jsonlines'))

    config_files = ['test_summary.cfg',
                    'test_summary_feature_hasher.cfg']
    for cf in config_files:
        if exists(join(config_dir, cf)):
            os.unlink(join(config_dir, cf))

    for output_file in glob.glob(join(output_dir, 'test_summary_*')) \
                       + glob.glob(join(output_dir, 'test_majority_class_custom_learner_*')):
        os.unlink(output_file)


def fill_in_config_paths(config_template_path):
    """
    Add paths to train, test, and output directories to a given config template
    file.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path)

    task = config.get("General", "task")
    # experiment_name = config.get("General", "experiment_name")

    config.set("Input", "train_location", train_dir)

    to_fill_in = ['log', 'predictions']

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


# Generate and write out data for the test that checks summary scores
def make_summary_data():
    train_fs, test_fs = make_classification_data(num_examples=600,
                                                 train_test_ratio=0.8,
                                                 num_labels=2,
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
        logistic_result_score = float(SCORE_OUTPUT_RE.search(outstr)
                                      .groups()[0])

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
            # the learner results dictionaries should have 27 rows,
            # and all of these except results_table
            # should be printed (though some columns will be blank).
            eq_(len(row), 27)
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
    """
    Test to validate backward compatibility
    """
    predict_path = join(_my_dir, 'backward_compatibility',
                        ('v0.9.17_test_summary_test_summary_'
                         'LogisticRegression.predictions'))
    model_path = join(_my_dir, 'backward_compatibility',
                      ('v0.9.17_test_summary_test_summary_LogisticRegression.'
                       '{}.model').format(sys.version_info[0]))
    test_path = join(_my_dir, 'backward_compatibility',
                     'v0.9.17_test_summary.jsonlines')

    learner = Learner.from_file(model_path)
    examples = Reader.for_path(test_path, quiet=True).read()
    new_predictions = learner.predict(examples)[:, 1]

    with open(predict_path) as predict_file:
        for line, new_val in zip(predict_file, new_predictions):
            assert_almost_equal(float(line.strip()), new_val)
