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
import json
import os
import sys
from glob import glob
from io import open
from os.path import abspath, dirname, exists, join

from numpy.testing import assert_almost_equal
from nose.tools import eq_
from skll.data import NDJWriter, Reader
from skll.experiments import run_configuration
from skll.learner import Learner
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import fill_in_config_paths, make_classification_data


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
_my_dir = abspath(dirname(__file__))


def setup():
    """
    Create necessary directories for testing.
    """
    for dir_name in ('train', 'test', 'output', 'evaluate'):
        new_dir = join(_my_dir, dir_name)
        if not exists(new_dir):
            os.makedirs(new_dir)


def tearDown():
    """
    Clean up after tests.
    """
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

    for output_file in (glob(join(output_dir, 'test_summary_*')) +
                        glob(join(output_dir,
                                  'test_majority_class_custom_learner_*'))):
        os.unlink(output_file)


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

    cfgfile = ('test_summary_feature_hasher.template.cfg' if
               use_feature_hashing else 'test_summary.template.cfg')
    config_template_path = join(_my_dir, 'configs', cfgfile)
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    outprefix = ('test_summary_feature_hasher_test_summary' if
                 use_feature_hashing else 'test_summary_test_summary')
    summprefix = ('test_summary_feature_hasher' if use_feature_hashing else
                  'test_summary')

    with open(join(_my_dir, 'output', ('{}_LogisticRegression.results.'
                                       'json'.format(outprefix)))) as f:
        outd = json.loads(f.read())
        logistic_result_score = outd[0]['score']

    with open(join(_my_dir, 'output',
                   '{}_SVC.results.json'.format(outprefix))) as f:
        outd = json.loads(f.read())
        svm_result_score = outd[0]['score']

    # note that Naive Bayes doesn't work with feature hashing
    if not use_feature_hashing:
        with open(join(_my_dir, 'output', ('{}_MultinomialNB.results.'
                                           'json'.format(outprefix)))) as f:
            outd = json.loads(f.read())
            naivebayes_result_score = outd[0]['score']

    with open(join(_my_dir, 'output', '{}_summary.tsv'.format(summprefix)),
              'r') as f:
        reader = csv.DictReader(f, dialect='excel-tab')

        for row in reader:
            # the learner results dictionaries should have 29 rows,
            # and all of these except results_table
            # should be printed (though some columns will be blank).
            eq_(len(row), 29)
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
                            err_msg=('mismatched scores for {} '
                                     '(result:{}, summary:'
                                     '{})').format(learner_name, result_score,
                                                   summary_score))

    # We itereate over each model with an expected
    # accuracy score. T est proves that the report
    # written out at least as a correct format for
    # this line. See _print_fancy_output
    for report_name, val in (("LogisticRegression", .5),
                             ("MultinomialNB", .5),
                             ("SVC", .7)):
        filename = "test_summary_test_summary_{}.results".format(report_name)
        results_path = join(_my_dir, 'output', filename)
        with open(results_path) as results_file:
            report = results_file.read()
            expected_string = "Accuracy = {:.1f}".format(val)
            eq_(expected_string in report,  # approximate
                True,
                msg="{} is not in {}".format(expected_string,
                                             report))


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
        old_predictions = [float(line.strip()) for
                           line in predict_file]
    assert_almost_equal(new_predictions, old_predictions)
