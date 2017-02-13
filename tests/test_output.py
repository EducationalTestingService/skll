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

import numpy as np
from numpy.testing import assert_almost_equal
from nose.plugins.attrib import attr
from nose.tools import eq_, ok_

from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.naive_bayes import MultinomialNB

from skll.data import FeatureSet, NDJWriter, Reader
from skll.experiments import _HAVE_PANDAS, _HAVE_SEABORN, run_configuration, _compute_ylimits_for_featureset
from skll.learner import Learner, _DEFAULT_PARAM_GRIDS

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

    for suffix in ['learning_curve', 'summary']:
        if exists(join(train_dir, 'test_{}.jsonlines'.format(suffix))):
            os.unlink(join(train_dir, 'test_{}.jsonlines'.format(suffix)))

        if exists(join(test_dir, 'test_{}.jsonlines'.format(suffix))):
            os.unlink(join(test_dir, 'test_{}.jsonlines'.format(suffix)))

        config_files = ['test_{}.cfg'.format(suffix),
                        'test_{}_feature_hasher.cfg'.format(suffix)]
        for cf in config_files:
            if exists(join(config_dir, cf)):
                os.unlink(join(config_dir, cf))

        for output_file in (glob(join(output_dir, 'test_{}_*'.format(suffix))) +
                            glob(join(output_dir, 'test_majority_class_custom_learner_*'))):
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


# Generate and write out data for the test that checks learning curve outputs
def make_learning_curve_data():

    # Load in the digits data set
    digits = load_digits()
    X, y = digits.data, digits.target

    # create featureset with all features
    feature_names = ['f{:02}'.format(n) for n in range(X.shape[1])]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))
    fs1 = FeatureSet('train1', features=features, labels=y, ids=list(range(X.shape[0])))

    # Write this feature set to file
    train_path = join(_my_dir, 'train', 'test_learning_curve1.jsonlines')
    writer = NDJWriter(train_path, fs1)
    writer.write()

    # create featureset with all except the last feature
    feature_names = ['f{:02}'.format(n) for n in range(X.shape[1])]
    features = []
    for row in X:
        features.append(dict(zip(feature_names[:-1], row)))
    fs2 = FeatureSet('train2', features=features, labels=y, ids=list(range(X.shape[0])))

    # Write this feature set to file
    train_path = join(_my_dir, 'train', 'test_learning_curve2.jsonlines')
    writer = NDJWriter(train_path, fs2)
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

    # We iterate over each model with an expected
    # accuracy score. Test proves that the report
    # written out at least has a correct format for
    # this line. See _print_fancy_output
    for report_name, val in (("LogisticRegression", .5),
                             ("MultinomialNB", .5),
                             ("SVC", .6333)):
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


def test_learning_curve_implementation():
    """
    Test to ensure that the learning curve results match scikit-learn
    """

    # This test is different from the other tests which just use regression data.
    # The reason is that we want this test to fail in case our implementation
    # diverges from the scikit-learn implementation. This test essentially
    # serves as a regression test as well.

    # Load in the digits data set
    digits = load_digits()
    X, y = digits.data, digits.target

    # get the learning curve results from scikit-learn for this data
    cv_folds = 10
    random_state = 123456789
    cv = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=random_state)
    estimator = MultinomialNB()
    train_sizes=np.linspace(.1, 1.0, 5)
    train_sizes1, train_scores1, test_scores1 = learning_curve(estimator, X, y, cv=cv,  train_sizes=train_sizes, scoring='accuracy')

    # get the features from this data into a FeatureSet instance we can use
    # with the SKLL API
    feature_names = ['f{:02}'.format(n) for n in range(X.shape[1])]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))
    fs = FeatureSet('train', features=features, labels=y, ids=list(range(X.shape[0])))

    # we don't want to filter out any features since scikit-learn
    # does not do that either
    l = Learner('MultinomialNB', min_feature_count=0)
    train_scores2, test_scores2, train_sizes2 = l.learning_curve(fs,
                                                                 cv_folds=cv_folds,
                                                                 train_sizes=train_sizes,
                                                                 objective='accuracy')

    assert np.all(train_sizes1 == train_sizes2)
    assert np.allclose(train_scores1, train_scores2)
    assert np.allclose(test_scores1, test_scores2)


def test_learning_curve_output():
    """
    Test that the TSV output of a learning curve experiment is as expected
    """

    # Test to validate learning curve output
    make_learning_curve_data()

    config_template_path = join(_my_dir, 'configs', 'test_learning_curve.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    # run the learning curve experiment
    run_configuration(config_path, quiet=True)
    outprefix = 'test_learning_curve'

    # make sure that the TSV file is created with the right columns
    output_tsv_path = join(_my_dir, 'output', '{}_summary.tsv'.format(outprefix))
    ok_(exists(output_tsv_path))
    with open(output_tsv_path, 'r') as tsvf:
        r = csv.reader(tsvf, dialect=csv.excel_tab)
        header = next(r)
        # make sure we have the expected number of columns
        eq_(len(header), 11)
        num_rows = len(list(r))
        # we should have 2 featuresets x 3 learners x 2 objectives x 5 (default)
        # training sizes = 60 rows
        eq_(num_rows, 60)

    # make sure that the two PNG files (one per featureset) are created
    # if the requirements are satisfied
    if _HAVE_PANDAS and _HAVE_SEABORN:
        for featureset_name in ["test_learning_curve1", "test_learning_curve2"]:
            ok_(exists(join(_my_dir,
                            'output',
                            '{}_{}.png'.format(outprefix, featureset_name))))


@attr('have_pandas_and_seaborn')
def test_learning_curve_plots():
    """
    Test that the plots of a learning curve experiment are generated as expected
    """

    # Test to validate learning curve output
    make_learning_curve_data()

    config_template_path = join(_my_dir, 'configs', 'test_learning_curve.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    # run the learning curve experiment
    run_configuration(config_path, quiet=True)
    outprefix = 'test_learning_curve'

    # make sure that the two PNG files (one per featureset) are created
    for featureset_name in ["test_learning_curve1", "test_learning_curve2"]:
        ok_(exists(join(_my_dir,
                        'output',
                        '{}_{}.png'.format(outprefix, featureset_name))))


@attr('have_pandas_and_seaborn')
def test_learning_curve_ylimits():
    """
    Test that the ylimits for learning curves are generated as expected.
    """

    if _HAVE_PANDAS:
        import pandas as pd

    # create a test data frame
    df_test = pd.DataFrame.from_dict({'test_score_std': {0: 0.16809136190418694, 1: 0.18556201422712379, 2: 0.15002727816517414, 3: 0.15301923832338646, 4: 0.15589815327205431, 5: 0.68205316443171948, 6: 0.77441075727706354, 7: 0.83838056331276678, 8: 0.84770116657005623, 9: 0.8708014559726478}, 'value': {0: 0.4092496971394447, 1: 0.2820507715115001, 2: 0.24533811547921261, 3: 0.21808651942296109, 4: 0.19767367891431534, 5: -2.3540980769230773, 6: -3.1312445327182394, 7: -3.2956790939674137, 8: -3.4843050005436713, 9: -3.6357879085645455}, 'train_score_std': {0: 0.15950199460682787, 1: 0.090992452273091703, 2: 0.068488654201949981, 3: 0.055223120652733763, 4: 0.03172452509259388, 5: 1.0561586240460523, 6: 0.53955995320300709, 7: 0.40477740983901211, 8: 0.34148185048394258, 9: 0.20791478156554272}, 'variable': {0: 'train_score_mean', 1: 'train_score_mean', 2: 'train_score_mean', 3: 'train_score_mean', 4: 'train_score_mean', 5: 'train_score_mean', 6: 'train_score_mean', 7: 'train_score_mean', 8: 'train_score_mean', 9: 'train_score_mean'}, 'objective': {0: 'r2', 1: 'r2', 2: 'r2', 3: 'r2', 4: 'r2', 5: 'neg_mean_squared_error', 6: 'neg_mean_squared_error', 7: 'neg_mean_squared_error', 8: 'neg_mean_squared_error', 9: 'neg_mean_squared_error'}})

    # compute the y-limits
    ylimits_dict = _compute_ylimits_for_featureset(df_test, ['r2', 'neg_mean_squared_error'])

    eq_(len(ylimits_dict), 2)
    eq_(ylimits_dict['neg_mean_squared_error'], (-4, 0))
    eq_(ylimits_dict['r2'], (0, 1.1))
