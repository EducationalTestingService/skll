# License: BSD 3 clause
"""
Tests related to output from run_experiment

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
"""

import csv
import json
import os
import re
import warnings

from ast import literal_eval
from collections import defaultdict
from glob import glob
from itertools import product
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal)

from nose.tools import eq_, ok_, assert_raises, raises

from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.naive_bayes import MultinomialNB

from skll.data import FeatureSet, NDJReader, NDJWriter, Reader
from skll.experiments import run_configuration
from skll.experiments.output import _compute_ylimits_for_featureset
from skll.learner import Learner
from skll.utils.constants import KNOWN_DEFAULT_PARAM_GRIDS, VALID_TASKS
from skll.utils.logging import get_skll_logger, close_and_remove_logger_handlers
from tests.utils import (create_jsonlines_feature_files,
                         fill_in_config_options,
                         fill_in_config_paths,
                         fill_in_config_paths_for_single_file,
                         make_classification_data,
                         make_regression_data)

_ALL_MODELS = list(KNOWN_DEFAULT_PARAM_GRIDS.keys())
_my_dir = abspath(dirname(__file__))


def setup():
    """
    Create necessary directories for testing.
    """
    for dir_name in ('train', 'test', 'output', 'evaluate'):
        new_dir = join(_my_dir, dir_name)
        if not exists(new_dir):
            os.makedirs(new_dir)

    # create jsonlines feature files
    train_dir = join(_my_dir, 'train')
    create_jsonlines_feature_files(train_dir)


def tearDown():
    """
    Clean up after tests.
    """
    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')
    for suffix in ['learning_curve', 'summary', 'fancy_xval',
                   'warning_multiple_featuresets']:
        if exists(join(train_dir, f'test_{suffix}.jsonlines')):
            os.unlink(join(train_dir, f'test_{suffix}.jsonlines'))

        if exists(join(test_dir, f'test_{suffix}.jsonlines')):
            os.unlink(join(test_dir, f'test_{suffix}.jsonlines'))

        config_files = [f'test_{suffix}.cfg',
                        f'test_{suffix}_with_metrics.cfg',
                        f'test_{suffix}_with_objectives.cfg',
                        f'test_{suffix}_feature_hasher.cfg',
                        f'test_{suffix}_feature_hasher_with_metrics.cfg']
        for cf in config_files:
            if exists(join(config_dir, cf)):
                os.unlink(join(config_dir, cf))

        for output_file in glob(join(output_dir, f'test_{suffix}_*')) \
                           + glob(join(output_dir, f'test_{suffix}.log')):
            os.unlink(output_file)

    for suffix in VALID_TASKS:
        config_files = [f'test_cv_results_{suffix}.cfg']
        for cf in config_files:
            if exists(join(config_dir, cf)):
                os.unlink(join(config_dir, cf))

    if exists(join(config_dir, 'test_send_warnings_to_log.cfg')):
        os.unlink(join(config_dir, 'test_send_warnings_to_log.cfg'))

    # adding all the suffix independent output patterns here that are not f'test_{SUFFIX}_*'
    clean_up_output_file_name_patterns = ['test_majority_class_custom_learner_*',
                                          'test_send_warnings_to_log*',
                                          'test_grid_search_cv_results_*.*',
                                          'test_check_override_learning_curve_min_examples*']
    for file_name_pattern in clean_up_output_file_name_patterns:
        for output_file in glob(join(output_dir, file_name_pattern)):
            os.unlink(output_file)

    if exists("test_current_directory.model"):
        os.unlink("test_current_directory.model")


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
    feature_names = [f'f{n:02}' for n in range(X.shape[1])]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))
    fs1 = FeatureSet('train1', features=features, labels=y, ids=list(range(X.shape[0])))

    # Write this feature set to file
    train_path = join(_my_dir, 'train', 'test_learning_curve1.jsonlines')
    writer = NDJWriter(train_path, fs1)
    writer.write()

    # create featureset with all except the last feature
    feature_names = [f'f{n:02}' for n in range(X.shape[1])]
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
def check_summary_score(use_feature_hashing,
                        use_additional_metrics):

    # Test to validate summary file scores
    make_summary_data()

    if use_feature_hashing:
        cfgfile = ('test_summary_feature_hasher_with_metrics.template.cfg' if
                   use_additional_metrics else 'test_summary_feature_hasher.template.cfg')
        outprefix = ('test_summary_feature_hasher_with_metrics_test_summary' if
                     use_additional_metrics else 'test_summary_feature_hasher_test_summary')
        summprefix = ('test_summary_feature_hasher_with_metrics' if
                      use_additional_metrics else 'test_summary_feature_hasher')
    else:
        cfgfile = ('test_summary_with_metrics.template.cfg' if
                   use_additional_metrics else 'test_summary.template.cfg')
        outprefix = ('test_summary_with_metrics_test_summary' if
                     use_additional_metrics else 'test_summary_test_summary')
        summprefix = ('test_summary_with_metrics' if
                      use_additional_metrics else 'test_summary')

    config_template_path = join(_my_dir, 'configs', cfgfile)
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True)

    with open(join(_my_dir, 'output',
                   f'{outprefix}_LogisticRegression.results.json')) as f:
        outd = json.loads(f.read())
        logistic_result_score = outd[0]['score']
        if use_additional_metrics:
            results_metrics_dict = outd[0]['additional_scores']
            logistic_result_additional_metric1 = results_metrics_dict['unweighted_kappa']
            logistic_result_additional_metric2 = results_metrics_dict['f1_score_micro']

    with open(join(_my_dir, 'output',
                   f'{outprefix}_SVC.results.json')) as f:
        outd = json.loads(f.read())
        svm_result_score = outd[0]['score']
        if use_additional_metrics:
            results_metrics_dict = outd[0]['additional_scores']
            svm_result_additional_metric1 = results_metrics_dict['unweighted_kappa']
            svm_result_additional_metric2 = results_metrics_dict['f1_score_micro']

    # note that Naive Bayes doesn't work with feature hashing
    if not use_feature_hashing:
        with open(join(_my_dir, 'output',
                       f'{outprefix}_MultinomialNB.results.json')) as f:
            outd = json.loads(f.read())
            naivebayes_result_score = outd[0]['score']
            if use_additional_metrics:
                results_metrics_dict = outd[0]['additional_scores']
                nb_result_additional_metric1 = results_metrics_dict['unweighted_kappa']
                nb_result_additional_metric2 = results_metrics_dict['f1_score_micro']

    with open(join(_my_dir, 'output', f'{summprefix}_summary.tsv'),
              'r') as f:
        reader = csv.DictReader(f, dialect='excel-tab')

        for row in reader:
            # the learner results dictionaries should have 34 rows,
            # and all of these except results_table
            # should be printed (though some columns will be blank).
            eq_(len(row), 35)
            assert row['model_params']
            assert row['grid_score']
            assert row['score']
            if use_additional_metrics:
                assert row['additional_scores']

            if row['learner_name'] == 'LogisticRegression':
                logistic_summary_score = float(row['score'])
                if use_additional_metrics:
                    summary_metrics_dict = literal_eval(row['additional_scores'])
                    logistic_summary_additional_score1 = float(summary_metrics_dict['unweighted_kappa'])
                    logistic_summary_additional_score2 = float(summary_metrics_dict['f1_score_micro'])
            elif row['learner_name'] == 'MultinomialNB':
                naivebayes_summary_score = float(row['score'])
                if use_additional_metrics:
                    summary_metrics_dict = literal_eval(row['additional_scores'])
                    nb_summary_additional_score1 = float(summary_metrics_dict['unweighted_kappa'])
                    nb_summary_additional_score2 = float(summary_metrics_dict['f1_score_micro'])
            elif row['learner_name'] == 'SVC':
                svm_summary_score = float(row['score'])
                if use_additional_metrics:
                    summary_metrics_dict = literal_eval(row['additional_scores'])
                    svm_summary_additional_score1 = float(summary_metrics_dict['unweighted_kappa'])
                    svm_summary_additional_score2 = float(summary_metrics_dict['f1_score_micro'])

    test_tuples = [(logistic_result_score,
                    logistic_summary_score,
                    'LogisticRegression'),
                   (svm_result_score,
                    svm_summary_score,
                    'SVC')]

    if use_additional_metrics:
        test_tuples.extend([(logistic_result_additional_metric1,
                             logistic_summary_additional_score1,
                             'LogisticRegression'),
                            (logistic_result_additional_metric2,
                             logistic_summary_additional_score2,
                             'LogisticRegression'),
                            (svm_result_additional_metric1,
                             svm_summary_additional_score1,
                             'SVC'),
                            (svm_result_additional_metric2,
                             svm_summary_additional_score2,
                             'SVC')])

    if not use_feature_hashing:
        test_tuples.append((naivebayes_result_score,
                            naivebayes_summary_score,
                            'MultinomialNB'))
        if use_additional_metrics:
            test_tuples.extend([(nb_result_additional_metric1,
                                 nb_summary_additional_score1,
                                 'MultinomialNB'),
                                (nb_result_additional_metric2,
                                 nb_summary_additional_score2,
                                 'MultinomialNB')])

    for result_score, summary_score, learner_name in test_tuples:
        assert_almost_equal(result_score, summary_score,
                            err_msg=f'mismatched scores for {learner_name} '
                                    f'(result:{result_score}, summary:'
                                    f'{summary_score})')

    # We iterate over each model with an expected
    # accuracy score. Test proves that the report
    # written out at least has a correct format for
    # this line. See _print_fancy_output
    if not use_feature_hashing and not use_additional_metrics:
        for report_name, val in (("LogisticRegression", .5),
                                 ("MultinomialNB", .5),
                                 ("SVC", .6333)):
            filename = f"test_summary_test_summary_{report_name}.results"
            results_path = join(_my_dir, 'output', filename)
            with open(results_path) as results_file:
                report = results_file.read()
                expected_string = f"Accuracy = {val:.1f}"
                eq_(expected_string in report,  # approximate
                    True,
                    msg=f"{expected_string} is not in {report}")


def test_summary():

    for (use_feature_hashing,
         use_additional_metrics) in product([True, False], [True, False]):
        yield check_summary_score, use_feature_hashing, use_additional_metrics


def check_xval_fancy_results_file(do_grid_search,
                                  use_folds_file,
                                  use_folds_file_for_grid_search,
                                  use_additional_metrics):

    train_path = join(_my_dir, 'train', 'f0.jsonlines')
    output_dir = join(_my_dir, 'output')

    # make a simple config file for cross-validation
    values_to_fill_dict = {'experiment_name': 'test_fancy_xval',
                           'train_file': train_path,
                           'task': 'cross_validate',
                           'grid_search': 'true',
                           'objectives': "['f1_score_micro']",
                           'featureset_names': '["f0"]',
                           'num_cv_folds': '6',
                           'grid_search_folds': '4',
                           'learners': "['LogisticRegression']",
                           'log': output_dir,
                           'predictions': output_dir,
                           'results': output_dir}

    folds_file_path = join(_my_dir, 'train', 'folds_file_test.csv')
    if use_folds_file:
        values_to_fill_dict['folds_file'] = folds_file_path
    values_to_fill_dict['grid_search'] = str(do_grid_search)
    values_to_fill_dict['use_folds_file_for_grid_search'] = str(use_folds_file_for_grid_search)

    if use_additional_metrics:
        values_to_fill_dict['metrics'] = str(["accuracy", "unweighted_kappa"])

    config_template_path = join(_my_dir,
                                'configs',
                                'test_fancy.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'xval')

    # run the experiment
    run_configuration(config_path, quiet=True)

    # now make sure that the results file was produced
    results_file_path = join(_my_dir, 'output', 'test_fancy_xval_f0_LogisticRegression.results')
    ok_(exists(results_file_path))

    # read in all the lines and look at the lines up to where we print the "Total Time"
    with open(results_file_path, 'r') as resultsf:
        results_lines = resultsf.readlines()
        end_idx = [results_lines.index(l) for l in results_lines if l.startswith('Total Time:')][0]
        results_lines = results_lines[:end_idx + 1]

        # read in the "keys" and "values" separated by colons into a dictionary
        results_dict = dict([rl.strip().split(': ') for rl in results_lines])

    # check that the fields we expect in the results file are there
    # and the ones that we do not expect aren't
    if do_grid_search:
        eq_(results_dict['Grid Search'], 'True')
        eq_(results_dict['Grid Objective Function'], 'f1_score_micro')
    else:
        eq_(results_dict['Grid Search'], 'False')
        ok_('Grid Search Folds' not in results_dict)
        ok_('Grid Objective Function' not in results_dict)

    if use_folds_file:
        eq_(results_dict['Number of Folds'], '5 via folds file')
        ok_('Stratified Folds' not in results_dict)
        eq_(results_dict['Specified Folds File'], folds_file_path)
        if do_grid_search:
            if use_folds_file_for_grid_search:
                eq_(results_dict['Grid Search Folds'], '5 via folds file')
                eq_(results_dict['Using Folds File for Grid Search'], 'True')
            else:
                eq_(results_dict['Grid Search Folds'], '4')
                eq_(results_dict['Using Folds File for Grid Search'], 'False')
    else:
        eq_(results_dict['Number of Folds'], '6')
        eq_(results_dict['Stratified Folds'], 'True')
        ok_('Using Folds File for Grid Search' not in results_dict)
        ok_('Specified Folds File' not in results_dict)
        if do_grid_search:
            eq_(results_dict['Grid Search Folds'], '4')

    if use_additional_metrics:
        expected_metrics = ["accuracy", "unweighted_kappa"]

        eq_(sorted(literal_eval(results_dict['Additional Evaluation Metrics'])),
            sorted(expected_metrics))


def test_xval_fancy_results_file():

    for (do_grid_search,
         use_folds_file,
         use_folds_file_for_grid_search,
         use_additional_metrics) in product([True, False],
                                            [True, False],
                                            [True, False],
                                            [True, False]):

        yield (check_xval_fancy_results_file, do_grid_search,
               use_folds_file, use_folds_file_for_grid_search,
               use_additional_metrics)


def check_grid_search_cv_results(task, do_grid_search):
    learners = ['LogisticRegression', 'SVC']
    expected_path = join(_my_dir, 'other', 'cv_results')

    def time_field(x):
        return x.endswith('_time')

    train_path = join(_my_dir, 'train', 'f0.jsonlines')
    output_dir = join(_my_dir, 'output')

    exp_name = (f'test_grid_search_cv_results_{task}_'
                f'{"gs" if do_grid_search else "nogs"}')

    # make a simple config file for cross-validation
    values_to_fill_dict = {'experiment_name': exp_name,
                           'train_file': train_path,
                           'task': task,
                           'grid_search': json.dumps(do_grid_search),
                           'objectives': "['f1_score_micro']",
                           'featureset_names': "['f0']",
                           'learners': json.dumps(learners),
                           'log': output_dir,
                           'results': output_dir}
    if task == 'train':
        values_to_fill_dict['models'] = output_dir
    elif task == 'cross_validate':
        values_to_fill_dict['predictions'] = output_dir
    elif task in ['evaluate', 'predict']:
        values_to_fill_dict['predictions'] = output_dir
        values_to_fill_dict['test_file'] = \
            values_to_fill_dict['train_file']

    # In the case where grid search is on and the task is
    # learning curve, grid search will automatically be turned
    # off, so simply turn it off here as well since it should
    # result in the same situation
    elif task == 'learning_curve':
        values_to_fill_dict['metrics'] = values_to_fill_dict.pop('objectives')
        if do_grid_search:
            do_grid_search = False

    config_template_path = join(_my_dir,
                                'configs',
                                'test_cv_results.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         task)

    # run the experiment
    if task in ['train', 'predict']:
        if do_grid_search:
            run_configuration(config_path, quiet=True)
        else:
            assert_raises(ValueError, run_configuration, config_path, quiet=True)
            # Short-circuit the test since a ValueError is
            # expected and is fatal
            return
    else:
        run_configuration(config_path, quiet=True)

    # now make sure that the results json file was produced
    for learner in learners:
        results_file_name = f'{exp_name}_f0_{learner}.results.json'
        actual_results_file_path = join(_my_dir, 'output',
                                        results_file_name)
        expected_results_file_path = join(expected_path,
                                          results_file_name)
        ok_(exists(actual_results_file_path))
        with open(expected_results_file_path) as expected, \
                open(actual_results_file_path) as actual:
            expected_lines = [json.loads(line) for line in expected][0]
            actual_lines = [json.loads(line) for line in actual][0]
            assert len(expected_lines) == len(actual_lines)
            if task == 'cross_validate':
                # All but the last line will have grid search-related
                # results
                for (expected_gs_cv_results,
                     actual_gs_cv_results) in zip(expected_lines[:-1],
                                                  actual_lines[:-1]):
                    assert len(expected_gs_cv_results) == len(actual_gs_cv_results)
                    for field in ['grid_score', 'grid_search_cv_results']:
                        if do_grid_search:
                            assert set(expected_gs_cv_results).intersection(actual_gs_cv_results) == \
                                set(expected_gs_cv_results)
                            if field == 'grid_score':
                                assert expected_gs_cv_results[field] == \
                                    actual_gs_cv_results[field]
                            else:
                                for subfield in expected_gs_cv_results[field]:
                                    if time_field(subfield):
                                        continue
                                    assert expected_gs_cv_results[field][subfield] == \
                                        actual_gs_cv_results[field][subfield]
                        else:
                            if field == 'grid_score':
                                assert actual_gs_cv_results[field] == 0.0
                            else:
                                assert actual_gs_cv_results[field] is None
                # The last line should be for the "average" and should
                # not contain any grid search results
                assert actual_lines[-1]['fold'] == 'average'
                for field in ['grid_score', 'grid_search_cv_results']:
                    assert field not in actual_lines[-1]
            elif task == 'evaluate':
                for (expected_gs_cv_results,
                     actual_gs_cv_results) in zip(expected_lines,
                                                  actual_lines):
                    assert len(expected_gs_cv_results) == len(actual_gs_cv_results)
                    for field in ['grid_score', 'grid_search_cv_results']:
                        if do_grid_search:
                            assert set(expected_gs_cv_results).intersection(actual_gs_cv_results) == \
                                set(expected_gs_cv_results)
                            if field == 'grid_score':
                                assert expected_gs_cv_results[field] == \
                                    actual_gs_cv_results[field]
                            else:
                                for subfield in expected_gs_cv_results[field]:
                                    if time_field(subfield):
                                        continue
                                    assert expected_gs_cv_results[field][subfield] == \
                                        actual_gs_cv_results[field][subfield]
                        else:
                            if field == 'grid_score':
                                assert actual_gs_cv_results[field] == 0.0
                            else:
                                assert actual_gs_cv_results[field] is None
            elif task in ['train', 'predict']:
                expected_gs_cv_results = expected_lines
                actual_gs_cv_results = actual_lines
                assert set(expected_gs_cv_results).intersection(actual_gs_cv_results) == \
                    set(expected_gs_cv_results)
                for field in ['grid_score', 'grid_search_cv_results']:
                    if field == 'grid_score':
                        assert expected_gs_cv_results[field] == \
                            actual_gs_cv_results[field]
                    else:
                        for subfield in expected_gs_cv_results[field]:
                            if time_field(subfield):
                                continue
                            assert expected_gs_cv_results[field][subfield] == \
                                actual_gs_cv_results[field][subfield]
            else:
                for expected_line, actual_line in zip(expected_lines,
                                                      actual_lines):
                    expected_fields = set(list(expected_line))
                    actual_fields = set(list(actual_line))
                    assert expected_fields.intersection(actual_fields) == \
                        expected_fields
                    assert all(field not in actual_fields
                               for field in ['grid_score',
                                             'grid_search_cv_results'])


def test_grid_search_cv_results():
    for task in VALID_TASKS:
        for do_grid_search in [True, False]:
            yield check_grid_search_cv_results, task, do_grid_search


def test_multiple_featuresets_and_featurehasher_throws_warning():
    '''
    test using multiple feature sets with feature hasher throws warning
    '''
    train_dir = join(_my_dir, 'train')
    output_dir = join(_my_dir, 'output')

    # make a simple config file for feature hasher warning test
    values_to_fill_dict = {'experiment_name': 'test_warning_multiple_featuresets',
                           'train_directory': train_dir,
                           'task': 'train',
                           'grid_search': 'false',
                           'objectives': "['f1_score_micro']",
                           'learners': "['LogisticRegression']",
                           'featuresets': ("[['test_input_3examples_1', "
                                           "'test_input_3examples_2']]"),
                           "featureset_names": "['feature_hasher']",
                           'suffix': '.jsonlines',
                           'log': output_dir,
                           'models': output_dir,
                           'feature_hasher': "true",
                           "hasher_features": "4"
                           }

    config_template_path = join(_my_dir,
                                'configs',
                                'test_warning_multiple_featuresets.template.cfg')

    config_path = fill_in_config_options(config_template_path,
                                         values_to_fill_dict,
                                         'feature_hasher')

    # run the experiment
    print(config_path)
    run_configuration(config_path, quiet=True)

    # test if it throws any warning
    logfile_path = join(_my_dir, "output",
                        "test_warning_multiple_featuresets_feature_hasher_LogisticRegression.log")
    with open(logfile_path) as f:
        warning_pattern = re.compile('Since there are multiple feature files, '
                                     'feature hashing applies to each '
                                     'specified feature file separately.')
        matches = re.findall(warning_pattern, f.read())
        eq_(len(matches), 1)


# Verify v0.23.1 model can still be loaded and generates the same predictions.
def test_backward_compatibility():
    """
    Test to validate backward compatibility
    """
    predict_path = join(_my_dir,
                        'backward_compatibility',
                        'v0.23.1_test_summary_test_summary_LogisticRegression_predictions.tsv')
    model_path = join(_my_dir,
                      'backward_compatibility',
                      'v0.23.1_test_summary_test_summary_LogisticRegression.model')
    test_path = join(_my_dir,
                     'backward_compatibility',
                     'v0.23.1_test_summary.jsonlines')

    learner = Learner.from_file(model_path)
    examples = Reader.for_path(test_path, quiet=True).read()
    new_predictions = learner.predict(examples, class_labels=False)[:, 1]

    with open(predict_path, 'r') as predict_file:
        reader = csv.DictReader(predict_file, dialect=csv.excel_tab)
        old_predictions = [float(row['1']) for row in reader]
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
    train_sizes = np.linspace(.1, 1.0, 5)
    train_sizes1, train_scores1, test_scores1 = learning_curve(estimator,
                                                               X,
                                                               y,
                                                               cv=cv,
                                                               train_sizes=train_sizes,
                                                               scoring='accuracy')

    # get the features from this data into a FeatureSet instance we can use
    # with the SKLL API
    feature_names = [f'f{n:02}' for n in range(X.shape[1])]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))
    fs = FeatureSet('train', features=features, labels=y, ids=list(range(X.shape[0])))

    # we don't want to filter out any features since scikit-learn
    # does not do that either
    learner = Learner('MultinomialNB', min_feature_count=0)
    (train_scores2,
     test_scores2,
     train_sizes2) = learner.learning_curve(fs,
                                            cv_folds=cv_folds,
                                            train_sizes=train_sizes,
                                            metric='accuracy')

    assert np.all(train_sizes1 == train_sizes2)
    assert np.allclose(train_scores1, train_scores2)
    assert np.allclose(test_scores1, test_scores2)


def test_learning_curve_output():
    """
    Test learning curve output for experiment with metrics option
    """

    # Test to validate learning curve output
    make_learning_curve_data()

    config_template_path = join(_my_dir, 'configs', 'test_learning_curve.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    # run the learning curve experiment
    run_configuration(config_path, quiet=True)
    outprefix = 'test_learning_curve'

    # make sure that the TSV file is created with the right columns
    output_tsv_path = join(_my_dir, 'output', f'{outprefix}_summary.tsv')
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
    for featureset_name in ["test_learning_curve1", "test_learning_curve2"]:
        ok_(exists(join(_my_dir,
                        'output',
                        f'{outprefix}_{featureset_name}.png')))


def test_learning_curve_output_with_objectives():
    """
    Test learning curve output for experiment with objectives option
    """

    # Test to validate learning curve output
    make_learning_curve_data()

    config_template_path = join(_my_dir,
                                'configs',
                                'test_learning_curve.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    # run the learning curve experiment
    run_configuration(config_path, quiet=True)
    outprefix = 'test_learning_curve'

    # make sure that the TSV file is created with the right columns
    output_tsv_path = join(_my_dir, 'output', f'{outprefix}_summary.tsv')
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
    for featureset_name in ["test_learning_curve1", "test_learning_curve2"]:
        ok_(exists(join(_my_dir,
                        'output',
                        f'{outprefix}_{featureset_name}.png')))


def test_learning_curve_plots():
    """
    Test learning curve plots for experiment with metrics option
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
                        f'{outprefix}_{featureset_name}.png')))


def test_learning_curve_plots_with_objectives():
    """
    Test learning curve plots for experiment with objectives option
    """

    # Test to validate learning curve output
    make_learning_curve_data()

    config_template_path = join(_my_dir,
                                'configs',
                                'test_learning_curve.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    # run the learning curve experiment
    run_configuration(config_path, quiet=True)
    outprefix = 'test_learning_curve'

    # make sure that the two PNG files (one per featureset) are created
    for featureset_name in ["test_learning_curve1", "test_learning_curve2"]:
        ok_(exists(join(_my_dir,
                        'output',
                        f'{outprefix}_{featureset_name}.png')))


def test_learning_curve_ylimits():
    """
    Test that the ylimits for learning curves are generated as expected.
    """

    # create a test data frame
    df_test = pd.DataFrame.from_dict({'test_score_std': {0: 0.16809136190418694,
                                                         1: 0.18556201422712379,
                                                         2: 0.15002727816517414,
                                                         3: 0.15301923832338646,
                                                         4: 0.15589815327205431,
                                                         5: 0.68205316443171948,
                                                         6: 0.77441075727706354,
                                                         7: 0.83838056331276678,
                                                         8: 0.84770116657005623,
                                                         9: 0.8708014559726478},
                                      'value': {0: 0.4092496971394447,
                                                1: 0.2820507715115001,
                                                2: 0.24533811547921261,
                                                3: 0.21808651942296109,
                                                4: 0.19767367891431534,
                                                5: -2.3540980769230773,
                                                6: -3.1312445327182394,
                                                7: -3.2956790939674137,
                                                8: -3.4843050005436713,
                                                9: -3.6357879085645455},
                                      'train_score_std': {0: 0.15950199460682787,
                                                          1: 0.090992452273091703,
                                                          2: 0.068488654201949981,
                                                          3: 0.055223120652733763,
                                                          4: 0.03172452509259388,
                                                          5: 1.0561586240460523,
                                                          6: 0.53955995320300709,
                                                          7: 0.40477740983901211,
                                                          8: 0.34148185048394258,
                                                          9: 0.20791478156554272},
                                      'variable': {0: 'train_score_mean',
                                                   1: 'train_score_mean',
                                                   2: 'train_score_mean',
                                                   3: 'train_score_mean',
                                                   4: 'train_score_mean',
                                                   5: 'train_score_mean',
                                                   6: 'train_score_mean',
                                                   7: 'train_score_mean',
                                                   8: 'train_score_mean',
                                                   9: 'train_score_mean'},
                                      'metric': {0: 'r2',
                                                 1: 'r2',
                                                 2: 'r2',
                                                 3: 'r2',
                                                 4: 'r2',
                                                 5: 'neg_mean_squared_error',
                                                 6: 'neg_mean_squared_error',
                                                 7: 'neg_mean_squared_error',
                                                 8: 'neg_mean_squared_error',
                                                 9: 'neg_mean_squared_error'}})

    # compute the y-limits
    ylimits_dict = _compute_ylimits_for_featureset(df_test, ['r2', 'neg_mean_squared_error'])

    eq_(len(ylimits_dict), 2)
    assert_almost_equal(ylimits_dict['neg_mean_squared_error'][0], -3.94, decimal=2)
    eq_(ylimits_dict['neg_mean_squared_error'][1], 0)
    eq_(ylimits_dict['r2'][0], 0)
    assert_almost_equal(ylimits_dict['r2'][1], 0.67, decimal=2)


@raises(ValueError)
def test_learning_curve_min_examples_check():
    """
    Test to check learning curve raises error with less than 500 examples
    """
    # generates a training split with less than 500 examples
    train_fs_less_than_500, _ = make_classification_data(num_examples=499,
                                                         train_test_ratio=1.0,
                                                         num_labels=3)

    # creating an example learner
    learner = Learner('LogisticRegression')

    # this must throw an error because `examples` has less than 500 items
    _ = learner.learning_curve(examples=train_fs_less_than_500, metric='accuracy')


def test_learning_curve_min_examples_check_override():
    """
    Test to check learning curve displays warning with less than 500 examples
    """

    # creates a logger which writes to a temporary log file
    log_file_path = join(_my_dir, "output",
                         "test_check_override_learning_curve_min_examples.log")
    logger = get_skll_logger("test_learning_curve_min_examples_check_override",
                             filepath=log_file_path)

    # generates a training split with less than 500 examples
    train_fs_less_than_500, _ = make_classification_data(num_examples=499,
                                                         train_test_ratio=1.0,
                                                         num_labels=3)

    # creating an example learner
    learner = Learner('LogisticRegression', logger=logger)

    # this must throw an error because `examples` has less than 500 items
    _ = learner.learning_curve(examples=train_fs_less_than_500, metric='accuracy',
                               override_minimum=True)

    # checks that the learning_curve warning message is contained in the log file
    with open(log_file_path) as tf:
        log_text = tf.read()
        learning_curve_warning_re = \
            re.compile(r'Because the number of training examples provided - '
                       r'\d+ - is less than the ideal minimum - \d+ - '
                       r'learning curve generation is unreliable'
                       r' and might break')
        assert learning_curve_warning_re.search(log_text)

    close_and_remove_logger_handlers(logger)


def check_pipeline_attribute(learner_name,
                             do_feature_hashing,
                             min_count,
                             scaling_type,
                             sampler_name,
                             learner,
                             function_args_dict):

    # look up the arguments that we computed earlier for
    # the current configuration
    estimator_type = learner.model_type._estimator_type
    train_fs, test_fs, feature_dicts, labels = function_args_dict[estimator_type][do_feature_hashing]

    # which metric score are we comparing between SKLL and sklearn
    metric = 'accuracy' if estimator_type == 'classifier' else 'r2'

    # train the given learner on the training set with the pipeline
    # attribute set to true
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.train(train_fs, grid_search=False)

    # get the predictions of the learner on the test set
    skll_predictions = learner.predict(test_fs)

    # get the metric score on the test set via SKLL
    (_,
     _,
     _,
     _,
     _,
     metric_scores) = learner.evaluate(test_fs, output_metrics=[metric])

    # now use the pipeline and get the predictions and get
    # the predictions and the score from the sklearn side
    pipeline = learner.pipeline
    sklearn_predictions = pipeline.predict(feature_dicts)
    sklearn_score = pipeline.score(feature_dicts, labels)

    # for classifiers, the predictions and scores should match exactly
    # but for regressors, they should be almost equal
    if estimator_type == 'classifier':
        assert_array_equal(skll_predictions, sklearn_predictions)
        eq_(metric_scores[metric], sklearn_score)
    else:
        assert_array_almost_equal(skll_predictions, sklearn_predictions)
        assert_almost_equal(metric_scores[metric], sklearn_score)


def test_pipeline_attribute():

    # define the classifier and regressor feature dictionaries and labels that we will test on
    # and also the classes and targets respectively
    cfeature_dicts = [{"f01": -2.87, "f02": 0.713, "f03": 2.86, "f04": 0.385, "f05": -0.989,
                       "f06": 0.380, "f07": -0.365, "f08": -0.224, "f09": 3.45, "f10": 0.622},
                      {"f01": 0.058, "f02": -1.14, "f03": 2.85, "f04": 1.41, "f05": 1.60,
                       "f06": 1.04, "f07": -0.669, "f08": -0.727, "f09": 1.82, "f10": 1.336},
                      {"f01": -1.80, "f02": 3.21, "f03": 0.79, "f04": -0.55, "f05": 0.059,
                       "f06": -5.66, "f07": -3.08, "f08": -0.95, "f09": 0.188, "f10": -1.24},
                      {"f01": 2.270, "f02": 2.271, "f03": 2.285, "f04": 2.951, "f05": 1.018,
                       "f06": -0.59, "f07": 0.432, "f08": 1.614, "f09": -0.69, "f10": -1.27},
                      {"f01": 2.98, "f02": 3.74, "f03": 1.96, "f04": 0.80, "f05": 0.425,
                       "f06": -0.76, "f07": 4.013, "f08": 3.119, "f09": 2.104, "f10": 0.195},
                      {"f01": 2.560, "f02": -2.05, "f03": 1.793, "f04": 0.955, "f05": 2.914,
                       "f06": 2.239, "f07": -1.41, "f08": -1.24, "f09": -4.44, "f10": 0.273},
                      {"f01": 1.86, "f02": -0.017, "f03": 1.337, "f04": -2.14, "f05": 2.255,
                       "f06": -1.21, "f07": -0.24, "f08": -0.66, "f09": -2.51, "f10": -1.06},
                      {"f01": -1.95, "f02": -1.81, "f03": 2.105, "f04": 0.976, "f05": -1.480,
                       "f06": 1.120, "f07": -1.22, "f08": 0.704, "f09": -3.66, "f10": -1.72},
                      {"f01": -1.54, "f02": -2.17, "f03": -4.18, "f04": 1.708, "f05": 0.514,
                       "f06": 0.354, "f07": -3.55, "f08": 2.285, "f09": -3.47, "f10": -0.79},
                      {"f01": 2.162, "f02": -0.71, "f03": -0.448, "f04": 0.326, "f05": 3.384,
                      "f06": -0.455, "f07": 1.253, "f08": 0.998, "f09": 3.193, "f10": 1.342}]
    classes = [1, 1, 0, 2, 1, 2, 0, 1, 2, 1]

    rfeature_dicts = [{'f1': 1.351, 'f2': -0.117, 'f3': 0.570, 'f4': 0.0619,
                       'f5': 1.569, 'f6': 0.805},
                      {'f1': -0.557, 'f2': -1.704, 'f3': 0.0913, 'f4': 0.767,
                       'f5': 1.281, 'f6': -0.803},
                      {'f1': 0.720, 'f2': -0.268, 'f3': 0.760, 'f4': 0.861,
                      'f5': -0.403, 'f6': 0.814},
                      {'f1': 1.737, 'f2': -0.228, 'f3': 1.340, 'f4': 2.031,
                      'f5': 2.170, 'f6': 1.498},
                      {'f1': 0.344, 'f2': 0.340, 'f3': 0.572, 'f4': -1.06,
                       'f5': 1.044, 'f6': 2.065},
                      {'f1': -0.489, 'f2': -0.420, 'f3': 0.428, 'f4': 0.707,
                       'f5': -1.306, 'f6': 0.0081},
                      {'f1': 0.805, 'f2': 0.570, 'f3': 1.351, 'f4': -0.117,
                       'f5': 0.0619, 'f6': 1.569},
                      {'f1': -1.083, 'f2': 0.0369, 'f3': -0.413, 'f4': 1.391,
                       'f5': 1.417, 'f6': -1.118},
                      {'f1': -1.945, 'f2': -0.332, 'f3': -1.393, 'f4': 0.952,
                       'f5': -0.816, 'f6': 1.417},
                      {'f1': 1.976, 'f2': -0.220, 'f3': -1.636, 'f4': 0.795,
                       'f5': -2.34, 'f6': -0.148}]
    targets = [96.057, -176.017, -182.32, -56.46, -50.14, -84.53, 241.71, -17.84,
               -47.09, 77.65]

    # create training featuresets that we will use to train our estimator
    function_args_dict = defaultdict(dict)
    for estimator_type in ['classifier', 'regressor']:
        for do_feature_hashing in [True, False]:
            if estimator_type == 'classifier':
                (train_fs, test_fs) = make_classification_data(num_examples=500,
                                                               num_features=10,
                                                               num_labels=3,
                                                               feature_bins=4,
                                                               non_negative=True,
                                                               use_feature_hashing=do_feature_hashing)
                labels = classes
                feature_dicts = cfeature_dicts
            else:
                (train_fs, test_fs, _) = make_regression_data(num_examples=500,
                                                              num_features=6,
                                                              feature_bins=4,
                                                              use_feature_hashing=do_feature_hashing)
                labels = targets
                feature_dicts = rfeature_dicts

            # if we are doing feature hashing, we need to transform our test
            # cases to the same space. If we are not, then we don't need to worry
            # beacuse we have manually ensured that the number of features are the
            # same for the non-hashing case (10 for classification, and 6 for
            # regression)
            test_fs = FeatureSet('test',
                                 ids=list(range(1, 11)),
                                 features=feature_dicts,
                                 labels=labels,
                                 vectorizer=train_fs.vectorizer if do_feature_hashing else None)
            function_args_dict[estimator_type][do_feature_hashing] = [train_fs,
                                                                      test_fs,
                                                                      feature_dicts,
                                                                      labels]
    function_args_dict = dict(function_args_dict)

    # now set up the test cases
    learners = ['LinearSVC', 'LogisticRegression',
                'MultinomialNB', 'SVC',
                'GradientBoostingClassifier', 'Lars',
                'LinearSVR', 'Ridge', 'SVR',
                'GradientBoostingRegressor']
    use_hashing = [True, False]
    min_feature_counts = [1, 2]
    samplers = [None, 'RBFSampler', 'SkewedChi2Sampler']
    scalers = ['none', 'with_mean', 'with_std', 'both']

    for (learner_name,
         do_feature_hashing,
         min_count,
         scaling_type,
         sampler_name) in product(learners,
                                  use_hashing,
                                  min_feature_counts,
                                  scalers,
                                  samplers):

        # skip the case for MultinomialNB with feature hashing
        # or feature sampling since it does not support those
        if learner_name == 'MultinomialNB':
            if do_feature_hashing or sampler_name is not None:
                continue

        # if we are using a SkewedChi2Sampler, we need to set the
        # some parameters to make sure it works as expected
        if sampler_name == 'SkewedChi2Sampler':
            sampler_kwargs = {'skewedness': 15, 'n_components': 10}
        else:
            sampler_kwargs = {}

        # create a learner instance with the given parameters
        # and with pipeline attribute set to True
        learner = Learner(learner_name,
                          min_feature_count=min_count,
                          sampler=sampler_name,
                          sampler_kwargs=sampler_kwargs,
                          feature_scaling=scaling_type,
                          pipeline=True)

        yield (check_pipeline_attribute,
               learner_name,
               do_feature_hashing,
               min_count,
               scaling_type,
               sampler_name,
               learner,
               function_args_dict)


def test_send_warnings_to_log():
    """
    Test that warnings get correctly sent to the logger.
    """
    # Run experiment

    suffix = '.jsonlines'
    train_path = join(_my_dir, 'train', f'test_send_warnings{suffix}')
    config_path = fill_in_config_paths_for_single_file(join(_my_dir,
                                                            "configs",
                                                            "test_send_warnings_to_log"
                                                            ".template.cfg"),
                                                       train_path,
                                                       None)
    run_configuration(config_path, quiet=True, local=True)

    # Check experiment log output
    # The experiment log file should contain warnings related
    # to the use of sklearn
    with open(join(_my_dir,
                   'output',
                   'test_send_warnings_to_log_train_test_send_warnings.'
                   'jsonlines_LinearSVC.log')) as f:
        log_content = f.read()
        convergence_sklearn_warning_re = \
            re.compile(r"WARNING - [^\n]+sklearn.svm._base\.py:\d+: "
                       r"ConvergenceWarning:Liblinear failed to converge, "
                       r"increase the number of iterations\.")
        assert convergence_sklearn_warning_re.search(log_content) is not None


def test_save_models_to_current_directory():
    """
    Test that saving models to current directory works.
    """

    # create a learner and train it on some data
    learner1 = Learner('LogisticRegression')
    train_path = join(_my_dir, 'train', 'f0.jsonlines')
    train_fs = NDJReader.for_path(train_path).read()
    learner1.train(train_fs, grid_search=False)

    # save this trained model into the current directory
    learner1.save("test_current_directory.model")

    # make sure that the model saved and that it's the same model
    ok_(exists("test_current_directory.model"))
    learner2 = Learner.from_file("test_current_directory.model")
    eq_(learner1.model_type, learner2.model_type)
    eq_(learner1.model_params, learner2.model_params)
    eq_(learner1.model_kwargs, learner2.model_kwargs)
