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
import json
import os
import re
from collections import OrderedDict
from io import open
from os.path import abspath, dirname, exists, join

import numpy as np
import scipy.sparse as sp
from nose.tools import eq_
from skll.data import FeatureSet, NDJWriter
from skll.experiments import _setup_config_parser, run_ablation
from skll.learner import _DEFAULT_PARAM_GRIDS


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
        train_fs = FeatureSet('ablation_cv', ids, features=sub_features, classes=classes)
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
