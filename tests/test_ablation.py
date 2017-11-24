# License: BSD 3 clause
"""
Tests related to ablation experiments.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import json
import os

from glob import glob
from io import open
from os.path import abspath, dirname, exists, join

from nose.tools import eq_
from skll.experiments import run_configuration
from skll.learner import _DEFAULT_PARAM_GRIDS

from utils import (create_jsonlines_feature_files,
                   fill_in_config_paths)


_ALL_MODELS = list(_DEFAULT_PARAM_GRIDS.keys())
_my_dir = abspath(dirname(__file__))


def setup():
    """
    Create necessary directories for testing.
    """
    train_dir = join(_my_dir, 'train')
    if not exists(train_dir):
        os.makedirs(train_dir)
    test_dir = join(_my_dir, 'test')
    if not exists(test_dir):
        os.makedirs(test_dir)
    output_dir = join(_my_dir, 'output')
    if not exists(output_dir):
        os.makedirs(output_dir)

    # create jsonlines feature files
    train_path = join(_my_dir, 'train')
    create_jsonlines_feature_files(train_path)


def tearDown():
    """
    Clean up after tests.
    """
    output_dir = join(_my_dir, 'output')
    config_dir = join(_my_dir, 'configs')

    for output_file in glob(join(output_dir, 'ablation_cv_*')):
        os.unlink(output_file)

    config_files = ['test_ablation.cfg',
                    'test_ablation_all_combos.cfg',
                    'test_ablation_feature_hasher.cfg',
                    'test_ablation_feature_hasher_all_combos.cfg',
                    'test_ablation_sampler.cfg',
                    'test_ablation_sampler_all_combos.cfg',
                    'test_ablation_feature_hasher_sampler.cfg',
                    'test_ablation_feature_hasher_sampler_all_combos.cfg']
    for cf in config_files:
        if exists(join(config_dir, cf)):
            os.unlink(join(config_dir, cf))


def check_ablation_rows(reader):
    """
    Helper function to ensure that all ablated_features and featureset values
    are correct for each row in results summary file.

    :returns: Number of items in reader
    """
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
    """
    Test ablation + cross-validation
    """

    config_template_path = join(_my_dir,
                                'configs',
                                'test_ablation.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=1)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_plain_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 7 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob(join(_my_dir,
                                     'output',
                                     'ablation_cv_plain*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_all_combos():
    """
    Test ablation all-combos + cross-validation
    """

    config_template_path = join(_my_dir,
                                'configs',
                                'test_ablation_all_combos.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=None)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_plain_all_combos_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob(join(_my_dir,
                                     'output',
                                     'ablation_cv_plain_all_combos*results')))
    eq_(num_result_files, 20)


def test_ablation_cv_feature_hasher():
    """
    Test ablation + cross-validation + feature hashing
    """

    config_template_path = join(_my_dir,
                                'configs',
                                'test_ablation_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=1)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(_my_dir,
                   'output',
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 7 ablated featuresets * 2 learners = 14 results files
    num_result_files = len(glob(join(_my_dir,
                                     'output',
                                     'ablation_cv_feature_hasher_*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_feature_hasher_all_combos():
    """
    Test ablation all-combos + cross-validation + feature hashing
    """

    config_template_path = join(_my_dir,
                                'configs',
                                'test_ablation_feature_hasher_all_combos.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=None)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(_my_dir,
                   'output',
                   'ablation_cv_feature_hasher_all_combos_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob(join(_my_dir,
                                    'output',
                                    'ablation_cv_feature_hasher_all_combos*.results')))
    eq_(num_result_files, 20)


def test_ablation_cv_sampler():
    """
    Test ablation + cross-validation + samplers
    """

    config_template_path = join(_my_dir,
                                'configs',
                                'test_ablation_sampler.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=1)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_sampler_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob(join(_my_dir,
                                    'output',
                                    'ablation_cv_sampler*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_all_combos_sampler():
    """
    Test ablation all-combos + cross-validation + samplers
    """

    config_template_path = join(_my_dir,
                                'configs',
                                'test_ablation_sampler_all_combos.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=None)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(_my_dir, 'output', 'ablation_cv_sampler_all_combos_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob(join(_my_dir,
                                     'output',
                                     'ablation_cv_sampler_all_combos*.results')))
    eq_(num_result_files, 20)


def test_ablation_cv_feature_hasher_sampler():
    """
    Test ablation + cross-validation + feature hashing + samplers
    """

    config_template_path = join(_my_dir,
                                'configs',
                                'test_ablation_feature_hasher_sampler.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=1)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(_my_dir,
                   'output',
                   'ablation_cv_feature_hasher_sampler_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 7 ablated featuresets * 2 learners = 14 results files
    num_result_files = len(glob(join(_my_dir,
                                     'output',
                                     'ablation_cv_feature_hasher_sampler*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_feature_hasher_all_combos_sampler():
    """
    Test ablation all-combos + cross-validation + feature hashing + samplers
    """

    config_template_path = join(_my_dir,
                                'configs',
                                'test_ablation_feature_hasher_sampler_all_combos.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=None)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(_my_dir,
                   'output',
                   'ablation_cv_feature_hasher_all_combos_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob(join(_my_dir,
                                     'output',
                                     'ablation_cv_feature_hasher_sampler_all_combos*.results')))
    eq_(num_result_files, 20)
