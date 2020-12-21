# License: BSD 3 clause
"""
Tests related to ablation experiments.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
"""

import csv
import json
from glob import glob
from os.path import join
from pathlib import Path

from nose.tools import eq_

from skll.experiments import run_configuration
from skll.utils.constants import KNOWN_DEFAULT_PARAM_GRIDS
from tests import config_dir, output_dir, test_dir, train_dir
from tests.utils import (
    create_jsonlines_feature_files,
    fill_in_config_paths,
    remove_jsonlines_feature_files,
    unlink,
)

_ALL_MODELS = list(KNOWN_DEFAULT_PARAM_GRIDS.keys())


def setup():
    """
    Create necessary directories for testing.
    """
    for dir_path in [train_dir, test_dir, output_dir]:
        Path(dir_path).mkdir(exist_ok=True)

    # create jsonlines feature files
    create_jsonlines_feature_files(train_dir)


def tearDown():
    """
    Clean up after tests.
    """

    for output_file in glob(join(output_dir, 'ablation_cv_*')):
        unlink(output_file)

    config_files = ['test_ablation.cfg',
                    'test_ablation_all_combos.cfg',
                    'test_ablation_feature_hasher.cfg',
                    'test_ablation_feature_hasher_all_combos.cfg',
                    'test_ablation_sampler.cfg',
                    'test_ablation_sampler_all_combos.cfg',
                    'test_ablation_feature_hasher_sampler.cfg',
                    'test_ablation_feature_hasher_sampler_all_combos.cfg']
    for cf in config_files:
        unlink(Path(config_dir) / cf)

    remove_jsonlines_feature_files(train_dir)


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

    config_template_path = join(config_dir, 'test_ablation.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=1)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(output_dir, 'ablation_cv_plain_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 7 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob(join(output_dir,
                                     'ablation_cv_plain*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_all_combos():
    """
    Test ablation all-combos + cross-validation
    """

    config_template_path = join(config_dir,
                                'test_ablation_all_combos.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=None)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(output_dir, 'ablation_cv_plain_all_combos_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob(join(output_dir,
                                     'ablation_cv_plain_all_combos*results')))
    eq_(num_result_files, 20)


def test_ablation_cv_feature_hasher():
    """
    Test ablation + cross-validation + feature hashing
    """

    config_template_path = join(config_dir,
                                'test_ablation_feature_hasher.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=1)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(output_dir,
                   'ablation_cv_feature_hasher_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 7 ablated featuresets * 2 learners = 14 results files
    num_result_files = len(glob(join(output_dir,
                                     'ablation_cv_feature_hasher_*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_feature_hasher_all_combos():
    """
    Test ablation all-combos + cross-validation + feature hashing
    """

    config_template_path = join(config_dir,
                                'test_ablation_feature_hasher_all_combos.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=None)

    # read in the summary file and make sure it has
    #    10 ablated featuresets
    #      * (10 folds + 1 average line)
    #      * 2 learners
    #    = 220 lines in total
    with open(join(output_dir,
                   'ablation_cv_feature_hasher_all_combos_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob(join(output_dir,
                                     'ablation_cv_feature_hasher_all_combos*.results')))
    eq_(num_result_files, 20)


def test_ablation_cv_sampler():
    """
    Test ablation + cross-validation + samplers
    """

    config_template_path = join(config_dir,
                                'test_ablation_sampler.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=1)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(output_dir, 'ablation_cv_sampler_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 6 ablated featuresets * 2 learners = 12 results files
    num_result_files = len(glob(join(output_dir,
                                     'ablation_cv_sampler*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_all_combos_sampler():
    """
    Test ablation all-combos + cross-validation + samplers
    """

    config_template_path = join(config_dir,
                                'test_ablation_sampler_all_combos.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=None)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(output_dir, 'ablation_cv_sampler_all_combos_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob(join(output_dir,
                                     'ablation_cv_sampler_all_combos*.results')))
    eq_(num_result_files, 20)


def test_ablation_cv_feature_hasher_sampler():
    """
    Test ablation + cross-validation + feature hashing + samplers
    """

    config_template_path = join(config_dir,
                                'test_ablation_feature_hasher_sampler.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=1)

    # read in the summary file and make sure it has
    # 7 ablated featuresets * (10 folds + 1 average line) * 2 learners = 154
    # lines
    with open(join(output_dir,
                   'ablation_cv_feature_hasher_sampler_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 154)

    # make sure there are 7 ablated featuresets * 2 learners = 14 results files
    num_result_files = len(glob(join(output_dir,
                                     'ablation_cv_feature_hasher_sampler*.results')))
    eq_(num_result_files, 14)


def test_ablation_cv_feature_hasher_all_combos_sampler():
    """
    Test ablation all-combos + cross-validation + feature hashing + samplers
    """

    config_template_path = join(config_dir,
                                'test_ablation_feature_hasher_sampler_all_combos.template.cfg')
    config_path = fill_in_config_paths(config_template_path)

    run_configuration(config_path, quiet=True, ablation=None)

    # read in the summary file and make sure it has
    # 10 ablated featuresets * (10 folds + 1 average line) * 2 learners = 220
    # lines
    with open(join(output_dir,
                   'ablation_cv_feature_hasher_all_combos_summary.tsv')) as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        num_rows = check_ablation_rows(reader)
        eq_(num_rows, 220)

    # make sure there are 10 ablated featuresets * 2 learners = 20 results
    # files
    num_result_files = len(glob(join(output_dir,
                                     'ablation_cv_feature_hasher_sampler_all_combos*.results')))
    eq_(num_result_files, 20)
