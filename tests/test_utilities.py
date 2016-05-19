# License: BSD 3 clause
"""
Module for running unit tests related to command line utilities.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import ast

import copy
import itertools
import os
import sys


from io import open
from glob import glob
from itertools import combinations, product
from os.path import abspath, dirname, exists, join
from six import StringIO

try:
    from unittest.mock import create_autospec, patch
except ImportError:
    from mock import create_autospec, patch

from nose.tools import eq_, assert_almost_equal, raises, assert_raises
from numpy.testing import assert_array_equal, assert_allclose, assert_array_almost_equal

import skll
import skll.utilities.compute_eval_from_predictions as cefp
import skll.utilities.filter_features as ff
import skll.utilities.generate_predictions as gp
import skll.utilities.print_model_weights as pmw
import skll.utilities.run_experiment as rex
import skll.utilities.skll_convert as sk
import skll.utilities.summarize_results as sr
import skll.utilities.filter_features as ff
import skll.utilities.join_features as jf

from skll.data import (FeatureSet, NDJWriter, LibSVMWriter,
                       MegaMWriter, LibSVMReader, safe_float)
from skll.data.readers import EXT_TO_READER
from skll.data.writers import EXT_TO_WRITER
from skll.experiments import _write_summary_file, run_configuration
from skll.learner import Learner, _DEFAULT_PARAM_GRIDS

from utils import make_classification_data, make_regression_data


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


def tearDown():
    """
    Clean up after tests.
    """
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')
    other_dir = join(_my_dir, 'other')

    if exists(join(test_dir, 'test_generate_predictions.jsonlines')):
        os.unlink(join(test_dir, 'test_generate_predictions.jsonlines'))

    for model_chunk in glob(join(output_dir,
                                 'test_print_model_weights.model*')):
        os.unlink(model_chunk)

    for model_chunk in glob(join(output_dir,
                                 'test_generate_predictions.model*')):
        os.unlink(model_chunk)

    for model_chunk in glob(join(output_dir,
                                 'test_generate_predictions_console.model*')):
        os.unlink(model_chunk)

    for f in glob(join(other_dir, 'test_skll_convert*')):
        os.unlink(f)

    if exists(join(other_dir, 'summary_file')):
        os.unlink(join(other_dir, 'summary_file'))

    if exists(join(other_dir, 'test_filter_features_input.arff')):
        os.unlink(join(other_dir, 'test_filter_features_input.arff'))
    for ffile in glob(join(other_dir,
                           'test_join_features*')):
        os.unlink(ffile)


def test_compute_eval_from_predictions():
    """
    Test compute_eval_from_predictions function console script
    """

    pred_path = join(_my_dir, 'other',
                     'test_compute_eval_from_predictions.predictions')
    input_path = join(_my_dir, 'other',
                      'test_compute_eval_from_predictions.jsonlines')

    # we need to capture stdout since that's what main() writes to
    compute_eval_from_predictions_cmd = [input_path, pred_path, 'pearson',
                                         'unweighted_kappa']
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = mystderr = StringIO()
        cefp.main(compute_eval_from_predictions_cmd)
        score_rows = mystdout.getvalue().strip().split('\n')
        err = mystderr.getvalue()
        print(err)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    scores = {}
    for score_row in score_rows:
        score, metric_name, pred_path = score_row.split('\t')
        scores[metric_name] = float(score)

    assert_almost_equal(scores['pearson'], 0.6197797868009122)
    assert_almost_equal(scores['unweighted_kappa'], 0.2)


def check_generate_predictions(use_feature_hashing=False, use_threshold=False):

    # create some simple classification data without feature hashing
    train_fs, test_fs = make_classification_data(
        num_examples=1000, num_features=5,
        use_feature_hashing=use_feature_hashing, feature_bins=4)

    # create a learner that uses an SGD classifier
    learner = Learner('SGDClassifier', probability=use_threshold)

    # train the learner with grid search
    learner.train(train_fs, grid_search=True)

    # get the predictions on the test featureset
    predictions = learner.predict(test_fs)

    # if we asked for probabilities, then use the threshold
    # to convert them into binary predictions
    if use_threshold:
        threshold = 0.6
        predictions = [int(p[1] >= threshold) for p in predictions]
    else:
        predictions = predictions.tolist()
        threshold = None

    # save the learner to a file
    model_file = join(_my_dir, 'output',
                      'test_generate_predictions.model')
    learner.save(model_file)

    # now use Predictor to generate the predictions and make
    # sure that they are the same as before saving the model
    p = gp.Predictor(model_file, threshold=threshold)
    predictions_after_saving = p.predict(test_fs)

    eq_(predictions, predictions_after_saving)


def test_generate_predictions():
    """
    Test generate predictions API with hashing and a threshold
    """

    yield check_generate_predictions, False, False
    yield check_generate_predictions, True, False
    yield check_generate_predictions, False, True
    yield check_generate_predictions, True, True


def check_generate_predictions_console(use_threshold=False):

    # create some simple classification data without feature hashing
    train_fs, test_fs = make_classification_data(num_examples=1000,
                                                 num_features=5)

    # save the test feature set to an NDJ file
    input_file = join(_my_dir, 'test',
                      'test_generate_predictions.jsonlines')
    writer = NDJWriter(input_file, test_fs)
    writer.write()

    # create a learner that uses an SGD classifier
    learner = Learner('SGDClassifier', probability=use_threshold)

    # train the learner with grid search
    learner.train(train_fs, grid_search=True)

    # get the predictions on the test featureset
    predictions = learner.predict(test_fs)

    # if we asked for probabilities, then use the threshold
    # to convert them into binary predictions
    if use_threshold:
        threshold = 0.6
        predictions = [int(p[1] >= threshold) for p in predictions]
    else:
        predictions = predictions.tolist()
        threshold = None

    # save the learner to a file
    model_file = join(_my_dir, 'output',
                      'test_generate_predictions_console.model')
    learner.save(model_file)

    # now call main() from generate_predictions.py
    generate_cmd = []
    if use_threshold:
        generate_cmd.append('-t {}'.format(threshold))
    generate_cmd.extend([model_file, input_file])

    # we need to capture stdout since that's what main() writes to
    err = ''
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = mystderr = StringIO()
        gp.main(generate_cmd)
        out = mystdout.getvalue()
        err = mystderr.getvalue()
        predictions_after_saving = [int(x) for x in out.strip().split('\n')]
        eq_(predictions, predictions_after_saving)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print(err)


def test_generate_predictions_console():
    """
    Test generate_predictions as a console script with/without a threshold
    """

    yield check_generate_predictions_console, False
    yield check_generate_predictions_console, True


def check_skll_convert(from_suffix, to_suffix):

    # create some simple classification data
    orig_fs, _ = make_classification_data(train_test_ratio=1.0,
                                          one_string_feature=True)

    # now write out this feature set in the given suffix
    from_suffix_file = join(_my_dir, 'other',
                            'test_skll_convert_in{}'.format(from_suffix))
    to_suffix_file = join(_my_dir, 'other',
                          'test_skll_convert_out{}'.format(to_suffix))

    writer = EXT_TO_WRITER[from_suffix](from_suffix_file, orig_fs, quiet=True)
    writer.write()

    # now run skll convert to convert the featureset into the other format
    skll_convert_cmd = [from_suffix_file, to_suffix_file, '--quiet']

    # we need to capture stderr to make sure we don't miss any errors
    err = ''
    try:
        old_stderr = sys.stderr
        sys.stderr = mystderr = StringIO()
        sk.main(skll_convert_cmd)
        err = mystderr.getvalue()
    finally:
        sys.stderr = old_stderr
        print(err)

    # now read the converted file
    reader = EXT_TO_READER[to_suffix](to_suffix_file, quiet=True)
    converted_fs = reader.read()

    # ensure that the original and the converted feature sets
    # are the same
    eq_(orig_fs, converted_fs)


def test_skll_convert():
    for from_suffix, to_suffix in itertools.permutations(['.jsonlines', '.ndj',
                                                          '.megam', '.tsv',
                                                          '.csv', '.arff',
                                                          '.libsvm'], 2):
        yield check_skll_convert, from_suffix, to_suffix


def test_skll_convert_libsvm_map():
    """
    Test to check whether the --reuse_libsvm_map option works for skll_convert
    """

    # create some simple classification data
    orig_fs, _ = make_classification_data(train_test_ratio=1.0,
                                          one_string_feature=True)

    # now write out this feature set as a libsvm file
    orig_libsvm_file = join(_my_dir, 'other',
                            'test_skll_convert_libsvm_map.libsvm')
    writer = LibSVMWriter(orig_libsvm_file, orig_fs, quiet=True)
    writer.write()

    # now make a copy of the dataset
    swapped_fs = copy.deepcopy(orig_fs)

    # now modify this new featureset to swap the first two columns
    del swapped_fs.vectorizer.vocabulary_['f01']
    del swapped_fs.vectorizer.vocabulary_['f02']
    swapped_fs.vectorizer.vocabulary_['f01'] = 1
    swapped_fs.vectorizer.vocabulary_['f02'] = 0
    tmp = swapped_fs.features[:, 0]
    swapped_fs.features[:, 0] = swapped_fs.features[:, 1]
    swapped_fs.features[:, 1] = tmp

    # now write out this new feature set as a MegaM file
    swapped_megam_file = join(_my_dir, 'other',
                              'test_skll_convert_libsvm_map.megam')
    writer = MegaMWriter(swapped_megam_file, swapped_fs, quiet=True)
    writer.write()

    # now run skll_convert to convert this into a libsvm file
    # but using the mapping specified in the first libsvm file
    converted_libsvm_file = join(_my_dir, 'other',
                                 'test_skll_convert_libsvm_map2.libsvm')

    # now call skll convert's main function
    skll_convert_cmd = ['--reuse_libsvm_map', orig_libsvm_file,
                        '--quiet', orig_libsvm_file,
                        converted_libsvm_file]
    err = ''
    try:
        old_stderr = sys.stderr
        sys.stderr = mystderr = StringIO()
        sk.main(skll_convert_cmd)
        err = mystderr.getvalue()
    finally:
        sys.stderr = old_stderr
        print(err)

    # now read the converted libsvm file into a featureset
    reader = LibSVMReader(converted_libsvm_file, quiet=True)
    converted_fs = reader.read()

    # now ensure that this new featureset and the original
    # featureset are the same
    eq_(orig_fs, converted_fs)


def check_print_model_weights(task='classification'):

    # create some simple classification or regression data
    if task == 'classification' or task == 'classification_no_intercept':
        train_fs, _ = make_classification_data(train_test_ratio=0.8)
    elif task == 'multiclass_classification':
        train_fs, _ = make_classification_data(train_test_ratio=0.8, num_labels=3)
    else:
        train_fs, _, _ = make_regression_data(num_features=4,
                                              train_test_ratio=0.8)

    # now train the appropriate model
    if task == 'classification' or task == 'multiclass_classification':
        learner = Learner('LogisticRegression')
        learner.train(train_fs, grid_objective='f1_score_micro')
    elif task == 'classification_no_intercept':
        learner = Learner('LogisticRegression')
        learner.train(train_fs, grid_objective='f1_score_micro', param_grid=[{'fit_intercept':[False]}])
    elif task == 'regression':
        learner = Learner('LinearRegression')
        learner.train(train_fs, grid_objective='pearson')
    else:
        learner = Learner('LinearSVR')
        learner.train(train_fs, grid_objective='pearson')

    # now save the model to disk
    model_file = join(_my_dir, 'output',
                      'test_print_model_weights.model')
    learner.save(model_file)

    # now call print_model_weights main() and capture the output
    print_model_weights_cmd = [model_file]
    err = ''
    try:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = mystderr = StringIO()
        sys.stdout = mystdout = StringIO()
        pmw.main(print_model_weights_cmd)
        out = mystdout.getvalue()
        err = mystderr.getvalue()
    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        print(err)

    # now parse the output of the print_model_weight command
    # and get the intercept and the feature values
    if task == 'classification':
        lines_to_parse = [l for l in out.split('\n')[1:] if l]
        intercept = safe_float(lines_to_parse[0].split('\t')[0])
        feature_values = []
        for ltp in lines_to_parse[1:]:
            fields = ltp.split('\t')
            feature_values.append((fields[2], safe_float(fields[0])))
        feature_values = [t[1] for t in sorted(feature_values)]
        assert_almost_equal(intercept, learner.model.intercept_[0])
        assert_allclose(learner.model.coef_[0], feature_values)
    elif task == 'multiclass_classification':
        # for multiple classes we get an intercept for each class
        # as well as a list of weights for each class

        lines_to_parse = [l for l in out.split('\n')[1:] if l]
        intercept = []
        for intercept_string in lines_to_parse[0:3]:
            intercept.append(safe_float(intercept_string.split('\t')[0]))

        feature_values = [[], [], []]
        for ltp in lines_to_parse[3:]:
            fields = ltp.split('\t')
            feature_values[int(fields[1])].append((fields[2], safe_float(fields[0])))

        for index, weights in enumerate(feature_values):
            feature_values[index] = [t[1] for t in sorted(weights)]

        for index, weights in enumerate(learner.model.coef_):
            assert_array_almost_equal(weights, feature_values[index])

        assert_array_almost_equal(intercept, learner.model.intercept_)
    elif task == 'classification_no_intercept':
        lines_to_parse = [l for l in out.split('\n')[0:] if l]
        intercept = safe_float(lines_to_parse[0].split('=')[1])
        feature_values = []
        for ltp in lines_to_parse[1:]:
            fields = ltp.split('\t')
            feature_values.append((fields[2], safe_float(fields[0])))
        feature_values = [t[1] for t in sorted(feature_values)]
        assert_almost_equal(intercept, learner.model.intercept_)
        assert_allclose(learner.model.coef_[0], feature_values)
    elif task == 'regression':
        lines_to_parse = [l for l in out.split('\n') if l]
        intercept = safe_float(lines_to_parse[0].split('=')[1])
        feature_values = []
        for ltp in lines_to_parse[1:]:
            fields = ltp.split('\t')
            feature_values.append((fields[1], safe_float(fields[0])))
        feature_values = [t[1] for t in sorted(feature_values)]
        assert_almost_equal(intercept, learner.model.intercept_)
        assert_allclose(learner.model.coef_, feature_values)
    else:
        lines_to_parse = [l for l in out.split('\n') if l]

        intercept_list = ast.literal_eval(lines_to_parse[0].split('=')[1].strip())
        intercept = []
        for intercept_string in intercept_list:
            intercept.append(safe_float(intercept_string))

        feature_values = []
        for ltp in lines_to_parse[1:]:
            fields = ltp.split('\t')
            feature_values.append((fields[1], safe_float(fields[0])))
        feature_values = [t[1] for t in sorted(feature_values)]

        assert_array_almost_equal(intercept, learner.model.intercept_)
        assert_allclose(learner.model.coef_, feature_values)


def test_print_model_weights():
    yield check_print_model_weights, 'classification'
    yield check_print_model_weights, 'multiclass_classification'
    yield check_print_model_weights, 'classification_no_intercept'
    yield check_print_model_weights, 'regression'
    yield check_print_model_weights, 'regression_linearSVR'


def check_summarize_results_argparse(use_ablation=False):
    """
    A utility function to check that we are setting up argument parsing
    correctly for summarize_results. We are not checking whether the summaries
    produced are accurate because we have separate tests for that.
    """

    # replace the _write_summary_file function that's called
    # by the main() in summarize_results with a mocked up version
    write_summary_file_mock = create_autospec(_write_summary_file)
    sr._write_summary_file = write_summary_file_mock

    # now call main with some arguments
    summary_file_name = join(_my_dir, 'other', 'summary_file')
    list_of_input_files = ['infile1', 'infile2', 'infile3']
    sr_cmd_args = [summary_file_name]
    sr_cmd_args.extend(list_of_input_files)
    if use_ablation:
        sr_cmd_args.append('--ablation')
    sr.main(argv=sr_cmd_args)

    # now check to make sure that _write_summary_file (or our mocked up version
    # of it) got the arguments that we passed
    positional_arguments, keyword_arguments = write_summary_file_mock.call_args
    eq_(positional_arguments[0], list_of_input_files)
    eq_(positional_arguments[1].name, summary_file_name)
    eq_(keyword_arguments['ablation'], int(use_ablation))


def test_summarize_results_argparse():
    yield check_summarize_results_argparse, False
    yield check_summarize_results_argparse, True


def check_run_experiments_argparse(multiple_config_files=False,
                                   n_ablated_features='1',
                                   keep_models=False,
                                   local=False,
                                   resume=False):
    """
    A utility function to check that we are setting up argument parsing
    correctly for run_experiment.  We are not checking whether the results are
    correct because we have separate tests for that.
    """

    # replace the run_configuration function that's called
    # by the main() in run_experiment with a mocked up version
    run_configuration_mock = create_autospec(run_configuration)
    rex.run_configuration = run_configuration_mock

    # now call main with some arguments
    config_file1_name = join(_my_dir, 'other', 'config_file1')
    config_files = [config_file1_name]
    rex_cmd_args = [config_file1_name]
    if multiple_config_files:
        config_file2_name = join(_my_dir, 'other', 'config_file2')
        rex_cmd_args.extend([config_file2_name])
        config_files.extend([config_file2_name])

    if n_ablated_features != 'all':
        rex_cmd_args.extend(['-a', '{}'.format(n_ablated_features)])
    else:
        rex_cmd_args.append('-A')

    if keep_models:
        rex_cmd_args.append('-k')

    if resume:
        rex_cmd_args.append('-r')

    if local:
        rex_cmd_args.append('-l')
    else:
        machine_list = ['"foo.1.org"', '"x.test.com"', '"z.a.b.d"']
        rex_cmd_args.append('-m')
        rex_cmd_args.append('{}'.format(','.join(machine_list)))

    rex_cmd_args.extend(['-q', 'foobar.q'])

    rex.main(argv=rex_cmd_args)

    # now check to make sure that run_configuration (or our mocked up version
    # of it) got the arguments that we passed
    positional_arguments, keyword_arguments = run_configuration_mock.call_args

    if multiple_config_files:
        eq_(positional_arguments[0], config_files[1])
    else:
        eq_(positional_arguments[0], config_file1_name)

    if n_ablated_features != 'all':
        eq_(keyword_arguments['ablation'], int(n_ablated_features))
    else:
        eq_(keyword_arguments['ablation'], None)

    if local:
        eq_(keyword_arguments['local'], local)
    else:
        eq_(keyword_arguments['hosts'], machine_list)

    eq_(keyword_arguments['overwrite'], not keep_models)
    eq_(keyword_arguments['queue'], 'foobar.q')
    eq_(keyword_arguments['resume'], resume)


def test_run_experiment_argparse():
    for (multiple_config_files,
         n_ablated_features,
         keep_models, local,
         resume) in product([True, False],
                            ['2', 'all'],
                            [True, False],
                            [True, False],
                            [True, False]):

        yield (check_run_experiments_argparse, multiple_config_files,
               n_ablated_features, keep_models, local, resume)


def check_filter_features_no_arff_argparse(extension, filter_type,
                                           label_col='y', id_col='id',
                                           inverse=False, quiet=False):
    """
    A utility function to check that we are setting up argument parsing
    correctly for filter_features for ALL file types except ARFF.
    We are not checking whether the results are correct because we
    have separate tests for that.
    """

    # replace the run_configuration function that's called
    # by the main() in filter_feature with a mocked up version
    reader_class = EXT_TO_READER[extension]
    writer_class = EXT_TO_WRITER[extension]

    # create some dummy input and output filenames
    infile = 'foo{}'.format(extension)
    outfile = 'bar{}'.format(extension)

    # create a simple featureset with actual ids, labels and features
    fs, _ = make_classification_data(num_labels=3, train_test_ratio=1.0)

    ff_cmd_args = [infile, outfile]

    if filter_type == 'feature':
        if inverse:
            features_to_keep = ['f01', 'f04', 'f07', 'f10']
        else:
            features_to_keep = ['f02', 'f03', 'f05', 'f06', 'f08', 'f09']

        ff_cmd_args.append('-f')

        for f in features_to_keep:
            ff_cmd_args.append(f)

    elif filter_type == 'id':
        if inverse:
            ids_to_keep = ['EXAMPLE_{}'.format(x) for x in range(1, 100, 2)]
        else:
            ids_to_keep = ['EXAMPLE_{}'.format(x) for x in range(2, 102, 2)]

        ff_cmd_args.append('-I')

        for idee in ids_to_keep:
            ff_cmd_args.append(idee)

    elif filter_type == 'label':
        if inverse:
            labels_to_keep = ['0', '1']
        else:
            labels_to_keep = ['2']

        ff_cmd_args.append('-L')

        for lbl in labels_to_keep:
            ff_cmd_args.append(lbl)

    ff_cmd_args.extend(['-l', label_col])
    ff_cmd_args.extend(['--id_col', id_col])

    if inverse:
        ff_cmd_args.append('-i')

    if quiet:
        ff_cmd_args.append('-q')

    # substitute mock methods for the three main methods that get called by
    # filter_features: the __init__() method of the appropriate reader,
    # FeatureSet.filter() and the __init__() method of the appropriate writer.
    # We also need to mock the read() and write() methods to prevent actual
    # reading and writing.
    with patch.object(reader_class, '__init__', autospec=True,
                      return_value=None) as read_init_mock, \
            patch.object(reader_class, 'read', autospec=True, return_value=fs),\
            patch.object(FeatureSet, 'filter', autospec=True) as filter_mock, \
            patch.object(writer_class, '__init__', autospec=True,
                         return_value=None) as write_init_mock, \
            patch.object(writer_class, 'write', autospec=True):

        ff.main(argv=ff_cmd_args)

        # get the various arguments from the three mocked up methods
        read_pos_arguments, read_kw_arguments = read_init_mock.call_args
        filter_pos_arguments, filter_kw_arguments = filter_mock.call_args
        write_pos_arguments, write_kw_arguments = write_init_mock.call_args

        # make sure that the arguments they got were the ones we specified
        eq_(read_pos_arguments[1], infile)
        eq_(read_kw_arguments['quiet'], quiet)
        eq_(read_kw_arguments['label_col'], label_col)
        eq_(read_kw_arguments['id_col'], id_col)

        eq_(write_pos_arguments[1], outfile)
        eq_(write_kw_arguments['quiet'], quiet)

        # Note that we cannot test the label_col column for the writer.
        # The reason is that it is set conditionally and those conditions
        # do not execute with mocking.
        eq_(filter_pos_arguments[0], fs)
        eq_(filter_kw_arguments['inverse'], inverse)

        if filter_type == 'feature':
            eq_(filter_kw_arguments['features'], features_to_keep)
        elif filter_type == 'id':
            eq_(filter_kw_arguments['ids'], ids_to_keep)
        elif filter_type == 'label':
            eq_(filter_kw_arguments['labels'], labels_to_keep)


def test_filter_features_no_arff_argparse():
    for (extension, filter_type, id_col,
         label_col, inverse, quiet) in product(['.jsonlines', '.ndj',
                                                '.megam', '.tsv',
                                                '.csv', ],
                                               ['feature', 'id',
                                                'label'],
                                               ['id', 'id_foo'],
                                               ['y', 'foo'],
                                               [True, False],
                                               [True, False]):

        yield (check_filter_features_no_arff_argparse, extension,
               filter_type, label_col, id_col, inverse, quiet)


def check_filter_features_arff_argparse(filter_type, label_col='y',
                                        id_col='id', inverse=False,
                                        quiet=False):
    """
    A utility function to check that we are setting up argument parsing
    correctly for filter_features for ARFF file types. We are not checking
    whether the results are correct because we have separate tests for that.
    """

    # replace the run_configuration function that's called
    # by the main() in filter_feature with a mocked up version
    writer_class = skll.data.writers.ARFFWriter

    # create some dummy input and output filenames
    infile = join(_my_dir, 'other', 'test_filter_features_input.arff')
    outfile = 'bar.arff'

    # create a simple featureset with actual ids, labels and features
    fs, _ = make_classification_data(num_labels=3, train_test_ratio=1.0)

    writer = writer_class(infile, fs, label_col=label_col, id_col=id_col)
    writer.write()

    ff_cmd_args = [infile, outfile]

    if filter_type == 'feature':
        if inverse:
            features_to_keep = ['f01', 'f04', 'f07', 'f10']
        else:
            features_to_keep = ['f02', 'f03', 'f05', 'f06', 'f08', 'f09']

        ff_cmd_args.append('-f')

        for f in features_to_keep:
            ff_cmd_args.append(f)

    elif filter_type == 'id':
        if inverse:
            ids_to_keep = ['EXAMPLE_{}'.format(x) for x in range(1, 100, 2)]
        else:
            ids_to_keep = ['EXAMPLE_{}'.format(x) for x in range(2, 102, 2)]

        ff_cmd_args.append('-I')

        for idee in ids_to_keep:
            ff_cmd_args.append(idee)

    elif filter_type == 'label':
        if inverse:
            labels_to_keep = ['0', '1']
        else:
            labels_to_keep = ['2']

        ff_cmd_args.append('-L')

        for lbl in labels_to_keep:
            ff_cmd_args.append(lbl)

    ff_cmd_args.extend(['-l', label_col])
    ff_cmd_args.extend(['--id_col', id_col])

    if inverse:
        ff_cmd_args.append('-i')

    if quiet:
        ff_cmd_args.append('-q')

    # Substitute mock methods for the main methods that get called by
    # filter_features for arff files: FeatureSet.filter() and the __init__()
    # method of the appropriate writer.  We also need to mock the write()
    # method to prevent actual writing.
    with patch.object(FeatureSet, 'filter', autospec=True) as filter_mock, \
            patch.object(writer_class, '__init__', autospec=True,
                         return_value=None) as write_init_mock, \
            patch.object(writer_class, 'write', autospec=True) as write_mock:

        ff.main(argv=ff_cmd_args)

        # get the various arguments from the three mocked up methods
        filter_pos_arguments, filter_kw_arguments = filter_mock.call_args
        write_pos_arguments, write_kw_arguments = write_init_mock.call_args

        # make sure that the arguments they got were the ones we specified
        eq_(write_pos_arguments[1], outfile)
        eq_(write_kw_arguments['quiet'], quiet)

        # note that we cannot test the label_col column for the writer
        # the reason is that is set conditionally and those conditions
        # do not execute with mocking

        eq_(filter_pos_arguments[0], fs)
        eq_(filter_kw_arguments['inverse'], inverse)

        if filter_type == 'feature':
            eq_(filter_kw_arguments['features'], features_to_keep)
        elif filter_type == 'id':
            eq_(filter_kw_arguments['ids'], ids_to_keep)
        elif filter_type == 'label':
            eq_(filter_kw_arguments['labels'], labels_to_keep)


def test_filter_features_arff_argparse():
    for (filter_type, label_col, id_col,
         inverse, quiet) in product(['feature', 'id',
                                     'label'],
                                    ['y', 'foo'],
                                    ['id', 'id_foo'],
                                    [True, False],
                                    [True, False]):

        yield (check_filter_features_arff_argparse, filter_type,
               label_col, id_col, inverse, quiet)


@raises(SystemExit)
def test_filter_features_libsvm_input_argparse():
    """
    Make sure filter_features exits when passing in input libsvm files
    """

    ff_cmd_args = ['foo.libsvm', 'bar.csv', '-f', 'a', 'b', 'c']
    ff.main(argv=ff_cmd_args)


@raises(SystemExit)
def test_filter_features_libsvm_output_argparse():
    """
    Make sure filter_features exits when passing in output libsvm files
    """

    ff_cmd_args = ['foo.csv', 'bar.libsvm', '-f', 'a', 'b', 'c']
    ff.main(argv=ff_cmd_args)


@raises(SystemExit)
def test_filter_features_unknown_input_format():
    """
    Make sure that filter_features exits when passing in an unknown input file format
    """

    ff_cmd_args = ['foo.xxx', 'bar.csv', '-f', 'a', 'b', 'c']
    ff.main(argv=ff_cmd_args)


@raises(SystemExit)
def test_filter_features_unknown_output_format():
    """
    Make sure that filter_features exits when passing in an unknown input file format
    """
    ff_cmd_args = ['foo.csv', 'bar.xxx', '-f', 'a', 'b', 'c']
    ff.main(argv=ff_cmd_args)


@raises(SystemExit)
def check_filter_features_raises_system_exit(cmd_args):
    """
    Little helper to make test output cleaner for tests that check that
    filter_features exits with the specified arguments.
    """
    ff.main(cmd_args)


def test_filter_features_unmatched_formats():
    # Make sure filter_feature exits when the output file is in a different
    # format
    for inext, outext in combinations(['.arff', '.megam', '.ndj', '.tsv',
                                       '.jsonlines', '.csv'], 2):
        ff_cmd_args = ['foo{}'.format(inext), 'bar{}'.format(outext), '-f',
                       'a', 'b', 'c']
        yield check_filter_features_raises_system_exit, ff_cmd_args


def check_join_features_argparse(extension, label_col='y', id_col='id',
                                 quiet=False):
    """
    A utility function to check that we are setting up argument parsing
    correctly for join_features for ALL file types. We are not checking
    whether the results are correct because we have separate tests for that.
    """

    # replace the run_configuration function that's called
    # by the main() in filter_feature with a mocked up version
    writer_class = EXT_TO_WRITER[extension]

    # create some dummy input and output filenames
    infile1 = join(_my_dir, 'other', 'test_join_features1{}'.format(extension))
    infile2 = join(_my_dir, 'other', 'test_join_features2{}'.format(extension))
    outfile = 'bar{}'.format(extension)

    # create a simple featureset with actual ids, labels and features
    fs1, _ = make_classification_data(num_labels=3,
                                      train_test_ratio=1.0,
                                      random_state=1234)
    fs2, _ = make_classification_data(num_labels=3,
                                      train_test_ratio=1.0,
                                      feature_prefix='g',
                                      random_state=5678)

    jf_cmd_args = [infile1, infile2, outfile]

    if extension in ['.tsv', '.csv', '.arff']:
        writer1 = writer_class(infile1, fs1, label_col=label_col,
                               id_col=id_col)
        writer2 = writer_class(infile2, fs2, label_col=label_col,
                               id_col=id_col)
        jf_cmd_args.extend(['-l', label_col])
        jf_cmd_args.extend(['--id_col', id_col])
    else:
        writer1 = writer_class(infile1, fs1)
        writer2 = writer_class(infile2, fs2)

    writer1.write()
    writer2.write()

    if quiet:
        jf_cmd_args.append('-q')

    # Substitute mock methods for the main methods that get called by
    # filter_features: FeatureSet.filter() and the __init__() method
    # of the appropriate writer. We also need to mock the write()
    # method to prevent actual writing.
    with patch.object(FeatureSet, '__add__', autospec=True) as add_mock, \
            patch.object(writer_class, '__init__', autospec=True,
                         return_value=None) as write_init_mock, \
            patch.object(writer_class, 'write', autospec=True) as write_mock:

        jf.main(argv=jf_cmd_args)

        # get the various arguments from the three mocked up methods
        add_pos_arguments, add_kw_arguments = add_mock.call_args
        write_pos_arguments, write_kw_arguments = write_init_mock.call_args

        # make sure that the arguments they got were the ones we specified
        eq_(write_pos_arguments[1], outfile)
        eq_(write_kw_arguments['quiet'], quiet)

        # note that we cannot test the label_col column for the writer
        # the reason is that is set conditionally and those conditions
        # do not execute with mocking

        eq_(add_pos_arguments[0], fs1)
        eq_(add_pos_arguments[1], fs2)


def test_join_features_argparse():
    for (extension, label_col, id_col, quiet) in product(['.jsonlines', '.ndj',
                                                          '.megam', '.tsv',
                                                          '.csv', '.arff'],
                                                         ['y', 'foo'],
                                                         ['id', 'id_foo'],
                                                         [True, False]):

        yield (check_join_features_argparse, extension, label_col, id_col,
               quiet)


@raises(SystemExit)
def test_join_features_libsvm_input_argparse():
    """
    Make sure that join_features exits when passing in input libsvm files
    """

    jf_cmd_args = ['foo.libsvm', 'bar.libsvm', 'baz.csv']
    jf.main(argv=jf_cmd_args)


@raises(SystemExit)
def test_join_features_libsvm_output_argparse():
    """
    Make sure that join_features exits when passing in output libsvm files
    """

    jf_cmd_args = ['foo.csv', 'bar.csv', 'baz.libsvm']
    jf.main(argv=jf_cmd_args)


@raises(SystemExit)
def test_join_features_unknown_input_format():
    """
    Make that join_features exits when passing in an unknown input file format
    """

    jf_cmd_args = ['foo.xxx', 'bar.tsv', 'baz.csv']
    jf.main(argv=jf_cmd_args)


@raises(SystemExit)
def test_join_features_unknown_output_format():
    """
    Make sure that join_features exits when passing in an unknown output file format
    """

    jf_cmd_args = ['foo.csv', 'bar.csv', 'baz.xxx']
    jf.main(argv=jf_cmd_args)


@raises(SystemExit)
def check_join_features_raises_system_exit(cmd_args):
    """
    Little helper to make test output cleaner for tests that check that
    join_features exits with the specified arguments.
    """
    jf.main(cmd_args)


def test_join_features_unmatched_input_formats():
    # Make sure that join_feature exits when the input files are in different
    # formats
    for ext1, ext2 in combinations(['.arff', '.megam', '.ndj', '.tsv',
                                    '.jsonlines', '.csv'], 2):
        jf_cmd_args = ['foo{}'.format(ext1), 'bar{}'.format(ext2),
                       'baz{}'.format(ext1)]
        yield check_join_features_raises_system_exit, jf_cmd_args


def test_join_features_unmatched_output_format():
    # Make sure join_features exits when the output file is in a different
    # format
    for ext1, ext2 in combinations(['.arff', '.megam', '.ndj', '.tsv',
                                    '.jsonlines', '.csv'], 2):
        jf_cmd_args = ['foo{}'.format(ext1), 'bar{}'.format(ext1),
                       'baz{}'.format(ext2)]
        yield check_join_features_raises_system_exit, jf_cmd_args
