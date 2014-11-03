# License: BSD 3 clause
'''
Module for running a bunch of simple unit tests. Should be expanded more in
the future.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import glob
import itertools
import os
import re
import sys

from six import StringIO

from io import open
from os.path import abspath, dirname, exists, join

from nose.tools import eq_, assert_almost_equal, nottest
from numpy.testing import assert_array_equal, assert_allclose

from skll.data import (FeatureSet, NDJWriter, LibSVMWriter,
                       MegaMWriter, LibSVMReader, safe_float)
from skll.data.readers import EXT_TO_READER
from skll.data.writers import EXT_TO_WRITER
from skll.learner import Learner, _DEFAULT_PARAM_GRIDS
import skll.utilities.compute_eval_from_predictions as cefp
import skll.utilities.generate_predictions as gp
import skll.utilities.skll_convert as sk
import skll.utilities.print_model_weights as pmw

from utils import make_classification_data, make_regression_data


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
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')
    other_dir = join(_my_dir, 'other')
    if exists(join(test_dir, 'test_generate_predictions.jsonlines')):
        os.unlink(join(test_dir, 'test_generate_predictions.jsonlines'))
    for model_chunk in glob.glob(join(output_dir, 'test_generate_predictions.model*')):
        os.unlink(model_chunk)
    for model_chunk in glob.glob(join(output_dir, 'test_generate_predictions_console.model*')):
        os.unlink(model_chunk)
    for f in glob.glob(join(other_dir, 'test_skll_convert*')):
        os.unlink(f)


def test_compute_eval_from_predictions():
    '''
    Test compute_eval_from_predictions function console script
    '''

    pred_path = join(_my_dir, 'other',
                     'test_compute_eval_from_predictions.predictions')
    input_path = join(_my_dir, 'other',
                      'test_compute_eval_from_predictions.jsonlines')

    # we need to capture stdout since that's what main() writes to
    compute_eval_from_predictions_cmd = [input_path, pred_path, 'pearson', 'unweighted_kappa']
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
    train_fs, test_fs = make_classification_data(num_examples=1000,
                                                 num_features=5,
                                                 use_feature_hashing=use_feature_hashing,
                                                 feature_bins=4)

    # create a learner that uses an SGD classifier
    learner = Learner('SGDClassifier', probability=use_threshold)

    # train the learner with grid search
    learner.train(train_fs, grid_search=True, feature_hasher=use_feature_hashing)

    # get the predictions on the test featureset
    predictions = learner.predict(test_fs, feature_hasher=use_feature_hashing)

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
    '''
    Test generate predictions API with hashing and a threshold
    '''

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
    '''
    Test generate_predictions as a console script with/without a threshold
    '''

    yield check_generate_predictions_console, False
    yield check_generate_predictions_console, True


def check_skll_convert(from_suffix, to_suffix):

    # create some simple classification data
    orig_fs, _ = make_classification_data(train_test_ratio=1.0, one_string_feature=True)

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
    '''
    Test to check whether the --reuse_libsvm_map option works for skll_convert
    '''

    # create some simple classification data
    orig_fs, _ = make_classification_data(train_test_ratio=1.0, one_string_feature=True)

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
    if task == 'classification':
        train_fs, _ = make_classification_data(train_test_ratio=0.8)
    else:
        train_fs, _, _ = make_regression_data(num_features=4, train_test_ratio=0.8)

    # now train the appropriate model
    if task == 'classification':
        learner = Learner('LogisticRegression')
        learner.train(train_fs)
    else:
        learner = Learner('LinearRegression')
        learner.train(train_fs, grid_objective='pearson')

    # now save the model to disk
    model_file = join(_my_dir, 'output',
                      'test_print_model_weights.model')
    learner.save(model_file)

    # now call print_model_weights main() and capture the output
    print_model_weights_cmd = [model_file]
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
    else:
        lines_to_parse = [l for l in out.split('\n') if l]
        intercept = safe_float(lines_to_parse[0].split('=')[1])
        feature_values = []
        for ltp in lines_to_parse[1:]:
            fields = ltp.split('\t')
            feature_values.append((fields[1], safe_float(fields[0])))
        feature_values = [t[1] for t in sorted(feature_values)]
        assert_almost_equal(intercept, learner.model.intercept_)
        assert_allclose(learner.model.coef_, feature_values)


def test_print_model_weights():
    yield check_print_model_weights, 'classification'
    yield check_print_model_weights, 'regression'
