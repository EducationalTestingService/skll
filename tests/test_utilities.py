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

import glob
import os
import re
import subprocess

from io import open
from os.path import abspath, dirname, exists, join

from nose.tools import eq_, assert_almost_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.datasets.samples_generator import (make_classification,
                                                make_regression)
from skll.data import FeatureSet, NDJWriter
from skll.experiments import _setup_config_parser
from skll.learner import Learner, _DEFAULT_PARAM_GRIDS
from skll.utilities.compute_eval_from_predictions \
    import compute_eval_from_predictions

from skll.utilities.generate_predictions import Predictor

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
    os.unlink(join(test_dir, 'test_generate_predictions.jsonlines'))
    for model_chunk in glob.glob(join(output_dir, 'test_generate_predictions.model*')):
        os.unlink(model_chunk)
    for model_chunk in glob.glob(join(output_dir, 'test_generate_predictions_console.model*')):
        os.unlink(model_chunk)


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


def make_classification_data(num_examples=100, train_test_ratio=0.5,
                             num_features=10, use_feature_hashing=False,
                             feature_bins=4, num_redundant=0, num_classes=2,
                             class_weights=None, non_negative=False,
                             random_state=1234567890):

    # use sklearn's make_classification to generate the data for us
    num_informative = num_features - num_redundant
    X, y = make_classification(n_samples=num_examples, n_features=num_features,
                               n_informative=num_informative, n_redundant=num_redundant,
                               n_classes=num_classes, weights=class_weights,
                               random_state=random_state)

    # if we were told to only generate non-negative features, then
    # we can simply take the absolute values of the generated features
    if non_negative:
        X = abs(X)

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, num_examples + 1)]

    # create a list of dictionaries as the features
    feature_names = ['f{}'.format(n) for n in range(1, num_features + 1)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples],
                                     features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # create a FeatureHasher if we are asked to use feature hashing
    # and use 2.5 times the number of features to be on the safe side
    vectorizer = (FeatureHasher(n_features=feature_bins)
                  if use_feature_hashing else None)
    train_fs = FeatureSet('classification_train', train_ids,
                          classes=train_y, features=train_features,
                          vectorizer=vectorizer)
    test_fs = FeatureSet('classification_test', test_ids,
                         classes=test_y, features=test_features,
                         vectorizer=vectorizer)

    return (train_fs, test_fs)


def test_compute_eval_from_predictions():
    pred_path = join(_my_dir, 'other',
                     'test_compute_eval_from_predictions.predictions')
    input_path = join(_my_dir, 'other',
                      'test_compute_eval_from_predictions.jsonlines')

    scores = compute_eval_from_predictions(input_path, pred_path,
                                           ['pearson', 'unweighted_kappa'])

    assert_almost_equal(scores['pearson'], 0.6197797868009122)
    assert_almost_equal(scores['unweighted_kappa'], 0.2)


def check_generate_predictions(use_feature_hashing=False, use_threshold=False):

    # create some simple classification data without feature hashing
    train_fs, test_fs = make_classification_data(num_examples=1000,
                                                 num_features=5,
                                                 use_feature_hashing=use_feature_hashing)

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
    learner.save('output/test_generate_predictions.model')

    # now use Predictor to generate the predictions and make
    # sure that they are the same as before saving the model
    p = Predictor('output/test_generate_predictions.model',
                  threshold=threshold)
    predictions_after_saving = p.predict(test_fs)

    eq_(predictions, predictions_after_saving)


def test_generate_predictions():
    '''
    Test generate predictions combined with hashing and a threshold
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
    input_file = 'test/test_generate_predictions.jsonlines'
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
    model_file = 'output/test_generate_predictions_console.model'
    learner.save(model_file)

    # now call generate_predictions.py on the command line
    generate_cmd = ['generate_predictions']
    if use_threshold:
        generate_cmd.append('-t {}'.format(threshold))
    generate_cmd.extend([model_file, input_file])

    p = subprocess.Popen(generate_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    predictions_after_saving = [int(x) for x in out.decode().strip().split('\n')]

    eq_(predictions, predictions_after_saving)


def test_generate_predictions_console():
    yield check_generate_predictions_console, False
    yield check_generate_predictions_console, True
