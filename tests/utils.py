"""
Utilities functions to make SKLL testing simpler
"""

import re

from collections import OrderedDict
from math import floor, log10
from os.path import abspath, dirname, exists, join

import numpy as np
from numpy.random import RandomState
from sklearn.datasets.samples_generator import (make_classification,
                                                make_regression)
from sklearn.feature_extraction import FeatureHasher

from skll.data import FeatureSet, NDJWriter
from skll.config import _setup_config_parser

_my_dir = abspath(dirname(__file__))


def fill_in_config_paths(config_template_path):
    """
    Add paths to train, test, and output directories to a given config template
    file.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path, validate=False)

    task = config.get("General", "task")

    config.set("Input", "train_directory", train_dir)

    to_fill_in = ['log']

    if task != 'learning_curve':
        to_fill_in.append('predictions')

    if task not in ['cross_validate', 'learning_curve']:
        to_fill_in.append('models')

    if task in ['cross_validate', 'evaluate', 'learning_curve', 'train']:
        to_fill_in.append('results')

    for d in to_fill_in:
        config.set("Output", d, join(output_dir))

    if task == 'cross_validate':
        folds_file = config.get("Input", "folds_file")
        if folds_file:
            config.set("Input", "folds_file", join(train_dir, folds_file))

    if task == 'predict' or task == 'evaluate':
        config.set("Input", "test_directory", test_dir)

    # set up custom learner path, if relevant
    custom_learner_path = config.get("Input", "custom_learner_path")
    custom_learner_abs_path = join(_my_dir, custom_learner_path)
    config.set("Input", "custom_learner_path", custom_learner_abs_path)

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}.cfg'.format(config_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def fill_in_config_paths_for_single_file(config_template_path, train_file,
                                         test_file, train_directory='',
                                         test_directory=''):
    """
    Add paths to train and test files, and output directories to a given config
    template file.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path, validate=False)

    task = config.get("General", "task")

    config.set("Input", "train_file", join(train_dir, train_file))
    if task == 'predict' or task == 'evaluate':
        config.set("Input", "test_file", join(test_dir, test_file))

    if train_directory:
        config.set("Input", "train_directory", join(train_dir, train_directory))

    if test_directory:
        config.set("Input", "test_directory", join(test_dir, test_directory))

    to_fill_in = ['log']
    if task != 'train':
        to_fill_in.append('predictions')

    if task == 'cross_validate':
        if config.get("Output", "save_cv_models"):
            to_fill_in.append('models')
    else:
        to_fill_in.append('models')

    if task == 'evaluate' or task == 'cross_validate':
        to_fill_in.append('results')

    for d in to_fill_in:
        config.set("Output", d, join(output_dir))

    if task == 'cross_validate':
        folds_file = config.get("Input", "folds_file")
        if folds_file:
            config.set("Input", "folds_file",
                       join(train_dir, folds_file))

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}.cfg'.format(config_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def fill_in_config_options(config_template_path,
                           values_to_fill_dict,
                           sub_prefix,
                           good_probability_option=False):
    """
    Fill in values in the given config template
    """

    config = _setup_config_parser(config_template_path, validate=False)

    # The dictionary that says which option to fill in where
    # Note: (a) `bad_option` and `duplicate_option` are needed
    # for test_config_parsing_invalid_option and
    # test_config_parsing_duplicate_option() respectively.
    # (b) `probability` is deliberately specified in the wrong
    # section for test_config_parsing_option_in_wrong_section().
    to_fill_in = {'General': ['experiment_name', 'task'],
                  'Input': ['train_directory', 'train_file', 'test_directory',
                            'test_file', 'featuresets', 'featureset_names',
                            'feature_hasher', 'hasher_features', 'learners',
                            'sampler', 'shuffle', 'feature_scaling',
                            'learning_curve_cv_folds_list', 'folds_file',
                            'learning_curve_train_sizes', 'fixed_parameters',
                            'num_cv_folds', 'bad_option', 'duplicate_option',
                            'suffix'],
                  'Tuning': ['grid_search', 'objective',
                             'use_folds_file_for_grid_search', 'grid_search_folds',
                             'pos_label_str', 'param_grids', 'objectives',
                             'duplicate_option'],
                  'Output': ['results', 'log', 'models', 'metrics',
                             'predictions', 'pipeline', 'save_cv_folds',
                             'save_cv_models']}

    if good_probability_option:
        to_fill_in['Output'].append('probability')
    else:
        to_fill_in['Tuning'].append('probability')

    for section in to_fill_in:
        for param_name in to_fill_in[section]:
            if param_name in values_to_fill_dict:
                config.set(section, param_name,
                           values_to_fill_dict[param_name])

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}_{}.cfg'.format(config_prefix, sub_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def fill_in_config_paths_for_fancy_output(config_template_path):
    """
    Add paths to train, test, and output directories to a given config template
    file.
    """

    train_dir = join(_my_dir, 'train')
    test_dir = join(_my_dir, 'test')
    output_dir = join(_my_dir, 'output')

    config = _setup_config_parser(config_template_path, validate=False)

    config.set("Input", "train_file", join(train_dir, "fancy_train.jsonlines"))
    config.set("Input", "test_file", join(test_dir,
                                          "fancy_test.jsonlines"))
    config.set("Output", "results", output_dir)
    config.set("Output", "log", output_dir)
    config.set("Output", "predictions", output_dir)

    config_prefix = re.search(r'^(.*)\.template\.cfg',
                              config_template_path).groups()[0]
    new_config_path = '{}.cfg'.format(config_prefix)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    return new_config_path


def create_jsonlines_feature_files(path):

    # we only need to create the feature files if they
    # don't already exist under the given path
    feature_files_to_create = [join(path, 'f{}.jsonlines'.format(i)) for i in range(6)]
    if all([exists(ff) for ff in feature_files_to_create]):
        return
    else:
        num_examples = 1000
        np.random.seed(1234567890)

        # Create lists we will write files from
        ids = []
        features = []
        labels = []
        for j in range(num_examples):
            y = "dog" if j % 2 == 0 else "cat"
            ex_id = "{}{}".format(y, j)
            x = {"f{}".format(feat_num): np.random.randint(0, 4) for feat_num in
                 range(5)}
            x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
            ids.append(ex_id)
            labels.append(y)
            features.append(x)

        for i in range(5):
            file_path = join(path, 'f{}.jsonlines'.format(i))
            sub_features = []
            for example_num in range(num_examples):
                feat_num = i
                x = {"f{}".format(feat_num):
                     features[example_num]["f{}".format(feat_num)]}
                sub_features.append(x)
            fs = FeatureSet('ablation_cv', ids, features=sub_features, labels=labels)

            writer = NDJWriter(file_path, fs)
            writer.write()

        # now write out the last file which is basically
        # identical to the last featureset we wrote
        # except that it has two extra instances
        fs = FeatureSet('extra',
                        ids + ['cat{}'.format(num_examples),
                               'dog{}'.format(num_examples + 1)],
                        features=sub_features + [{}, {}],
                        labels=labels + ['cat', 'dog'])
        file_path = join(path, 'f5.jsonlines')
        writer = NDJWriter(file_path, fs)
        writer.write()


def make_classification_data(num_examples=100, train_test_ratio=0.5,
                             num_features=10, use_feature_hashing=False,
                             feature_bins=4, num_labels=2,
                             empty_labels=False, string_label_list=None,
                             feature_prefix='f', id_type='string',
                             class_weights=None, non_negative=False,
                             one_string_feature=False, num_string_values=4,
                             random_state=1234567890):

    # use sklearn's make_classification to generate the data for us
    num_numeric_features = (num_features - 1 if one_string_feature else
                            num_features)
    X, y = make_classification(n_samples=num_examples,
                               n_features=num_numeric_features,
                               n_informative=num_numeric_features,
                               n_redundant=0, n_classes=num_labels,
                               weights=class_weights,
                               random_state=random_state)

    if string_label_list:
        assert(len(string_label_list) == num_labels)
        label_to_string = np.vectorize(lambda n: string_label_list[n])
        y = label_to_string(y)

    # if we were told to only generate non-negative features, then
    # we can simply take the absolute values of the generated features
    if non_negative:
        X = abs(X)

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs; we create IDs that either can also
    # be numbers or pure strings
    if id_type == 'string':
        ids = ['EXAMPLE_{}'.format(n) for n in range(1, num_examples + 1)]
    elif id_type == 'integer_string':
        ids = ['{}'.format(n) for n in range(1, num_examples + 1)]
    elif id_type == 'float':
        ids = [float(n) for n in range(1, num_examples + 1)]
    elif id_type == 'integer':
        ids = list(range(1, num_examples + 1))

    # create a string feature that has four possible values
    # 'a', 'b', 'c' and 'd' and add it to X at the end
    if one_string_feature:
        prng = RandomState(random_state)
        random_indices = prng.random_integers(0, num_string_values - 1,
                                              num_examples)
        possible_values = [chr(x) for x in range(97, 97 + num_string_values)]
        string_feature_values = [possible_values[i] for i in random_indices]
        string_feature_column = np.array(string_feature_values,
                                         dtype=object).reshape(100, 1)
        X = np.append(X, string_feature_column, 1)

    # create a list of dictionaries as the features
    feature_names = ['{}{:02d}'.format(feature_prefix, n) for n in
                     range(1, num_features + 1)]
    features = [dict(zip(feature_names, row)) for row in X]

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples],
                                     features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # are we told to generate empty labels
    train_labels = None if empty_labels else train_y
    test_labels = None if empty_labels else test_y

    # create a FeatureHasher if we are asked to use feature hashing
    # with the specified number of feature bins
    vectorizer = (FeatureHasher(n_features=feature_bins)
                  if use_feature_hashing else None)
    train_fs = FeatureSet('classification_train', train_ids,
                          labels=train_labels, features=train_features,
                          vectorizer=vectorizer)
    if train_test_ratio < 1.0:
        test_fs = FeatureSet('classification_test', test_ids,
                             labels=test_labels, features=test_features,
                             vectorizer=vectorizer)
    else:
        test_fs = None

    return (train_fs, test_fs)


def make_regression_data(num_examples=100,
                         train_test_ratio=0.5,
                         num_features=2,
                         sd_noise=1.0,
                         use_feature_hashing=False,
                         feature_bins=4,
                         start_feature_num=1,
                         random_state=1234567890):

    # if we are doing feature hashing and we have asked for more
    # feature bins than number of total features, we need to
    # handle that because `make_regression()` doesn't know
    # about hashing
    if use_feature_hashing and num_features < feature_bins:
        num_features = feature_bins

    # use sklearn's make_regression to generate the data for us
    X, y, weights = make_regression(n_samples=num_examples,
                                    n_features=num_features,
                                    noise=sd_noise,
                                    random_state=random_state,
                                    coef=True)

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, num_examples + 1)]

    # create a list of dictionaries as the features
    index_width_for_feature_name = int(floor(log10(num_features))) + 1
    feature_names = []
    for n in range(start_feature_num, start_feature_num + num_features):
        index_str = str(n).zfill(index_width_for_feature_name)
        feature_name = 'f{}'.format(index_str)
        feature_names.append(feature_name)
    features = [dict(zip(feature_names, row)) for row in X]

    # At this point the labels are generated using unhashed features
    # even if we want to do feature hashing. `make_regression()` from
    # sklearn doesn't know anything about feature hashing, so we need
    # a hack here to compute the updated labels ourselves
    # using the same command that sklearn uses inside `make_regression()`
    # which is to generate the X and the weights and then compute the
    # y as the dot product of the two. This y will then be used as our
    # labels instead of the original y we got from `make_regression()`.
    # Note that we only want to use the number of weights that are
    # equal to the number of feature bins for the hashing
    if use_feature_hashing:
        feature_hasher = FeatureHasher(n_features=feature_bins)
        hashed_X = feature_hasher.fit_transform(features)
        y = hashed_X.dot(weights[:feature_bins])

    # convert the weights array into a dictionary for convenience
    # if we are using feature hashing, we need to use the names
    # that would be output by `model_params()` instead of the
    # original names since that's what we would get from SKLL
    if use_feature_hashing:
        index_width_for_feature_name = int(floor(log10(feature_bins))) + 1
        hashed_feature_names = []
        for i in range(feature_bins):
            index_str = str(i + 1).zfill(index_width_for_feature_name)
            feature_name = 'hashed_feature_{}'.format(index_str)
            hashed_feature_names.append(feature_name)
        weightdict = dict(zip(hashed_feature_names, weights[:feature_bins]))
    else:
        weightdict = dict(zip(feature_names, weights))

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples],
                                     features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # create a FeatureHasher if we are asked to use feature hashing
    # with the specified number of feature bins
    vectorizer = (FeatureHasher(n_features=feature_bins) if
                  use_feature_hashing else None)
    train_fs = FeatureSet('regression_train', train_ids,
                          labels=train_y, features=train_features,
                          vectorizer=vectorizer)
    test_fs = FeatureSet('regression_test', test_ids,
                         labels=test_y, features=test_features,
                         vectorizer=vectorizer)

    return (train_fs, test_fs, weightdict)


def make_sparse_data(use_feature_hashing=False):
    """
    Function to create sparse data with two features always zero
    in the training set and a different one always zero in the
    test set
    """
    # Create training data
    X, y = make_classification(n_samples=500, n_features=3,
                               n_informative=3, n_redundant=0,
                               n_classes=2, random_state=1234567890)

    # we need features to be non-negative since we will be
    # using naive bayes laster
    X = np.abs(X)

    # make sure that none of the features are zero
    X[np.where(X == 0)] += 1

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 501)]

    # create a list of dictionaries as the features
    # with f1 and f5 always 0
    feature_names = ['f{}'.format(n) for n in range(1, 6)]
    features = []
    for row in X:
        row = [0] + row.tolist() + [0]
        features.append(dict(zip(feature_names, row)))

    # use a FeatureHasher if we are asked to do feature hashing
    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    train_fs = FeatureSet('train_sparse', ids,
                          features=features, labels=y,
                          vectorizer=vectorizer)

    # now create the test set with f4 always 0 but nothing else
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=4, n_redundant=0,
                               n_classes=2, random_state=1234567890)
    X = np.abs(X)
    X[np.where(X == 0)] += 1
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, 101)]

    # create a list of dictionaries as the features
    # with f4 always 0
    feature_names = ['f{}'.format(n) for n in range(1, 6)]
    features = []
    for row in X:
        row = row.tolist()
        row = row[:3] + [0] + row[3:]
        features.append(dict(zip(feature_names, row)))

    test_fs = FeatureSet('test_sparse', ids,
                         features=features, labels=y,
                         vectorizer=vectorizer)

    return train_fs, test_fs
