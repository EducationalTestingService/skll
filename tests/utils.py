import numpy as np
from numpy.random import RandomState

from skll.data import FeatureSet
from sklearn.datasets.samples_generator import (make_classification,
                                                make_regression)
from sklearn.feature_extraction import FeatureHasher


def make_classification_data(num_examples=100, train_test_ratio=0.5,
                             num_features=10, use_feature_hashing=False,
                             feature_bins=4, num_classes=2,
                             empty_classes=False, feature_prefix='f',
                             class_weights=None, non_negative=False,
                             one_string_feature=False, num_string_values=4,
                             random_state=1234567890):

    # use sklearn's make_classification to generate the data for us
    num_numeric_features = num_features -1 if one_string_feature else num_features
    X, y = make_classification(n_samples=num_examples, n_features=num_numeric_features,
                               n_informative=num_numeric_features, n_redundant=0,
                               n_classes=num_classes, weights=class_weights,
                               random_state=random_state)

    # if we were told to only generate non-negative features, then
    # we can simply take the absolute values of the generated features
    if non_negative:
        X = abs(X)

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, num_examples + 1)]

    # create a string feature that has four possible values
    # 'a', 'b', 'c' and 'd' and add it to X at the end
    if one_string_feature:
        prng = RandomState(random_state)
        random_indices = prng.random_integers(0, num_string_values - 1, num_examples)
        possible_values = [chr(x) for x in range(97, 97 + num_string_values)]
        string_feature_values = [possible_values[i] for i in random_indices]
        string_feature_column = np.array(string_feature_values, dtype=object).reshape(100, 1)
        X = np.append(X, string_feature_column, 1)

    # create a list of dictionaries as the features
    feature_names = ['{}{:02d}'.format(feature_prefix, n) for n in range(1, num_features + 1)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples],
                                     features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # are we told to generate empty classes
    train_classes = None if empty_classes else train_y
    test_classes = None if empty_classes else test_y

    # create a FeatureHasher if we are asked to use feature hashing
    # and use 2.5 times the number of features to be on the safe side
    vectorizer = (FeatureHasher(n_features=feature_bins)
                  if use_feature_hashing else None)
    train_fs = FeatureSet('classification_train', train_ids,
                          classes=train_classes, features=train_features,
                          vectorizer=vectorizer)
    if train_test_ratio < 1.0:
        test_fs = FeatureSet('classification_test', test_ids,
                             classes=test_classes, features=test_features,
                             vectorizer=vectorizer)
    else:
        test_fs = None

    return (train_fs, test_fs)


def make_regression_data(num_examples=100, train_test_ratio=0.5,
                         num_features=2, sd_noise=1.0,
                         use_feature_hashing=False,
                         feature_bins=4,
                         start_feature_num=1,
                         random_state=1234567890):

    # use sklearn's make_regression to generate the data for us
    X, y, weights = make_regression(n_samples=num_examples,
                                    n_features=num_features,
                                    noise=sd_noise, random_state=random_state,
                                    coef=True)

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = ['EXAMPLE_{}'.format(n) for n in range(1, num_examples + 1)]

    # create a list of dictionaries as the features
    feature_names = ['f{:02d}'.format(n) for n
                     in range(start_feature_num,
                              start_feature_num + num_features)]
    features = []
    for row in X:
        features.append(dict(zip(feature_names, row)))

    # convert the weights array into a dictionary for convenience
    weightdict = dict(zip(feature_names, weights))

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples],
                                     features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # create a FeatureHasher if we are asked to use feature hashing
    # and use 2.5 times the number of features to be on the safe side
    vectorizer = (FeatureHasher(n_features=feature_bins) if
                  use_feature_hashing else None)
    train_fs = FeatureSet('regression_train', train_ids,
                          classes=train_y, features=train_features,
                          vectorizer=vectorizer)
    test_fs = FeatureSet('regression_test', test_ids,
                         classes=test_y, features=test_features,
                         vectorizer=vectorizer)

    return (train_fs, test_fs, weightdict)


