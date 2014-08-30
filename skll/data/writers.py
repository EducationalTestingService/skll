# License: BSD 3 clause
'''
Handles loading data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
'''

from __future__ import absolute_import, print_function, unicode_literals

import csv
import json
import logging
import os
import re
from csv import DictWriter
from decimal import Decimal
from io import open

import numpy as np
from six import iteritems, PY2, PY3, string_types, text_type
from six.moves import map, zip
from sklearn.feature_extraction import DictVectorizer

# Import QueueHandler and QueueListener for multiprocess-safe logging
if PY2:
    from logutils.queue import QueueHandler, QueueListener
else:
    from logging.handlers import QueueHandler, QueueListener


# Constants
LIBSVM_REPLACE_DICT = {':': '\u2236',
                       '#': '\uFF03',
                       ' ': '\u2002',
                       '=': '\ua78a',
                       '|': '\u2223'}


def _replace_non_ascii(line):
    '''
    :param line: The line to clean up.
    :type line: str

    :returns: Copy of line with all non-ASCII characters replaced with
    <U1234> sequences where 1234 is the value of ord() for the character.
    '''
    char_list = []
    for char in line:
        char_num = ord(char)
        char_list.append('<U{}>'.format(char_num) if char_num > 127 else char)
    return ''.join(char_list)


def _examples_to_dictwriter_inputs(ids, classes, features, label_col):
    '''
    Takes lists of IDs, classes, and features and converts them to a set of
    fields and a list of instance dictionaries to use as input for DictWriter.
    '''
    instances = []
    fields = set()
    # Iterate through examples
    for ex_id, class_name, feature_dict in zip(ids, classes, features):
        # Don't try to add class column if this is label-less data
        if label_col not in feature_dict:
            if class_name is not None:
                feature_dict[label_col] = class_name
        else:
            raise ValueError(('Class column name "{0}" already used ' +
                              'as feature name!').format(label_col))

        if 'id' not in feature_dict:
            if ex_id is not None:
                feature_dict['id'] = ex_id
        else:
            raise ValueError('ID column name "id" already used as ' +
                             'feature name!')
        fields.update(feature_dict.keys())
        instances.append(feature_dict)
    return fields, instances


def _write_arff_file(path, ids, classes, features, label_col,
                     relation='skll_relation', arff_regression=False):
    '''
    Writes a feature file in .csv or .tsv format with the given a list of IDs,
    classes, and features.

    :param path: A path to the feature file we would like to create.
    :type path: str
    :param ids: The IDs for each instance in the feature list/array. If None,
                IDs will be automatically generated with the prefix specified
                by `id_prefix` followed by the row number. If `id_prefix` is
                also None, no IDs will be written to the file.
    :type ids: list of str
    :param classes: The class labels for each instance in the feature
                    list/array. If None, no class labels will be added to
                    output file.
    :type classes: list of str
    :param features: The features for each instance.
    :type features: list of dict
    :param label_col: The column which should contain the label.
    :type label_col: str
    :param relation: The name to give the relationship represented in this
                     featureset.
    :type relation: str
    :param arff_regression: Make the class variable numeric instead of nominal
                            and remove all non-numeric attributes.
    :type relation: bool
    '''
    # Convert IDs, classes, and features into format for DictWriter
    fields, instances = _examples_to_dictwriter_inputs(ids, classes, features,
                                                       label_col)
    if label_col in fields:
        fields.remove(label_col)

    # a list to keep track of any non-numeric fields
    non_numeric_fields = []

    # Write file
    if PY3:
        file_mode = 'w'
    else:
        file_mode = 'wb'
    with open(path, file_mode) as f:
        # Add relation to header
        print("@relation '{}'\n".format(relation), file=f)

        # Loop through fields writing the header info for the ARFF file
        sorted_fields = sorted(fields)
        for field in sorted_fields:
            # Check field type
            numeric = True
            for instance in instances:
                if field in instance:
                    numeric = not isinstance(instance[field], string_types)
                    break

            # ignore non-numeric fields if we are dealing with regression
            # but keep track of them for later use
            if arff_regression and not numeric:
                non_numeric_fields.append(field)
                continue

            print("@attribute '{}'".format(field.replace('\\', '\\\\')
                                                .replace("'", "\\'")),
                  end=" ", file=f)
            print("numeric" if numeric else "string", file=f)

        # Print class label header if necessary
        if arff_regression:
            print("@attribute {} numeric".format(label_col), file=f)
        else:
            if set(classes) != {None}:
                print("@attribute {} ".format(label_col) +
                      "{" + ','.join(map(str, sorted(set(classes)))) + "}",
                      file=f)
        sorted_fields.append(label_col)

        # throw out any non-numeric fields if we are writing for regression
        if arff_regression:
            sorted_fields = [fld for fld in sorted_fields if fld not in
                             non_numeric_fields]

        # Create CSV writer to handle missing values for lines in data section
        # and to ignore the instance values for non-numeric attributes
        writer = csv.DictWriter(f, sorted_fields, restval=0,
                                extrasaction='ignore', dialect='arff')

        # Output instances
        print("\n@data", file=f)
        writer.writerows(instances)


def _write_delimited_file(path, ids, classes, features, label_col, dialect):
    '''
    Writes a feature file in .csv or .tsv format with the given a list of IDs,
    classes, and features.

    :param path: A path to the feature file we would like to create.
    :type path: str
    :param ids: The IDs for each instance in the feature list/array. If None,
                IDs will be automatically generated with the prefix specified
                by `id_prefix` followed by the row number. If `id_prefix` is
                also None, no IDs will be written to the file.
    :type ids: list of str
    :param classes: The class labels for each instance in the feature
                    list/array. If None, no class labels will be added to
                    output file.
    :type classes: list of str
    :param features: The features for each instance.
    :type features: list of dict
    :param label_col: The column which should contain the label.
    :type label_col: str
    :param dialect: The CSV dialect to write the output file in. Should be
                    'excel-tab' for .tsv and 'excel' for .csv.
    :type dialect: str
    '''

    # Convert IDs, classes, and features into format for DictWriter
    fields, instances = _examples_to_dictwriter_inputs(ids, classes, features,
                                                       label_col)

    # Write file
    if PY3:
        file_mode = 'w'
    else:
        file_mode = 'wb'
    with open(path, file_mode) as f:
        # Create writer
        writer = DictWriter(f, fieldnames=sorted(fields), restval=0,
                            dialect=dialect)
        # Output instance
        writer.writeheader()
        writer.writerows(instances)


def _write_jsonlines_file(path, ids, classes, features):
    '''
    Writes a feature file in .jsonlines/.ndj format with the given a list of
    IDs, classes, and features.

    :param path: A path to the feature file we would like to create.
    :type path: str
    :param ids: The IDs for each instance in the feature list/array. If None,
                IDs will be automatically generated with the prefix specified
                by `id_prefix` followed by the row number. If `id_prefix` is
                also None, no IDs will be written to the file.
    :type ids: list of str
    :param classes: The class labels for each instance in the feature
                    list/array. If None, no class labels will be added to
                    output file.
    :type classes: list of str
    :param features: The features for each instance.
    :type features: list of dict
    '''
    file_mode = 'w' if PY3 else 'wb'
    with open(path, file_mode) as f:
        # Iterate through examples
        for ex_id, class_name, feature_dict in zip(ids, classes, features):
            # Don't try to add class column if this is label-less data
            example_dict = {}
            if class_name is not None:
                example_dict['y'] = class_name
            if ex_id is not None:
                example_dict['id'] = ex_id
            example_dict["x"] = feature_dict
            print(json.dumps(example_dict, sort_keys=True), file=f)


def _sanitize_libsvm_name(name):
    '''
    Replace illegal characters in class names with close unicode equivalents
    to make things loadable in by LibSVM or LibLinear.
    '''
    if isinstance(name, string_types):
        for orig, replace in LIBSVM_REPLACE_DICT.items():
            name = name.replace(orig, replace)
    return name


def _write_libsvm_file(path, ids, classes, features, feat_vectorizer,
                       label_map):
    '''
    Writes a feature file in .libsvm format with the given a list of IDs,
    classes, and features.

    :param path: A path to the feature file we would like to create.
    :type path: str
    :param ids: The IDs for each instance in the feature list/array. If None,
                IDs will be automatically generated with the prefix specified
                by `id_prefix` followed by the row number. If `id_prefix` is
                also None, no IDs will be written to the file.
    :type ids: list of str
    :param classes: The class labels for each instance in the feature
                    list/array. If None, no class labels will be added to
                    output file.
    :type classes: list of str
    :param features: The features for each instance.
    :type features: list of dict
    :param feat_vectorizer: A `DictVectorizer` to map to/from feature columns
                            indices and names.
    :type feat_vectorizer: DictVectorizer
    :param label_map: Mapping from class name to numbers to use for writing
                      LibSVM files.
    :type label_map: dict (str -> int)
    '''
    with open(path, 'w') as f:
        # Iterate through examples
        for ex_id, class_name, feature_dict in zip(ids, classes, features):
            field_values = [(feat_vectorizer.vocabulary_[field] + 1, value) for
                            field, value in iteritems(feature_dict)
                            if Decimal(value) != 0]
            field_values.sort()
            # Print label
            if class_name in label_map:
                print('{}'.format(label_map[class_name]), end=' ',
                      file=f)
            else:
                print('{}'.format(class_name), end=' ', file=f)
            # Print features
            print(' '.join(('{}:{}'.format(field, value) for field, value in
                            field_values)), end=' ', file=f)
            # Print comment with id and mappings
            print('#', end=' ', file=f)
            if ex_id is not None:
                print(_sanitize_libsvm_name('{}'.format(ex_id)), end='',
                      file=f)
            print(' |', end=' ', file=f)
            if PY2 and class_name is not None and isinstance(class_name,
                                                             text_type):
                class_name = class_name.encode('utf-8')
            if class_name in label_map:
                print('%s=%s' % (_sanitize_libsvm_name(label_map[class_name]),
                                 _sanitize_libsvm_name(class_name)),
                      end=' | ', file=f)
            else:
                print(' |', end=' ', file=f)
            line = ' '.join(('%s=%s' % (feat_vectorizer.vocabulary_[field] + 1,
                                        _sanitize_libsvm_name(field)) for
                             field, value in feature_dict.items() if
                             Decimal(value) != 0))
            print(line, file=f)


def _write_megam_file(path, ids, classes, features):
    '''
    Writes a feature file in .megam format with the given a list of IDs,
    classes, and features.

    :param path: A path to the feature file we would like to create.
    :type path: str
    :param ids: The IDs for each instance in the feature list/array. If None,
                IDs will be automatically generated with the prefix specified
                by `id_prefix` followed by the row number. If `id_prefix` is
                also None, no IDs will be written to the file.
    :type ids: list of str
    :param classes: The class labels for each instance in the feature
                    list/array. If None, no class labels will be added to
                    output file.
    :type classes: list of str
    :param features: The features for each instance.
    :type features: list of dict
    '''
    with open(path, 'w') as f:
        # Iterate through examples
        for ex_id, class_name, feature_dict in zip(ids, classes, features):
            # Don't try to add class column if this is label-less data
            if ex_id is not None:
                print('# {}'.format(ex_id), file=f)
            if class_name is not None:
                print(class_name, end='\t', file=f)
            print(_replace_non_ascii(' '.join(('{} {}'.format(field, value) for
                                               field, value in
                                               sorted(feature_dict.items()) if
                                               Decimal(value) != 0))),
                  file=f)


def _write_sub_feature_file(path, ids, classes, features, filter_features,
                            label_col='y', arff_regression=False,
                            arff_relation='skll_relation',
                            feat_vectorizer=None, label_map=None):
    '''
    Writes a feature file in either ``.arff``, ``.csv``, ``.jsonlines``,
    ``.libsvm``, ``.megam``, ``.ndj``, or ``.tsv`` formats with the given a
    list of IDs, classes, and features.

    :param path: A path to the feature file we would like to create. The suffix
                 to this filename must be ``.arff``, ``.csv``, ``.jsonlines``,
                 ``.libsvm``, ``.megam``, ``.ndj``, or ``.tsv``
    :type path: str
    :param ids: The IDs for each instance in the feature list/array. If None,
                IDs will be automatically generated with the prefix specified by
                `id_prefix` followed by the row number. If `id_prefix` is also
                None, no IDs will be written to the file.
    :type ids: list of str
    :param classes: The class labels for each instance in the feature
                    list/array. If None, no class labels will be added to output
                    file.
    :type classes: list of str
    :param features: The features for each instance represented as either a list
                     of dictionaries.
    :type features: list of dict
    :param filter_features: Set of features to include in current feature file.
    :type filter_features: set of str
    :param label_col: Name of the column which contains the class labels for
                      CSV/TSV files. If no column with that name exists, or
                      `None` is specified, the data is considered to be
                      unlabelled.
    :type label_col: str
    :param arff_regression: A boolean value indicating whether the ARFF files
                            that are written should be written for regression
                            rather than classification, i.e., the class
                            variable y is numerical rather than an enumeration
                            of classes and all non-numeric attributes are
                            removed.
    :type arff_regression: bool
    :param arff_relation: Relation name for ARFF file.
    :type arff_relation: str
    :param feat_vectorizer: A `DictVectorizer` to map to/from feature columns
                            indices and names.
    :type feat_vectorizer: DictVectorizer
    :param label_map: Mapping from class name to numbers to use for writing
                      LibSVM files.
    :type label_map: dict (str -> int)
    '''
    # Setup logger
    logger = logging.getLogger(__name__)

    # Get lowercase extension for file extension checking
    ext = os.path.splitext(path)[1].lower()

    # Filter feature dictionaries if asked
    if filter_features:
        # Print pre-filtered list of feature names if debugging
        if logger.getEffectiveLevel() == logging.DEBUG:
            feat_names = set()
            for feat_dict in features:
                feat_names.update(feat_dict.keys())
            feat_names = sorted(feat_names)
            logger.debug('Original features: %s', feat_names)

        # Apply filtering
        features = [{feat_name: feat_value for feat_name, feat_value in
                     iteritems(feat_dict) if (feat_name in filter_features or
                                              feat_name.split('=', 1)[0] in
                                              filter_features)} for feat_dict
                    in features]

        # Print list post-filtering
        if logger.getEffectiveLevel() == logging.DEBUG:
            feat_names = set()
            for feat_dict in features:
                feat_names.update(feat_dict.keys())
            feat_names = sorted(feat_names)
            logger.debug('Filtered features: %s', feat_names)

    # Create TSV file if asked
    if ext == ".tsv":
        _write_delimited_file(path, ids, classes, features, label_col,
                              'excel-tab')
    # Create CSV file if asked
    elif ext == ".csv":
        _write_delimited_file(path, ids, classes, features, label_col,
                              'excel')
    # Create .jsonlines file if asked
    elif ext == ".jsonlines" or ext == '.ndj':
        _write_jsonlines_file(path, ids, classes, features)
    # Create .libsvm file if asked
    elif ext == ".libsvm":
        _write_libsvm_file(path, ids, classes, features, feat_vectorizer,
                           label_map)
    # Create .megam file if asked
    elif ext == ".megam":
        _write_megam_file(path, ids, classes, features)
    # Create ARFF file if asked
    elif ext == ".arff":
        _write_arff_file(path, ids, classes, features, label_col,
                         arff_regression=arff_regression,
                         relation=arff_relation)
    # Invalid file suffix, raise error
    else:
        raise ValueError(('Output file must be in either .arff, .csv, '
                          '.jsonlines, .libsvm, .megam, .ndj, or .tsv format. '
                          'You specified: {}').format(path))


def write_feature_file(path, ids, classes, features, feat_vectorizer=None,
                       id_prefix='EXAMPLE_', label_col='y',
                       arff_regression=False, arff_relation='skll_relation',
                       subsets=None, label_map=None):
    '''
    Writes a feature file in either ``.arff``, ``.csv``, ``.jsonlines``,
    ``.megam``, ``.ndj``, or ``.tsv`` formats with the given a list of IDs,
    classes, and features.

    :param path: A path to the feature file we would like to create. The suffix
                 to this filename must be ``.arff``, ``.csv``, ``.jsonlines``,
                 ``.libsvm``, ``.megam``, ``.ndj``, or ``.tsv``. If ``subsets``
                 is not ``None``, this is assumed to be a string containing the
                 path to the directory to write the feature files with an
                 additional file extension specifying the file type. For
                 example ``/foo/.csv``.
    :type path: str
    :param ids: The IDs for each instance in the feature list/array. If None,
                IDs will be automatically generated with the prefix specified
                by `id_prefix` followed by the row number. If `id_prefix` is
                also None, no IDs will be written to the file.
    :type ids: list of str
    :param classes: The class labels for each instance in the feature
                    list/array. If None, no class labels will be added to
                    output file.
    :type classes: list of str
    :param features: The features for each instance represented as either a
                     list of dictionaries or an array-like (if
                     `feat_vectorizer` is also specified).
    :type features: list of dict or array-like
    :param feat_vectorizer: A `DictVectorizer` to map to/from feature columns
                            indices and names.
    :type feat_vectorizer: DictVectorizer
    :param id_prefix: If we need to automatically generate IDs, put this prefix
                      infront of the numerical IDs.
    :type id_prefix: str
    :param label_col: Name of the column which contains the class labels for
                      CSV/TSV files. If no column with that name exists, or
                      `None` is specified, the data is considered to be
                      unlabelled.
    :type label_col: str
    :param arff_regression: A boolean value indicating whether the ARFF files
                            that are written should be written for regression
                            rather than classification, i.e., the class
                            variable y is numerical rather than an enumeration
                            of classes and all non-numeric attributes are
                            removed.
    :type arff_regression: bool
    :param arff_relation: Relation name for ARFF file.
    :type arff_relation: str
    :param subsets: A mapping from subset names to lists of feature names that
                    are included in those sets. If given, a feature file will
                    be written for every subset (with the name containing the
                    subset name as suffix to ``path``). Note, since string-
                    valued features are automatically converted into boolean
                    features with names of the form
                    ``FEATURE_NAME=STRING_VALUE``, when doing the filtering,
                    the portion before the ``=`` is all that's used for
                    matching. Therefore, you do not need to enumerate all of
                    these boolean feature names in your mapping.
    :type subsets: dict (str to list of str)
    :param label_map: Mapping from class name to numbers to use for writing
                      LibSVM files.
    :type label_map: dict (str -> int)
    '''
    # Setup logger
    logger = logging.getLogger(__name__)

    logger.debug('Feature vectorizer: %s', feat_vectorizer)
    logger.debug('Features: %s', features)

    # Convert feature array to list of dicts if given a feat vectorizer,
    # otherwise fail.  Only necessary if we were given an array.
    if isinstance(features, np.ndarray):
        if feat_vectorizer is None:
            raise ValueError('If `feat_vectorizer` is unspecified, you must '
                             'pass a list of dicts for `features`.')
        # Convert features to list of dicts if given an array-like & vectorizer
        else:
            features = feat_vectorizer.inverse_transform(features)

    # Create missing vectorizers if writing libsvm
    if os.path.splitext(path)[1].lower() == '.libsvm':
        if label_map is None:
            label_map = {}
            if classes is not None:
                label_map = {label: num for num, label in
                             enumerate(sorted({label for label in classes if
                                               not isinstance(label,
                                                              (int, float))}))}
            # Add fake item to vectorizer for None
            label_map[None] = '00000'
        # Create feature vectorizer if unspecified and writing libsvm
        if feat_vectorizer is None or not feat_vectorizer.vocabulary_:
            feat_vectorizer = DictVectorizer(sparse=True)
            feat_vectorizer.fit(features)
    else:
        label_map = None

    # Create ID generator if necessary
    if ids is None:
        if id_prefix is not None:
            ids = ('{}{}'.format(id_prefix, num) for num in
                   range(len(features)))
        else:
            ids = (None for _ in range(len(features)))

    # Create class generator if necessary
    if classes is None:
        classes = (None for _ in range(len(features)))

    # Get prefix and extension for checking file types and writing subset files
    root, ext = re.search(r'^(.*)(\.[^.]*)$', path).groups()

    # Write one feature file if we weren't given a dict of subsets
    if subsets is None:
        _write_sub_feature_file(path, ids, classes, features, [],
                                label_col=label_col,
                                arff_regression=arff_regression,
                                arff_relation=arff_relation,
                                feat_vectorizer=feat_vectorizer,
                                label_map=label_map)
    # Otherwise write one feature file per subset
    else:
        ids = list(ids)
        classes = list(classes)
        for subset_name, filter_features in iteritems(subsets):
            logger.debug('Subset (%s) features: %s', subset_name,
                         filter_features)
            sub_path = os.path.join(root, '{}{}'.format(subset_name, ext))
            _write_sub_feature_file(sub_path, ids, classes, features,
                                    set(filter_features), label_col=label_col,
                                    arff_regression=arff_regression,
                                    arff_relation=arff_relation,
                                    feat_vectorizer=feat_vectorizer,
                                    label_map=label_map)

