# Copyright (C) 2012-2013 Educational Testing Service

# This file is part of SciKit-Learn Laboratory.

# SciKit-Learn Laboratory is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

# SciKit-Learn Laboratory is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# SciKit-Learn Laboratory.  If not, see <http://www.gnu.org/licenses/>.

'''
Handles loading data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
'''

from __future__ import print_function, unicode_literals

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from csv import DictReader, DictWriter, excel_tab
from decimal import Decimal
from itertools import islice
from io import open
from operator import itemgetter

import numpy as np
from bs4 import UnicodeDammit
from collections import namedtuple
from six import iteritems, string_types
from six.moves import map, zip
from sklearn.feature_extraction import DictVectorizer

MAX_CONCURRENT_PROCESSES = int(os.getenv('SKLL_MAX_CONCURRENT_PROCESSES', '5'))


ExamplesTuple = namedtuple('ExamplesTuple', ['ids', 'classes', 'features',
                                             'feat_vectorizer'])


class _DictIter(object):
    """
    A little helper class to make picklable iterators out of example
    dictionary generators

    :param path_or_list: Path or a list of example dictionaries.
    :type path_or_list: str or list of dict
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    """

    def __init__(self, path_or_list, quiet=True, ids_to_floats=False,
                 tsv_label='y'):
        super(_DictIter, self).__init__()
        self.path_or_list = path_or_list
        self.quiet = quiet
        self.ids_to_floats = ids_to_floats
        self.tsv_label = tsv_label

    def __iter__(self):
        raise NotImplementedError


class _DummyDictIter(_DictIter):
    '''
    This class is to facilitate programmatic use of ``Learner.predict()``
    and other functions that take ``ExamplesTuple`` objects as input.
    It iterates over examples in the same way as other ``_DictIter``s, but uses
    a list of example dictionaries instead of a path to a file.

    :param path_or_list: List of example dictionaries.
    :type path_or_list: list of dict
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''

    def __iter__(self):
        if not self.quiet:
            print("Converting examples...", end="", file=sys.stderr)
            sys.stderr.flush()
        for example_num, example in enumerate(self.path_or_list):
            curr_id = str(example.get("id",
                                      "EXAMPLE_{}".format(example_num)))
            if self.ids_to_floats:
                try:
                    curr_id = float(curr_id)
                except ValueError:
                    raise ValueError(('You set ids_to_floats to true,' +
                                      ' but ID {} could not be ' +
                                      'converted to in ' +
                                      '{}').format(curr_id,
                                                   example))
            class_name = _safe_float(example['y']) if 'y' in example else None
            example = example['x']
            yield curr_id, class_name, example

            if not self.quiet and example_num % 100 == 0:
                print(".", end="", file=sys.stderr)
        if not self.quiet:
            print("done", file=sys.stderr)


class _JSONDictIter(_DictIter):
    '''
    Iterator to convert current line in .jsonlines file to a dictionary with the
    following fields: "SKLL_ID" (originally "id"), "SKLL_CLASS_LABEL"
    (originally "y"), and all of the feature-values in the sub-dictionary "x".
    Basically, we're flattening the structure, but renaming "y" and "id" to
    prevent possible conflicts with feature names in "x".

    :param path_or_list: Path to .jsonlines file
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''

    def __iter__(self):
        with open(self.path_or_list) as f:
            if not self.quiet:
                print("Loading {}...".format(self.path_or_list), end="",
                      file=sys.stderr)
                sys.stderr.flush()
            for example_num, line in enumerate(f):
                example = json.loads(line.strip())
                # Convert all IDs to strings initially,
                # for consistency with csv and megam formats.
                curr_id = str(example.get("id",
                                          "EXAMPLE_{}".format(example_num)))
                class_name = (_safe_float(example["y"]) if 'y' in example
                              else None)
                example = example["x"]

                if self.ids_to_floats:
                    try:
                        curr_id = float(curr_id)
                    except ValueError:
                        raise ValueError(('You set ids_to_floats to true, but' +
                                          ' ID {} could not be converted to ' +
                                          'float').format(curr_id))

                yield curr_id, class_name, example

                if not self.quiet and example_num % 100 == 0:
                    print(".", end="", file=sys.stderr)
            if not self.quiet:
                print("done", file=sys.stderr)


class _MegaMDictIter(_DictIter):
    '''
    Iterator that yields tuples of IDs, classes, and dictionaries mapping from
    features to values for each pair of lines in the MegaM -fvals file specified
    by path.

    :param path_or_list: Path to .megam file (-fvals format)
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''

    def __iter__(self):
        if not self.quiet:
            print("Loading {}...".format(self.path_or_list), end="",
                  file=sys.stderr)
            sys.stderr.flush()
        with open(self.path_or_list, 'rb') as megam_file:
            example_num = 0
            curr_id = 'EXAMPLE_0'
            for line in megam_file:
                # Process encoding
                line = UnicodeDammit(line, ['utf-8',
                                            'windows-1252']).unicode_markup
                line = _sanitize_line(line.strip())
                # Handle instance lines
                if line.startswith('#'):
                    curr_id = line[1:].strip()
                elif line and line not in ['TRAIN', 'TEST', 'DEV']:
                    split_line = line.split()
                    num_cols = len(split_line)
                    del line
                    # Line is just a class label
                    if num_cols == 1:
                        class_name = _safe_float(split_line[0])
                        field_pairs = []
                    # Line has a class label and feature-value pairs
                    elif num_cols % 2 == 1:
                        class_name = _safe_float(split_line[0])
                        field_pairs = split_line[1:]
                    # Line just has feature-value pairs
                    elif num_cols % 2 == 0:
                        class_name = None
                        field_pairs = split_line

                    curr_info_dict = {}
                    if len(field_pairs) > 0:
                        # Get current instances feature-value pairs
                        field_names = islice(field_pairs, 0, None, 2)
                        # Convert values to floats, because otherwise
                        # features'll be categorical
                        field_values = (_safe_float(val) for val in
                                        islice(field_pairs, 1, None, 2))

                        # Add the feature-value pairs to dictionary
                        curr_info_dict.update(zip(field_names, field_values))

                        if len(curr_info_dict) != len(field_pairs) / 2:
                            raise ValueError(('There are duplicate feature ' +
                                              'names in {} for example ' +
                                              '{}.').format(self.path_or_list,
                                                            curr_id))

                    if self.ids_to_floats:
                        try:
                            curr_id = float(curr_id)
                        except ValueError:
                            raise ValueError(('You set ids_to_floats to true,' +
                                              ' but ID {} could not be ' +
                                              'converted to in ' +
                                              '{}').format(curr_id,
                                                           self.path_or_list))

                    yield curr_id, class_name, curr_info_dict

                    # Set default example ID for next instance, in case we see a
                    # line without an ID.
                    example_num += 1
                    curr_id = 'EXAMPLE_{}'.format(example_num)

                    if not self.quiet and example_num % 100 == 0:
                        print(".", end="", file=sys.stderr)
            if not self.quiet:
                print("done", file=sys.stderr)


class _TSVDictIter(_DictIter):
    '''
    Iterator that yields tuples of IDs, classes, and dictionaries mapping from
    features to values for each line in a specified TSV file.

    :param path_or_list: Path to .tsv file
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param tsv_label: Name of the column which contains the class labels.
                      If no column with that name exists, or `None` is
                      specified, the data is considered to be unlabelled.
    :type tsv_label: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''
    def __iter__(self):
        if not self.quiet:
            print("Loading {}...".format(self.path_or_list), end="",
                  file=sys.stderr)
            sys.stderr.flush()
        with open(self.path_or_list) as f:
            reader = DictReader(f, dialect=excel_tab)
            for example_num, row in enumerate(reader):
                if self.tsv_label is not None and self.tsv_label in row:
                    class_name = _safe_float(row[self.tsv_label])
                    del row[self.tsv_label]
                else:
                    class_name = None

                if "id" not in row:
                    curr_id = "EXAMPLE_{}".format(example_num)
                else:
                    curr_id = row["id"]
                    del row["id"]

                # Convert features to floats and if a feature is 0
                # then store the name of the feature so we can
                # delete it later since we don't need to explicitly
                # store zeros in the feature hash
                columns_to_delete = []
                if sys.version_info < (3, 0):
                    columns_to_convert_to_unicode = []
                for fname, fval in iteritems(row):
                    fval_float = _safe_float(fval)
                    # we don't need to explicitly store zeros
                    if fval_float != 0.0:
                        row[fname] = fval_float
                        if sys.version_info < (3, 0):
                            columns_to_convert_to_unicode.append(fname)
                    else:
                        columns_to_delete.append(fname)

                # remove the columns with zero values
                for cname in columns_to_delete:
                    del row[cname]

                # convert the names of all the other columns to
                # unicode for python 2
                if sys.version_info < (3, 0):
                    for cname in columns_to_convert_to_unicode:
                        fval = row[cname]
                        del row[cname]
                        row[cname.decode('utf-8')] = fval

                if self.ids_to_floats:
                    try:
                        curr_id = float(curr_id)
                    except ValueError:
                        raise ValueError(('You set ids_to_floats to true, but' +
                                          ' ID {} could not be converted to ' +
                                          'float').format(curr_id))
                elif sys.version_info < (3, 0):
                    curr_id = curr_id.decode('utf-8')

                yield curr_id, class_name, row

                if not self.quiet and example_num % 100 == 0:
                    print(".", end="", file=sys.stderr)
            if not self.quiet:
                print("done", file=sys.stderr)


def _ids_for_iter_type(example_iter_type, path, ids_to_floats):
    '''
    Little helper function to return an array of IDs for a given example
    generator (and whether or not the examples have labels).
    '''
    example_iter = example_iter_type(path, ids_to_floats=ids_to_floats)
    return np.array([curr_id for curr_id, _, _ in example_iter])


def _classes_for_iter_type(example_iter_type, path, tsv_label):
    '''
    Little helper function to return an array of classes for a given example
    generator (and whether or not the examples have labels).
    '''
    example_iter = example_iter_type(path, tsv_label=tsv_label)
    return np.array([class_name for _, class_name, _ in example_iter])


def _features_for_iter_type(example_iter_type, path, quiet, sparse, tsv_label):
    '''
    Little helper function to return a sparse matrix of features and feature
    vectorizer for a given example generator (and whether or not the examples
    have labels).
    '''
    example_iter = example_iter_type(path, quiet=quiet, tsv_label=tsv_label)
    feat_vectorizer = DictVectorizer(sparse=sparse)
    feat_dict_generator = map(itemgetter(2), example_iter)
    try:
        features = feat_vectorizer.fit_transform(feat_dict_generator)
    except ValueError:
        raise ValueError('The last feature file did not include any features.')
    return features, feat_vectorizer


def load_examples(path, quiet=False, sparse=True, tsv_label='y',
                  ids_to_floats=False):
    '''
    Loads examples in the TSV, JSONLINES (a json dict per line), or MegaM
    formats.

    If you would like to include example/instance IDs in your files, they must
    be specified in the following ways:

    * MegaM: As a comment line directly preceding the line with feature values.
    * TSV: An "id" column.
    * JSONLINES: An "id" key in each JSON dictionary.

    Also, for TSV files, there must be a column with the name specified by
    `tsv_label` if the data is labelled.

    :param path: The path to the file to load the examples from, or a list of
                 example dictionaries (like you would pass to
                 `convert_examples`).
    :type path: str or dict
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param sparse: Whether or not to store the features in a numpy CSR matrix.
    :type sparse: bool
    :param tsv_label: Name of the column which contains the class labels for
                      TSV files. If no column with that name exists, or `None`
                      is specified, the data is considered to be unlabelled.
    :type tsv_label: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool

    :return: 4-tuple of an array example ids, an array of class labels, a
             scipy CSR matrix of features, and a DictVectorizer containing
             the mapping between feature names and the column indices in
             the feature matrix.
    '''
    # Setup logger
    logger = logging.getLogger(__name__)

    logger.debug('Path: {}'.format(path))

    # Build an appropriate generator for examples so we process the input file
    # through the feature vectorizer without using tons of memory
    if not isinstance(path, string_types):
        example_iter_type = _DummyDictIter
    elif path.endswith(".tsv"):
        example_iter_type = _TSVDictIter
    elif path.endswith(".jsonlines"):
        example_iter_type = _JSONDictIter
    elif path.endswith(".megam"):
        example_iter_type = _MegaMDictIter
    else:
        raise ValueError('Example files must be in either .tsv, .megam, or ' +
                         '.jsonlines format. You specified: {}'.format(path))

    logger.debug('Example iterator type: {}'.format(example_iter_type))

    # Generators can't be pickled, so unfortunately we have to turn them into
    # lists. Would love a workaround for this.
    if example_iter_type == _DummyDictIter:
        path = list(path)

    if MAX_CONCURRENT_PROCESSES == 1:
        executor_type = ThreadPoolExecutor
    else:
        executor_type = ProcessPoolExecutor

    # Create generators that we can use to create numpy arrays without wasting
    # memory (even though this requires reading the file multiple times).
    # Do this using a process pool so that we can clear out the temporary
    # variables more easily and do these in parallel.
    with executor_type(max_workers=3) as executor:
        ids_future = executor.submit(_ids_for_iter_type, example_iter_type,
                                     path, ids_to_floats)
        classes_future = executor.submit(_classes_for_iter_type,
                                         example_iter_type, path, tsv_label)
        features_future = executor.submit(_features_for_iter_type,
                                          example_iter_type, path, quiet,
                                          sparse, tsv_label)

        # Wait for processes/threads to complete and store results
        ids = ids_future.result()
        classes = classes_future.result()
        features, feat_vectorizer = features_future.result()

    # Make sure we have the same number of ids, classes, and features
    assert ids.shape[0] == classes.shape[0] == features.shape[0]

    return ExamplesTuple(ids, classes, features, feat_vectorizer)


def convert_examples(example_dicts, sparse=True, ids_to_floats=False):
    '''
    This function is to facilitate programmatic use of Learner.predict()
    and other functions that take ExamplesTuple objects as input.
    It converts a .jsonlines-style list of dictionaries into
    an ExamplesTuple.

    :param example_dicts: An list of dictionaries following the .jsonlines
                          format (i.e., features 'x', label 'y', and 'id').
    :type example_dicts: iterable of dicts

    :return an ExamplesTuple representing the examples in example_dicts.
    '''
    return load_examples(example_dicts, sparse=sparse, quiet=True,
                         ids_to_floats=ids_to_floats)


def _sanitize_line(line):
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


def _safe_float(text):
    '''
    Attempts to convert a string to a float, but if that's not possible, just
    returns the original value.
    '''
    try:
        return float(text)
    except ValueError:
        return text.decode('utf-8') if sys.version_info < (3, 0) else text


def write_feature_file(path, ids, classes, features, feat_vectorizer=None,
                       id_prefix='EXAMPLE_', tsv_label='y'):
    '''
    Writes output a feature file in either .jsonlines, .megam, or .tsv formats
    with the given a list of IDs, classes, and features.

    :param path: A path to the feature file we would like to create. The suffix
                 to this filename must be .jsonlines, .megam, or .tsv.
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
                     of dictionaries or an array-like (if `feat_vectorizer`
                     is also specified).
    :type features: list of dict or array-like
    :param feat_vectorizer: A `DictVectorizer` to map to/from feature columns
                            indices and names.
    :type feat_vectorizer: DictVectorizer
    :param id_prefix: If we need to automatically generate IDs, put this prefix
                      infront of the numerical IDs.
    :type id_prefix: str
    :param tsv_label: Name of the column which contains the class labels for
                      TSV files. If no column with that name exists, or `None`
                      is specified, the data is considered to be unlabelled.
    :type tsv_label: str
    '''
    # Setup logger
    logger = logging.getLogger(__name__)

    logger.debug('Feature vectorizer: {}'.format(feat_vectorizer))
    logger.debug('Features: {}'.format(features))

    # Check for valid features
    if feat_vectorizer is None and features and not isinstance(features[0],
                                                               dict):
        raise ValueError('If `feat_vectorizer` is unspecified, you must pass ' +
                         'a list of dicts for `features`.')

    # Convert features to list of dicts if given an array-like and a vectorizer
    if feat_vectorizer is not None:
        features = feat_vectorizer.inverse_transform(features)

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

    # Create TSV file if asked
    if path.endswith(".tsv"):
        if sys.version_info >= (3, 0):
            file_mode = 'w'
            delimiter = '\t'
        else:
            file_mode = 'wb'
            delimiter = b'\t'
        with open(path, file_mode) as f:
            instances = []
            fields = set()
            # Iterate through examples
            for ex_id, class_name, feature_dict in zip(ids, classes, features):
                # Don't try to add class column if this is label-less data
                if tsv_label not in feature_dict:
                    if class_name is not None:
                        feature_dict[tsv_label] = class_name
                else:
                    raise ValueError(('Class column name "{0}" already used ' +
                                      'as feature name!').format(tsv_label))

                if 'id' not in feature_dict:
                    if ex_id is not None:
                        feature_dict['id'] = ex_id
                else:
                    raise ValueError('ID column name "id" already used as ' +
                                     'feature name!')
                fields.update(feature_dict.keys())
                instances.append(feature_dict)

            # Create writer
            writer = DictWriter(f, fieldnames=sorted(fields),
                                delimiter=delimiter, restval=0)
            # Output instance
            writer.writeheader()
            writer.writerows(instances)

    # Create .jsonlines file if asked
    elif path.endswith(".jsonlines"):
        file_mode = 'w' if sys.version_info >= (3, 0) else 'wb'
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

    # Create .jsonlines file if asked
    elif path.endswith(".megam"):
        with open(path, 'w') as f:
            # Iterate through examples
            for ex_id, class_name, feature_dict in zip(ids, classes, features):
                # Don't try to add class column if this is label-less data
                if ex_id is not None:
                    print('# {}'.format(ex_id), file=f)
                if class_name is not None:
                    print(class_name, end='\t', file=f)
                print(' '.join(('{} {}'.format(field, value) for field, value in
                                sorted(feature_dict.items()) if
                                Decimal(value) != 0)),
                      file=f)

    # Invalid file suffix, raise error
    else:
        raise ValueError('Output file must be in either .tsv, .megam, or ' +
                         '.jsonlines format. You specified: {}'.format(path))
