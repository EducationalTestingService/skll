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
import sys
from csv import DictReader, DictWriter, excel_tab
from decimal import Decimal
from itertools import islice
from io import open
from multiprocessing import Pool
from operator import itemgetter

import numpy as np
from bs4 import UnicodeDammit
from collections import namedtuple
from six import iteritems
from six.moves import map, zip
from sklearn.feature_extraction import DictVectorizer


ExamplesTuple = namedtuple('ExamplesTuple', ['ids', 'classes', 'features',
                                             'feat_vectorizer'])


def _make_examples_generator(example_gen_func, path, quiet=True,
                             tsv_label='y', ids_to_floats=True):
    '''
    This function simply calls example_gen_func, which should be
    one of the data loading functions (e.g., _tsv_dict_iter),
    with the appropriate arguments.

    Note: ids_to_floats defaults to True to save memory.
    '''
    if example_gen_func == _tsv_dict_iter:
        return example_gen_func(path, quiet=quiet, tsv_label=tsv_label,
                                ids_to_floats=ids_to_floats)
    else:
        return example_gen_func(path, quiet=quiet,
                                ids_to_floats=ids_to_floats)


def _ids_for_gen_func(example_gen_func, path, ids_to_floats=False):
    '''
    Little helper function to return an array of IDs for a given example
    generator (and whether or not the examples have labels).
    '''
    gen_results = _make_examples_generator(example_gen_func, path,
                                           ids_to_floats=ids_to_floats)
    return _ids_for_gen_func_helper(gen_results)


def _ids_for_gen_func_helper(gen_results):
    '''
    See _ids_for_gen_func.
    '''
    return np.array([curr_id for curr_id, _, _ in gen_results])


def _classes_for_gen_func(example_gen_func, path, tsv_label):
    '''
    Little helper function to return an array of classes for a given example
    generator (and whether or not the examples have labels).
    '''
    gen_results = _make_examples_generator(example_gen_func, path,
                                           tsv_label=tsv_label)
    return _classes_for_gen_func_helper(gen_results)


def _classes_for_gen_func_helper(gen_results):
    '''
    See _classes_for_gen_func.
    '''
    return np.array([class_name for _, class_name, _ in gen_results])


def _features_for_gen_func(example_gen_func, path, quiet, sparse):
    '''
    Little helper function to return a sparse matrix of features and feature
    vectorizer for a given example generator (and whether or not the examples
    have labels).
    '''
    gen_results = _make_examples_generator(example_gen_func, path,
                                           quiet=quiet)
    return _features_for_gen_func_helper(gen_results, sparse)


def _features_for_gen_func_helper(gen_results, sparse):
    '''
    See _features_for_gen_func.
    '''
    feat_vectorizer = DictVectorizer(sparse=sparse)
    feat_dict_generator = map(itemgetter(2), gen_results)
    features = feat_vectorizer.fit_transform(feat_dict_generator)
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

    :param path: The path to the file to load the examples from.
    :type path: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param sparse: Whether or not to store the features in a numpy CSR matrix.
    :type sparse: bool
    :param tsv_label: Name of the column which contains the class labels for
                      TSV files. If no column with that name exists, or `None`
                      is specified, the data is considered to be unlabelled.
    :type tsv_label: str

    :return: 4-tuple of an array example ids, an array of class labels, a
             scipy CSR matrix of features, and a DictVectorizer containing
             the mapping between feature names and the column indices in
             the feature matrix.
    '''
    # Build an appropriate generator for examples so we process the input file
    # through the feature vectorizer without using tons of memory
    if path.endswith(".tsv"):
        example_gen_func = _tsv_dict_iter
    elif path.endswith(".jsonlines"):
        example_gen_func = _json_dict_iter
    elif path.endswith(".megam"):
        example_gen_func = _megam_dict_iter
    else:
        raise ValueError('Example files must be in either .tsv, .megam, or ' +
                         '.jsonlines format. You specified: {}'.format(path))

    # Create generators that we can use to create numpy arrays without wasting
    # memory (even though this requires reading the file multiple times).
    # Do this using a process pool so that we can clear out the temporary
    # variables more easily and do these in parallel.
    pool = Pool(3)

    ids_result = pool.apply_async(_ids_for_gen_func,
                                  args=(example_gen_func, path,
                                        ids_to_floats))
    classes_result = pool.apply_async(_classes_for_gen_func,
                                      args=(example_gen_func, path,
                                            tsv_label))
    features_result = pool.apply_async(_features_for_gen_func,
                                       args=(example_gen_func, path, quiet,
                                             sparse))

    # Wait for processes to complete and store results
    pool.close()
    pool.join()
    ids = ids_result.get()
    classes = classes_result.get()
    features, feat_vectorizer = features_result.get()

    return ExamplesTuple(ids, classes, features, feat_vectorizer)


def convert_examples(example_dicts, sparse=True, ids_to_floats=False):
    '''
    This function is to facilitate programmatic use of Learner.predict()
    and other functions that take ExamplesTuple objects as input.
    It converts a .jsonlines-style list of dictionaries into
    an ExamplesTuple.

    :param example_dicts: A list of dictionaries following the .jsonlines
                          format (i.e., features 'x', label 'y', and 'id').
    :type example_dicts: list of dict

    :return an ExamplesTuple representing the examples in example_dicts.
    '''

    gen_results = [(_safe_float(d['id']) if ids_to_floats else d['id'],
                    d['y'] if 'y' in d else None,
                    d['x'])
                   for d in example_dicts]
    ids = _ids_for_gen_func_helper(gen_results)
    classes = _classes_for_gen_func_helper(gen_results)
    features, feat_vectorizer = _features_for_gen_func_helper(gen_results,
                                                              sparse)

    return ExamplesTuple(ids, classes, features, feat_vectorizer)


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
        return text


def _json_dict_iter(path, quiet=False, ids_to_floats=False):
    '''
    Convert current line in .jsonlines file to a dictionary with the
    following fields: "SKLL_ID" (originally "id"), "SKLL_CLASS_LABEL"
    (originally "y"), and all of the feature-values in the sub-dictionary
    "x". Basically, we're flattening the structure, but renaming "y" and "id"
    to prevent possible conflicts with feature names in "x".

    :param path: Path to .jsonlines file
    :type path: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    '''
    with open(path) as f:
        if not quiet:
            print("Loading {}...".format(path), end="", file=sys.stderr)
            sys.stderr.flush()
        for example_num, line in enumerate(f):
            example = json.loads(line.strip())
            # Convert all IDs to strings initially,
            # for consistency with csv and megam formats.
            curr_id = str(example.get("id", "EXAMPLE_{}".format(example_num)))
            class_name = _safe_float(example["y"]) if 'y' in example else None
            example = example["x"]

            if ids_to_floats:
                curr_id = _safe_float(curr_id)

            yield curr_id, class_name, example

            if not quiet and example_num % 100 == 0:
                print(".", end="", file=sys.stderr)
        if not quiet:
            print("done", file=sys.stderr)


def _megam_dict_iter(path, quiet=False, ids_to_floats=False):
    '''
    Generator that yields tuples of IDs, classes, and dictionaries mapping from
    features to values for each pair of lines in the MegaM -fvals file specified
    by path.

    :param path: Path to MegaM file (-fvals format)
    :type path: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    '''

    if not quiet:
        print("Loading {}...".format(path), end="", file=sys.stderr)
        sys.stderr.flush()
    with open(path, 'rb') as megam_file:
        example_num = 0
        curr_id = 'EXAMPLE_0'
        for line in megam_file:
            # Process encoding
            line = UnicodeDammit(line, ['utf-8', 'windows-1252']).unicode_markup
            line = _sanitize_line(line.strip())
            # Handle instance lines
            if line.startswith('#'):
                curr_id = line[1:].strip()
            elif line and line not in ['TRAIN', 'TEST', 'DEV']:
                has_labels = '\t' in line
                split_line = line.split()
                del line
                curr_info_dict = {}

                if has_labels:
                    class_name = _safe_float(split_line[0])
                    field_pairs = split_line[1:]
                else:
                    class_name = None
                    field_pairs = split_line

                if len(field_pairs) > 0:
                    # Get current instances feature-value pairs
                    field_names = islice(field_pairs, 0, None, 2)
                    # Convert values to floats, because otherwise features'll
                    # be categorical
                    field_values = (_safe_float(val) for val in
                                    islice(field_pairs, 1, None, 2))

                    # TODO: Add some sort of check for duplicate feature names

                    # Add the feature-value pairs to dictionary
                    curr_info_dict.update(zip(field_names, field_values))

                if ids_to_floats:
                    curr_id = _safe_float(curr_id)

                yield curr_id, class_name, curr_info_dict

                # Set default example ID for next instance, in case we see a
                # line without an ID.
                example_num += 1
                curr_id = 'EXAMPLE_{}'.format(example_num)

                if not quiet and example_num % 100 == 0:
                    print(".", end="", file=sys.stderr)
        if not quiet:
            print("done", file=sys.stderr)


def _tsv_dict_iter(path, quiet=False, tsv_label='y', ids_to_floats=False):
    '''
    Generator that yields tuples of IDs, classes, and dictionaries mapping from
    features to values for each pair of lines in the MegaM -fvals file specified
    by path.

    :param path: Path to TSV
    :type path: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param tsv_label: Name of the column which contains the class labels.
                      If no column with that name exists, or `None` is
                      specified, the data is considered to be unlabelled.
    :type tsv_label: str
    '''
    if not quiet:
        print("Loading {}...".format(path), end="", file=sys.stderr)
        sys.stderr.flush()
    with open(path) as f:
        reader = DictReader(f, dialect=excel_tab)
        for example_num, row in enumerate(reader):
            if tsv_label is not None and tsv_label in row:
                class_name = _safe_float(row[tsv_label])
                del row[tsv_label]
            else:
                class_name = None

            if "id" not in row:
                curr_id = "EXAMPLE_{}".format(example_num)
            else:
                curr_id = row["id"]
                del row["id"]

            # Convert features to floats
            for fname, fval in iteritems(row):
                fval_float = _safe_float(fval)
                # we don't need to explicitly store zeros
                if fval_float != 0.0:
                    row[fname] = fval_float

            if ids_to_floats:
                curr_id = _safe_float(curr_id)

            yield curr_id, class_name, row

            if not quiet and example_num % 100 == 0:
                print(".", end="", file=sys.stderr)
        if not quiet:
            print("done", file=sys.stderr)


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
            ids = ('{}{}'.format(id_prefix, num) for num in range(len(features)))
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
                                sorted(feature_dict.items()) if Decimal(value) != 0)),
                      file=f)

    # Invalid file suffix, raise error
    else:
        raise ValueError('Output file must be in either .tsv, .megam, or ' +
                         '.jsonlines format. You specified: {}'.format(path))
