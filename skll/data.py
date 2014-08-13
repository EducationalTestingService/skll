# License: BSD 3 clause
'''
Handles loading data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
'''

from __future__ import print_function, unicode_literals

import csv
import json
import logging
import os
import re
import sys
from csv import DictReader, DictWriter
from decimal import Decimal
from itertools import chain, islice
from io import open, BytesIO, StringIO
from multiprocessing import Queue
from operator import itemgetter

import numpy as np
from bs4 import UnicodeDammit
from collections import namedtuple
from joblib.pool import MemmapingPool
from six import iteritems, PY2, string_types, text_type
from six.moves import map, zip
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

# Import QueueHandler and QueueListener for multiprocess-safe logging
if PY2:
    from logutils.queue import QueueHandler, QueueListener
else:
    from logging.handlers import QueueHandler, QueueListener

MAX_CONCURRENT_PROCESSES = int(os.getenv('SKLL_MAX_CONCURRENT_PROCESSES', '5'))


ExamplesTuple = namedtuple('ExamplesTuple', ['ids', 'classes', 'features',
                                             'feat_vectorizer'])

# Register dialect for handling ARFF files
if sys.version_info >= (3, 0):
    csv.register_dialect('arff', delimiter=',', quotechar="'",
                         escapechar='\\', doublequote=False,
                         lineterminator='\n', skipinitialspace=True)
else:
    csv.register_dialect('arff', delimiter=b',', quotechar=b"'",
                         escapechar=b'\\', doublequote=False,
                         lineterminator=b'\n', skipinitialspace=True)


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
    :param class_map: Mapping from original class labels to new ones. This is
                      mainly used for collapsing multiple classes into a single
                      class. Anything not in the mapping will be kept the same.
    :type class_map: dict from str to str
    """
    def __init__(self, path_or_list, quiet=True, ids_to_floats=False,
                 label_col='y', class_map=None):
        super(_DictIter, self).__init__()
        self.path_or_list = path_or_list
        self.quiet = quiet
        self.ids_to_floats = ids_to_floats
        self.label_col = label_col
        self.class_map = class_map

    def __iter__(self):
        # Setup logger
        logger = logging.getLogger(__name__)

        logger.debug('DictIter type: %s', type(self))
        logger.debug('path_or_list: %s', self.path_or_list)

        # Check if we're given a path to a file, and if so, open it
        if isinstance(self.path_or_list, string_types):
            if sys.version_info >= (3, 0):
                file_mode = 'r'
            else:
                file_mode = 'rb'
            if not self.quiet:
                print("Loading {}...".format(self.path_or_list), end="",
                      file=sys.stderr)
                sys.stderr.flush()
            with open(self.path_or_list, file_mode) as f:
                for ret_tuple in self._sub_iter(f):
                    yield ret_tuple
        else:
            if not self.quiet:
                print("Loading...".format(self.path_or_list), end="",
                      file=sys.stderr)
                sys.stderr.flush()
            for ret_tuple in self._sub_iter(self.path_or_list):
                yield ret_tuple

    def _sub_iter(self, file_or_list):
        '''
        Iterates through a given file or list.
        '''
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
                                      'converted to float in ' +
                                      '{}').format(curr_id, example))
            class_name = (_safe_float(example['y'],
                                      replace_dict=self.class_map)
                          if 'y' in example else None)
            example = example['x']
            yield curr_id, class_name, example

            if not self.quiet and example_num % 100 == 0:
                print(".", end="", file=sys.stderr)
        if not self.quiet:
            print("done", file=sys.stderr)


class _JSONDictIter(_DictIter):
    '''
    Iterator to convert current line in .jsonlines/.ndj file to a dictionary
    with the following fields: "SKLL_ID" (originally "id"), "SKLL_CLASS_LABEL"
    (originally "y"), and all of the feature-values in the sub-dictionary "x".
    Basically, we're flattening the structure, but renaming "y" and "id" to
    prevent possible conflicts with feature names in "x".

    :param path_or_list: Path to .jsonlines/.ndj file
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''

    def _sub_iter(self, file_or_list):
        for example_num, line in enumerate(file_or_list):
            # Remove extraneous whitespace
            line = line.strip()

            # If this is a comment line or a blank line, move on
            if line.startswith('//') or not line:
                continue

            # Process good lines
            example = json.loads(line)
            # Convert all IDs to strings initially,
            # for consistency with csv and megam formats.
            curr_id = str(example.get("id",
                                      "EXAMPLE_{}".format(example_num)))
            class_name = (_safe_float(example['y'],
                                      replace_dict=self.class_map)
                          if 'y' in example else None)
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
    features to values for each pair of lines in the MegaM -fvals file
    specified by path.

    :param path_or_list: Path to .megam file (-fvals format)
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''

    def _sub_iter(self, file_or_list):
        example_num = 0
        curr_id = 'EXAMPLE_0'
        for line in file_or_list:
            # Process encoding
            if not isinstance(line, text_type):
                line = UnicodeDammit(line, ['utf-8',
                                            'windows-1252']).unicode_markup
            line = line.strip()
            # Handle instance lines
            if line.startswith('#'):
                curr_id = line[1:].strip()
            elif line and line not in ['TRAIN', 'TEST', 'DEV']:
                split_line = line.split()
                num_cols = len(split_line)
                del line
                # Line is just a class label
                if num_cols == 1:
                    class_name = _safe_float(split_line[0],
                                             replace_dict=self.class_map)
                    field_pairs = []
                # Line has a class label and feature-value pairs
                elif num_cols % 2 == 1:
                    class_name = _safe_float(split_line[0],
                                             replace_dict=self.class_map)
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
                                          'converted to float in ' +
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


class _LibSVMDictIter(_DictIter):
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
    line_regex = re.compile(r'^(?P<label_num>[^ ]+)\s+(?P<features>[^#]*)\s*'
                            r'(?P<comments>#\s*(?P<example_id>[^|]+)\s*\|\s*'
                            r'(?P<label_map>[^|]+)\s*\|\s*'
                            r'(?P<feat_map>.*)\s*)?$')

    @staticmethod
    def _pair_to_tuple(pair, feat_map):
        '''
        Split a feature-value pair separated by a colon into a tuple.  Also
        do _safe_float conversion on the value.
        '''
        name, value = pair.split(':')
        if feat_map is not None:
            name = feat_map[name]
        value = _safe_float(value)
        return (name, value)

    def _sub_iter(self, file_or_list):
        for example_num, line in enumerate(file_or_list):
            curr_id = ''
            # Decode line if it's not already str
            if isinstance(line, bytes):
                line = UnicodeDammit(line, ['utf-8',
                                            'windows-1252']).unicode_markup
            match = self.line_regex.search(line.strip())
            if not match:
                raise ValueError('Line does not look like valid libsvm format'
                                 '\n{}'.format(line))
            # Metadata is stored in comments if this was produced by SKLL
            if match.group('comments') is not None:
                # Store mapping from feature numbers to names
                if match.group('feat_map'):
                    feat_map = dict(pair.split('=') for pair in
                                    match.group('feat_map').split())
                else:
                    feat_map = None
                # Store mapping from label/class numbers to names
                if match.group('label_map'):
                    label_map = dict(pair.split('=') for pair in
                                     match.group('label_map').strip().split())
                else:
                    label_map = None
                curr_id = match.group('example_id').strip()

            if not curr_id:
                curr_id = 'EXAMPLE_{}'.format(example_num)

            class_num = match.group('label_num')
            # If we have a mapping from class numbers to labels, get label
            if label_map is not None:
                class_name = label_map[class_num]
            class_name = _safe_float(class_name,
                                     replace_dict=self.class_map)

            curr_info_dict = dict(self._pair_to_tuple(pair, feat_map) for pair
                                  in match.group('features').strip().split())

            if self.ids_to_floats:
                try:
                    curr_id = float(curr_id)
                except ValueError:
                    raise ValueError(('You set ids_to_floats to true,' +
                                      ' but ID {} could not be ' +
                                      'converted to float in ' +
                                      '{}').format(curr_id,
                                                   self.path_or_list))

            yield curr_id, class_name, curr_info_dict

            if not self.quiet and example_num % 100 == 0:
                print(".", end="", file=sys.stderr)
        if not self.quiet:
            print("done", file=sys.stderr)


class _DelimitedDictIter(_DictIter):
    '''
    Iterator that yields tuples of IDs, classes, and dictionaries mapping from
    features to values for each line in a specified delimited (CSV/TSV) file.

    :param path_or_list: Path to .tsv file
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param label_col: Name of the column which contains the class labels.
                      If no column with that name exists, or `None` is
                      specified, the data is considered to be unlabelled.
    :type label_col: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    :param class_map: Mapping from original class labels to new ones. This is
                      mainly used for collapsing multiple classes into a single
                      class. Anything not in the mapping will be kept the same.
    :type class_map: dict from str to str
    :param dialect: The dialect of to pass on to the underlying CSV reader.
    :type dialect: str
    '''
    def __init__(self, path_or_list, quiet=True, ids_to_floats=False,
                 label_col='y', class_map=None, dialect=None):
        super(_DelimitedDictIter, self).__init__(path_or_list, quiet=quiet,
                                                 ids_to_floats=ids_to_floats,
                                                 label_col=label_col,
                                                 class_map=class_map)
        self.dialect = dialect

    def _sub_iter(self, file_or_list):
        reader = DictReader(file_or_list, dialect=self.dialect)
        for example_num, row in enumerate(reader):
            if self.label_col is not None and self.label_col in row:
                class_name = _safe_float(row[self.label_col],
                                         replace_dict=self.class_map)
                del row[self.label_col]
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
            if PY2:
                columns_to_convert_to_unicode = []
            for fname, fval in iteritems(row):
                fval_float = _safe_float(fval)
                # we don't need to explicitly store zeros
                if fval_float:
                    row[fname] = fval_float
                    if PY2:
                        columns_to_convert_to_unicode.append(fname)
                else:
                    columns_to_delete.append(fname)

            # remove the columns with zero values
            for cname in columns_to_delete:
                del row[cname]

            # convert the names of all the other columns to
            # unicode for python 2
            if PY2:
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
            elif PY2:
                curr_id = curr_id.decode('utf-8')

            yield curr_id, class_name, row

            if not self.quiet and example_num % 100 == 0:
                print(".", end="", file=sys.stderr)
        if not self.quiet:
            print("done", file=sys.stderr)


class _CSVDictIter(_DelimitedDictIter):
    '''
    Iterator that yields tuples of IDs, classes, and dictionaries mapping from
    features to values for each line in a specified CSV file.

    :param path_or_list: Path to .tsv file
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param label_col: Name of the column which contains the class labels.
                      If no column with that name exists, or `None` is
                      specified, the data is considered to be unlabelled.
    :type label_col: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''
    def __init__(self, path_or_list, quiet=True, ids_to_floats=False,
                 label_col='y', class_map=None):
        super(_CSVDictIter, self).__init__(path_or_list, quiet=quiet,
                                           ids_to_floats=ids_to_floats,
                                           label_col=label_col,
                                           dialect='excel',
                                           class_map=class_map)


class _ARFFDictIter(_DelimitedDictIter):
    '''
    Iterator that yields tuples of IDs, classes, and dictionaries mapping from
    features to values for each data line in a specified ARFF file.

    :param path_or_list: Path to .tsv file
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param label_col: Name of the column which contains the class labels.
                      If no column with that name exists, or `None` is
                      specified, the data is considered to be unlabelled.
    :type label_col: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''
    def __init__(self, path_or_list, quiet=True, ids_to_floats=False,
                 label_col='y', class_map=None):
        super(_ARFFDictIter, self).__init__(path_or_list, quiet=quiet,
                                            ids_to_floats=ids_to_floats,
                                            label_col=label_col,
                                            dialect='arff',
                                            class_map=class_map)

    @staticmethod
    def split_with_quotes(s, delimiter=' ', quote_char="'", escape_char='\\'):
        '''
        A replacement for string.split that won't split delimiters enclosed in
        quotes.
        '''
        if PY2:
            delimiter = delimiter.encode()
            quote_char = quote_char.encode()
            escape_char = escape_char.encode()
        return next(csv.reader([s], delimiter=delimiter, quotechar=quote_char,
                               escapechar=escape_char))

    def _sub_iter(self, file_or_list):
        field_names = []
        # Process ARFF header
        for line in file_or_list:
            # Process encoding
            if not isinstance(line, text_type):
                decoded_line = UnicodeDammit(line, ['utf-8',
                                             'windows-1252']).unicode_markup
            else:
                decoded_line = line
            line = decoded_line.strip()
            # Skip empty lines
            if line:
                # Split the line using CSV reader because it can handle
                # quoted delimiters.
                split_header = self.split_with_quotes(line)
                row_type = split_header[0].lower()
                if row_type == '@attribute':
                    # Add field name to list
                    field_names.append(split_header[1])
                # Stop at data
                elif row_type == '@data':
                    break
                # Skip other types of rows (relations)

        # Create header for CSV
        if PY2:
            io_type = BytesIO
        else:
            io_type = StringIO
        with io_type() as field_buffer:
            csv.writer(field_buffer, dialect='arff').writerow(field_names)
            field_str = field_buffer.getvalue()

        # Set label_col to be the name of the last field, since that's standard
        # for ARFF files
        if self.label_col != field_names[-1]:
            self.label_col = None

        # Process data as CSV file
        return super(_ARFFDictIter, self)._sub_iter(chain([field_str],
                                                          file_or_list))


class _TSVDictIter(_DelimitedDictIter):
    '''
    Iterator that yields tuples of IDs, classes, and dictionaries mapping from
    features to values for each line in a specified TSV file.

    :param path_or_list: Path to .tsv file
    :type path_or_list: str
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param label_col: Name of the column which contains the class labels.
                      If no column with that name exists, or `None` is
                      specified, the data is considered to be unlabelled.
    :type label_col: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    '''
    def __init__(self, path_or_list, quiet=True, ids_to_floats=False,
                 label_col='y', class_map=None):
        super(_TSVDictIter, self).__init__(path_or_list, quiet=quiet,
                                           ids_to_floats=ids_to_floats,
                                           label_col=label_col,
                                           dialect='excel-tab',
                                           class_map=class_map)


def _ids_for_iter_type(example_iter_type, path, ids_to_floats):
    '''
    Little helper function to return an array of IDs for a given example
    generator (and whether or not the examples have labels).
    '''
    try:
        example_iter = example_iter_type(path, ids_to_floats=ids_to_floats)
        res_array = np.array([curr_id for curr_id, _, _ in example_iter])
    except Exception as e:
        # Setup logger
        logger = logging.getLogger(__name__)
        logger.exception('Failed to load IDs for %s.', path)
        raise e
    return res_array


def _classes_for_iter_type(example_iter_type, path, label_col, class_map):
    '''
    Little helper function to return an array of classes for a given example
    generator (and whether or not the examples have labels).
    '''
    try:
        example_iter = example_iter_type(path, label_col=label_col,
                                         class_map=class_map)
        res_array = np.array([class_name for _, class_name, _ in example_iter])
    except Exception as e:
        # Setup logger
        logger = logging.getLogger(__name__)
        logger.exception('Failed to load classes for %s.', path)
        raise e
    return res_array


def _features_for_iter_type(example_iter_type, path, quiet, sparse, label_col,
                            feature_hasher, num_features):
    '''
    Little helper function to return a sparse matrix of features and feature
    vectorizer for a given example generator (and whether or not the examples
    have labels).
    '''
    try:
        example_iter = example_iter_type(path, quiet=quiet,
                                         label_col=label_col)
        if feature_hasher:
            feat_vectorizer = FeatureHasher(n_features=num_features)
        else:
            feat_vectorizer = DictVectorizer(sparse=sparse)
        feat_dict_generator = map(itemgetter(2), example_iter)
    except Exception:
        # Setup logger
        logger = logging.getLogger(__name__)
        logger.exception('Failed to load features for %s.', path)
        raise
    try:
        if feature_hasher:
            features = feat_vectorizer.transform(feat_dict_generator)
        else:
            features = feat_vectorizer.fit_transform(feat_dict_generator)
    except ValueError:
        logger = logging.getLogger(__name__)
        logger.error('The last feature file did not include any features.')
        raise
    return features, feat_vectorizer


def load_examples(path, quiet=False, sparse=True, label_col='y',
                  ids_to_floats=False, class_map=None, feature_hasher=False,
                  num_features=None):
    '''
    Loads examples in the ``.arff``, ``.csv``, ``.jsonlines``, ``.libsvm``,
    ``.megam``, ``.ndj``, or ``.tsv`` formats.

    If you would like to include example/instance IDs in your files, they must
    be specified in the following ways:

    * MegaM: As a comment line directly preceding the line with feature values.
    * LibSVM: As the first item in the three-part comment described below.
    * CSV/TSV/ARFF: An "id" column.
    * JSONLINES: An "id" key in each JSON dictionary.

    Also, for ARFF, CSV, and TSV files, there must be a column with the name
    specified by `label_col` if the data is labelled. For ARFF files, this
    column must also be the final one (as it is in Weka).

    For LibSVM files, we use a specially formatted comment for storing example
    IDs, class names, and feature names, which are normally not supported by
    the format.  The comment is not mandatory, but without it, your classes
    and features will not have names.  The comment is structured as follows:

        ExampleID | 1=FirstClass | 1=FirstFeature 2=SecondFeature

    :param path: The path to the file to load the examples from, or a list of
                 example dictionaries (like you would pass to
                 `convert_examples`).
    :type path: str or dict
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param sparse: Whether or not to store the features in a numpy CSR matrix.
    :type sparse: bool
    :param label_col: Name of the column which contains the class labels for
                      ARFF/CSV/TSV files. If no column with that name exists, or
                      `None` is specified, the data is considered to be
                      unlabelled.
    :type label_col: str
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    :param class_map: Mapping from original class labels to new ones. This is
                      mainly used for collapsing multiple classes into a single
                      class. Anything not in the mapping will be kept the same.
    :type class_map: dict from str to str

    :returns: 4-tuple of an array example ids, an array of class labels, a
              scipy CSR matrix of features, and a DictVectorizer containing
              the mapping between feature names and the column indices in
              the feature matrix.
    '''
    # Setup logger
    logger = logging.getLogger(__name__)

    logger.debug('Path: %s', path)

    # Build an appropriate generator for examples so we process the input file
    # through the feature vectorizer without using tons of memory
    if not isinstance(path, string_types):
        example_iter_type = _DummyDictIter
    # Lowercase path for file extension checking, if it's a string
    else:
        lc_path = path.lower()
        if lc_path.endswith(".tsv"):
            example_iter_type = _TSVDictIter
        elif lc_path.endswith(".csv"):
            example_iter_type = _CSVDictIter
        elif lc_path.endswith(".arff"):
            example_iter_type = _ARFFDictIter
        elif lc_path.endswith(".jsonlines") or lc_path.endswith('.ndj'):
            example_iter_type = _JSONDictIter
        elif lc_path.endswith(".megam"):
            example_iter_type = _MegaMDictIter
        elif lc_path.endswith(".libsvm"):
            example_iter_type = _LibSVMDictIter
        else:
            raise ValueError(('Example files must be in either .arff, .csv, ' +
                              '.jsonlines, .megam, .ndj, or .tsv format. You ' +
                              'specified: {}').format(path))

    logger.debug('Example iterator type: %s', example_iter_type)

    # Generators can't be pickled, so unfortunately we have to turn them into
    # lists. Would love a workaround for this.
    if example_iter_type == _DummyDictIter:
        path = list(path)

    # Create thread/process-safe logger stuff
    queue = Queue(-1)
    q_handler = QueueHandler(queue)
    logger = logging.getLogger(__name__)
    logger.addHandler(q_handler)
    q_listener = QueueListener(queue)
    q_listener.start()

    # Create generators that we can use to create numpy arrays without wasting
    # memory (even though this requires reading the file multiple times).
    # Do this using a process pool so that we can clear out the temporary
    # variables more easily and do these in parallel.
    if MAX_CONCURRENT_PROCESSES == 1:
        ids = _ids_for_iter_type(example_iter_type, path, ids_to_floats)
        classes = _classes_for_iter_type(example_iter_type, path, label_col,
                                         class_map)
        features, feat_vectorizer = _features_for_iter_type(example_iter_type,
                                                            path, quiet,
                                                            sparse, label_col,
                                                            feature_hasher,
                                                            num_features)
    else:
        pool = MemmapingPool(min(3, MAX_CONCURRENT_PROCESSES))

        ids_result = pool.apply_async(_ids_for_iter_type,
                                      args=(example_iter_type, path,
                                            ids_to_floats))
        classes_result = pool.apply_async(_classes_for_iter_type,
                                          args=(example_iter_type, path,
                                                label_col, class_map))
        features_result = pool.apply_async(_features_for_iter_type,
                                           args=(example_iter_type, path,
                                                 quiet, sparse, label_col,
                                                 feature_hasher, num_features))

        # Wait for processes to complete and store results
        pool.close()
        pool.join()
        ids = ids_result.get()
        classes = classes_result.get()
        features, feat_vectorizer = features_result.get()
        # Need to call terminate to clear up temporary directory
        pool.terminate()

    # Tear-down thread/process-safe logging and switch back to regular
    q_listener.stop()
    logger.removeHandler(q_handler)

    # Make sure we have the same number of ids, classes, and features
    assert ids.shape[0] == classes.shape[0] == features.shape[0]

    return ExamplesTuple(ids, classes, features, feat_vectorizer)


def convert_examples(example_dicts, sparse=True, ids_to_floats=False):
    '''
    This function is to facilitate programmatic use of Learner.predict()
    and other functions that take ExamplesTuple objects as input.
    It converts a .jsonlines/.ndj-style list of dictionaries into
    an ExamplesTuple.

    :param example_dicts: An list of dictionaries following the .jsonlines/.ndj
                          format (i.e., features 'x', label 'y', and 'id').
    :type example_dicts: iterable of dicts

    :returns: an ExamplesTuple representing the examples in example_dicts.
    '''
    return load_examples(example_dicts, sparse=sparse, quiet=True,
                         ids_to_floats=ids_to_floats)


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


def _safe_float(text, replace_dict=None):
    '''
    Attempts to convert a string to a float, but if that's not possible, just
    returns the original value.

    :param text: The text to convert.
    :type text: str
    :param replace_dict: Mapping from text to replacement text values. This is
                         mainly used for collapsing multiple classes into a
                         single class. Replacing happens before conversion to
                         floats. Anything not in the mapping will be kept the
                         same.
    :type replace_dict: dict from str to str
    '''
    if replace_dict is not None:
        if text in replace_dict:
            text = replace_dict[text]
        else:
            logging.getLogger(__name__).warning('Encountered value that was '
                                                'not in replacement '
                                                'dictionary (e.g., class_map):'
                                                ' {}'.format(text))
    try:
        return float(text)
    except ValueError:
        return text.decode('utf-8') if PY2 else text
    except TypeError:
        return 0.0


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
    if sys.version_info >= (3, 0):
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
    if sys.version_info >= (3, 0):
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
                print('{}'.format(ex_id), end='', file=f)
            print(' |', end=' ', file=f)
            if PY2 and class_name is not None and isinstance(class_name,
                                                             text_type):
                class_name = class_name.encode('utf-8')
            if class_name in label_map:
                print('{}={}'.format(label_map[class_name],
                                          class_name),
                      end=' | ', file=f)
            else:
                print(' |', end=' ', file=f)
            print(' '.join(('{}={}'.format(feat_vectorizer.vocabulary_[field]
                                           + 1, field) for field, value in
                           feature_dict.items() if Decimal(value) != 0)),
                  file=f)


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

