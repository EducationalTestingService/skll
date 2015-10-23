# License: BSD 3 clause
"""
Handles loading data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import json
import logging
import re
import sys
from csv import DictReader
from itertools import chain, islice
from io import open, BytesIO, StringIO

import numpy as np
from bs4 import UnicodeDammit
from six import iteritems, PY2, PY3, string_types, text_type
from six.moves import map, zip
from sklearn.feature_extraction import FeatureHasher

from skll.data import FeatureSet
from skll.data.dict_vectorizer import DictVectorizer


class Reader(object):

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
    :param id_col: Name of the column which contains the instance IDs for
                   ARFF/CSV/TSV files. If no column with that name exists, or
                   `None` is specified, the IDs will be generated
                   automatically.
    :type id_col: str
    :param label_col: Name of the column which contains the class labels
                      for ARFF/CSV/TSV files. If no column with that name
                      exists, or `None` is specified, the data is
                      considered to be unlabelled.
    :type label_col: str
    :param class_map: Mapping from original class labels to new ones. This is
                      mainly used for collapsing multiple labels into a single
                      class. Anything not in the mapping will be kept the same.
    :type class_map: dict from str to str
    :param sparse: Whether or not to store the features in a numpy CSR
                   matrix when using a DictVectorizer to vectorize the
                   features.
    :type sparse: bool
    :param feature_hasher: Whether or not a FeatureHasher should be used to
                           vectorize the features.
    :type feature_hasher: bool
    :param num_features: If using a FeatureHasher, how many features should the
                         resulting matrix have?  You should set this to a power
                         of 2 greater than the actual number of features to
                         avoid collisions.
    :type num_features: int
    """

    def __init__(self, path_or_list, quiet=True, ids_to_floats=False,
                 label_col='y', id_col='id', class_map=None, sparse=True,
                 feature_hasher=False, num_features=None):
        super(Reader, self).__init__()
        self.path_or_list = path_or_list
        self.quiet = quiet
        self.ids_to_floats = ids_to_floats
        self.label_col = label_col
        self.id_col = id_col
        self.class_map = class_map
        self._progress_msg = ''
        if feature_hasher:
            self.vectorizer = FeatureHasher(n_features=num_features)
        else:
            self.vectorizer = DictVectorizer(sparse=sparse)

    @classmethod
    def for_path(cls, path_or_list, **kwargs):
        """
        :param path: The path to the file to load the examples from, or a list
                     of example dictionaries.
        :type path: str or dict
        :param quiet: Do not print "Loading..." status message to stderr.
        :type quiet: bool
        :param sparse: Whether or not to store the features in a numpy CSR
                       matrix.
        :type sparse: bool
        :param id_col: Name of the column which contains the instance IDs for
                       ARFF/CSV/TSV files. If no column with that name exists,
                       or `None` is specified, the IDs will be generated
                       automatically.
        :type id_col: str
        :param label_col: Name of the column which contains the class labels
                          for ARFF/CSV/TSV files. If no column with that name
                          exists, or `None` is specified, the data is
                          considered to be unlabelled.
        :type label_col: str
        :param ids_to_floats: Convert IDs to float to save memory. Will raise
                              error if we encounter an a non-numeric ID.
        :type ids_to_floats: bool
        :param class_map: Mapping from original class labels to new ones. This
                          is mainly used for collapsing multiple classes into a
                          single class. Anything not in the mapping will be
                          kept the same.
        :type class_map: dict from str to str

        :returns: New instance of the :class:`Reader` sub-class that is
                  appropriate for the given path, or :class:`DictListReader` if
                  given a list of dictionaries.
        """
        if not isinstance(path_or_list, string_types):
            return DictListReader(path_or_list)
        else:
            # Get lowercase extension for file extension checking
            ext = '.' + path_or_list.rsplit('.', 1)[-1].lower()
            if ext not in EXT_TO_READER:
                raise ValueError(('Example files must be in either .arff, '
                                  '.csv, .jsonlines, .megam, .ndj, or .tsv '
                                  'format. You specified: '
                                  '{}').format(path_or_list))
        return EXT_TO_READER[ext](path_or_list, **kwargs)

    def _sub_read(self, f):
        """
        Does the actual reading of the given file or list.

        :param f: An open file to iterate through
        :type f: file
        """
        raise NotImplementedError

    def _print_progress(self, progress_num, end="\r"):
        """
        Little helper to print out progress numbers in proper format.

        Nothing gets printed if ``self.quiet`` is ``True``.

        :param progress_num: Progress indicator value.  Usually either a line
                             number or a percentage.
        :type progress_num: anything that can be converted to str
        :param end: The string to put at the end of the line.  "\\r" should be
                    used for every update except for the final one.
        :type end: str
        """
        # Print out status
        if not self.quiet:
            print("{}{:>15}".format(self._progress_msg, progress_num),
                  end=end, file=sys.stderr)
            sys.stderr.flush()

    def read(self):
        """
        Loads examples in the ``.arff``, ``.csv``, ``.jsonlines``, ``.libsvm``,
        ``.megam``, ``.ndj``, or ``.tsv`` formats.

        :returns: :class:`~skll.data.featureset.FeatureSet` representing the
                  file we read in.
        """
        # Setup logger
        logger = logging.getLogger(__name__)

        logger.debug('Path: %s', self.path_or_list)

        if not self.quiet:
            self._progress_msg = "Loading {}...".format(self.path_or_list)
            print(self._progress_msg, end="\r", file=sys.stderr)
            sys.stderr.flush()

        # Get labels and IDs
        ids = []
        labels = []
        with open(self.path_or_list, 'r' if PY3 else 'rb') as f:
            for ex_num, (id_, class_, _) in enumerate(self._sub_read(f), start=1):
                # Update lists of IDs, clases, and features
                if self.ids_to_floats:
                    try:
                        id_ = float(id_)
                    except ValueError:
                        raise ValueError(('You set ids_to_floats to true,'
                                          ' but ID {} could not be '
                                          'converted to float in '
                                          '{}').format(id_,
                                                       self.path_or_list))
                ids.append(id_)
                labels.append(class_)
                if ex_num % 100 == 0:
                    self._print_progress(ex_num)
            self._print_progress(ex_num)

        # Remember total number of examples for percentage progress meter
        total = ex_num

        # Convert everything to numpy arrays
        ids = np.array(ids)
        labels = np.array(labels)

        def feat_dict_generator():
            with open(self.path_or_list, 'r' if PY3 else 'rb') as f:
                for ex_num, (_, _, feat_dict) in enumerate(self._sub_read(f)):
                    yield feat_dict
                    if ex_num % 100 == 0:
                        self._print_progress('{:.8}%'.format(100 * ((ex_num /
                                                                    total))))
                self._print_progress("100%")

        # Convert everything to numpy arrays
        features = self.vectorizer.fit_transform(feat_dict_generator())

        # Report that loading is complete
        self._print_progress("done", end="\n")

        # Make sure we have the same number of ids, labels, and features
        assert ids.shape[0] == labels.shape[0] == features.shape[0]

        if ids.shape[0] != len(set(ids)):
            raise ValueError('The example IDs are not unique in %s.' %
                             self.path_or_list)

        return FeatureSet(self.path_or_list, ids, labels=labels,
                          features=features, vectorizer=self.vectorizer)


class DictListReader(Reader):

    """
    This class is to facilitate programmatic use of
    :meth:`~skll.learner.Learner.predict` and other functions that take
    :class:`~skll.data.featureset.FeatureSet` objects as input. It iterates
    over examples in the same way as other :class:`Reader` clases, but uses a
    list of example dictionaries instead of a path to a file.

    :param path_or_list: List of example dictionaries.
    :type path_or_list: Iterable of dict
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    :param ids_to_floats: Convert IDs to float to save memory. Will raise error
                          if we encounter an a non-numeric ID.
    :type ids_to_floats: bool
    """

    def read(self):
        ids = []
        labels = []
        feat_dicts = []
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
            class_name = (safe_float(example['y'],
                                     replace_dict=self.class_map)
                          if 'y' in example else None)
            example = example['x']
            # Update lists of IDs, labels, and feature dictionaries
            if self.ids_to_floats:
                try:
                    curr_id = float(curr_id)
                except ValueError:
                    raise ValueError(('You set ids_to_floats to true, but ID '
                                      '{} could not be converted to float in '
                                      '{}').format(curr_id, self.path_or_list))
            ids.append(curr_id)
            labels.append(class_name)
            feat_dicts.append(example)
            # Print out status
            if example_num % 100 == 0:
                self._print_progress(example_num)
        # Convert lists to numpy arrays
        ids = np.array(ids)
        labels = np.array(labels)
        features = self.vectorizer.fit_transform(feat_dicts)

        return FeatureSet('converted', ids, labels=labels,
                          features=features, vectorizer=self.vectorizer)


class NDJReader(Reader):

    """
    Reader to create a :class:`~skll.data.featureset.FeatureSet` out of a
    .jsonlines/.ndj file

    If you would like to include example/instance IDs in your files, they
    must be specified in the following ways as an "id" key in each JSON
    dictionary.
    """

    def _sub_read(self, f):
        for example_num, line in enumerate(f):
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
            class_name = (safe_float(example['y'],
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


class MegaMReader(Reader):

    """
    Reader to create a :class:`~skll.data.featureset.FeatureSet` out ouf a
    MegaM -fvals file.

    If you would like to include example/instance IDs in your files, they
    must be specified as a comment line directly preceding the line with
    feature values.
    """

    def _sub_read(self, f):
        example_num = 0
        curr_id = 'EXAMPLE_0'
        for line in f:
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
                    class_name = safe_float(split_line[0],
                                            replace_dict=self.class_map)
                    field_pairs = []
                # Line has a class label and feature-value pairs
                elif num_cols % 2 == 1:
                    class_name = safe_float(split_line[0],
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
                    field_values = (safe_float(val) for val in
                                    islice(field_pairs, 1, None, 2))

                    # Add the feature-value pairs to dictionary
                    curr_info_dict.update(zip(field_names, field_values))

                    if len(curr_info_dict) != len(field_pairs) / 2:
                        raise ValueError(('There are duplicate feature ' +
                                          'names in {} for example ' +
                                          '{}.').format(self.path_or_list,
                                                        curr_id))

                yield curr_id, class_name, curr_info_dict

                # Set default example ID for next instance, in case we see a
                # line without an ID.
                example_num += 1
                curr_id = 'EXAMPLE_{}'.format(example_num)


class LibSVMReader(Reader):

    """
    Reader to create a :class:`~skll.data.featureset.FeatureSet` out ouf a
    LibSVM/LibLinear/SVMLight file.

    We use a specially formatted comment for storing example IDs, class names,
    and feature names, which are normally not supported by the format.  The
    comment is not mandatory, but without it, your labels and features will
    not have names.  The comment is structured as follows::

        ExampleID | 1=FirstClass | 1=FirstFeature 2=SecondFeature
    """

    line_regex = re.compile(r'^(?P<label_num>[^ ]+)\s+(?P<features>[^#]*)\s*'
                            r'(?P<comments>#\s*(?P<example_id>[^|]+)\s*\|\s*'
                            r'(?P<label_map>[^|]+)\s*\|\s*'
                            r'(?P<feat_map>.*)\s*)?$', flags=re.UNICODE)

    LIBSVM_REPLACE_DICT = {'\u2236': ':',
                           '\uFF03': '#',
                           '\u2002': ' ',
                           '\ua78a': '=',
                           '\u2223': '|'}

    @staticmethod
    def _pair_to_tuple(pair, feat_map):
        """
        Split a feature-value pair separated by a colon into a tuple.  Also
        do safe_float conversion on the value.
        """
        name, value = pair.split(':')
        if feat_map is not None:
            name = feat_map[name]
        value = safe_float(value)
        return (name, value)

    def _sub_read(self, f):
        for example_num, line in enumerate(f):
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
                    feat_map = {}
                    for pair in match.group('feat_map').split():
                        number, name = pair.split('=')
                        for orig, replacement in \
                                LibSVMReader.LIBSVM_REPLACE_DICT.items():
                            name = name.replace(orig, replacement)
                        feat_map[number] = name
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
            if label_map:
                class_name = label_map[class_num]
            else:
                class_name = class_num
            class_name = safe_float(class_name,
                                    replace_dict=self.class_map)

            curr_info_dict = dict(self._pair_to_tuple(pair, feat_map) for pair
                                  in match.group('features').strip().split())

            yield curr_id, class_name, curr_info_dict


class DelimitedReader(Reader):

    """
    Reader for creating a :class:`~skll.data.featureset.FeatureSet` out of a
    delimited (CSV/TSV) file.

    If you would like to include example/instance IDs in your files, they
    must be specified as an ``id`` column.

    Also, for ARFF, CSV, and TSV files, there must be a column with the
    name specified by `label_col` if the data is labelled. For ARFF files,
    this column must also be the final one (as it is in Weka).

    :param dialect: The dialect of to pass on to the underlying CSV reader.
                    Default: ``excel-tab``
    :type dialect: str
    """

    def __init__(self, path_or_list, **kwargs):
        self.dialect = kwargs.pop('dialect', 'excel-tab')
        super(DelimitedReader, self).__init__(path_or_list, **kwargs)

    def _sub_read(self, f):
        reader = DictReader(f, dialect=self.dialect)
        for example_num, row in enumerate(reader):
            if self.label_col is not None and self.label_col in row:
                class_name = safe_float(row[self.label_col],
                                        replace_dict=self.class_map)
                del row[self.label_col]
            else:
                class_name = None

            if self.id_col not in row:
                curr_id = "EXAMPLE_{}".format(example_num)
            else:
                curr_id = row[self.id_col]
                del row[self.id_col]

            # Convert features to floats and if a feature is 0
            # then store the name of the feature so we can
            # delete it later since we don't need to explicitly
            # store zeros in the feature hash
            columns_to_delete = []
            if PY2:
                columns_to_convert_to_unicode = []
            for fname, fval in iteritems(row):
                fval_float = safe_float(fval)
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
                if not self.ids_to_floats:
                    curr_id = curr_id.decode('utf-8')

            yield curr_id, class_name, row


class CSVReader(DelimitedReader):

    """
    Reader for creating a :class:`~skll.data.featureset.FeatureSet` out of a
    CSV file.

    If you would like to include example/instance IDs in your files, they
    must be specified as an "id" column.

    Also, there must be a column with the name specified by `label_col` if the
    data is labelled.
    """

    def __init__(self, path_or_list, **kwargs):
        kwargs['dialect'] = 'excel'
        super(CSVReader, self).__init__(path_or_list, **kwargs)


class ARFFReader(DelimitedReader):

    """
    Reader for creating a :class:`~skll.data.featureset.FeatureSet` out of an
    ARFF file.

    If you would like to include example/instance IDs in your files, they
    must be specified as an "id" column.

    Also, there must be a column with the name specified by `label_col` if the
    data is labelled, and this column must be the final one (as it is in Weka).
    """

    def __init__(self, path_or_list, **kwargs):
        kwargs['dialect'] = 'arff'
        super(ARFFReader, self).__init__(path_or_list, **kwargs)
        self.relation = ''
        self.regression = False

    @staticmethod
    def split_with_quotes(s, delimiter=' ', quote_char="'", escape_char='\\'):
        """
        A replacement for string.split that won't split delimiters enclosed in
        quotes.
        """
        if PY2:
            delimiter = delimiter.encode()
            quote_char = quote_char.encode()
            escape_char = escape_char.encode()
        return next(csv.reader([s], delimiter=delimiter, quotechar=quote_char,
                               escapechar=escape_char))

    def _sub_read(self, f):
        field_names = []
        # Process ARFF header
        for line in f:
            # Process encoding
            if not isinstance(line, text_type):
                decoded_line = UnicodeDammit(line,
                                             ['utf-8',
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
                    field_name = split_header[1]
                    field_names.append(field_name)
                    # Check if we're doing regression
                    if field_name == self.label_col:
                        self.regression = (len(split_header) > 2 and
                                           split_header[2] == 'numeric')
                # Save relation if specified
                elif row_type == '@relation':
                    self.relation = split_header[1]
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
        return super(ARFFReader, self)._sub_read(chain([field_str], f))


class TSVReader(DelimitedReader):

    """
    Reader for creating a :class:`~skll.data.featureset.FeatureSet` out of a
    TSV file.

    If you would like to include example/instance IDs in your files, they
    must be specified as an "id" column.

    Also there must be a column with the name specified by `label_col` if the
    data is labelled.
    """

    def __init__(self, path_or_list, **kwargs):
        kwargs['dialect'] = 'excel-tab'
        super(TSVReader, self).__init__(path_or_list, **kwargs)


def safe_float(text, replace_dict=None):
    """
    Attempts to convert a string to an int, and then a float, but if neither is
    possible, just returns the original string value.

    :param text: The text to convert.
    :type text: str
    :param replace_dict: Mapping from text to replacement text values. This is
                         mainly used for collapsing multiple labels into a
                         single class. Replacing happens before conversion to
                         floats. Anything not in the mapping will be kept the
                         same.
    :type replace_dict: dict from str to str
    """

    # convert to text to be "Safe"!
    text = text_type(text)

    if replace_dict is not None:
        if text in replace_dict:
            text = replace_dict[text]
        else:
            logging.getLogger(__name__).warning('Encountered value that was '
                                                'not in replacement '
                                                'dictionary (e.g., class_map):'
                                                ' {}'.format(text))
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text.decode('utf-8') if PY2 else text
        except TypeError:
            return 0.0
    except TypeError:
        return 0


# Constants
EXT_TO_READER = {".arff": ARFFReader,
                 ".csv": CSVReader,
                 ".jsonlines": NDJReader,
                 ".libsvm": LibSVMReader,
                 ".megam": MegaMReader,
                 '.ndj': NDJReader,
                 ".tsv": TSVReader}
