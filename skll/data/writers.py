# License: BSD 3 clause
'''
Handles loading data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
'''

from __future__ import absolute_import, print_function, unicode_literals

import json
import logging
import os
import re
import sys
from csv import DictWriter
from decimal import Decimal
from io import open
from warnings import warn

import numpy as np
from six import iteritems, PY2, string_types, text_type
from six.moves import map, zip
from sklearn.feature_extraction import FeatureHasher

from skll.data import FeatureSet


class FeatureSetWriter(object):

    """
    Helper class for writing out FeatureSets to files.

    :param path: A path to the feature file we would like to create. The suffix
                 to this filename must be ``.arff``, ``.csv``, ``.jsonlines``,
                 ``.libsvm``, ``.megam``, ``.ndj``, or ``.tsv``. If ``subsets``
                 is not ``None``, path is assumed to be a string containing the
                 path to the directory to write the feature files with an
                 additional file extension specifying the file type. For
                 example ``/foo/.csv``.
    :type path: str
    :param feature_set: The FeatureSet to dump a file.
    :type feature_set: FeatureSet
    :param requires_binary: Whether or not the FeatureSetWriter must open the
                            file in binary mode for writing with Python 2.
    :type requires_binary: bool
    :param quiet: Do not print "Writing..." status message to stderr.
    :type quiet: bool
    """

    def __init__(self, path, feature_set, **kwargs):
        super(FeatureSetWriter, self).__init__()
        self.requires_binary = kwargs.pop('requires_binary', False)
        self.quiet = kwargs.pop('quiet', True)
        self.path = path
        self.feat_set = feature_set
        # Get prefix & extension for checking file types & writing subset files
        # TODO: Determine if we purposefully used this instead of os.path.split
        self.root, self.ext = re.search(r'^(.*)(\.[^.]*)$', path).groups()
        self._progress_msg = ''
        if kwargs:
            raise ValueError('Passed extra keyword arguments to '
                             'FeatureSetWriter constructor: {}'.format(kwargs))

    def write(self, subsets=None):
        '''
        Writes out this FeatureSetWriter's FeatureSet to a file in its
        format.

        :param subsets: A mapping from subset names to lists of feature names
                        that are included in those sets. If given, a feature
                        file will be written for every subset (with the name
                        containing the subset name as suffix to ``path``).
                        Note, since string- valued features are automatically
                        converted into boolean features with names of the form
                        ``FEATURE_NAME=STRING_VALUE``, when doing the
                        filtering, the portion before the ``=`` is all that's
                        used for matching. Therefore, you do not need to
                        enumerate all of these boolean feature names in your
                        mapping.
        :type subsets: dict (str to list of str)
        '''
        # Setup logger
        logger = logging.getLogger(__name__)

        if isinstance(self.feat_set.vectorizer, FeatureHasher):
            raise ValueError('FeatureSetWriter cannot write sets that use'
                             'FeatureHasher for vectorization.')

        # Write one feature file if we weren't given a dict of subsets
        if subsets is None:
            self._write_subset(self.path, None)
        # Otherwise write one feature file per subset
        else:
            for subset_name, filter_features in iteritems(subsets):
                logger.debug('Subset (%s) features: %s', subset_name,
                             filter_features)
                sub_path = os.path.join(self.root, '{}{}'.format(subset_name,
                                                                 self.ext))
                self._write_subset(sub_path, set(filter_features))

    def _write_subset(self, sub_path, filter_features):
        '''
        Writes out the given FeatureSet to a file in this class's format.

        :param filter_features: Set of features to include in current feature
                                file.
        :type filter_features: set of str
        '''
        # Setup logger
        logger = logging.getLogger(__name__)

        logger.debug('sub_path: %s', sub_path)
        logger.debug('feature_set: %s', self.feat_set.name)
        logger.debug('filter_features: %s', filter_features)

        if not self.quiet:
            self._progress_msg = "Writing {}...".format(sub_path)
            print(self._progress_msg, end="\r", file=sys.stderr)
            sys.stderr.flush()

        # Apply filtering
        filtered_set = (self.feat_set.filtered_iter(features=filter_features)
                        if filter_features is not None else self.feat_set)

        # Open file for writing and write each line
        file_mode = 'wb' if (self.requires_binary and PY2) else 'w'
        with open(sub_path, file_mode) as output_file:
            # Write out the header if this format requires it
            self._write_header(filtered_set, output_file)
            # Write individual lines
            for ex_num, (id_, class_, feat_dict) in enumerate(filtered_set):
                self._write_line(id_, class_, feat_dict, output_file)
                if not self.quiet and ex_num % 100 == 0:
                    print("{}{:>15}".format(self._progress_msg, ex_num),
                          end="\r", file=sys.stderr)
                    sys.stderr.flush()
            if not self.quiet:
                print("{}{:<15}".format(self._progress_msg, "done"),
                      file=sys.stderr)
                sys.stderr.flush()

    def _write_header(self, feature_set, output_file):
        '''
        Called before lines are written to file, so that headers can be written
        for files that need them.
        '''
        pass

    def _write_line(self, id_, class_, feat_dict, output_file):
        '''
        Write the current line in the file in this FeatureSetWriter's format.
        '''
        raise NotImplementedError


class DelimitedFileWriter(FeatureSetWriter):

    """
    FeatureSetWriter for writing out FeatureSets as TSV/CSV files.

    :param path: A path to the feature file we would like to create.
                 If ``subsets`` is not ``None``, this is assumed to be a string
                 containing the path to the directory to write the feature
                 files with an additional file extension specifying the file
                 type. For example ``/foo/.csv``.
    :type path: str
    :param feature_set: The FeatureSet to dump to a file.
    :type feature_set: FeatureSet
    :param quiet: Do not print "Writing..." status message to stderr.
    :type quiet: bool
    :param label_col: Name of the column which contains the class labels for
                      CSV/TSV files. If no column with that name exists, or
                      `None` is specified, the data is considered to be
                      unlabelled.
    :type label_col: str
    :param dialect: The dialect to use for the underlying ``csv.DictWriter``
                    Default: 'excel-tab'
    :type dialect: str
    """

    def __init__(self, path, feature_set, **kwargs):
        kwargs['requires_binary'] = True
        self.dialect = kwargs.pop('dialect', 'excel-tab')
        self.label_col = kwargs.pop('label_col', 'y')
        super(DelimitedFileWriter, self).__init__(path, feature_set, **kwargs)
        self._dict_writer = None

    def _get_fieldnames(self):
        '''
        Build list of fieldnames for DictWriter from self.feat_set.
        '''
        # Build list of fieldnames (features + 'id' + label_col)
        fieldnames = set(self.feat_set.vectorizer.get_feature_names())
        if self.feat_set.has_ids:
            fieldnames.add('id')
        if self.feat_set.has_classes:
            fieldnames.add(self.label_col)
        return sorted(fieldnames)

    def _write_header(self, feature_set, output_file):
        '''
        Called before lines are written to file, so that headers can be written
        for files that need them.
        '''
        # Initialize DictWriter that will be used to write header and rows
        self._dict_writer = DictWriter(output_file,
                                       self._get_fieldnames(),
                                       restval=0, dialect=self.dialect)
        # Actually output the header to the file
        self._dict_writer.writeheader()

    def _write_line(self, id_, class_, feat_dict, output_file):
        '''
        Write the current line in the file in this FeatureSetWriter's format.
        '''
        # Add class column to feat_dict (unless this is unlabelled data)
        if self.label_col not in feat_dict:
            if self.feat_set.has_classes:
                feat_dict[self.label_col] = class_
        else:
            raise ValueError(('Class column name "{}" already used as feature '
                              'name.').format(self.label_col))
        # Add id column to feat_dict if id is provided
        if 'id' not in feat_dict:
            if self.feat_set.has_ids:
                feat_dict['id'] = id_
        else:
            raise ValueError('ID column name "id" already used as feature '
                             'name.')
        # Write out line
        self._dict_writer.writerow(feat_dict)


class CSVWriter(DelimitedFileWriter):

    """
    FeatureSetWriter for writing out FeatureSets as TSV files.

    :param path: A path to the feature file we would like to create.
                 If ``subsets`` is not ``None``, this is assumed to be a string
                 containing the path to the directory to write the feature
                 files with an additional file extension specifying the file
                 type. For example ``/foo/.csv``.
    :type path: str
    :param feature_set: The FeatureSet to dump a file.
    :type feature_set: FeatureSet
    :param quiet: Do not print "Writing..." status message to stderr.
    :type quiet: bool
    """

    def __init__(self, path, feature_set, **kwargs):
        kwargs['dialect'] = 'excel'
        super(CSVWriter, self).__init__(path, feature_set, **kwargs)
        self._dict_writer = None


class TSVWriter(DelimitedFileWriter):

    """
    FeatureSetWriter for writing out FeatureSets as TSV files.

    :param path: A path to the feature file we would like to create.
                 If ``subsets`` is not ``None``, this is assumed to be a string
                 containing the path to the directory to write the feature
                 files with an additional file extension specifying the file
                 type. For example ``/foo/.csv``.
    :type path: str
    :param feature_set: The FeatureSet to dump a file.
    :type feature_set: FeatureSet
    :param quiet: Do not print "Writing..." status message to stderr.
    :type quiet: bool
    """

    def __init__(self, path, feature_set, **kwargs):
        kwargs['dialect'] = 'excel-tab'
        super(TSVWriter, self).__init__(path, feature_set, **kwargs)
        self._dict_writer = None


class ARFFWriter(DelimitedFileWriter):

    """
    FeatureSetWriter for writing out FeatureSets as ARFF files.

    :param path: A path to the feature file we would like to create. If
                 ``subsets`` is not ``None``, this is assumed to be a string
                 containing the path to the directory to write the feature
                 files with an additional file extension specifying the file
                 type. For example ``/foo/.csv``.
    :type path: str
    :param feature_set: The FeatureSet to dump a file.
    :type feature_set: FeatureSet
    :param requires_binary: Whether or not the FeatureSetWriter must open the
                            file in binary mode for writing with Python 2.
    :type requires_binary: bool
    :param quiet: Do not print "Writing..." status message to stderr.
    :type quiet: bool
    :param relation: The name of the relation in the ARFF file.
                     Default: ``'skll_relation'``
    :type relation: str
    :param regression: Is this an ARFF file to be used for regression?
                       Default: ``False``
    :type regression: bool
    """

    def __init__(self, path, feature_set, **kwargs):
        self.relation = kwargs.pop('relation', 'skll_relation')
        self.regression = kwargs.pop('regression', False)
        kwargs['dialect'] = 'arff'
        super(ARFFWriter, self).__init__(path, feature_set, **kwargs)
        self._dict_writer = None

    def _write_header(self, feature_set, output_file):
        '''
        Called before lines are written to file, so that headers can be written
        for files that need them.
        '''
        fieldnames = self._get_fieldnames()
        if self.label_col in fieldnames:
            fieldnames.remove(self.label_col)

        # Add relation to header
        print("@relation '{}'\n".format(self.relation), file=output_file)

        # Loop through fields writing the header info for the ARFF file
        for field in fieldnames:
            print("@attribute '{}' numeric".format(field.replace('\\', '\\\\')
                                                   .replace("'", "\\'")),
                  file=output_file)

        # Print class label header if necessary
        if self.regression:
            print("@attribute {} numeric".format(self.label_col),
                  file=output_file)
        else:
            if self.feat_set.has_classes:
                print("@attribute {} ".format(self.label_col) +
                      "{" + ','.join(map(str,
                                         sorted(set(self.feat_set.classes)))) +
                      "}", file=output_file)
        fieldnames.append(self.label_col)

        # Create CSV writer to handle missing values for lines in data section
        # and to ignore the instance values for non-numeric attributes
        self._dict_writer = DictWriter(output_file, fieldnames, restval=0,
                                       extrasaction='ignore', dialect='arff')

        # Finish header and start data section
        print("\n@data", file=output_file)


class MegaMWriter(FeatureSetWriter):

    """ FeatureSetWriter for writing out FeatureSets as MegaM files. """
    @staticmethod
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
            char_list.append(
                '<U{}>'.format(char_num) if char_num > 127 else char)
        return ''.join(char_list)

    def _write_line(self, id_, class_, feat_dict, output_file):
        '''
        Write the current line in the file in MegaM format.
        '''
        # Don't try to add class column if this is label-less data
        if self.feat_set.has_ids:
            print('# {}'.format(id_), file=output_file)
        if self.feat_set.has_classes:
            print(class_, end='\t', file=output_file)
        print(self._replace_non_ascii(' '.join(('{} {}'.format(field,
                                                               value) for
                                                field, value in
                                                sorted(feat_dict.items()) if
                                                Decimal(value) != 0))),
              file=output_file)


class NDJWriter(FeatureSetWriter):

    """
    FeatureSetWriter for writing out FeatureSets as .jsonlines/.ndj files.
    """

    def __init__(self, path, feature_set, **kwargs):
        kwargs['requires_binary'] = True
        super(NDJWriter, self).__init__(path, feature_set, **kwargs)

    def _write_line(self, id_, class_, feat_dict, output_file):
        '''
        Write the current line in the file in MegaM format.
        '''
        example_dict = {}
        # Don't try to add class column if this is label-less data
        if self.feat_set.has_classes:
            example_dict['y'] = np.asscalar(class_)
        if self.feat_set.has_ids:
            example_dict['id'] = np.asscalar(id_)
        example_dict["x"] = feat_dict
        print(json.dumps(example_dict, sort_keys=True), file=output_file)


class LibSVMWriter(FeatureSetWriter):

    """
    FeatureSetWriter for writing out FeatureSets as LibSVM/SVMLight files.
    """

    LIBSVM_REPLACE_DICT = {':': '\u2236',
                           '#': '\uFF03',
                           ' ': '\u2002',
                           '=': '\ua78a',
                           '|': '\u2223'}

    def __init__(self, path, feature_set, **kwargs):
        self.label_map = kwargs.pop('label_map', None)
        super(LibSVMWriter, self).__init__(path, feature_set, **kwargs)
        if self.label_map is None:
            self.label_map = {}
            if feature_set.has_classes:
                self.label_map = {label: num for num, label in
                                  enumerate(sorted({label for label in
                                                    feature_set.classes if
                                                    not isinstance(label,
                                                                   (int,
                                                                    float))}))}
            # Add fake item to vectorizer for None
            self.label_map[None] = '00000'

    @staticmethod
    def _sanitize(name):
        '''
        Replace illegal characters in class names with close unicode
        equivalents to make things loadable in by LibSVM, LibLinear, or
        SVMLight.
        '''
        if isinstance(name, string_types):
            for orig, replacement in LibSVMWriter.LIBSVM_REPLACE_DICT.items():
                name = name.replace(orig, replacement)
        return name

    def _write_line(self, id_, class_, feat_dict, output_file):
        '''
        Write the current line in the file in MegaM format.
        '''
        field_values = sorted([(self.feat_set.vectorizer.vocabulary_[field] +
                                1, value) for field, value in
                               iteritems(feat_dict) if Decimal(value) != 0])
        # Print label
        if class_ in self.label_map:
            print('{}'.format(self.label_map[class_]), end=' ',
                  file=output_file)
        else:
            print('{}'.format(class_), end=' ', file=output_file)
        # Print features
        print(' '.join(('{}:{}'.format(field, value) for field, value in
                        field_values)), end=' ', file=output_file)
        # Print comment with id and mappings
        print('#', end=' ', file=output_file)
        if self.feat_set.has_ids:
            print(self._sanitize('{}'.format(id_)), end='',
                  file=output_file)
        print(' |', end=' ', file=output_file)
        if (PY2 and self.feat_set.has_classes and isinstance(class_,
                                                             text_type)):
            class_ = class_.encode('utf-8')
        if class_ in self.label_map:
            print('%s=%s' % (self._sanitize(self.label_map[class_]),
                             self._sanitize(class_)),
                  end=' | ', file=output_file)
        else:
            print(' |', end=' ', file=output_file)
        line = ' '.join(('%s=%s' % (self.feat_set.vectorizer.vocabulary_[field]
                                    + 1, self._sanitize(field)) for
                         field, value in feat_dict.items() if
                         Decimal(value) != 0))
        print(line, file=output_file)


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
    # logger.debug('Features: %s', features)

    warn('The write_feature_file function will be removed in SKLL 1.0.0. '
         'Please switch to using a Writer (e.g., NDJWriter) directly.',
         DeprecationWarning)

    feature_set = FeatureSet(path, ids=ids, classes=classes, features=features,
                             vectorizer=feat_vectorizer)

    writer_args = {}

    # Get lowercase extension for file extension checking
    ext = '.' + path.rsplit('.', 1)[-1].lower()

    if ext not in EXT_TO_WRITER:
        raise ValueError(('Output file must be in either .arff, .csv, '
                          '.jsonlines, .libsvm, .megam, .ndj, or .tsv format. '
                          'You specified: {}').format(path))
    else:
        writer_type = EXT_TO_WRITER[ext]

    # Setup writer-type-specific arguments
    if issubclass(writer_type, DelimitedFileWriter):
        writer_args['label_col'] = label_col
        if writer_type is ARFFWriter:
            writer_args['relation'] = arff_relation
            writer_args['regression'] = arff_regression
    elif issubclass(writer_type, LibSVMWriter):
        writer_args['label_map'] = label_map

    writer = writer_type(path, feature_set, **writer_args)
    writer.write(subsets=subsets)


# Constants
EXT_TO_WRITER = {".arff": ARFFWriter,
                 ".csv": CSVWriter,
                 ".jsonlines": NDJWriter,
                 ".libsvm": LibSVMWriter,
                 ".megam": MegaMWriter,
                 '.ndj': NDJWriter,
                 ".tsv": TSVWriter}
