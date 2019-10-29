# License: BSD 3 clause
"""
Handles loading data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:organization: ETS
"""

import json
import logging
import os
import re
import sys
import pandas as pd

from csv import DictWriter
from decimal import Decimal

from scipy.sparse import issparse
from sklearn.feature_extraction import FeatureHasher


class Writer(object):
    """
    Helper class for writing out FeatureSets to files on disk.

    Parameters
    ----------
    path : str
        A path to the feature file we would like to create. The suffix
        to this filename must be ``.arff``, ``.csv``, ``.jsonlines``,
        ``.libsvm``, ``.megam``, ``.ndj``, or ``.tsv``. If ``subsets``
        is not ``None``, when calling the ``write()`` method, path is
        assumed to be a string containing the path to the directory to
        write the feature files with an additional file extension
        specifying the file type. For example ``/foo/.csv``.
    feature_set : skll.FeatureSet
        The ``FeatureSet`` instance to dump to the file.
    quiet : bool
        Do not print "Writing..." status message to stderr.
        Defaults to ``True``.
    subsets : dict (str to list of str)
        A mapping from subset names to lists of feature names
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
        Defaults to ``None``.
    logger : logging.Logger
        A logger instance to use to log messages instead of creating
        a new one by default.
        Defaults to ``None``.
    """

    def __init__(self, path, feature_set, **kwargs):
        super(Writer, self).__init__()

        self.quiet = kwargs.pop('quiet', True)
        self.path = path
        self.feat_set = feature_set
        self.subsets = kwargs.pop('subsets', None)
        logger = kwargs.pop('logger', None)
        self.logger = logger if logger else logging.getLogger(__name__)

        # Get prefix & extension for checking file types & writing subset files
        # TODO: Determine if we purposefully used this instead of os.path.split
        self.root, self.ext = re.search(r'^(.*)(\.[^.]*)$', path).groups()
        self._progress_msg = ''
        self._use_pandas = False
        if kwargs:
            raise ValueError('Passed extra keyword arguments to '
                             'Writer constructor: {}'.format(kwargs))

    @classmethod
    def for_path(cls, path, feature_set, **kwargs):
        """
        Retrieve object of ``Writer`` sub-class that is
        appropriate for given path.

        Parameters
        ----------
        path : str
            A path to the feature file we would like to create. The
            suffix to this filename must be ``.arff``, ``.csv``,
            ``.jsonlines``, ``.libsvm``, ``.megam``, ``.ndj``, or
            ``.tsv``. If ``subsets`` is not ``None``, when calling the
            ``write()`` method, path is assumed to be a string
            containing the path to the directory to write the feature
            files with an additional file extension specifying the
            file type. For example ``/foo/.csv``.
        feature_set : skll.FeatureSet
            The ``FeatureSet`` instance to dump to the output file.
        kwargs : dict
            The keyword arguments for ``for_path`` are the same as
            the initializer for the desired ``Writer`` subclass.

        Returns
        -------
        writer : skll.data.writers.Writer
            New instance of the Writer sub-class that is
            appropriate for the given path.
        """
        # Get lowercase extension for file extension checking
        ext = '.' + path.rsplit('.', 1)[-1].lower()
        return EXT_TO_WRITER[ext](path, feature_set, **kwargs)

    def write(self):
        """
        Writes out this Writer's ``FeatureSet`` to a file in its
        format.
        """
        if isinstance(self.feat_set.vectorizer, FeatureHasher):
            raise ValueError('Writer cannot write sets that use a '
                             'FeatureHasher for vectorization.')

        # Write one feature file if we weren't given a dict of subsets
        if self.subsets is None:
            self._write_subset(self.path, None)

        # Otherwise write one feature file per subset
        else:
            for subset_name, filter_features in self.subsets.items():
                self.logger.debug('Subset ({}) features: {}'.format(subset_name,
                                                                    filter_features))
                sub_path = os.path.join(self.root, '{}{}'.format(subset_name,
                                                                 self.ext))
                self._write_subset(sub_path, set(filter_features))

    def _write_subset(self, sub_path, filter_features):
        """
        Writes out the given ``FeatureSet`` instance to a file in this class's format.

        Parameters
        ----------
        sub_path : str
            The path to the file we want to create for this subset
            of our data.
        filter_features : set of str
            Set of features to include in current feature file.
        """
        self.logger.debug('sub_path: %s', sub_path)
        self.logger.debug('feature_set: %s', self.feat_set.name)
        self.logger.debug('filter_features: %s', filter_features)

        if not self.quiet:
            self._progress_msg = "Writing {}...".format(sub_path)
            print(self._progress_msg, end="\r", file=sys.stderr)
            sys.stderr.flush()

        if not self._use_pandas:

            # Apply filtering
            filtered_set = (self.feat_set.filtered_iter(features=filter_features)
                            if filter_features is not None else self.feat_set)

            # Open file for writing and write each line
            with open(sub_path, 'w', encoding='utf-8') as output_file:
                # Write out the header if this format requires it
                self._write_header(filtered_set, output_file, filter_features)
                # Write individual lines
                for ex_num, (id_, label_, feat_dict) in enumerate(filtered_set):
                    self._write_line(id_, label_, feat_dict, output_file)
                    if not self.quiet and ex_num % 100 == 0:
                        print("{}{:>15}".format(self._progress_msg, ex_num),
                              end="\r", file=sys.stderr)
                        sys.stderr.flush()
        else:
            self._write_data(self.feat_set, sub_path, filter_features)

        if not self.quiet:
            print("{}{:<15}".format(self._progress_msg, "done"),
                  file=sys.stderr)
            sys.stderr.flush()

    def _write_header(self, feature_set, output_file, filter_features):
        """
        Called before lines are written to file, so that headers can be written
        for files that need them.

        Parameters
        ----------
        feature_set : skll.FeatureSet
            The ``FeatureSet`` instance being written to a file.
        output_file : file buffer
            The file being written to.
        filter_features : set of str
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.
        """
        pass

    def _write_line(self, id_, label_, feat_dict, output_file):
        """
        Write the current line in the file in this Writer's format.

        Parameters
        ----------
        id_ : str
            The ID for the current instance.
        label_ : str
            The label for the current instance.
        feat_dict : dict
            The feature dictionary for the current instance.
        output_file : file buff
             The file being written to.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def _write_data(self, feature_set, output_file, filter_features):
        """
        Write the the full data set in the Writer's format using `pandas`,
        rather than writing row-by-row.

        Parameters
        ----------
        feature_set : skll.FeatureSet
            The ``FeatureSet`` instance being written to a file.
        output_file : file buffer
            The file being written to.
        filter_features : set of str
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def _get_column_names_and_indexes(self, feature_set, filter_features=None):
        """
        Get the names of the columns and the associated
        index numbers for the (possibly filtered) features.

        Parameters
        ----------
        feature_set : skll.FeatureSet
            The ``FeatureSet`` instance being written to a file.
        filter_features : set of str
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.

        Returns
        -------
        column_names : list of str
            A list of the (possibly
            filtered) column names.
        column_indexes : list of int
            A list of the (possibly
            filtered) column indexes.
        """
        # if we're not doing filtering,
        # then just take all the feature names
        self.logger.debug(feature_set)
        if filter_features is None:
            filter_features = feature_set.vectorizer.feature_names_

        # create a list of tuples with (column names, column indexes)
        # so that we can correctly extract the appropriate columns
        columns = sorted([(col_name, col_idx) for col_name, col_idx
                          in feature_set.vectorizer.vocabulary_.items()
                          if (col_name in filter_features or
                              col_name.split('=', 1)[0] in filter_features)],
                         key=lambda x: x[1])

        # then, split the names and indexes into separate lists
        column_names, column_indexes = zip(*columns)
        return list(column_names), list(column_indexes)

    def _build_dataframe_with_features(self, feature_set, filter_features=None):
        """
        Create a data frame with the (possibly filtered) features from the current
        feature set.

        Parameters
        ----------
        feature_set : skll.FeatureSet
            The ``FeatureSet`` instance being written to a file.
        filter_features : set of str or None, optional
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.
            Defaults to None.

        Returns
        -------
        df_features : pd.DataFrame
            The data frame constructed from
            the feature set.

        Raises
        ------
        ValueError
            If ID column is already used as feature.
            If label column is already used as feature.
        """
        # if there is no filtering, then just keep all the names
        (column_names,
         column_idxs) = self._get_column_names_and_indexes(feature_set,
                                                           filter_features)

        # create the data frame from the feature set;
        # then, select only the columns that we want,
        # and give the columns their correct names
        if issparse(feature_set.features):
            df_features = pd.DataFrame(feature_set.features.todense())
        else:
            df_features = pd.DataFrame(feature_set.features)
        df_features = df_features.iloc[:, column_idxs].copy()
        df_features.columns = column_names
        return df_features

    def _build_dataframe(self, feature_set, filter_features=None, df_features=None):
        """
        Create the data frame with the (possibly filtered) features from the current
        feature set. Then, add the IDs and labels, if applicable. If the data frame
        with features already exists, pass `df_features`. Then the IDs and labels will
        simply be added to the existing data frame containing the features.

        Parameters
        ----------
        feature_set : skll.FeatureSet
            The ``FeatureSet`` instance being written to a file.
        filter_features : set of str or None, optional
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.
            Defaults to None.
        df_features : pd.DataFrame or None, optional
            If the data frame with features already exists,
            then we use it and add IDs and labels; otherwise,
            the feature data frame will be created from the feature set.
            Defaults to None.

        Returns
        -------
        df_features : pd.DataFrame
            The data frame constructed from
            the feature set.

        Raises
        ------
        ValueError
            If ID column is already used as feature.
            If label column is already used as feature.
        """
        # create the data frame with just the features
        # from the feature set, at this point
        if df_features is None:
            df_features = self._build_dataframe_with_features(feature_set,
                                                              filter_features)

        # if the id column is already in the data frame,
        # then raise an error; otherwise, just add the ids
        if self.id_col in df_features:
            raise ValueError('ID column name "{}" already used as feature '
                             'name.'.format(self.id_col))
        df_features[self.id_col] = feature_set.ids

        # if the the labels should exist but the column is already
        # in the data frame, then raise an error; otherwise, just add the labels
        if feature_set.has_labels:
            if self.label_col in df_features:
                raise ValueError(('Class column name "{}" already used as '
                                  'feature name.').format(self.label_col))
            df_features[self.label_col] = feature_set.labels

        return df_features


class CSVWriter(Writer):

    """
    Writer for writing out ``FeatureSet`` instances as CSV files.

    Parameters
    ----------
    path : str
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.csv``.
    feature_set : skll.FeatureSet
        The ``FeatureSet`` instance to dump to the output file.
    pandas_kwargs : dict or None, optional
        Arguments that will be passed directly
        to the `pandas` I/O reader.
        Defaults to None.
    kwargs : dict, optional
        The arguments to the ``Writer`` object being instantiated.
    """

    def __init__(self, path, feature_set, pandas_kwargs=None, **kwargs):
        self.label_col = kwargs.pop('label_col', 'y')
        self.id_col = kwargs.pop('id_col', 'id')
        super(CSVWriter, self).__init__(path, feature_set, **kwargs)
        self._pandas_kwargs = {} if pandas_kwargs is None else pandas_kwargs
        self._sep = self._pandas_kwargs.pop('sep', str(','))
        self._index = self._pandas_kwargs.pop('index', False)
        self._use_pandas = True

    def _write_data(self, feature_set, output_file, filter_features):
        """
        Write the data in CSV format.

        Parameters
        ----------
        feature_set : skll.FeatureSet
            The ``FeatureSet`` instance being written to a file.
        output_file : file buffer
            The file being written to.
        filter_features : set of str
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.
        """
        df = self._build_dataframe(feature_set, filter_features)
        df.to_csv(output_file, sep=self._sep, index=self._index, **self._pandas_kwargs)


class TSVWriter(CSVWriter):

    """
    Writer for writing out FeatureSets as TSV files.

    Parameters
    ----------
    path : str
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.tsv``.
    feature_set : skll.FeatureSet
        The ``FeatureSet`` instance to dump to the output file.
    pandas_kwargs : dict or None, optional
        Arguments that will be passed directly
        to the `pandas` I/O reader.
        Defaults to None.
    kwargs : dict, optional
        The arguments to the ``Writer`` object being instantiated.
    """

    def __init__(self, path, feature_set, pandas_kwargs=None, **kwargs):

        super(TSVWriter, self).__init__(path, feature_set, pandas_kwargs, **kwargs)
        self._sep = str('\t')


class ARFFWriter(Writer):

    """
    Writer for writing out FeatureSets as ARFF files.

    Parameters
    ----------
    path : str
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.arff``.
    feature_set : skll.FeatureSet
        The ``FeatureSet`` instance to dump to the output file.
    relation : str, optional
        The name of the relation in the ARFF file.
        Defaults to ``'skll_relation'``.
    regression : bool, optional
        Is this an ARFF file to be used for regression?
        Defaults to ``False``.
    kwargs : dict, optional
        The arguments to the ``Writer`` object being instantiated.
    """

    def __init__(self, path, feature_set, **kwargs):
        self.relation = kwargs.pop('relation', 'skll_relation')
        self.regression = kwargs.pop('regression', False)
        self.dialect = kwargs.pop('dialect', 'excel-tab')
        self.label_col = kwargs.pop('label_col', 'y')
        self.id_col = kwargs.pop('id_col', 'id')
        super(ARFFWriter, self).__init__(path, feature_set, **kwargs)
        self._dict_writer = None

    def _write_header(self, feature_set, output_file, filter_features):
        """
        Called before lines are written to file, so that headers can be written
        for files that need them.

        Parameters
        ----------
        feature_set : skll.FeatureSet
            The FeatureSet being written to a file.
        output_file : file buffer
            The file being written to.
        filter_features : set of str
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.
        """
        fieldnames, _ = self._get_column_names_and_indexes(self.feat_set, filter_features)
        fieldnames.append(self.id_col)

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
            if self.feat_set.has_labels:
                print("@attribute {} ".format(self.label_col) +
                      "{" + ','.join(list(map(str,
                                              sorted(set(self.feat_set.labels))))) +
                      "}", file=output_file)
        if self.label_col:
            fieldnames.append(self.label_col)

        # Create CSV writer to handle missing values for lines in data section
        # and to ignore the instance values for non-numeric attributes
        self._dict_writer = DictWriter(output_file, fieldnames, restval=0,
                                       extrasaction='ignore', dialect='arff')

        # Finish header and start data section
        print("\n@data", file=output_file)

    def _write_line(self, id_, label_, feat_dict, output_file):
        """
        Write the current line in the file in this Writer's format.

        Parameters
        ----------
        id_ : str
            The ID for the current instance.
        label_ : str
            The label for the current instance.
        feat_dict : dict
            The feature dictionary for the current instance.
        output_file : file buffer
            The file being written to.

        Raises
        ------
        ValueError
            If class column name is already use as a feature
        ValueError
            If ID column name is already used as a feature.
        """
        # Add class column to feat_dict (unless this is unlabeled data)
        if self.label_col not in feat_dict:
            if self.feat_set.has_labels:
                feat_dict[self.label_col] = label_
        else:
            raise ValueError(('Class column name "{}" already used as feature '
                              'name.').format(self.label_col))
        # Add id column to feat_dict if id is provided
        if self.id_col not in feat_dict:
            feat_dict[self.id_col] = id_
        else:
            raise ValueError('ID column name "{}" already used as feature '
                             'name.'.format(self.id_col))
        # Write out line
        self._dict_writer.writerow(feat_dict)


class MegaMWriter(Writer):

    """
    Writer for writing out FeatureSets as MegaM files.
    """

    @staticmethod
    def _replace_non_ascii(line):
        """
        Parameters
        ----------
        line : str
            The line to clean up.

        Returns
        -------
        char_list : str
            Copy of line with all non-ASCII characters replaced with
            <U1234> sequences where 1234 is the value of ord() for the character.
        """
        char_list = []
        for char in line:
            char_num = ord(char)
            char_list.append(
                '<U{}>'.format(char_num) if char_num > 127 else char)
        return ''.join(char_list)

    def _write_line(self, id_, label_, feat_dict, output_file):
        """
        Write the current line in the file in MegaM format.

        Parameters
        ----------
        id_ : str
            The ID for the current instance.
        label_ : str
            The label for the current instance.
        feat_dict : dict
            The feature dictionary for the current instance.
        output_file : file buffer
            The file being written to.
        """
        # Don't try to add class column if this is label-less data
        print('# {}'.format(id_), file=output_file)
        if self.feat_set.has_labels:
            print('{}'.format(label_), end='\t', file=output_file)
        print(self._replace_non_ascii(' '.join(('{} {}'.format(field,
                                                               value) for
                                                field, value in
                                                sorted(feat_dict.items()) if
                                                Decimal(value) != 0))),
              file=output_file)


class NDJWriter(Writer):

    """
    Writer for writing out FeatureSets as .jsonlines/.ndj files.

    Parameters
    ----------
    path : str
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.ndj``.
    feature_set : skll.FeatureSet
        The ``FeatureSet`` instance to dump to the output file.
    kwargs : dict, optional
        The arguments to the ``Writer`` object being instantiated.
    """

    def __init__(self, path, feature_set, **kwargs):
        super(NDJWriter, self).__init__(path, feature_set, **kwargs)

    def _write_line(self, id_, label_, feat_dict, output_file):
        """
        Write the current line in the file in NDJ format.

        Parameters
        ----------
        id_ : str
            The ID for the current instance.
        label_ : str
            The label for the current instance.
        feat_dict : dict
            The feature dictionary for the current instance.
        output_file : file buffer
            The file being written to.
        """
        example_dict = {}
        # Don't try to add class column if this is label-less data
        # Try to convert the label to a scalar assuming it'a numpy
        # non-scalar type (e.g., int64) but if that doesn't work
        # then use it as is
        if self.feat_set.has_labels:
            try:
                example_dict['y'] = label_.item()
            except AttributeError:
                example_dict['y'] = label_
        # Try to convert the ID to a scalar assuming it'a numpy
        # non-scalar type (e.g., int64) but if that doesn't work
        # then use it as is
        try:
            example_dict['id'] = id_.item()
        except AttributeError:
            example_dict['id'] = id_
        example_dict["x"] = feat_dict
        print(json.dumps(example_dict, sort_keys=True), file=output_file)


class LibSVMWriter(Writer):

    """
    Writer for writing out FeatureSets as LibSVM/SVMLight files.

    Parameters
    ----------
    path : str
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.libsvm``.
    feature_set : skll.FeatureSet
        The ``FeatureSet`` instance to dump to the output file.
    kwargs : dict, optional
        The arguments to the ``Writer`` object being instantiated.
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
            if feature_set.has_labels:
                self.label_map = {label: num for num, label in
                                  enumerate(sorted({label for label in
                                                    feature_set.labels if
                                                    not isinstance(label,
                                                                   (int,
                                                                    float))}))}
            # Add fake item to vectorizer for None
            self.label_map[None] = '00000'

    @staticmethod
    def _sanitize(name):
        """
        Replace illegal characters in class names with close unicode
        equivalents to make things loadable in by LibSVM, LibLinear, or
        SVMLight.

        Parameters
        ----------
        name : str
            The class names to replace with unicode equivalents.

        Returns
        -------
        name : str
            The class names with unicode equivalent replacements.
        """
        if isinstance(name, str):
            for orig, replacement in LibSVMWriter.LIBSVM_REPLACE_DICT.items():
                name = name.replace(orig, replacement)
        return name

    def _write_line(self, id_, label_, feat_dict, output_file):
        """
        Write the current line in the file in MegaM format.

        Parameters
        ----------
        id_ : str
            The ID for the current instance.
        label_ : str
            The label for the current instance.
        feat_dict : dict
            The feature dictionary for the current instance.
        output_file : file buffer
            The file being written to.
        """
        field_values = sorted([(self.feat_set.vectorizer.vocabulary_[field] +
                                1, value) for field, value in
                               feat_dict.items() if Decimal(value) != 0])
        # Print label
        if label_ in self.label_map:
            print('{}'.format(self.label_map[label_]), end=' ',
                  file=output_file)
        else:
            print('{}'.format(label_), end=' ', file=output_file)
        # Print features
        print(' '.join(('{}:{}'.format(field, value) for field, value in
                        field_values)), end=' ', file=output_file)
        # Print comment with id and mappings
        print('#', end=' ', file=output_file)
        print(self._sanitize('{}'.format(id_)), end='',
              file=output_file)
        print(' |', end=' ', file=output_file)

        if label_ in self.label_map:
            print('%s=%s' % (self._sanitize(self.label_map[label_]),
                             self._sanitize(label_)),
                  end=' | ', file=output_file)
        else:
            print(' |', end=' ', file=output_file)
        line = ' '.join(('%s=%s' % (self.feat_set.vectorizer.vocabulary_[field] + 1,
                                    self._sanitize(field))
                         for field, value in feat_dict.items()
                         if Decimal(value) != 0))
        print(line, file=output_file)


# Constants
EXT_TO_WRITER = {".arff": ARFFWriter,
                 ".csv": CSVWriter,
                 ".jsonlines": NDJWriter,
                 ".libsvm": LibSVMWriter,
                 ".megam": MegaMWriter,
                 '.ndj': NDJWriter,
                 ".tsv": TSVWriter}
