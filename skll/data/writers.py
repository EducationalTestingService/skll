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
import sys
from csv import DictWriter
from decimal import Decimal
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

from skll.data import FeatureSet
from skll.types import FeatGenerator, FeatureDict, IdType, LabelType, PathOrStr


class Writer(object):
    """
    Write out FeatureSets to files on disk.

    This is the base class used to create featureset writers for different
    file types.

    Parameters
    ----------
    path : :class:`skll.types.PathOrStr`
        A path to the feature file we would like to create. The suffix
        to this filename must be ``.arff``, ``.csv``, ``.jsonlines``,
        ``.libsvm``, ``.ndj``, or ``.tsv``. If ``subsets``
        is not ``None``, when calling the ``write()`` method, path is
        assumed to be a string containing the path to the directory to
        write the feature files with an additional file extension
        specifying the file type. For example ``/foo/.csv``.

    feature_set : :class:`skll.data.featureset.FeatureSet`
        The ``FeatureSet`` instance to dump to the file.

    quiet : bool, default=True
        Do not print "Writing..." status message to stderr.

    subsets : Optional[Dict[str, List[str]]], default=None
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

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.

    """

    def __init__(
        self,
        path: PathOrStr,
        feature_set: FeatureSet,
        quiet: bool = True,
        subsets: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize base Writer class."""
        super(Writer, self).__init__()

        self.quiet = quiet
        self.path = Path(path)
        self.feat_set = feature_set
        self.subsets = subsets
        self.logger = logger if logger else logging.getLogger(__name__)

        # Get prefix & extension for checking file types & writing subset files;
        # since we also need to handle paths like "foo/.csv" for subset-writing
        # we have to do a bit of introspection before figuring out the various
        # parts of the path
        parent, stem, suffix = self.path.parent, self.path.stem, self.path.suffix
        if stem.startswith(".") and not suffix:
            self.root = parent
            self.ext = stem.lower()
        else:
            self.root = parent / stem
            self.ext = suffix.lower()
        self._progress_msg = ""
        self._use_pandas = False

    @classmethod
    def for_path(cls, path: PathOrStr, feature_set: FeatureSet, **kwargs) -> "Writer":
        """
        Retrieve object of ``Writer`` sub-class appropriate for given path.

        Parameters
        ----------
        path : :class:`skll.types.PathOrStr`
            A path to the feature file we would like to create. The
            suffix to this filename must be ``.arff``, ``.csv``,
            ``.jsonlines``, ``.libsvm``, ``.ndj``, or
            ``.tsv``. If ``subsets`` is not ``None``, when calling the
            ``write()`` method, path is assumed to be a string
            containing the path to the directory to write the feature
            files with an additional file extension specifying the
            file type. For example ``/foo/.csv``.

        feature_set : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance to dump to the output file.

        kwargs : Optional[Dict[str, Any]]
            The keyword arguments for ``for_path`` are the same as
            the initializer for the desired ``Writer`` subclass.

        Returns
        -------
        writer : :class:`skll.data.Writer`
            New instance of the Writer sub-class that is
            appropriate for the given path.

        """
        # Get lowercase extension for file extension checking
        # NOTE: the reason we are doing this complicated gymnastics
        # instead of just using `path.suffix` is because sometimes
        # `path` may look like `foo/.jsonlines` when we are writing
        # subsets so we need to handle that edge case
        path = Path(path)
        stem, suffix = path.stem, path.suffix
        if stem.startswith(".") and not suffix:
            ext = stem.lower()
        else:
            ext = suffix.lower()
        return EXT_TO_WRITER[ext](path, feature_set, **kwargs)

    def write(self) -> None:
        """Write out this Writer's ``FeatureSet`` to a file in its format."""
        if isinstance(self.feat_set.vectorizer, FeatureHasher):
            raise ValueError(
                "Writer cannot write sets that use a " "FeatureHasher for vectorization."
            )

        # Write one feature file if we weren't given a dict of subsets
        if self.subsets is None:
            self._write_subset(self.path)

        # Otherwise write one feature file per subset
        else:
            for subset_name, filter_features in self.subsets.items():
                self.logger.debug(f"Subset ({subset_name}) features: " f"{filter_features}")
                sub_path = self.root / f"{subset_name}{self.ext}"
                self._write_subset(sub_path, set(filter_features))

    def _write_subset(
        self, sub_path: PathOrStr, filter_features: Optional[Set[str]] = None
    ) -> None:
        """
        Write out given ``FeatureSet`` instance to a file in this class's format.

        Parameters
        ----------
        sub_path : :class:`skll.types.PathOrStr`
            The path to the file we want to create for this subset
            of our data.

        filter_features : Optional[Set[str]], default=None
            Set of features to include in current feature file.

        """
        self.logger.debug(f"sub_path: {sub_path}")
        self.logger.debug(f"feature_set: {self.feat_set.name}")
        self.logger.debug(f"filter_features: {filter_features}")

        if not self.quiet:
            self._progress_msg = f"Writing {sub_path}..."
            print(self._progress_msg, end="\r", file=sys.stderr)
            sys.stderr.flush()

        if not self._use_pandas:
            # Apply filtering
            filtered_set: Union[FeatGenerator, FeatureSet] = (
                self.feat_set.filtered_iter(features=filter_features)
                if filter_features is not None
                else self.feat_set
            )

            # Open file for writing and write each line
            with open(sub_path, "w", encoding="utf-8") as output_file:
                # Write out the header if this format requires it
                self._write_header(filtered_set, output_file, filter_features)
                # Write individual lines
                for ex_num, (id_, label_, feat_dict) in enumerate(filtered_set):
                    self._write_line(id_, label_, feat_dict, output_file)
                    if not self.quiet and ex_num % 100 == 0:
                        print(f"{self._progress_msg}{ex_num:>15}", end="\r", file=sys.stderr)
                        sys.stderr.flush()
        else:
            self._write_data(self.feat_set, sub_path, filter_features)

        if not self.quiet:
            print(f"{self._progress_msg}{'done':<15}", file=sys.stderr)
            sys.stderr.flush()

    def _write_header(self, feature_set, output_file, filter_features):
        """
        Write header to file.

        Called before lines are written to file, so that headers can be written
        for files that need them.

        Parameters
        ----------
        feature_set : Ignored
            Not used.

        output_file : Ignored
            Not used.

        filter_features : Ignored
           Not used.

        """
        pass

    def _write_line(self, id_, label_, feat_dict, output_file):
        """
        Write the current line in the file in this Writer's format.

        Parameters
        ----------
        id_ : Ignored
            Not used.

        label_ : Ignored
            Not used.

        feat_dict : Ignored
            Not used.

        output_file : Ignored
             Not used.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def _write_data(self, feature_set, output_file, filter_features):
        """
        Write full data set in Writer's format using `pandas`, rather than row-by-row.

        Parameters
        ----------
        feature_set : Ignored
            Not used.

        output_file : Ignored
            Not used.

        filter_features : Ignored
            Not used.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def _get_column_names_and_indexes(
        self, feature_set: FeatureSet, filter_features: Optional[Set[str]] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Get names of columns and associated indices for (possibly filtered) features.

        Parameters
        ----------
        feature_set : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance being written to a file.

        filter_features : Optional[Set[str]], default=None
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.

        Returns
        -------
        column_names : List[str]
            A list of the (possibly filtered) column names.

        column_indexes : List[int]
            A list of the (possibly filtered) column indexes.

        """
        # if we're not doing filtering,
        # then just take all the feature names
        self.logger.debug(feature_set)
        if isinstance(feature_set.vectorizer, DictVectorizer):
            if filter_features is None:
                filter_features = feature_set.vectorizer.feature_names_

            # create a list of tuples with (column names, column indexes)
            # so that we can correctly extract the appropriate columns
            columns = sorted(
                [
                    (col_name, col_idx)
                    for col_name, col_idx in feature_set.vectorizer.vocabulary_.items()
                    if (col_name in filter_features or col_name.split("=", 1)[0] in filter_features)
                ],
                key=lambda x: x[1],
            )

            # then, split the names and indexes into separate lists
            column_names, column_indexes = zip(*columns)
            return list(column_names), list(column_indexes)
        else:
            return [], []


class CSVWriter(Writer):
    """
    Writer for writing out ``FeatureSet`` instances as CSV files.

    Parameters
    ----------
    path : :class:`skll.types.PathOrStr`
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.csv``.

    feature_set : :class:`skll.data.featureset.FeatureSet`
        The ``FeatureSet`` instance to dump to the output file.

    quiet : bool, default=True
        Do not print "Writing..." status message to stderr.

    subsets : Optional[Dict[str, List[str]]], default=None
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

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.

    label_col : str, default="y"
        The column name containing the label.

    id_col : str, default="id"
        The column name containing the ID.

    pandas_kwargs : Optional[Dict[str], Any], default=None
        Arguments that will be passed directly to the `pandas` I/O reader.

    """

    def __init__(
        self,
        path: PathOrStr,
        feature_set: FeatureSet,
        quiet: bool = True,
        subsets: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
        label_col: str = "y",
        id_col: str = "id",
        pandas_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the CSVWriter class."""
        self.label_col = label_col
        self.id_col = id_col
        self._pandas_kwargs = {} if pandas_kwargs is None else pandas_kwargs
        self._sep = self._pandas_kwargs.pop("sep", ",")
        self._index = self._pandas_kwargs.pop("index", False)
        super(CSVWriter, self).__init__(
            path, feature_set, quiet=quiet, subsets=subsets, logger=logger
        )
        self._use_pandas = True

    def _build_dataframe_with_features(
        self, feature_set: FeatureSet, filter_features: Optional[Set[str]] = None
    ) -> pd.DataFrame:
        """
        Create and filter data frame from features in given feature set.

        Parameters
        ----------
        feature_set : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance being written to a file.

        filter_features : Optional[Set[str]], default=None
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.

        Returns
        -------
        df_features : pandas.DataFrame
            The data frame constructed from the feature set. The frame may be
            empty are not features in the feature set.

        Raises
        ------
        ValueError
            If ID column is already used as feature.
            If label column is already used as feature.

        """
        # if there is no filtering, then just keep all the names
        (column_names, column_idxs) = self._get_column_names_and_indexes(
            feature_set, filter_features
        )

        # create the data frame from the feature set;
        # then, select only the columns that we want,
        # and give the columns their correct names
        if feature_set.features is not None:
            if issparse(feature_set.features):
                df_features = pd.DataFrame(feature_set.features.toarray())
            else:
                df_features = pd.DataFrame(feature_set.features)
            df_features = df_features.iloc[:, column_idxs].copy()
            df_features.columns = column_names
            return df_features
        else:
            return pd.DataFrame()

    def _build_dataframe(
        self,
        feature_set: FeatureSet,
        filter_features: Optional[Set[str]] = None,
        df_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create and filter data frame with features in given feature set.

        Add the IDs and labels, if applicable. If the data frame
        with features already exists, pass `df_features`. Then the IDs and labels will
        simply be added to the existing data frame containing the features.

        Parameters
        ----------
        feature_set : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance being written to a file.

        filter_features : Optional[Set[str]], default=None
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.

        df_features : Optional[pandas.DataFrame], default=None
            If the data frame with features already exists,
            then we use it and add IDs and labels; otherwise,
            the feature data frame will be created from the feature set.

        Returns
        -------
        df_features : pandas.DataFrame
            The data frame constructed from the feature set.

        Raises
        ------
        ValueError
            If ID column is already used as feature.
            If label column is already used as feature.

        """
        # create the data frame with just the features
        # from the feature set, at this point
        if df_features is None:
            df_features = self._build_dataframe_with_features(feature_set, filter_features)

        # if the id column is already in the data frame,
        # then raise an error; otherwise, just add the ids
        if self.id_col in df_features:
            raise ValueError(f"ID column name {self.id_col} already used as feature name.")
        df_features[self.id_col] = feature_set.ids

        # if the the labels should exist but the column is already
        # in the data frame, then raise an error; otherwise, just add the labels
        if feature_set.has_labels:
            if self.label_col in df_features:
                raise ValueError(
                    f'Class column name "{self.label_col}" ' "already used as feature name."
                )
            df_features[self.label_col] = feature_set.labels

        return df_features

    def _write_data(
        self,
        feature_set: FeatureSet,
        output_file: PathOrStr,
        filter_features: Optional[Set[str]] = None,
    ) -> None:
        """
        Write the data in CSV format.

        Parameters
        ----------
        feature_set : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance being written to a file.

        output_file : :class:`skll.types.PathOrStr`
            The path of the file being written to

        filter_features : Optional[Set[str]], default=None
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.

        """
        df = self._build_dataframe(feature_set, filter_features=filter_features)
        df.to_csv(output_file, sep=self._sep, index=self._index, **self._pandas_kwargs)


class TSVWriter(CSVWriter):
    """
    Writer for writing out FeatureSets as TSV files.

    Parameters
    ----------
    path : :class:`skll.types.PathOrStr`
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.tsv``.

    feature_set : :class:`skll.data.featureset.FeatureSet`
        The ``FeatureSet`` instance to dump to the output file.

    quiet : bool, default=True
        Do not print "Writing..." status message to stderr.

    subsets : Optional[Dict[str, List[str]]], default=None
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

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.

    label_col: str, default="y"
        The column name containing the label.

    id_col: str, default="id"
        The column name containing the ID.

    pandas_kwargs : Optional[Dict[str, Any]], default=None
        Arguments that will be passed directly to the `pandas` I/O reader.

    """

    def __init__(
        self,
        path: PathOrStr,
        feature_set: FeatureSet,
        quiet: bool = True,
        subsets: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
        label_col: str = "y",
        id_col: str = "id",
        pandas_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the TSVWriter class."""
        super(TSVWriter, self).__init__(
            path,
            feature_set,
            quiet=quiet,
            subsets=subsets,
            logger=logger,
            label_col=label_col,
            id_col=id_col,
            pandas_kwargs=pandas_kwargs,
        )
        self._sep = str("\t")


class ARFFWriter(Writer):
    """
    Writer for writing out FeatureSets as ARFF files.

    Parameters
    ----------
    path : :class:`skll.types.PathOrStr`
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.arff``.

    feature_set : :class:`skll.data.featureset.FeatureSet`
        The ``FeatureSet`` instance to dump to the output file.

    quiet : bool, default=True
        Do not print "Writing..." status message to stderr.

    subsets : Optional[Dict[str, List[str]]], default=None
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

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.

    relation : str, default='skll_relation'
        The name of the relation in the ARFF file.

    regression : bool, default=False
        Is this an ARFF file to be used for regression?

    kwargs : Optional[Dict[str, Any]]
        The arguments to the ``Writer`` object being instantiated.

    """

    def __init__(
        self,
        path: PathOrStr,
        feature_set: FeatureSet,
        quiet: bool = True,
        subsets: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
        relation="skll_relation",
        regression=False,
        dialect="excel-tab",
        label_col="y",
        id_col="id",
    ):
        """Initialize the ARFFWRiter class."""
        self.relation = relation
        self.regression = regression
        self.dialect = dialect
        self.label_col = label_col
        self.id_col = id_col
        super(ARFFWriter, self).__init__(
            path, feature_set, quiet=quiet, subsets=subsets, logger=logger
        )
        self._dict_writer: Optional[DictWriter[str]] = None

    def _write_header(
        self, feature_set: FeatureSet, output_file: IO[str], filter_features: Set[str]
    ) -> None:
        """
        Write headers to ARFF file.

        Called before lines are written to file, so that headers can be written
        for files that need them.

        Parameters
        ----------
        feature_set : Ignored
            Not used.

        output_file : IO[str]
            The file being written to.

        filter_features : Set[str]
            If only writing a subset of the features in the
            FeatureSet to ``output_file``, these are the
            features to include in this file.

        """
        fieldnames, _ = self._get_column_names_and_indexes(self.feat_set, filter_features)
        fieldnames.append(self.id_col)

        # Add relation to header
        print(f"@relation '{self.relation}'\n", file=output_file)

        # Loop through fields writing the header info for the ARFF file
        for field in fieldnames:
            field = field.replace("\\", "\\\\").replace("'", "\\'")
            print(f"@attribute '{field}' numeric", file=output_file)

        # Print class label header if necessary
        if self.regression:
            print(f"@attribute {self.label_col} numeric", file=output_file)
        else:
            if self.feat_set.has_labels:
                sorted_features = sorted(set(self.feat_set.labels))  # type: ignore
                labels_str = ",".join([str(feat) for feat in sorted_features])
                labels_str = "{" + labels_str + "}"
                print(f"@attribute {self.label_col} {labels_str}", file=output_file)
        if self.label_col:
            fieldnames.append(self.label_col)

        # Create CSV writer to handle missing values for lines in data section
        # and to ignore the instance values for non-numeric attributes
        self._dict_writer = DictWriter(
            output_file, fieldnames, restval=0, extrasaction="ignore", dialect="arff"
        )

        # Finish header and start data section
        print("\n@data", file=output_file)

    def _write_line(
        self, id_: IdType, label_: LabelType, feat_dict: FeatureDict, output_file: IO[str]
    ) -> None:
        """
        Write the current line in the file in this Writer's format.

        Parameters
        ----------
        id_ : :class:`skll.types.IdType`
            The ID for the current instance.

        label_ : :class:`skll.types.LabelType`
            The label for the current instance.

        feat_dict : :class:`skll.types.FeatureDict`
            The feature dictionary for the current instance.

        output_file : Ignored.
            Not used.

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
            raise ValueError(
                f'Class column name "{self.label_col}" already ' "used as feature name."
            )
        # Add id column to feat_dict if id is provided
        if self.id_col not in feat_dict:
            feat_dict[self.id_col] = id_
        else:
            raise ValueError(f'ID column name "{self.id_col}" already used as' " feature name.")

        # Write out line
        if self._dict_writer:
            self._dict_writer.writerow(feat_dict)


class NDJWriter(Writer):
    """
    Writer for writing out FeatureSets as .jsonlines/.ndj files.

    Parameters
    ----------
    path : :class:`skll.types.PathOrStr`
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.ndj``.

    feature_set : :class:`skll.data.featureset.FeatureSet`
        The ``FeatureSet`` instance to dump to the output file.

    quiet : bool, default=True
        Do not print "Writing..." status message to stderr.

    subsets : Optional[Dict[str, List[str]]], default=None
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

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.

    """

    def __init__(
        self,
        path: PathOrStr,
        feature_set: FeatureSet,
        quiet: bool = True,
        subsets: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the NDJWriter class."""
        super(NDJWriter, self).__init__(
            path, feature_set, quiet=quiet, subsets=subsets, logger=logger
        )

    def _write_line(
        self,
        id_: IdType,
        label_: Union[LabelType, np.int64, np.float64],
        feat_dict: FeatureDict,
        output_file: IO[str],
    ) -> None:
        """
        Write the current line in the file in NDJ format.

        Parameters
        ----------
        id_ : :class:`skll.types.IdType`
            The ID for the current instance.

        label_ : :class:`skll.types.LabelType`
            The label for the current instance.

        feat_dict : :class:`skll.types.FeatureDict`
            The feature dictionary for the current instance.

        output_file : IO[str]
            The file being written to.

        """
        example_dict: FeatureDict = {}
        # Don't try to add class column if this is label-less data
        # Try to convert the label to a scalar assuming it'a numpy
        # non-scalar type (e.g., int64) but if that doesn't work
        # then use it as is
        if self.feat_set.has_labels:
            if hasattr(label_, "item"):
                example_dict["y"] = label_.item()
            else:
                example_dict["y"] = label_
        # Try to convert the ID to a scalar assuming it'a numpy
        # non-scalar type (e.g., int64) but if that doesn't work
        # then use it as is
        if hasattr(id_, "item"):
            example_dict["id"] = id_.item()
        else:
            example_dict["id"] = id_
        example_dict["x"] = feat_dict
        print(json.dumps(example_dict, sort_keys=True), file=output_file)


class LibSVMWriter(Writer):
    """
    Writer for writing out FeatureSets as LibSVM/SVMLight files.

    Parameters
    ----------
    path : :class:`skll.types.PathOrStr`
        A path to the feature file we would like to create.
        If ``subsets`` is not ``None``, this is assumed to be a string
        containing the path to the directory to write the feature
        files with an additional file extension specifying the file
        type. For example ``/foo/.libsvm``.

    feature_set : :class:`skll.data.featureset.FeatureSet`
        The ``FeatureSet`` instance to dump to the output file.

    quiet : bool, default=True
        Do not print "Writing..." status message to stderr.

    subsets : Optional[Dict[str, List[str]]], default=None
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

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.

    label_map : Optional[Dict[str, int]], default=None
        A mapping from label strings to integers.

    """

    LIBSVM_REPLACE_DICT = {
        ":": "\u2236",
        "#": "\uFF03",
        " ": "\u2002",
        "=": "\ua78a",
        "|": "\u2223",
    }

    def __init__(
        self,
        path: PathOrStr,
        feature_set: FeatureSet,
        quiet: bool = True,
        subsets: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
        label_map: Optional[Dict[Any, Any]] = None,
    ):
        """Initialize the LibSVMWriter class."""
        self.label_map = label_map
        super(LibSVMWriter, self).__init__(
            path, feature_set, quiet=quiet, subsets=subsets, logger=logger
        )
        if self.label_map is None:
            fs_labels = feature_set.labels if feature_set.has_labels else np.array([])
            self.label_map = {
                label: num
                for num, label in enumerate(
                    sorted(
                        {
                            label
                            for label in fs_labels  # type: ignore
                            if not isinstance(label, (int, float))
                        }
                    )
                )
            }
            # Add fake item to vectorizer for None
            self.label_map[None] = "00000"

    @staticmethod
    def _sanitize(name: Union[IdType, LabelType]) -> Union[IdType, LabelType]:
        """
        Sanitize feature names for older feature formats.

        Replace special characters in names with close unicode
        equivalents to make things loadable in by LibSVM, LibLinear, or
        SVMLight.

        Parameters
        ----------
        name : Union[:class:`skll.types.IdType`, :class:`skll.types.LabelType`]
            Input name in which special characters are replaced with unicode
            equivalents.

        Returns
        -------
        Union[:class:`skll.types.IdType`, :class:`skll.types.LabelType`]
            The sanitized name with special characters replaced.

        """
        sanitized_name = name
        if isinstance(sanitized_name, str):
            for orig, replacement in LibSVMWriter.LIBSVM_REPLACE_DICT.items():
                sanitized_name = sanitized_name.replace(orig, replacement)
        return sanitized_name

    def _write_line(
        self, id_: IdType, label_: LabelType, feat_dict: FeatureDict, output_file: IO[str]
    ) -> None:
        """
        Write the current line in the file in this Writer's format.

        Parameters
        ----------
        id_ : :class:`skll.types.IdType`
            The ID for the current instance.

        label_ : :class:`skll.types.LabelType`
            The label for the current instance.

        feat_dict : :class:`skll.types.FeatureDict`
            The feature dictionary for the current instance.

        output_file : IO[str]
            The file being written to.

        """
        field_values = (
            sorted(
                [
                    (self.feat_set.vectorizer.vocabulary_[field] + 1, value)
                    for field, value in feat_dict.items()
                    if Decimal(value) != 0
                ]
            )
            if self.feat_set.vectorizer
            else []
        )

        # Print label
        if self.label_map:
            if label_ in self.label_map:
                print(self.label_map[label_], end=" ", file=output_file)
            else:
                print(label_, end=" ", file=output_file)

        # Print features
        print(
            " ".join((f"{field}:{value}" for field, value in field_values)),
            end=" ",
            file=output_file,
        )

        # Print comment with id and mappings
        print("#", end=" ", file=output_file)
        print(self._sanitize(id_), end="", file=output_file)
        print(" |", end=" ", file=output_file)

        if self.label_map:
            if label_ in self.label_map:
                print(
                    f"{self._sanitize(self.label_map[label_])}={self._sanitize(label_)}",
                    end=" | ",
                    file=output_file,
                )
            else:
                print(" |", end=" ", file=output_file)

        line = (
            " ".join(
                f"{self.feat_set.vectorizer.vocabulary_[field] + 1}={self._sanitize(field)}"
                for field, value in feat_dict.items()
                if Decimal(value) != 0
            )
            if self.feat_set.vectorizer
            else ""
        )
        print(line, file=output_file)


# Constants
EXT_TO_WRITER = {
    ".arff": ARFFWriter,
    ".csv": CSVWriter,
    ".jsonlines": NDJWriter,
    ".libsvm": LibSVMWriter,
    ".ndj": NDJWriter,
    ".tsv": TSVWriter,
}
