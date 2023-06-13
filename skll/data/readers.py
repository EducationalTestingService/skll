# License: BSD 3 clause
"""
Load features and labels from various file types.

Handle loading data from various types of data files. A
base ``Reader`` class is provided that is sub-classed for each data
file type that is supported, e.g. ``CSVReader``.

Notes about IDs & Label Conversion
-----------------------------------
All ``Reader`` sub-classes are designed to read in example IDs
as strings unless ``ids_to_floats`` is set to ``True`` in which
case they will be read in as floats, if possible. In the latter
case, an exception will be raised if they cannot be converted to
floats.

All ``Reader`` sub-classes also use the ``safe_float`` function internally
to read in labels. This function tries to convert a single label
first to ``int``, then to ``float``. If neither conversion is
possible, the label remains a ``str``. It should be noted that, if
classification is being done with a feature set that is read in with
one of the ``Reader`` sub-classes, care must be taken to ensure that
labels do not get converted in unexpected ways. For example,
classification labels should not be a mixture of ``int``-converting
and ``float``-converting labels. Consider the situation below:

>>> import numpy as np
>>> from skll.data.readers import safe_float
>>> np.array([safe_float(x) for x in ["2", "2.2", "2.21"]]) # array([2.  , 2.2 , 2.21])

The labels will all be converted to floats and any classification
model generated with this data will predict labels such as ``2.0``,
``2.2``, etc., not ``str`` values that exactly match the input
labels, as might be expected. Be aware that it may be best to make
use of the ``class_map`` keyword argument in such cases to map
original labels to labels that convert only to ``str``.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:organization: ETS
"""

import csv
import json
import logging
import re
import sys
from io import StringIO
from itertools import chain
from numbers import Number
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bs4 import UnicodeDammit
from sklearn.feature_extraction import FeatureHasher

from skll.data import FeatureSet
from skll.data.dict_vectorizer import DictVectorizer
from skll.types import ClassMap, FeatGenerator, FeatureDictList, IdType, LabelType, PathOrStr


class Reader(object):
    """
    Load FeatureSets from files on disk.

    This is the base class used to create featureset readers for different
    file types.

    Parameters
    ----------
    path_or_list : Union[:class:`skll.types.PathOrStr`, List[Dict[str, Any]]
        Path or a list of example dictionaries.

    quiet : bool, default=True
        Do not print "Loading..." status message to stderr.

    ids_to_floats : bool, default=False
        Convert IDs to float to save memory. Will raise error
        if we encounter an a non-numeric ID.

    label_col : Optional[str], default='y'
        Name of the column which contains the class labels
        for ARFF/CSV/TSV files. If no column with that name
        exists, or ``None`` is specified, the data is
        considered to be unlabelled.

    id_col : str, default='id'
        Name of the column which contains the instance IDs.
        If no column with that name exists, or ``None`` is
        specified, example IDs will be automatically generated.

    class_map : Optional[:class:`skll.types.ClassMap`], default=None
        Mapping from original class labels to new ones. This is
        mainly used for collapsing multiple labels into a single
        class. Anything not in the mapping will be kept the same.
        The keys are the new labels and the list of values for each
        key is the labels to be collapsed to said new label.

    sparse : bool, default=True
        Whether or not to store the features in a numpy CSR
        matrix when using a DictVectorizer to vectorize the
        features.

    feature_hasher : bool, default=False
        Whether or not a FeatureHasher should be used to
        vectorize the features.

    num_features : Optional[int], default=None
        If using a FeatureHasher, how many features should the
        resulting matrix have?  You should set this to a power
        of 2 greater than the actual number of features to
        avoid collisions.

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.
    """

    def __init__(
        self,
        path_or_list: Union[PathOrStr, FeatureDictList],
        quiet: bool = True,
        ids_to_floats: bool = False,
        label_col: Optional[str] = "y",
        id_col: str = "id",
        class_map: Optional[ClassMap] = None,
        sparse: bool = True,
        feature_hasher: bool = False,
        num_features: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the base class."""
        super(Reader, self).__init__()
        self.path_or_list = path_or_list
        self.quiet = quiet
        self.ids_to_floats = ids_to_floats
        self.label_col = label_col
        self.id_col = id_col
        self.class_map = class_map
        self._progress_msg = ""
        self._use_pandas = False

        self.vectorizer: Union[DictVectorizer, FeatureHasher]
        if feature_hasher:
            self.vectorizer = FeatureHasher(n_features=num_features)
        else:
            self.vectorizer = DictVectorizer(sparse=sparse)
        self.logger = logger if logger else logging.getLogger(__name__)

    @classmethod
    def for_path(cls, path_or_list: Union[PathOrStr, FeatureDictList], **kwargs) -> "Reader":
        """
        Instantiate Reader sub-class based on the file extension.

        If the input is a list of dictionaries instead of a path, use a
        dictionary reader instead.

        Parameters
        ----------
        path_or_list : Union[:class:`skll.types.PathOrStr`, :class:`skll.types.FeatureDictList`]
            A path or list of example dictionaries.

        kwargs : Optional[Dict[str, Any]]
            The arguments to the Reader object being instantiated.

        Returns
        -------
        reader : :class:`skll.data.readers.Reader`
            A new instance of the Reader sub-class that is
            appropriate for the given path.

        Raises
        ------
        ValueError
            If file does not have a valid extension.
        """
        if not isinstance(path_or_list, (str, Path)):
            return DictListReader(path_or_list)
        else:
            # Get lowercase extension for file extension checking
            path_or_list = Path(path_or_list)
            ext = path_or_list.suffix.lower()
            if ext not in EXT_TO_READER:
                raise ValueError(
                    "Example files must be in either .arff, "
                    ".csv, .jsonlines, .ndj, or .tsv format. You"
                    f"specified: {path_or_list}"
                )
        return EXT_TO_READER[ext](path_or_list, **kwargs)

    def _sub_read(self, file):
        """
        Perform the actual reading of the given file or list.

        For ``Reader`` objects that do not rely on ``pandas``
        (and therefore read row-by-row), this function will
        be called by  ``_sub_read_rows()`` and will take a file
        buffer rather than a file path. Otherwise, it will
        take a path and will be called directly in the ``read()``
        method.

        Parameters
        ----------
        file : Ignored
            Not used.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def _print_progress(self, progress_num: Union[int, str], end="\r"):
        r"""
        Print out progress numbers in proper format.

        Nothing gets printed if ``self.quiet`` is ``True``.

        Parameters
        ----------
        progress_num: Union[int, str]
            Progress indicator value. Usually either a line
            number or a percentage. Must be able to convert to string.

        end : str, default='\r'
            The string to put at the end of the line.  "\r" should be
            used for every update except for the final one.
        """
        # Print out status
        if not self.quiet:
            print(f"{self._progress_msg}{progress_num:>15}", end=end, file=sys.stderr)
            sys.stderr.flush()

    def _sub_read_rows(self, file: PathOrStr) -> Tuple[np.ndarray, np.ndarray, FeatureDictList]:
        """
        Read the file in row-by-row.

        This method is used for ``Reader`` objects that do not rely on ``pandas``,
        and are instead read line-by-line into a FeatureSet object, unlike
        pandas-based reader object, which will read everything into memory in
        a data frame object before converting to a ``FeatureSet``.

        Parameters
        ----------
        file : :class:`skll.types.PathOrStr`
            The path to the input file.

        Returns
        -------
        ids : numpy.ndarray of shape (n_ids,)
            The ids array.

        labels : numpy.ndarray of shape (n_labels,)
            The labels array.

        features : :class:`skll.types.FeatureDictList`
            List containing feature dictionaries.

        Raises
        ------
        ValueError
            If ``ids_to_floats`` is True, but IDs cannot be converted.

        ValueError
            If no features are found.

        ValueError
            If the example IDs are not unique.
        """
        # Get labels and IDs
        ids_list: List[IdType] = []
        labels_list: List[LabelType] = []
        ex_num = 0
        with open(file, encoding="utf-8") as f:
            for ex_num, (id_, class_, _) in enumerate(self._sub_read(f), start=1):
                # Update lists of IDs, classes, and features
                if self.ids_to_floats:
                    try:
                        id_ = float(id_)
                    except ValueError:
                        raise ValueError(
                            "You set ids_to_floats to true, but "
                            f"ID {id_} could not be converted to"
                            f" float in {self.path_or_list}"
                        )
                ids_list.append(id_)
                labels_list.append(class_)
                if ex_num % 100 == 0:
                    self._print_progress(ex_num)
            self._print_progress(ex_num)

        # Remember total number of examples for percentage progress meter
        total = ex_num
        if total == 0:
            raise ValueError("No features found in possibly empty file " f"'{self.path_or_list}'.")

        # Convert everything to numpy arrays
        ids = np.array(ids_list)
        labels = np.array(labels_list)

        def feat_dict_generator():
            with open(self.path_or_list, encoding="utf-8") as f:
                for ex_num, (_, _, feat_dict) in enumerate(self._sub_read(f)):
                    yield feat_dict
                    if ex_num % 100 == 0:
                        self._print_progress(f"{100 * ex_num / total:.8}%")
                self._print_progress("100%")

        # extract the features dictionary
        features = feat_dict_generator()

        return ids, labels, features

    def _parse_dataframe(
        self,
        df: pd.DataFrame,
        id_col: Optional[str],
        label_col: Optional[str],
        replace_blanks_with: Optional[Union[Number, Dict[str, Number]]] = None,
        drop_blanks: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, FeatureDictList]:
        """
        Parse the data frame into ids, labels, and features.

        For ``Reader`` objects that rely on ``pandas``, this function
        will be called in the ``_sub_read()`` method to parse the
        data frame into the expected format. It will not be used
        by ``Reader`` classes that read row-by-row (and therefore
        use the ``_sub_read_rows()`` function).

        Parameters
        ----------
        df : pandas.DataFrame
            The pandas data frame to parse.

        id_col : Optional[str]
            The id column.

        label_col : Optional[str]
            The label column.

        replace_blanks_with : Optional[Union[Number, Dict[str, Number]]], default=None
            Specifies a new value with which to replace blank values.
            Options are:

                - ``Number`` : A (numeric) value with which to replace blank values.
                - ``Dict[str, Number]`` : A dictionary specifying the replacemen
                  value for each column.
                - ``None`` : Blank values will be left as blanks, and not replaced.

        drop_blanks : bool, default=False
            If ``True``, remove lines/rows that have any blank values.

        Returns
        -------
        ids : numpy.ndarray of shape (n_ids,)
            The ids for the feature set.

        labels : numpy.ndarray of shape (n_labels,)
            The labels for the feature set.

        features : :class:`skll.types.FeatureDictList`
            List of feature dictionaries.
        """
        if df.empty:
            raise ValueError("No features found in possibly empty file " f"'{self.path_or_list}'.")

        if drop_blanks and replace_blanks_with is not None:
            raise ValueError(
                "You cannot both drop blanks and replace them. "
                "'replace_blanks_with' can only have a value when "
                "'drop_blanks' is `False`."
            )

        # should we replace blank values with something?
        if replace_blanks_with is not None:
            self.logger.info(
                "Blank values in all rows/lines will be replaced with " "user-specified value(s)."
            )
            df = df.fillna(replace_blanks_with)

        # should we remove lines that have any NaNs?
        if drop_blanks:
            self.logger.info("Rows/lines with any blank values will be dropped.")
            df = df.dropna().reset_index(drop=True)

            # if the dataframe has no rows left after removing blanks,
            # raise an exception here because downstream processing
            # will run into issues
            if df.empty:
                raise ValueError(
                    "No rows/lines left in the feature file " "after dropping blank values."
                )

        # if the id column exists,
        # get them from the data frame and
        # delete the column; otherwise, just
        # set it to None
        if id_col is not None and id_col in df:
            ids = df[id_col].astype(str)
            del df[id_col]
            # if `ids_to_floats` is True,
            # then convert the ids to floats
            if self.ids_to_floats:
                ids = ids.astype(float)
            ids = ids.values
        else:
            # create ids with the prefix `EXAMPLE_`
            ids = np.array([f"EXAMPLE_{i}" for i in range(df.shape[0])])

        # if the label column exists,
        # get them from the data frame and
        # delete the column; otherwise, just
        # set it to None
        if label_col is not None and label_col in df:
            labels = df[label_col]
            del df[label_col]
            # if `class_map` exists, then
            # map the new classes to the labels;
            # otherwise, just convert them to floats
            if self.class_map is not None:
                labels = labels.apply(safe_float, replace_dict=self.class_map)
            else:
                labels = labels.apply(safe_float)
            labels = labels.values
        else:
            # create an array of Nones
            labels = np.array([None] * df.shape[0])

        # convert the remaining features to
        # a list of dictionaries
        features = df.to_dict(orient="records")

        return ids, labels, features

    def read(self) -> FeatureSet:
        """
        Load examples from various file formats.

        The following formats are supported: ``.arff``, ``.csv``, ``.jsonlines``,
        ``.libsvm``, ``.ndj``, or ``.tsv`` formats.

        Returns
        -------
        :class:`skll.data.featureset.FeatureSet`
            A ``FeatureSet`` instance representing the input file.

        Raises
        ------
        ValueError
            If ``ids_to_floats`` is True, but IDs cannot be converted.

        ValueError
            If no features are found.

        ValueError
            If the example IDs are not unique.
        """
        self.logger.debug(f"Path: {self.path_or_list}")

        if not self.quiet:
            self._progress_msg = f"Loading {self.path_or_list}..."
            print(self._progress_msg, end="\r", file=sys.stderr)
            sys.stderr.flush()

        # if we are in this method, self.path_or_file must be a path
        assert isinstance(self.path_or_list, (str, Path)), "file path or path object required"

        if self._use_pandas:
            ids, labels, features = self._sub_read(self.path_or_list)
        else:
            ids, labels, features = self._sub_read_rows(self.path_or_list)

        # Convert everything to numpy arrays
        features = self.vectorizer.fit_transform(features)

        # Report that loading is complete
        self._print_progress("done", end="\n")

        # Make sure we have the same number of ids, labels, and features
        assert ids.shape[0] == labels.shape[0] == features.shape[0]

        if ids.shape[0] != len(set(ids)):
            raise ValueError("The example IDs are not unique in " f"{self.path_or_list}.")

        featureset_name = str(self.path_or_list)
        return FeatureSet(
            featureset_name, ids, labels=labels, features=features, vectorizer=self.vectorizer
        )


class DictListReader(Reader):
    """
    Facilitate programmatic use of methods that take ``FeatureSet`` as input.

    Support ``Learner.predict()`` and other methods that take ``FeatureSet``
    objects as input. It iterates over examples in the same way as other
    ``Reader`` classes, but uses a list of example dictionaries instead of
    a path to a file.

    Parameters
    ----------
    path_or_list : Union[:class:`skll.types.PathOrStr`, List[Dict[str, Any]]
        Path or a list of example dictionaries.

    quiet : bool, default=True
        Do not print "Loading..." status message to stderr.

    ids_to_floats : bool, default=False
        Convert IDs to float to save memory. Will raise error
        if we encounter an a non-numeric ID.

    label_col : Optional[str], default='y'
        Name of the column which contains the class labels
        for ARFF/CSV/TSV files. If no column with that name
        exists, or ``None`` is specified, the data is
        considered to be unlabelled.

    id_col : str, default='id'
        Name of the column which contains the instance IDs.
        If no column with that name exists, or ``None`` is
        specified, example IDs will be automatically generated.

    class_map : Optional[:class:`skll.types.ClassMap`], default=None
        Mapping from original class labels to new ones. This is
        mainly used for collapsing multiple labels into a single
        class. Anything not in the mapping will be kept the same.
        The keys are the new labels and the list of values for each
        key is the labels to be collapsed to said new label.

    sparse : bool, default=True
        Whether or not to store the features in a numpy CSR
        matrix when using a DictVectorizer to vectorize the
        features.

    feature_hasher : bool, default=False
        Whether or not a FeatureHasher should be used to
        vectorize the features.

    num_features : Optional[int], default=None
        If using a FeatureHasher, how many features should the
        resulting matrix have?  You should set this to a power
        of 2 greater than the actual number of features to
        avoid collisions.

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.
    """

    def read(self) -> FeatureSet:
        """
        Read examples from list of dictionaries.

        Returns
        -------
        :class:`skll.data.FeatureSet`
            A ``FeatureSet`` representing the list of dictionaries we read in.
        """
        # if we are in this method, `self.path_or_list` must be a
        # list of dictionaries
        assert isinstance(self.path_or_list, list)

        # initialize some variables
        ids_list: List[IdType] = []
        labels_list: List[Optional[LabelType]] = []
        feat_dicts: FeatureDictList = []

        for example_num, example in enumerate(self.path_or_list):
            curr_id: Union[float, str] = str(example.get("id", f"EXAMPLE_{example_num}"))
            if self.ids_to_floats:
                try:
                    curr_id = float(curr_id)
                except ValueError:
                    raise ValueError(
                        "You set ids_to_floats to true, but ID "
                        f"{curr_id} could not be converted to "
                        f"float in {example}"
                    )
            class_name = (
                safe_float(example["y"], replace_dict=self.class_map) if "y" in example else None
            )
            example = example["x"]

            # Update lists of IDs, labels, and feature dictionaries
            if self.ids_to_floats:
                try:
                    curr_id = float(curr_id)
                except ValueError:
                    raise ValueError(
                        "You set ids_to_floats to true, but ID "
                        f"{curr_id} could not be converted to "
                        f"float in {self.path_or_list}"
                    )
            ids_list.append(curr_id)
            labels_list.append(class_name)
            feat_dicts.append(example)

            # Print out status
            if example_num % 100 == 0:
                self._print_progress(example_num)

        # Convert lists to numpy arrays
        ids = np.array(ids_list)
        labels = np.array(labels_list)
        features = self.vectorizer.fit_transform(feat_dicts)

        return FeatureSet(
            "converted", ids, labels=labels, features=features, vectorizer=self.vectorizer
        )


class NDJReader(Reader):
    """
    Create a ``FeatureSet`` instance from a JSONlines/NDJ file.

    If example/instance IDs are included in the files, they
    must be specified as the  "id" key in each JSON dictionary.

    Parameters
    ----------
    path_or_list : Union[:class:`skll.types.PathOrStr`, List[Dict[str, Any]]
        Path or a list of example dictionaries.

    quiet : bool, default=True
        Do not print "Loading..." status message to stderr.

    ids_to_floats : bool, default=False
        Convert IDs to float to save memory. Will raise error
        if we encounter an a non-numeric ID.

    label_col : Optional[str], default='y'
        Name of the column which contains the class labels
        for ARFF/CSV/TSV files. If no column with that name
        exists, or ``None`` is specified, the data is
        considered to be unlabelled.

    id_col : str, default='id'
        Name of the column which contains the instance IDs.
        If no column with that name exists, or ``None`` is
        specified, example IDs will be automatically generated.

    class_map : Optional[:class:`skll.types.ClassMap`], default=None
        Mapping from original class labels to new ones. This is
        mainly used for collapsing multiple labels into a single
        class. Anything not in the mapping will be kept the same.
        The keys are the new labels and the list of values for each
        key is the labels to be collapsed to said new label.

    sparse : bool, default=True
        Whether or not to store the features in a numpy CSR
        matrix when using a DictVectorizer to vectorize the
        features.

    feature_hasher : bool, default=False
        Whether or not a FeatureHasher should be used to
        vectorize the features.

    num_features : Optional[int], default=None
        If using a FeatureHasher, how many features should the
        resulting matrix have?  You should set this to a power
        of 2 greater than the actual number of features to
        avoid collisions.

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.

    """

    def _sub_read(self, file: IO[str]) -> FeatGenerator:
        """
        Iterate through the rows of the file buffer.

        Parameters
        ----------
        file : IO[str]
            A file buffer for an NDJ file.

        Yields
        ------
        curr_id : :class:`skll.types.IdType`
            The current ID for the example.

        class_name : Optional[:class:`skll.types.LabelType`]
            The name of the class label for the example.

        example : :class:`skll.types.FeatureDict`
            The example value in dictionary format, with 'x'
            as list of features.

        Raises
        ------
        ValueError
            If IDs cannot be converted to floats, and ``ids_to_floats``
            is ``True``.
        """
        for example_num, line in enumerate(file):
            # Remove extraneous whitespace
            line = line.strip()

            # If this is a comment line or a blank line, move on
            if line.startswith("//") or not line:
                continue

            # Process good lines
            example = json.loads(line)
            # Convert all IDs to strings initially,
            # for consistency with csv formats.
            curr_id: IdType = str(example.get("id", f"EXAMPLE_{example_num}"))
            class_name: Optional[Union[float, str]] = (
                safe_float(example["y"], replace_dict=self.class_map) if "y" in example else None
            )
            example = example["x"]

            if self.ids_to_floats:
                try:
                    curr_id = float(curr_id)
                except ValueError:
                    raise ValueError(
                        "You set ids_to_floats to true, but ID "
                        f"{curr_id} could not be converted to "
                        "float"
                    )

            yield curr_id, class_name, example


class LibSVMReader(Reader):
    """
    Create a ``FeatureSet`` instance from a LibSVM/LibLinear/SVMLight file.

    We use a specially formatted comment for storing example IDs, class names,
    and feature names, which are normally not supported by the format.  The
    comment is not mandatory, but without it, your labels and features will
    not have names.  The comment is structured as follows::

        ExampleID | 1=FirstClass | 1=FirstFeature 2=SecondFeature

    Parameters
    ----------
    path_or_list : Union[:class:`skll.types.PathOrStr`, List[Dict[str, Any]]
        Path or a list of example dictionaries.

    quiet : bool, default=True
        Do not print "Loading..." status message to stderr.

    ids_to_floats : bool, default=False
        Convert IDs to float to save memory. Will raise error
        if we encounter an a non-numeric ID.

    label_col : Optional[str], default='y'
        Name of the column which contains the class labels
        for ARFF/CSV/TSV files. If no column with that name
        exists, or ``None`` is specified, the data is
        considered to be unlabelled.

    id_col : str, default='id'
        Name of the column which contains the instance IDs.
        If no column with that name exists, or ``None`` is
        specified, example IDs will be automatically generated.

    class_map : Optional[:class:`skll.types.ClassMap`], default=None
        Mapping from original class labels to new ones. This is
        mainly used for collapsing multiple labels into a single
        class. Anything not in the mapping will be kept the same.
        The keys are the new labels and the list of values for each
        key is the labels to be collapsed to said new label.

    sparse : bool, default=True
        Whether or not to store the features in a numpy CSR
        matrix when using a DictVectorizer to vectorize the
        features.

    feature_hasher : bool, default=False
        Whether or not a FeatureHasher should be used to
        vectorize the features.

    num_features : Optional[int], default=None
        If using a FeatureHasher, how many features should the
        resulting matrix have?  You should set this to a power
        of 2 greater than the actual number of features to
        avoid collisions.

    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.
    """

    line_regex = re.compile(
        r"^(?P<label_num>[^ ]+)\s+(?P<features>[^#]*)\s*"
        r"(?P<comments>#\s*(?P<example_id>[^|]+)\s*\|\s*"
        r"(?P<label_map>[^|]+)\s*\|\s*"
        r"(?P<feat_map>.*)\s*)?$",
        flags=re.UNICODE,
    )

    LIBSVM_REPLACE_DICT = {
        "\u2236": ":",
        "\uFF03": "#",
        "\u2002": " ",
        "\ua78a": "=",
        "\u2223": "|",
    }

    @staticmethod
    def _pair_to_tuple(pair: str, feat_map: Dict[str, str]) -> Tuple[str, Union[float, int, str]]:
        """
        Split feature-value pair separated by a colon into tuple.

        Also perform `safe_float` conversion on the value.

        Parameters
        ----------
        pair: str
            A feature-value pair to split.
        feat_map : Dict[str, str]
            Feature name mapping.

        Returns
        -------
        name : str
            The name of the feature.
        value : Union[float, int, str]
            The value of the example.
        """
        name, value = pair.split(":")
        if feat_map is not None:
            ret_name = feat_map[name]
        ret_value = safe_float(value)
        return (ret_name, ret_value)

    def _sub_read(self, file: IO[str]) -> FeatGenerator:
        """
        Parse rows of LibSVM file.

        Parameters
        ----------
        file : IO[str]
            A file buffer for an LibSVM file.

        Yields
        ------
        curr_id : :class:`skll.types.IdType`
            The current ID for the example.

        class_ : :class:`skll.types.LabelType`
            The name of the class label for the example.

        example : :class:`skll.types.FeatureDict`
            The example valued in dictionary format, with 'x'
            as list of features.

        Raises
        ------
        ValueError
            If line does not look like valid libsvm format.
        """
        feat_map: Optional[Dict[str, str]]
        for example_num, line in enumerate(file):
            curr_id = ""
            # Decode line if it's not already str
            if isinstance(line, bytes):
                line = UnicodeDammit(line, ["utf-8", "windows-1252"]).unicode_markup
            match = self.line_regex.search(line.strip())
            if not match:
                raise ValueError("Line does not look like valid libsvm format" f"\n{line}")
            # Metadata is stored in comments if this was produced by SKLL
            if match.group("comments") is not None:
                # Store mapping from feature numbers to names
                if match.group("feat_map"):
                    feat_map = {}
                    for pair in match.group("feat_map").split():
                        number, name = pair.split("=")
                        for orig, replacement in LibSVMReader.LIBSVM_REPLACE_DICT.items():
                            name = name.replace(orig, replacement)
                        feat_map[number] = name
                else:
                    feat_map = None
                # Store mapping from label/class numbers to names
                if match.group("label_map"):
                    label_map = dict(
                        pair.split("=") for pair in match.group("label_map").strip().split()
                    )
                else:
                    label_map = None
                curr_id = match.group("example_id").strip()

            if not curr_id:
                curr_id = f"EXAMPLE_{example_num}"

            # the class can be either a float, an int, or a string;
            # so we can use our `LabelType` type alias for this
            class_: LabelType

            # get the class number
            class_num = match.group("label_num")

            # If we have a mapping from class numbers to labels, get label
            if label_map:
                class_ = label_map[class_num]
            else:
                class_ = class_num
            class_ = safe_float(class_, replace_dict=self.class_map)

            if feat_map:
                curr_info_dict = dict(
                    self._pair_to_tuple(pair, feat_map)
                    for pair in match.group("features").strip().split()
                )
            else:
                curr_info_dict = {}

            yield curr_id, class_, curr_info_dict


class CSVReader(Reader):
    """
    Create a ``FeatureSet`` instance from a CSV file.

    If example/instance IDs are included in the files, they
    must be specified in the ``id`` column.
    Also, there must be a column with the name specified by ``label_col`` if the
    data is labeled.

    Parameters
    ----------
    path_or_list : Union[:class:`skll.types.PathOrStr`, List[Dict[str, Any]]]
        The path to a comma-delimited file.

    replace_blanks_with : Optional[Union[Number, Dict[str, Number]]], default=None
        Specifies a new value with which to replace blank values.
        Options are:

        - ``Number`` : A (numeric) value with which to replace blank values.
        - ``dict`` : A dictionary specifying the replacement value for each column.
        - ``None`` : Blank values will be left as blanks, and not replaced.

        The replacement occurs after the data set is read into a ``pd.DataFrame``.

    drop_blanks : bool, default=False
        If ``True``, remove lines/rows that have any blank
        values. These lines/rows are removed after the
        the data set is read into a ``pd.DataFrame``.

    pandas_kwargs : Optional[Dict[str, Any]], default=None
        Arguments that will be passed directly to the ``pandas`` I/O reader.

    kwargs : Optional[Dict[str, Any]]
        Other arguments to the Reader object.
    """

    def __init__(
        self,
        path_or_list: Union[PathOrStr, List[Dict[str, Any]]],
        replace_blanks_with: Optional[Union[Number, Dict[str, Number]]] = None,
        drop_blanks: bool = False,
        pandas_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize CSVReader class."""
        super(CSVReader, self).__init__(path_or_list, **kwargs)
        self._replace_blanks_with = replace_blanks_with
        self._drop_blanks = drop_blanks
        self._pandas_kwargs = {} if pandas_kwargs is None else pandas_kwargs
        self._sep = self._pandas_kwargs.pop("sep", str(","))
        self._engine = self._pandas_kwargs.pop("engine", "c")
        self._use_pandas = True

    def _sub_read(self, file: PathOrStr) -> Tuple[np.ndarray, np.ndarray, FeatureDictList]:
        """
        Parse rows of CSV file.

        Parameters
        ----------
        file : :class:`skll.types.PathOrStr`
            The path to the CSV file.

        Returns
        -------
        ids : numpy.ndarray of shape (n_ids,)
            The ids for the feature set.

        labels : numpy.ndarray of shape (n_labels,)
            The labels for the feature set.

        features : :class:`skll.types.FeatureDictList`
            The list of feature dictionaries for the feature set.
        """
        df = pd.read_csv(file, sep=self._sep, engine=self._engine, **self._pandas_kwargs)
        return self._parse_dataframe(
            df,
            self.id_col,
            self.label_col,
            replace_blanks_with=self._replace_blanks_with,
            drop_blanks=self._drop_blanks,
        )


class TSVReader(CSVReader):
    """
    Create a ``FeatureSet`` instance from a TSV file.

    If example/instance IDs are included in the files, they
    must be specified in the ``id`` column.
    Also there must be a column with the name specified by ``label_col``
    if the data is labeled.

    Parameters
    ----------
    path_or_list : str
        The path to a comma-delimited file.

    replace_blanks_with : Optional[Union[Number, Dict[str, Number]]], default=None
        Specifies a new value with which to replace blank values.
        Options are:

            - ``Number`` : A (numeric) value with which to replace blank values.
            - ``dict`` : A dictionary specifying the replacement value for each column.
            - ``None`` : Blank values will be left as blanks, and not replaced.

        The replacement occurs after the data set is read into a ``pd.DataFrame``.

    drop_blanks : bool, default=False
        If ``True``, remove lines/rows that have any blank values. These
        lines/rows are removed after the the data set is read into a
        ``pd.DataFrame``.

    pandas_kwargs : Optional[Dict[str, Any]], default=None
        Arguments that will be passed directly to the ``pandas`` I/O reader.

    kwargs : Optional[Dict[str, Any]]
        Other arguments to the Reader object.
    """

    def __init__(
        self,
        path_or_list: Union[PathOrStr, List[Dict[str, Any]]],
        replace_blanks_with: Optional[Union[Number, Dict[str, Number]]] = None,
        drop_blanks: bool = False,
        pandas_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize TSVReader class."""
        super(TSVReader, self).__init__(
            path_or_list,
            replace_blanks_with=replace_blanks_with,
            drop_blanks=drop_blanks,
            pandas_kwargs=pandas_kwargs,
            **kwargs,
        )
        self._sep = str("\t")


class ARFFReader(Reader):
    """
    Create a ``FeatureSet`` instance from an ARFF file.

    If example/instance IDs are included in the files, they
    must be specified in the ``id`` column.
    Also, there must be a column with the name specified by ``label_col`` if the
    data is labeled, and this column must be the final one (as it is in Weka).

    Parameters
    ----------
    path_or_list : Union[:class:`skll.types.PathOrStr`, List[Dict[str, Any]]]
        The path to the ARFF file.

    kwargs : Optional[Dict[str, Any]]
        Other arguments to the Reader object.
    """

    def __init__(self, path_or_list: Union[PathOrStr, List[Dict[str, Any]]], **kwargs):
        """Initialize ARFFReader class."""
        super(ARFFReader, self).__init__(path_or_list, **kwargs)
        self.dialect = "arff"
        self.relation = ""
        self.regression = False

    @staticmethod
    def split_with_quotes(
        string: str, delimiter: str = " ", quote_char: str = "'", escape_char: str = "\\"
    ) -> List[str]:
        r"""
        Split strings but not on split delimiters enclosed in quotes.

        Parameters
        ----------
        string : str
            The string with quotes to split

        delimiter : str, default=' '
            The delimiter to split on.

        quote_char : str, default="'"
            The quote character to ignore.

        escape_char : str, default='\\'
            The escape character.
        """
        return next(
            csv.reader([string], delimiter=delimiter, quotechar=quote_char, escapechar=escape_char)
        )

    def _sub_read(self, file: IO[str]) -> FeatGenerator:
        """
        Parse rows of ARFF file.

        Parameters
        ----------
        file : IO[str]
            A file buffer for the ARFF file.

        Yields
        ------
        curr_id : :class:`skll.types.IdType`
            The current ID for the example.

        class_name : :class:`skll.types.LabelType`
            The name of the class label for the example.

        example : :class:`skll.types.FeatureDict`
            The example features in dictionary format.
        """
        field_names = []
        # Process ARFF header
        for line in file:
            # Process encoding
            if not isinstance(line, str):
                decoded_line = UnicodeDammit(line, ["utf-8", "windows-1252"]).unicode_markup
            else:
                decoded_line = line
            line = decoded_line.strip()
            # Skip empty lines
            if line:
                # Split the line using CSV reader because it can handle
                # quoted delimiters.
                split_header = self.split_with_quotes(line)
                row_type = split_header[0].lower()
                if row_type == "@attribute":
                    # Add field name to list
                    field_name = split_header[1]
                    field_names.append(field_name)
                    # Check if we're doing regression
                    if field_name == self.label_col:
                        self.regression = len(split_header) > 2 and split_header[2] == "numeric"
                # Save relation if specified
                elif row_type == "@relation":
                    self.relation = split_header[1]
                # Stop at data
                elif row_type == "@data":
                    break
                # Skip other types of rows (relations)

        # Create header for CSV
        io_type = StringIO
        with io_type() as field_buffer:
            csv.writer(field_buffer, dialect="arff").writerow(field_names)
            field_str = field_buffer.getvalue()

        # Set label_col to be the name of the last field, since that's standard
        # for ARFF files
        if self.label_col != field_names[-1]:
            self.label_col = None

        # Process iterator as a CSV file
        csv_file_buffer = chain([field_str], file)
        reader = csv.DictReader(csv_file_buffer, dialect=self.dialect)
        for example_num, row in enumerate(reader):
            if self.label_col is not None and self.label_col in row:
                class_name = safe_float(row[self.label_col], replace_dict=self.class_map)
                del row[self.label_col]
            else:
                class_name = None

            if self.id_col not in row:
                curr_id = f"EXAMPLE_{example_num}"
            else:
                curr_id = row[self.id_col]
                del row[self.id_col]

            # Convert features to floats and if a feature is 0
            # then store the name of the feature so we can
            # delete it later since we don't need to explicitly
            # store zeros in the feature hash
            columns_to_delete = []
            for fname, fval in row.items():
                fval_float = safe_float(fval)
                # we don't need to explicitly store zeros
                if fval_float:
                    row[fname] = fval_float
                else:
                    columns_to_delete.append(fname)

            # remove the columns with zero values
            for cname in columns_to_delete:
                del row[cname]

            yield curr_id, class_name, row


def safe_float(
    text: Any,
    replace_dict: Optional[Dict[str, List[str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Union[float, int, str]:
    """
    Convert string to a float.

    It first tries to convert to an integer and then to a float if that fails.
    If neither is possible, the originals string value is returned.

    Parameters
    ----------
    text : Any
        The text to convert.

    replace_dict : Optional[Dict[str, List[str]]], default=None
        Mapping from text to replacement text values. This is
        mainly used for collapsing multiple labels into a
        single class. Replacing happens before conversion to
        floats. Anything not in the mapping will be kept the
        same.

    logger : Optional[logging.Logger], default=None
        The Logger instance to use to log messages. Used instead of
        creating a new Logger instance by default.

    Returns
    -------
    Union[float, int, str]
        The text value converted to int or float, if possible. Otherwise
        it's a string.
    """
    # convert to str to be "Safe"!
    text = str(text)

    # get a logger unless we are passed one
    if not logger:
        logger = logging.getLogger(__name__)

    if replace_dict is not None:
        if text in replace_dict:
            text = replace_dict[text]
        else:
            logger.warning(
                "Encountered value that was not in replacement "
                f"dictionary (e.g., class_map): {text}"
            )
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text
        except TypeError:
            return 0.0
    except TypeError:
        return 0


# Constants
EXT_TO_READER = {
    ".arff": ARFFReader,
    ".csv": CSVReader,
    ".jsonlines": NDJReader,
    ".libsvm": LibSVMReader,
    ".ndj": NDJReader,
    ".tsv": TSVReader,
}
