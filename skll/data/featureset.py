# License: BSD 3 clause
"""
Classes related to storing/merging feature sets.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:organization: ETS
"""

from copy import deepcopy
from typing import Collection, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

from skll.data.dict_vectorizer import DictVectorizer as SkllDictVectorizer
from skll.types import FeatGenerator, FeatureDictList, IdType, LabelType, SparseFeatureMatrix


class FeatureSet(object):
    """
    Encapsulate features, labels, and metadata for a given dataset.

    Parameters
    ----------
    name : str
        The name of this feature set.

    ids : Union[List[str], numpy.ndarray]
        Example IDs for this set.

    labels : Optional[Union[List[str], numpy.ndarray], default=None
        Labels for this set.

    features : Optional[Union[:class:`skll.types.FeatureDictList`, :class:`numpy.ndarray`]], default=None
        The features for each instance represented as either a
        list of dictionaries or a numpy array (if ``vectorizer`` is
        also specified).

    vectorizer : Optional[Union[:class:`sklearn.feature_extraction.DictVectorizer`, :class:`sklearn.feature_extraction.FeatureHasher`], default=None
        Vectorizer which will be used to generate the feature matrix.

    Warnings
    --------
    FeatureSets can only be equal if the order of the instances is
    identical because these are stored as lists/arrays. Since scikit-learn's
    ``DictVectorizer`` automatically sorts the underlying feature matrix
    if it is sparse, we do not do any sorting before checking for equality.
    This is not a problem because we _always_ use sparse matrices with
    ``DictVectorizer`` when creating FeatureSets.

    Notes
    -----
    If ids, labels, and/or features are not None, the number of rows in
    each array must be equal.

    """

    def __init__(
        self,
        name: str,
        ids: Union[List[str], np.ndarray],
        labels: Optional[Union[List[str], np.ndarray]] = None,
        features: Optional[Union[FeatureDictList, SparseFeatureMatrix]] = None,
        vectorizer: Optional[Union[DictVectorizer, FeatureHasher]] = None,
    ):
        """Initialize a FeatureSet instance."""
        super(FeatureSet, self).__init__()

        # clearly define the attribute types
        self.ids: np.ndarray
        self.labels: Optional[np.ndarray]
        self.features: Optional[SparseFeatureMatrix]
        self.vectorizer: Optional[Union[DictVectorizer, FeatureHasher]]

        self.name = name

        if isinstance(ids, list):
            self.ids = np.array(ids)
        elif isinstance(ids, np.ndarray):
            self.ids = ids
        else:
            raise ValueError("Ids must be a list or numpy array.")

        if isinstance(labels, list):
            labels = np.array(labels)
        self.labels = labels

        self.vectorizer = vectorizer

        # convert features from list of dictionaries to sparse array, if needed
        if isinstance(features, list):
            if self.vectorizer is None:
                self.vectorizer = SkllDictVectorizer(sparse=True)
            features_array: SparseFeatureMatrix = self.vectorizer.fit_transform(features)
            self.features = features_array
        else:
            self.features = features

        if self.features is not None:
            num_feats = self.features.shape[0]
            num_ids = self.ids.shape[0]
            if num_feats != num_ids:
                raise ValueError(
                    f"Number of IDs ({num_ids}) does not equal "
                    f"number of feature rows ({num_feats})"
                )
            if self.labels is None:
                self.labels = np.empty(num_feats)
                self.labels.fill(None)
            num_labels = self.labels.shape[0]
            if num_feats != num_labels:
                raise ValueError(
                    f"Number of labels ({num_labels}) does not "
                    f"equal number of feature rows ({num_feats})"
                )

    def __contains__(self, value):
        """
        Check if example ID is in the FeatureSet.

        Parameters
        ----------
        value
            The value to check.

        """
        return value in self.ids

    def __eq__(self, other):
        """
        Check whether two featuresets are the same.

        Parameters
        ----------
        other : :class:`skll.data.featureset.FeatureSet`
            The other ``FeatureSet`` to check equivalence with.

        Returns
        -------
        bool
            ``True`` if they are the same, ``False`` otherwise.

        Notes
        -----
        We consider feature values to be equal if any differences are in the
        sixth decimal place or higher.

        """
        return (
            self.ids.shape == other.ids.shape
            and self.labels.shape == other.labels.shape
            and self.features.shape == other.features.shape
            and (self.ids == other.ids).all()
            and (self.labels == other.labels).all()
            and np.allclose(self.features.data, other.features.data, rtol=1e-6)
            and (self.features.indices == other.features.indices).all()
            and (self.features.indptr == other.features.indptr).all()
            and self.vectorizer == other.vectorizer
        )

    def __iter__(self):
        """Iterate through (ID, label, feature_dict) tuples in feature set."""
        if self.features is not None:
            if not isinstance(self.vectorizer, DictVectorizer):
                raise ValueError(
                    "FeatureSets can only be iterated through if "
                    "they use a DictVectorizer for their feature "
                    "vectorizer."
                )
            for id_, label_, feats in zip(self.ids, self.labels, self.features):
                # reshape to a 2D matrix if we are not using a sparse matrix
                # to store the features
                feats = feats.reshape(1, -1) if not sp.issparse(feats) else feats

                # When calling inverse_transform we have to add [0] to get the
                # results for the current instance because it always returns a
                # 2D array
                yield (id_, label_, self.vectorizer.inverse_transform(feats)[0])
        else:
            return

    def __len__(self) -> int:
        """Return number of rows in the ``FeatureSet`` instance."""
        return self.features.shape[0] if self.features is not None else 0

    def __add__(self, other: "FeatureSet") -> "FeatureSet":
        """
        Combine two feature sets to create a new one.

        The combination is done assuming they both have the same instances
        with the same IDs in the same order.

        Parameters
        ----------
        other : :class:`skll.data.featureset.FeatureSet`
            The other ``FeatureSet`` to add to this one.

        Returns
        -------
        :class:`skll.data.featureset.FeatureSet
            The combined feature set.

        Raises
        ------
        ValueError
            If IDs are not in the same order in each ``FeatureSet`` instance.

        ValueError
            If either the 'features' or 'vectorizer' attributes are
            ``None`` for either of the two ``FeatureSet`` instances.

        ValueError
            If vectorizers are different between the two ``FeatureSet`` instances.

        ValueError
            If there are duplicate feature names.

        ValueError
            If there are conflicting labels.

        """
        # Check that the sets of IDs are equal
        if set(self.ids) != set(other.ids):
            raise ValueError("IDs are not in the same order in each " "feature set")
        # Compute the relative ordering of IDs for merging the features
        # and labels.
        ids_indices = dict((y, x) for x, y in enumerate(other.ids))
        relative_order = [ids_indices[self_id] for self_id in self.ids]

        # Initialize the new feature set with a name and the IDs.
        new_set = FeatureSet("+".join(sorted([self.name, other.name])), deepcopy(self.ids))

        # Make sure that features and vectorizer in either feature set are not None
        if (
            self.vectorizer is None
            or other.vectorizer is None
            or self.features is None
            or other.features is None
        ):
            raise ValueError(
                "Cannot combine FeatureSets since either the vectorizer "
                "or the features are not defined."
            )

        # Make sure the two vectorizers are the same type
        if not isinstance(self.vectorizer, type(other.vectorizer)):
            raise ValueError(
                "Cannot combine FeatureSets because they are "
                "not both using the same type of feature "
                "vectorizer (e.g., DictVectorizer, "
                "FeatureHasher)"
            )
        # they have to be the same types in this block
        else:
            uses_feature_hasher = isinstance(self.vectorizer, FeatureHasher)
            if uses_feature_hasher:
                if self.vectorizer.n_features != other.vectorizer.n_features:
                    raise ValueError(
                        "Cannot combine FeatureSets that use "
                        "FeatureHashers with different values of "
                        "n_features setting."
                    )
            else:
                # Check for duplicate feature names.
                if set(self.vectorizer.feature_names_) & set(other.vectorizer.feature_names_):
                    raise ValueError(
                        "Cannot combine FeatureSets because they have duplicate feature names."
                    )
            num_feats = self.features.shape[1]

            new_set.features = sp.hstack([self.features, other.features[relative_order]], "csr")
            new_set.vectorizer = deepcopy(self.vectorizer)
            if not uses_feature_hasher:
                for feat_name, index in other.vectorizer.vocabulary_.items():
                    new_set.vectorizer.vocabulary_[feat_name] = index + num_feats
                other_names = other.vectorizer.feature_names_
                new_set.vectorizer.feature_names_.extend(other_names)

            # If either set has labels, check that they don't conflict.
            if self.has_labels:
                # labels should be the same for each FeatureSet, so store once.
                conflicts = not np.all(self.labels == other.labels[relative_order])  # type: ignore
                if other.has_labels and conflicts:
                    raise ValueError(
                        "Feature sets have conflicting labels for examples with the same ID."
                    )
                new_set.labels = deepcopy(self.labels)
            else:
                labels = other.labels
                if other.has_labels:
                    labels = deepcopy(other.labels[relative_order])  # type: ignore
                new_set.labels = labels

        return new_set

    def filter(
        self,
        ids: Optional[List[IdType]] = None,
        labels: Optional[List[LabelType]] = None,
        features: Optional[List[str]] = None,
        inverse: bool = False,
    ) -> None:
        """
        Remove or keep features and/or examples from the given feature set.

        Filtering is done in-place.

        Parameters
        ----------
        ids : Optional[List[:class:`skll.types.IdType`]], default=None
            Examples to keep in the FeatureSet. If ``None``, no ID
            filtering takes place.

        labels : Optional[List[:class:`skll.types.LabelType`]], default=None
            Labels that we want to retain examples for. If ``None``,
            no label filtering takes place.

        features : Optional[List[str]], default=None
            Features to keep in the FeatureSet. To help with
            filtering string-valued features that were converted
            to sequences of boolean features when read in, any
            features in the FeatureSet that contain a ``=`` will be
            split on the first occurrence and the prefix will be
            checked to see if it is in ``features``.
            If ``None``, no feature filtering takes place.
            Cannot be used if FeatureSet uses a FeatureHasher for
            vectorization.

        inverse : bool, default=False
            Instead of keeping features and/or examples in lists,
            remove them.

        Raises
        ------
        ValueError
            If attempting to use features to filter a ``FeatureSet`` that
            uses a ``FeatureHasher`` vectorizer.

        """
        # Construct mask that indicates which examples to keep
        mask = np.ones(len(self), dtype=bool)
        if ids:
            mask = np.logical_and(mask, np.in1d(self.ids, ids))
        if labels and self.labels is not None:
            mask = np.logical_and(mask, np.in1d(self.labels, labels))

        if inverse and (labels is not None or ids is not None):
            mask = np.logical_not(mask)

        # Remove examples not in mask
        self.ids = self.ids[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]
        if self.features is not None:
            self.features = self.features[mask, :]

        # Filter features
        if features and self.features is not None and self.vectorizer is not None:
            if isinstance(self.vectorizer, FeatureHasher):
                raise ValueError(
                    "FeatureSets with FeatureHasher vectorizers cannot be filtered by feature."
                )
            columns = np.array(
                sorted(
                    {
                        feat_num
                        for feat_name, feat_num in self.vectorizer.vocabulary_.items()
                        if (feat_name in features or feat_name.split("=", 1)[0] in features)
                    }
                )
            )
            if inverse:
                all_columns = np.arange(self.features.shape[1])
                columns = all_columns[np.logical_not(np.in1d(all_columns, columns))]
            self.features = self.features[:, columns]
            self.vectorizer.restrict(columns, indices=True)

    def filtered_iter(
        self,
        ids: Optional[List[IdType]] = None,
        labels: Optional[List[LabelType]] = None,
        features: Optional[Collection[str]] = None,
        inverse: bool = False,
    ) -> FeatGenerator:
        """
        Retain only the specified features and/or examples from the output.

        Parameters
        ----------
        ids : Optional[List[:class:`skll.types.IdType`]], default=None
            Examples to keep in the ``FeatureSet``. If ``None``, no ID
            filtering takes place.

        labels : Optional[List[:class:`skll.types.LabelType`]], default=None
            Labels that we want to retain examples for. If ``None``,
            no label filtering takes place.

        features : Optional[Collection[str]], default=None
            Features to keep in the ``FeatureSet``. To help with
            filtering string-valued features that were converted
            to sequences of boolean features when read in, any
            features in the ``FeatureSet`` that contain a `=` will be
            split on the first occurrence and the prefix will be
            checked to see if it is in ``features``.
            If `None`, no feature filtering takes place.
            Cannot be used if ``FeatureSet`` uses a FeatureHasher for
            vectorization.

        inverse : bool, default=False
            Instead of keeping features and/or examples in lists,
            remove them.

        Returns
        -------
        :class:`skll.types.FeatGenerator`

            A generator that yields 3-tuples containing:

              - :class:`skll.types.IdType`  - The ID of the example.

              - :class:`skll.types.LabelType` - The label of the example.

              - :class:`skll.types.FeatureDict` - The feature dictionary, with
                feature name as the key and example value as the value.

        Raises
        ------
        ValueError
            If the vectorizer is not a ``DictVectorizer``.

        ValueError
            If any of the "labels", "features", or "vectorizer" attribute
            is ``None``.

        """
        if self.features is not None and not isinstance(self.vectorizer, DictVectorizer):
            raise ValueError(
                "FeatureSets can only be iterated through if they"
                " use a DictVectorizer for their feature "
                "vectorizer."
            )

        if self.labels is None or self.features is None or self.vectorizer is None:
            raise ValueError("Cannot filter featureset with no labels, features, or vectorizer.")
        else:
            for id_, label_, feats in zip(self.ids, self.labels, self.features):
                # Skip instances with IDs not in filter
                if ids is not None and (id_ in ids) == inverse:
                    continue
                # Skip instances with labels not in filter
                if labels is not None and (label_ in labels) == inverse:
                    continue

                # reshape to a 2D matrix if we are not using a sparse matrix
                # to store the features
                feats = feats.reshape(1, -1) if not sp.issparse(feats) else feats
                feat_dict = self.vectorizer.inverse_transform(feats)[0]
                if features is not None:
                    feat_dict = {
                        name: value
                        for name, value in feat_dict.items()
                        if (inverse != (name in features or name.split("=", 1)[0] in features))
                    }
                elif not inverse:
                    feat_dict = {}
                yield id_, label_, feat_dict

    def __sub__(self, other: "FeatureSet") -> "FeatureSet":
        """
        Subset ``FeatureSet`` instance by removing all features from ``other`` instance.

        Parameters
        ----------
        other : :class:`skll.data.featureset.FeatureSet`
            The other ``FeatureSet`` containing the features that should
            be removed from this ``FeatureSet``.

        Returns
        -------
        :class:`skll.data.featureset.FeatureSet`
            A copy of ``self`` with all features in ``other`` removed.

        """
        new_set = deepcopy(self)
        if other.vectorizer:
            new_set.filter(features=other.vectorizer.feature_names_, inverse=True)
        return new_set

    @property
    def has_labels(self):
        """
        Check if ``FeatureSet`` has finite labels.

        Returns
        -------
        has_labels : bool
            Whether or not this FeatureSet has any finite labels.

        """
        # make sure that labels is not None or a list of Nones
        if self.labels is not None and not all(label is None for label in self.labels):
            # then check that they are not a list of NaNs
            return not (
                np.issubdtype(self.labels.dtype, np.floating) and np.isnan(np.min(self.labels))
            )
        else:
            return False

    def __str__(self):
        """
        Return a string representation of ``FeatureSet``.

        Returns
        -------
        str:
            A string representation of ``FeatureSet``.

        """
        return str(self.__dict__)

    def __repr__(self):
        """
        Return a string representation of ``FeatureSet``.

        Returns
        -------
        str:
            A string representation of ``FeatureSet``.

        """
        return repr(self.__dict__)

    def __getitem__(
        self, value: Union[int, slice]
    ) -> Union["FeatureSet", Tuple[IdType, LabelType, FeatureDictList]]:
        """
        Get new feature subset or specific example.

        Parameters
        ----------
        value: Union[int, slice]
            The value to use for retrieval. This can either be a slice or
            an index.

        Returns
        -------
        Union[:class:`skll.data.featureset.FeatureSet`, Tuple[:class:`skll.types.IdType`, :class:`skll.types.LabelType`, :class:`skll.types.FeatureDictList`]]  # noqa: E501
            If `value` is a slice, then return a new ``FeatureSet`` instance
            containing a subset of the data. If it's an index, return the
            specific example by row number.

        """
        # Check if we're slicing
        if isinstance(value, slice):
            sliced_ids = self.ids[value]
            sliced_feats = self.features[value] if self.features is not None else None
            sliced_labels = self.labels[value] if self.labels is not None else None
            return FeatureSet(
                f"{self.name}_{value}",
                sliced_ids,
                features=sliced_feats,
                labels=sliced_labels,
                vectorizer=self.vectorizer,
            )
        else:
            label = self.labels[value] if self.labels is not None else ""
            if self.features is not None and self.vectorizer:
                submatrix = self.features[value, :]
                features = self.vectorizer.inverse_transform(submatrix)[0]
            else:
                features = [{}]
            return self.ids[value], label, features

    @staticmethod
    def split(
        fs: "FeatureSet", ids_for_split1: List[int], ids_for_split2: Optional[List[int]] = None
    ) -> Tuple["FeatureSet", "FeatureSet"]:
        """
        Split ``FeatureSet`` into two new ``FeatureSet`` instances.

        The splitting is done based on the given indices for the two splits.

        Parameters
        ----------
        fs : skll.data.featureset.FeatureSet
            The ``FeatureSet`` instance to split.

        ids_for_split1 : List[int]
            A list of example indices which will be split out into
            the first ``FeatureSet`` instance. Note that the
            FeatureSet instance will respect the order of the
            specified indices.

        ids_for_split2 : Optional[List[int]], default=None
            An optional list of example indices which will be
            split out into the second ``FeatureSet`` instance.
            Note that the ``FeatureSet`` instance will respect
            the order of the specified indices. If this is
            not specified, then the second ``FeatureSet``
            instance will contain the complement of the
            first set of indices sorted in ascending order.

        Returns
        -------
        Tuple[:class:`skll.data.featureset.FeatureSet`, :class:`skll.data.featureset.FeatureSet`]
            A tuple containing the two featureset instances.

        """
        # Note: an alternative way to implement this is to make copies
        # of the given FeatureSet instance and then use the `filter()`
        # method but that wastes too much memory since it requires making
        # two copies of the original FeatureSet which may be huge. With
        # the current implementation, we are creating new objects but
        # they should be much smaller than the original FeatureSet.
        ids1 = fs.ids[ids_for_split1]
        labels1 = fs.labels[ids_for_split1] if fs.labels is not None else None
        features1 = fs.features[ids_for_split1] if fs.features is not None else None

        # if ids_for_split2 is not given, it will be the complement of ids_split1
        if ids_for_split2 is None:
            ids_for_split2 = [ind for ind in range(len(fs.ids)) if ind not in ids_for_split1]

        ids2 = fs.ids[ids_for_split2]
        labels2 = fs.labels[ids_for_split2] if fs.labels is not None else None
        features2 = fs.features[ids_for_split2] if fs.features is not None else None

        fs1 = FeatureSet(
            f"{fs.name}_1", ids1, labels=labels1, features=features1, vectorizer=fs.vectorizer
        )
        fs2 = FeatureSet(
            f"{fs.name}_2", ids2, labels=labels2, features=features2, vectorizer=fs.vectorizer
        )
        return fs1, fs2

    @staticmethod
    def from_data_frame(
        df: DataFrame,
        name: str,
        labels_column: Optional[str] = None,
        vectorizer: Optional[Union[DictVectorizer, FeatureHasher]] = None,
    ) -> "FeatureSet":
        """
        Create a ``FeatureSet`` instance from a pandas data frame.

        Will raise an Exception if pandas is not installed in your environment.
        The ``ids`` in the ``FeatureSet`` will be the index from the given frame.

        Parameters
        ----------
        df : pandas.DataFrame
            The pandas.DataFrame object to use as a ``FeatureSet``.

        name : str
            The name of the output ``FeatureSet`` instance.

        labels_column : Optional[str], default=None
            The name of the column containing the labels (data to predict).

        vectorizer : Optional[Union[:class:`sklearn.feature_extraction.DictVectorizer`, :class:`sklearn.feature_extraction.FeatureHasher`]], default=None
            Vectorizer which will be used to generate the feature matrix.

        Returns
        -------
        :class:`skll.data.featureset.FeatureSet`
            A ``FeatureSet`` instance generated from from the given data frame.

        """
        if labels_column:
            feature_columns = [column for column in df.columns if column != labels_column]
            labels = df[labels_column].tolist()
        else:
            feature_columns = df.columns
            labels = None

        features = df[feature_columns].to_dict(orient="records")
        return FeatureSet(
            name, ids=df.index.tolist(), labels=labels, features=features, vectorizer=vectorizer
        )
