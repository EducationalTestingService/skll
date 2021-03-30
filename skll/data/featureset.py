# License: BSD 3 clause
"""
Classes related to storing/merging feature sets.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:organization: ETS
"""

from copy import deepcopy

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

from skll.data.dict_vectorizer import DictVectorizer as NewDictVectorizer


class FeatureSet(object):

    """
    Encapsulation of all of the features, values, and metadata about a given
    set of data. This replaces ``ExamplesTuple`` from older versions of SKLL.

    Parameters
    ----------
    name : str
        The name of this feature set.

    ids : np.array of shape (n_ids,)
        Example IDs for this set.

    labels : np.array of shape (n_labels,), default=None
        labels for this set.

    feature : list of dict or an array-like of shape (n_samples, n_features), default=None
        The features for each instance represented as either a
        list of dictionaries or an array-like (if ``vectorizer`` is
        also specified).

    vectorizer : DictVectorizer or FeatureHasher, default=None
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

    def __init__(self, name, ids, labels=None, features=None,
                 vectorizer=None):
        super(FeatureSet, self).__init__()
        self.name = name
        if isinstance(ids, list):
            ids = np.array(ids)
        self.ids = ids
        if isinstance(labels, list):
            labels = np.array(labels)
        self.labels = labels
        self.features = features
        self.vectorizer = vectorizer
        # Convert list of dicts to numpy array
        if isinstance(self.features, list):
            if self.vectorizer is None:
                self.vectorizer = NewDictVectorizer(sparse=True)
            self.features = self.vectorizer.fit_transform(self.features)
        if self.features is not None:
            num_feats = self.features.shape[0]
            if self.ids is None:
                raise ValueError('A list of IDs is required')
            num_ids = self.ids.shape[0]
            if num_feats != num_ids:
                raise ValueError(f'Number of IDs ({num_ids}) does not equal '
                                 f'number of feature rows ({num_feats})')
            if self.labels is None:
                self.labels = np.empty(num_feats)
                self.labels.fill(None)
            num_labels = self.labels.shape[0]
            if num_feats != num_labels:
                raise ValueError(f'Number of labels ({num_labels}) does not '
                                 f'equal number of feature rows ({num_feats})')

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
        other : skll.data.FeatureSet
            The other ``FeatureSet`` to check equivalence with.

        Note
        ----
        We consider feature values to be equal if any differences are in the
        sixth decimal place or higher.
        """

        return (self.ids.shape == other.ids.shape and
                self.labels.shape == other.labels.shape and
                self.features.shape == other.features.shape and
                (self.ids == other.ids).all() and
                (self.labels == other.labels).all() and
                np.allclose(self.features.data, other.features.data,
                            rtol=1e-6) and
                (self.features.indices == other.features.indices).all() and
                (self.features.indptr == other.features.indptr).all() and
                self.vectorizer == other.vectorizer)

    def __iter__(self):
        """
        Iterate through (ID, label, feature_dict) tuples in feature set.
        """
        if self.features is not None:
            if not isinstance(self.vectorizer, DictVectorizer):
                raise ValueError('FeatureSets can only be iterated through if '
                                 'they use a DictVectorizer for their feature '
                                 'vectorizer.')
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

    def __len__(self):
        """
        The number of rows in the ``FeatureSet`` instance.
        """
        return self.features.shape[0]

    def __add__(self, other):
        """
        Combine two feature sets to create a new one.  This is done assuming
        they both have the same instances with the same IDs in the same order.

        Parameters
        ----------
        other : skll.data.FeatureSet
            The other ``FeatureSet`` to add to this one.

        Raises
        ------
        ValueError
            If IDs are not in the same order in each ``FeatureSet`` instance.

        ValueError
            If vectorizers are different between the two ``FeatureSet`` instances.

        ValueError
            If there are duplicate feature names.

        ValueError
            If there are conflicting labels.
        """

        # Check that the sets of IDs are equal
        if set(self.ids) != set(other.ids):
            raise ValueError('IDs are not in the same order in each '
                             'feature set')
        # Compute the relative ordering of IDs for merging the features
        # and labels.
        ids_indices = dict((y, x) for x, y in enumerate(other.ids))
        relative_order = [ids_indices[self_id] for self_id in self.ids]

        # Initialize the new feature set with a name and the IDs.
        new_set = FeatureSet('+'.join(sorted([self.name, other.name])),
                             deepcopy(self.ids))

        # Combine feature matrices and vectorizers.
        if not isinstance(self.vectorizer, type(other.vectorizer)):
            raise ValueError('Cannot combine FeatureSets because they are '
                             'not both using the same type of feature '
                             'vectorizer (e.g., DictVectorizer, '
                             'FeatureHasher)')
        uses_feature_hasher = isinstance(self.vectorizer, FeatureHasher)
        if uses_feature_hasher:
            if (self.vectorizer.n_features !=
                    other.vectorizer.n_features):
                raise ValueError('Cannot combine FeatureSets that uses '
                                 'FeatureHashers with different values of '
                                 'n_features setting.')
        else:
            # Check for duplicate feature names.
            if (set(self.vectorizer.feature_names_) &
                    set(other.vectorizer.feature_names_)):
                raise ValueError('Cannot combine FeatureSets because they '
                                 'have duplicate feature names.')
        num_feats = self.features.shape[1]

        new_set.features = sp.hstack([self.features,
                                      other.features[relative_order]],
                                     'csr')
        new_set.vectorizer = deepcopy(self.vectorizer)
        if not uses_feature_hasher:
            for feat_name, index in other.vectorizer.vocabulary_.items():
                new_set.vectorizer.vocabulary_[feat_name] = (index +
                                                             num_feats)
            other_names = other.vectorizer.feature_names_
            new_set.vectorizer.feature_names_.extend(other_names)

        # If either set has labels, check that they don't conflict.
        if self.has_labels:
            # labels should be the same for each FeatureSet, so store once.
            if other.has_labels and \
                    not np.all(self.labels == other.labels[relative_order]):
                raise ValueError('Feature sets have conflicting labels for '
                                 'examples with the same ID.')
            new_set.labels = deepcopy(self.labels)
        else:
            new_set.labels = deepcopy(other.labels[relative_order])

        return new_set

    def filter(self, ids=None, labels=None, features=None, inverse=False):
        """
        Removes or keeps features and/or examples from the `Featureset` depending
        on the parameters. Filtering is done in-place.

        Parameters
        ----------
        ids : list of str/float, default=None
            Examples to keep in the FeatureSet. If ``None``, no ID
            filtering takes place.

        labels : list of str/float, default=None
            Labels that we want to retain examples for. If ``None``,
            no label filtering takes place.

        features : list of str, default=None
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
        if ids is not None:
            mask = np.logical_and(mask, np.in1d(self.ids, ids))
        if labels is not None:
            mask = np.logical_and(mask, np.in1d(self.labels, labels))

        if inverse and (labels is not None or ids is not None):
            mask = np.logical_not(mask)

        # Remove examples not in mask
        self.ids = self.ids[mask]
        self.labels = self.labels[mask]
        self.features = self.features[mask, :]

        # Filter features
        if features is not None:
            if isinstance(self.vectorizer, FeatureHasher):
                raise ValueError('FeatureSets with FeatureHasher vectorizers'
                                 ' cannot be filtered by feature.')
            columns = np.array(sorted({feat_num for feat_name, feat_num in
                                       self.vectorizer.vocabulary_.items()
                                       if (feat_name in features or
                                           feat_name.split('=', 1)[0] in
                                           features)}))
            if inverse:
                all_columns = np.arange(self.features.shape[1])
                columns = all_columns[np.logical_not(np.in1d(all_columns,
                                                             columns))]
            self.features = self.features[:, columns]
            self.vectorizer.restrict(columns, indices=True)

    def filtered_iter(self, ids=None, labels=None, features=None,
                      inverse=False):
        """
        A version of `__iter__` that retains only the specified features
        and/or examples from the output.

        Parameters
        ----------
        ids : list of str/float, default=None
            Examples to keep in the ``FeatureSet``. If ``None``, no ID
            filtering takes place.

        labels : list of str/float, default=None
            Labels that we want to retain examples for. If ``None``,
            no label filtering takes place.

        features : list of str, default=None
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

        Yields
        ------
        id_ : str
            The ID of the example.

        label_ : str
            The label of the example.

        feat_dict : dict
            The feature dictionary, with feature name as the key
            and example value as the value.

        Raises
        ------
        ValueError
            If the vectorizer is not a ``DictVectorizer``.
        """
        if self.features is not None and not isinstance(self.vectorizer,
                                                        DictVectorizer):
            raise ValueError('FeatureSets can only be iterated through if they'
                             ' use a DictVectorizer for their feature '
                             'vectorizer.')

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
                feat_dict = {name: value for name, value in
                             feat_dict.items() if
                             (inverse != (name in features or
                                          name.split('=', 1)[0] in features))}
            elif not inverse:
                feat_dict = {}
            yield id_, label_, feat_dict

    def __sub__(self, other):
        """
        Subset ``FeatureSet`` instance by removing all the features from the
        other ``FeatureSet`` instance.

        Parameters
        ----------
        other : skll.data.FeatureSet
            The other ``FeatureSet`` containing the features that should
            be removed from this ``FeatureSet``.

        Returns
        -------
        A copy of ``self`` with all features in ``other`` removed.
        """
        new_set = deepcopy(self)
        new_set.filter(features=other.vectorizer.feature_names_,
                       inverse=True)
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
            return not (np.issubdtype(self.labels.dtype, np.floating) and
                        np.isnan(np.min(self.labels)))
        else:
            return False

    def __str__(self):
        """
        Returns
        -------
        A string representation of ``FeatureSet``.
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        Returns
        -------
        A string representation of ``FeatureSet``.
        """
        return repr(self.__dict__)

    def __getitem__(self, value):
        """
        Parameters
        ----------
        value
            The value to retrieve.

        Returns
        -------
        A specific example by row number or, if given a slice,
        a new ``FeatureSet`` instance containing a subset of the data.
        """
        # Check if we're slicing
        if isinstance(value, slice):
            sliced_ids = self.ids[value]
            sliced_feats = (self.features[value] if self.features is not None
                            else None)
            sliced_labels = (self.labels[value] if self.labels is not None
                             else None)
            return FeatureSet(f'{self.name}_{value}', sliced_ids,
                              features=sliced_feats, labels=sliced_labels,
                              vectorizer=self.vectorizer)
        else:
            label = self.labels[value] if self.labels is not None else None
            feats = self.features[value, :]
            features = (self.vectorizer.inverse_transform(feats)[0] if
                        self.features is not None else {})
            return self.ids[value], label, features

    @staticmethod
    def split_by_ids(fs, ids_for_split1, ids_for_split2=None):
        """
        Split the ``FeatureSet`` into two new ``FeatureSet`` instances based on
        the given IDs for the two splits.

        Parameters
        ----------
        fs : skll.data.FeatureSet
            The ``FeatureSet`` instance to split.

        ids_for_split1 : list of int
            A list of example IDs which will be split out into
            the first ``FeatureSet`` instance. Note that the
            FeatureSet instance will respect the order of the
            specified IDs.

        ids_for_split2 : list of int, default=None
            An optional ist of example IDs which will be
            split out into the second ``FeatureSet`` instance.
            Note that the ``FeatureSet`` instance will respect
            the order of the specified IDs. If this is
            not specified, then the second ``FeatureSet``
            instance will contain the complement of the
            first set of IDs sorted in ascending order.

        Returns
        -------
        fs1 : skll.data.FeatureSet
            The first ``FeatureSet``.

        fs2 : skll.data.FeatureSet
            The second ``FeatureSet``.
        """

        # Note: an alternative way to implement this is to make copies
        # of the given FeatureSet instance and then use the `filter()`
        # method but that wastes too much memory since it requires making
        # two copies of the original FeatureSet which may be huge. With
        # the current implementation, we are creating new objects but
        # they should be much smaller than the original FeatureSet.
        ids1 = fs.ids[ids_for_split1]
        labels1 = fs.labels[ids_for_split1]
        features1 = fs.features[ids_for_split1]
        if ids_for_split2 is None:
            ids2 = fs.ids[~np.in1d(fs.ids, ids_for_split1)]
            labels2 = fs.labels[~np.in1d(fs.ids, ids_for_split1)]
            features2 = fs.features[~np.in1d(fs.ids, ids_for_split1)]
        else:
            ids2 = fs.ids[ids_for_split2]
            labels2 = fs.labels[ids_for_split2]
            features2 = fs.features[ids_for_split2]

        fs1 = FeatureSet(f'{fs.name}_1',
                         ids1,
                         labels=labels1,
                         features=features1,
                         vectorizer=fs.vectorizer)
        fs2 = FeatureSet(f'{fs.name}_2',
                         ids2,
                         labels=labels2,
                         features=features2,
                         vectorizer=fs.vectorizer)
        return fs1, fs2

    @staticmethod
    def from_data_frame(df, name, labels_column=None, vectorizer=None):
        """
        Helper function to create a ``FeatureSet`` instance from a `pandas.DataFrame`.
        Will raise an Exception if pandas is not installed in your environment.
        The ``ids`` in the ``FeatureSet`` will be the index from the given frame.

        Parameters
        ----------
        df : pd.DataFrame
            The pandas.DataFrame object to use as a ``FeatureSet``.

        name : str
            The name of the output ``FeatureSet`` instance.

        labels_column : str, default=None
            The name of the column containing the labels (data to predict).

        vectorizer : DictVectorizer or FeatureHasher, default=None
            Vectorizer which will be used to generate the feature matrix.

        Returns
        -------
        feature_set : skll.data.FeatureSet
            A ``FeatureSet`` instance generated from from the given data frame.
        """
        if labels_column:
            feature_columns = [column for column in df.columns if column != labels_column]
            labels = df[labels_column].tolist()
        else:
            feature_columns = df.columns
            labels = None

        features = df[feature_columns].to_dict(orient='records')
        return FeatureSet(name,
                          ids=df.index.tolist(),
                          labels=labels,
                          features=features,
                          vectorizer=vectorizer)
