# License: BSD 3 clause
"""
Classes related to storing/merging feature sets.

:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
"""

from __future__ import absolute_import, print_function, unicode_literals

from copy import deepcopy

import numpy as np
import scipy.sparse as sp
from six import iteritems
from six.moves import zip
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

from skll.data.dict_vectorizer import DictVectorizer as NewDictVectorizer


class FeatureSet(object):

    """
    Encapsulation of all of the features, values, and metadata about a given
    set of data.

    .. warning::
        FeatureSets can only be equal if the order of the instances is
        identical because these are stored as lists/arrays.

    This replaces ``ExamplesTuple`` from older versions.

    :param name: The name of this feature set.
    :type name: str
    :param ids: Example IDs for this set.
    :type ids: np.array
    :param labels: labels for this set.
    :type labels: np.array
    :param features: The features for each instance represented as either a
                     list of dictionaries or an array-like (if `vectorizer` is
                     also specified).
    :type features: list of dict or array-like
    :param vectorizer: Vectorizer which will be used to generate the feature matrix.
    :type vectorizer: DictVectorizer or FeatureHasher

    .. note::
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
                raise ValueError(('Number of IDs (%s) does not equal '
                                  'number of feature rows (%s)') % (num_ids,
                                                                    num_feats))
            if self.labels is None:
                self.labels = np.empty(num_feats)
                self.labels.fill(None)
            num_labels = self.labels.shape[0]
            if num_feats != num_labels:
                raise ValueError(('Number of labels (%s) does not equal '
                                  'number of feature rows (%s)') % (num_labels,
                                                                    num_feats))

    def __contains__(self, value):
        """
        Check if example ID is in set
        """
        return value in self.ids

    def __eq__(self, other):
        """
        Check whether two featuresets are the same.

        .. note::
           We consider feature values to be equal if any differences are in the
           sixth decimal place or higher.
        """

        # We need to sort the indices for the underlying
        # feature sparse matrix in case we haven't done
        # so already.
        if not self.features.has_sorted_indices:
            self.features.sort_indices()
        if not other.features.has_sorted_indices:
            other.features.sort_indices()

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
            for id_, label_, feats in zip(self.ids, self.labels,
                                          self.features):
                # When calling inverse_transform we have to add [0] to get the
                # results for the current instance because it always returns a
                # 2D array
                yield (id_, label_,
                       self.vectorizer.inverse_transform(feats)[0])
        else:
            return

    def __len__(self):
        return self.features.shape[0]

    def __add__(self, other):
        """
        Combine two feature sets to create a new one.  This is done assuming
        they both have the same instances with the same IDs in the same order.
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
        Removes or keeps features and/or examples from the Featureset depending
        on the passed in parameters.

        :param ids: Examples to keep in the FeatureSet. If `None`, no ID
                    filtering takes place.
        :type ids: list of str/float
        :param labels: labels that we want to retain examples for. If `None`,
                        no label filtering takes place.
        :type labels: list of str/float
        :param features: Features to keep in the FeatureSet. To help with
                         filtering string-valued features that were converted
                         to sequences of boolean features when read in, any
                         features in the FeatureSet that contain a `=` will be
                         split on the first occurrence and the prefix will be
                         checked to see if it is in `features`.
                         If `None`, no feature filtering takes place.
                         Cannot be used if FeatureSet uses a FeatureHasher for
                         vectorization.
        :type features: list of str
        :param inverse: Instead of keeping features and/or examples in lists,
                        remove them.
        :type inverse: bool
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
                                       iteritems(self.vectorizer.vocabulary_)
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
        A version of ``__iter__`` that retains only the specified features
        and/or examples from the output.

        :param ids: Examples in the FeatureSet to keep. If `None`, no ID
                    filtering takes place.
        :type ids: list of str/float
        :param labels: labels that we want to retain examples for. If `None`,
                       no label filtering takes place.
        :type labels: list of str/float
        :param features: Features in the FeatureSet to keep. To help with
                         filtering string-valued features that were converted
                         to sequences of boolean features when read in, any
                         features in the FeatureSet that contain a `=` will be
                         split on the first occurrence and the prefix will be
                         checked to see if it is in `features`.
                         If `None`, no feature filtering takes place.
                         Cannot be used if FeatureSet uses a FeatureHasher for
                         vectorization.
        :type features: list of str
        :param inverse: Instead of keeping features and/or examples in lists,
                        remove them.
        :type inverse: bool
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
            feat_dict = self.vectorizer.inverse_transform(feats)[0]
            if features is not None:
                feat_dict = {name: value for name, value in
                             iteritems(feat_dict) if
                             (inverse != (name in features or
                                          name.split('=', 1)[0] in features))}
            elif not inverse:
                feat_dict = {}
            yield id_, label_, feat_dict

    def __sub__(self, other):
        """
        :returns: a copy of ``self`` with all features in ``other`` removed.
        """
        new_set = deepcopy(self)
        new_set.filter(features=other.vectorizer.feature_names_,
                       inverse=True)
        return new_set

    @property
    def has_labels(self):
        """
        :returns: Whether or not this FeatureSet has any finite labels.
        """
        if self.labels is not None:
            return not (np.issubdtype(self.labels.dtype, float) and
                        np.isnan(np.min(self.labels)))
        else:
            return False

    def __str__(self):
        """
        :returns: a string representation of FeatureSet
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        :returns:  a string representation of FeatureSet
        """
        return repr(self.__dict__)

    def __getitem__(self, value):
        """
        :returns: A specific example by row number, or if given a slice,
                  a new FeatureSet containing a subset of the data.
        """
        # Check if we're slicing
        if isinstance(value, slice):
            sliced_ids = self.ids[value]
            sliced_feats = (self.features[value] if self.features is not None
                            else None)
            sliced_labels = (self.labels[value] if self.labels is not None
                             else None)
            return FeatureSet('{}_{}'.format(self.name, value), sliced_ids,
                              features=sliced_feats, labels=sliced_labels,
                              vectorizer=self.vectorizer)
        else:
            label = self.labels[value] if self.labels is not None else None
            feats = self.features[value, :]
            features = (self.vectorizer.inverse_transform(feats)[0] if
                        self.features is not None else {})
            return self.ids[value], label, features

    @staticmethod
    def from_data_frame(df, name, labels_column=None, vectorizer=None):
        '''
        Helper function to create a FeatureSet object from a `pandas.DataFrame`.
        Will raise an Exception if pandas is not installed in your environment.
        `FeatureSet` `ids` will be the index on `df`.

        :param df: The pandas.DataFrame object you'd like to use as a feature set.
        :type df: pandas.DataFrame
        :param name: The name of this feature set.
        :type name: str
        :param labels_column: The name of the column containing the labels (data to predict).
        :type labels_column: str or None
        :param vectorizer: Vectorizer which will be used to generate the feature matrix.
        :type vectorizer: DictVectorizer or FeatureHasher
        '''
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
