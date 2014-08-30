# License: BSD 3 clause
'''
Classes related to storing/merging feature sets.

:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
'''

from __future__ import absolute_import, print_function, unicode_literals

from collections import namedtuple
from copy import deepcopy
from itertools import repeat
try:
    from collections import Set
except ImportError:
    from collections.abc import Set

import numpy as np
import scipy.sparse as sp
from six import iteritems, PY2, PY3, string_types, text_type
from six.moves import map, zip
from sklearn.feature_extraction import FeatureHasher


class FeatureSet(object):
    """
    Encapsulation of all of the features, values, and metadata about a given
    set of data.

    This replaces ExamplesTuple in older versions.

    :param ids: Example IDs for this set.
    :type ids: np.array
    :param classes: Classes for this set.
    :type classes: np.array
    :param features: Feature matrix as created by the given feature vectorizer.
    :type features: np.array
    :param feat_vectorizer: Vectorizer that created feature matrix.
    :type feat_vectorizer: DictVectorizer or FeatureHasher

    .. note::
       If ids, classes, and/or features are not None, the number of rows in
       each array must be equal.
    """
    def __init__(self, ids=None, classes=None, features=None,
                 feat_vectorizer=None):
        super(FeatureSet, self).__init__()
        self.ids = ids
        self.classes = classes
        self.features = features
        self.feat_vectorizer = feat_vectorizer
        if self.features is not None:
            num_feats = self.features.shape[1]
            if self.ids is None:
                self.ids = np.empty(num_feats)
                self.ids.fill(None)
            if self.classes is None:
                self.classes = np.empty(num_feats)
                self.classes.fill(None)
            num_ids = self.ids.shape[1]
            num_classes = self.classes.shape[1]
            if num_feats != num_ids:
                raise ValueError(('Number of IDs (%s) does not equal number of'
                                  ' features (%s)') % (num_ids, num_feats))
            if num_feats != num_classes:
                raise ValueError(('Number of classes ({}) does not equal '
                                  'number of features ({})') % (num_classes,
                                                                num_feats))
    def __contains__(self, value):
        pass

    def __iter__(self):
        '''
        Iterate through (ID, class, features) tuples in feature set.
        '''
        if self.features is not None:
            return zip(self.ids, self.classes, self.features)
        else:
            return iter([])

    def __len__(self):
        return self.features.shape[1]

    def __add__(self, other):
        '''
        Combine two feature sets to create a new one.  This is done assuming
        they both have the same instances with the same IDs in the same order.
        '''
        new_set = FeatureSet()
        # Combine feature matrices and vectorizers
        if self.features is not None:
            if type(self.feat_vectorizer) != type(other.feat_vectorizer):
                raise ValueError('Cannot combine FeatureSets because they are '
                                 'not both using the same type of feature '
                                 'vectorizer (e.g., DictVectorizer, '
                                 'FeatureHasher)')
            feature_hasher = isinstance(self.feat_vectorizer, FeatureHasher)
            if not feature_hasher:
                # Check for duplicate feature names
                if (set(self.feat_vectorizer.get_feature_names()) &
                        set(other.feat_vectorizer.get_feature_names())):
                    raise ValueError('Two feature files have the same '
                                     'feature!')
            num_feats = self.features.shape[1]
            new_set.features = sp.hstack([self.features, other.features],
                                         'csr')
            new_set.feat_vectorizer = deepcopy(self.feat_vectorizer)
            if not feature_hasher:
                # dictvectorizer sorts the vocabularies within each file
                vocab = sorted(new_set.feat_vectorizer.vocabulary_.items(),
                               key=lambda x: x[1])
                for feat_name, index in vocab:
                    new_set.feat_vectorizer.vocabulary_[feat_name] = (index +
                                                                      num_feats)
                    new_set.feat_vectorizer.feature_names_.append(feat_name)
        else:
            new_set.features = deepcopy(other.features)
            new_set.feat_vectorizer = deepcopy(other.feat_vectorizer)

        # Check that IDs are in the same order
        if not np.all(self.ids == other.ids):
            raise ValueError('IDs are not in the same order in each feature '
                             'file!')

        # If either set has labels, check that they don't conflict
        if any(x is not None for x in other.classes):
            # Classes should be the same for each ExamplesTuple, so store once
            if self.classes is None:
                new_set.classes = deepcopy(other.classes)
            # Check that classes don't conflict, when specified
            elif not np.all(self.classes == other.classes):
                raise ValueError('Feature files have conflicting labels for '
                                 'examples with the same ID!')
        return new_set

    def filter(self, ids=None, classes=None, features=None, inverse=False):
        '''
        Removes features and/or examples from the Featureset depending on the
        passed in parameters.

        :param ids: Examples to keep in the FeatureSet
        :type ids: list of str/float
        :param classes: Classes that we want to retain examples for.
        :type classes: list of str/float
        :param features: Features to keep in the FeatureSet
        :type features: list of str
        :param inverse: Instead of keeping features and ids in lists, remove
                        them.
        :type inverse: bool
        '''
        pass

    def __sub__(self, other):
        '''
        Return a copy of ``self`` with all features in ``other`` removed.
        '''
        new_set = deepcopy(self)
        new_set.filter(features=other.features, inverse=True)
        return new_set


# Alias ExamplesTuple to FeatureSet for backward-compatibility
ExamplesTuple = FeatureSet
