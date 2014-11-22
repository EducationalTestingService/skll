"""
This module is just here until a version of scikit-learn is released with the
changes present here.  It has already been merged in scikit-learn master via
https://github.com/scikit-learn/scikit-learn/pull/3683.
"""

# Authors: Lars Buitinck <L.J.Buitinck@uva.nl>
#          Dan Blanchard <dblanchard@ets.org>
# License: BSD 3 clause

from array import array
from collections import Mapping

import numpy as np
import scipy.sparse as sp
import six

from sklearn.feature_extraction import DictVectorizer as OldDictVectorizer


class DictVectorizer(OldDictVectorizer):
    """Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator: string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse: boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.
    sort: boolean, optional.
        Whether feature_names_ and vocabulary_ should be sorted when fitting.
        True by default.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[ 2.,  0.,  1.],
           [ 0.,  1.,  3.]])
    >>> v.inverse_transform(X) == \
        [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[ 0.,  0.,  4.]])

    See also
    --------
    FeatureHasher : performs vectorization using only a hash function.
    sklearn.preprocessing.OneHotEncoder : handles nominal/categorical features
      encoded as columns of integers.
    """
    def __init__(self, dtype=np.float64, separator="=", sparse=True,
                 sort=True):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = sort
        self.feature_names_ = []
        self.vocabulary_ = {}

    def __eq__(self, other):
        """
        Check whether two vectorizers are the same
        """
        return (self.dtype == other.dtype and
                self.vocabulary_ == other.vocabulary_)

    def fit(self, X, y=None):
        """Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        self
        """
        # collect all the possible feature names
        self.feature_names_ = []
        self.vocabulary_ = {}

        vocab = self.vocabulary_

        for x in X:
            for f, v in six.iteritems(x):
                if isinstance(v, six.string_types):
                    f = "%s%s%s" % (f, self.separator, v)
                if f not in vocab:
                    self.feature_names_.append(f)
                    vocab[f] = len(vocab)

        if self.sort:
            self.feature_names_.sort()
            self.vocabulary_ = dict((f, i) for i, f in
                                    enumerate(self.feature_names_))

        return self

    def fit_transform(self, X, y=None):
        """Learn a list of feature name -> indices mappings and transform X.

        Like fit(X) followed by transform(X), but does not require
        materializing X in memory.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        # Sanity check: Python's array has no way of explicitly requesting the
        # signed 32-bit integers that scipy.sparse needs, so we use the next
        # best thing: typecode "i" (int). However, if that gives larger or
        # smaller integers than 32-bit ones, np.frombuffer screws up.
        assert array("i").itemsize == 4, (
            "sizeof(int) != 4 on your platform; please report this at"
            " https://github.com/scikit-learn/scikit-learn/issues and"
            " include the output from platform.platform() in your bug report")

        self.vocabulary_ = {}
        self.feature_names_ = []

        dtype = self.dtype
        vocab = self.vocabulary_

        # Process everything as sparse regardless of setting
        X = [X] if isinstance(X, Mapping) else X

        indices = array("i")
        indptr = array("i", [0])
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        # collect all the possible feature names and build sparse matrix at
        # same time
        for x in X:
            for f, v in six.iteritems(x):
                if isinstance(v, six.string_types):
                    f = "%s%s%s" % (f, self.separator, v)
                    v = 1
                if f not in vocab:
                    self.feature_names_.append(f)
                    vocab[f] = len(vocab)
                indices.append(vocab[f])
                values.append(dtype(v))

            indptr.append(len(indices))

        if len(indptr) == 1:
            raise ValueError("Sample sequence X is empty.")

        if len(indices) > 0:
            # workaround for bug in older NumPy:
            # http://projects.scipy.org/numpy/ticket/1943
            indices = np.frombuffer(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        shape = (len(indptr) - 1, len(vocab))

        result_matrix = sp.csr_matrix((values, indices, indptr),
                                      shape=shape, dtype=dtype)

        # Sort everything if asked
        if self.sort:
            self.feature_names_.sort()
            map_index = np.empty(len(self.feature_names_), dtype=np.int32)
            for new_val, f in enumerate(self.feature_names_):
                map_index[new_val] = self.vocabulary_[f]
                self.vocabulary_[f] = new_val
            result_matrix = result_matrix[:, map_index]

        # Convert to dense if asked
        if not self.sparse:
            result_matrix = result_matrix.toarray()

        return result_matrix
