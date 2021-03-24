"""
This module is here because the scikit-learn version of `DictVectorizer`
does not contain an `__eq__` method for vectorizer equality which we need
for SKLL.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from sklearn.feature_extraction import DictVectorizer as OldDictVectorizer


class DictVectorizer(OldDictVectorizer):
    """
    Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    However, note that this transformer will only do a binary one-hot encoding
    when feature values are of type string. If categorical features are
    represented as numeric values such as int, the DictVectorizer can be
    followed by OneHotEncoder to complete binary one-hot encoding.

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Read more in the :ref:`User Guide <dict_feature_extraction>`.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.

    separator : string, optional
        Separator string used when constructing new features for one-hot
        coding.

    sparse : boolean, default=True
        Whether transform should produce scipy.sparse matrices.

    sort : boolean, default=True
        Whether `feature_names_` and `vocabulary_` should be sorted when fitting.

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
    def __eq__(self, other):
        """
        Check whether two vectorizers are the same, assuming
        we are actually comparing to a vectorizer
        """
        return (isinstance(other, OldDictVectorizer) and
                self.dtype == other.dtype and
                self.vocabulary_ == other.vocabulary_)
