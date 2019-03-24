"""
Class for transforming features.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:date: 03/19/2019
:organization: ETS
"""

import logging
import numpy as np

from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs import mean_variance_axis, min_max_axis

from skll.config import _VALID_TRANSFORM_OPTIONS


def check_negatives(X,
                    axis=None,
                    raise_error=True,
                    log_warning=True,
                    name=None):
    """
    Check whether a numpy array or sparse matrix has any negative values,
    ignoring NaN values.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
         The data to check.
    axis : None or int, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (axis = None) is to perform a logical OR over all
        the dimensions of the input array.
        Defaults to None.
    raise_error : bool, optional
        Whether to raise an error, if negatives
        exist in the data set.
        Defaults to True.
    log_warning : bool, optional
        Whether to log a warning, if negatives
        exist in the data set. Note the warning
        will only be logged if `raise_error=False`.
        Defaults to True.
    name : str or None, optional
        The name of the transformation that will be ultimately be performed
        on the data set. This is only used if an error is raised, to be
        more informative to users. If None, then the generic 'transformation'
        will be be used.
        Defaults to None.

    Returns
    -------
    negative_values : bool or array of bool
        Whether the data set has negative values.
        If axis is 0 or 1, and array of boolean
        values is returned.
    """
    # if the matrix is sparse, we need to use a different approach
    # to check negatives; we find the minimum across all features,
    # and if it is less than zero, we know there are negatives
    if issparse(X):
        if axis is None:
            negative_values = min_max_axis(X, 0, True)[0].min() < 0
        else:
            negative_values = min_max_axis(X, axis, True)[0] < 0
    else:
        with np.errstate(invalid='ignore'):
            negative_values = np.any(X < 0, axis=axis)

    if np.any(negative_values):
        name = 'transformation' if name is None else name
        if raise_error:
            raise ValueError("Your data set contains negative values, so the {} "
                             "cannot be performed.".format(name))
        if log_warning:
            logging.warning("The {} will be applied to a data set with negative values."
                            "".format(name))

    return negative_values


def check_positives(X,
                    axis=None,
                    raise_error=True,
                    log_warning=True,
                    name=None):
    """
    Check whether a numpy array or sparse matrix has any positive values,
    ignoring NaN values.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
         The data to check.
    axis : None or int, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (axis = None) is to perform a logical OR over all
        the dimensions of the input array.
        Defaults to None.
    raise_error : bool, optional
        Whether to raise an error, if positives
        exist in the data set.
        Defaults to True.
    log_warning : bool, optional
        Whether to log a warning, if positives
        exist in the data set. Note the warning
        will only be logged if `raise_error=False`.
        Defaults to True.
    name : str or None, optional
        The name of the transformation that will be ultimately be performed
        on the data set. This is only used if an error is raised, to be
        more informative to users. If None, then the generic 'transformation'
        will be be used.
        Defaults to None.

    Returns
    -------
    positive_values : bool or array of bool
        Whether the data set has positive values.
        If axis is 0 or 1, and array of boolean
        values is returned.
    """
    # if the matrix is sparse, we need to use a different approach
    # to check negatives; we find the minimum across all features,
    # and if it is less than zero, we know there are negatives
    if issparse(X):
        if axis is None:
            positive_values = min_max_axis(X, 0, True)[1].max() > 0
        else:
            positive_values = min_max_axis(X, axis, True)[1] > 0
    else:
        with np.errstate(invalid='ignore'):
            positive_values = np.any(X > 0, axis=axis)

    if np.any(positive_values):
        name = 'transformation' if name is None else name
        if raise_error:
            raise ValueError("Your data set contains positive values, so the {} "
                             "cannot be performed.".format(name))
        if log_warning:
            logging.warning("The {} will be applied to a data set with positive values."
                            "".format(name))

    return positive_values


def check_zeros(X,
                axis=None,
                raise_error=True,
                log_warning=True,
                name=None):
    """
    Check whether a numpy array has any zero values, ignoring NaN values.
    If `X` is a sparse matrix, it will not be checked.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
         The data to check.
    axis : None or int, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (axis = None) is to perform a logical OR over all
        the dimensions of the input array.
        Defaults to None.
    raise_error : bool, optional
        Whether to raise an error, if zeros
        exist in the data set.
        Defaults to True.
    log_warning : bool, optional
        Whether to log a warning, if zeros
        exist in the data set. Note the warning
        will only be logged if `raise_error=False`.
        Defaults to True.
    name : str or None, optional
        The name of the transformation that will be ultimately be performed
        on the data set. This is only used if an error is raised, to be
        more informative to users. If None, then the generic 'transformation'
        will be be used.
        Defaults to None.

    Returns
    -------
    zero_values : bool or array of bool or None
        Whether the data set has zero values.
        If axis is 0 or 1, and array of boolean
        values is returned.
        If the data is a sparse matrix,
        then `None` is returned.
    """
    # if the matrix is sparse, we shouldn't check whether it has zeros,
    # since we are going to ignore zero elements anyway
    if issparse(X):
        return

    with np.errstate(invalid='ignore'):
        zero_values = np.any(X == 0, axis=axis)

    if np.any(zero_values):
        name = 'transformation' if name is None else name
        if raise_error:
            raise ValueError("Your data set contains zero values, so the {} "
                             "cannot be performed.".format(name))
        if log_warning:
            logging.warning("The {} will be applied to a data set with zero values."
                            "".format(name))

    return zero_values


class Transformer(BaseEstimator, TransformerMixin):
    """
    A class to perform feature transformations.

    Parameters
    ----------
    transformation : str, optional
        The type of transformation to perform. The
        possible transformations are:
          - raw: no transformation, use original feature value
          - log: natural log
          - inv: 1/x
          - sqrt: square root
          - addOneInv: 1/(x+1)
          - addOneLn: ln(x+1)
        Defaults to 'raw'.
    sd_multiplier : int, optional
        Use this std. dev. multiplier to compute the ceiling
        and floor for outlier removal and check that these
        are not equal to zero.
        Only used if transformation is in {'inv'}.
        Defaults to 4.
    raise_error : bool, optional
        Whether to raise errors.
        Defaults to True.
    log_warning : bool, optional
        Whether to log a warning, if negatives
        exist in the data set. Note the warning
        will only be logged if `raise_error=False`.
        Defaults to True.
    copy : bool, optional
        Set to False to perform in-place transformation.
        Defaults to True.
    """

    def __init__(self,
                 transformation='raw',
                 sd_multiplier=4,
                 raise_error=True,
                 log_warning=True,
                 copy=True):

        if transformation not in _VALID_TRANSFORM_OPTIONS:
            raise ValueError('The transformation you specified, `{}`, is not'
                             'in the list of possible transformations. '
                             'Please choose one of the following:\n \{{}\}'
                             ''.format(transformation,
                                       ', '.join(_VALID_TRANSFORM_OPTIONS)))

        self.transformation = transformation
        self.sd_multiplier = sd_multiplier
        self.log_warning = log_warning
        self.raise_error = raise_error
        self.copy = copy

    def fit(self, X, y=None):
        """
        Do nothing and return the estimator unchanged
        This method is just there to implement the usual
        sklearn API and hence work in pipelines.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self
        """
        return self

    def transform(self, X, copy=None):
        """
        Transform each column of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to transform, column-by-column;
            scipy.sparse matrices should be in CSR format.
        copy : bool, optional (default: None)
            Copy the input X or not.
            If None, then the value of
            `self.copy` will be used.
            Defaults to None.

        Returns
        -------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The transformed data.
        """
        copy = copy if copy is not None else self.copy
        X = check_array(X,
                        accept_sparse='csr',
                        force_all_finite='allow-nan',
                        estimator=self,
                        copy=copy)

        # if the transformation is `raw`, we just return
        # the untransformed array that was originally provided
        if self.transformation == 'raw':
            return X

        # if the data set is sparse, then we assign to the
        # the `data` attribute, which is a one-dimensional
        # ndarray which contains all the non-zero elements;
        # so we are ignoring zero elements in our transformation
        if issparse(X):
            X.data = self._transform_feature(X)
        else:
            X = self._transform_feature(X)
        return X

    def _apply_sqrt_transform(self, X):
        """
        Apply the `sqrt` transform to `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
             The data set to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape [n_samples, n_features]
             The transformed data set.

        Raises
        ------
        ValueError
            If the transform is applied to a data set
            that has negative values and `raise_error` is set to True.
        """
        check_negatives(X,
                        raise_error=self.raise_error,
                        log_warning=self.log_warning,
                        name=self.transformation)
        with np.errstate(invalid='ignore'):
            if issparse(X):
                X_new = np.sqrt(X.data)
            else:
                X_new = np.sqrt(X)
        return X_new

    def _apply_log_transform(self, X):
        """
        Apply the `log` transform to `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
             The data set to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape [n_samples, n_features]
             The transformed data set.

        Raises
        ------
        ValueError
            If the transform is applied to a data set
            can be zero or negative and `raise_error` is set to True.
            If the matrix is sparse, then zeros are ignored.
        """
        check_negatives(X,
                        raise_error=self.raise_error,
                        log_warning=self.log_warning,
                        name=self.transformation)
        check_zeros(X,
                    raise_error=self.raise_error,
                    log_warning=self.log_warning,
                    name=self.transformation)
        with np.errstate(invalid='ignore'):
            if issparse(X):
                X_new = np.log(X.data)
            else:
                X_new = np.log(X)
        return X_new

    def _apply_inverse_transform(self, X):
        """
        Apply the inverse transform to `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
             The data set to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape [n_samples, n_features]
             The transformed data set.

        Raises
        ------
        ValueError
            If the transform is applied to a data set that can
            be zero or to a features that can have different
            signs and `raise_error` is set to 'True'
        """
        check_zeros(X,
                    raise_error=self.raise_error,
                    log_warning=self.log_warning,
                    name=self.transformation)
        if issparse(X):
            data_means, data_vars = mean_variance_axis(X, 0)
            # calculate the standard deviations from the variances;
            # since this is the population variance, we perform Bessel's
            # correction to get the approximate sample variance: N / (N - 1.0)
            data_sds = np.sqrt(data_vars * (X.shape[0] / (X.shape[0] - 1.0)))
        else:
            data_means, data_sds = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1)

        floors = data_means - self.sd_multiplier * data_sds
        ceilings = data_means + self.sd_multiplier * data_sds
        if np.any(floors == 0) or np.any(ceilings == 0):
            logging.warning("The floor/ceiling for one of your features "
                            "is zero after applying the inverse "
                            "transformation")

        # check to make sure each feature is either all positives or all negatives;
        # apply this along the feature axis, because different features may have different signs
        all_negatives = ~check_positives(X, axis=0, raise_error=False, log_warning=False)
        all_positives = ~check_negatives(X, axis=0, raise_error=False, log_warning=False)
        all_positives_or_negatives = np.all(np.any(np.array(zip(all_negatives,
                                                                all_positives)), axis=1))
        if not all_positives_or_negatives:
            if self.raise_error:
                raise ValueError("The inverse transformation should not be "
                                 "applied to if the values of any feature can "
                                 "have different signs.")
            if self.log_warning:
                logging.warning("The inverse transformation was applied to "
                                "a features with different signs. This can change "
                                "the ranking of the responses")

        with np.errstate(divide='ignore'):
            if issparse(X):
                X_new = 1 / X.data
            else:
                X_new = 1 / X
        return X_new

    def _apply_add_one_inverse_transform(self, X):
        """
        Apply the add one and invert transform to `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
             The data set to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape [n_samples, n_features]
             The transformed data set.

        Raises
        ------
        ValueError
            If the transform is applied to a data set
            that has negatives and `raise_error` is set to True.
        """
        check_negatives(X,
                        raise_error=self.raise_error,
                        log_warning=self.log_warning,
                        name=self.transformation)
        if issparse(X):
            X_new = 1 / (X.data + 1)
        else:
            X_new = 1 / (X + 1)
        return X_new

    def _apply_add_one_log_transform(self, X):
        """
        Apply the add one and log transform to `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
             The data set to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape [n_samples, n_features]
             The transformed data set.

        Raises
        ------
        ValueError
            If the transform is applied to a data set
            that has negatives and `raise_error` is set to True.
        """
        check_negatives(X,
                        raise_error=self.raise_error,
                        log_warning=self.log_warning,
                        name=self.transformation)
        with np.errstate(invalid='ignore'):
            if issparse(X):
                X_new = np.log1p(X.data)
            else:
                X_new = np.log1p(X)
        return X_new

    def _transform_feature(self, X):
        """
        Applies the given transform to all of the values in the given
        numpy array or sparse matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
             The data set to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape [n_samples, n_features]
             The transformed data set.

        Note
        ----
        Many of these transformations may be meaningless for features which
        span both negative and positive values. Some transformations may
        throw errors for negative feature values.
        """
        transforms = {'inv': self._apply_inverse_transform,
                      'sqrt': self._apply_sqrt_transform,
                      'log': self._apply_log_transform,
                      'addOneInv': self._apply_add_one_inverse_transform,
                      'addOneLn': self._apply_add_one_log_transform}

        transformer = transforms.get(self.transformation)
        X_new = transformer(X)
        return X_new
