"""
Class for transforming features.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:date: 03/14/2019
:organization: ETS
"""

import logging
import numpy as np

from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin


POSSIBLE_TRANSFORMATIONS = {'inv', 'sqrt', 'log', 'addOneInv', 'addOneLn', 'raw'}


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Encapsulate feature transformation methods.

    Parameters
    ----------
    transformation : str, optional
        The type of transformation to perform. The
        possible transformations are:
          - raw: no transformation, use original feature value
          - inv: 1/x
          - sqrt: square root
          - addOneInv: 1/(x+1)
          - addOneLn: ln(x+1)
        Defaults to 'raw'.
    raise_error : bool, optional
        When set to true, raises an error if the transform is applied to
        a feature that can have negative values.
        Defaults to True.
    """

    def __init__(self,
                 transformation='raw',
                 raise_error=True):

        transformation = str(transformation)
        if transformation not in POSSIBLE_TRANSFORMATIONS:
            possible_transformations_str = ', '.join(POSSIBLE_TRANSFORMATIONS)
            raise ValueError('The transformation you specified, `{}`, is not'
                             'in the list of possible transformations. '
                             'Please choose one of the following: {}'
                             ''.format(transformation, possible_transformations_str))
        self.transformation = transformation
        self.raise_error = raise_error

    def fit(self, X, y=None):
        """
        No actual fitting is performed.

        Returns
        -------
        self
        """
        return self

    def fit_transform(self, X, y=None):
        """
        Transform the features using the appropriate
        transformation, and return the transformed
        features object.

        Returns
        -------
        X : numpy array or scipy sparse array
            The transformed features
        """
        return self.transform(X)

    def transform(self, X):
        """
        Transform the features using the appropriate
        transformation, and return the transformed
        features object.

        Returns
        -------
        X : numpy array or scipy sparse array
            The transformed features
        """
        X = deepcopy(X)
        if self.transformation == 'raw':
            return X

        for col_index, _ in enumerate(X.T):
            X.T[col_index] = self.transform_feature(X.T[col_index],
                                                    self.transformation,
                                                    self.raise_error)
        return X

    @classmethod
    def apply_sqrt_transform(cls,
                             values,
                             raise_error=True):
        """
        Apply the `sqrt` transform to `values`.

        Parameters
        ----------
        values : numpy array or scipy sparse array
            Numpy array or Scipy array containing the feature values.
        raise_error : bool, optional
            When set to true, raises an error if the transform is applied to
            a feature that can have negative values.

        Returns
        -------
        new_data : numpy array
            Numpy array or Scipy array containing the transformed feature
            values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature
            that has negative values and `raise_error` is set to true.
        """
        # check if the feature has any negative values
        if values.min() < 0:
            if raise_error:
                raise ValueError("The square root transformation should not be "
                                 "applied, since at least one of your features "
                                 "has negative values")
            else:
                logging.warning("The square root  transformation was not applied to "
                                "a feature in your data, which had negative values.")
                return values

        with np.errstate(invalid='ignore'):
            new_data = np.sqrt(values)
        return new_data

    @classmethod
    def apply_log_transform(cls,
                            values,
                            raise_error=True):
        """
        Apply the `log` transform to `values`.

        Parameters
        ----------
        values : numpy array or scipy sparse array
            Numpy array or Scipy array containing the feature values.
        raise_error : bool, optional
            When set to true, raises an error if the transform is applied to
            a feature that has zero or negative values.

        Returns
        -------
        new_data : numpy array or scipy sparse array
            Numpy array or Scipy array containing the transformed feature
            values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature that
            can be zero or negative and `raise_error` is set to true.
        """
        # check if the feature has any negative values
        if values.min() < 0:
            if raise_error:
                raise ValueError("The log transformation should not be "
                                 "applied, since at least one of your features "
                                 "has negative values")
            else:
                logging.warning("The log transformation was not applied to "
                                "a feature in your data, which had negative values.")
                return values

        # check if the feature has any zeros
        if not ((values != 0).sum() == values.shape[0]):
            if raise_error:
                raise ValueError("The log transformation should not be "
                                 "applied, since at least one of your features "
                                 "has 0 values")
            else:
                logging.warning("The log transformation was not applied to "
                                "a feature in your data, which had 0 values.")
                return values

        new_data = np.log(values)
        return new_data

    @classmethod
    def apply_inverse_transform(cls,
                                values,
                                raise_error=True,
                                sd_multiplier=4):
        """
        Apply the inverse transform to `values`.

        Parameters
        ----------
        values : numpy array or scipy.sparse.csr_matrix
            Numpy or Scipy array containing the feature values.
        raise_error : bool, optional
            When set to true, raises an error if the transform is applied to
            a feature that can be zero or to a feature that can have
            different signs.
        sd_multiplier : int, optional
            Use this std. dev. multiplier to compute the ceiling
            and floor for outlier removal and check that these
            are not equal to zero.

        Returns
        -------
        new_data: numpy array
            Numpy array containing the transformed feature
            values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature that can
            be zero or to a feature that can have different
            signs and `raise_error` is set to 'True'
        """
        # check if the feature has any zeros
        if not ((values != 0).sum() == values.shape[0]):
            if raise_error:
                raise ValueError("The inverse transformation should not be "
                                 "applied, since at least one of your features "
                                 "has 0 values")
            else:
                logging.warning("The inverse transformation was not applied to "
                                "a feature in your data, which had 0 values.")
                return values

        # check if the floor or ceiling are zero
        data_mean = np.mean(values)
        data_sd = np.std(values, ddof=1)
        floor = data_mean - sd_multiplier * data_sd
        ceiling = data_mean + sd_multiplier * data_sd
        if floor == 0 or ceiling == 0:
            logging.warning("The floor/ceiling for one of your "
                            "is zero after applying the inverse "
                            "transformation")

        # check if the feature can be both positive and negative
        all_positive = np.all(np.abs(values) == values)
        all_negative = np.all(np.abs(values) == -values)
        if not (all_positive or all_negative):
            if raise_error:
                raise ValueError("The inverse transformation should not be "
                                 "applied to feature where the values can "
                                 "have different signs")
            else:
                logging.warning("The inverse transformation was "
                                "applied to feature where the values can"
                                "have different signs. This can change "
                                "the ranking of the responses")

        with np.errstate(divide='ignore'):
            new_data = 1 / values

        return new_data

    @classmethod
    def apply_add_one_inverse_transform(cls,
                                        values,
                                        raise_error=True):
        """
        Apply the add one and invert transform to `values`.

        Parameters
        ----------
        values : numpy array or scipy.sparse.csr_matrix
            Numpy or Scipy array containing the feature values.
            Numpy array containing the feature values.
        raise_error : bool, optional
            When set to true, raises an error if the transform is applied to
            a feature that has zero or negative values.

        Returns
        -------
        new_data : np.array
            Numpy array containing the transformed feature
            values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature
            that can be negative and `raise_error` is set to True.
        """
        # check if the feature has any negative values
        if values.min() < 0:
            if raise_error:
                raise ValueError("The addOneInv transformation should not be "
                                 "applied, since at least one of your features "
                                 "has negative values")
            else:
                logging.warning("The addOneInv transformation was not applied to "
                                "a feature in your data, which had negative values.")
                return values

        new_data = 1 / (values + 1)
        return new_data

    @classmethod
    def apply_add_one_log_transform(cls,
                                    values,
                                    raise_error=True):
        """
        Apply the add one and log transform to `values`.

        Parameters
        ----------
        values : numpy array or scipy.sparse.csr_matrix
            Numpy or Scipy array containing the feature values.
        raise_error : bool, optional
            When set to true, raises an error if the transform is applied to
            a feature that has zero or negative values.

        Returns
        -------
        new_data : numpy array
            Numpy array that contains the transformed feature
            values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature that
            can be negative.
        """
        # check if the feature has any negative values
        if values.min() < 0:
            if raise_error:
                raise ValueError("The addOneLn transformation should not be "
                                 "applied, since at least one of your features "
                                 "has negative values")
            else:
                logging.warning("The addOneLn transformation was not applied to "
                                "a feature in your data, which had negative values.")
                return values

        new_data = np.log(values + 1)
        return new_data

    @classmethod
    def transform_feature(cls,
                          values,
                          transform,
                          raise_error=True):
        """
        Applies the given transform to all of the values in the given
        numpy array. The values are assumed to be for the feature with
        the given name.

        Parameters
        ----------
        values : numpy array or scipy.sparse.csr_matrix
            Numpy or Scipy array containing the feature values.
        transform : str
            Name of the transform to apply.
            Valid options include ::
                {'inv', 'sqrt', 'log', 'addOneInv', 'addOneLn'}
        raise_error : bool, optional
            Raise a ValueError if a transformation leads to `Inf` values or may
            change the ranking of the responses

        Returns
        -------
        new_data : np.array
            Numpy or Scipy array containing the transformed feature
            values.

        Note
        ----
        Many of these transformations may be meaningless for features which
        span both negative and positive values. Some transformations may
        throw errors for negative feature values.
        """

        transforms = {'inv': FeatureTransformer.apply_inverse_transform,
                      'sqrt': FeatureTransformer.apply_sqrt_transform,
                      'log': FeatureTransformer.apply_log_transform,
                      'addOneInv': FeatureTransformer.apply_add_one_inverse_transform,
                      'addOneLn': FeatureTransformer.apply_add_one_log_transform}

        transformer = transforms.get(transform)
        new_data = transformer(values, raise_error)
        return new_data
