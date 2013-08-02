# Copyright (C) 2012-2013 Educational Testing Service

# This file is part of SciKit-Learn Laboratory.

# SciKit-Learn Laboratory is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SciKit-Learn Laboratory is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SciKit-Learn Laboratory.  If not, see <http://www.gnu.org/licenses/>.

'''
StandardScaler has a bug in that it always scales by the standard
deviation for sparse matrices, i.e., it ignores the value of with_std.
This is a fixed version. This is just temporary until the bug is fixed in
sklearn.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
'''

from __future__ import print_function, unicode_literals

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import _mean_and_std, StandardScaler
from sklearn.utils import warn_if_not_float, check_arrays
from sklearn.utils.sparsefuncs import inplace_csr_column_scale
from sklearn.utils.sparsefuncs import mean_variance_axis0


class FixedStandardScaler(StandardScaler):

    '''
    StandardScaler has a bug in that it always scales by the standard
    deviation for sparse matrices, i.e., it ignores the value of with_std.
    This is a fixed version. This is just temporary until the bug is fixed in
    sklearn.
    '''

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        X = check_arrays(X, copy=self.copy, sparse_format="csr")[0]
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError("Cannot center sparse matrices: pass " +
                                 "`with_mean=False` instead. See docstring " +
                                 "for motivation and alternatives.")
            warn_if_not_float(X, estimator=self)
            self.mean_ = None

            # we added this check for with_std
            if self.with_std:
                var = mean_variance_axis0(X)[1]
                self.std_ = np.sqrt(var)
                self.std_[var == 0.0] = 1.0
            else:
                self.std_ = None

            return self
        else:
            warn_if_not_float(X, estimator=self)
            self.mean_, self.std_ = _mean_and_std(X, axis=0,
                                                  with_mean=self.with_mean,
                                                  with_std=self.with_std)
            return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        X = check_arrays(X, copy=copy, sparse_format="csr")[0]
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError("Cannot center sparse matrices: pass " +
                                 "`with_mean=False` instead. See docstring " +
                                 "for motivation and alternatives.")
            warn_if_not_float(X, estimator=self)
            if self.with_std:
                inplace_csr_column_scale(X, 1 / self.std_)
        else:
            warn_if_not_float(X, estimator=self)
            if self.with_mean:
                X -= self.mean_
            # we added this check for with_std
            if self.with_std:
                X /= self.std_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError("Cannot center sparse matrices: pass " +
                                 "`with_mean=False` instead. See docstring " +
                                 "for motivation and alternatives.")
            if not sp.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            # we added this check for with_std
            if self.with_std:
                inplace_csr_column_scale(X, self.std_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.std_
            if self.with_mean:
                X += self.mean_
        return X
