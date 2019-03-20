

import numpy as np
import pandas as pd

from copy import deepcopy
from functools import wraps
from nose.plugins.logcapture import LogCapture

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from nose.tools import eq_, raises

from skll.transformer import check_negatives, check_positives, check_zeros
from skll.transformer import FeatureTransformer
from skll import FeatureSet


def checklogging(log_msg_txt=None):

    def _check_logging(fun):
        @wraps(fun)
        def _wrapped(*args, **kwargs):
            lc = LogCapture()
            lc.begin()
            res = fun(*args, **kwargs)
            # make sure there are logging messages
            assert len(lc.handler.buffer) > 0, 'No logging message found.'
            # check the last logging message
            if log_msg_txt is not None:
                eq_(lc.handler.buffer[-1], log_msg_txt)
            return res
        return _wrapped
    return _check_logging


@checklogging('root: WARNING: The ~~TESTING~~ will be '
              'applied to a data set with zero values.')
def test_check_zero_log_warning():

    data = np.random.randn(1000, 10)
    data[45, 4] = 0
    check_zeros(data, raise_error=False, name='~~TESTING~~')


@checklogging('root: WARNING: The ~~TESTING~~ will be '
              'applied to a data set with positive values.')
def test_check_positives_log_warning():

    data = np.random.randn(1000, 10)
    data = np.abs(data)
    check_positives(data, raise_error=False, name='~~TESTING~~')


@checklogging('root: WARNING: The ~~TESTING~~ will be '
              'applied to a data set with negative values.')
def test_check_negatives_log_warning():

    data = np.random.randn(1000, 10)
    data = -data
    check_negatives(data, raise_error=False, name='~~TESTING~~')


class TestCheckZero:

    def setUp(self):

        self.data = np.random.randn(1000, 10)

    def test_check_zero_with_numpy(self):

        data = self.data.copy()
        eq_(check_zeros(data), False)
        assert_array_equal(check_zeros(data, 0), np.array([False] * 10))
        assert_array_equal(check_zeros(data, 1), np.array([False] * 1000))

    def test_check_zero_with_pandas(self):

        data = self.data.copy()
        data = pd.DataFrame(data)
        eq_(check_zeros(data), False)
        assert_array_equal(check_zeros(data, 0), np.array([False] * 10))
        assert_array_equal(check_zeros(data, 1), np.array([False] * 1000))

    def test_check_zero_with_sparse(self):

        data = self.data.copy()
        data = csr_matrix(data)
        eq_(check_zeros(data), None)
        eq_(check_zeros(data, 0), None)
        eq_(check_zeros(data, 1), None)

    @raises(ValueError)
    def test_check_zero_with_numpy_raise_error(self):

        data = self.data.copy()
        data[45, 4] = 0
        check_zeros(data)

    @raises(ValueError)
    def test_check_zero_with_pandas_raise_error(self):

        data = self.data.copy()
        data = pd.DataFrame(data)
        data.iloc[45, 4] = 0
        check_zeros(data)

    def test_check_zero_with_sparse_does_not_raise_error(self):

        data = self.data.copy()
        data[45, 4] = 0
        data = csr_matrix(data)
        eq_(check_zeros(data), None)


class TestCheckNegatives:

    def setUp(self):

        data = np.random.randn(1000, 10)
        data = np.abs(data)
        data[[0, 1], [0, 1]] = -1
        self.data = data

    def test_check_negative_with_numpy(self):

        data = self.data.copy()
        eq_(check_negatives(data, raise_error=False, log_warning=False), True)
        assert_array_equal(check_negatives(data, 0, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 8))
        assert_array_equal(check_negatives(data, 1, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 998))

    def test_check_negative_with_pandas(self):

        data = self.data.copy()
        data = pd.DataFrame(data)
        eq_(check_negatives(data, raise_error=False, log_warning=False), True)
        assert_array_equal(check_negatives(data, 0, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 8))
        assert_array_equal(check_negatives(data, 1, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 998))

    def test_check_negative_with_sparse(self):

        data = self.data.copy()
        data = csr_matrix(data)
        # add some zeros
        data[[100, 100], [8, 7]] = 0
        eq_(check_negatives(data, raise_error=False, log_warning=False), True)
        assert_array_equal(check_negatives(data, 0, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 8))
        assert_array_equal(check_negatives(data, 1, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 998))

    @raises(ValueError)
    def test_check_negatives_with_numpy_raise_error(self):

        data = self.data.copy()
        check_negatives(data)

    @raises(ValueError)
    def test_check_negatives_with_sparse_raise_error(self):

        data = self.data.copy()
        data = csr_matrix(data)
        # add some zeros
        data[[100, 100], [8, 7]] = 0
        check_negatives(data)


class TestCheckPositives:

    def setUp(self):

        data = np.random.randn(1000, 10)
        data = -np.abs(data)
        data[[0, 1], [0, 1]] = 1
        self.data = data

    def test_check_positive_with_numpy(self):

        data = self.data.copy()
        eq_(check_positives(data, raise_error=False, log_warning=False), True)
        assert_array_equal(check_positives(data, 0, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 8))
        assert_array_equal(check_positives(data, 1, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 998))

    def test_check_positive_with_pandas(self):

        data = self.data.copy()
        data = pd.DataFrame(data)
        eq_(check_positives(data, raise_error=False, log_warning=False), True)
        assert_array_equal(check_positives(data, 0, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 8))
        assert_array_equal(check_positives(data, 1, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 998))

    def test_check_positive_with_sparse(self):

        data = self.data.copy()
        data = csr_matrix(data)
        # add some zeros
        data[[100, 100], [8, 7]] = 0
        eq_(check_positives(data, raise_error=False, log_warning=False), True)
        assert_array_equal(check_positives(data, 0, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 8))
        assert_array_equal(check_positives(data, 1, raise_error=False, log_warning=False),
                           np.array([True] * 2 + [False] * 998))

    @raises(ValueError)
    def test_check_positive_with_numpy_raise_error(self):

        data = self.data.copy()
        check_positives(data)

    @raises(ValueError)
    def test_check_positive_with_sparse_raise_error(self):

        data = self.data.copy()
        data = csr_matrix(data)
        # add some zeros
        data[[100, 100], [8, 7]] = 0
        check_positives(data)


class TestFeatureTransformer:

    def setUp(self):

        self.data = np.random.randn(1000, 10)
        self.data_all_pos = np.abs(self.data.copy())
        self.data_all_neg = -np.abs(self.data.copy())

        data = pd.DataFrame(self.data, columns=['y'] + ['f{}'.format(i) for i in range(1, 10)])
        data.index.name = 'id'
        data = data.abs()
        self.data_pd = data.copy()
        self.data_fs = FeatureSet.from_data_frame(data, 'test', labels_column='y')

    @staticmethod
    def add_random_values(data, values, rng=100):
        data = deepcopy(data)
        for _ in range(rng):
            row_idx = np.random.randint(0, 1000)
            col_idx = np.random.randint(0, 10)
            data[row_idx, col_idx] = np.nan
        return data

    def test_raw_transformation_numpy(self):

        ft = FeatureTransformer()
        data = ft.transform(self.data)
        assert_array_equal(data, self.data)

    def test_log_transformation_numpy(self):

        ft = FeatureTransformer('log')
        data = ft.transform(self.data_all_pos)
        assert_array_equal(data, np.log(self.data_all_pos))

    def test_inv_transformation_numpy_pos(self):

        ft = FeatureTransformer('inv')
        data = ft.transform(self.data_all_pos)
        assert_array_equal(data, 1 / self.data_all_pos)

    def test_inv_transformation_numpy_neg(self):

        ft = FeatureTransformer('inv')
        data = ft.transform(self.data_all_neg)
        assert_array_equal(data, 1 / self.data_all_neg)

    @raises(ValueError)
    def test_inv_transformation_numpy_neg_error(self):

        data_all_neg = self.data_all_neg.copy()
        data_all_neg[0, 0] = 1
        ft = FeatureTransformer('inv')
        ft.transform(data_all_neg)

    @raises(ValueError)
    def test_inv_transformation_numpy_pos_error(self):

        data_all_pos = self.data_all_pos.copy()
        data_all_pos[0, 0] = -1
        ft = FeatureTransformer('inv')
        ft.transform(data_all_pos)

    def test_inv_transformation_numpy_one_col_different(self):

        data_all_pos = self.data_all_pos.copy()
        data_all_pos[:, 0] = -np.abs(np.random.randn(1000,))
        ft = FeatureTransformer('inv')
        data = ft.transform(data_all_pos)
        assert_array_equal(data, 1 / data_all_pos)
