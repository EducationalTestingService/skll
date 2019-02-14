# License: BSD 3 clause
"""
A simple wrapper around the existing LogisticRegression class, for testing
custom learners functionality.

:author: Michael Heilman (mheilman@ets.org)
"""

from sklearn.linear_model import LogisticRegression


class CustomLogisticRegressionWrapper(LogisticRegression):
    pass
