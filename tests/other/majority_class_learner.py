# License: BSD 3 clause
"""
A simple majority class classifier, an example of a custom classifier.

:author: Michael Heilman (mheilman@ets.org)
"""

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClassLearner(BaseEstimator, ClassifierMixin):
    """A simple majority class classifier."""

    def __init__(self):
        """Initialize class."""
        self.majority_class = None

    def fit(self, X, y):
        """Set the majority class based on the given data."""
        counts = Counter(y)
        max_count = -1
        for label, count in counts.items():
            if count > max_count:
                self.majority_class = label
                max_count = count
        return self

    def predict(self, X):
        """Return the prediction (majority class) for the given data."""
        return np.array([self.majority_class for x in range(X.shape[0])])
