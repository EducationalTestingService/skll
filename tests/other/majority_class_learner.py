# License: BSD 3 clause
"""
A simple majority class classifier, an example of a custom classifier.

:author: Michael Heilman (mheilman@ets.org)
"""

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClassLearner(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.majority_class = None

    def fit(self, X, y):
        counts = Counter(y)
        max_count = -1
        for label, count in counts.items():
            if count > max_count:
                self.majority_class = label
                max_count = count

    def predict(self, X):
        return np.array([self.majority_class for x in range(X.shape[0])])

