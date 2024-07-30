"""Additional custom metrics module for testing purposes."""
from sklearn.metrics import fbeta_score


def f06_micro(y_true, y_pred):
    """Define a custom metric for testing purposes."""
    return fbeta_score(y_true, y_pred, beta=0.6, average="micro")
