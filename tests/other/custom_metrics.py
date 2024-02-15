"""Custom metrics for testing purposes."""
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    r2_score,
    roc_auc_score,
)


def f075_macro(y_true, y_pred):  # noqa: D103
    return fbeta_score(y_true, y_pred, beta=0.75, average="macro")


def ratio_of_ones(y_true, y_pred):  # noqa: D103
    true_ones = [label for label in y_true if label == 1]
    pred_ones = [label for label in y_pred if label == 1]
    return len(pred_ones) / (len(true_ones) + len(pred_ones))


def r2(y_true, y_pred):  # noqa: D103
    return r2_score(y_true, y_pred)


def one_minus_precision(y_true, y_pred, greater_is_better=False):  # noqa: D103
    return 1 - precision_score(y_true, y_pred, average="binary")


def one_minus_f1_macro(y_true, y_pred, greater_is_better=False):  # noqa: D103
    return 1 - f1_score(y_true, y_pred, average="macro")


def fake_prob_metric(y_true, y_pred, response_method="predict_proba"):  # noqa: D103
    return average_precision_score(y_true, y_pred)


def fake_prob_metric_multiclass(y_true, y_pred, response_method="predict_proba"):  # noqa: D103
    return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovo")
