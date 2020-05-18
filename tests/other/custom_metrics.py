from sklearn.metrics import fbeta_score, precision_score, r2_score


def f075_macro(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 0.75, average='macro')


def ratio_of_ones(y_true, y_pred):
    true_ones = [label for label in y_true if label == 1]
    pred_ones = [label for label in y_pred if label == 1]
    return len(pred_ones) / (len(true_ones) + len(pred_ones))


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def one_minus_precision(y_true, y_pred, greater_is_better=False):
    return 1 - precision_score(y_true, y_pred, average='binary')
