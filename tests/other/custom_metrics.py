from sklearn.metrics import fbeta_score


def f075_macro(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 0.75, average='macro')


def ratio_of_ones(y_true, y_pred):
    true_ones = [label for label in y_true if label == 1]
    pred_ones = [label for label in y_pred if label == 1]
    return len(pred_ones) / (len(true_ones) + len(pred_ones))
