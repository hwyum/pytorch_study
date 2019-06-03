import numpy as np

from sklearn.metrics import f1_score


def f1(pred, target, labels=None):
    if isinstance(pred[0], np.ndarray):
        pred = pred.flatten()
    if isinstance(target[0], np.ndarray):
        target = target.flatten()

    if len(pred) != len(target):
        raise ValueError()

    return f1_score(target, pred, labels=labels, average='weighted')