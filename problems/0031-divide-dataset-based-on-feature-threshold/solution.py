import numpy as np


def divide_on_feature(X, feature_i, threshold):
    # Your code here
    X = np.array(X)
    mask = X[:, feature_i] >= threshold
    matched = X[mask]
    unmatched = X[~mask]
    return [matched, unmatched]
    pass
