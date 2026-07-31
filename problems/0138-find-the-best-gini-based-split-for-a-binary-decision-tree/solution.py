import numpy as np
from typing import Tuple


def gini(y) -> float:
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)


def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    best_score = float("inf")
    best_feature = -1
    best_threshold = 0.0
    for feature in range(X.shape[1]):
        for threshold in np.unique(X[:, feature]):
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            left_labels = y[left_mask]
            right_labels = y[right_mask]
            left_weight = len(left_labels) / len(y)
            right_weight = len(right_labels) / len(y)
            score = left_weight * gini(left_labels) + right_weight * gini(right_labels)
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = float(threshold)
    return best_feature, best_threshold
    pass
