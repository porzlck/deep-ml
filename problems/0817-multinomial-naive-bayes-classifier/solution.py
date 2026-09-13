import numpy as np
from collections import Counter


def multinomial_naive_bayes(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """
    Implements Multinomial Naive Bayes classifier.

    Args:
        X_train: Training count features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test count features (shape: N_test x D)
        alpha: Laplace smoothing parameter

    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    cnt = Counter(y_train)
    classes = np.unique(y_train)
    log_priors = []
    log_likelihoods = []
    N = len(y_train)
    D = X_train.shape[1]
    for c in classes:
        X_c = X_train[c == y_train]
        prior = cnt[c] / N
        log_priors.append(np.log(prior))
        likelihood = np.sum(X_c, axis=0)
        likelihood = (likelihood + alpha) / (np.sum(likelihood) + alpha * D)
        log_likelihoods.append(np.log(likelihood))
    log_likelihoods = np.array(log_likelihoods)
    log_priors = np.array(log_priors)
    scores = X_test @ log_likelihoods.T + log_priors
    predictions = classes[np.argmax(scores, axis=1)]
    return predictions
