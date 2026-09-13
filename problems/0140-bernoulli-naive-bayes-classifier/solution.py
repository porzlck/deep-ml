import numpy as np
from collections import Counter


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        cnt = Counter(y)
        self.classes = np.unique(y)
        N = len(y)
        log_likelihoods = []
        log_inv_likelihoods = []
        log_priors = []
        for c in self.classes:
            prior = cnt[c] / N
            log_priors.append(np.log(prior))
            X_c = X[y == c]
            N_c = X_c.shape[0]
            feature_count = np.sum(X_c, axis=0)
            likelihood = (feature_count + self.smoothing) / (N_c + 2 * self.smoothing)
            log_likelihoods.append(np.log(likelihood))
            log_inv_likelihoods.append(np.log(1 - likelihood))
        self.log_priors = np.array(log_priors)
        self.log_likelihoods = np.array(log_likelihoods)
        self.log_inv_likelihoods = np.array(log_inv_likelihoods)

    def predict(self, X):
        # Predict class labels for test set X
        scores = []
        for i in range(len(self.classes)):
            score = (
                self.log_priors[i]
                + X @ self.log_likelihoods[i]
                + (1 - X) @ self.log_inv_likelihoods[i]
            )
            scores.append(score)
        scores = np.array(scores).T
        return self.classes[np.argmax(scores, axis=1)]
