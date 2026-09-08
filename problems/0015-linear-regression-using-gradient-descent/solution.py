import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int
) -> np.ndarray:

    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for _ in range(iterations):
        gradient = (1 / m) * X.T @ (X @ theta - y)
        theta = theta - alpha * gradient

    return theta.flatten()