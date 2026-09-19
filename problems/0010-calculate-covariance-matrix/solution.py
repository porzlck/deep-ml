import numpy as np


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    v = np.array(vectors)
    mat = np.cov(v, rowvar=True)
    return mat.tolist()
