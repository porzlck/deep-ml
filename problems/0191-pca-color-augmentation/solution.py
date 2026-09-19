import numpy as np


def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image.

    Args:
        image: RGB image of shape (H, W, 3) with values in [0, 255]
        alpha: Array of 3 random coefficients for principal components

    Returns:
        Augmented image of shape (H, W, 3) with values clamped to [0, 255]
    """
    pixels = image.reshape(image.shape[0] * image.shape[1], 3)
    cov = np.cov(pixels, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.maximum(eigenvalues, 0)
    delta = eigenvectors @ (alpha * np.sqrt(eigenvalues))
    image = image.astype(float) + delta
    image = np.clip(image, 0, 255)
    return image
