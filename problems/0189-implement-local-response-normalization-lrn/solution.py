import numpy as np


def local_response_normalization(
    x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75
) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size
        k: Additive constant
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of same shape as input.
    """
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    ans = np.empty((N, C, H, W), dtype=x.dtype)
    len = n // 2
    for i in range(N):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    left = max(0, c - len)
                    right = min(C, c + len + 1)
                    p = x[i, left:right, h, w]
                    cur_val = x[i, c, h, w]
                    cur_div = (k + alpha * (np.sum(p**2))) ** beta
                    ans[i, c, h, w] = cur_val / cur_div
    return ans
