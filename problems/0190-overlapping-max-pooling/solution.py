import numpy as np
import math


def overlapping_max_pool2d(
    x: np.ndarray, kernel_size: int = 3, stride: int = 2
) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    N = x.shape[0]
    C = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    out_h = math.ceil((height - kernel_size) / stride) + 1
    out_w = math.ceil((width - kernel_size) / stride) + 1
    ans = np.empty((N, C, out_h, out_w), dtype=x.dtype)
    for i in range(N):
        for j in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = min(h_start + kernel_size, height)
                    w_end = min(w_start + kernel_size, width)
                    window = x[i, j, h_start:h_end, w_start:w_end]
                    ans[i, j, h, w] = np.max(window)
    return ans
