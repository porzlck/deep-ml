import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    out = w1 @ x
    out = ReLU(out)

    out = w2 @ out

    out = out + x
    out = ReLU(out)

    return out