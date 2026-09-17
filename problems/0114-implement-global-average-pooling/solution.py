import numpy as np


def global_avg_pool(x: np.ndarray) -> np.ndarray:
    pool = []
    for i in range(x.shape[-1]):
        pool.append(np.mean(x[:, :, i]))
    pool = np.array(pool)
    return pool
