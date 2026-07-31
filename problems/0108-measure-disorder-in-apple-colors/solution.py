import numpy as np


def disorder(apples: list) -> float:
    """
    Compute the disorder in a basket of apples.
    """
    # Your code here
    apples = np.array(apples)
    _, counts = np.unique(apples, return_counts=True)
    probs = counts / len(apples)
    return 1 - np.sum(probs**2)
    pass
