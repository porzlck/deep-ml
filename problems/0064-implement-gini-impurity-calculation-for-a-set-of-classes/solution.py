from collections import Counter
import numpy as np


def gini_impurity(y):
    cnt = Counter(y)
    n = len(y)
    val = 1 - sum((count / n) ** 2 for count in cnt.values())
    return round(val, 3)
