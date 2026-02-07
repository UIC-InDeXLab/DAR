import numpy as np
import random


def calc_score(point: np.ndarray, q: np.ndarray, metric):
    if metric == "l2":
        return np.linalg.norm(point - q)  # k smallest
    elif metric == "cosine":
        denom = float(np.linalg.norm(point) * np.linalg.norm(q))
        score = float(np.dot(point, q)) / denom  # k smallest

        return 1 - score
    else:
        raise ValueError(f"Unknown metric: {metric}")


def quickselect(points: np.ndarray, k, q: np.ndarray, metric):
    if k < 0 or k >= len(points):
        raise ValueError(f"k must be in [0, {len(points) - 1}], got {k}")
    items = [(calc_score(point, q, metric), i) for i, point in enumerate(points)]
    idx = _qs(items, k)
    return idx  # return index


def _qs(items, k):
    pivot_score, _ = random.choice(items)

    lows = [x for x in items if x[0] < pivot_score]
    pivots = [x for x in items if x[0] == pivot_score]
    highs = [x for x in items if x[0] > pivot_score]

    if k < len(lows):
        return _qs(lows, k)
    if k < len(lows) + len(pivots):
        return min(i for _, i in pivots) # tie break
    return _qs(highs, k - len(lows) - len(pivots))
