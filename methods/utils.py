from ranges.stripe_range import StripeRange
import numpy as np


def linear_search(points: np.ndarray, q: StripeRange):
    in_range_nodes = []
    for i, point in enumerate(points):
        if q.is_in(point):
            in_range_nodes.append(i)
    return in_range_nodes


def epsilon_sample(points: np.ndarray, size: int):
    """
    Sample a subset of points from the given points.
    """
    if size >= len(points):
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return indices, points[indices]


def find_kth_exhaustive(points: np.ndarray, scoring_fn, k: int):
    """
    Find the k-th point in the points based on the scoring function.
    """
    scores = np.array([scoring_fn(point) for point in points])
    sorted_indices = np.argsort(scores)
    return points[sorted_indices[k - 1]], scores[sorted_indices[k - 1]]
