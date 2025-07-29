from ranges.stripe_range import StripeRange
import numpy as np
import sys
from collections.abc import Mapping, Container
from numbers import Number


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


def find_kth_exhaustive(points: np.ndarray, weights, k: int):
    scores_with_points = []

    for point in points:
        # Compute dot product manually
        score = sum(p * w for p, w in zip(point, weights))
        scores_with_points.append((score, point))

    # Sort by score in descending order
    scores_with_points.sort(reverse=True, key=lambda x: x[0])

    # Get the k-th (1-based) point
    score, point = scores_with_points[k - 1]
    return point, score


def get_size_recursive(obj, seen=None):
    """
    Recursively calculate the size of a Python object in bytes.

    Args:
        obj: The object to measure
        seen: Set of already seen object IDs to avoid infinite recursion

    Returns:
        int: Size in bytes
    """
    size = sys.getsizeof(obj)

    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Mark this object as seen
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size_recursive(v, seen) for v in obj.values()])
        size += sum([get_size_recursive(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size_recursive(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size_recursive(i, seen) for i in obj])

    return size
