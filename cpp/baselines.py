from ranges.stripe_range import StripeRange
import numpy as np


def linear_search(points: np.ndarray, q: StripeRange):
    in_range_nodes = []
    for i, point in enumerate(points):
        if q.is_in(point):
            in_range_nodes.append(i)
    return in_range_nodes
