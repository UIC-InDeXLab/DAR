import numpy as np
import math

from ranges.stripe_range import StripeRange


def compute_sample_size(d, epsilon, delta=0.1, constant=1 / 1024):
    vc_dim = d + 1
    m = (constant / (epsilon**2)) * (vc_dim * np.log(1 / epsilon) + np.log(1 / delta))
    return math.ceil(m)


def epsilon_sample(points: np.ndarray, epsilon: float, delta: float = 0.1):
    n, d = points.shape
    sample_size = compute_sample_size(d, epsilon, delta)
    sample_size = min(sample_size, n)  # Can't sample more than population
    indices = np.random.choice(n, size=sample_size, replace=False)
    return points[indices]


def find_stripe(eps_sample, eps, weights, k, n):
    # weights = weights / np.linalg.norm(weights)

    m = len(eps_sample)
    scores = np.dot(eps_sample, weights)

    # print(k / n * m)
    # print("m", m)
    rank_low = int(max(0, math.floor((k / n - eps) * m)))
    rank_high = int(min(m - 1, math.ceil((k / n + eps) * m)))
    # print(f"Rank low: {rank_low}, Rank high: {rank_high} (m={m})")

    sorted_indices = np.argsort(-scores)

    return StripeRange(
        normal_vector=weights,
        start_dot=scores[sorted_indices[rank_high]],
        end_dot=scores[sorted_indices[rank_low]],
    )


def query(index, stripe):
    return index.query(stripe)


def preprocess(points, index_fn, eps=0.1):
    n = len(points)
    # Build index
    index = index_fn(points)

    # Build eps sample
    eps_sample = epsilon_sample(points, epsilon=eps)

    return (index, eps_sample, eps, n)


def find_kth(index, eps_sample, eps, n, weights, k):
    stripe = find_stripe(eps_sample, eps, weights, k, n)
    return query(index, stripe)
