import numpy as np
from collections import defaultdict


def fagins_algorithm(points: np.ndarray, weights: np.ndarray, k: int):
    """
    Fagin's Algorithm to find top-k points based on dot product scoring.

    Args:
        points (np.ndarray): An (n, d) array of n points in d-dimensional space.
        weights (np.ndarray): A (d,) array representing the scoring weight vector.
        k (int): The number of top results to return.

    Returns:
        List of tuples: Each tuple is (index, score, point).
    """
    n, d = points.shape
    assert weights.shape[0] == d, "Weight vector dimension mismatch."

    # Step 1: Create sorted lists for each dimension
    sorted_lists = []
    for dim in range(d):
        sorted_indices = np.argsort(points[:, dim] * weights[dim])[
            ::-1
        ]  # descending order
        sorted_lists.append(sorted_indices)

    # Step 2: Sorted Access Phase
    seen = defaultdict(set)  # key: point index, value: set of dimensions seen
    fully_seen = set()
    ptrs = [0] * d
    while len(fully_seen) < k:
        for dim in range(d):
            if ptrs[dim] >= n:
                continue
            idx = sorted_lists[dim][ptrs[dim]]
            seen[idx].add(dim)
            if len(seen[idx]) == d:
                fully_seen.add(idx)
            ptrs[dim] += 1
        if all(ptr >= n for ptr in ptrs):
            break  # All lists exhausted

    # Step 3: Random Access Phase - compute scores
    candidates = set(seen.keys())
    scores = {}
    for idx in candidates:
        score = np.dot(points[idx], weights)
        scores[idx] = score

    # Step 4: Select Top-k
    topk = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    # result = [(idx, score, points[idx]) for idx, score in topk]
    return points[topk[-1][0]], topk[-1][1]


# Example Usage:
if __name__ == "__main__":
    np.random.seed(42)
    n_points = 100
    dim = 4
    k = 5

    points = np.random.rand(n_points, dim)
    weights = np.random.rand(dim)

    top_k_results = fagins_algorithm(points, weights, k)

    for idx, score, point in top_k_results:
        print(f"Index: {idx}, Score: {score:.4f}, Point: {point}")
