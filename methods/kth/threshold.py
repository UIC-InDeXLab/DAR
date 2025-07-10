import numpy as np
from collections import defaultdict
import heapq


def threshold_algorithm(points: np.ndarray, weights: np.ndarray, k: int):
    """
    Threshold Algorithm (TA) to find top-k points based on dot product scoring.

    Args:
        points (np.ndarray): An (n, d) array of points.
        weights (np.ndarray): A (d,) weight vector.
        k (int): Number of top results to return.

    Returns:
        List of tuples: (index, score, point)
    """
    n, d = points.shape
    assert weights.shape[0] == d, "Dimension mismatch."

    # Step 1: Create sorted lists for each dimension
    sorted_lists = []
    for dim in range(d):
        scores = points[:, dim] * weights[dim]
        sorted_indices = np.argsort(-scores)  # descending
        sorted_lists.append(sorted_indices)

    # Data structures
    seen = {}
    heap = []  # Min-heap to maintain top-k: (score, index)
    threshold = 0
    ptr = 0  # global position in lists

    while True:
        # Step 2: Sorted Access: look at the next unseen item in each list
        for dim in range(d):
            if ptr >= n:
                continue
            idx = sorted_lists[dim][ptr]
            if idx not in seen:
                # Random Access: compute full score
                score = np.dot(points[idx], weights)
                seen[idx] = score
                if len(heap) < k:
                    heapq.heappush(heap, (score, idx))
                else:
                    if score > heap[0][0]:
                        heapq.heappushpop(heap, (score, idx))

        # Step 3: Threshold calculation
        threshold = sum(
            points[sorted_lists[dim][ptr], dim] * weights[dim] for dim in range(d)
        )

        # Step 4: Check stopping condition
        if len(heap) == k and heap[0][0] >= threshold:
            break

        ptr += 1
        if ptr >= n:
            break

    # Step 5: Prepare results
    topk = sorted(heap, key=lambda x: -x[0])  # descending order
    # return points[topk[-1][0]], topk[-1][1]
    return points[topk[-1][1]], topk[-1][0]


# Example Usage:
if __name__ == "__main__":
    np.random.seed(0)
    n_points = 100
    dim = 4
    k = 5

    points = np.random.rand(n_points, dim)
    weights = np.random.rand(dim)

    top_k = threshold_algorithm(points, weights, k)
    for idx, score, point in top_k:
        print(f"Index: {idx}, Score: {score:.4f}, Point: {point}")
