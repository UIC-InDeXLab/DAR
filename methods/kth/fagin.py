import numpy as np
import heapq
from collections import defaultdict


def fagins_algorithm_points(points, k, scoring_fn):
    """
    points: List of n points, each is a list or np.ndarray of d real values.
    k: Number of top items to retrieve.
    scoring_fn: Monotonic scoring function that takes a d-dimensional list and returns a score.

    Returns: List of (index, aggregate_score) for top-k points (using their original indices).
    """
    n, d = points.shape

    # Step 1: Create sorted lists for each dimension
    lists = []
    for dim in range(d):
        sorted_indices = np.argsort(-points[:, dim])  # descending order
        sorted_list = [(idx, points[idx, dim]) for idx in sorted_indices]
        lists.append(sorted_list)

    seen = defaultdict(set)  # index -> set of list indices where seen
    scores = defaultdict(lambda: [None] * d)  # index -> list of scores
    objects_seen_in_all = set()

    max_length = n

    # Step 2: Sorted Access Phase
    for i in range(max_length):
        for list_idx, lst in enumerate(lists):
            if i < len(lst):
                obj_idx, score = lst[i]
                seen[obj_idx].add(list_idx)
                scores[obj_idx][list_idx] = score

                if len(seen[obj_idx]) == d:
                    objects_seen_in_all.add(obj_idx)

        if len(objects_seen_in_all) >= k:
            break

    # Step 3: Random Access Phase
    for obj_idx in scores:
        for list_idx in range(d):
            if scores[obj_idx][list_idx] is None:
                scores[obj_idx][list_idx] = points[obj_idx, list_idx]

    # Step 4: Aggregate using scoring function
    aggregate_scores = []
    for obj_idx, score_vector in scores.items():
        total_score = scoring_fn(score_vector)
        aggregate_scores.append((total_score, obj_idx))

    top_k = heapq.nlargest(k, aggregate_scores)

    return [(idx, score) for score, idx in top_k]
