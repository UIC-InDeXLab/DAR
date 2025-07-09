import numpy as np
import heapq
from collections import defaultdict


def threshold_algorithm(points, k, scoring_fn):
    """
    points: np.ndarray of shape (n, d)
    k: int, number of top results
    scoring_fn: monotonic scoring function on complete d-dimensional vectors

    Returns: List of (point_index, aggregate_score) for top-k points
    """
    n, d = points.shape

    # Step 1: Precompute sorted lists (descending) and build lookup tables
    sorted_lists = []
    lookup_tables = []
    for dim in range(d):
        sorted_idx = np.argsort(-points[:, dim])  # descending sort
        sorted_list = [(idx, points[idx, dim]) for idx in sorted_idx]
        lookup = {idx: points[idx, dim] for idx in sorted_idx}
        sorted_lists.append(sorted_list)
        lookup_tables.append(lookup)

    # Initialize data structures
    seen_scores = defaultdict(lambda: [None] * d)
    complete_objects = set()
    accessed = set()
    top_k_heap = []  # min-heap of (score, point_idx)
    max_iters = max(len(lst) for lst in sorted_lists)

    for i in range(max_iters):
        # Step 2: Sorted access round
        current_frontier = []
        for dim_idx, lst in enumerate(sorted_lists):
            if i < len(lst):
                obj_idx, value = lst[i]
                seen_scores[obj_idx][dim_idx] = value
                current_frontier.append(value)
                accessed.add(obj_idx)

        # Step 3: Random access for missing dimensions
        for obj_idx in accessed:
            vector = seen_scores[obj_idx]
            for dim_idx in range(d):
                if vector[dim_idx] is None:
                    vector[dim_idx] = lookup_tables[dim_idx][obj_idx]

            # Score the fully observed vector
            score = scoring_fn(vector)
            seen_scores[obj_idx] = vector  # make sure it's saved
            if obj_idx not in complete_objects:
                complete_objects.add(obj_idx)
                heapq.heappush(top_k_heap, (score, obj_idx))
                if len(top_k_heap) > k:
                    heapq.heappop(top_k_heap)

        # Step 4: Compute threshold
        threshold_vector = []
        for dim_idx, lst in enumerate(sorted_lists):
            if i < len(lst):
                threshold_vector.append(lst[i][1])
            else:
                threshold_vector.append(0.0)  # conservative default

        threshold_score = scoring_fn(threshold_vector)

        # Step 5: Check stopping condition
        if len(top_k_heap) == k and top_k_heap[0][0] >= threshold_score:
            break

    # Step 6: Extract and sort final top-k
    result = sorted(top_k_heap, key=lambda x: -x[0])
    return [(idx, score) for score, idx in result]
