import numpy as np
from ranges.stripe_range import StripeRange
from methods.utils import linear_search


class KDNode:
    def __init__(
        self, point, left=None, right=None, axis=0, bbox=None, subtree_points=None
    ):
        self.point = point  # np.ndarray (1D)
        self.left = left  # KDNode
        self.right = right  # KDNode
        self.axis = axis  # int
        self.bbox = bbox  # (min_corner, max_corner)
        self.subtree_points = subtree_points  # np.ndarray of shape (n_subtree, k)


class KDTree:
    def __init__(self, points: np.ndarray, depth=0, bbox=None):
        if points.shape[0] == 0:
            self.node = None
            return

        k = points.shape[1]
        axis = depth % k

        # Sort points along the current axis
        sorted_indices = points[:, axis].argsort()
        points = points[sorted_indices]
        median_idx = len(points) // 2
        median_point = points[median_idx]

        # Initialize bounding box if not provided
        if bbox is None:
            min_corner = np.min(points, axis=0)
            max_corner = np.max(points, axis=0)
            bbox = (min_corner, max_corner)

        # Split the bounding boxes for children
        left_bbox, right_bbox = self._split_bbox(bbox, median_point, axis)

        # Split points
        left_points = points[:median_idx]
        right_points = points[median_idx + 1 :]

        # Build children
        left_tree = (
            KDTree(left_points, depth + 1, left_bbox)
            if left_points.shape[0] > 0
            else None
        )
        right_tree = (
            KDTree(right_points, depth + 1, right_bbox)
            if right_points.shape[0] > 0
            else None
        )

        left_node = left_tree.node if left_tree else None
        right_node = right_tree.node if right_tree else None

        # Collect subtree points
        subtree_points = [median_point]
        if left_node:
            subtree_points.append(left_node.subtree_points)
        if right_node:
            subtree_points.append(right_node.subtree_points)
        subtree_points = np.vstack(subtree_points)  # shape: (n_subtree, k)

        self.node = KDNode(
            point=median_point,
            left=left_node,
            right=right_node,
            axis=axis,
            bbox=bbox,
            subtree_points=subtree_points,
        )

    def _split_bbox(self, bbox, point, axis):
        min_corner, max_corner = bbox
        left_max = max_corner.copy()
        left_max[axis] = point[axis]
        right_min = min_corner.copy()
        right_min[axis] = point[axis]
        left_bbox = (min_corner.copy(), left_max)
        right_bbox = (right_min, max_corner.copy())
        return left_bbox, right_bbox

    def get_root(self):
        return self.node

    def query(self, range: StripeRange, node=None):
        """Query the KD-Tree for points within a given stripe range."""
        if node is None:
            node = self.get_root()

        result = []

        if range.hyper_rectangle_intersect(node.bbox) == 1:  # partial overlap
            if node.right is not None:
                result += self.query(range, node.right)
            if node.left is not None:
                result += self.query(range, node.left)
            if range.is_in(node.point):
                result.append(node.point)
        elif range.hyper_rectangle_intersect(node.bbox) == 2:  # full overlap
            result += node.subtree_points.tolist()
        else:
            # No overlap, do not traverse further
            pass

        return result
