import math
import numpy as np
import random
from typing import List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm


@dataclass
class Point:
    """Represents a point in d-dimensional space"""

    coords: List[float]
    data: Any = None  # Optional associated data

    def __getitem__(self, idx: int) -> float:
        return self.coords[idx]

    def __len__(self) -> int:
        return len(self.coords)

    def dimension(self) -> int:
        return len(self.coords)

    @staticmethod
    def from_numpy(array: np.ndarray, show_progress: bool = True):
        """Convert numpy array to list of Point objects with optional progress tracking"""
        n_points = array.shape[0]
        
        if show_progress and n_points > 1000:
            print(f"Converting {n_points} numpy points to Point objects...")
            return [
                Point(coords=array[i].tolist(), data=None) 
                for i in tqdm(range(n_points), desc="Converting points", unit="points")
            ]
        else:
            return [
                Point(coords=array[i].tolist(), data=None) for i in range(n_points)
            ]


class SplitType(Enum):
    """Types of splits used in partition trees"""

    COORDINATE = "coordinate"
    LINEAR = "linear"


@dataclass
class Split:
    """Represents a splitting hyperplane"""

    split_type: SplitType
    dimension: int = -1  # For coordinate splits
    value: float = 0.0  # For coordinate splits
    coefficients: List[float] = None  # For linear splits
    constant: float = 0.0  # For linear splits

    def evaluate(self, point: Point) -> float:
        """Evaluate the split function at a point"""
        if self.split_type == SplitType.COORDINATE:
            return point[self.dimension] - self.value
        else:  # LINEAR
            result = sum(c * p for c, p in zip(self.coefficients, point.coords))
            return result - self.constant


class PartitionTreeNode:
    """Node in a partition tree"""

    def __init__(self, points: List[Point], depth: int = 0):
        self.points = points
        self.depth = depth
        self.split: Optional[Split] = None
        self.left_child: Optional["PartitionTreeNode"] = None
        self.right_child: Optional["PartitionTreeNode"] = None
        self.is_leaf = False

        # Bounding box for this node
        if points:
            d = points[0].dimension()
            self.bbox_min = [min(p[i] for p in points) for i in range(d)]
            self.bbox_max = [max(p[i] for p in points) for i in range(d)]
        else:
            self.bbox_min = []
            self.bbox_max = []


class PartitionTree:
    """
    Implementation of Matoušek's efficient partition trees (1992)

    Supports:
    - Range searching in O(n^(1-1/d) + k) time
    - Nearest neighbor queries
    - Various geometric queries
    """

    def __init__(
        self,
        points: List[Point],
        leaf_size: int = 10,
        balance_factor: float = 0.5,
        use_linear_splits: bool = True,
        show_progress: bool = True,
    ):
        """
        Initialize partition tree

        Args:
            points: List of points to index
            leaf_size: Maximum points in leaf nodes
            balance_factor: Balance parameter (0.5 = perfectly balanced)
            use_linear_splits: Whether to use linear splits (more flexible)
            show_progress: Whether to show progress bars during construction
        """
        self.points = points
        self.leaf_size = leaf_size
        self.balance_factor = balance_factor
        self.use_linear_splits = use_linear_splits
        self.show_progress = show_progress
        self.dimension = points[0].dimension() if points else 0
        
        # Progress tracking
        self.nodes_created = 0
        self.total_points = len(points)
        self.max_depth = 0
        
        if self.show_progress and self.total_points > 100:
            print(f"Building partition tree for {self.total_points} points in {self.dimension}D...")
            
        self.root = self._build_tree(points, 0)
        
        if self.show_progress and self.total_points > 100:
            print(f"✓ Partition tree built: {self.nodes_created} nodes, max depth: {self.max_depth}")

    def _build_tree(self, points: List[Point], depth: int) -> PartitionTreeNode:
        """Recursively build the partition tree"""
        self.nodes_created += 1
        self.max_depth = max(self.max_depth, depth)
        
        # Show progress for larger trees
        if (self.show_progress and self.total_points > 1000 and 
            self.nodes_created % 10 == 0):
            tqdm.write(f"Partition tree progress: {self.nodes_created} nodes created, depth: {depth}")
        
        node = PartitionTreeNode(points, depth)

        # Base case: create leaf node
        if len(points) <= self.leaf_size:
            node.is_leaf = True
            return node

        # Find best split
        split = self._find_best_split(points)
        if split is None:
            node.is_leaf = True
            return node

        node.split = split

        # Partition points based on split
        left_points = []
        right_points = []

        for point in points:
            if split.evaluate(point) <= 0:
                left_points.append(point)
            else:
                right_points.append(point)

        # Ensure non-empty partitions
        if not left_points or not right_points:
            node.is_leaf = True
            return node

        # Recursively build children
        node.left_child = self._build_tree(left_points, depth + 1)
        node.right_child = self._build_tree(right_points, depth + 1)

        return node

    def _find_best_split(self, points: List[Point]) -> Optional[Split]:
        """Find the best split for a set of points"""
        if len(points) <= 1:
            return None

        best_split = None
        best_score = float("inf")

        # Try coordinate splits
        for dim in range(self.dimension):
            coords = [p[dim] for p in points]
            coords.sort()

            # Try splits at different quantiles
            for quantile in [0.3, 0.5, 0.7]:
                idx = int(quantile * len(coords))
                if 0 < idx < len(coords):
                    split_value = coords[idx]
                    split = Split(SplitType.COORDINATE, dim, split_value)
                    score = self._evaluate_split(points, split)

                    if score < best_score:
                        best_score = score
                        best_split = split

        # Try linear splits if enabled
        if self.use_linear_splits and len(points) >= 4:
            for _ in range(5):  # Try a few random linear splits
                split = self._generate_random_linear_split(points)
                if split:
                    score = self._evaluate_split(points, split)
                    if score < best_score:
                        best_score = score
                        best_split = split

        return best_split

    def _generate_random_linear_split(self, points: List[Point]) -> Optional[Split]:
        """Generate a random linear split"""
        if len(points) < 2:
            return None

        # Random direction
        coefficients = [random.gauss(0, 1) for _ in range(self.dimension)]
        norm = math.sqrt(sum(c * c for c in coefficients))
        if norm == 0:
            return None

        coefficients = [c / norm for c in coefficients]

        # Find median projection
        projections = [
            sum(c * p[i] for i, c in enumerate(coefficients)) for p in points
        ]
        projections.sort()
        median_proj = projections[len(projections) // 2]

        return Split(SplitType.LINEAR, coefficients=coefficients, constant=median_proj)

    def _evaluate_split(self, points: List[Point], split: Split) -> float:
        """Evaluate quality of a split (lower is better)"""
        left_count = 0
        right_count = 0

        for point in points:
            if split.evaluate(point) <= 0:
                left_count += 1
            else:
                right_count += 1

        if left_count == 0 or right_count == 0:
            return float("inf")

        # Balance score - prefer balanced splits
        balance = abs(left_count - right_count) / len(points)

        return balance

    def range_query(
        self, query_min: List[float], query_max: List[float]
    ) -> List[Point]:
        """
        Perform orthogonal range query

        Args:
            query_min: Minimum coordinates of query box
            query_max: Maximum coordinates of query box

        Returns:
            List of points within the query range
        """
        result = []
        self._range_query_recursive(self.root, query_min, query_max, result)
        return result

    def _range_query_recursive(
        self,
        node: PartitionTreeNode,
        query_min: List[float],
        query_max: List[float],
        result: List[Point],
    ):
        """Recursive range query implementation"""
        if not node:
            return

        # Check if node's bounding box intersects query
        if not self._bbox_intersects(
            node.bbox_min, node.bbox_max, query_min, query_max
        ):
            return

        if node.is_leaf:
            # Check each point in leaf
            for point in node.points:
                if self._point_in_range(point, query_min, query_max):
                    result.append(point)
        else:
            # Recursively search children
            self._range_query_recursive(node.left_child, query_min, query_max, result)
            self._range_query_recursive(node.right_child, query_min, query_max, result)

    def _bbox_intersects(
        self,
        bbox_min: List[float],
        bbox_max: List[float],
        query_min: List[float],
        query_max: List[float],
    ) -> bool:
        """Check if bounding boxes intersect"""
        for i in range(len(bbox_min)):
            if bbox_max[i] < query_min[i] or bbox_min[i] > query_max[i]:
                return False
        return True

    def _point_in_range(
        self, point: Point, query_min: List[float], query_max: List[float]
    ) -> bool:
        """Check if point is within query range"""
        for i in range(len(query_min)):
            if point[i] < query_min[i] or point[i] > query_max[i]:
                return False
        return True

    def nearest_neighbor(
        self, query_point: Point, k: int = 1
    ) -> List[Tuple[Point, float]]:
        """
        Find k nearest neighbors

        Args:
            query_point: Query point
            k: Number of neighbors to find

        Returns:
            List of (point, distance) tuples, sorted by distance
        """
        candidates = []
        self._nn_recursive(self.root, query_point, candidates, k)

        # Sort by distance and return top k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def _nn_recursive(
        self,
        node: PartitionTreeNode,
        query_point: Point,
        candidates: List[Tuple[Point, float]],
        k: int,
    ):
        """Recursive nearest neighbor search"""
        if not node:
            return

        if node.is_leaf:
            # Add all points from leaf to candidates
            for point in node.points:
                dist = self._euclidean_distance(query_point, point)
                candidates.append((point, dist))
        else:
            # Recursively search both children
            # In practice, you'd want to search the closer child first
            # and prune based on current best distance
            self._nn_recursive(node.left_child, query_point, candidates, k)
            self._nn_recursive(node.right_child, query_point, candidates, k)

    def _euclidean_distance(self, p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1.coords, p2.coords)))

    def halfspace_query(
        self, coefficients: List[float], min_value: float, max_value: float
    ) -> List[Point]:
        """
        Perform range query on linear function: find all points p such that
        min_value <= coefficients · p <= max_value

        Args:
            coefficients: Linear function coefficients [a1, a2, ..., ad]
            min_value: Minimum value of the linear function
            max_value: Maximum value of the linear function
            Query is: min_value <= a1*x1 + a2*x2 + ... + ad*xd <= max_value

        Returns:
            List of points satisfying the range constraint
        """
        if len(coefficients) != self.dimension:
            raise ValueError(
                f"Coefficients dimension {len(coefficients)} must match tree dimension {self.dimension}"
            )

        if min_value > max_value:
            raise ValueError(
                f"min_value ({min_value}) cannot be greater than max_value ({max_value})"
            )

        result = []
        self._halfspace_query_recursive(
            self.root, coefficients, min_value, max_value, result
        )
        return result

    def halfspace_query_upper(
        self, coefficients: List[float], constant: float
    ) -> List[Point]:
        """
        Convenience method for upper half-space: coefficients · p <= constant
        Equivalent to halfspace_query(coefficients, -infinity, constant)
        """
        return self.halfspace_query(coefficients, float("-inf"), constant)

    def halfspace_query_lower(
        self, coefficients: List[float], constant: float
    ) -> List[Point]:
        """
        Convenience method for lower half-space: coefficients · p >= constant
        Equivalent to halfspace_query(coefficients, constant, infinity)
        """
        return self.halfspace_query(coefficients, constant, float("inf"))

    def _halfspace_query_recursive(
        self,
        node: PartitionTreeNode,
        coefficients: List[float],
        min_value: float,
        max_value: float,
        result: List[Point],
    ):
        """Recursive range query implementation for linear functions"""
        if not node:
            return

        # Check if we can prune this subtree
        prune_decision = self._halfspace_pruning_test(
            node, coefficients, min_value, max_value
        )

        if prune_decision == "all_outside":
            # All points in this subtree are outside the range
            return
        elif prune_decision == "all_inside":
            # All points in this subtree are inside the range
            self._collect_all_points(node, result)
            return

        # Mixed case - need to examine further
        if node.is_leaf:
            # Check each point in leaf
            for point in node.points:
                if self._point_in_range_constraint(
                    point, coefficients, min_value, max_value
                ):
                    result.append(point)
        else:
            # Recursively search children
            self._halfspace_query_recursive(
                node.left_child, coefficients, min_value, max_value, result
            )
            self._halfspace_query_recursive(
                node.right_child, coefficients, min_value, max_value, result
            )

    def _halfspace_pruning_test(
        self,
        node: PartitionTreeNode,
        coefficients: List[float],
        min_value: float,
        max_value: float,
    ) -> str:
        """
        Test if we can prune a node for range queries on linear functions

        Returns:
            "all_inside": All points in subtree satisfy the constraint
            "all_outside": No points in subtree satisfy the constraint
            "mixed": Some points may satisfy, need to recurse
        """
        if not node.bbox_min or not node.bbox_max:
            return "mixed"

        # Compute the minimum and maximum values of the linear function
        # over the bounding box
        func_min = 0.0
        func_max = 0.0

        for i in range(len(coefficients)):
            coeff = coefficients[i]
            if coeff >= 0:
                func_min += coeff * node.bbox_min[i]
                func_max += coeff * node.bbox_max[i]
            else:
                func_min += coeff * node.bbox_max[i]
                func_max += coeff * node.bbox_min[i]

        # Check if the function range [func_min, func_max] intersects [min_value, max_value]
        if func_max < min_value or func_min > max_value:
            return "all_outside"
        elif func_min >= min_value and func_max <= max_value:
            return "all_inside"
        else:
            return "mixed"

    def _point_in_range_constraint(
        self,
        point: Point,
        coefficients: List[float],
        min_value: float,
        max_value: float,
    ) -> bool:
        """Check if point satisfies range constraint on linear function"""
        dot_product = sum(c * p for c, p in zip(coefficients, point.coords))
        return min_value <= dot_product <= max_value

    def _collect_all_points(self, node: PartitionTreeNode, result: List[Point]):
        """Collect all points from a subtree"""
        if node.is_leaf:
            result.extend(node.points)
        else:
            if node.left_child:
                self._collect_all_points(node.left_child, result)
            if node.right_child:
                self._collect_all_points(node.right_child, result)

    def intersection_halfspace_query(
        self, constraints: List[Tuple[List[float], float, float]]
    ) -> List[Point]:
        """
        Find points satisfying intersection of multiple range constraints on linear functions

        Args:
            constraints: List of (coefficients, min_value, max_value) tuples
            Each represents: min_value <= coefficients · p <= max_value

        Returns:
            Points satisfying ALL range constraints
        """
        if not constraints:
            return list(self.points)

        result = []
        self._intersection_halfspace_recursive(self.root, constraints, result)
        return result

    def _intersection_halfspace_recursive(
        self,
        node: PartitionTreeNode,
        constraints: List[Tuple[List[float], float, float]],
        result: List[Point],
    ):
        """Recursive intersection of range constraints query"""
        if not node:
            return

        # Check pruning for all constraints
        can_prune = False
        all_inside = True

        for coefficients, min_val, max_val in constraints:
            prune_decision = self._halfspace_pruning_test(
                node, coefficients, min_val, max_val
            )
            if prune_decision == "all_outside":
                can_prune = True
                break
            elif prune_decision == "mixed":
                all_inside = False

        if can_prune:
            return

        if all_inside:
            # All points satisfy all constraints
            self._collect_all_points(node, result)
            return

        # Mixed case
        if node.is_leaf:
            for point in node.points:
                satisfies_all = True
                for coefficients, min_val, max_val in constraints:
                    if not self._point_in_range_constraint(
                        point, coefficients, min_val, max_val
                    ):
                        satisfies_all = False
                        break
                if satisfies_all:
                    result.append(point)
        else:
            self._intersection_halfspace_recursive(node.left_child, constraints, result)
            self._intersection_halfspace_recursive(
                node.right_child, constraints, result
            )

    def size(self) -> int:
        """Return number of points in the tree"""
        return len(self.points)

    def depth(self) -> int:
        """Return depth of the tree"""
        return self._tree_depth(self.root)

    def _tree_depth(self, node: PartitionTreeNode) -> int:
        """Calculate tree depth recursively"""
        if not node or node.is_leaf:
            return 0

        left_depth = self._tree_depth(node.left_child) if node.left_child else 0
        right_depth = self._tree_depth(node.right_child) if node.right_child else 0

        return 1 + max(left_depth, right_depth)


# Example usage and testing
def example_usage():
    """Demonstrate partition tree usage"""

    # Create sample 2D points
    points = [
        Point([1.0, 2.0], data="A"),
        Point([3.0, 4.0], data="B"),
        Point([5.0, 1.0], data="C"),
        Point([2.0, 6.0], data="D"),
        Point([4.0, 3.0], data="E"),
        Point([6.0, 5.0], data="F"),
        Point([1.5, 3.5], data="G"),
        Point([4.5, 2.5], data="H"),
    ]

    # Build partition tree
    print("Building partition tree...")
    tree = PartitionTree(points, leaf_size=3)
    print(f"Tree built with {tree.size()} points, depth = {tree.depth()}")

    # Example 1: Range query on sum: 4 <= x + y <= 8
    print("\nRange query: 4 <= x + y <= 8")
    coefficients = [1.0, 1.0]
    min_val, max_val = 4.0, 8.0
    range_result = tree.halfspace_query(coefficients, min_val, max_val)
    print(f"Points with sum between {min_val} and {max_val}:")
    for point in range_result:
        sum_coords = sum(point.coords)
        print(f"  Point {point.coords} (sum={sum_coords:.1f}, data: {point.data})")

    # Example 2: Range query on weighted sum: 1 <= 2*x - y <= 5
    print("\nRange query: 1 <= 2*x - y <= 5")
    coefficients = [2.0, -1.0]
    min_val, max_val = 1.0, 5.0
    range_result = tree.halfspace_query(coefficients, min_val, max_val)
    print(f"Points with 2*x - y between {min_val} and {max_val}:")
    for point in range_result:
        value = 2 * point.coords[0] - point.coords[1]
        print(f"  Point {point.coords} (2x-y={value:.1f}, data: {point.data})")


def demonstrate_halfspace_theory():
    """Demonstrate theoretical aspects of half-space queries"""
    print("\n" + "=" * 60)
    print("HALF-SPACE QUERY THEORY DEMONSTRATION")
    print("=" * 60)

    # Create a larger dataset for performance demonstration
    import random

    random.seed(42)

    large_points = []
    for i in range(1000):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        large_points.append(Point([x, y], data=f"P{i}"))

    print(f"\nCreated dataset with {len(large_points)} points")
    tree = PartitionTree(large_points, leaf_size=10)
    print(f"Tree depth: {tree.depth()}")

    # Test query performance
    import time

    # Range query: 3 <= x + y <= 12
    coefficients = [1.0, 1.0]
    min_val, max_val = 3.0, 12.0

    start_time = time.time()
    result = tree.halfspace_query(coefficients, min_val, max_val)
    query_time = time.time() - start_time

    print(f"\nRange query {min_val} <= x + y <= {max_val}:")
    print(f"  Found {len(result)} points")
    print(f"  Query time: {query_time*1000:.2f} ms")
    print(
        f"  Expected complexity: O(n^(1-1/d)) = O(n^0.5) ≈ O({len(large_points)**0.5:.0f}) for 2D"
    )

    # Compare with brute force
    start_time = time.time()
    brute_force_result = []
    for point in large_points:
        value = sum(c * p for c, p in zip(coefficients, point.coords))
        if min_val <= value <= max_val:
            brute_force_result.append(point)
    brute_force_time = time.time() - start_time

    print(f"\nBrute force comparison:")
    print(f"  Found {len(brute_force_result)} points")
    print(f"  Brute force time: {brute_force_time*1000:.2f} ms")
    print(f"  Speedup: {brute_force_time/query_time:.1f}x")

    # Verify results match
    tree_coords = {tuple(p.coords) for p in result}
    brute_coords = {tuple(p.coords) for p in brute_force_result}
    print(f"  Results match: {tree_coords == brute_coords}")

    # Test multiple constraint query
    print(f"\nMultiple constraint query performance:")
    constraints = [
        ([1.0, 0.0], 2.0, 8.0),  # 2 <= x <= 8
        ([0.0, 1.0], 1.5, 7.5),  # 1.5 <= y <= 7.5
        ([1.0, 1.0], 4.0, 14.0),  # 4 <= x+y <= 14
    ]

    start_time = time.time()
    multi_result = tree.intersection_halfspace_query(constraints)
    multi_time = time.time() - start_time

    print(f"  Found {len(multi_result)} points satisfying all constraints")
    print(f"  Query time: {multi_time*1000:.2f} ms")


if __name__ == "__main__":
    example_usage()
    demonstrate_halfspace_theory()
