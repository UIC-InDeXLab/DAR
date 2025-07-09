import numpy as np
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ranges.stripe_range import StripeRange


class MBR:
    """Minimum Bounding Rectangle for d-dimensional space"""

    def __init__(self, min_coords: np.ndarray, max_coords: np.ndarray):
        """
        Initialize MBR with min and max coordinates
        Args:
            min_coords: numpy array of minimum coordinates for each dimension
            max_coords: numpy array of maximum coordinates for each dimension
        """
        self.min_coords = np.array(min_coords, dtype=float)
        self.max_coords = np.array(max_coords, dtype=float)
        self.dimensions = len(self.min_coords)

        if len(self.max_coords) != self.dimensions:
            raise ValueError("min_coords and max_coords must have same dimensionality")

    @classmethod
    def from_point(cls, point: np.ndarray):
        """Create MBR from a single point"""
        point = np.array(point)
        return cls(point, point)

    @classmethod
    def from_points(cls, points: np.ndarray):
        """Create MBR that encompasses all points"""
        if len(points) == 0:
            return cls(np.array([0.0]), np.array([0.0]))

        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return cls(min_coords, max_coords)

    def area(self) -> float:
        """Calculate hypervolume of the MBR"""
        extents = self.max_coords - self.min_coords
        # Avoid zero area for point MBRs by using small epsilon
        extents = np.maximum(extents, 1e-10)
        return np.prod(extents)

    def union(self, other: "MBR") -> "MBR":
        """Create union of two MBRs"""
        if self.dimensions != other.dimensions:
            raise ValueError("Cannot union MBRs of different dimensions")

        min_coords = np.minimum(self.min_coords, other.min_coords)
        max_coords = np.maximum(self.max_coords, other.max_coords)
        return MBR(min_coords, max_coords)

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if MBR contains a point"""
        point = np.array(point)
        if len(point) != self.dimensions:
            raise ValueError("Point dimension doesn't match MBR dimension")

        return np.all(self.min_coords <= point) and np.all(point <= self.max_coords)

    def intersects(self, other: "MBR") -> bool:
        """Check if this MBR intersects with another MBR"""
        if self.dimensions != other.dimensions:
            return False

        return np.all(self.max_coords >= other.min_coords) and np.all(
            other.max_coords >= self.min_coords
        )

    def __str__(self):
        return f"MBR(min={self.min_coords}, max={self.max_coords})"


class RTreeNode:
    """R-tree node (can be leaf or internal)"""

    def __init__(
        self, is_leaf: bool = False, max_entries: int = 4, dimensions: int = 2
    ):
        self.is_leaf = is_leaf
        self.max_entries = max_entries
        self.dimensions = dimensions
        self.entries = []  # List of (MBR, data/child_node)
        self.parent = None

    def is_full(self) -> bool:
        return len(self.entries) >= self.max_entries

    def add_entry(self, mbr: MBR, data_or_child):
        """Add an entry to this node"""
        self.entries.append((mbr, data_or_child))
        if not self.is_leaf:
            data_or_child.parent = self

    def get_mbr(self) -> MBR:
        """Get the MBR that encompasses all entries in this node"""
        if not self.entries:
            return MBR(np.zeros(self.dimensions), np.zeros(self.dimensions))

        mbr = self.entries[0][0]
        for entry_mbr, _ in self.entries[1:]:
            mbr = mbr.union(entry_mbr)
        return mbr


class RTree:
    """R-tree implementation for d-dimensional points"""

    def __init__(self, max_entries: int = 4, dimensions: int = 2):
        self.max_entries = max_entries
        self.min_entries = max_entries // 2
        self.dimensions = dimensions
        self.root = RTreeNode(
            is_leaf=True, max_entries=max_entries, dimensions=dimensions
        )
        self.size = 0

    def insert_points(self, points: np.ndarray):
        """Insert multiple points into the R-tree"""
        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Validate dimensions
        if points.shape[1] != self.dimensions:
            raise ValueError(f"Points must have {self.dimensions} dimensions")

        for i, point in enumerate(points):
            self.insert_point(point, i)  # Use index as data
        self.points = points.copy()

    def insert_point(self, point: np.ndarray, data=None):
        """Insert a single point into the R-tree"""
        point = np.array(point)
        if len(point) != self.dimensions:
            raise ValueError(f"Point must have {self.dimensions} dimensions")

        mbr = MBR.from_point(point)
        self._insert(mbr, data if data is not None else point)
        self.size += 1

    def _insert(self, mbr: MBR, data):
        """Internal insert method"""
        leaf = self._choose_leaf(mbr)
        leaf.add_entry(mbr, data)

        if leaf.is_full():
            self._split_node(leaf)

    def _choose_leaf(self, mbr: MBR) -> RTreeNode:
        """Choose the best leaf node to insert the new entry"""
        node = self.root

        while not node.is_leaf:
            best_child = None
            min_enlargement = float("inf")
            min_area = float("inf")

            for entry_mbr, child in node.entries:
                enlarged_mbr = entry_mbr.union(mbr)
                enlargement = enlarged_mbr.area() - entry_mbr.area()

                if enlargement < min_enlargement or (
                    enlargement == min_enlargement and entry_mbr.area() < min_area
                ):
                    min_enlargement = enlargement
                    min_area = entry_mbr.area()
                    best_child = child

            node = best_child

        return node

    def _split_node(self, node: RTreeNode):
        """Split an overflowing node using linear split algorithm"""
        entries = node.entries[:]

        # Find the pair of entries with maximum separation
        max_separation = -1
        seed1_idx = seed2_idx = 0

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                mbr1, mbr2 = entries[i][0], entries[j][0]
                union_mbr = mbr1.union(mbr2)
                separation = union_mbr.area() - mbr1.area() - mbr2.area()

                if separation > max_separation:
                    max_separation = separation
                    seed1_idx, seed2_idx = i, j

        # Create two new nodes
        node1 = RTreeNode(
            is_leaf=node.is_leaf,
            max_entries=self.max_entries,
            dimensions=self.dimensions,
        )
        node2 = RTreeNode(
            is_leaf=node.is_leaf,
            max_entries=self.max_entries,
            dimensions=self.dimensions,
        )

        # Add seed entries
        node1.add_entry(*entries[seed1_idx])
        node2.add_entry(*entries[seed2_idx])

        # Distribute remaining entries
        remaining = [
            entries[i] for i in range(len(entries)) if i not in (seed1_idx, seed2_idx)
        ]

        for mbr, data_or_child in remaining:
            mbr1 = node1.get_mbr().union(mbr)
            mbr2 = node2.get_mbr().union(mbr)

            enlargement1 = mbr1.area() - node1.get_mbr().area()
            enlargement2 = mbr2.area() - node2.get_mbr().area()

            if enlargement1 < enlargement2:
                node1.add_entry(mbr, data_or_child)
            else:
                node2.add_entry(mbr, data_or_child)

        # Handle root split
        if node == self.root:
            new_root = RTreeNode(
                is_leaf=False, max_entries=self.max_entries, dimensions=self.dimensions
            )
            new_root.add_entry(node1.get_mbr(), node1)
            new_root.add_entry(node2.get_mbr(), node2)
            self.root = new_root
        else:
            # Replace the old node with the two new nodes in parent
            parent = node.parent
            parent.entries = [
                (mbr, child) for mbr, child in parent.entries if child != node
            ]
            parent.add_entry(node1.get_mbr(), node1)
            parent.add_entry(node2.get_mbr(), node2)

            if parent.is_full():
                self._split_node(parent)

    def traverse_tree(self, visit_func=None):
        """Traverse the entire tree structure"""
        if visit_func is None:
            visit_func = lambda node, level, mbr, data: print(
                f"{'  ' * level}Level {level}: {mbr} -> {data}"
            )

        def _traverse_recursive(node: RTreeNode, level: int):
            node_mbr = node.get_mbr()

            if node.is_leaf:
                print(
                    f"{'  ' * level}LEAF Node (Level {level}): {len(node.entries)} points"
                )
                for mbr, data in node.entries:
                    visit_func(node, level + 1, mbr, data)
            else:
                print(
                    f"{'  ' * level}INTERNAL Node (Level {level}): {len(node.entries)} children"
                )
                for mbr, child in node.entries:
                    visit_func(
                        node,
                        level + 1,
                        mbr,
                        f"Child node with {len(child.entries)} entries",
                    )
                    _traverse_recursive(child, level + 1)

        print(f"=== R-tree Structure ({self.dimensions}D) ===")
        print(f"Tree size: {self.size} points")
        _traverse_recursive(self.root, 0)
        print("========================")

    def range_query(self, query_mbr: MBR) -> List:
        """Find all points within the query MBR"""
        results = []

        def _search_recursive(node: RTreeNode):
            for mbr, data_or_child in node.entries:
                if mbr.intersects(query_mbr):
                    if node.is_leaf:
                        if isinstance(data_or_child, np.ndarray):
                            if query_mbr.contains_point(data_or_child):
                                results.append(data_or_child)
                        else:
                            results.append(data_or_child)
                    else:
                        _search_recursive(data_or_child)

        _search_recursive(self.root)
        return results

    def query(
        self,
        query: StripeRange,
        node: Optional[RTreeNode] = None,
    ) -> List:
        if node is None:
            node = self.root

        result = []

        # Check if the node's MBR intersects with the stripe
        node_mbr = node.get_mbr()
        if (
            query.hyper_rectangle_intersect((node_mbr.min_coords, node_mbr.max_coords))
            != 0
        ):
            if node.is_leaf:
                # Check each point in the leaf
                for mbr, data in node.entries:
                    if isinstance(data, np.ndarray):
                        point = data
                    else:
                        # If data is not the point itself, get it from MBR center
                        point = (mbr.min_coords + mbr.max_coords) / 2

                    # print(point, data)
                    if query.is_in(point):
                        result.append(self.points[data])
            else:
                # Recursively search child nodes
                for mbr, child in node.entries:
                    result.extend(self.query(query, child))

        return result
