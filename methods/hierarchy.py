import numpy as np
import math
from ranges.stripe_range import StripeRange
from scipy.spatial import cKDTree
import miniball


class HierarchicalIndex:
    def __init__(self, points: np.ndarray, decay=2):
        """_summary_

        Args:
            decay (int, optional): _description_. The decay in the size of layers as we go higher in the hierarchy. the default is 2, meaning that each layer will have half the points of the previous layer.
        """
        self.points = points
        self.layers = []
        self.decay = decay
        self.L = math.ceil(math.log(len(points), decay))
        print("L:", self.L)
        self.edges = [
            {} for _ in range(self.L)
        ]  # self.edges[layer] = {"point": [points]}
        self.coverage = [{} for _ in range(self.L)]
        self.stats = [{} for _ in range(self.L)]

    def build_index(self):
        current_layer = np.arange(len(self.points))
        self.layers.append(current_layer)
        self.edges[0] = {point: [point] for point in current_layer}
        layer_idx = 0

        while len(current_layer) >= self.decay:
            # print("Building layer", layer_idx)
            next_layer_size = len(current_layer) // self.decay
            next_layer = current_layer[
                np.random.choice(len(current_layer), next_layer_size, replace=False)
            ]
            self.layers.append(next_layer)
            # print(len(current_layer))
            self.edges[layer_idx + 1] = {point: [point] for point in next_layer}

            # Build graph connections
            tree = cKDTree(self.points[next_layer])

            for point_index in current_layer:
                point = self.points[point_index]
                _, idx = tree.query(
                    point, k=1
                )  # Find the closest centroid in next_layer
                idx = next_layer[idx]
                self.edges[layer_idx + 1][idx].append(
                    point_index
                )  # unidirectional edge from top layer to bottom layer

            current_layer = next_layer
            layer_idx += 1

    def find_coverage(self):
        self.coverage[0] = {point: set([point]) for point in self.layers[0]}
        for layer in range(1, self.L):
            for point, neighbors in self.edges[layer].items():
                point_coverage = set()
                for neighbor in neighbors:
                    point_coverage.update(self.coverage[layer - 1][neighbor])
                    point_coverage.add(neighbor)
                self.coverage[layer][point] = point_coverage

    """
    The stats is either the max radius, ellipses, or triangle
    """

    def find_neighbor_stats(self):
        for layer in range(self.L):
            for point, coverage in self.coverage[layer].items():
                # Compute the smallest circle enclosing all points
                points = self.points[list(coverage)]
                # center = np.mean(points, axis=0)
                if len(points) == 0:
                    self.stats[layer][point] = {
                        "center": center,
                        "max_radius": 0,
                    }
                try:
                    center, radius_squared = miniball.get_bounding_ball(points)
                except np.linalg.LinAlgError:
                    # fallback: bounding circle of two furthest points
                    from scipy.spatial.distance import pdist, squareform

                    D = squareform(pdist(points))
                    i, j = np.unravel_index(np.argmax(D), D.shape)
                    center = (points[i] + points[j]) / 2
                    radius_squared = np.linalg.norm(points[i] - center) ** 2

                # max_distance = np.max(np.linalg.norm(points - center, axis=1))
                self.stats[layer][point] = {
                    "center": center,
                    "max_radius": math.sqrt(radius_squared),
                }

    def query(self, q: StripeRange):
        """
        Start from the top layer. At each layer, find all points that the max radius hits the stripe range.
        Repeat in the bottom layer only for those hit points until the end.
        """
        candidate_points = set()
        # Start from the top layer
        for point in self.layers[-1]:
            candidate_points.add(point)

        # Traverse down the layers
        for layer in reversed(range(self.L)):
            if layer == 0:
                break
            next_candidates = set()
            for point in candidate_points:
                if point not in self.edges[layer]:
                    continue
                # print(layer, point)
                stat = self.stats[layer][point]
                max_radius = stat["max_radius"]
                center = stat["center"]
                # Check if the stripe intersects with the circle of max_radius
                dist_to_line = np.dot(q.normal_vector, center)
                # dist_to_line = abs(np.dot(range.normal_vector, center) - range.start)
                # print(
                #     "layer: ",
                #     layer,
                #     "point: ",
                #     point,
                #     self.points[point],
                #     "dist_to_line: ",
                #     dist_to_line,
                #     "max_radius: ",
                #     max_radius,
                # )
                if (
                    (q.start_dot - max_radius)
                    <= dist_to_line
                    <= (q.end_dot + max_radius)
                ):
                    # Add all neighbors to next candidates
                    next_candidates.update(self.edges[layer][point])
            candidate_points = next_candidates
            # print(candidate_points)

        # Final filtering at the bottom layer
        print("candidates: ", len(candidate_points), "all: ", len(self.points))
        result = []
        for point in candidate_points:
            if q.is_in(self.points[point]):
                result.append(point)

        return result
