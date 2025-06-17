import numpy as np
import math
from ranges.stripe_range import StripeRange
from scipy.spatial import cKDTree


class HierarchicalIndex:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.layers = []
        self.radii = []

    def build_index(self):
        current_layer = self.points
        self.layers.append(current_layer)

        while len(current_layer) > 1:
            # Sample half of the points randomly to create the next layer
            next_layer_size = len(current_layer) // 2
            next_layer_indices = np.random.choice(
                len(current_layer), next_layer_size, replace=False
            )
            next_layer = current_layer[next_layer_indices]
            self.layers.append(next_layer)

            # Build a KDTree for the current layer
            tree = cKDTree(current_layer)

            # Find the closest centroid for each point in the next layer
            distances, indices = tree.query(next_layer, k=1)

            # Track the largest radius for each centroid
            radii = np.zeros(len(current_layer))
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                radii[idx] = max(radii[idx], dist)

            self.radii.append(radii)

            # Move to the next layer
            current_layer = next_layer

    def get_layers(self):
        return self.layers

    def get_radii(self):
        return self.radii

    def query(self, range: StripeRange):
        pass


# Example usage
if __name__ == "__main__":
    points = np.random.rand(100, 2)  # Generate 100 random 2D points
    index = HierarchicalIndex(points)
    index.build_index()

    print("Layers:")
    for i, layer in enumerate(index.get_layers()):
        print(f"Layer {i}: {layer.shape[0]} points")

    print("Radii:")
    for i, radii in enumerate(index.get_radii()):
        print(f"Layer {i}: {radii}")
