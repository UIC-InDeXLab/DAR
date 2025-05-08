import numpy as np
from sklearn.preprocessing import normalize

from graphs.base import Graph


class ThetaGraph(Graph):
    def __init__(self, points, num_directions):
        super().__init__(points)
        self.num_directions = num_directions
        self.directions = []
        self.compute_directions()
        self.build_graph()
        # TODO: add distance r (both r and radius)

    def compute_directions(self, seed=40):
        """
        Generate k approximately uniform unit vectors on the sphere S^{d-1}.

        Args:
            k (int): number of directions
            d (int): dimension of the space
            seed (int): random seed for reproducibility

        Returns:
            numpy.ndarray: shape (k, d), each row is a unit vector
        """
        np.random.seed(seed)
        d = self.points.shape[1]  # dimension
        dirs = np.random.randn(self.num_directions, d)
        dirs = normalize(dirs, axis=1)
        self.directions = dirs

    def build_graph(self):
        k = self.directions.shape[0]

        for i in range(self.n):
            p = self.points[i]
            best_proj = [-np.inf] * k
            best_index = [None] * k

            for j in range(self.n):
                if i == j:
                    continue
                q = self.points[j]
                diff = q - p
                for idx, v in enumerate(self.directions):
                    proj = np.dot(diff, v)
                    if proj > 0 and proj > best_proj[idx]:
                        best_proj[idx] = proj
                        best_index[idx] = j

            for j in best_index:
                if j is not None:
                    self.adj_matrix[i].append(j)
                    self.adj_matrix[j].append(i)
