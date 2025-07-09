import numpy as np

from graphs.base import Graph


class KNNGraph(Graph):
    def __init__(self, points, k):
        super().__init__(points)
        self.k = k
        self.build_graph()

    def build_graph(self):
        num_points = len(self.points)

        for i in range(num_points):
            distances = []
            for j in range(num_points):
                if i != j:
                    dist = np.linalg.norm(self.points[i] - self.points[j])
                    distances.append((dist, j))
            distances.sort(key=lambda x: x[0])
            for _, j in distances[: self.k]:
                self.adj_matrix[i].append(j)
                self.adj_matrix[j].append(i)
