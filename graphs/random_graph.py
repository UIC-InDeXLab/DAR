import random
import numpy as np

from graphs.base import Graph


class RandomGraph(Graph):
    def __init__(self, points, degree=10):
        super().__init__(points)
        self.degree = degree
        self.build_graph()

    def build_graph(self):
        num_points = len(self.points)

        for i in range(num_points):
            neighbors = random.sample(range(num_points), self.degree)
            for j in neighbors:
                if j != i and j not in self.adj_matrix[i]:
                    self.adj_matrix[i].append(j)
                    self.adj_matrix[j].append(i)
