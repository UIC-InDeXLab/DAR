from collections import defaultdict


class Graph:
    def __init__(self, points):
        self.points = points
        self.adj_matrix = defaultdict(list)
        self.n = len(points)

    def build_graph(self):
        raise NotImplementedError("Subclasses should implement this method.")
