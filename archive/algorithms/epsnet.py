import numpy as np


class EpsNet:
    def __init__(self, points: np.ndarray, epsnet_size):
        self.epsnet_size = epsnet_size
        self.points = points
        self.epsnet_indices = self.find_epsnet()

    def find_epsnet(self):
        """
        Randomly draw a sample of size epsnet_size from the points.
        """
        if self.epsnet_size > len(self.points):
            raise ValueError("epsnet_size cannot be larger than the number of points.")
        indices = np.random.choice(len(self.points), self.epsnet_size, replace=False)
        return indices
