from ranges.base import Range
import numpy as np
import itertools


"""
represents a stripe between two parallel hyperplanes in a d-dimensional space.
"""


class StripeRange(Range):
    def __init__(self, normal_vector, start_dot, end_dot):
        """
        Initialize the StripeRange with a normal vector and start/end points.

        Args:
            normal_vector (list): Normal vector of the stripe.
            start (list): Start point of the stripe.
            end (list): End point of the stripe.
        """
        super().__init__()
        self.normal_vector = normal_vector
        # self.start = start
        # self.end = end
        self.start_dot = start_dot
        self.end_dot = end_dot

    def is_in(self, point):
        """
        check the dot product of point and normal vector
        if the dot product is between the two hyperplanes, return True
        else return False
        """
        dot = np.dot(point, self.normal_vector)
        # dot_product = sum(p * n for p, n in zip(point, self.normal_vector))
        return dot >= self.start_dot and dot <= self.end_dot

    def hyper_rectangle_intersect(self, hyper_rectangle):
        """
        Check if the stripe intersects with a hyper_rectangle.

        Args:
            hyper_rectangle (tuple): A tuple of two points representing the corners of the hyper_rectangle.

        Returns:
            Either: partial overlap (1), no overlap (0), or full overlap (2).
        """
        min_corner, max_corner = hyper_rectangle
        dims = len(min_corner)

        # Generate all corners of the hyperrectangle (2^d corners)
        corners = []
        for bits in itertools.product([0, 1], repeat=dims):
            corner = np.where(bits, max_corner, min_corner)
            corners.append(corner)

        corners = np.array(corners)  # shape: (2^d, d)

        # Compute dot products
        dot_products = corners @ self.normal_vector

        min_dot = np.min(dot_products)
        max_dot = np.max(dot_products)

        if self.start_dot <= min_dot and self.end_dot >= max_dot:
            # Full overlap
            return 2
        elif self.end_dot < min_dot or self.start_dot > max_dot:
            # No overlap
            return 0
        else:
            # Partial overlap
            return 1

    @staticmethod
    def sample_stripe(points: np.ndarray, r=0.5, tolerance=0.01):
        """

        Args:
            points (_type_): The point set
            r (_type_): Fraction of points to be included in the stripe.
            tolerance (float, optional): Epsilon error.

        Returns:
            _type_: _description_
        """
        n, d = points.shape
        best_stripe = None
        best_ratio_error = float("inf")

        for _ in range(10000):
            # print("Sampling stripe...", 1 / r)
            # Sample a random unit direction vector
            v = np.random.randn(d)
            v /= np.linalg.norm(v)

            projections = points @ v
            projections.sort()

            # Find the interval [a, b] with r-fraction of points
            window_size = int(r * n)
            for i in range(n - window_size + 1):
                a = projections[i]
                b = projections[i + window_size - 1]
                # actual_r = (b >= a) * window_size / n
                count = np.sum((projections >= a) & (projections <= b))
                error = abs(count / n - r)
                if error < best_ratio_error:
                    best_ratio_error = error
                    best_stripe = (v.copy(), a, b)
                if best_ratio_error <= tolerance:
                    print("Found!")
                    return StripeRange(
                        normal_vector=best_stripe[0],
                        start_dot=best_stripe[1],
                        end_dot=best_stripe[2],
                    )

        print("Didn't find!")
        return StripeRange(
            normal_vector=best_stripe[0],
            start_dot=best_stripe[1],
            end_dot=best_stripe[2],
        )
