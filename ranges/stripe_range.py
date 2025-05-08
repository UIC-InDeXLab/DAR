from ranges.base import Range


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
        dot_product = sum(p * n for p, n in zip(point, self.normal_vector))
        return dot_product >= self.start_dot and dot_product <= self.end_dot
