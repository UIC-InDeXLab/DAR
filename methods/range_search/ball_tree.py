import numpy as np
import math
from ranges.stripe_range import StripeRange
from sklearn.neighbors import BallTree
from tqdm import tqdm


class BallTreeIndex:
    def __init__(self, points: np.ndarray, leaf_size=40, metric="euclidean"):
        """
        Initialize Ball Tree for range searching within stripe regions.

        Args:
            points (np.ndarray): Points to index
            leaf_size (int): Maximum number of points in leaf nodes
            metric (str): Distance metric to use
        """
        self.points = points
        self.leaf_size = leaf_size
        self.metric = metric
        self.tree = None

    def build_index(self):
        """
        Build the ball tree index.
        """
        print("Building Ball Tree index...")
        self.tree = BallTree(self.points, leaf_size=self.leaf_size, metric=self.metric)
        print(
            f"Ball Tree built with {len(self.points)} points, leaf_size={self.leaf_size}"
        )

    def query(self, q: StripeRange):
        """
        Query the ball tree for points within the stripe range using hierarchical filtering.
        
        This approach follows the strategy from hierarchy.py:
        1. Use multiple scales/layers of ball queries
        2. Filter balls based on intersection with stripe range
        3. Progressively refine candidate set

        Args:
            q (StripeRange): The stripe range query

        Returns:
            list: Indices of points within the stripe
        """
        if self.tree is None:
            raise ValueError("Index not built. Call build_index() first.")

        result_indices = set()
        
        # Get data bounds for scale estimation
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        data_extent = np.linalg.norm(max_coords - min_coords)
        stripe_width = abs(q.end_dot - q.start_dot)
        
        # Hierarchical filtering approach with multiple scales
        # Start with large scale and progressively refine
        scales = [
            data_extent * 0.8,    # Large scale - capture broad regions
            data_extent * 0.4,    # Medium scale - refine regions  
            data_extent * 0.2,    # Small scale - local neighborhoods
            stripe_width * 2      # Fine scale - stripe-specific
        ]
        
        # Generate sample centers along the stripe for each scale
        num_centers_per_scale = [3, 5, 7, 10]  # More centers for smaller scales
        
        for scale_idx, radius in enumerate(scales):
            if radius <= 0:
                continue
                
            num_centers = num_centers_per_scale[scale_idx]
            sample_centers = self._generate_stripe_samples(q, min_coords, max_coords, num_centers)
            
            # For each sample center, create a "ball" and test intersection with stripe
            for center in sample_centers:
                # Test if this ball could intersect with the stripe
                if self._ball_intersects_stripe(center, radius, q):
                    # Query points within this ball
                    candidate_indices = self.tree.query_radius(
                        [center], r=radius, return_distance=False
                    )
                    
                    # Add candidates from this ball
                    for idx in candidate_indices[0]:
                        result_indices.add(int(idx))
        
        # Additional sampling along stripe boundaries for edge cases
        boundary_samples = self._generate_boundary_samples(q, min_coords, max_coords)
        boundary_radius = min(stripe_width * 1.5, data_extent * 0.1)
        
        for boundary_center in boundary_samples:
            if self._ball_intersects_stripe(boundary_center, boundary_radius, q):
                candidate_indices = self.tree.query_radius(
                    [boundary_center], r=boundary_radius, return_distance=False
                )
                
                for idx in candidate_indices[0]:
                    result_indices.add(int(idx))
        
        # Final filtering: check each candidate against the actual stripe condition
        final_results = []
        for idx in result_indices:
            if q.is_in(self.points[idx]):
                final_results.append(idx)
        
        return final_results
    
    def _generate_stripe_samples(self, q: StripeRange, min_coords, max_coords, num_samples=5):
        """
        Generate sample points along the stripe to use for radius queries.
        
        Args:
            q (StripeRange): The stripe query
            min_coords (np.ndarray): Minimum coordinates of the data
            max_coords (np.ndarray): Maximum coordinates of the data
            num_samples (int): Number of sample points to generate
            
        Returns:
            list: List of sample points
        """
        samples = []
        
        # Start with the basic stripe center
        stripe_center = self._estimate_stripe_center(q, min_coords, max_coords)
        samples.append(stripe_center)
        
        if num_samples <= 1:
            return samples
        
        # Generate additional samples by moving perpendicular to the normal vector
        # Find two orthogonal directions to the normal vector
        d = len(q.normal_vector)
        
        # Create orthogonal vectors
        orthogonal_vectors = []
        for i in range(min(d-1, num_samples-1)):
            # Create a vector orthogonal to the normal
            ortho = np.zeros(d)
            if i == 0:
                # Find the coordinate with the smallest absolute value in normal_vector
                min_idx = np.argmin(np.abs(q.normal_vector))
                ortho[min_idx] = 1.0
            else:
                ortho[(i-1) % d] = 1.0
                ortho[i % d] = -1.0
            
            # Make it orthogonal to the normal vector using Gram-Schmidt
            ortho = ortho - np.dot(ortho, q.normal_vector) * q.normal_vector
            if np.linalg.norm(ortho) > 1e-10:
                ortho = ortho / np.linalg.norm(ortho)
                orthogonal_vectors.append(ortho)
        
        # Generate samples along orthogonal directions
        data_extent = np.linalg.norm(max_coords - min_coords)
        step_size = data_extent * 0.2
        
        for i, ortho_vec in enumerate(orthogonal_vectors):
            if len(samples) >= num_samples:
                break
            
            # Add samples in both directions along this orthogonal vector
            for direction in [-1, 1]:
                if len(samples) >= num_samples:
                    break
                sample = stripe_center + direction * step_size * ortho_vec
                samples.append(sample)
        
        return samples

    def _ball_intersects_stripe(self, center: np.ndarray, radius: float, q: StripeRange) -> bool:
        """
        Test if a ball intersects with the stripe range.
        
        This follows the same logic as hierarchy.py:
        Check if the stripe intersects with the ball by testing if the distance
        from the ball center to the stripe is less than the ball radius.
        
        Args:
            center (np.ndarray): Center of the ball
            radius (float): Radius of the ball  
            q (StripeRange): The stripe range query
            
        Returns:
            bool: True if the ball intersects the stripe
        """
        # Calculate distance from ball center to the stripe
        # This is the signed distance to the line defined by the normal vector
        dist_to_line = np.dot(q.normal_vector, center)
        
        # Check if the stripe intersects with the ball
        # This is the same condition as in hierarchy.py:
        # (q.start_dot - radius) <= dist_to_line <= (q.end_dot + radius)
        return (q.start_dot - radius) <= dist_to_line <= (q.end_dot + radius)
    
    def _generate_boundary_samples(self, q: StripeRange, min_coords: np.ndarray, max_coords: np.ndarray, num_samples: int = 6) -> list:
        """
        Generate sample points near the stripe boundaries to catch edge cases.
        
        Args:
            q (StripeRange): The stripe query
            min_coords (np.ndarray): Minimum coordinates of the data
            max_coords (np.ndarray): Maximum coordinates of the data
            num_samples (int): Number of boundary samples to generate
            
        Returns:
            list: List of boundary sample points
        """
        boundary_samples = []
        
        # Generate samples slightly outside the stripe boundaries
        stripe_center = self._estimate_stripe_center(q, min_coords, max_coords)
        stripe_width = abs(q.end_dot - q.start_dot)
        
        # Create samples at different distances from the stripe center along the normal direction
        offsets = np.linspace(-stripe_width * 0.6, stripe_width * 0.6, num_samples)
        
        for offset in offsets:
            boundary_sample = stripe_center + offset * q.normal_vector
            boundary_samples.append(boundary_sample)
        
        return boundary_samples

    def _estimate_stripe_center(self, q: StripeRange, min_coords, max_coords):
        """
        Estimate a center point for the stripe to use in radius queries.
        
        Args:
            q (StripeRange): The stripe query
            min_coords (np.ndarray): Minimum coordinates of the data
            max_coords (np.ndarray): Maximum coordinates of the data
            
        Returns:
            np.ndarray: Estimated center point
        """
        # The stripe is defined by two parallel hyperplanes
        # We want to find a point that lies in the middle of the stripe
        
        # Start with the center of the data bounding box
        center = (min_coords + max_coords) / 2
        
        # Project this center onto the normal vector
        center_dot = np.dot(center, q.normal_vector)
        
        # Calculate the desired dot product (middle of the stripe)
        target_dot = (q.start_dot + q.end_dot) / 2
        
        # Adjust the center to lie in the middle of the stripe
        adjustment = (target_dot - center_dot) * q.normal_vector
        stripe_center = center + adjustment
        
        return stripe_center
