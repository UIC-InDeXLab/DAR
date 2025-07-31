import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LineSegment:
    """Represents a line segment with start and end points."""

    start: Tuple[float, float]
    end: Tuple[float, float]
    line_index: int  # Index of the original line this segment comes from
    start_angle: float = 0.0  # Starting angle for sorting purposes

    def __str__(self):
        return f"Segment from {self.start} to {self.end} (Line {self.line_index}, angle {self.start_angle:.3f})"


def line_intersection(
    line1: Tuple[float, float, float], line2: Tuple[float, float, float]
) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two lines ax + by + c = 0.

    Args:
        line1, line2: Tuples (a, b, c) representing lines

    Returns:
        Intersection point (x, y) or None if lines are parallel
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2

    # Calculate determinant
    det = a1 * b2 - a2 * b1

    if abs(det) < 1e-10:  # Lines are parallel
        return None

    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det

    return (x, y)


def ray_line_intersection(
    ray_angle: float, line: Tuple[float, float, float]
) -> Optional[Tuple[float, float]]:
    """
    Find intersection of a ray from origin with a line.

    Args:
        ray_angle: Angle of ray in radians (0 = positive x-axis)
        line: Tuple (a, b, c) representing line ax + by + c = 0

    Returns:
        Intersection point or None if no intersection in positive ray direction
    """
    a, b, c = line

    # Ray direction vector
    dx, dy = math.cos(ray_angle), math.sin(ray_angle)

    # Ray equation: (x, y) = t * (dx, dy) for t >= 0
    # Substitute into line equation: a(t*dx) + b(t*dy) + c = 0
    # Solve for t: t = -c / (a*dx + b*dy)

    denominator = a * dx + b * dy

    if abs(denominator) < 1e-10:  # Ray is parallel to line
        return None

    t = -c / denominator

    if t < 1e-10:  # Intersection is behind the origin
        return None

    x = t * dx
    y = t * dy

    return (x, y)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2π) range."""
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle


def point_to_angle(point: Tuple[float, float]) -> float:
    """Convert a point to its angle from origin."""
    x, y = point
    angle = math.atan2(y, x)
    return normalize_angle(angle)


def segment_contains_angle(segment: LineSegment, target_angle: float) -> bool:
    """Check if a line segment's angular span contains the target angle."""
    start_angle = point_to_angle(segment.start)
    end_angle = point_to_angle(segment.end)
    target_angle = normalize_angle(target_angle)

    # Handle wrap-around case
    if start_angle <= end_angle:
        return start_angle <= target_angle <= end_angle
    else:  # Segment crosses 0° line
        return target_angle >= start_angle or target_angle <= end_angle


def binary_search_segment_by_ray(
    segments: List[LineSegment], ray_angle: float
) -> Optional[LineSegment]:
    """
    Binary search to find the segment that intersects with the given ray angle.

    Args:
        segments: List of LineSegment objects sorted by start_angle (counter-clockwise)
        ray_angle: Angle of the ray in radians (from positive x-axis, counter-clockwise)

    Returns:
        LineSegment that intersects the ray, or None if no intersection found
    """
    if not segments:
        return None

    ray_angle = normalize_angle(ray_angle)

    # Check each segment to see if it contains the ray angle
    # Since segments might not be contiguous in angle space, we need to check each one
    for segment in segments:
        if segment_contains_angle(segment, ray_angle):
            # Verify by computing actual intersection
            intersection = ray_line_intersection_with_segment(ray_angle, segment)
            if intersection is not None:
                return segment

    return None


def ray_line_intersection_with_segment(
    ray_angle: float, segment: LineSegment
) -> Optional[Tuple[float, float]]:
    """
    Find intersection of a ray from origin with a line segment.

    Args:
        ray_angle: Angle of ray in radians
        segment: LineSegment object

    Returns:
        Intersection point or None if no intersection within the segment
    """
    # Get the line equation from the segment
    # We need to reconstruct the line equation from the segment
    x1, y1 = segment.start
    x2, y2 = segment.end

    # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    # Simplify to ax + by + c = 0
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1

    # Find intersection with ray
    intersection = ray_line_intersection(ray_angle, (a, b, c))

    if intersection is None:
        return None

    # Check if intersection point lies within the segment bounds
    ix, iy = intersection

    # Check if the intersection point is within the segment
    # Use parametric form: P = P1 + t(P2 - P1), where 0 <= t <= 1
    if abs(x2 - x1) > abs(y2 - y1):
        t = (ix - x1) / (x2 - x1) if abs(x2 - x1) > 1e-10 else 0
    else:
        t = (iy - y1) / (y2 - y1) if abs(y2 - y1) > 1e-10 else 0

    if 0 <= t <= 1:
        return intersection

    return None


def find_intersecting_segment_optimized(
    segments: List[LineSegment], ray_angle: float
) -> Optional[Tuple[LineSegment, Tuple[float, float]]]:
    """
    Optimized function to find the segment intersecting a ray and return both segment and intersection point.

    Args:
        segments: List of LineSegment objects sorted by start_angle
        ray_angle: Angle of the ray in radians

    Returns:
        Tuple of (intersecting_segment, intersection_point) or None if no intersection
    """
    if not segments:
        return None

    ray_angle = normalize_angle(ray_angle)

    # Binary search approach: find the segment whose angular range contains the ray
    left, right = 0, len(segments) - 1

    while left <= right:
        mid = (left + right) // 2
        segment = segments[mid]

        # Check if this segment intersects the ray
        intersection = ray_line_intersection_with_segment(ray_angle, segment)
        if intersection is not None:
            return (segment, intersection)

        # Determine which direction to search
        segment_start_angle = normalize_angle(segment.start_angle)
        segment_end_angle = point_to_angle(segment.end)

        # Handle the complex case of angular ordering
        if segment_start_angle <= ray_angle:
            if (
                segment_end_angle >= ray_angle
                or segment_end_angle < segment_start_angle
            ):
                # Ray might be in this segment's range, but no intersection found
                # Try adjacent segments
                break
            else:
                left = mid + 1
        else:
            right = mid - 1

    # If binary search doesn't find it, fall back to linear search
    # This handles edge cases and wrap-around situations
    for segment in segments:
        intersection = ray_line_intersection_with_segment(ray_angle, segment)
        if intersection is not None:
            return (segment, intersection)

    return None


def get_line_intersections_on_ray(
    ray_angle: float, lines: List[Tuple[float, float, float]]
) -> List[Tuple[float, int]]:
    """
    Get all line intersections with a ray, sorted by distance from origin.

    Args:
        ray_angle: Ray angle in radians
        lines: List of lines as (a, b, c) tuples

    Returns:
        List of (distance, line_index) tuples sorted by distance
    """
    intersections = []

    for i, line in enumerate(lines):
        point = ray_line_intersection(ray_angle, line)
        if point is not None:
            x, y = point
            distance = math.sqrt(x * x + y * y)
            intersections.append((distance, i))

    # Sort by distance from origin
    intersections.sort(key=lambda x: x[0])

    return intersections


def get_kth_level_arrangement(
    lines: List[Tuple[float, float, float]], k: int, num_rays: int = 3600
) -> List[LineSegment]:
    """
    Compute the k-th level of line arrangement using ray sweeping.

    Args:
        lines: List of lines as (a, b, c) tuples representing ax + by + c = 0
        k: Level number (1-indexed, 1st level is closest to origin)
        num_rays: Number of rays to use for sweeping (more rays = higher precision)

    Returns:
        List of LineSegment objects representing the k-th level, sorted by angle
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    if k > len(lines):
        return []  # k-th level doesn't exist

    # Get all intersection points between lines
    intersection_points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            point = line_intersection(lines[i], lines[j])
            if point is not None:
                x, y = point
                angle = math.atan2(y, x)
                if angle < 0:
                    angle += 2 * math.pi
                intersection_points.append(angle)

    # Add some extra angles to ensure we capture all changes
    intersection_points.extend([0, math.pi / 2, math.pi, 3 * math.pi / 2])
    intersection_points = sorted(set(intersection_points))

    # Create rays at critical angles and between them
    ray_angles = []
    for i in range(len(intersection_points)):
        ray_angles.append(intersection_points[i])
        # Add angle slightly after each intersection
        next_angle = intersection_points[(i + 1) % len(intersection_points)]
        if next_angle <= intersection_points[i]:
            next_angle += 2 * math.pi
        mid_angle = (intersection_points[i] + next_angle) / 2
        if mid_angle >= 2 * math.pi:
            mid_angle -= 2 * math.pi
        ray_angles.append(mid_angle)

    # Remove duplicates and sort
    ray_angles = sorted(set(ray_angles))

    # For each ray, find which line is at the k-th level
    kth_level_data = []

    for angle in ray_angles:
        intersections = get_line_intersections_on_ray(angle, lines)
        if len(intersections) >= k:
            _, line_idx = intersections[k - 1]  # k-1 because of 0-indexing

            # Get the intersection point
            point = ray_line_intersection(angle, lines[line_idx])
            if point is not None:
                kth_level_data.append((angle, line_idx, point))

    # Group consecutive segments belonging to the same line and store with start angle
    segments = []
    if not kth_level_data:
        return segments

    current_line = kth_level_data[0][1]
    segment_start_angle = kth_level_data[0][0]
    segment_start_point = kth_level_data[0][2]

    for i in range(1, len(kth_level_data)):
        angle, line_idx, point = kth_level_data[i]

        if line_idx != current_line:
            # End current segment
            prev_angle, prev_line_idx, prev_point = kth_level_data[i - 1]
            segment = LineSegment(segment_start_point, prev_point, current_line)
            # Store the start angle for sorting
            segment.start_angle = segment_start_angle
            segments.append(segment)

            # Start new segment
            current_line = line_idx
            segment_start_angle = angle
            segment_start_point = point

    # Add the last segment
    if kth_level_data:
        last_angle, last_line_idx, last_point = kth_level_data[-1]
        segment = LineSegment(segment_start_point, last_point, current_line)
        segment.start_angle = segment_start_angle
        segments.append(segment)

    # Sort segments by their start angle (counter-clockwise order)
    segments.sort(key=lambda seg: getattr(seg, "start_angle", 0))

    return segments


def plot_points_dual_lines_and_kth_level(
    points: np.ndarray, k: int, plot_range: float = 5
):
    """
    Plot original points, their dual lines, and the k-th level arrangement.

    Args:
        points: numpy array of shape (n, 2) containing 2D points
        k: Level number to visualize
        plot_range: Range for x and y axes ([-plot_range, plot_range])
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # Convert points to dual lines
    dual_lines = []
    for point in points:
        a, b = point
        dual_lines.append((a, b, -1))  # ax + by = 1

    # Get k-th level arrangement
    kth_level_segments = get_kth_level_arrangement(dual_lines, k)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Original points
    ax1.scatter(
        points[:, 0], points[:, 1], c="red", s=100, zorder=5, label="Original Points"
    )
    for i, (x, y) in enumerate(points):
        ax1.annotate(
            f"P{i}({x:.1f}, {y:.1f})",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    # Add origin
    ax1.scatter(0, 0, c="black", s=100, marker="x", zorder=5, label="Origin")

    ax1.set_xlim(-plot_range, plot_range)
    ax1.set_ylim(-plot_range, plot_range)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Original Points")
    ax1.legend()
    ax1.set_aspect("equal")

    # Plot 2: Dual lines and k-th level
    colors = plt.cm.tab10(np.linspace(0, 1, len(dual_lines)))

    # Plot dual lines
    x_vals = np.linspace(-plot_range, plot_range, 1000)
    for i, (a, b, c) in enumerate(dual_lines):
        if abs(b) > 1e-10:  # Not vertical line
            y_vals = (-a * x_vals - c) / b
        else:  # Vertical line
            x_line = -c / a
            y_vals = np.linspace(-plot_range, plot_range, 1000)
            x_vals_line = np.full_like(y_vals, x_line)
            ax2.plot(
                x_vals_line,
                y_vals,
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=f"Line {i}: {a:.1f}x + {b:.1f}y = 1",
            )
            continue

        # Only plot within the range
        mask = (y_vals >= -plot_range) & (y_vals <= plot_range)
        if np.any(mask):
            ax2.plot(
                x_vals[mask],
                y_vals[mask],
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=f"Line {i}: {a:.1f}x + {b:.1f}y = 1",
            )

    # Plot k-th level segments
    if kth_level_segments:
        for i, segment in enumerate(kth_level_segments):
            x_coords = [segment.start[0], segment.end[0]]
            y_coords = [segment.start[1], segment.end[1]]
            ax2.plot(
                x_coords,
                y_coords,
                "red",
                linewidth=4,
                alpha=0.8,
                label=f"{k}-th Level" if i == 0 else "",
            )

            # Mark endpoints
            ax2.scatter(x_coords, y_coords, c="red", s=50, zorder=5)

    # Add origin
    ax2.scatter(0, 0, c="black", s=100, marker="x", zorder=5, label="Origin")

    ax2.set_xlim(-plot_range, plot_range)
    ax2.set_ylim(-plot_range, plot_range)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title(f"Dual Lines and {k}-th Level Arrangement")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    # Print information
    print(f"Original points: {points.tolist()}")
    print(f"\nDual lines:")
    for i, (a, b, c) in enumerate(dual_lines):
        print(f"  Line {i}: {a}x + {b}y = 1")

    print(f"\n{k}-th level arrangement (sorted by sweep angle):")
    if kth_level_segments:
        for i, segment in enumerate(kth_level_segments):
            angle_deg = math.degrees(segment.start_angle)
            print(
                f"  Segment {i+1}: from {segment.start} to {segment.end} (Line {segment.line_index}, start angle: {angle_deg:.1f}°)"
            )
    else:
        print(f"  No {k}-th level exists")