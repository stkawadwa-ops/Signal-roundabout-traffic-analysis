"""
Shapely-based geometric operations for roundabout analysis.

This module uses Shapely for robust geometric computations:
1. Zone definitions (entry, circulatory, exit)
2. Turning radius calculations
3. Trajectory intersection detection
4. Lane boundary detection
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from shapely.geometry import (
    Point, LineString, Polygon, MultiPolygon,
    box,
)
from shapely.ops import nearest_points, split, unary_union
import math

logger = logging.getLogger(__name__)


@dataclass
class RoundaboutGeometry:
    """Represents the geometric structure of a roundabout."""
    center: Tuple[float, float]
    outer_radius: float
    inner_radius: float
    entry_zones: List[Polygon]  # Entry lane polygons
    exit_zones: List[Polygon]   # Exit lane polygons
    circulatory_zone: Polygon   # Circulatory lane
    
    def __post_init__(self):
        """Validate and compute derived geometry."""
        if self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must be > inner_radius")
        
        # Create circulatory zone if not provided
        if self.circulatory_zone is None:
            self.circulatory_zone = self._create_circulatory_zone()
    
    def _create_circulatory_zone(self) -> Polygon:
        """Create circulatory zone as annulus (ring)."""
        cx, cy = self.center
        
        # Outer circle
        outer_circle = Point(cx, cy).buffer(self.outer_radius)
        
        # Inner circle (hole)
        inner_circle = Point(cx, cy).buffer(self.inner_radius)
        
        # Annulus
        circulatory = outer_circle.difference(inner_circle)
        
        return circulatory


class ZoneProcessor:
    """
    Classify vehicle trajectories into roundabout zones:
    - Entry: Entering the roundabout
    - Circulatory: Within the roundabout
    - Exit: Leaving the roundabout
    """
    
    def __init__(self, roundabout_geom: RoundaboutGeometry):
        self.geom = roundabout_geom
        self.entry_polygons = [Polygon(zone) if isinstance(zone, (list, tuple)) else zone 
                               for zone in roundabout_geom.entry_zones]
        self.exit_polygons = [Polygon(zone) if isinstance(zone, (list, tuple)) else zone 
                              for zone in roundabout_geom.exit_zones]
        self.circulatory_polygon = roundabout_geom.circulatory_zone
    
    def classify_point(self, point: Tuple[float, float]) -> str:
        """
        Classify a single point into zone.
        
        Args:
            point: (x, y) coordinate
            
        Returns:
            Zone: 'entry', 'circulatory', 'exit', or 'outside'
        """
        p = Point(point)
        
        # Check entry zones
        for entry_zone in self.entry_polygons:
            if p.within(entry_zone) or p.touches(entry_zone):
                return 'entry'
        
        # Check circulatory zone
        if p.within(self.circulatory_polygon) or p.touches(self.circulatory_polygon):
            return 'circulatory'
        
        # Check exit zones
        for exit_zone in self.exit_polygons:
            if p.within(exit_zone) or p.touches(exit_zone):
                return 'exit'
        
        return 'outside'
    
    def classify_trajectory(self, trajectory: np.ndarray) -> Dict[str, List[int]]:
        """
        Classify trajectory points into zones.
        
        Args:
            trajectory: (N, 2) array of points
            
        Returns:
            Dict mapping zone name to list of frame indices
        """
        zones = {
            'entry': [],
            'circulatory': [],
            'exit': [],
            'outside': []
        }
        
        for i, point in enumerate(trajectory):
            zone = self.classify_point(tuple(point))
            zones[zone].append(i)
        
        return zones
    
    def get_zone_transitions(self, trajectory: np.ndarray) -> List[Tuple[int, str, str]]:
        """
        Get transitions between zones along trajectory.
        
        Args:
            trajectory: (N, 2) array of points
            
        Returns:
            List of (frame_idx, from_zone, to_zone) tuples
        """
        transitions = []
        prev_zone = None
        
        for i, point in enumerate(trajectory):
            curr_zone = self.classify_point(tuple(point))
            
            if prev_zone is not None and curr_zone != prev_zone:
                transitions.append((i, prev_zone, curr_zone))
            
            prev_zone = curr_zone
        
        return transitions


class TurningGeometry:
    """
    Calculate turning radius and other geometric parameters from trajectories.
    
    Uses circle fitting to estimate turning radius from vehicle path.
    """
    
    @staticmethod
    def fit_circle(points: np.ndarray, min_points: int = 5) -> Optional[Dict]:
        """
        Fit circle to trajectory points using least squares.
        
        Args:
            points: (N, 2) array of trajectory points
            min_points: Minimum points needed for fitting
            
        Returns:
            Dict with 'center', 'radius', 'residual' or None if fit fails
        """
        if len(points) < min_points:
            return None
        
        try:
            # Convert to homogeneous coordinates for least squares
            x = points[:, 0]
            y = points[:, 1]
            
            # Set up system: (x - cx)^2 + (y - cy)^2 = r^2
            # Rearrange to linear system
            A = np.column_stack([x, y, np.ones(len(x))])
            b = x**2 + y**2
            
            # Solve: A^T @ A @ c = A^T @ b
            c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            cx, cy, c3 = c
            cx /= 2
            cy /= 2
            
            # Compute radius
            radius = np.sqrt(cx**2 + cy**2 + c3 / 2)
            
            # Compute residual
            distances = np.sqrt((x - cx)**2 + (y - cy)**2)
            residual = np.mean(np.abs(distances - radius))
            
            return {
                'center': np.array([cx, cy]),
                'radius': radius,
                'residual': residual,
                'num_points': len(points)
            }
        
        except Exception as e:
            logger.warning(f"Circle fitting failed: {e}")
            return None
    
    @staticmethod
    def compute_curvature(trajectory: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Compute curvature along trajectory (inverse of turning radius).
        
        Args:
            trajectory: (N, 2) array of points
            window: Window size for local curvature
            
        Returns:
            (N,) array of curvature values
        """
        if len(trajectory) < window:
            return np.zeros(len(trajectory))
        
        curvatures = np.zeros(len(trajectory))
        
        for i in range(window // 2, len(trajectory) - window // 2):
            # Get local window
            start = i - window // 2
            end = i + window // 2 + 1
            local = trajectory[start:end]
            
            # Fit circle to local window
            fit_result = TurningGeometry.fit_circle(local, min_points=3)
            
            if fit_result:
                curvatures[i] = 1.0 / (fit_result['radius'] + 1e-6)
            else:
                curvatures[i] = 0.0
        
        return curvatures
    
    @staticmethod
    def estimate_turning_radius(trajectory: np.ndarray) -> Optional[float]:
        """
        Estimate overall turning radius of trajectory.
        
        Args:
            trajectory: (N, 2) array of points
            
        Returns:
            Estimated turning radius in pixels or None
        """
        fit_result = TurningGeometry.fit_circle(trajectory, min_points=10)
        
        if fit_result:
            return fit_result['radius']
        
        return None
    
    @staticmethod
    def compute_path_length(trajectory: np.ndarray) -> float:
        """
        Compute total path length of trajectory.
        
        Args:
            trajectory: (N, 2) array of points
            
        Returns:
            Total path length
        """
        if len(trajectory) < 2:
            return 0.0
        
        diffs = np.diff(trajectory, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        return np.sum(distances)
    
    @staticmethod
    def compute_angular_velocity(trajectory: np.ndarray, frame_rate: float = 30.0) -> np.ndarray:
        """
        Compute angular velocity along trajectory.
        
        Args:
            trajectory: (N, 2) array of points
            frame_rate: Frames per second
            
        Returns:
            (N-1,) array of angular velocities (radians/second)
        """
        if len(trajectory) < 3:
            return np.array([])
        
        # Compute angles at each point
        angles = np.arctan2(
            trajectory[:, 1] - trajectory[0, 1],
            trajectory[:, 0] - trajectory[0, 0]
        )
        
        # Compute angular differences
        angle_diffs = np.diff(angles)
        
        # Normalize to [-pi, pi]
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
        
        # Convert to angular velocity (rad/s)
        dt = 1.0 / frame_rate
        angular_velocity = angle_diffs / dt
        
        return angular_velocity


class TrajectoryIntersection:
    """
    Detect and analyze trajectory intersections.
    
    Useful for understanding vehicle interactions and conflicts.
    """
    
    @staticmethod
    def compute_trajectory_line(trajectory: np.ndarray) -> LineString:
        """
        Convert trajectory points to LineString.
        
        Args:
            trajectory: (N, 2) array of points
            
        Returns:
            Shapely LineString object
        """
        return LineString(trajectory)
    
    @staticmethod
    def detect_intersections(traj1: np.ndarray, traj2: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect intersection points between two trajectories.
        
        Args:
            traj1: (N, 2) array of trajectory 1 points
            traj2: (M, 2) array of trajectory 2 points
            
        Returns:
            List of intersection points (x, y)
        """
        line1 = TrajectoryIntersection.compute_trajectory_line(traj1)
        line2 = TrajectoryIntersection.compute_trajectory_line(traj2)
        
        intersection = line1.intersection(line2)
        
        if intersection.is_empty:
            return []
        
        if intersection.geom_type == 'Point':
            return [(intersection.x, intersection.y)]
        elif intersection.geom_type == 'LineString':
            # Trajectories overlap
            return [(pt[0], pt[1]) for pt in intersection.coords]
        elif intersection.geom_type == 'MultiPoint':
            return [(pt.x, pt.y) for pt in intersection.geoms]
        else:
            return []
    
    @staticmethod
    def compute_minimum_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Compute minimum distance between two trajectories.
        
        Args:
            traj1: (N, 2) array of trajectory 1 points
            traj2: (M, 2) array of trajectory 2 points
            
        Returns:
            Minimum distance between trajectories
        """
        line1 = TrajectoryIntersection.compute_trajectory_line(traj1)
        line2 = TrajectoryIntersection.compute_trajectory_line(traj2)
        
        return line1.distance(line2)
    
    @staticmethod
    def detect_conflict(traj1: np.ndarray, traj2: np.ndarray, 
                       distance_threshold: float = 50.0) -> bool:
        """
        Detect if trajectories represent a conflict (vehicles in close proximity).
        
        Args:
            traj1: (N, 2) array of trajectory 1 points
            traj2: (M, 2) array of trajectory 2 points
            distance_threshold: Maximum distance for conflict
            
        Returns:
            True if conflict detected
        """
        min_dist = TrajectoryIntersection.compute_minimum_distance(traj1, traj2)
        intersections = TrajectoryIntersection.detect_intersections(traj1, traj2)
        
        # Conflict if trajectories intersect or come very close
        return len(intersections) > 0 or min_dist < distance_threshold


class LaneBoundaryDetector:
    """
    Detect lane boundaries within roundabout zones.
    
    Uses trajectory clustering to identify marking patterns.
    """
    
    @staticmethod
    def detect_lane_boundaries(trajectories: List[np.ndarray], 
                              num_lanes: int = 2) -> List[LineString]:
        """
        Detect lane boundaries from collection of trajectories.
        
        Args:
            trajectories: List of (N, 2) trajectory arrays
            num_lanes: Expected number of lanes
            
        Returns:
            List of lane boundary LineStrings
        """
        from sklearn.cluster import KMeans
        
        # Collect all trajectory points
        all_points = np.vstack(trajectories)
        
        # Cluster points by lateral position
        if len(all_points) < num_lanes:
            return []
        
        # Use KMeans to find lane centers
        kmeans = KMeans(n_clusters=num_lanes, random_state=42)
        labels = kmeans.fit_predict(all_points[:, [0]])  # Cluster by x-coordinate
        
        # Create boundaries between clusters
        boundaries = []
        centers = sorted(kmeans.cluster_centers_.flatten())
        
        for i in range(len(centers) - 1):
            # Boundary is midpoint between adjacent cluster centers
            boundary_x = (centers[i] + centers[i + 1]) / 2
            
            # Create vertical line at this x-coordinate
            y_min = all_points[:, 1].min()
            y_max = all_points[:, 1].max()
            
            line = LineString([(boundary_x, y_min), (boundary_x, y_max)])
            boundaries.append(line)
        
        return boundaries
