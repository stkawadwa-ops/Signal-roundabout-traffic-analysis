"""
Enhanced Traffic Flow Analysis with Multi-Exit Roundabout Support.

This module extends the existing framework to capture:
1. Multiple entry/exit points (3-5+ per roundabout)
2. Complex circulation patterns (1st, 2nd, 3rd exit)
3. Error recovery (re-circulation, u-turns)
4. Continuous speed measurement
5. Overpass occlusion tracking per vehicle
6. Complete journey logging per vehicle

Built on existing framework:
- Video stabilization ✓
- Detection & tracking ✓
- Geometry (enhanced) ✓
- Database (extended schema) ✓
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CirculationState(Enum):
    """Vehicle circulation state within roundabout."""
    NOT_IN_ROUNDABOUT = 0
    ENTERING = 1              # Approaching entry zone
    IN_ENTRY_ZONE = 2         # In entry lane
    IN_CIRCULATION = 3        # Actively circulating
    EXITING = 4               # Moving toward exit
    IN_EXIT_ZONE = 5          # In exit lane
    EXITED = 6                # Left roundabout
    RE_CIRCULATING = 7        # Missed exit, re-circulating
    IN_OVERPASS = 8           # Blocked by overpass


@dataclass
class EntryPoint:
    """Represents a roundabout entry point."""
    entry_id: int
    name: str                 # e.g., "North Entry", "South Entry"
    polygon: List[Tuple[float, float]]  # Entry zone boundary
    heading: float            # Direction of traffic entering (degrees)
    lane_count: int           # Number of lanes at entry


@dataclass
class ExitPoint:
    """Represents a roundabout exit point."""
    exit_id: int
    name: str                 # e.g., "1st Exit", "2nd Exit"
    polygon: List[Tuple[float, float]]  # Exit zone boundary
    bearing_from_center: float  # Angle from roundabout center (degrees)
    lane_count: int           # Number of lanes at exit


@dataclass
class OverpassSegment:
    """Represents an overpass that can occlude traffic."""
    overpass_id: int
    name: str                 # e.g., "Highway Bridge", "Rail Overpass"
    polygon: List[Tuple[float, float]]  # Overpass boundary
    affects_direction: str    # "clockwise", "counter-clockwise", "both"


@dataclass
class CirculationArc:
    """Records a single circulation arc within roundabout."""
    arc_id: int
    entry_point_id: int       # Which entry was used
    exit_point_id: int        # Which exit was taken
    entry_time_frame: int
    exit_time_frame: int
    duration_frames: int
    exit_order: int           # 1st, 2nd, 3rd exit encountered, etc.
    is_error_recovery: bool   # Was this a re-circulation (error)?
    mean_speed_kmh: float
    speed_samples: List[float] = field(default_factory=list)


@dataclass
class OverpassOcclusion:
    """Records vehicle occlusion by overpass."""
    occlusion_id: int
    overpass_id: int
    entry_frame: int
    exit_frame: int
    duration_frames: int
    confidence: float         # How confident is this occlusion?


@dataclass
class VehicleJourney:
    """Complete journey record for a single vehicle through roundabout."""
    track_id: int
    vehicle_class: str        # car, bus, truck, motorcycle, pedestrian
    vehicle_type: str         # sedan, SUV, articulated_bus, rigid_truck, etc.
    
    # Entry phase
    entry_point_id: int
    entry_time_frame: int
    entry_time_seconds: float
    entry_speed_kmh: float
    
    # Circulation phases (list of arcs)
    circulation_arcs: List[CirculationArc] = field(default_factory=list)
    
    # Exit phase
    exit_point_id: int        # Final exit used
    exit_time_frame: int
    exit_time_seconds: float
    exit_speed_kmh: float
    total_time_in_roundabout_seconds: float = 0.0
    
    # Overpass interactions
    overpass_occlusions: List[OverpassOcclusion] = field(default_factory=list)
    total_overpass_time_seconds: float = 0.0
    
    # Speed profile
    speed_samples: List[Tuple[int, float]] = field(default_factory=list)  # (frame, speed_kmh)
    speed_avg_kmh: float = 0.0
    speed_max_kmh: float = 0.0
    speed_min_kmh: float = 0.0
    speed_std_kmh: float = 0.0
    
    # Trajectory summary
    total_path_length_m: float = 0.0
    circulation_count: int = 0  # How many times circulated
    trajectory_points: List[Tuple[float, float]] = field(default_factory=list)
    
    def get_journey_summary(self) -> Dict:
        """Get human-readable journey summary."""
        return {
            'track_id': self.track_id,
            'vehicle': f"{self.vehicle_type} ({self.vehicle_class})",
            'entry': f"Entry {self.entry_point_id} @ {self.entry_time_seconds:.1f}s ({self.entry_speed_kmh:.1f} km/h)",
            'exit': f"Exit {self.exit_point_id} @ {self.exit_time_seconds:.1f}s ({self.exit_speed_kmh:.1f} km/h)",
            'time_in_roundabout': f"{self.total_time_in_roundabout_seconds:.1f}s",
            'circulation_count': self.circulation_count,
            're_circulations': sum(1 for arc in self.circulation_arcs if arc.is_error_recovery),
            'overpass_blocks': len(self.overpass_occlusions),
            'overpass_time': f"{self.total_overpass_time_seconds:.1f}s",
            'speed_profile': f"avg={self.speed_avg_kmh:.1f}, min={self.speed_min_kmh:.1f}, max={self.speed_max_kmh:.1f} km/h",
            'path_length': f"{self.total_path_length_m:.1f}m",
        }


class RoundaboutConfiguration:
    """Complete roundabout configuration with multiple entries/exits."""
    
    def __init__(self, roundabout_id: int, name: str, center: Tuple[float, float],
                 outer_radius: float, inner_radius: float):
        self.roundabout_id = roundabout_id
        self.name = name
        self.center = center
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        
        self.entry_points: Dict[int, EntryPoint] = {}
        self.exit_points: Dict[int, ExitPoint] = {}
        self.overpass_segments: Dict[int, OverpassSegment] = {}
    
    def add_entry_point(self, entry_point: EntryPoint):
        """Add entry point to roundabout."""
        self.entry_points[entry_point.entry_id] = entry_point
        logger.info(f"Added entry point: {entry_point.name} (ID: {entry_point.entry_id})")
    
    def add_exit_point(self, exit_point: ExitPoint):
        """Add exit point to roundabout."""
        self.exit_points[exit_point.exit_id] = exit_point
        logger.info(f"Added exit point: {exit_point.name} (ID: {exit_point.exit_id})")
    
    def add_overpass(self, overpass: OverpassSegment):
        """Add overpass segment."""
        self.overpass_segments[overpass.overpass_id] = overpass
        logger.info(f"Added overpass: {overpass.name} (ID: {overpass.overpass_id})")
    
    def get_entry_sequence(self) -> List[ExitPoint]:
        """Get exits in clockwise order for circulation analysis."""
        exits_sorted = sorted(
            self.exit_points.values(),
            key=lambda e: e.bearing_from_center
        )
        return exits_sorted


class CirculationAnalyzer:
    """Analyze vehicle circulation patterns through multi-exit roundabouts."""
    
    def __init__(self, config: RoundaboutConfiguration, frame_rate: float = 30.0):
        self.config = config
        self.frame_rate = frame_rate
        self.dt_seconds = 1.0 / frame_rate
    
    def detect_entry_point(self, trajectory: np.ndarray, start_frame: int) -> Optional[Tuple[int, float]]:
        """
        Detect which entry point vehicle used.
        
        Args:
            trajectory: (N, 2) array of points
            start_frame: Frame number of start
            
        Returns:
            (entry_id, confidence) or None
        """
        from shapely.geometry import Point, Polygon
        
        entry_point = Point(trajectory[0])
        
        best_entry_id = None
        best_distance = float('inf')
        
        for entry_id, entry in self.config.entry_points.items():
            poly = Polygon(entry.polygon)
            if entry_point.within(poly) or entry_point.touches(poly):
                return entry_id, 1.0  # High confidence if within entry zone
            
            # Distance-based fallback
            dist = entry_point.distance(poly)
            if dist < best_distance:
                best_distance = dist
                best_entry_id = entry_id
        
        if best_distance < 50:  # Within 50 pixels
            return best_entry_id, max(0.5, 1.0 - best_distance / 100)
        
        return None
    
    def detect_exit_point(self, point: Tuple[float, float]) -> Optional[Tuple[int, float]]:
        """
        Detect which exit point a vehicle is near.
        
        Args:
            point: (x, y) coordinate
            
        Returns:
            (exit_id, confidence) or None
        """
        from shapely.geometry import Point, Polygon
        
        query_point = Point(point)
        
        for exit_id, exit in self.config.exit_points.items():
            poly = Polygon(exit.polygon)
            if query_point.within(poly) or query_point.touches(poly):
                return exit_id, 1.0
            
            dist = query_point.distance(poly)
            if dist < 30:  # Close to exit
                confidence = 1.0 - (dist / 100)
                return exit_id, confidence
        
        return None
    
    def count_exits_encountered(self, trajectory: np.ndarray, 
                               start_frame: int, end_frame: int) -> int:
        """
        Count how many exit zones vehicle passed before exiting.
        
        Useful for determining "1st exit", "2nd exit", etc.
        
        Args:
            trajectory: Full trajectory
            start_frame: Frame when entered circulatory zone
            end_frame: Frame when exited roundabout
            
        Returns:
            Number of exit points encountered
        """
        exit_sequence = self.config.get_entry_sequence()
        first_exit_id = None
        exits_passed = 0
        
        for i, point in enumerate(trajectory):
            exit_info = self.detect_exit_point(tuple(point))
            if exit_info:
                exit_id, confidence = exit_info
                if confidence > 0.7:
                    if first_exit_id is None:
                        first_exit_id = exit_id
                    elif exit_id != first_exit_id:
                        exits_passed += 1
                        first_exit_id = exit_id
        
        return exits_passed
    
    def detect_circulation_arcs(self, trajectory: np.ndarray, 
                               speed_profile: np.ndarray,
                               entry_frame: int, exit_frame: int) -> List[CirculationArc]:
        """
        Detect individual circulation arcs through roundabout.
        
        A circulation arc starts when vehicle enters circulatory zone
        and ends when it approaches an exit.
        
        Args:
            trajectory: (N, 2) vehicle trajectory
            speed_profile: (N,) speed in km/h for each frame
            entry_frame: Frame number of entry
            exit_frame: Frame number of exit
            
        Returns:
            List of CirculationArc objects
        """
        arcs = []
        arc_count = 0
        
        # Find transitions to/from exit zones
        in_exit_zone = False
        arc_start_frame = entry_frame
        
        for i, point in enumerate(trajectory):
            frame_idx = entry_frame + i
            
            exit_info = self.detect_exit_point(tuple(point))
            is_near_exit = exit_info is not None and exit_info[1] > 0.7
            
            if is_near_exit and not in_exit_zone:
                # Entering exit zone
                in_exit_zone = True
            
            elif not is_near_exit and in_exit_zone:
                # Left exit zone without exiting - re-circulating
                in_exit_zone = False
                
                # Create arc for this circulation
                arc_frames = i
                speed_samples = speed_profile[max(0, i-30):i]
                
                arc = CirculationArc(
                    arc_id=arc_count,
                    entry_point_id=1,  # Placeholder
                    exit_point_id=1,   # Placeholder
                    entry_time_frame=arc_start_frame,
                    exit_time_frame=frame_idx,
                    duration_frames=arc_frames,
                    exit_order=arc_count + 1,
                    is_error_recovery=(arc_count > 0),  # 2nd+ arc = error
                    mean_speed_kmh=float(np.mean(speed_samples)) if len(speed_samples) > 0 else 0.0,
                    speed_samples=speed_samples.tolist()
                )
                
                arcs.append(arc)
                arc_count += 1
                arc_start_frame = frame_idx
            
            if frame_idx >= exit_frame:
                break
        
        # Final arc
        if len(trajectory) > 0:
            final_frames = len(trajectory) - max(0, arc_start_frame - entry_frame)
            speed_samples = speed_profile[max(0, len(trajectory)-30):]
            
            arc = CirculationArc(
                arc_id=arc_count,
                entry_point_id=1,
                exit_point_id=1,
                entry_time_frame=arc_start_frame,
                exit_time_frame=exit_frame,
                duration_frames=final_frames,
                exit_order=arc_count + 1,
                is_error_recovery=(arc_count > 0),
                mean_speed_kmh=float(np.mean(speed_samples)) if len(speed_samples) > 0 else 0.0,
                speed_samples=speed_samples.tolist()
            )
            arcs.append(arc)
        
        return arcs
    
    def detect_overpass_occlusions(self, trajectory: np.ndarray, 
                                   track_id: int,
                                   entry_frame: int, exit_frame: int) -> List[OverpassOcclusion]:
        """
        Detect when vehicle is occluded by overpass.
        
        Args:
            trajectory: Vehicle trajectory
            track_id: Vehicle track ID
            entry_frame: Frame when entered
            exit_frame: Frame when exited
            
        Returns:
            List of OverpassOcclusion events
        """
        from shapely.geometry import Point, Polygon
        
        occlusions = []
        occlusion_id = 0
        in_overpass = False
        occlusion_start_frame = None
        
        for i, point in enumerate(trajectory):
            frame_idx = entry_frame + i
            query_point = Point(point)
            
            # Check all overpass segments
            is_in_any_overpass = False
            for overpass in self.config.overpass_segments.values():
                poly = Polygon(overpass.polygon)
                if query_point.within(poly):
                    is_in_any_overpass = True
                    break
            
            if is_in_any_overpass and not in_overpass:
                # Entering overpass
                in_overpass = True
                occlusion_start_frame = frame_idx
            
            elif not is_in_any_overpass and in_overpass:
                # Exiting overpass
                in_overpass = False
                
                occlusion = OverpassOcclusion(
                    occlusion_id=occlusion_id,
                    overpass_id=0,  # Placeholder
                    entry_frame=occlusion_start_frame,
                    exit_frame=frame_idx,
                    duration_frames=frame_idx - occlusion_start_frame,
                    confidence=0.9
                )
                occlusions.append(occlusion)
                occlusion_id += 1
        
        return occlusions
    
    def build_vehicle_journey(self, track_id: int, trajectory: np.ndarray,
                             speed_profile: np.ndarray, class_info: Dict,
                             entry_frame: int, exit_frame: int) -> VehicleJourney:
        """
        Construct complete vehicle journey record.
        
        Args:
            track_id: Vehicle track ID
            trajectory: (N, 2) trajectory points
            speed_profile: (N,) speed in km/h
            class_info: {'class': 'car', 'type': 'sedan'}
            entry_frame: Entry frame number
            exit_frame: Exit frame number
            
        Returns:
            VehicleJourney object with all details
        """
        frame_rate = self.frame_rate
        dt_seconds = 1.0 / frame_rate
        
        # Detect entry/exit points
        entry_info = self.detect_entry_point(trajectory, entry_frame)
        entry_point_id = entry_info[0] if entry_info else 0
        
        exit_info = self.detect_exit_point(tuple(trajectory[-1]))
        exit_point_id = exit_info[0] if exit_info else 0
        
        # Calculate journey timing
        entry_time_seconds = entry_frame * dt_seconds
        exit_time_seconds = exit_frame * dt_seconds
        total_time_seconds = exit_time_seconds - entry_time_seconds
        
        # Build journey
        journey = VehicleJourney(
            track_id=track_id,
            vehicle_class=class_info.get('class', 'unknown'),
            vehicle_type=class_info.get('type', 'unknown'),
            entry_point_id=entry_point_id,
            entry_time_frame=entry_frame,
            entry_time_seconds=entry_time_seconds,
            entry_speed_kmh=float(speed_profile[0]) if len(speed_profile) > 0 else 0.0,
            exit_point_id=exit_point_id,
            exit_time_frame=exit_frame,
            exit_time_seconds=exit_time_seconds,
            exit_speed_kmh=float(speed_profile[-1]) if len(speed_profile) > 0 else 0.0,
            total_time_in_roundabout_seconds=total_time_seconds,
            trajectory_points=[tuple(p) for p in trajectory],
            speed_samples=[(entry_frame + i, float(sp)) for i, sp in enumerate(speed_profile)],
            speed_avg_kmh=float(np.mean(speed_profile)) if len(speed_profile) > 0 else 0.0,
            speed_max_kmh=float(np.max(speed_profile)) if len(speed_profile) > 0 else 0.0,
            speed_min_kmh=float(np.min(speed_profile)) if len(speed_profile) > 0 else 0.0,
            speed_std_kmh=float(np.std(speed_profile)) if len(speed_profile) > 1 else 0.0,
            total_path_length_m=float(self._compute_path_length(trajectory)),
        )
        
        # Detect circulation patterns
        journey.circulation_arcs = self.detect_circulation_arcs(
            trajectory, speed_profile, entry_frame, exit_frame
        )
        journey.circulation_count = len(journey.circulation_arcs)
        
        # Detect overpass occlusions
        journey.overpass_occlusions = self.detect_overpass_occlusions(
            trajectory, track_id, entry_frame, exit_frame
        )
        journey.total_overpass_time_seconds = sum(
            oc.duration_frames * dt_seconds for oc in journey.overpass_occlusions
        )
        
        return journey
    
    @staticmethod
    def _compute_path_length(trajectory: np.ndarray, pixel_to_meter: float = 0.05) -> float:
        """Compute total path length in meters."""
        if len(trajectory) < 2:
            return 0.0
        
        diffs = np.diff(trajectory, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        total_pixels = np.sum(distances)
        
        return total_pixels * pixel_to_meter


class JourneyDatabase:
    """Store and query vehicle journeys."""
    
    def __init__(self):
        self.journeys: Dict[int, VehicleJourney] = {}
    
    def add_journey(self, journey: VehicleJourney):
        """Add journey to database."""
        self.journeys[journey.track_id] = journey
    
    def get_journeys_by_entry(self, entry_id: int) -> List[VehicleJourney]:
        """Get all journeys entering at specific point."""
        return [j for j in self.journeys.values() if j.entry_point_id == entry_id]
    
    def get_journeys_by_exit(self, exit_id: int) -> List[VehicleJourney]:
        """Get all journeys exiting at specific point."""
        return [j for j in self.journeys.values() if j.exit_point_id == exit_id]
    
    def get_journeys_by_class(self, vehicle_class: str) -> List[VehicleJourney]:
        """Get all journeys for specific vehicle class."""
        return [j for j in self.journeys.values() if j.vehicle_class == vehicle_class]
    
    def get_re_circulating_journeys(self) -> List[VehicleJourney]:
        """Get journeys with re-circulation (error recovery)."""
        return [
            j for j in self.journeys.values()
            if any(arc.is_error_recovery for arc in j.circulation_arcs)
        ]
    
    def get_overpass_affected_journeys(self) -> List[VehicleJourney]:
        """Get journeys affected by overpass occlusion."""
        return [j for j in self.journeys.values() if len(j.overpass_occlusions) > 0]
    
    def compute_statistics(self) -> Dict:
        """Compute aggregate statistics."""
        if not self.journeys:
            return {}
        
        journeys = list(self.journeys.values())
        
        stats = {
            'total_vehicles': len(journeys),
            'avg_time_in_roundabout_s': np.mean([j.total_time_in_roundabout_seconds for j in journeys]),
            'avg_speed_kmh': np.mean([j.speed_avg_kmh for j in journeys]),
            'vehicles_with_recirculation': len(self.get_re_circulating_journeys()),
            'recirculation_rate': len(self.get_re_circulating_journeys()) / len(journeys) * 100,
            'vehicles_affected_by_overpass': len(self.get_overpass_affected_journeys()),
            'avg_overpass_delay_s': np.mean([
                j.total_overpass_time_seconds for j in self.get_overpass_affected_journeys()
            ]) if self.get_overpass_affected_journeys() else 0.0,
        }
        
        return stats
    
    def print_journey_report(self, track_id: int):
        """Print detailed journey report for single vehicle."""
        if track_id not in self.journeys:
            logger.warning(f"Journey {track_id} not found")
            return
        
        journey = self.journeys[track_id]
        summary = journey.get_journey_summary()
        
        print(f"\n{'='*60}")
        print(f"VEHICLE JOURNEY REPORT - Track ID: {track_id}")
        print(f"{'='*60}")
        for key, value in summary.items():
            print(f"  {key:.<40} {value}")
        print(f"{'='*60}\n")
