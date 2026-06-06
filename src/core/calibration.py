"""
Camera Calibration for Pixel-to-Meter Conversion

Handles:
- Calibration from known lane widths
- Calibration from roundabout radius
- Validation of calibration accuracy
- Perspective correction
"""

import logging
from typing import Tuple, Optional, List, Dict
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Calibration method types"""
    LANE_WIDTH = "lane_width"  # Standard road lane = 3.5m
    ROUNDABOUT_RADIUS = "roundabout_radius"  # Known roundabout radius
    MANUAL_POINTS = "manual_points"  # Manually marked known distance


@dataclass
class CalibrationResult:
    """Result of calibration"""
    method: CalibrationMethod
    ratio: float  # meters per pixel
    confidence: float  # 0.0-1.0, quality of calibration
    reference_distance_m: float  # Distance used for calibration
    measured_pixels: float  # Pixels measured for reference
    notes: str
    
    def __str__(self) -> str:
        return (
            f"Calibration ({self.method.value}):\n"
            f"  Ratio: {self.ratio:.6f} m/pixel\n"
            f"  Confidence: {self.confidence:.2%}\n"
            f"  Reference: {self.reference_distance_m}m = {self.measured_pixels:.1f} pixels"
        )


class CalibrationManager:
    """
    Manages camera calibration for pixel-to-meter conversion.
    
    Critical for accurate dimension extraction in top-down UAV view.
    """
    
    # Standard dimensions
    STANDARD_LANE_WIDTH_M = 3.5  # Global standard
    
    def __init__(self):
        self.calibration: Optional[CalibrationResult] = None
        self.altitude_m: Optional[float] = None
    
    def calibrate_from_lane_width(
        self,
        frame: np.ndarray,
        lane_width_m: float = STANDARD_LANE_WIDTH_M,
    ) -> CalibrationResult:
        """
        Calibrate using standard road lane width.
        
        Procedure:
        1. User marks start and end of lane marking
        2. System measures pixel distance
        3. Calculates ratio
        
        Args:
            frame: Frame with visible lane marking
            lane_width_m: Lane width in meters (default: 3.5m)
            
        Returns:
            CalibrationResult
        """
        logger.info("Starting lane width calibration...")
        logger.info("Click on two points marking the lane width boundaries")
        logger.info("Then press ENTER")
        
        points = self._interactive_point_selection(frame, num_points=2)
        
        if len(points) != 2:
            raise ValueError("Exactly 2 points required for lane width calibration")
        
        p1, p2 = points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        ratio = lane_width_m / pixel_distance
        
        calibration = CalibrationResult(
            method=CalibrationMethod.LANE_WIDTH,
            ratio=ratio,
            confidence=0.9,  # High confidence for standard lane
            reference_distance_m=lane_width_m,
            measured_pixels=pixel_distance,
            notes=f"Calibrated from {lane_width_m}m lane marking",
        )
        
        self.calibration = calibration
        logger.info(f"Calibration complete:\n{calibration}")
        
        return calibration
    
    def calibrate_from_roundabout_radius(
        self,
        frame: np.ndarray,
        roundabout_radius_m: float,
    ) -> CalibrationResult:
        """
        Calibrate using known roundabout radius.
        
        Args:
            frame: Frame with roundabout visible
            roundabout_radius_m: Roundabout radius in meters
            
        Returns:
            CalibrationResult
        """
        logger.info("Starting roundabout radius calibration...")
        logger.info(f"Roundabout radius: {roundabout_radius_m}m")
        logger.info("Click on two points marking the roundabout radius")
        logger.info("Then press ENTER")
        
        points = self._interactive_point_selection(frame, num_points=2)
        
        if len(points) != 2:
            raise ValueError("Exactly 2 points required for radius calibration")
        
        p1, p2 = points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        ratio = roundabout_radius_m / pixel_distance
        
        calibration = CalibrationResult(
            method=CalibrationMethod.ROUNDABOUT_RADIUS,
            ratio=ratio,
            confidence=0.85,
            reference_distance_m=roundabout_radius_m,
            measured_pixels=pixel_distance,
            notes=f"Calibrated from {roundabout_radius_m}m roundabout radius",
        )
        
        self.calibration = calibration
        logger.info(f"Calibration complete:\n{calibration}")
        
        return calibration
    
    def calibrate_manual(
        self,
        known_distance_m: float,
        pixel_distance: float,
        confidence: float = 0.85,
    ) -> CalibrationResult:
        """
        Programmatic calibration with known distance and pixel count.
        
        Args:
            known_distance_m: Known distance in meters
            pixel_distance: Corresponding distance in pixels
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            CalibrationResult
        """
        ratio = known_distance_m / pixel_distance
        
        calibration = CalibrationResult(
            method=CalibrationMethod.MANUAL_POINTS,
            ratio=ratio,
            confidence=confidence,
            reference_distance_m=known_distance_m,
            measured_pixels=pixel_distance,
            notes=f"Manual calibration: {known_distance_m}m = {pixel_distance:.1f}px",
        )
        
        self.calibration = calibration
        logger.info(f"Calibration set:\n{calibration}")
        
        return calibration
    
    def validate_calibration(
        self,
        frame: np.ndarray,
        known_features: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate calibration by checking against known features.
        
        Args:
            frame: Frame for validation
            known_features: Dict of {feature_name: distance_in_meters}
                           Example: {"vehicle_length": 4.5, "lane_width": 3.5}
            
        Returns:
            (is_valid, error_report)
        """
        if self.calibration is None:
            logger.error("No calibration set")
            return False, {}
        
        logger.info("Validating calibration against known features...")
        logger.info(f"Known features: {known_features}")
        
        errors = {}
        
        for feature_name, actual_distance_m in known_features.items():
            logger.info(f"\nValidating {feature_name}: {actual_distance_m}m")
            logger.info("Click on two points marking this feature")
            logger.info("Then press ENTER")
            
            points = self._interactive_point_selection(frame, num_points=2)
            
            if len(points) < 2:
                logger.warning(f"Skipped {feature_name}")
                continue
            
            p1, p2 = points
            pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            extracted_distance_m = pixel_distance * self.calibration.ratio
            
            error_percent = abs(extracted_distance_m - actual_distance_m) / actual_distance_m * 100
            errors[feature_name] = {
                'actual_m': actual_distance_m,
                'extracted_m': extracted_distance_m,
                'error_percent': error_percent,
                'pixels': pixel_distance,
            }
            
            status = "✓" if error_percent < 10 else "✗"
            logger.info(
                f"{status} {feature_name}: {actual_distance_m:.2f}m actual, "
                f"{extracted_distance_m:.2f}m extracted (error: {error_percent:.1f}%)"
            )
        
        # Overall validation
        avg_error = np.mean([e['error_percent'] for e in errors.values()])
        is_valid = avg_error < 10.0
        
        logger.info(f"\nAverage error: {avg_error:.1f}%")
        logger.info(f"Calibration {'VALID ✓' if is_valid else 'NEEDS ADJUSTMENT ✗'}")
        
        return is_valid, errors
    
    def pixels_to_meters(
        self,
        pixel_distance: float,
    ) -> float:
        """
        Convert pixel distance to meters.
        
        Args:
            pixel_distance: Distance in pixels
            
        Returns:
            Distance in meters
        """
        if self.calibration is None:
            raise RuntimeError("No calibration set. Call calibrate_* first.")
        
        return pixel_distance * self.calibration.ratio
    
    def meters_to_pixels(
        self,
        meter_distance: float,
    ) -> float:
        """
        Convert meter distance to pixels.
        
        Args:
            meter_distance: Distance in meters
            
        Returns:
            Distance in pixels
        """
        if self.calibration is None:
            raise RuntimeError("No calibration set. Call calibrate_* first.")
        
        return meter_distance / self.calibration.ratio
    
    def get_ratio(self) -> float:
        """Get current ratio (m/pixel)"""
        if self.calibration is None:
            raise RuntimeError("No calibration set")
        return self.calibration.ratio
    
    def _interactive_point_selection(
        self,
        frame: np.ndarray,
        num_points: int,
    ) -> List[Tuple[int, int]]:
        """
        Interactive point selection on frame.
        
        Args:
            frame: Frame to select points on
            num_points: Number of points to select
            
        Returns:
            List of (x, y) coordinates
        """
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                logger.debug(f"Point selected: ({x}, {y})")
        
        window_name = "Select Points (Press ENTER when done)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        display_frame = frame.copy()
        
        while len(points) < num_points:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(0)
            
            if key == 13:  # ENTER
                break
            
            # Redraw points
            display_frame = frame.copy()
            for i, (x, y) in enumerate(points):
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    str(i + 1),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            
            # Draw line if 2+ points
            if len(points) >= 2:
                cv2.line(display_frame, points[0], points[-1], (0, 255, 0), 2)
        
        cv2.destroyWindow(window_name)
        
        return points
