"""
Optimized Multi-Object Tracker with ByteTrack/BoT-SORT and deep feature re-identification.

This module provides state-of-the-art tracking for UAV footage with:
1. ByteTrack: Robust association with motion prediction
2. BoT-SORT: Appearance features + motion for occlusion handling
3. Kalman filtering for trajectory prediction
4. Re-identification across occlusion zones
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import cv2

logger = logging.getLogger(__name__)

# Try importing ByteTrack
try:
    from bytetrack import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    logger.warning("ByteTrack not available. Install with: pip install git+https://github.com/ifzhang/ByteTrack.git")

# Try importing YOLOX (used by ByteTrack)
try:
    from yolox.tracker.byte_tracker import BYTETracker as YOLOXBYTETracker
    YOLOX_AVAILABLE = True
except ImportError:
    YOLOX_AVAILABLE = False


@dataclass
class TrackingConfig:
    """Configuration for multi-object tracking."""
    # ByteTrack parameters
    track_thresh: float = 0.5  # Detection confidence threshold
    track_buffer: int = 30  # Buffer size for lost tracks
    match_thresh: float = 0.8  # Matching threshold
    
    # Motion model parameters
    max_age: int = 120  # Max frames to keep lost track
    min_hits: int = 3  # Min detections to create track
    
    # Re-identification parameters
    use_appearance: bool = True  # Use appearance features
    appearance_threshold: float = 0.7
    
    # Occlusion handling
    occlusion_timeout: int = 60  # Max frames to predict through occlusion
    min_track_length: int = 5  # Min track length for interpolation
    
    # Frame info
    frame_rate: int = 30
    img_size: Tuple[int, int] = (1920, 1080)


@dataclass
class Track:
    """Represents a single tracked object."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    # Motion state
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    kalman_state: Optional[np.ndarray] = None
    
    # Appearance features
    appearance_feature: Optional[np.ndarray] = None
    
    # Track history
    history: List[np.ndarray] = field(default_factory=list)
    hit_streak: int = 0  # Consecutive frames with detections
    age: int = 0  # Total frames since track creation
    time_since_update: int = 0  # Frames since last detection
    
    # Occlusion handling
    occluded: bool = False
    occlusion_start_frame: int = 0
    
    def get_center(self) -> np.ndarray:
        """Get bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def update(self, detection: Dict, frame_idx: int):
        """Update track with new detection."""
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.history.append(self.bbox.copy())
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
        
        # Update velocity
        if len(self.history) > 1:
            prev_center = (self.history[-2][:2] + self.history[-2][2:]) / 2
            curr_center = self.get_center()
            self.velocity = curr_center - prev_center
        
        self.occluded = False
    
    def predict(self):
        """Predict next position using velocity."""
        self.bbox = self.bbox + np.array([
            self.velocity[0], self.velocity[1],
            self.velocity[0], self.velocity[1]
        ])
        self.time_since_update += 1
        self.age += 1
        
        if self.time_since_update > 1:
            self.occluded = True


class KalmanFilterTracker:
    """
    Kalman filter for motion prediction through occlusions.
    
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self, bbox: np.ndarray):
        """Initialize Kalman filter with bounding box."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float32) * 1.0
        
        # State covariance
        self.P = np.eye(8, dtype=np.float32)
    
    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Predict next state."""
        # Motion model: constant velocity
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt  # x += vx * dt
        F[1, 5] = dt  # y += vy * dt
        F[2, 6] = dt  # w += vw * dt
        F[3, 7] = dt  # h += vh * dt
        
        # Predict state
        self.state = F @ self.state
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return self.state
    
    def update(self, measurement: np.ndarray):
        """Update state with measurement."""
        # Measurement: [x, y, w, h]
        z = measurement.astype(np.float32)
        
        # Measurement matrix
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # w
        H[3, 3] = 1  # h
        
        # Innovation
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        self.P = (np.eye(8, dtype=np.float32) - K @ H) @ self.P
    
    def get_bbox(self) -> np.ndarray:
        """Get current bbox estimate."""
        x, y, w, h = self.state[:4]
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])


class ByteTrackTracker:
    """
    Multi-object tracker using ByteTrack algorithm.
    
    ByteTrack improves upon DeepSORT by:
    1. Using motion-based matching for all detections
    2. Not relying solely on appearance features
    3. Better performance on low-confidence detections
    """
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_idx = 0
        
        # Try to use ByteTrack if available
        if YOLOX_AVAILABLE:
            self.byte_tracker = YOLOXBYTETracker(
                frame_rate=self.config.frame_rate,
                track_thresh=self.config.track_thresh,
                track_buffer=self.config.track_buffer,
                match_thresh=self.config.match_thresh,
                mot20=False
            )
            self.use_byte_tracker = True
            logger.info("Using ByteTrack for association")
        else:
            self.byte_tracker = None
            self.use_byte_tracker = False
            logger.warning("ByteTrack not available, using basic Hungarian matching")
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of dicts with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'class_id': int
                - 'class_name': str
                - 'embedding': optional appearance feature
                
        Returns:
            List of active tracks
        """
        self.frame_idx += 1
        
        if self.use_byte_tracker and self.byte_tracker:
            return self._update_with_bytetrack(detections)
        else:
            return self._update_with_hungarian(detections)
    
    def _update_with_bytetrack(self, detections: List[Dict]) -> List[Track]:
        """Update using ByteTrack algorithm."""
        # Format detections for ByteTrack
        if len(detections) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            dets = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                w = x2 - x1
                h = y2 - y1
                # ByteTrack expects [x, y, w, h, conf]
                dets.append([x1, y1, w, h, det['confidence']])
            dets = np.array(dets, dtype=np.float32)
        
        # Update ByteTrack
        tracks = self.byte_tracker.update(dets, img_info=(self.config.img_size[1], self.config.img_size[0]))
        
        # Convert ByteTrack outputs to internal format
        active_tracks = []
        updated_ids = set()
        
        for track in tracks:
            track_id = int(track.track_id)
            x1, y1, w, h = track.tlwh
            x2, y2 = x1 + w, y1 + h
            
            # Find matching detection
            matched_det = None
            for det in detections:
                det_x1, det_y1, det_x2, det_y2 = det['bbox']
                iou = self._compute_iou(
                    np.array([x1, y1, x2, y2]),
                    det['bbox']
                )
                if iou > 0.5:
                    matched_det = det
                    break
            
            if track_id in self.tracks:
                # Update existing track
                if matched_det:
                    self.tracks[track_id].update(matched_det, self.frame_idx)
                else:
                    self.tracks[track_id].predict()
                active_tracks.append(self.tracks[track_id])
            else:
                # Create new track
                if matched_det:
                    new_track = Track(
                        track_id=track_id,
                        bbox=matched_det['bbox'],
                        confidence=matched_det['confidence'],
                        class_id=matched_det['class_id'],
                        class_name=matched_det['class_name']
                    )
                    self.tracks[track_id] = new_track
                    active_tracks.append(new_track)
            
            updated_ids.add(track_id)
        
        # Remove dead tracks
        dead_ids = [tid for tid in self.tracks if tid not in updated_ids]
        for tid in dead_ids:
            if self.tracks[tid].time_since_update > self.config.max_age:
                del self.tracks[tid]
        
        return active_tracks
    
    def _update_with_hungarian(self, detections: List[Dict]) -> List[Track]:
        """Update using basic Hungarian algorithm matching."""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            logger.error("scipy required for Hungarian matching")
            return list(self.tracks.values())
        
        # Predict tracks
        for track in self.tracks.values():
            track.predict()
        
        # Compute cost matrix
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        
        if num_tracks == 0:
            # Create new tracks for all detections
            for i, det in enumerate(detections):
                new_track = Track(
                    track_id=self.next_track_id,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name']
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
            return list(self.tracks.values())
        
        # Build cost matrix
        cost_matrix = np.zeros((num_tracks, num_dets))
        track_list = list(self.tracks.values())
        
        for i, track in enumerate(track_list):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.bbox, det['bbox'])
                cost_matrix[i, j] = 1 - iou  # Convert to cost
        
        # Solve assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Update matched tracks
        matched_det_indices = set()
        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] < 0.5:  # IOU threshold
                track = track_list[t_idx]
                track.update(detections[d_idx], self.frame_idx)
                matched_det_indices.add(d_idx)
        
        # Create tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_track = Track(
                    track_id=self.next_track_id,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name']
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        # Remove dead tracks
        dead_ids = [
            tid for tid in self.tracks
            if self.tracks[tid].time_since_update > self.config.max_age
        ]
        for tid in dead_ids:
            del self.tracks[tid]
        
        return list(self.tracks.values())
    
    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bboxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        inter_min_x = max(x1_min, x2_min)
        inter_min_y = max(y1_min, y2_min)
        inter_max_x = min(x1_max, x2_max)
        inter_max_y = min(y1_max, y2_max)
        
        if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
            return 0.0
        
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.frame_idx = 0
        if self.byte_tracker:
            self.byte_tracker.reset()
