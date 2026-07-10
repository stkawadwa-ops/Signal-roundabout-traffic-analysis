"""
Optimized Video Stabilizer using OpenCV's GoodFeaturesToTrack and Optical Flow.

This module stabilizes UAV drone footage by:
1. Detecting stable features (GoodFeaturesToTrack)
2. Tracking features across frames (calcOpticalFlowPyrLK)
3. Computing homography matrices for perspective correction
4. Applying frame alignment and stabilization
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StabilizationConfig:
    """Configuration for video stabilization."""
    max_features: int = 200  # GoodFeaturesToTrack parameter
    feature_quality: float = 0.01
    min_distance: float = 30.0
    block_size: int = 3
    
    # Optical flow parameters
    win_size: Tuple[int, int] = (15, 15)
    max_level: int = 2
    criteria: Tuple = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    
    # Homography parameters
    ransac_threshold: float = 5.0
    ransac_method: int = cv2.RANSAC


class VideoStabilizer:
    """
    Stabilizes drone footage using feature tracking and homography estimation.
    
    Pipeline:
    1. Detect stable features in current frame (GoodFeaturesToTrack)
    2. Track features to next frame (calcOpticalFlowPyrLK)
    3. Estimate homography matrix (perspective transformation)
    4. Warp frames to stabilized coordinate system
    """
    
    def __init__(self, config: Optional[StabilizationConfig] = None):
        self.config = config or StabilizationConfig()
        self.prev_frame = None
        self.prev_features = None
        self.accumulated_homography = np.eye(3)
        
    def detect_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect good features to track using Harris corner detection.
        
        Args:
            frame: Input frame (grayscale or color)
            
        Returns:
            Array of feature coordinates (N, 1, 2)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # GoodFeaturesToTrack: Detect Harris corners
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.feature_quality,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size
        )
        
        return features
    
    def track_features(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track features across frames using Lukas-Kanade optical flow.
        
        Args:
            prev_frame: Previous frame (grayscale)
            curr_frame: Current frame (grayscale)
            prev_features: Features from previous frame
            
        Returns:
            Tuple of (next_features, status, error)
        """
        if prev_features is None or len(prev_features) == 0:
            return None, None, None
        
        # calcOpticalFlowPyrLK: Pyramid Lucas-Kanade tracker
        next_features, status, error = cv2.calcOpticalFlowPyrLK(
            prev_frame,
            curr_frame,
            prev_features,
            None,
            winSize=self.config.win_size,
            maxLevel=self.config.max_level,
            criteria=self.config.criteria
        )
        
        return next_features, status, error
    
    def estimate_homography(
        self,
        prev_features: np.ndarray,
        curr_features: np.ndarray,
        status: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate homography matrix from feature correspondences.
        
        Uses RANSAC to find robust perspective transformation despite outliers.
        
        Args:
            prev_features: Features in previous frame
            curr_features: Features in current frame
            status: Tracking status (1 = good, 0 = lost)
            
        Returns:
            3x3 homography matrix or None if not enough matches
        """
        if status is None or len(status) < 4:
            return None
        
        # Filter good matches
        good_prev = prev_features[status.flatten() == 1]
        good_curr = curr_features[status.flatten() == 1]
        
        if len(good_prev) < 4:
            logger.warning(f"Insufficient good features for homography: {len(good_prev)}")
            return None
        
        # Compute homography with RANSAC
        try:
            H, _ = cv2.findHomography(
                good_prev,
                good_curr,
                method=self.config.ransac_method,
                ransacReprojThreshold=self.config.ransac_threshold
            )
            return H
        except Exception as e:
            logger.error(f"Homography estimation failed: {e}")
            return None
    
    def stabilize_frame(
        self,
        frame: np.ndarray,
        homography: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Warp frame using homography matrix for stabilization.
        
        Args:
            frame: Input frame to stabilize
            homography: 3x3 homography matrix
            output_size: Output frame size (height, width) or None to preserve
            
        Returns:
            Stabilized frame
        """
        if homography is None:
            return frame
        
        h, w = frame.shape[:2]
        if output_size is None:
            output_size = (h, w)
        
        # Apply perspective transformation
        stabilized = cv2.warpPerspective(
            frame,
            homography,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return stabilized
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Main stabilization pipeline: detect, track, estimate, warp.
        
        Args:
            frame: Current frame (BGR or grayscale)
            
        Returns:
            Tuple of (stabilized_frame, homography_matrix)
        """
        h, w = frame.shape[:2]
        
        # Convert to grayscale for processing
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_features = self.detect_features(gray)
            return frame, np.eye(3)
        
        # Step 1: Track features using optical flow
        next_features, status, error = self.track_features(
            self.prev_frame,
            gray,
            self.prev_features
        )
        
        if next_features is None or status is None:
            logger.warning("Feature tracking failed, returning unmodified frame")
            self.prev_frame = gray
            self.prev_features = self.detect_features(gray)
            return frame, np.eye(3)
        
        # Step 2: Estimate homography
        H = self.estimate_homography(self.prev_features, next_features, status)
        
        if H is None:
            H = np.eye(3)
        
        # Step 3: Accumulate homography for global stabilization
        self.accumulated_homography = self.accumulated_homography @ H
        
        # Step 4: Warp frame to stabilized coordinates
        stabilized = self.stabilize_frame(frame, self.accumulated_homography)
        
        # Update for next iteration
        self.prev_frame = gray
        
        # Re-detect features if too few remaining
        if np.sum(status) < self.config.max_features // 2:
            self.prev_features = self.detect_features(gray)
        else:
            # Track only good features
            self.prev_features = next_features[status.flatten() == 1]
        
        return stabilized, H
    
    def reset(self):
        """Reset stabilizer state for new video segment."""
        self.prev_frame = None
        self.prev_features = None
        self.accumulated_homography = np.eye(3)


class OpticalFlowAnalyzer:
    """
    Analyze optical flow for motion estimation and compensation.
    
    Useful for detecting motion patterns, estimating drone drift, etc.
    """
    
    def __init__(self, win_size: Tuple[int, int] = (15, 15)):
        self.win_size = win_size
    
    def compute_dense_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> np.ndarray:
        """
        Compute dense optical flow (Farneback method).
        
        Args:
            prev_frame: Previous frame (grayscale)
            curr_frame: Current frame (grayscale)
            
        Returns:
            Flow field as (height, width, 2) array
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame,
            curr_frame,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            n8=True,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )
        return flow
    
    def estimate_global_motion(self, flow: np.ndarray) -> Tuple[float, float]:
        """
        Estimate global motion (drone drift) from optical flow.
        
        Args:
            flow: Dense optical flow field
            
        Returns:
            Tuple of (dx, dy) representing global motion in pixels
        """
        # Use median flow as robust estimate of global motion
        dx_median = np.median(flow[:, :, 0])
        dy_median = np.median(flow[:, :, 1])
        
        return dx_median, dy_median
