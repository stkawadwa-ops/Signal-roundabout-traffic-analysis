"""
Vehicle and Pedestrian Detection using YOLOv8

Handles:
- Real-time detection of vehicles and pedestrians
- Bounding box extraction
- Confidence filtering
- Frame-by-frame processing
"""

import logging
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Standardized bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float
    
    @property
    def width(self) -> float:
        """Width in pixels"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Height in pixels"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Area in square pixels"""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinates"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'center': self.center,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
        }


@dataclass
class DetectionFrame:
    """Frame with all detections"""
    frame_id: int
    timestamp: float
    frame_shape: Tuple[int, int, int]  # (height, width, channels)
    detections: List[BoundingBox]
    raw_frame: Optional[np.ndarray] = None
    
    def get_vehicles(self) -> List[BoundingBox]:
        """Get vehicle detections only"""
        return [d for d in self.detections if d.class_name != 'person']
    
    def get_pedestrians(self) -> List[BoundingBox]:
        """Get pedestrian detections only"""
        return [d for d in self.detections if d.class_name == 'person']
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionFrame':
        """Filter detections by confidence threshold"""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionFrame(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            frame_shape=self.frame_shape,
            detections=filtered,
            raw_frame=self.raw_frame,
        )


class VehicleDetector:
    """
    YOLO-based vehicle and pedestrian detector.
    
    Features:
    - Loads pre-trained YOLOv8 models
    - Processes video frames
    - Returns standardized bounding boxes
    - Handles confidence thresholding
    """
    
    # YOLO class IDs
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        6: 'train',
        7: 'truck',
    }
    PERSON_CLASS = 0
    
    def __init__(
        self,
        model_name: str = 'yolov8x.pt',
        device: str = 'cuda',
        conf_threshold: float = 0.5,
    ):
        """
        Initialize detector.
        
        Args:
            model_name: YOLOv8 model size ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            device: 'cuda' or 'cpu'
            conf_threshold: Confidence threshold for detections
        """
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        
        logger.info(f"Loading YOLOv8 model: {model_name} on device: {device}")
        self.model = YOLO(model_name)
        self.model.to(device)
        logger.info("Model loaded successfully")
    
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> DetectionFrame:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            frame_id: Frame number/ID
            timestamp: Timestamp in seconds
            
        Returns:
            DetectionFrame with all detections
        """
        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
        )
        
        detections = []
        if len(results) > 0:
            result = results[0]
            
            # Extract boxes
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # Determine class name
                class_name = self.model.names[class_id]
                
                detection = BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )
                detections.append(detection)
        
        detection_frame = DetectionFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=frame.shape,
            detections=detections,
            raw_frame=frame.copy(),
        )
        
        return detection_frame
    
    def detect_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 1,
    ) -> List[DetectionFrame]:
        """
        Detect objects in video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            skip_frames: Process every Nth frame (default: all frames)
            
        Yields:
            DetectionFrame objects
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {fps}, Total frames: {total_frames}")
        
        frame_id = 0
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                # Calculate timestamp
                timestamp = frame_id / fps
                
                # Detect
                detection_frame = self.detect_frame(frame, frame_id, timestamp)
                
                logger.debug(
                    f"Frame {frame_id}: {len(detection_frame.detections)} detections"
                )
                
                yield detection_frame
                
                frame_count += 1
                frame_id += skip_frames
                
                # Check max frames
                if max_frames and frame_id >= max_frames:
                    break
        
        finally:
            cap.release()
            logger.info(f"Video processing complete. Processed {frame_id} frames")
    
    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[BoundingBox],
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame (will be copied)
            detections: List of BoundingBox objects
            show_confidence: Whether to show confidence scores
            
        Returns:
            Frame with drawn detections
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)
            
            # Draw box
            color = (0, 255, 0) if detection.class_name != 'person' else (255, 0, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = detection.class_name
            if show_confidence:
                label += f" ({detection.confidence:.2f})"
            
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        
        return vis_frame
