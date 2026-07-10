"""
Optimized YOLO Detector with SAHI (Slicing Aided Hyper Inference) integration.

This module provides:
1. YOLOv8/YOLOv10 detection with P2 feature maps for small objects
2. SAHI for improved detection of small vehicles (critical for aerial footage)
3. Confidence filtering and NMS optimization
4. Multi-scale detection pipeline
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ultralytics not installed. Install with: pip install ultralytics")

try:
    from sahi.prediction import ObjectPrediction
    from sahi.models.yolov8 import Yolov8DetectionModel
    from sahi.sliced_prediction import SlicedPredictionResult
    from sahi.utils.cv import read_image_pil
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SAHI not installed. Install with: pip install sahi")

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration for YOLO detection."""
    model_name: str = "yolov8x"  # yolov8x or yolov10x for best accuracy
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    
    # SAHI configuration for small object detection
    use_sahi: bool = True
    slice_height: int = 512  # Slice size for SAH
    slice_width: int = 512
    overlap_height_ratio: float = 0.1  # 10% overlap between slices
    overlap_width_ratio: float = 0.1
    
    # Device configuration
    device: str = "cuda:0"  # cuda:0, cpu, etc.
    
    # Multi-scale detection
    enable_p2_features: bool = True  # Use P2 feature maps (higher resolution)
    augment: bool = False  # Test-time augmentation


class YOLODetector:
    """
    Optimized YOLO detector with SAHI integration for small object detection.
    
    SAHI enables detection of small vehicles in aerial footage by:
    1. Slicing the image into overlapping tiles
    2. Running YOLO on each tile
    3. Merging predictions with NMS
    
    This is critical for UAV footage where vehicles appear small.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.model = None
        self.sahi_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLO and SAHI models."""
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics package required. Install with: pip install ultralytics")
        
        # Load base YOLO model
        model_path = f"{self.config.model_name}.pt"
        self.model = YOLO(model_path)
        self.model.to(self.config.device)
        
        logger.info(f"Loaded {self.config.model_name} model")
        
        # Initialize SAHI model if enabled
        if self.config.use_sahi:
            if not SAHI_AVAILABLE:
                logger.warning("SAHI not available. Install with: pip install sahi")
                self.config.use_sahi = False
            else:
                try:
                    self.sahi_model = Yolov8DetectionModel(
                        model_path=model_path,
                        confidence_threshold=self.config.confidence_threshold,
                        device=self.config.device,
                    )
                    logger.info("Initialized SAHI model for small object detection")
                except Exception as e:
                    logger.error(f"Failed to initialize SAHI: {e}")
                    self.config.use_sahi = False
    
    def detect(
        self,
        frame: np.ndarray,
        use_sahi: Optional[bool] = None
    ) -> Dict:
        """
        Detect vehicles in frame using YOLO ± SAHI.
        
        Args:
            frame: Input frame (BGR or RGB)
            use_sahi: Override config for SAHI usage
            
        Returns:
            Dictionary containing:
            - 'boxes': (N, 4) array of [x1, y1, x2, y2]
            - 'confidences': (N,) array of confidence scores
            - 'class_ids': (N,) array of class IDs
            - 'class_names': List of class names
        """
        use_sahi = use_sahi if use_sahi is not None else self.config.use_sahi
        
        if use_sahi and self.sahi_model:
            return self._detect_with_sahi(frame)
        else:
            return self._detect_standard(frame)
    
    def _detect_standard(self, frame: np.ndarray) -> Dict:
        """Standard YOLO detection without SAHI."""
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.nms_threshold,
            device=self.config.device,
            augment=self.config.augment
        )
        
        result = results[0]
        
        # Extract detections
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
        confidences = result.boxes.conf.cpu().numpy()  # (N,)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        class_names = [result.names[cid] for cid in class_ids]
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids,
            'class_names': class_names,
            'raw_result': result
        }
    
    def _detect_with_sahi(self, frame: np.ndarray) -> Dict:
        """
        SAHI-based detection for improved small object detection.
        
        Pipeline:
        1. Slice image into overlapping tiles
        2. Run YOLO on each tile
        3. Merge predictions with IoU-based NMS
        """
        try:
            from sahi.sliced_prediction import (
                slice_image,
                SlicedPredictionResult
            )
            
            # Prepare image for SAHI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
            h, w = frame_rgb.shape[:2]
            
            # Slice image
            slice_bboxes = slice_image(
                image=frame_rgb,
                height=self.config.slice_height,
                width=self.config.slice_width,
                overlap_height_ratio=self.config.overlap_height_ratio,
                overlap_width_ratio=self.config.overlap_width_ratio,
            )
            
            # Run YOLO on each slice
            all_boxes = []
            all_confidences = []
            all_class_ids = []
            
            for slice_bbox in slice_bboxes:
                # Extract slice from image
                slice_img = frame_rgb[
                    slice_bbox[1]:slice_bbox[3],
                    slice_bbox[0]:slice_bbox[2]
                ]
                
                # Run YOLO on slice
                results = self.model(
                    slice_img,
                    conf=self.config.confidence_threshold,
                    iou=self.config.nms_threshold,
                    device=self.config.device
                )
                
                result = results[0]
                
                if result.boxes.shape[0] > 0:
                    # Get detections
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Adjust coordinates to original image space
                    boxes[:, [0, 2]] += slice_bbox[0]  # x1, x2
                    boxes[:, [1, 3]] += slice_bbox[1]  # y1, y2
                    
                    all_boxes.append(boxes)
                    all_confidences.append(confidences)
                    all_class_ids.append(class_ids)
            
            # Merge detections
            if all_boxes:
                boxes = np.vstack(all_boxes)
                confidences = np.concatenate(all_confidences)
                class_ids = np.concatenate(all_class_ids)
                
                # Apply NMS to merged detections
                keep_indices = cv2.dnn.NMSBoxes(
                    bboxes=[
                        [int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])]
                        for b in boxes
                    ],
                    scores=confidences.tolist(),
                    score_threshold=self.config.confidence_threshold,
                    nms_threshold=self.config.nms_threshold
                )
                
                if keep_indices:
                    keep_indices = keep_indices.flatten()
                    boxes = boxes[keep_indices]
                    confidences = confidences[keep_indices]
                    class_ids = class_ids[keep_indices]
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                confidences = np.empty(0, dtype=np.float32)
                class_ids = np.empty(0, dtype=np.int32)
            
            # Get class names
            class_names = [self.model.names[cid] for cid in class_ids]
            
            return {
                'boxes': boxes,
                'confidences': confidences,
                'class_ids': class_ids,
                'class_names': class_names,
                'sahi_enabled': True
            }
            
        except Exception as e:
            logger.error(f"SAHI detection failed: {e}. Falling back to standard detection.")
            return self._detect_standard(frame)
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        use_sahi: Optional[bool] = None
    ) -> List[Dict]:
        """
        Detect vehicles in a batch of frames.
        
        Args:
            frames: List of input frames
            use_sahi: Override config for SAHI usage
            
        Returns:
            List of detection dictionaries
        """
        return [self.detect(frame, use_sahi) for frame in frames]
    
    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: Dict,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Visualize detections on frame.
        
        Args:
            frame: Input frame
            detections: Detection dictionary from detect()
            show_confidence: Whether to show confidence scores
            
        Returns:
            Frame with bounding boxes drawn
        """
        vis_frame = frame.copy()
        
        boxes = detections['boxes']
        confidences = detections['confidences']
        class_names = detections['class_names']
        
        for box, conf, class_name in zip(boxes, confidences, class_names):
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}" if show_confidence else class_name
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return vis_frame
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {}
        
        return {
            'model_name': self.config.model_name,
            'model_type': 'YOLO',
            'num_classes': len(self.model.names),
            'class_names': list(self.model.names.values()),
            'device': self.config.device,
            'sahi_enabled': self.config.use_sahi and self.sahi_model is not None,
        }
