"""Detection components."""

try:
    from .yolo_detector import DetectionConfig, YOLODetector
    __all__ = ["DetectionConfig", "YOLODetector"]
except ImportError:
    # ultralytics not installed
    __all__ = []
