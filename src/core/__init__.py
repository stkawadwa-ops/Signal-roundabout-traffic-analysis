"""Core modules for vehicle detection and classification"""

from .classifier import VehicleClassifier
from .calibration import CalibrationManager

try:
    from .detector import VehicleDetector
    __all__ = [
        "VehicleDetector",
        "VehicleClassifier",
        "CalibrationManager",
    ]
except ImportError:
    # ultralytics / torch not installed — classifier and calibrator still work
    __all__ = [
        "VehicleClassifier",
        "CalibrationManager",
    ]
