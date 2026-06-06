"""Core modules for vehicle detection and classification"""

from .detector import VehicleDetector
from .classifier import VehicleClassifier
from .calibration import CalibrationManager

__all__ = [
    "VehicleDetector",
    "VehicleClassifier",
    "CalibrationManager",
]
