"""
Vehicle Classification by Physical Dimensions

Classification Method:
- Based ONLY on Length (L) and Width (W) extracted from bounding box
- No attempted inference from other features
- 8 standard traffic categories + pedestrians/cycles
- Maps to TRL/ORN11 and FHWA standards
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class VehicleCategory(Enum):
    """Standard vehicle categories for traffic analysis"""
    MOTORCYCLE_CYCLE = 1      # Motorcycle, scooter, cycle
    SMALL_CAR = 2             # Compact car
    STANDARD_CAR = 3          # Regular sedan/hatchback
    LARGE_CAR_VAN = 4         # SUV, wagon, small van
    LIGHT_COMMERCIAL = 5      # Small truck, large van
    HEAVY_VEHICLE = 6         # Large truck
    ARTICULATED_TRUCK = 7     # HGV with trailer
    BUS = 8                   # Bus
    PEDESTRIAN = 99           # Pedestrian
    UNKNOWN = 0               # Unclassifiable


@dataclass
class DimensionThresholds:
    """
    Decision thresholds for dimension-based classification.
    All values in meters.
    """
    motorcycle_max_length: float = 2.5
    motorcycle_max_width: float = 1.6
    
    small_car_max_length: float = 4.0
    small_car_max_width: float = 1.8
    
    standard_car_max_length: float = 5.0
    standard_car_max_width: float = 1.9
    
    large_car_max_length: float = 6.5
    large_car_max_width: float = 2.1
    
    light_comm_max_length: float = 8.0
    light_comm_max_width: float = 2.3
    
    heavy_max_length: float = 12.0
    heavy_max_width: float = 2.5
    
    articulated_min_length: float = 12.0
    bus_min_area: float = 20.0
    bus_typical_width: float = 2.5
    bus_max_length: float = 14.0


@dataclass
class ClassificationResult:
    """
    Result of vehicle classification
    """
    category: VehicleCategory
    category_code: int
    category_name: str
    
    # Dimensions
    length_m: Optional[float]
    width_m: Optional[float]
    area_m2: Optional[float]
    aspect_ratio: Optional[float]
    
    # Standards mapping
    trl_class: Optional[int]          # TRL/ORN11 class (1-6)
    fhwa_class: Optional[int]         # FHWA class (1-13)
    pcu: float                         # Passenger Car Unit
    
    # Quality
    confidence: float                 # 0.0-1.0
    flagged: bool = False
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'category': self.category_name,
            'category_code': self.category_code,
            'dimensions': {
                'length_m': self.length_m,
                'width_m': self.width_m,
                'area_m2': self.area_m2,
                'aspect_ratio': self.aspect_ratio,
            },
            'standards': {
                'trl_class': self.trl_class,
                'fhwa_class': self.fhwa_class,
                'pcu': self.pcu,
            },
            'quality': {
                'confidence': self.confidence,
                'flagged': self.flagged,
                'reason': self.reason,
            },
        }


class VehicleClassifier:
    """
    Classify vehicles based on physical dimensions (Length × Width).
    
    This is the ONLY reliable method at UAV altitude in top-down view.
    """
    
    # Mapping: VehicleCategory -> (TRL class, FHWA class, PCU)
    STANDARDS_MAPPING = {
        VehicleCategory.MOTORCYCLE_CYCLE: {'trl': 1, 'fhwa': 1, 'pcu': 0.60},
        VehicleCategory.SMALL_CAR: {'trl': 2, 'fhwa': 2, 'pcu': 0.90},
        VehicleCategory.STANDARD_CAR: {'trl': 2, 'fhwa': 2, 'pcu': 1.00},
        VehicleCategory.LARGE_CAR_VAN: {'trl': 3, 'fhwa': 3, 'pcu': 1.30},
        VehicleCategory.LIGHT_COMMERCIAL: {'trl': 3, 'fhwa': 4, 'pcu': 1.60},
        VehicleCategory.HEAVY_VEHICLE: {'trl': 5, 'fhwa': 7, 'pcu': 2.50},
        VehicleCategory.ARTICULATED_TRUCK: {'trl': 5, 'fhwa': 10, 'pcu': 3.20},
        VehicleCategory.BUS: {'trl': 6, 'fhwa': 6, 'pcu': 2.80},
        VehicleCategory.PEDESTRIAN: {'trl': None, 'fhwa': None, 'pcu': 0.10},
        VehicleCategory.UNKNOWN: {'trl': 2, 'fhwa': 2, 'pcu': 1.00},
    }
    
    def __init__(
        self,
        thresholds: Optional[DimensionThresholds] = None,
    ):
        """
        Initialize classifier.
        
        Args:
            thresholds: Custom dimension thresholds (use defaults if None)
        """
        self.thresholds = thresholds or DimensionThresholds()
        logger.info("VehicleClassifier initialized")
    
    def classify_vehicle(
        self,
        length_m: float,
        width_m: float,
        confidence: float = 0.85,
    ) -> ClassificationResult:
        """
        Classify vehicle based on dimensions.
        
        Decision tree (order matters - most specific first):
        1. Motorcycle/Scooter (very small)
        2. Articulated Truck (very long)
        3. Bus (large rectangular)
        4. Heavy Vehicle (8-12m long)
        5. Light Commercial (6-8m)
        6. Large Car/Van (4.5-6.5m)
        7. Standard Car (4.0-5.0m)
        8. Small Car (3.5-4.0m)
        
        Args:
            length_m: Vehicle length in meters
            width_m: Vehicle width in meters
            confidence: Detector confidence (affects classification confidence)
            
        Returns:
            ClassificationResult
        """
        area_m2 = length_m * width_m
        aspect_ratio = length_m / width_m if width_m > 0 else 1.0
        
        # Decision logic (order matters)
        
        # 1. Motorcycle/Scooter/Cycle (very small)
        if (length_m < self.thresholds.motorcycle_max_length and
            width_m < self.thresholds.motorcycle_max_width):
            return self._create_result(
                category=VehicleCategory.MOTORCYCLE_CYCLE,
                length_m=length_m,
                width_m=width_m,
                area_m2=area_m2,
                aspect_ratio=aspect_ratio,
                confidence=0.85 * confidence,
            )
        
        # 2. Articulated Truck (very long)
        if length_m > self.thresholds.articulated_min_length:
            return self._create_result(
                category=VehicleCategory.ARTICULATED_TRUCK,
                length_m=length_m,
                width_m=width_m,
                area_m2=area_m2,
                aspect_ratio=aspect_ratio,
                confidence=0.80 * confidence,
            )
        
        # 3. Bus (large area, rectangular, 10-14m long)
        if (area_m2 > self.thresholds.bus_min_area and
            length_m >= 10.0 and
            length_m <= self.thresholds.bus_max_length):
            return self._create_result(
                category=VehicleCategory.BUS,
                length_m=length_m,
                width_m=width_m,
                area_m2=area_m2,
                aspect_ratio=aspect_ratio,
                confidence=0.82 * confidence,
            )
        
        # 4. Heavy Vehicle (8-12m long)
        if (length_m >= 8.0 and
            length_m < self.thresholds.heavy_max_length):
            return self._create_result(
                category=VehicleCategory.HEAVY_VEHICLE,
                length_m=length_m,
                width_m=width_m,
                area_m2=area_m2,
                aspect_ratio=aspect_ratio,
                confidence=0.78 * confidence,
            )
        
        # 5. Light Commercial (6-8m)
        if (length_m >= 6.0 and
            length_m < self.thresholds.light_comm_max_length):
            return self._create_result(
                category=VehicleCategory.LIGHT_COMMERCIAL,
                length_m=length_m,
                width_m=width_m,
                area_m2=area_m2,
                aspect_ratio=aspect_ratio,
                confidence=0.80 * confidence,
            )
        
        # 6. Large Car/Van (4.5-6.5m)
        if (length_m >= 4.5 and
            length_m <= self.thresholds.large_car_max_length):
            return self._create_result(
                category=VehicleCategory.LARGE_CAR_VAN,
                length_m=length_m,
                width_m=width_m,
                area_m2=area_m2,
                aspect_ratio=aspect_ratio,
                confidence=0.82 * confidence,
            )
        
        # 7. Standard Car (4.0-5.0m)
        if (length_m >= 4.0 and
            length_m < self.thresholds.standard_car_max_length):
            return self._create_result(
                category=VehicleCategory.STANDARD_CAR,
                length_m=length_m,
                width_m=width_m,
                area_m2=area_m2,
                aspect_ratio=aspect_ratio,
                confidence=0.85 * confidence,
            )
        
        # 8. Small Car (3.5-4.0m) - default car-like
        if length_m >= 3.5:
            return self._create_result(
                category=VehicleCategory.SMALL_CAR,
                length_m=length_m,
                width_m=width_m,
                area_m2=area_m2,
                aspect_ratio=aspect_ratio,
                confidence=0.85 * confidence,
            )
        
        # Fallback: Unknown
        result = self._create_result(
            category=VehicleCategory.UNKNOWN,
            length_m=length_m,
            width_m=width_m,
            area_m2=area_m2,
            aspect_ratio=aspect_ratio,
            confidence=0.50 * confidence,
        )
        result.flagged = True
        result.reason = f"Dimensions do not match any category (L={length_m:.1f}m, W={width_m:.1f}m)"
        
        return result
    
    def classify_pedestrian(self) -> ClassificationResult:
        """
        Create pedestrian classification.
        
        Returns:
            ClassificationResult for pedestrian
        """
        return self._create_result(
            category=VehicleCategory.PEDESTRIAN,
            length_m=None,
            width_m=None,
            area_m2=None,
            aspect_ratio=None,
            confidence=0.90,
        )
    
    def batch_classify(
        self,
        dimensions_list: List[Dict],
    ) -> List[ClassificationResult]:
        """
        Classify multiple vehicles.
        
        Args:
            dimensions_list: List of dicts with 'length_m', 'width_m', 'confidence'
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        for dims in dimensions_list:
            result = self.classify_vehicle(
                length_m=dims['length_m'],
                width_m=dims['width_m'],
                confidence=dims.get('confidence', 0.85),
            )
            results.append(result)
        
        return results
    
    def _create_result(
        self,
        category: VehicleCategory,
        length_m: Optional[float],
        width_m: Optional[float],
        area_m2: Optional[float],
        aspect_ratio: Optional[float],
        confidence: float,
    ) -> ClassificationResult:
        """
        Create a ClassificationResult.
        """
        standards = self.STANDARDS_MAPPING[category]
        
        return ClassificationResult(
            category=category,
            category_code=category.value,
            category_name=category.name.replace('_', ' '),
            length_m=length_m,
            width_m=width_m,
            area_m2=area_m2,
            aspect_ratio=aspect_ratio,
            trl_class=standards['trl'],
            fhwa_class=standards['fhwa'],
            pcu=standards['pcu'],
            confidence=min(confidence, 1.0),
        )
