# Vehicle Detection & Classification - Realistic UAV Implementation

**Practical Implementation for High-Altitude UAV Traffic Analysis**  
*Based ONLY on Visible Physical Characteristics: Length and Width*

---

## Table of Contents

1. [Core Principle](#core-principle)
2. [What IS Actually Visible from UAV](#what-is-actually-visible-from-uav)
3. [Classification System (Length + Width Based)](#classification-system-length--width-based)
4. [Detection Implementation](#detection-implementation)
5. [Category Definitions](#category-definitions)
6. [Pedestrian & Cycle Detection](#pedestrian--cycle-detection)
7. [Implementation Architecture](#implementation-architecture)
8. [Calibration Procedure](#calibration-procedure)
9. [Quality Assurance](#quality-assurance)

---

## Core Principle

**What matters at UAV altitude:**
- Vehicle **footprint** (length × width as seen from directly above)
- That's it. This is the ONLY reliable visible feature.

**What we cannot reliably see:**
- ❌ Passenger windows (too small, variable angles)
- ❌ Vehicle height (2D top-down view doesn't show height)
- ❌ Wheels/axles (under vehicle body)
- ❌ Cargo area details (orientation dependent)
- ❌ Markings/branding (poor resolution from altitude)
- ❌ Pedestrian characteristics (all look the same from above)
- ❌ Motorcycle vs bicycle (can't distinguish)

---

## What IS Actually Visible from UAV

### Directly Observable at 30-50m Altitude

| Feature | Visible | Confidence | Use |
|---------|---------|------------|-----|
| **Vehicle length (L/W direction)** | ✅ Yes | High | PRIMARY classifier |
| **Vehicle width (perpendicular)** | ✅ Yes | High | PRIMARY classifier |
| **Vehicle footprint area** | ✅ Yes | High | Secondary classifier |
| **Aspect ratio (L:W ratio)** | ✅ Yes | High | Shape indicator |
| **Overall silhouette shape** | ✅ Yes | Medium | Rough type hint |
| **Color** | ✅ Yes | Medium | Reference only |
| **Motion pattern** | ✅ Yes | High | Speed/direction |

### NOT Reliably Observable

| Feature | Why Not | Impact |
|---------|--------|--------|
| **Wheels/Tires** | Under vehicle body | Cannot count axles/tires |
| **Height** | 2D top-down view | No vertical dimension |
| **Windows** | Too small/resolution | Cannot count or identify |
| **Doors** | Low resolution | Cannot distinguish |
| **Cargo** | Obscured | Cannot see inside/under |
| **Age/gender** | Pedestrians indistinguishable | Cannot sub-categorize |
| **Motorcycle vs Bicycle** | Similar sizes/speeds in roundabout | Cannot reliably separate |

---

## Classification System (Length + Width Based)

### Single Decision Tree: L × W Dimensions

```
Vehicle Detected (Bounding Box)
    ↓
Extract Dimensions:
- Length (L) = bbox height (travel direction)
- Width (W) = bbox width (perpendicular)
    ↓
    ├─ If L < 2.5m → Motorcycle/Scooter/Tuk-tuk
    │  (Too small for car)
    │
    ├─ If L = 2.5-4.0m and W = 1.5-1.8m → Small Car
    │  (Compact car footprint)
    │
    ├─ If L = 4.0-5.0m and W = 1.7-1.9m → Standard Car
    │  (Regular sedan/hatchback)
    │
    ├─ If L = 4.5-6.5m and W = 1.8-2.1m → Large Car / Van
    │  (SUV, wagon, small van)
    │
    ├─ If L = 6.0-8.0m and W = 2.0-2.3m → Light Commercial
    │  (Small truck, large van)
    │
    ├─ If L = 8.0-12.0m and W = 2.3-2.5m → Heavy Vehicle
    │  (Medium/large truck)
    │
    ├─ If L > 12.0m and W = 2.3-2.5m → Articulated Truck
    │  (HGV with visible trailer section - clear L/W discontinuity)
    │
    └─ If area > 20 m² and L = 10-14m → Bus
       (Large rectangular footprint, distinct from trucks)
```

---

## Category Definitions

### 8 Practical Vehicle Categories

| ID | Category | Length Range | Width Range | Footprint | TRL | PCU | Characteristics |
|----|----|---------|---------|-----------|-----|-----|---|
| **1** | Motorcycle/Scooter/Cycle | <2.5m | <1.6m | <4 m² | 1 | 0.6 | Small 2-wheeled |
| **2** | Small Car | 3.5-4.0m | 1.5-1.8m | 5-7 m² | 2 | 0.9 | Compact footprint |
| **3** | Standard Car | 4.0-5.0m | 1.7-1.9m | 7-9.5 m² | 2 | 1.0 | Regular sedan/hatch |
| **4** | Large Car/Van | 4.5-6.5m | 1.8-2.1m | 8-14 m² | 3 | 1.3 | SUV/wagon/van mix |
| **5** | Light Commercial | 6.0-8.0m | 2.0-2.3m | 12-18 m² | 3-4 | 1.6 | Small truck/van |
| **6** | Heavy Vehicle | 8.0-12.0m | 2.3-2.5m | 18-30 m² | 5 | 2.5 | Large truck |
| **7** | Articulated Truck | >12.0m | 2.3-2.5m | >30 m² | 5 | 3.2 | HGV with trailer |
| **8** | Bus | 10-14m | 2.4-2.8m | 24-39 m² | 6 | 2.8 | Tall rectangular |

### Pedestrian Category

| Category | Detection | TRL | PCU | Characteristics |
|----------|-----------|-----|-----|---|
| **Pedestrian** | YOLO person detection | - | 0.10 | Walking/standing, bipedal motion |
| **Pedestrians (Group)** | 2+ close persons | - | 0.15 per person | Moving together |

---

## Detection Implementation

### Step 1: Calibration (Pixel → Meters)

```python
class CalibrationManager:
    """
    Calibrate pixel-to-meter conversion using standard road markings
    """
    
    @staticmethod
    def calibrate_from_lane_markings(
        frame,
        known_lane_width_m: float = 3.5
    ) -> float:
        """
        Use standard road lane marking to calibrate
        - Standard lane = 3.5m width
        - Measure pixels across lane marking
        - Calculate ratio
        """
        pass
    
    @staticmethod
    def calibrate_from_road_width(
        frame,
        roundabout_radius_m: float
    ) -> float:
        """
        Use known roundabout radius to calibrate
        - Measure roundabout diameter in pixels
        - Calculate ratio
        """
        pass
    
    @staticmethod
    def validate_calibration(
        ratio: float,
        frame,
        known_features: dict
    ) -> bool:
        """
        Validate calibration by checking against multiple known features
        """
        pass
```

### Step 2: Dimension Extraction

```python
class DimensionExtractor:
    """
    Extract length and width from bounding box
    Using calibrated pixel-to-meter ratio
    """
    
    def __init__(self, pixel_to_meter_ratio: float):
        """
        ratio: meters per pixel
        Example: if 100 pixels = 3.5m (lane width), ratio = 0.035
        """
        self.ratio = pixel_to_meter_ratio
    
    def extract_dimensions(self, bbox: tuple) -> dict:
        """
        Extract from bounding box: (x1, y1, x2, y2)
        
        Assumption: Vehicle aligned with direction of travel
        (reasonable for traffic on roundabout)
        """
        
        x1, y1, x2, y2 = bbox
        
        width_pixels = x2 - x1  # Perpendicular to travel
        length_pixels = y2 - y1  # Direction of travel (top-down view)
        
        width_m = width_pixels * self.ratio
        length_m = length_pixels * self.ratio
        
        area_m2 = width_m * length_m
        aspect_ratio = length_m / width_m if width_m > 0 else 1.0
        
        return {
            'length_m': length_m,
            'width_m': width_m,
            'area_m2': area_m2,
            'aspect_ratio': aspect_ratio,
            'length_pixels': length_pixels,
            'width_pixels': width_pixels,
        }
```

### Step 3: Classification by Dimensions

```python
class DimensionClassifier:
    """
    Classify vehicles based ONLY on length and width
    """
    
    # Decision boundaries (meters)
    THRESHOLDS = {
        'motorcycle_max_length': 2.5,
        'motorcycle_max_width': 1.6,
        
        'small_car_max_length': 4.0,
        'small_car_max_width': 1.8,
        
        'standard_car_max_length': 5.0,
        'standard_car_max_width': 1.9,
        
        'large_car_max_length': 6.5,
        'large_car_max_width': 2.1,
        
        'light_comm_max_length': 8.0,
        'light_comm_max_width': 2.3,
        
        'heavy_max_length': 12.0,
        'heavy_max_width': 2.5,
        
        'articulated_min_length': 12.0,
        'bus_min_area': 20.0,  # m²
        'bus_typical_width': 2.5,
    }
    
    def classify(self, dimensions: dict) -> dict:
        """
        Classify vehicle based on L and W only
        """
        
        length = dimensions['length_m']
        width = dimensions['width_m']
        area = dimensions['area_m2']
        
        # Decision logic (order matters - most specific first)
        
        # Check 1: Motorcycle/Scooter/Cycle (too small)
        if (length < self.THRESHOLDS['motorcycle_max_length'] and
            width < self.THRESHOLDS['motorcycle_max_width']):
            return {
                'category': 'Motorcycle/Cycle',
                'category_code': 1,
                'trl_class': 1,
                'pcu': 0.60,
                'length_m': length,
                'width_m': width,
                'confidence': 0.85
            }
        
        # Check 2: Articulated Truck (very long)
        if length > self.THRESHOLDS['articulated_min_length']:
            return {
                'category': 'Articulated Truck',
                'category_code': 7,
                'trl_class': 5,
                'pcu': 3.2,
                'length_m': length,
                'width_m': width,
                'confidence': 0.80
            }
        
        # Check 3: Bus (large area, typical width)
        if (area > self.THRESHOLDS['bus_min_area'] and
            length >= 10.0 and length <= 14.0):
            return {
                'category': 'Bus',
                'category_code': 8,
                'trl_class': 6,
                'pcu': 2.8,
                'length_m': length,
                'width_m': width,
                'confidence': 0.82
            }
        
        # Check 4: Heavy Vehicle (8-12m long)
        if (length >= 8.0 and 
            length < self.THRESHOLDS['heavy_max_length']):
            return {
                'category': 'Heavy Vehicle/Truck',
                'category_code': 6,
                'trl_class': 5,
                'pcu': 2.5,
                'length_m': length,
                'width_m': width,
                'confidence': 0.78
            }
        
        # Check 5: Light Commercial (6-8m)
        if (length >= 6.0 and
            length < self.THRESHOLDS['light_comm_max_length']):
            return {
                'category': 'Light Commercial',
                'category_code': 5,
                'trl_class': 3,
                'pcu': 1.6,
                'length_m': length,
                'width_m': width,
                'confidence': 0.80
            }
        
        # Check 6: Large Car/Van (4.5-6.5m)
        if (length >= 4.5 and
            length <= self.THRESHOLDS['large_car_max_length']):
            return {
                'category': 'Large Car/Van',
                'category_code': 4,
                'trl_class': 3,
                'pcu': 1.3,
                'length_m': length,
                'width_m': width,
                'confidence': 0.82
            }
        
        # Check 7: Standard Car (4.0-5.0m)
        if (length >= 4.0 and
            length < self.THRESHOLDS['standard_car_max_length']):
            return {
                'category': 'Standard Car',
                'category_code': 3,
                'trl_class': 2,
                'pcu': 1.0,
                'length_m': length,
                'width_m': width,
                'confidence': 0.85
            }
        
        # Check 8: Small Car (3.5-4.0m) - default for car-sized
        if length >= 3.5:
            return {
                'category': 'Small Car',
                'category_code': 2,
                'trl_class': 2,
                'pcu': 0.9,
                'length_m': length,
                'width_m': width,
                'confidence': 0.85
            }
        
        # Fallback: Unknown/ambiguous
        return {
            'category': 'Unknown',
            'category_code': 0,
            'trl_class': 2,
            'pcu': 1.0,
            'length_m': length,
            'width_m': width,
            'confidence': 0.50,
            'flagged': True,
            'reason': 'Dimensions do not match any category'
        }
```

---

## Pedestrian & Cycle Detection

### Pedestrian Detection (Simple)

```python
class PedestrianDetector:
    """
    Pedestrians detected from YOLO person detection
    No sub-categorization (cannot distinguish from above)
    """
    
    def classify_pedestrian(self, bbox: tuple) -> dict:
        """
        Simple pedestrian detection
        """
        return {
            'category': 'Pedestrian',
            'category_code': None,
            'trl_class': None,
            'pcu': 0.10,
            'confidence': 0.85
        }
    
    def detect_pedestrian_group(self, nearby_pedestrians: list) -> dict:
        """
        Group of pedestrians crossing together
        """
        group_size = len(nearby_pedestrians)
        
        return {
            'category': 'Pedestrian Group',
            'category_code': None,
            'group_size': group_size,
            'trl_class': None,
            'pcu': 0.10 * group_size,
            'confidence': 0.80
        }
```

### Cycle Detection (Unified)

```python
class CycleDetector:
    """
    Motorcycles, scooters, bicycles, mopeds all classified as 'Cycle'
    Cannot distinguish at UAV altitude + same traffic impact in roundabout
    """
    
    def classify_cycle(self, dimensions: dict) -> dict:
        """
        Small 2-wheeled vehicle (bicycle or motorcycle)
        """
        
        length = dimensions['length_m']
        width = dimensions['width_m']
        
        if (length < 2.5 and width < 1.6):
            return {
                'category': 'Cycle',
                'subcategory': None,  # No distinction
                'category_code': 1,
                'trl_class': 1,
                'pcu': 0.60,
                'length_m': length,
                'width_m': width,
                'confidence': 0.80
            }
        
        return None  # Not a cycle
```

---

## Implementation Architecture

### Simplified Pipeline

```
Raw UAV Frame (30-50m altitude, looking down)
    ↓
┌─────────────────────────────┐
│ YOLO Person-Vehicle Det     │
│ (Basic: person vs vehicle)  │
└────────────┬────────────────┘
             │
    ┌────────┴──────────┐
    │                   │
┌───▼──────────┐   ┌────▼────────────────┐
│ PERSON       │   │ VEHICLE (bounding   │
│ Detected     │   │ box)                │
└───┬──────────┘   └────┬────────────────┘
    │                   │
    │            ┌──────▼────────────────┐
    │            │ Extract Dimensions   │
    │            │ from Bounding Box    │
    │            │ (L, W using ratio)   │
    │            └──────┬────────────────┘
    │                   │
    │            ┌──────▼────────────────┐
    │            │ DimensionClassifier  │
    │            │ Classify by L + W    │
    │            │ only                 │
    │            └──────┬────────────────┘
    │                   │
    │            ┌──────▼────────────────┐
    │            │ Classification       │
    │            │ Result (8 categories)│
    │            └──────┬────────────────┘
    │                   │
    │      ┌────────────┘
    │      │
    ├──────┴─────────────────────────┐
    │                                │
┌───▼──────────────┐         ┌───────▼──────┐
│ Pedestrian       │         │ Vehicle      │
│ Record           │         │ Record       │
│ (TRL: -)         │         │ (TRL 1-6)    │
└──────────────────┘         └───────┬──────┘
                                     │
                            ┌────────▼─────────┐
                            │ Final Detection  │
                            │ Record           │
                            │ (L, W, Category, │
                            │  TRL, PCU)       │
                            └──────────────────┘
```

---

## Calibration Procedure

### Recommended Calibration Steps

**Step 1: Before flight**
- Measure actual lane width at roundabout (standard = 3.5m)
- Mark start and end points clearly

**Step 2: During flight**
- Capture frame with lane markings clearly visible
- Measure pixel distance across marked lane
- Calculate: ratio = 3.5m / measured_pixels

**Example:**
```
Lane width = 3.5m (physical)
Pixels across lane = 100 pixels
Ratio = 3.5 / 100 = 0.035 m/pixel
```

**Step 3: Validation**
- Test on known vehicles parked in view
- Measure actual vehicle dimensions
- Compare with extracted dimensions
- Should be within ±10% error

```python
# Validation example
measured_car_length = 4.5  # meters (physical measurement)
extracted_car_length = 4.4  # meters (from pixels + ratio)
error_percent = abs(measured_car_length - extracted_car_length) / measured_car_length * 100
# error_percent = 2.2% ✓ Good
```

---

## Detection Record Format

### Simplified Output

```python
class DetectionRecord:
    detection_id: int
    frame_number: int
    timestamp: float
    
    # Bounding box
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float  # 0.0-1.0
    
    # Classification
    category: str  # One of 8 categories
    category_code: int  # 1-8
    
    # Dimensions (if vehicle)
    length_m: Optional[float]
    width_m: Optional[float]
    area_m2: Optional[float]
    
    # Standards
    trl_class: Optional[int]  # TRL/ORN11 mapping
    pcu: float  # Passenger Car Unit
    
    # Speed (from tracking)
    speed_kmh: Optional[float]
    
    # Quality
    flagged: bool
    reason: Optional[str]
```

### JSON Export Example

```json
{
  "detection_id": 1042,
  "frame_number": 1250,
  "timestamp": 125.5,
  "bbox": [640, 360, 720, 420],
  "confidence": 0.88,
  "category": "Standard Car",
  "category_code": 3,
  "dimensions": {
    "length_m": 4.5,
    "width_m": 1.8,
    "area_m2": 8.1
  },
  "classification": {
    "trl_class": 2,
    "pcu": 1.0
  },
  "speed_kmh": 35.2,
  "quality": {
    "flagged": false,
    "reason": null
  }
}
```

### CSV Format

```csv
frame, timestamp, category, length_m, width_m, trl_class, pcu, speed_kmh, confidence, flagged

1250, 125.5, "Standard Car", 4.5, 1.8, 2, 1.0, 35.2, 0.88, false
1251, 125.6, "Pedestrian", null, null, null, 0.10, 1.2, 0.92, false
1252, 125.7, "Cycle", 2.2, 0.8, 1, 0.60, 8.5, 0.75, false
1253, 125.8, "Bus", 12.0, 2.6, 6, 2.8, 25.0, 0.85, false
1254, 125.9, "Heavy Vehicle", 10.0, 2.4, 5, 2.5, 30.0, 0.80, false
```

---

## Quality Assurance

### Validation Rules

| Check | Rule | Action if Violated |
|-------|------|-------------------|
| **Dimension Plausibility** | 2.0m ≤ L ≤ 20m, 1.4m ≤ W ≤ 2.8m | Flag for review |
| **Aspect Ratio** | L:W between 1.3 and 8.0 | Flag anomaly |
| **Category Consistency** | Vehicle stays in category ±1 frame | Log switches |
| **Speed Sanity** | L < 3m: speed <15 m/s; L > 8m: speed <12 m/s | Flag violators |
| **Calibration Check** | Known features match ±10% | Re-calibrate if worse |

### Flagging for Manual Review

```python
def should_flag_for_review(detection: DetectionRecord) -> bool:
    """Flag detections needing manual verification"""
    
    reasons = []
    
    # Low confidence
    if detection.confidence < 0.70:
        reasons.append("Low confidence (<0.70)")
    
    # Dimension anomalies
    if detection.length_m:
        if detection.length_m < 2.0 or detection.length_m > 20.0:
            reasons.append(f"Anomalous length: {detection.length_m:.1f}m")
    
    if detection.width_m:
        if detection.width_m < 1.4 or detection.width_m > 2.8:
            reasons.append(f"Anomalous width: {detection.width_m:.1f}m")
    
    # Weird aspect ratios
    if (detection.length_m and detection.width_m and 
        detection.length_m / detection.width_m > 8.0):
        reasons.append("Extreme aspect ratio (possibly FP)")
    
    if reasons:
        detection.flagged = True
        detection.reason = "; ".join(reasons)
        return True
    
    return False
```

---

## Implementation Checklist

### Phase 1: Setup & Calibration (Week 1)
- [ ] Set up YOLO person-vehicle detector
- [ ] Implement dimension extraction from bounding box
- [ ] Develop calibration procedure (lane markings)
- [ ] Test on sample video
- [ ] **Deliverable**: Calibrated pixel-to-meter ratio

### Phase 2: Classification (Week 2)
- [ ] Implement DimensionClassifier (8 categories)
- [ ] Implement decision tree logic
- [ ] Test on diverse vehicles
- [ ] Create validation dataset
- [ ] **Deliverable**: Classification working on test data

### Phase 3: Integration (Week 3)
- [ ] Integrate with pedestrian detection
- [ ] Add cycle detection
- [ ] Create detection records
- [ ] Implement QA flagging
- [ ] **Deliverable**: Complete pipeline end-to-end

### Phase 4: Testing & Refinement (Week 4)
- [ ] Test on all 7 roundabout sample videos
- [ ] Validate dimensions against ground truth
- [ ] Refine thresholds if needed
- [ ] Create test dataset
- [ ] **Deliverable**: Production-ready detector

---

## Key Advantages of This Approach

✅ **Simple**: Only uses L × W (two numbers)  
✅ **Reliable**: Dimensions always visible from above  
✅ **Fast**: No complex feature extraction  
✅ **Maintainable**: Clear decision logic  
✅ **Calibratable**: Single one-time calibration  
✅ **Practical**: Works with UAV altitude constraints  
✅ **Relevant**: Categories match roundabout traffic patterns  
✅ **Traceable**: Easy to validate and explain results  

