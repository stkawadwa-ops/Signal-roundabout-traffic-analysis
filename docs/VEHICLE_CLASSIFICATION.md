# Vehicle Detection & Classification Implementation Guide

**Comprehensive Implementation for Detection & Parameter Extraction**  
*FHWA 13-Class + TRL/ORN11 Standards + Pedestrian Detection*

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [YOLO-Based Base Detection](#yolo-based-base-detection)
3. [Vehicle Classification Pipeline](#vehicle-classification-pipeline)
4. [Pedestrian & Cyclist Detection](#pedestrian--cyclist-detection)
5. [Multi-Standard Output](#multi-standard-output)
6. [Confidence & Quality Management](#confidence--quality-management)
7. [Data Export Format](#data-export-format)
8. [Implementation Roadmap](#implementation-roadmap)

---

## System Architecture

### Complete Detection Pipeline

```
Raw UAV Frame (1920x1080, 30 FPS)
    ↓
┌───────────────────────────────────────────┐
│   YOLO v8 Person-Vehicle Detection       │
│   (Pre-trained on COCO + custom traffic) │
├───────────────────────────────────────────┤
│ Outputs:                                  │
│ - Bounding boxes for all road users       │
│ - Confidence scores (0.5 - 1.0)           │
│ - Rough class: person / vehicle types    │
└───────────┬───────────────────────────────┘
            │
    ┌───────┴────────┐
    │                │
┌───▼───────────┐  ┌▼──────────────┐
│ PERSON        │  │ VEHICLE       │
│ Detections    │  │ Detections    │
└───┬───────────┘  └┬──────────────┘
    │              │
    │         ┌────┴──────────────────────┐
    │         │                           │
    │    ┌────▼──────────┐      ┌─────────▼──────┐
    │    │Bicycle CNN    │      │Vehicle Type    │
    │    │Classifier     │      │CNN Classifier  │
    │    │               │      │(FHWA 13 cls)   │
    │    └────┬──────────┘      └────┬──────────┘
    │         │ Yes (Bicycle)        │
    │    ┌────▼──────────┐      ┌────▼──────────┐
    │    │Bicycle        │      │FHWA Class     │
    │    │Record         │      │Result (1-13)  │
    │    └───────────────┘      └────┬──────────┘
    │                                │
    │         ┌──────────────────────┤
    │         │                      │
    │    ┌────▼────────────┐    ┌────▼───────────┐
    │    │Map to TRL/ORN11 │    │Create Vehicle  │
    │    │(ORN11 classes)  │    │Detection Record│
    │    │                 │    │(FHWA+TRL)      │
    │    └────┬────────────┘    └────┬───────────┘
    │         │                      │
    │         └──────────┬───────────┘
    │                    │
    │                    ▼
    │          ┌────────────────┐
    │          │Vehicle Result  │
    │          └────────────────┘
    │
    └──────────┬──────────────────┐
               │                  │
        ┌──────▼──────┐   ┌──────▼────────┐
        │Pedestrian   │   │Cyclist        │
        │Classifier   │   │(Processed)    │
        │(Sub-types)  │   └───────────────┘
        └──────┬──────┘
               │
        ┌──────▼──────────┐
        │Pedestrian       │
        │Record           │
        │(Adult/Child/    │
        │ Elderly/        │
        │ Assisted/Group) │
        └─────────────────┘
                   
            ↓ (All Results Combined)
            
     ┌──────────────────────────────┐
     │ Frame-Level Detection Output │
     │ (Pedestrians + Vehicles)     │
     │ Both Standards Mapped        │
     │ (FHWA + TRL/ORN11)           │
     └──────────────────────────────┘
```

---

## YOLO-Based Base Detection

### Model Selection & Configuration

**Why YOLOv8?**
- Real-time inference (20-30 FPS on GPU)
- Pre-trained on COCO (includes pedestrians, vehicles)
- Excellent mAP for traffic scenarios
- Small model size (suitable for batch processing)

### Base Detection Classes

```python
YOLO_CLASSES = {
    0: 'person',        # For pedestrian detection
    1: 'bicycle',       # For cyclist detection
    2: 'car',           # Generic vehicle
    3: 'motorcycle',    # Two-wheeled motor
    5: 'bus',           # Public transport
    7: 'truck',         # Commercial vehicle
    # Others (less relevant to traffic roundabouts)
}
```

### Detection Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Confidence Threshold** | 0.5 | Balance between recall and precision |
| **NMS Threshold** | 0.45 | Prevent duplicate detections |
| **Image Size** | 1024 | Speed/accuracy trade-off for UAV |
| **Batch Size** | 8 | GPU memory optimization |
| **Device** | GPU (CUDA) | Real-time processing required |
| **Inference Mode** | FP32 | Precision for edge cases |

### Detection Output

```python
class YOLODetection:
    """Raw YOLO detection output per frame"""
    
    bbox: tuple              # (x1, y1, x2, y2) in pixels
    confidence: float        # 0.0 - 1.0
    class_id: int           # YOLO class index (0-80)
    class_name: str         # YOLO class name
    area_pixels: int        # Width × Height
    center: tuple           # (cx, cy)
    
    # Computed properties
    width_pixels: int
    height_pixels: int
    aspect_ratio: float     # width / height
```

### GPU Memory Requirements

| Operation | Memory |
|-----------|--------|
| Model load | 800 MB |
| Batch inference (8 frames) | 2 GB |
| NMS post-processing | 200 MB |
| **Total** | ~3 GB |

---

## Vehicle Classification Pipeline

### Stage 1: Bicycle Detection (Binary Classifier)

**Purpose**: Separate bicycles from motorcycles/scooters (often confused)

```python
class BicycleDetector:
    """
    Specialized binary classifier to detect bicycles
    Reasons for separate detection:
    - Bicycles often misclassified as motorcycles/scooters
    - Different tracking and flow analysis
    - Essential for TRL/ORN11 "Cycle" (Class 7) category
    - Critical for roundabout safety (cyclist protection)
    """
    
    def __init__(self):
        self.model = load_pretrained_bicycle_classifier()
        self.confidence_threshold = 0.75
    
    def is_bicycle(self, frame, bbox, yolo_class) -> bool:
        """
        Returns True if detection is bicycle
        Features used:
        - Shape (thin profile, visible frame geometry)
        - Motion (pedal rotation patterns)
        - Texture (spoke patterns, tire treads)
        """
        
        roi = extract_roi(frame, bbox)
        
        # Feature extraction
        features = {
            'shape_features': self.extract_shape_features(roi),
            'texture_features': self.extract_texture_features(roi),
            'edge_features': self.extract_edge_features(roi),
            'size_ratios': self.compute_size_ratios(roi),
        }
        
        # Classification
        confidence = self.model.predict(features)
        
        # Decision logic
        if confidence > self.confidence_threshold:
            return True
        
        # Heuristic: YOLO says "bicycle" + features support it
        if yolo_class == 'bicycle' and confidence > 0.5:
            return True
        
        return False
    
    def extract_shape_features(self, roi) -> ndarray:
        """
        Extract shape features from ROI
        - Frame lines (vertical/diagonal)
        - Wheel roundness
        - Handlebars
        """
        pass
    
    def extract_texture_features(self, roi) -> ndarray:
        """
        Extract texture patterns
        - Spoke patterns
        - Seat texture
        - Tire treads
        """
        pass
```

### Stage 2: Vehicle Type Classification (FHWA 13 Classes)

**Purpose**: Classify vehicles into FHWA 13-class system (most detailed technical)

```python
class FHWAVehicleClassifier:
    """
    CNN-based classifier for FHWA 13-class system
    Outputs: FHWA class (1-13) which maps to TRL/ORN11
    
    Classes:
    1: Motorcycles
    2: Passenger Cars
    3: Other 2-Axle 4-Tire Vehicles (vans, pickups)
    4: Buses
    5-7: Various truck configurations
    8-13: Trailer configurations
    """
    
    FHWA_CLASSES = {
        1: 'Motorcycles',
        2: 'Passenger Cars',
        3: 'Other 2-Axle 4-Tire Vehicles',
        4: 'Buses',
        5: '2-Axle 6-Tire Single-Unit Trucks',
        6: '3-Axle Single-Unit Trucks',
        7: '4+ Axle Single-Unit Trucks',
        8: '4-5 Axle Single-Trailer Trucks',
        9: '5-Axle Single-Trailer Trucks',
        10: '6+ Axle Single-Trailer Trucks',
        11: '5- Axle Multi-Trailer Trucks',
        12: '6-Axle Multi-Trailer Trucks',
        13: '7+ Axle Multi-Trailer Trucks',
    }
    
    def __init__(self):
        self.model = load_pretrained_fhwa_classifier()
        self.confidence_threshold = 0.6
    
    def classify(self, frame, bbox) -> dict:
        """
        Classify vehicle into FHWA class
        """
        
        roi = extract_roi(frame, bbox)
        
        # Feature extraction (multi-modal)
        features = {
            'visual_features': self.extract_visual_features(roi),
            'shape_features': self.extract_shape_features(roi),
            'axle_indicators': self.detect_axle_count(roi),
            'tire_indicators': self.detect_tire_count(roi),
            'size_features': self.extract_size_features(bbox),
            'length_to_height': self.compute_ratios(bbox),
        }
        
        # Classification with softmax probabilities
        logits = self.model(features)
        probabilities = softmax(logits)
        
        # Get top-3 predictions for uncertainty handling
        top_indices = argsort(probabilities)[-3:][::-1]
        
        return {
            'fhwa_class': top_indices[0],
            'fhwa_name': self.FHWA_CLASSES[top_indices[0]],
            'confidence': probabilities[top_indices[0]],
            'alternatives': [
                {
                    'class': idx,
                    'name': self.FHWA_CLASSES[idx],
                    'probability': probabilities[idx]
                }
                for idx in top_indices[1:]
            ]
        }
    
    def detect_axle_count(self, roi) -> int:
        """
        Estimate axle count from visual indicators:
        - Wheel spacing patterns
        - Suspension gap signatures
        - Trailer connection points
        """
        # CNN-based axle detection
        pass
    
    def detect_tire_count(self, roi) -> int:
        """
        Count visible/estimated tires
        Critical for distinguishing:
        - Class 2 (4 tires) vs Class 3 (4 tires, but van)
        - Class 5 (6 tires dual rear) vs others
        """
        pass
```

### Stage 3: Standard Mapping (FHWA → TRL/ORN11)

```python
class ClassificationMapper:
    """
    Map FHWA (detailed technical) to TRL/ORN11 (practical)
    Also assigns PCU values for flow analysis
    """
    
    # FHWA → TRL/ORN11 Class Mapping
    FHWA_TO_TRL = {
        1: 1,   # Motorcycle → TRL Motorcycle
        2: 2,   # Passenger Car → TRL Passenger Car
        3: 3,   # Other 2-Axle 4-Tire → TRL LGV
        4: 6,   # Bus → TRL Bus
        5: 3,   # 2-Axle 6-Tire Truck → TRL LGV/MGV
        6: 4,   # 3-Axle Single-Unit → TRL MGV
        7: 5,   # 4+ Axle Single-Unit → TRL HGV
        8: 5,   # 4-5 Axle Single-Trailer → TRL HGV
        9: 5,   # 5-Axle Single-Trailer → TRL HGV
        10: 5,  # 6+ Axle Single-Trailer → TRL HGV
        11: 5,  # Multi-Trailer → TRL HGV
        12: 5,  # 6-Axle Multi-Trailer → TRL HGV
        13: 5,  # 7+ Axle Multi-Trailer → TRL HGV
    }
    
    # Passenger Car Units (PCU) for flow analysis
    FHWA_TO_PCU = {
        1: 0.6,     # Motorcycle - nimble
        2: 1.0,     # Car - reference
        3: 1.2,     # Van/Pickup
        4: 2.8,     # Bus - high capacity
        5: 1.5,     # Dual-tire truck
        6: 2.0,     # 3-axle truck
        7: 2.5,     # Large single-unit
        8: 3.0,     # 4-5 axle combo
        9: 3.0,     # 5-axle (18-wheeler)
        10: 3.5,    # 6+ axle
        11: 3.0,    # Multi-trailer
        12: 3.2,    # 6-axle multi
        13: 3.5,    # 7+ axle multi
    }
    
    @staticmethod
    def map_to_trl(fhwa_class: int) -> int:
        """Map FHWA class to TRL/ORN11 class"""
        return ClassificationMapper.FHWA_TO_TRL.get(fhwa_class, 5)
    
    @staticmethod
    def get_pcu(fhwa_class: int) -> float:
        """Get Passenger Car Unit value"""
        return ClassificationMapper.FHWA_TO_PCU.get(fhwa_class, 1.5)
```

---

## Pedestrian & Cyclist Detection

### Pedestrian Sub-Classification

```python
class PedestrianClassifier:
    """
    Classify pedestrians into sub-categories critical for roundabout safety
    """
    
    PEDESTRIAN_TYPES = {
        'adult': 'Adult Pedestrian (15-65 years)',
        'child': 'Child Pedestrian (<15 years)',
        'elderly': 'Elderly Pedestrian (65+ years)',
        'assisted': 'Assisted Pedestrian (crutches, walker, wheelchair)',
        'group': 'Pedestrian Group (2+ persons)',
    }
    
    def __init__(self):
        self.age_classifier = load_age_classifier()
        self.mobility_aid_detector = load_aid_detector()
        self.confidence_threshold = 0.65
    
    def classify(self, frame, bbox, track_history=None) -> dict:
        """
        Classify pedestrian with temporal consistency
        """
        
        roi = extract_roi(frame, bbox)
        
        # Extract ROI properties
        height_pixels = bbox[3] - bbox[1]
        width_pixels = bbox[2] - bbox[0]
        
        # Check 1: Is this a group?
        if track_history and len(track_history) > 1:
            is_group = self.detect_pedestrian_group(track_history)
            if is_group:
                return {
                    'pedestrian_type': 'group',
                    'description': self.PEDESTRIAN_TYPES['group'],
                    'confidence': 0.95,
                    'group_size': len(track_history)
                }
        
        # Check 2: Does pedestrian have mobility aid?
        has_mobility_aid, aid_type = self.detect_mobility_aid(roi)
        
        if has_mobility_aid:
            return {
                'pedestrian_type': 'assisted',
                'description': self.PEDESTRIAN_TYPES['assisted'],
                'mobility_aid': aid_type,
                'confidence': 0.85
            }
        
        # Check 3: Estimate age from appearance
        age_predictions = self.age_classifier(roi)
        
        # Decision logic
        child_prob = age_predictions.get('child_prob', 0.0)
        elderly_prob = age_predictions.get('elderly_prob', 0.0)
        
        if child_prob > 0.6:
            ped_type = 'child'
        elif elderly_prob > 0.6:
            ped_type = 'elderly'
        else:
            ped_type = 'adult'
        
        return {
            'pedestrian_type': ped_type,
            'description': self.PEDESTRIAN_TYPES[ped_type],
            'confidence': age_predictions[f'{ped_type}_prob'],
            'age_distribution': age_predictions
        }
    
    def detect_mobility_aid(self, roi) -> tuple:
        """
        Detect crutches, walkers, wheelchairs, canes
        Returns: (has_aid: bool, aid_type: str)
        """
        pass
    
    def detect_pedestrian_group(self, track_history) -> bool:
        """
        Detect if pedestrians are moving as a coordinated group
        Uses: spatial proximity + correlated velocity
        """
        pass
```

### Pedestrian Speed Validation

```python
class PedestrianSpeedValidator:
    """
    Validate pedestrian/cyclist speeds for QA
    Flags anomalies for manual review
    """
    
    SPEED_RANGES_MS = {
        'adult': (0.9, 2.0),           # m/s
        'child': (0.8, 1.8),
        'elderly': (0.5, 1.2),
        'assisted': (0.3, 1.0),
        'cyclist': (4.0, 6.0),
    }
    
    @staticmethod
    def validate_speed(pedestrian_type: str, speed_ms: float) -> tuple:
        """
        Check if speed is within expected range
        Returns: (is_valid: bool, deviation: float)
        """
        if pedestrian_type not in PedestrianSpeedValidator.SPEED_RANGES_MS:
            return True, 0.0
        
        min_speed, max_speed = PedestrianSpeedValidator.SPEED_RANGES_MS[
            pedestrian_type
        ]
        
        is_valid = min_speed <= speed_ms <= max_speed
        
        # Calculate how far outside range
        if not is_valid:
            if speed_ms < min_speed:
                deviation = min_speed - speed_ms
            else:
                deviation = speed_ms - max_speed
        else:
            deviation = 0.0
        
        return is_valid, deviation
```

---

## Multi-Standard Output

### Detection Record Structure

```python
class DetectionRecord:
    """
    Complete detection output per frame/object
    Includes both FHWA and TRL/ORN11 classifications
    """
    
    # Identification
    detection_id: int
    frame_number: int
    timestamp: float
    roundabout_id: str
    
    # Bounding box & confidence
    bbox: tuple                    # (x1, y1, x2, y2)
    confidence: float              # 0.0-1.0
    area_pixels: int
    center: tuple
    
    # YOLO base classification
    yolo_class: str
    yolo_confidence: float
    
    # FHWA Classification (detailed technical)
    fhwa_class: int                # 1-13
    fhwa_name: str
    fhwa_confidence: float
    fhwa_alternatives: list
    
    # TRL/ORN11 Classification (practical)
    trl_class: int                 # 1-8
    trl_name: str
    
    # Pedestrian Data
    is_pedestrian: bool
    pedestrian_type: Optional[str]  # adult/child/elderly/assisted/group
    pedestrian_confidence: float
    mobility_aid: Optional[str]
    
    # Cyclist Data
    is_cyclist: bool
    
    # Physical Attributes (estimated)
    vehicle_length_m: Optional[float]
    vehicle_height_m: Optional[float]
    vehicle_width_m: Optional[float]
    axle_count_estimate: Optional[int]
    tire_count_estimate: Optional[int]
    
    # Flow Properties
    pcu_value: float
    speed_kmh: Optional[float]
    heading_degrees: Optional[float]
    
    # Quality Indicators
    flagged_for_review: bool
    review_reason: Optional[str]
    quality_score: float  # 0.0-1.0
```

### JSON Export Format

```json
{
  "detection_id": 1042,
  "frame_number": 1250,
  "timestamp": 125.5,
  "roundabout_id": "Roundabout_1",
  "bbox": [640, 360, 720, 420],
  "confidence": 0.96,
  "yolo_class": "car",
  "yolo_confidence": 0.92,
  "classification": {
    "fhwa": {
      "class": 2,
      "name": "Passenger Cars",
      "confidence": 0.96,
      "alternatives": [
        {
          "class": 3,
          "name": "Other 2-Axle 4-Tire Vehicles",
          "probability": 0.03
        }
      ]
    },
    "trl_orn11": {
      "class": 2,
      "name": "Passenger Car",
      "pcu": 1.0
    }
  },
  "pedestrian": null,
  "cyclist": null,
  "physical_attributes": {
    "length_m": 4.5,
    "height_m": 1.6,
    "width_m": 1.8,
    "axle_count": 2,
    "tire_count": 4
  },
  "quality": {
    "flagged": false,
    "reason": null,
    "score": 0.96
  }
}
```

---

## Confidence & Quality Management

### Multi-Level Confidence Scoring

```python
class ConfidenceManager:
    """
    Manage confidence scores across detection pipeline
    Identify low-confidence detections for manual review
    """
    
    def compute_overall_confidence(
        self,
        yolo_confidence: float,
        classification_confidence: float,
        feature_consistency: float,
        temporal_consistency: float = 1.0
    ) -> float:
        """
        Weighted combination of confidence sources
        """
        
        weights = {
            'yolo': 0.30,            # Base detection
            'classification': 0.50,  # Vehicle/pedestrian type
            'features': 0.15,        # Feature matching
            'temporal': 0.05,        # Tracking consistency
        }
        
        overall = (
            weights['yolo'] * yolo_confidence +
            weights['classification'] * classification_confidence +
            weights['features'] * feature_consistency +
            weights['temporal'] * temporal_consistency
        )
        
        return overall
    
    def flag_for_manual_review(
        self,
        detection: DetectionRecord
    ) -> None:
        """
        Flag detections requiring human verification
        """
        
        reasons = []
        
        if detection.confidence < 0.55:
            reasons.append("Low overall confidence")
        
        if detection.fhwa_confidence < 0.60:
            reasons.append("Low classification confidence")
        
        # Check if top 2 FHWA alternatives too close
        if len(detection.fhwa_alternatives) > 0:
            gap = (detection.fhwa_confidence - 
                   detection.fhwa_alternatives[0]['probability'])
            if gap < 0.05:
                reasons.append("Ambiguous FHWA classification")
        
        if reasons:
            detection.flagged_for_review = True
            detection.review_reason = "; ".join(reasons)
```

---

## Data Export Format

### CSV Output

```csv
frame_num, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
fhwa_class, fhwa_name, trl_class, trl_name,
is_pedestrian, pedestrian_type, is_cyclist, 
confidence, pcu_value, speed_kmh, flagged, review_reason

1250, 125.5, 640, 360, 720, 420, 2, "Passenger Cars", 
2, "Passenger Car", false, null, false, 
0.96, 1.0, 35.2, false, null
```

### Database Schema

```sql
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY,
    frame_number INTEGER,
    timestamp REAL,
    roundabout_id TEXT,
    
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    
    -- FHWA Classification (detailed technical)
    fhwa_class INTEGER,
    fhwa_name TEXT,
    fhwa_confidence REAL,
    
    -- TRL/ORN11 Classification (practical)
    trl_class INTEGER,
    trl_name TEXT,
    
    -- Pedestrian/Cyclist
    is_pedestrian BOOLEAN,
    pedestrian_type TEXT,  -- adult, child, elderly, assisted, group
    pedestrian_confidence REAL,
    is_cyclist BOOLEAN,
    
    -- Quality
    confidence REAL,
    pcu_value REAL,
    flagged_for_review BOOLEAN,
    review_reason TEXT,
    
    FOREIGN KEY (frame_number) REFERENCES frames(frame_number),
    FOREIGN KEY (roundabout_id) REFERENCES roundabouts(roundabout_id)
);
```

---

## Implementation Roadmap

### Phase 1: Base Detection Infrastructure (Week 1)
- [ ] Set up YOLOv8 model loading and inference
- [ ] Implement GPU batch processing
- [ ] Create NMS and confidence filtering
- [ ] Test on sample roundabout video (first 5 min)
- **Deliverable**: Raw detections CSV

### Phase 2: Bicycle Classification (Week 2)
- [ ] Source/train bicycle binary classifier
- [ ] Implement bicycle detection logic
- [ ] Integrate into detection pipeline
- [ ] Test on mixed traffic scenarios
- **Deliverable**: Bicycle/motorcycle distinction validated

### Phase 3: FHWA Vehicle Classification (Week 3)
- [ ] Train/fine-tune CNN on FHWA 13 classes
- [ ] Implement axle/tire detection features
- [ ] Add alternative prediction ranking
- [ ] Validate on diverse truck types
- **Deliverable**: FHWA classification working

### Phase 4: TRL/ORN11 Mapping (Week 4)
- [ ] Implement FHWA → TRL mapping
- [ ] Add PCU value assignment
- [ ] Create dual-standard output
- [ ] Validate mapping consistency
- **Deliverable**: Dual standard output (FHWA + TRL)

### Phase 5: Pedestrian Classification (Week 5)
- [ ] Train age classifier (adult/child/elderly)
- [ ] Implement mobility aid detection
- [ ] Add group detection logic
- [ ] Test on crowded scenarios
- **Deliverable**: Pedestrian sub-classification working

### Phase 6: Quality & Integration (Week 6)
- [ ] Implement confidence scoring system
- [ ] Add manual review flagging
- [ ] Create CSV/JSON export
- [ ] Integrate with tracking module
- [ ] End-to-end testing
- **Deliverable**: Complete detection module ready

---

## Testing Strategy

### Unit Tests

```python
def test_bicycle_detection():
    """Verify bicycle classification accuracy >= 95%"""
    pass

def test_fhwa_classification():
    """Test FHWA 13-class classification on diverse vehicles"""
    pass

def test_fhwa_to_trl_mapping():
    """Verify correct mapping FHWA → TRL/ORN11"""
    pass

def test_pedestrian_subtype():
    """Test pedestrian sub-classification (adult/child/elderly)"""
    pass

def test_confidence_scoring():
    """Validate confidence score calculation logic"""
    pass

def test_speed_validation():
    """Test pedestrian speed validation ranges"""
    pass
```

### Integration Tests

```python
def test_end_to_end_detection():
    """Full pipeline: frame → FHWA + TRL + Pedestrian"""
    pass

def test_tracking_consistency():
    """Verify classification consistency across frames"""
    pass

def test_all_standards_export():
    """Test CSV/JSON export with both standards"""
    pass
```

---

## Performance Benchmarks (Target)

| Metric | Target | Current |
|--------|--------|---------|
| YOLO Detection | 20-30 FPS | - |
| FHWA Classification | 10-20 FPS | - |
| Pedestrian Classification | 15-25 FPS | - |
| Overall Frame Rate | 15 FPS | - |
| Detection Precision | >95% | - |
| Pedestrian Detection | >90% | - |
| Bicycle Accuracy | >95% | - |

