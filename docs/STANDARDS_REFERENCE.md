# Vehicle Classification Standards Reference

**Signal-Roundabout Traffic Analysis System**  
*Comprehensive Classification Framework for Detection & Parameter Extraction*

---

## Table of Contents
1. [Overview](#overview)
2. [FHWA 13-Class System](#fhwa-13-class-system)
3. [TRL Classification Framework (ORN 11)](#trl-classification-framework-orn-11)
4. [Pedestrian Classification](#pedestrian-classification)
5. [Unified Classification Matrix](#unified-classification-matrix)
6. [Detection Hierarchy](#detection-hierarchy)
7. [PCU Values for Flow Analysis](#pcu-values-for-flow-analysis)

---

## Overview

This project requires classification across **two major international standards** plus **pedestrian detection**:

- **FHWA (Federal Highway Administration)**: 13-class axle-based system (US standard - detailed technical classification)
- **TRL (Transport Research Laboratory) / ORN 11**: Practical urban classification framework (UK/International - simplified for practical application)
- **Pedestrians & Cyclists**: Non-motorized road users (mandatory for roundabout analysis)

The system will detect and classify **all road users**, not just vehicles, providing output in both standards simultaneously with consistent pedestrian classification.

---

## FHWA 13-Class System

**Standard**: Federal Highway Administration (United States)  
**Basis**: Axle configuration and wheel count  
**Application**: Traffic monitoring, capacity analysis, safety studies  
**Use Case**: Detailed technical traffic analysis, capacity modeling

### Class Definitions

| Class | Name | Description | Axles | Tires | Detection Features |
|-------|------|-------------|-------|-------|-------------------|
| **1** | Motorcycles | Motorcycles, scooters, mopeds | 2 | 2 | Small profile, single track |
| **2** | Passenger Cars | Sedans, coupes, station wagons, hatchbacks | 2 | 4 | Standard car shape, 4 tires |
| **3** | Other 2-Axle, 4-Tire Vehicles | Pickups, vans, campers (non-passenger) | 2 | 4 | Van/SUV shape, similar to car but higher |
| **4** | Buses | Vehicles designed for 10+ persons | 2-3 | 6+ | Long profile, passenger windows |
| **5** | 2-Axle, 6-Tire Single-Unit Trucks | Pickup trucks with dual rear wheels | 2 | 6 | Dual rear tires visible |
| **6** | 3-Axle Single-Unit Trucks | Single trucks with 3+ axles, no trailer | 3 | 8+ | 3-wheel axle configuration |
| **7** | 4+ Axle Single-Unit Trucks | Single trucks with 4+ axles | 4+ | 10+ | Extended truck bed |
| **8** | 4- or 5-Axle Single-Trailer Trucks | Tractor + single trailer (≤5 axles) | 4-5 | 10-12 | Tractor + trailer visible |
| **9** | 5-Axle Single-Trailer Trucks | Classic "18-wheeler" tractor + trailer | 5 | 10 | Long tractor-trailer combination |
| **10** | 6+ Axle Single-Trailer Trucks | Single trailer with >5 axles | 6+ | 12+ | Extended trailer |
| **11** | 5- or Fewer Axle Multi-Trailer Trucks | Tractor towing 2+ trailers (≤5 axles) | 5 | 10 | Tandem trailer configuration |
| **12** | 6-Axle Multi-Trailer Trucks | Multi-trailer with exactly 6 axles | 6 | 12 | Double/triple trailer |
| **13** | 7+ Axle Multi-Trailer Trucks | Multi-trailer with 7+ axles | 7+ | 14+ | Extended multi-trailer |

### FHWA Groupings for Analysis

**Light Vehicles**: Classes 1-3  
**Medium Vehicles**: Classes 4-5  
**Heavy Vehicles**: Classes 6-13

---

## TRL Classification Framework (ORN 11)

**Standard**: Transport Research Laboratory (UK)  
**Document**: ORN 11 - Urban Road Traffic Surveys  
**Basis**: Practical urban classification for manual and automated surveys  
**Application**: Urban traffic analysis, practical implementation, international developing countries

### Primary Classification (ORN 11 Standard)

| Class | Name | Description | Typical Vehicle | TRL Research Class | PCU |
|-------|------|-------------|------------------|-------------------|-----|
| **1** | Motorcycle | Motorcycles, scooters, mopeds, three-wheelers | Honda motorcycle, Vespa, auto-rickshaw | Motorcycles | 0.5-0.75 |
| **2** | Passenger Car | Sedans, hatchbacks, station wagons | Toyota Camry, Ford Focus | Passenger Cars | 1.0 |
| **3** | Light Goods Vehicle (LGV) | Small vans, pickup trucks (≤3.5 ton) | Ford Transit, Nissan Pickup | Vans/Light Commercial | 1.0-1.5 |
| **4** | Medium Goods Vehicle (MGV) | Medium vans/trucks (3.5-7 ton) | MAN TGL, truck with 2-3 axles | Medium Commercial | 1.5-2.0 |
| **5** | Heavy Goods Vehicle (HGV) | Trucks/lorries, articulated (>7 ton) | Scania articulated, Mercedes Actros | Heavy Commercial | 2.0-3.5 |
| **6** | Bus | Buses, minibuses (10+ persons) | Standard transit bus, articulated bus | Public Transport | 2.0-3.0 |
| **7** | Bicycle/Cycle | Pedal bicycles, rickshaws | Standard bicycle, cargo bike | Non-Motorized | 0.5 |
| **8** | Auto Rickshaw/3-Wheeler | Three-wheeled motorized taxis | Tuk-tuk, auto-rickshaw, Bajaj | Special Category | 0.75 |

### ORN 11 Simplified Groupings for Flow Analysis

| Grouping | Classes | Description | Use Case |
|----------|---------|-------------|----------|
| **Non-Motorized** | Bicycle | Pedal-powered vehicles | Separate cycling flows |
| **Motorcycles** | Motorcycle, Auto Rickshaw | Two/three-wheeled motorized | Mixed traffic analysis |
| **Cars** | Passenger Car, LGV | Personal and light commercial | Dominant flow category |
| **Large Vehicles** | MGV, HGV, Bus | Commercial transport | Capacity-determining flows |

---

## Pedestrian Classification

**Critical Component**: Pedestrians are **mandatory** in roundabout analysis

### Pedestrian Sub-Categories

| Type | Description | Detection Criteria | Priority | Speed Range (m/s) |
|------|-------------|-------------------|----------|-------------------|
| **Adult Pedestrian** | Walking adult (15-65 years) | Bipedal motion, normal speed (1-2 m/s) | High | 0.9-2.0 |
| **Child Pedestrian** | Walking child (<15 years) | Smaller body size, variable speed | **Critical** | 0.8-1.8 |
| **Elderly Pedestrian** | Older adult (65+ years) | Slower speed, sometimes assisted | High | 0.5-1.2 |
| **Assisted Pedestrian** | With crutches, walker, wheelchair | Support device visible, slow speed | High | 0.3-1.0 |
| **Pedestrian Group** | 2+ pedestrians moving together | Multiple bounding boxes, correlated motion | Medium | 0.8-1.8 |
| **Cyclist** | Cyclist on bicycle | Human + bicycle frame, pedal motion | High | 4.0-6.0 |

### Pedestrian Detection Challenges
- Small bounding boxes (harder to detect at high altitude UAV footage)
- Rapid motion variations
- Occlusion by vehicles
- Variable clothing and appearance
- Groups moving as units
- Crossings may be obscured by overpass structures

### Pedestrian Speed Validation Ranges
- **Normal Walking**: 1.0-1.5 m/s (3.6-5.4 km/h)
- **Fast Walking**: 1.5-2.0 m/s (5.4-7.2 km/h)
- **Slow/Elderly**: 0.5-1.0 m/s (1.8-3.6 km/h)
- **Cyclist**: 4-6 m/s (14.4-21.6 km/h)

---

## Unified Classification Matrix

### System Output Format

The system will simultaneously output **both standards** for each detected vehicle:

```
Detection Record:
{
  "detection_id": 1042,
  "frame_number": 1250,
  "timestamp": 125.5,
  "bounding_box": [640, 360, 720, 420],
  "confidence": 0.98,
  
  // Primary classification (FHWA - most detailed)
  "fhwa_class": 2,
  "fhwa_name": "Passenger Cars",
  
  // Secondary classification (TRL/ORN11 - practical)
  "trl_class": 2,  // ORN 11 code
  "trl_name": "Passenger Car",
  
  // Additional attributes
  "vehicle_length_m": 4.5,
  "vehicle_height_m": 1.6,
  "axle_count": 2,
  "tire_count": 4,
  "pedestrian_subcategory": null,
  
  // Detailed properties for parameter extraction
  "is_pedestrian": false,
  "is_bicycle": false,
  "is_motorcycle": false,
  "is_bus": false,
  "is_truck": false,
  
  // Color and appearance
  "primary_color": "white",
  "window_count": 4,
  "cargo_visible": false
}
```

### Cross-Standard Mapping Table

| FHWA | FHWA Name | TRL/ORN11 | TRL Name | PCU | Roundabout Impact |
|------|-----------|----------|----------|-----|-------------------|
| 1 | Motorcycles | 1 | Motorcycle | 0.5-0.75 | Low - nimble |
| 2 | Passenger Cars | 2 | Passenger Car | 1.0 | Reference |
| 3 | Other 2-Axle 4-Tire | 3 | Light Goods Vehicle | 1.0-1.5 | Moderate |
| 4 | Buses | 6 | Bus | 2.0-3.0 | High - capacity |
| 5 | 2-Axle 6-Tire Trucks | 3 | LGV / 4 | MGV | 1.5-2.0 | Moderate |
| 6 | 3-Axle Single-Unit | 4 | Medium Goods Vehicle | 1.5-2.0 | Moderate-High |
| 7 | 4+ Axle Single-Unit | 5 | Heavy Goods Vehicle | 2.0-2.5 | **Critical** |
| 8-13 | Multi-Axle/Multi-Trailer | 5 | Heavy Goods Vehicle | 2.5-3.5 | **Critical** |
| - | - | 7 | Bicycle | 0.5 | Special handling |
| - | - | 8 | Auto Rickshaw | 0.75 | Local context |
| - | Pedestrian | - | Pedestrian | 0.1-0.15 | Safety priority |

---

## Detection Hierarchy

### System Detection Strategy

The system uses a **hierarchical detection approach** to handle overlapping categories:

```
┌─────────────────────────────────────┐
│  Raw Image/Frame (UAV Footage)      │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────┐
        │ YOLO v8     │ Pre-trained detector
        │ Detection   │ (vehicle + person)
        └──────┬──────┘
               │
    ┌──────────┴───────────┐
    │                      │
┌───▼──────┐        ┌──────▼───────┐
│ Person   │        │ Vehicle      │
│ Detected │        │ Detected     │
└───┬──────┘        └──────┬───────┘
    │                      │
    │          ┌───────────┼──────────────┐
    │          │           │              │
    │    ┌─────▼──┐   ┌────▼───┐   ┌────▼───┐
    │    │Bicycle │   │Motorcycle│  │Car/Van │
    │    │Check   │   │/Scooter  │  │/Truck  │
    │    └─────┬──┘   │Check     │  │Check   │
    │          │      └────┬────┘  └────┬───┘
    │          │           │            │
    │    ┌─────▼──────────┴────────────▼──┐
    │    │  Vehicle Type CNN Classifier   │
    │    │  (FHWA 13 classes)             │
    │    └─────┬──────────────────────┬──┘
    │          │                      │
    │    ┌─────▼──┐       ┌──────────▼──┐
    │    │Specific│       │Specific     │
    │    │FHWA    │       │FHWA Class   │
    │    │Class   │       │Result       │
    │    └────┬───┘       └──────┬──────┘
    │         │                  │
    │         └──────┬───────────┘
    │                │
    │         ┌──────▼──────┐
    │         │ Map to TRL  │
    │         │ (ORN 11)    │
    │         └──────┬──────┘
    │                │
    │         ┌──────▼──────────┐
    │         │Vehicle Result   │
    │         │(FHWA+TRL)       │
    │         └─────────────────┘
    │
    └──────────┬─────────────────────────────┐
               │                             │
            ┌──▼──┐                    ┌────▼───┐
            │Pedestrian               │ Bicycle│
            │Sub-Classifier           │(Processed)
            │(Adult/Child/Elderly/    │
            │ Assisted/Group)         │
            └──┬──┘                    └────────┘
               │
            ┌──▼──────────┐
            │Pedestrian   │
            │Record       │
            │(classified) │
            └─────────────┘
                   
            ↓ (All Results Combined)
            
     ┌──────────────────────────────┐
     │ Frame-Level Detection Output │
     │ (Pedestrians + Vehicles)     │
     │ Both Standards Mapped        │
     │ (FHWA + TRL/ORN11)           │
     └──────────────────────────────┘
```

---

## PCU Values for Flow Analysis

**Passenger Car Unit (PCU)**: Standardized measure of vehicle impact on traffic flow

### PCU Conversion Table (Used for delay analysis)

| Vehicle Type | FHWA/TRL Class | PCU Value | Basis | Roundabout Usage |
|--------------|----------|-----------|-------|-------------------|
| Pedestrian | - | 0.1-0.15 | Foot-space equivalent | Crossing analysis |
| Bicycle | 7 (TRL) | 0.5 | Space requirement | Cycle flow |
| Motorcycle | 1 (FHWA) / 1 (TRL) | 0.5-0.75 | Space + performance | Mixed traffic |
| Compact Car | 2 (FHWA) / 2 (TRL) | 0.9 | Small footprint | Urban | 
| Standard Car | 2 (FHWA) / 2 (TRL) | 1.0 | Reference vehicle | Baseline |
| Large Car/SUV | 2 (FHWA) / 2 (TRL) | 1.2 | Large footprint | Space-intensive |
| LGV/Van | 3 (FHWA) / 3 (TRL) | 1.2-1.5 | Height + size | More space |
| Bus | 4 (FHWA) / 6 (TRL) | 2.5-3.0 | Large, slow | Capacity-based |
| Medium Truck | 6 (FHWA) / 4 (TRL) | 1.8-2.2 | Size + mass | Performance impact |
| Heavy Truck/HGV | 7-10 (FHWA) / 5 (TRL) | 2.5-3.5 | Large, slow | Major impact |
| Articulated Truck | 8-13 (FHWA) / 5 (TRL) | 3.0-3.5 | Complex movement | **Roundabout critical** |
| Auto Rickshaw | 8 (TRL) | 0.75 | Context-specific | Regional |

### Flow Equivalency Example

```
100 vehicles consisting of:
- 80 cars @ 1.0 PCU = 80 PCU
- 15 motorcycles @ 0.6 PCU = 9 PCU
- 4 buses @ 2.8 PCU = 11.2 PCU
- 1 HGV @ 3.0 PCU = 3.0 PCU

Total Flow = 103.2 PCU ≈ equivalent to 103 cars in terms of flow impact
```

---

## Implementation Strategy

### Phase 1: Detection Architecture

```python
# Pseudo-code for detection pipeline

class UnifiedVehicleDetector:
    """
    Detects and classifies ALL road users using FHWA + TRL/ORN11 standards
    """
    
    def __init__(self):
        self.yolo_detector = YOLOv8('person-vehicle')  # Base detection
        self.vehicle_classifier = CNNClassifier()      # 13-class FHWA
        self.pedestrian_classifier = CNNClassifier()   # Adult/Child/Elderly
        self.bicycle_detector = SpecializedDetector()  # Bicycle detection
        
    def detect_and_classify(self, frame):
        """
        Single frame processing - outputs both FHWA and TRL
        """
        # Step 1: Base detection
        detections = self.yolo_detector.detect(frame)
        
        # Step 2: Separate persons and vehicles
        persons = [d for d in detections if d.class == 'person']
        vehicles = [d for d in detections if d.class == 'vehicle']
        
        results = []
        
        # Step 3: Classify persons
        for person_det in persons:
            pedestrian_type = self.pedestrian_classifier(
                frame, person_det.bbox
            )
            result = create_pedestrian_record(person_det, pedestrian_type)
            results.append(result)
        
        # Step 4: Classify vehicles (FHWA 13 classes)
        for vehicle_det in vehicles:
            # Check for bicycle first
            is_bicycle = self.bicycle_detector(frame, vehicle_det.bbox)
            
            if is_bicycle:
                result = create_bicycle_record(vehicle_det)
            else:
                # Run FHWA vehicle classifier
                fhwa_class = self.vehicle_classifier(
                    frame, vehicle_det.bbox
                )
                # Map to TRL/ORN11
                trl_class = self.map_fhwa_to_trl(fhwa_class)
                
                result = create_vehicle_record(
                    vehicle_det, 
                    fhwa_class,
                    trl_class
                )
            
            results.append(result)
        
        return results
```

### Phase 2: Standard Mapping

```python
class ClassificationMapper:
    """
    Map FHWA classification to TRL/ORN11
    """
    
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
    
    FHWA_TO_PCU = {
        1: 0.6,
        2: 1.0,
        3: 1.2,
        4: 2.8,
        5: 1.5,
        6: 2.0,
        7: 2.5,
        8: 3.0,
        9: 3.0,
        10: 3.5,
        11: 3.0,
        12: 3.2,
        13: 3.5,
    }
    
    @staticmethod
    def map_to_trl(fhwa_class: int) -> int:
        return ClassificationMapper.FHWA_TO_TRL.get(fhwa_class, 5)
    
    @staticmethod
    def get_pcu(fhwa_class: int) -> float:
        return ClassificationMapper.FHWA_TO_PCU.get(fhwa_class, 1.5)
```

---

## Quality Validation

### Per-Standard Validation Rules

| Standard | Validation Rule | Action |
|----------|-----------------|--------|
| **FHWA** | Axle count matches class | Flag if mismatch |
| **TRL/ORN11** | Vehicle size reasonable for category | Apply constraints |
| **Pedestrian** | Speed 0.5-2.0 m/s (or cycling 4-6 m/s) | Flag if outside range |
| **Consistency** | FHWA class maps consistently to TRL | Log warning if unusual |

---

## Data Export Schema

### CSV Output Format

```csv
track_id, frame_start, frame_end, fhwa_class, fhwa_name, trl_class, 
trl_name, pedestrian_type, speed_kmh, length_m, pcu_value, confidence
```

Example row:
```csv
1042, 1200, 1450, 2, "Passenger Cars", 2, "Passenger Car", 
null, 35.2, 4.5, 1.0, 0.96
```

### Database Schema (Simplified)

```sql
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY,
    frame_number INTEGER,
    timestamp REAL,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    
    -- FHWA Classification (detailed)
    fhwa_class INTEGER,
    fhwa_name TEXT,
    fhwa_confidence REAL,
    
    -- TRL/ORN11 Classification (practical)
    trl_class INTEGER,
    trl_name TEXT,
    
    -- Pedestrian/Cyclist
    is_pedestrian BOOLEAN,
    pedestrian_type TEXT,  -- adult, child, elderly, assisted, group
    is_cyclist BOOLEAN,
    
    -- Confidence & Quality
    confidence REAL,
    pcu_value REAL,
    flagged_for_review BOOLEAN,
    review_reason TEXT,
    
    FOREIGN KEY (frame_number) REFERENCES frames(frame_number)
);
```

---

## References

- **FHWA Traffic Monitoring Guide**: https://www.fhwa.dot.gov/policyinformation/tmguide/tmg.pdf
- **ORN 11 (TRL) Urban Road Traffic Surveys**: UK Department for International Development
- **TRL Manual Classified Counts**: Transport Research Laboratory, UK

