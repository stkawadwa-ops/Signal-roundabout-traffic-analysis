# Signal-Roundabout Traffic Analysis System

## Project Overview

A sophisticated computer vision and data analysis system designed to extract precise traffic flow parameters from Unmanned Aerial Vehicle (UAV) footage at signalized roundabouts. This system specializes in handling multi-segment video streams (due to UAV battery constraints) and video instability while maintaining continuous vehicle tracking across occlusions caused by overhead structures.

**Research Objective**: Optimize traffic flow and minimize delays at 7 signalized roundabouts by analyzing comprehensive traffic parameters classified according to international standards (TRL/AASHTO).

---

## Key Challenges Addressed

### 1. Multi-Segment Video Concatenation
- **Problem**: UAVs provide ~20-minute video segments; total footage per facility spans 4+ hours
- **Solution**: Automatic video stitching with temporal calibration and continuity verification
- **Implementation**: Frame overlap detection, timestamp reconciliation, vehicle re-identification across segments

### 2. Video Instability/Drift Compensation
- **Problem**: Drone altitude/heading variations cause frame jitter and pixel drift
- **Solution**: Real-time video stabilization using optical flow and gyroscopic-based correction
- **Implementation**: Template matching, perspective transformation, frame alignment

### 3. Vehicle Tracking with Occlusion Handling
- **Problem**: 6/7 roundabouts have overhead structures creating blind zones
- **Solution**: Kalman filter-based trajectory prediction with re-identification logic
- **Implementation**: Hungarian algorithm matching, temporal gap interpolation, vehicle fingerprinting

### 4. Traffic Classification & Parameter Extraction
- **Problem**: Multiple vehicle classes (pedestrians, cars, buses, HGVs) per international standards
- **Solution**: Deep learning classifier + rule-based parameter extraction
- **Implementation**: YOLO-based detection, CNN-based classification, geometric flow analysis

---

## Technical Architecture

```
Signal-roundabout-traffic-analysis/
│
├── README.md                              # This file
├── LICENSE
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package setup
├── config.yaml                            # Global configuration
│
├── docs/
│   ├── PROJECT_DESIGN.md                  # High-level system design
│   ├── STANDARDS_REFERENCE.md             # TRL/AASHTO classification
│   ├── DATA_PIPELINE.md                   # Data flow architecture
│   ├── VIDEO_PROCESSING.md                # Video handling procedures
│   ├── PARAMETER_DEFINITIONS.md           # All extracted parameters
│   ├── API_REFERENCE.md                   # Code API documentation
│   ├── INSTALLATION_GUIDE.md              # Setup instructions
│   ├── USER_MANUAL.md                     # How to use the system
│   └── TROUBLESHOOTING.md                 # Common issues & solutions
│
├── src/
│   ├── __init__.py
│   ├── constants.py                       # Global constants & enums
│   ├── config_loader.py                   # Configuration management
│   ├── logger_setup.py                    # Logging configuration
│   │
│   ├── video_processing/
│   │   ├── __init__.py
│   │   ├── video_reader.py                # Multi-segment video loading
│   │   ├── video_concatenator.py          # Segment stitching & temporal sync
│   │   ├── video_stabilizer.py            # Frame stabilization
│   │   ├── frame_aligner.py               # Perspective correction
│   │   ├── quality_checker.py             # Segment overlap validation
│   │   └── preprocessing.py               # Frame normalization & enhancement
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py               # YOLO v8 vehicle detection
│   │   ├── classifier.py                  # Vehicle type classification
│   │   ├── bounding_box_utils.py          # BBox manipulation & validation
│   │   ├── confidence_filter.py           # Detection filtering
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── pretrained_yolo.pt         # Pre-trained YOLO weights
│   │       └── vehicle_classifier.pth     # Pre-trained classifier
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── kalman_filter.py               # Kalman filter implementation
│   │   ├── hungarian_matcher.py           # Hungarian algorithm for matching
│   │   ├── tracker.py                     # Multi-object tracker (MOT)
│   │   ├── occlusion_handler.py           # Trajectory prediction & gap handling
│   │   ├── reidentification.py            # Vehicle re-identification logic
│   │   ├── trajectory.py                  # Trajectory data structure
│   │   ├── track_management.py            # Track lifecycle management
│   │   └── continuity_validator.py        # Cross-segment continuity checks
│   │
│   ├── parameter_extraction/
│   │   ├── __init__.py
│   │   ├── speed_calculator.py            # Velocity estimation (px/frame → km/h)
│   │   ├── volume_analyzer.py             # Traffic volume counting
│   │   ├── turning_geometry.py            # Turning radius calculation
│   │   ├── trajectory_analyzer.py         # Trajectory-based parameters
│   │   ├── flow_classifier.py             # Entry/circulatory/exit classification
│   │   ├── zone_processor.py              # Roundabout zone identification
│   │   ├── parameter_validator.py         # Sanity checking of parameters
│   │   ├── standards_mapper.py            # TRL/AASHTO classification mapping
│   │   └── metrics_aggregator.py          # Parameter aggregation & statistics
│   │
│   ├── data_management/
│   │   ├── __init__.py
│   │   ├── database_schema.py             # SQLite schema definition
│   │   ├── database_manager.py            # Database operations (CRUD)
│   │   ├── data_exporter.py               # Export to CSV/Excel/JSON
│   │   ├── data_validator.py              # Data integrity checks
│   │   ├── cache_manager.py               # Intermediate results caching
│   │   └── backup_manager.py              # Data backup utilities
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── delay_model.py                 # Delay calculation models
│   │   ├── flow_simulator.py              # Traffic flow simulation
│   │   ├── optimizer.py                   # Optimization algorithms
│   │   ├── signal_timing.py               # Signal timing recommendations
│   │   ├── sensitivity_analysis.py        # Parameter sensitivity analysis
│   │   └── scenario_builder.py            # What-if scenario generation
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── video_annotator.py             # Overlay tracks on video
│   │   ├── trajectory_plotter.py          # Plot trajectories
│   │   ├── parameter_visualizer.py        # Parameter distribution plots
│   │   ├── flow_diagram.py                # Roundabout flow diagrams
│   │   ├── heatmap_generator.py           # Spatial density heatmaps
│   │   └── report_generator.py            # PDF report generation
│   │
│   ├── quality_assurance/
│   │   ├── __init__.py
│   │   ├── tracker_validator.py           # Track accuracy validation
│   │   ├── parameter_auditor.py           # Parameter correctness checks
│   │   ├── statistical_validator.py       # Statistical anomaly detection
│   │   ├── manual_review_tool.py          # Interactive frame-by-frame review
│   │   └── error_reporter.py              # Error logging & reporting
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline_orchestrator.py       # Main processing pipeline
│   │   ├── job_manager.py                 # Batch job management
│   │   ├── progress_tracker.py            # Progress monitoring
│   │   └── error_handler.py               # Error recovery & logging
│   │
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py                    # Geometric calculations
│       ├── frame_utils.py                 # Frame manipulation utilities
│       ├── math_utils.py                  # Mathematical utilities
│       ├── file_utils.py                  # File I/O utilities
│       ├── performance_profiler.py        # Performance monitoring
│       └── time_utils.py                  # Time-related utilities
│
├── models/
│   ├── yolo/
│   │   ├── README.md
│   │   └── download_models.py             # Script to download YOLO weights
│   └── classifiers/
│       ├── README.md
│       └── train_classifier.py            # Fine-tune classifier (optional)
│
├── data/
│   ├── raw/
│   │   ├── roundabout_1/
│   │   │   ├── segment_1.mp4
│   │   │   ├── segment_2.mp4
│   │   │   └── metadata.json              # Video metadata & timestamps
│   │   ├── roundabout_2/
│   │   └── ...
│   │
│   ├── processed/
│   │   ├── roundabout_1/
│   │   │   ├── concatenated_video.mp4     # Stitched video
│   │   │   ├── tracks.json                # Extracted tracks
│   │   │   ├── parameters.csv             # Extracted parameters
��   │   │   └── diagnostics/
│   │   └── ...
│   │
│   └── analysis/
│       ├── flow_analysis.csv
│       ├── delay_analysis.csv
│       ├── optimization_results/
│       └── reports/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Exploratory data analysis
│   ├── 02_parameter_validation.ipynb      # Parameter sanity checks
│   ├── 03_flow_analysis.ipynb             # Flow pattern analysis
│   ├── 04_optimization_results.ipynb      # Optimization findings
│   └── 05_comparative_analysis.ipynb      # Cross-roundabout comparison
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                        # Pytest fixtures
│   │
│   ├── unit/
│   │   ├── test_video_processing.py
│   │   ├── test_detection.py
│   │   ├── test_tracking.py
│   │   ├── test_parameter_extraction.py
│   │   ├── test_data_management.py
│   │   └── test_utils.py
│   │
│   ├── integration/
│   │   ├── test_pipeline_end_to_end.py
│   │   ├── test_video_concatenation.py
│   │   ├── test_tracking_continuity.py
│   │   └── test_data_export.py
│   │
│   └── fixtures/
│       ├── sample_video.mp4               # Test video segment
│       ├── sample_detections.json         # Test detection data
│       └── expected_outputs.json          # Expected results
│
├── scripts/
│   ├── run_full_pipeline.py               # Main execution script
│   ├── process_single_roundabout.py       # Single facility processing
│   ├── validate_video_segments.py         # Pre-processing validation
│   ├── inspect_tracks.py                  # Interactive track inspection
│   ├── generate_report.py                 # Generate analysis reports
│   ├── batch_process.py                   # Batch processing multiple facilities
│   ├── optimize_signal_timing.py          # Run optimization algorithms
│   ├── download_models.py                 # Download pre-trained models
│   └── cli.py                             # Command-line interface
│
├── configs/
│   ├── default.yaml                       # Default configuration
│   ├── production.yaml                    # Production settings
│   ├── development.yaml                   # Development settings
│   ├── roundabout_1.yaml                  # Facility-specific configs
│   ├── roundabout_2.yaml
│   └── ...
│
├── logs/
│   └── .gitkeep                           # Directory for runtime logs
│
├── .github/
│   ├── workflows/
│   │   ├── tests.yml                      # Automated testing
│   │   ├── code_quality.yml               # Linting & formatting
│   │   └── documentation.yml              # Build documentation
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── .gitignore
├── CONTRIBUTING.md                        # Contribution guidelines
├── CHANGELOG.md                           # Version history
└── environment.yml                        # Conda environment specification

```

---

## Core Modules Explained

### 1. **Video Processing** (`src/video_processing/`)
- **Purpose**: Handle multi-segment UAV footage with instability
- **Key Functions**:
  - `video_concatenator.py`: Align timestamps across segments, detect overlaps
  - `video_stabilizer.py`: Compensate for drone drift using optical flow
  - `quality_checker.py`: Verify concatenation integrity

### 2. **Detection** (`src/detection/`)
- **Purpose**: Identify and classify all vehicles per TRL/AASHTO standards
- **Key Functions**:
  - `yolo_detector.py`: Real-time vehicle detection
  - `classifier.py`: Categorize vehicles (pedestrian, car, bus, HGV, etc.)
  - `confidence_filter.py`: Remove low-confidence detections

### 3. **Tracking** (`src/tracking/`)
- **Purpose**: Maintain continuous vehicle trajectories across occlusions
- **Key Functions**:
  - `kalman_filter.py`: Predict vehicle motion through blind zones
  - `hungarian_matcher.py`: Associate detections across frames
  - `occlusion_handler.py`: Interpolate missing trajectories under overpasses
  - `reidentification.py`: Match vehicles exiting occlusion zones

### 4. **Parameter Extraction** (`src/parameter_extraction/`)
- **Purpose**: Calculate all traffic flow metrics
- **Key Functions**:
  - `speed_calculator.py`: Convert pixel coordinates to real-world speeds
  - `turning_geometry.py`: Compute turning radii per AASHTO guidelines
  - `flow_classifier.py`: Classify vehicles as entry/circulatory/exit
  - `standards_mapper.py`: Map to TRL/AASHTO classes

### 5. **Data Management** (`src/data_management/`)
- **Purpose**: Store and retrieve processed data reliably
- **Key Functions**:
  - SQLite database with schema for tracks, parameters, metadata
  - Export to CSV/Excel for analysis
  - Data validation and integrity checks

### 6. **Optimization** (`src/optimization/`)
- **Purpose**: Analyze delay and recommend improvements
- **Key Functions**:
  - `delay_model.py`: Calculate vehicle delays
  - `optimizer.py`: Recommend signal timing changes
  - `scenario_builder.py`: Generate what-if scenarios

### 7. **Visualization** (`src/visualization/`)
- **Purpose**: Generate outputs for verification and reporting
- **Key Functions**:
  - Annotated videos with tracked vehicles
  - Flow heatmaps and trajectory plots
  - PDF reports with findings

### 8. **Quality Assurance** (`src/quality_assurance/`)
- **Purpose**: Validate all results before analysis
- **Key Functions**:
  - `tracker_validator.py`: Verify tracking accuracy
  - `manual_review_tool.py`: Interactive verification UI

---

## Data Flow Architecture

```
Raw UAV Videos (4 hours per roundabout)
        ↓
[Video Preprocessing]
  ├─ Segment loading
  ├─ Temporal synchronization
  ├─ Stability correction
  └─ Concatenation validation
        ↓
Stabilized Continuous Video
        ↓
[Vehicle Detection]
  ├─ YOLO detection (per frame)
  ├─ Confidence filtering
  └─ Bounding box normalization
        ↓
Detections (frame-by-frame)
        ↓
[Multi-Object Tracking]
  ├─ Hungarian matching
  ├─ Kalman filtering
  ├─ Occlusion prediction
  └─ Re-identification
        ↓
Continuous Tracks (with metadata)
        ↓
[Parameter Extraction]
  ├─ Speed calculation
  ├─ Volume counting
  ├─ Turning radius computation
  ├─ Zone classification
  └─ Standards mapping
        ↓
Raw Parameters
        ↓
[Quality Assurance]
  ├─ Sanity validation
  ├─ Statistical checks
  ├─ Manual review flagging
  └─ Error reporting
        ↓
Validated Parameters
        ↓
[Data Storage]
  ├─ SQLite database
  ├─ CSV export
  └─ Backup creation
        ↓
[Analysis & Optimization]
  ├─ Delay calculation
  ├─ Flow simulation
  ├─ Signal timing optimization
  └─ Report generation
        ↓
Final Outputs
  ├─ Annotated videos
  ├─ Parameter tables
  ├─ Flow diagrams
  ├─ Optimization recommendations
  └─ PDF reports
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for real-time processing)
- 8GB+ RAM
- 200GB+ storage for raw videos

### Quick Installation

```bash
# Clone repository
git clone https://github.com/stkawadwa-ops/Signal-roundabout-traffic-analysis.git
cd Signal-roundabout-traffic-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run CPU-safe smoke pipeline (no model download required)
python scripts/run_pipeline.py --smoke --device cpu --max-frames 20 --output-dir ./data/processed/smoke_run

# Run smoke test
pytest tests/test_smoke_pipeline.py -q
```

### Process Your First Roundabout

```bash
python scripts/run_pipeline.py \
    --input-video path/to/video.mp4 \
    --device auto \
    --max-frames 300 \
    --output-dir ./data/processed/run_01
```

---

## Key Features

✅ **Robust Video Handling**
- Multi-segment concatenation with overlap detection
- Drone instability compensation
- Temporal continuity validation

✅ **Advanced Tracking**
- Kalman filter-based prediction
- Occlusion handling with trajectory interpolation
- Vehicle re-identification across blind zones

✅ **Comprehensive Parameters**
- Speed (entry, circulatory, exit)
- Volume (per vehicle class)
- Turning radii
- Classification per TRL/AASHTO standards

✅ **Quality Assurance**
- Automatic validation checks
- Manual review tools
- Statistical anomaly detection

✅ **Analysis & Optimization**
- Delay quantification
- Flow simulation
- Signal timing recommendations

---

## Configuration

All processing controlled via YAML files in `configs/`:

```yaml
video:
  overlap_tolerance_frames: 30
  stabilization_method: "optical_flow"
  
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.45
  
tracking:
  kalman_process_noise: 0.01
  kalman_measurement_noise: 4.0
  occlusion_timeout_frames: 120
  
parameters:
  speed_unit: "kmh"
  pixel_to_meter_ratio: 0.05  # Calibrate per roundabout
  
database:
  type: "sqlite"
  path: "./data/processed/roundabout_1/database.db"
```

---

## Project Status

🔄 **In Development** - Active implementation phase

---

## Documentation

Comprehensive guides available in `docs/`:
- **PROJECT_DESIGN.md**: System architecture deep-dive
- **STANDARDS_REFERENCE.md**: TRL/AASHTO classification details
- **VIDEO_PROCESSING.md**: Video handling specifics
- **INSTALLATION_GUIDE.md**: Detailed setup instructions
- **USER_MANUAL.md**: Step-by-step usage guide

---

## Support & Issues

If you encounter issues:
1. Check `docs/TROUBLESHOOTING.md`
2. Review test cases in `tests/`
3. Inspect logs in `logs/`
4. Create issue with error traceback

---

## License

[Add your license here]

---

## Contact

**Project Lead**: stkawadwa-ops

---

**Last Updated**: June 6, 2026
