"""
Configuration Loader for Signal Roundabout Traffic Analysis Pipeline.

Loads production.yaml (or any facility-specific YAML) and converts it
into typed dataclass objects used by PipelineOrchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed config sections
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    stabilization_method: str = "optical_flow"
    max_features: int = 200
    feature_quality: float = 0.01
    min_distance: float = 30.0
    block_size: int = 3
    win_size: Tuple[int, int] = (15, 15)
    max_level: int = 2
    ransac_threshold: float = 5.0
    ransac_method: str = "RANSAC"
    # Set to False to skip stabilization entirely (e.g. already-stabilised footage)
    enable_stabilization: bool = True


@dataclass
class DetectionConfig:
    # Model options: yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
    # Use a smaller model (yolov8n) for fast CPU testing; switch to yolov8x for production GPU
    model_name: str = "yolov8n"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    use_sahi: bool = False          # Disable by default — enable on GPU for small objects
    slice_height: int = 512
    slice_width: int = 512
    overlap_height_ratio: float = 0.1
    overlap_width_ratio: float = 0.1
    device: str = "cpu"             # "cpu" or "cuda:0"
    batch_size: int = 1
    augment: bool = False
    enable_p2_features: bool = True


@dataclass
class TrackingConfig:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    max_age: int = 120
    min_hits: int = 3
    use_appearance: bool = False
    appearance_threshold: float = 0.7
    occlusion_timeout: int = 60
    min_track_length: int = 5
    frame_rate: int = 30
    img_size: List[int] = field(default_factory=lambda: [1920, 1080])


@dataclass
class GeometryConfig:
    """
    Roundabout geometric parameters — MUST be calibrated per facility.

    roundabout_center : pixel coordinates of the roundabout centre
    outer_radius      : pixel distance from centre to outer kerb
    inner_radius      : pixel distance from centre to inner island
    entry_zones       : list of polygon point-lists (each [[x,y], ...], min 3 points)
    exit_zones        : list of polygon point-lists (each [[x,y], ...], min 3 points)
    pixel_to_meter_ratio : metres per pixel — calibrate from known ground distance
    """
    roundabout_center: List[float] = field(default_factory=lambda: [960.0, 540.0])
    outer_radius: float = 200.0
    inner_radius: float = 80.0
    entry_zones: List[List[List[float]]] = field(default_factory=list)
    exit_zones: List[List[List[float]]] = field(default_factory=list)
    pixel_to_meter_ratio: float = 0.05


@dataclass
class ParameterExtractionConfig:
    speed_unit: str = "kmh"
    speed_filter_window: int = 5
    speed_outlier_threshold: float = 2.0
    # Upper bound for speed clipping — values above this are treated as noise
    max_speed_kmh: float = 200.0
    vehicle_classes: List[str] = field(
        default_factory=lambda: ["pedestrian", "car", "bus", "truck", "motorcycle"]
    )
    circle_fit_min_points: int = 10
    curvature_window: int = 5


@dataclass
class DatabaseConfig:
    type: str = "sqlite"
    path: str = "./data/processed/{facility_id}/database.db"
    batch_size: int = 1000
    commit_interval: int = 10000


@dataclass
class OutputConfig:
    export_formats: List[str] = field(default_factory=lambda: ["csv", "json"])
    generate_annotated_video: bool = False
    annotate_detections: bool = True
    annotate_tracks: bool = True
    annotate_zones: bool = True
    generate_report: bool = False
    output_dir: str = "./data/analysis/{facility_id}"
    tracks_file: str = "tracks.json"
    parameters_file: str = "parameters.csv"
    video_output: str = "annotated_output.mp4"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "./logs/{facility_id}_{timestamp}.log"
    log_detections: bool = False
    log_tracks: bool = True


@dataclass
class PipelineConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    parameter_extraction: ParameterExtractionConfig = field(
        default_factory=ParameterExtractionConfig
    )
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Load and validate YAML configuration into a PipelineConfig."""

    @staticmethod
    def load(config_path: str) -> PipelineConfig:
        """Load config from YAML file.  Returns defaults if file is missing."""
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed.  Run: pip install pyyaml")
            return PipelineConfig()

        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found at {path}; using defaults.")
            return PipelineConfig()

        with path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        config = ConfigLoader._parse(raw)
        logger.info(f"Config loaded from {path}")
        return config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_db_path(config: PipelineConfig, facility_id: str) -> str:
        return config.database.path.replace("{facility_id}", facility_id)

    @staticmethod
    def get_output_dir(config: PipelineConfig, facility_id: str) -> str:
        return config.output.output_dir.replace("{facility_id}", facility_id)

    @staticmethod
    def get_log_file(config: PipelineConfig, facility_id: str, timestamp: str) -> str:
        return (
            config.logging.log_file
            .replace("{facility_id}", facility_id)
            .replace("{timestamp}", timestamp)
        )

    # ------------------------------------------------------------------
    # Internal parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: Dict[str, Any]) -> PipelineConfig:  # noqa: C901
        cfg = PipelineConfig()

        if "video" in raw:
            v = raw["video"]
            cfg.video = VideoConfig(
                stabilization_method=v.get("stabilization_method", "optical_flow"),
                max_features=int(v.get("max_features", 200)),
                feature_quality=float(v.get("feature_quality", 0.01)),
                min_distance=float(v.get("min_distance", 30.0)),
                block_size=int(v.get("block_size", 3)),
                win_size=tuple(v.get("win_size", [15, 15])),
                max_level=int(v.get("max_level", 2)),
                ransac_threshold=float(v.get("ransac_threshold", 5.0)),
                ransac_method=v.get("ransac_method", "RANSAC"),
                enable_stabilization=bool(v.get("enable_stabilization", True)),
            )

        if "detection" in raw:
            d = raw["detection"]
            cfg.detection = DetectionConfig(
                model_name=d.get("model_name", "yolov8n"),
                confidence_threshold=float(d.get("confidence_threshold", 0.5)),
                nms_threshold=float(d.get("nms_threshold", 0.45)),
                use_sahi=bool(d.get("use_sahi", False)),
                slice_height=int(d.get("slice_height", 512)),
                slice_width=int(d.get("slice_width", 512)),
                overlap_height_ratio=float(d.get("overlap_height_ratio", 0.1)),
                overlap_width_ratio=float(d.get("overlap_width_ratio", 0.1)),
                device=d.get("device", "cpu"),
                batch_size=int(d.get("batch_size", 1)),
                augment=bool(d.get("augment", False)),
                enable_p2_features=bool(d.get("enable_p2_features", True)),
            )

        if "tracking" in raw:
            t = raw["tracking"]
            cfg.tracking = TrackingConfig(
                track_thresh=float(t.get("track_thresh", 0.5)),
                track_buffer=int(t.get("track_buffer", 30)),
                match_thresh=float(t.get("match_thresh", 0.8)),
                max_age=int(t.get("max_age", 120)),
                min_hits=int(t.get("min_hits", 3)),
                use_appearance=bool(t.get("use_appearance", False)),
                appearance_threshold=float(t.get("appearance_threshold", 0.7)),
                occlusion_timeout=int(t.get("occlusion_timeout", 60)),
                min_track_length=int(t.get("min_track_length", 5)),
                frame_rate=int(t.get("frame_rate", 30)),
                img_size=list(t.get("img_size", [1920, 1080])),
            )

        if "geometry" in raw:
            g = raw["geometry"]
            cfg.geometry = GeometryConfig(
                roundabout_center=[float(x) for x in g.get("roundabout_center", [960.0, 540.0])],
                outer_radius=float(g.get("outer_radius", 200.0)),
                inner_radius=float(g.get("inner_radius", 80.0)),
                entry_zones=g.get("entry_zones", []),
                exit_zones=g.get("exit_zones", []),
                pixel_to_meter_ratio=float(g.get("pixel_to_meter_ratio", 0.05)),
            )

        if "parameter_extraction" in raw:
            p = raw["parameter_extraction"]
            cfg.parameter_extraction = ParameterExtractionConfig(
                speed_unit=p.get("speed_unit", "kmh"),
                speed_filter_window=int(p.get("speed_filter_window", 5)),
                speed_outlier_threshold=float(p.get("speed_outlier_threshold", 2.0)),
                max_speed_kmh=float(p.get("max_speed_kmh", 200.0)),
                vehicle_classes=p.get(
                    "vehicle_classes", ["pedestrian", "car", "bus", "truck", "motorcycle"]
                ),
                circle_fit_min_points=int(p.get("circle_fit_min_points", 10)),
                curvature_window=int(p.get("curvature_window", 5)),
            )

        if "database" in raw:
            db = raw["database"]
            cfg.database = DatabaseConfig(
                type=db.get("type", "sqlite"),
                path=db.get("path", "./data/processed/{facility_id}/database.db"),
                batch_size=int(db.get("batch_size", 1000)),
                commit_interval=int(db.get("commit_interval", 10000)),
            )

        if "output" in raw:
            o = raw["output"]
            cfg.output = OutputConfig(
                export_formats=o.get("export_formats", ["csv", "json"]),
                generate_annotated_video=bool(o.get("generate_annotated_video", False)),
                annotate_detections=bool(o.get("annotate_detections", True)),
                annotate_tracks=bool(o.get("annotate_tracks", True)),
                annotate_zones=bool(o.get("annotate_zones", True)),
                generate_report=bool(o.get("generate_report", False)),
                output_dir=o.get("output_dir", "./data/analysis/{facility_id}"),
                tracks_file=o.get("tracks_file", "tracks.json"),
                parameters_file=o.get("parameters_file", "parameters.csv"),
                video_output=o.get("video_output", "annotated_output.mp4"),
            )

        if "logging" in raw:
            lg = raw["logging"]
            cfg.logging = LoggingConfig(
                level=lg.get("level", "INFO"),
                format=lg.get(
                    "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ),
                log_file=lg.get("log_file", "./logs/{facility_id}_{timestamp}.log"),
                log_detections=bool(lg.get("log_detections", False)),
                log_tracks=bool(lg.get("log_tracks", True)),
            )

        return cfg
