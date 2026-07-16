"""Minimal end-to-end pipeline orchestration."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.data_management.database_schema import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    output_dir: Path
    smoke: bool = False
    device: str = "auto"
    max_frames: int = 30
    input_video: Optional[Path] = None
    frame_rate: float = 30.0
    pixel_to_meter_ratio: float = 0.05


class PipelineRunner:
    """Minimal glue runner across stabilization, detection, tracking, extraction, persistence."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(self.config.device)

    def run(self) -> Dict:
        start_time = time.time()
        if self.config.smoke:
            summary = self._run_smoke_pipeline()
        else:
            summary = self._run_real_pipeline()
        summary["duration_seconds"] = round(time.time() - start_time, 3)
        summary_path = self.config.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Pipeline summary written to %s", summary_path)
        return summary

    def _resolve_device(self, requested_device: str) -> str:
        if self.config.smoke:
            return "cpu"
        normalized_device = requested_device.lower()
        if normalized_device == "cpu":
            return "cpu"
        if normalized_device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return requested_device
                logger.warning("CUDA requested but unavailable, falling back to CPU")
            except (ImportError, ModuleNotFoundError):
                logger.warning("Torch unavailable, falling back to CPU")
            return "cpu"
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except (ImportError, ModuleNotFoundError):
            return "cpu"

    def _run_smoke_pipeline(self) -> Dict:
        db_path = self.config.output_dir / "database.db"
        if db_path.exists():
            db_path.unlink()

        max_frames = max(1, self.config.max_frames)
        detections_by_frame: Dict[int, List[Dict]] = {}
        track_points: List[Dict] = []
        prev_center: Optional[List[float]] = None
        speed_samples_kmh: List[float] = []

        for frame_idx in range(max_frames):
            x1 = 40 + frame_idx * 3
            y1 = 60 + frame_idx * 2
            x2 = x1 + 22
            y2 = y1 + 14
            detection = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": 0.99,
                "class_id": 2,
                "class_name": "car",
            }
            detections_by_frame[frame_idx] = [detection]

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            point = {"frame_number": frame_idx, "x": cx, "y": cy, "vx": None, "vy": None}
            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                point["vx"] = dx
                point["vy"] = dy
                dt_seconds = 1.0 / self.config.frame_rate
                distance_meters = math.sqrt(dx * dx + dy * dy) * self.config.pixel_to_meter_ratio
                speed_meters_per_second = distance_meters / dt_seconds
                speed_samples_kmh.append(speed_meters_per_second * 3.6)
            track_points.append(point)
            prev_center = [cx, cy]

        speed_avg = sum(speed_samples_kmh) / len(speed_samples_kmh) if speed_samples_kmh else 0.0
        speed_max = max(speed_samples_kmh) if speed_samples_kmh else 0.0
        speed_min = min(speed_samples_kmh) if speed_samples_kmh else 0.0
        entry_frames = 1
        exit_frames = 1 if max_frames > 1 else 0

        with DatabaseManager(str(db_path), auto_init=True) as db:
            facility_id = db.insert_facility(name=f"smoke_facility_{int(time.time())}", location="synthetic")
            video_id = db.insert_video(
                facility_id=facility_id,
                segment_number=1,
                filename="synthetic_smoke_sequence",
                duration_seconds=max_frames / self.config.frame_rate,
                frame_rate=self.config.frame_rate,
                resolution_width=320,
                resolution_height=240,
                file_size_bytes=0,
            )
            track_id = db.insert_track(
                facility_id=facility_id,
                video_id=video_id,
                external_track_id=1,
                class_id=2,
                class_name="car",
                start_frame=0,
                end_frame=max_frames - 1,
                confidence_mean=0.99,
                confidence_min=0.99,
                confidence_max=0.99,
            )
            db.batch_insert_track_points(track_id=track_id, points=track_points)
            db.insert_track_parameters(
                track_id=track_id,
                parameters={
                    "speed_avg": speed_avg,
                    "speed_max": speed_max,
                    "speed_min": speed_min,
                    "entry_frames": entry_frames,
                    "circulatory_frames": max(0, max_frames - entry_frames - exit_frames),
                    "exit_frames": exit_frames,
                    "turning_radius": None,
                    "path_length": None,
                    "curvature_avg": None,
                    "vehicle_class": "car",
                    "vehicle_category": "STANDARD_CAR",
                    "first_zone": "entry",
                    "last_zone": "exit",
                    "zone_sequence": "entry->circulatory->exit",
                    "time_in_roundabout_seconds": max_frames / self.config.frame_rate,
                    "dwell_time_seconds": 0.0,
                    "track_quality": 1.0,
                    "confidence_score": 0.99,
                },
            )

        return {
            "status": "success",
            "mode": "smoke",
            "device": self.device,
            "frames_processed": max_frames,
            "detections_total": sum(len(v) for v in detections_by_frame.values()),
            "tracks_total": 1,
            "database_path": str(db_path),
            "stages": {
                "stabilization": "synthetic_noop",
                "detection": "synthetic_detector",
                "tracking": "single_track_glue",
                "parameter_extraction": "speed_and_zone_summary",
                "persistence": "sqlite",
            },
        }

    def _run_real_pipeline(self) -> Dict:
        if not self.config.input_video:
            raise ValueError("Non-smoke mode requires --input-video")
        if not self.config.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {self.config.input_video}")

        try:
            import cv2
            from src.video_processing.video_stabilizer import VideoStabilizer
            from src.detection.yolo_detector import DetectionConfig, YOLODetector
            from src.tracking.tracker import ByteTrackTracker, TrackingConfig
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise RuntimeError(
                "Required runtime dependencies are unavailable for non-smoke mode. "
                "Use --smoke for CPU-safe execution."
            ) from exc

        cap = cv2.VideoCapture(str(self.config.input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {self.config.input_video}")

        stabilizer = VideoStabilizer()
        detector = YOLODetector(DetectionConfig(device=self.device, use_sahi=False))
        tracker = ByteTrackTracker(TrackingConfig(frame_rate=int(round(self.config.frame_rate))))

        frame_idx = 0
        detections_total = 0
        tracks_seen = set()
        while cap.isOpened() and frame_idx < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            stable_frame, _ = stabilizer.process_frame(frame)
            detections = detector.detect(stable_frame)

            tracking_input = []
            boxes = detections.get("boxes", [])
            confidences = detections.get("confidences", [])
            class_ids = detections.get("class_ids", [])
            class_names = detections.get("class_names", [])
            for i, box in enumerate(boxes):
                tracking_input.append(
                    {
                        "bbox": box,
                        "confidence": float(confidences[i]),
                        "class_id": int(class_ids[i]),
                        "class_name": class_names[i],
                    }
                )
            active_tracks = tracker.update(tracking_input)
            tracks_seen.update(t.track_id for t in active_tracks)
            detections_total += len(tracking_input)
            frame_idx += 1

        cap.release()

        return {
            "status": "success",
            "mode": "full",
            "device": self.device,
            "frames_processed": frame_idx,
            "detections_total": detections_total,
            "tracks_total": len(tracks_seen),
            "database_path": None,
            "stages": {
                "stabilization": "opencv_optical_flow",
                "detection": "yolo",
                "tracking": "bytetrack_or_hungarian",
                "parameter_extraction": "minimal_summary",
                "persistence": "summary_json",
            },
        }
