"""
Pipeline Orchestrator — the single object that runs the full analysis.

Frame-by-frame flow
-------------------
Video frame
  └── VideoStabilizer   (optional, homography-based)
       └── YOLODetector  (vehicle/pedestrian detection)
            └── ByteTrackTracker  (multi-object tracking)
                 └── per-frame track positions accumulated

Post-processing (after all frames)
  For each completed track:
    • Speed profile       (displacement × pixel_to_meter / dt)
    • Vehicle class       (bounding-box dimensions → VehicleClassifier)
    • Zone classification (entry / circulatory / exit per trajectory point)
    • Turning radius      (circle fit to trajectory)
    • Results → SQLite database + CSV/JSON export
    • Optional annotated video written alongside
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config_loader import ConfigLoader, PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class IDs that correspond to road users
# ---------------------------------------------------------------------------
_ROAD_USER_CLASS_IDS = {
    0,   # person
    2,   # car
    3,   # motorcycle
    5,   # bus
    6,   # train
    7,   # truck
}

# Per-class display colours (BGR)
_CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "person":     (255, 100,  50),
    "car":        ( 50, 220,  50),
    "motorcycle": (200, 200,  50),
    "bus":        ( 50, 150, 255),
    "truck":      (200,  50, 255),
    "train":      (100, 100, 200),
}
_DEFAULT_COLOUR = (180, 180, 180)


class PipelineOrchestrator:
    """
    Ties together stabilisation, detection, tracking, parameter extraction
    and persistence into a single callable object.

    Typical usage
    -------------
    >>> from src.config_loader import ConfigLoader
    >>> from src.pipeline import PipelineOrchestrator
    >>> config = ConfigLoader.load("configs/production.yaml")
    >>> orchestrator = PipelineOrchestrator(config)
    >>> results = orchestrator.run(
    ...     video_path="data/raw/roundabout_1/segment_1.mp4",
    ...     facility_name="roundabout_1",
    ... )
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._init_detector()
        self._init_tracker()
        self._init_stabilizer()
        self._init_classifier()
        self._init_calibration()
        self._init_zone_processor()
        logger.info("PipelineOrchestrator ready")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_detector(self) -> None:
        from src.detection.yolo_detector import DetectionConfig as _DC, YOLODetector

        dc = self.config.detection
        det_cfg = _DC(
            model_name=dc.model_name,
            confidence_threshold=dc.confidence_threshold,
            nms_threshold=dc.nms_threshold,
            use_sahi=dc.use_sahi,
            slice_height=dc.slice_height,
            slice_width=dc.slice_width,
            overlap_height_ratio=dc.overlap_height_ratio,
            overlap_width_ratio=dc.overlap_width_ratio,
            device=dc.device,
            augment=dc.augment,
        )
        self.detector = YOLODetector(det_cfg)
        logger.info(f"Detector: {dc.model_name} on {dc.device} (SAHI={dc.use_sahi})")

    def _init_tracker(self) -> None:
        from src.tracking.tracker import ByteTrackTracker
        from src.tracking.tracker import TrackingConfig as _TC

        tc = self.config.tracking
        tracker_cfg = _TC(
            track_thresh=tc.track_thresh,
            track_buffer=tc.track_buffer,
            match_thresh=tc.match_thresh,
            max_age=tc.max_age,
            min_hits=tc.min_hits,
            frame_rate=tc.frame_rate,
            img_size=tuple(tc.img_size),
        )
        self.tracker = ByteTrackTracker(tracker_cfg)

    def _init_stabilizer(self) -> None:
        if not self.config.video.enable_stabilization:
            self.stabilizer = None
            return
        from src.video_processing.video_stabilizer import StabilizationConfig, VideoStabilizer

        vc = self.config.video
        stab_cfg = StabilizationConfig(
            max_features=vc.max_features,
            feature_quality=vc.feature_quality,
            min_distance=vc.min_distance,
            block_size=vc.block_size,
            win_size=vc.win_size,
            max_level=vc.max_level,
            ransac_threshold=vc.ransac_threshold,
        )
        self.stabilizer = VideoStabilizer(stab_cfg)
        logger.info("Video stabilizer enabled")

    def _init_classifier(self) -> None:
        from src.core.classifier import VehicleClassifier

        self.classifier = VehicleClassifier()

    def _init_calibration(self) -> None:
        from src.core.calibration import CalibrationManager

        self.calibrator = CalibrationManager()
        p2m = self.config.geometry.pixel_to_meter_ratio
        self.calibrator.calibrate_manual(
            known_distance_m=1.0,
            pixel_distance=1.0 / p2m if p2m > 0 else 1000.0,
        )
        logger.info(f"Calibration: {p2m:.4f} m/pixel")

    def _init_zone_processor(self) -> None:
        """Build a ZoneProcessor from the geometry config."""
        from shapely.geometry import Polygon

        from src.parameter_extraction.geometry import RoundaboutGeometry, ZoneProcessor

        gc = self.config.geometry

        # Convert coordinate lists → Shapely Polygons (skip degenerate ones)
        entry_polys = [
            Polygon(zone) for zone in gc.entry_zones if len(zone) >= 3
        ]
        exit_polys = [
            Polygon(zone) for zone in gc.exit_zones if len(zone) >= 3
        ]

        geom = RoundaboutGeometry(
            center=tuple(gc.roundabout_center),
            outer_radius=gc.outer_radius,
            inner_radius=gc.inner_radius,
            entry_zones=entry_polys,
            exit_zones=exit_polys,
            circulatory_zone=None,  # auto-created in __post_init__
        )
        self.zone_processor = ZoneProcessor(geom)
        self.roundabout_geom = geom
        logger.info(
            f"Zone processor: centre={gc.roundabout_center}, "
            f"r_outer={gc.outer_radius}, r_inner={gc.inner_radius}, "
            f"entries={len(entry_polys)}, exits={len(exit_polys)}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        facility_name: str,
        segment_number: int = 1,
        output_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        skip_frames: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline on a single video file.

        Parameters
        ----------
        video_path      : path to the input .mp4/.avi video
        facility_name   : human-readable roundabout name (used as folder key)
        segment_number  : ordinal segment number within the facility's footage
        output_dir      : where to write CSV / JSON / annotated video
        db_path         : override the SQLite path from config
        max_frames      : stop after N frames (useful for quick tests)
        skip_frames     : process every Nth frame (1 = every frame)

        Returns
        -------
        dict with summary statistics
        """
        from src.data_management.database_schema import DataExporter, DatabaseManager

        # --- resolve paths ---------------------------------------------------
        facility_id = facility_name.replace(" ", "_")
        if output_dir is None:
            output_dir = ConfigLoader.get_output_dir(self.config, facility_id)
        if db_path is None:
            db_path = ConfigLoader.get_db_path(self.config, facility_id)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        t_start = time.time()

        # --- validate video file BEFORE touching the database ----------------
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap_test = cv2.VideoCapture(video_path)
        if not cap_test.isOpened():
            cap_test.release()
            raise FileNotFoundError(f"Cannot open video (unsupported format?): {video_path}")
        fps = cap_test.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_test.release()

        # --- database (opened only after video is confirmed readable) --------
        db = DatabaseManager(db_path, auto_init=True)
        facility_row_id = self._get_or_create_facility(db, facility_name)

        video_id = db.insert_video(
            facility_id=facility_row_id,
            segment_number=segment_number,
            filename=Path(video_path).name,
            duration_seconds=total_frames / fps,
            frame_rate=fps,
            resolution_width=width,
            resolution_height=height,
            file_size_bytes=Path(video_path).stat().st_size if Path(video_path).exists() else None,
        )

        logger.info("=" * 60)
        logger.info(f"STARTING ANALYSIS: {facility_name}")
        logger.info(f"  Video     : {video_path}")
        logger.info(f"  FPS       : {fps:.1f}  |  Frames: {total_frames}")
        logger.info(f"  Size      : {width}x{height}")
        logger.info(f"  Database  : {db_path}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info("=" * 60)

        # --- main processing loop --------------------------------------------
        track_histories = self._process_video(
            video_path=video_path,
            video_id=video_id,
            facility_id=facility_row_id,
            fps=fps,
            output_dir=output_dir,
            max_frames=max_frames,
            skip_frames=skip_frames,
        )

        # --- parameter extraction -------------------------------------------
        logger.info(f"Post-processing {len(track_histories)} track(s) …")
        proc_results = self._post_process_tracks(
            track_histories=track_histories,
            db=db,
            video_id=video_id,
            facility_id=facility_row_id,
            fps=fps,
        )

        # --- export results --------------------------------------------------
        if "csv" in self.config.output.export_formats:
            csv_path = str(Path(output_dir) / self.config.output.parameters_file)
            try:
                DataExporter.export_to_csv(db, facility_row_id, csv_path)
                logger.info(f"CSV export: {csv_path}")
            except Exception as exc:
                logger.warning(f"CSV export failed: {exc}")

        if "json" in self.config.output.export_formats:
            json_path = str(Path(output_dir) / self.config.output.tracks_file)
            try:
                DataExporter.export_to_json(db, facility_row_id, json_path)
                logger.info(f"JSON export: {json_path}")
            except Exception as exc:
                logger.warning(f"JSON export failed: {exc}")

        db.close()

        elapsed = time.time() - t_start

        summary: Dict[str, Any] = {
            "facility_id": facility_row_id,
            "video_id": video_id,
            "facility_name": facility_name,
            "video_path": video_path,
            "num_tracks_detected": len(track_histories),
            "num_tracks_processed": proc_results["num_processed"],
            "database_path": db_path,
            "output_dir": output_dir,
            "processing_time_seconds": round(elapsed, 1),
            "fps_video": fps,
            "total_frames": total_frames,
        }

        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"  Tracks detected  : {summary['num_tracks_detected']}")
        logger.info(f"  Tracks processed : {summary['num_tracks_processed']}")
        logger.info(f"  Processing time  : {elapsed:.1f}s")
        logger.info(f"  Output           : {output_dir}")
        logger.info("=" * 60)

        return summary

    # ------------------------------------------------------------------
    # Frame-level processing
    # ------------------------------------------------------------------

    def _process_video(  # noqa: C901
        self,
        video_path: str,
        video_id: int,
        facility_id: int,
        fps: float,
        output_dir: str,
        max_frames: Optional[int],
        skip_frames: int,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Iterate over video frames, stabilise, detect, and track.

        Returns
        -------
        track_histories : dict  { external_track_id → history_dict }
            Each history_dict contains:
                'positions'    list[np.ndarray(2,)]  – bbox centre (x,y)
                'frames'       list[int]
                'bboxes'       list[np.ndarray(4,)]  – [x1,y1,x2,y2]
                'class_id'     int
                'class_name'   str
                'confidences'  list[float]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Reset tracker state for this video
        self.tracker.reset()
        if self.stabilizer:
            self.stabilizer.reset()

        # Optional annotated video writer
        video_writer: Optional[cv2.VideoWriter] = None
        if self.config.output.generate_annotated_video:
            out_path = str(
                Path(output_dir) / self.config.output.video_output
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            logger.info(f"Writing annotated video → {out_path}")

        track_histories: Dict[int, Dict[str, Any]] = {}

        frame_count = 0
        processed_count = 0
        log_interval = max(1, total_frames // 20)  # log ~20 progress updates

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue

                if max_frames and processed_count >= max_frames:
                    logger.info(f"Reached max_frames limit ({max_frames})")
                    break

                # --- stabilise ---
                if self.stabilizer is not None:
                    try:
                        frame, _ = self.stabilizer.process_frame(frame)
                    except Exception as exc:
                        logger.debug(f"Stabilisation skipped frame {frame_count}: {exc}")

                # --- detect ---
                try:
                    detections_raw = self.detector.detect(frame)
                except Exception as exc:
                    logger.warning(f"Detection failed on frame {frame_count}: {exc}")
                    frame_count += 1
                    processed_count += 1
                    continue

                # Filter to road-user classes only and build tracker input
                formatted_dets = self._format_detections(detections_raw)

                if self.config.logging.log_detections:
                    logger.debug(
                        f"Frame {frame_count}: {len(formatted_dets)} road users detected"
                    )

                # --- track ---
                try:
                    active_tracks = self.tracker.update(formatted_dets)
                except Exception as exc:
                    logger.warning(f"Tracker failed on frame {frame_count}: {exc}")
                    active_tracks = []

                # --- accumulate track histories ---
                for trk in active_tracks:
                    tid = trk.track_id
                    if tid not in track_histories:
                        track_histories[tid] = {
                            "positions":   [],
                            "frames":      [],
                            "bboxes":      [],
                            "class_id":    trk.class_id,
                            "class_name":  trk.class_name,
                            "confidences": [],
                        }
                    cx, cy = trk.get_center()
                    track_histories[tid]["positions"].append(np.array([cx, cy]))
                    track_histories[tid]["frames"].append(frame_count)
                    track_histories[tid]["bboxes"].append(trk.bbox.copy())
                    track_histories[tid]["confidences"].append(trk.confidence)

                # --- draw annotations onto frame ---
                if video_writer is not None:
                    anno_frame = self._draw_annotations(frame, active_tracks)
                    video_writer.write(anno_frame)

                # --- progress log ---
                if processed_count % log_interval == 0 and processed_count > 0:
                    pct = (frame_count / max(total_frames, 1)) * 100
                    logger.info(
                        f"  Frame {frame_count}/{total_frames} ({pct:.0f}%) — "
                        f"active tracks: {len(active_tracks)}"
                    )

                frame_count += 1
                processed_count += 1

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()

        logger.info(
            f"Video processing done. Frames processed: {processed_count}. "
            f"Unique tracks seen: {len(track_histories)}"
        )
        return track_histories

    def _format_detections(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert YOLODetector output dict into ByteTrackTracker input format."""
        boxes       = raw.get("boxes", np.empty((0, 4)))
        confidences = raw.get("confidences", np.empty(0))
        class_ids   = raw.get("class_ids", np.empty(0, dtype=int))
        class_names = raw.get("class_names", [])

        formatted = []
        for i, (box, conf, cid) in enumerate(zip(boxes, confidences, class_ids)):
            cid_int = int(cid)
            if cid_int not in _ROAD_USER_CLASS_IDS:
                continue
            cname = class_names[i] if i < len(class_names) else str(cid_int)
            formatted.append(
                {
                    "bbox":       np.array(box, dtype=np.float32),
                    "confidence": float(conf),
                    "class_id":   cid_int,
                    "class_name": cname,
                }
            )
        return formatted

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process_tracks(
        self,
        track_histories: Dict[int, Dict[str, Any]],
        db: Any,
        video_id: int,
        facility_id: int,
        fps: float,
    ) -> Dict[str, Any]:
        """Extract parameters from completed tracks and write to DB."""
        from src.parameter_extraction.geometry import TurningGeometry

        pixel_to_meter = self.config.geometry.pixel_to_meter_ratio
        min_len = self.config.tracking.min_track_length

        num_processed = 0

        for ext_id, history in track_histories.items():
            positions   = history["positions"]
            frames      = history["frames"]
            bboxes      = history["bboxes"]
            class_name  = history["class_name"]
            class_id    = history["class_id"]
            confidences = history["confidences"]

            if len(positions) < min_len:
                continue  # too short to be meaningful

            trajectory = np.array(positions, dtype=np.float64)
            conf_arr   = np.array(confidences, dtype=np.float64)

            start_frame = frames[0]
            end_frame   = frames[-1]

            # ---- insert track ----
            track_db_id = db.insert_track(
                facility_id=facility_id,
                video_id=video_id,
                external_track_id=ext_id,
                class_id=class_id,
                class_name=class_name,
                start_frame=start_frame,
                end_frame=end_frame,
                confidence_mean=float(np.mean(conf_arr)),
                confidence_min=float(np.min(conf_arr)),
                confidence_max=float(np.max(conf_arr)),
            )

            # ---- trajectory points (every frame) ----
            track_points = [
                {"frame_number": f, "x": float(pos[0]), "y": float(pos[1])}
                for f, pos in zip(frames, positions)
            ]
            db.batch_insert_track_points(track_db_id, track_points)

            # ---- speed profile ----
            speed_profile_kmh = self._compute_speed_profile(trajectory, fps, pixel_to_meter)

            # ---- vehicle classification from bbox dimensions ----
            classification = self._classify_from_bboxes(bboxes, class_name, pixel_to_meter)

            # ---- zone analysis ----
            zone_data = self._analyze_zones(trajectory)

            # ---- turning radius ----
            turning_radius_px = TurningGeometry.estimate_turning_radius(trajectory)
            turning_radius_m = (
                turning_radius_px * pixel_to_meter if turning_radius_px is not None else None
            )

            # ---- path length ----
            if len(trajectory) > 1:
                diffs = np.diff(trajectory, axis=0)
                path_length_m = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))) * pixel_to_meter)
            else:
                path_length_m = 0.0

            # ---- curvature ----
            curvature_arr = TurningGeometry.compute_curvature(
                trajectory, window=self.config.parameter_extraction.curvature_window
            )
            curvature_avg = float(np.mean(curvature_arr)) if len(curvature_arr) > 0 else None

            # ---- time in roundabout ----
            time_in_roundabout_s = (end_frame - start_frame) / fps

            # ---- build parameters dict ----
            params: Dict[str, Any] = {
                "speed_avg": float(np.mean(speed_profile_kmh)) if len(speed_profile_kmh) > 0 else None,
                "speed_max": float(np.max(speed_profile_kmh)) if len(speed_profile_kmh) > 0 else None,
                "speed_min": float(np.min(speed_profile_kmh)) if len(speed_profile_kmh) > 0 else None,
                "entry_frames":        zone_data.get("entry_frames"),
                "circulatory_frames":  zone_data.get("circulatory_frames"),
                "exit_frames":         zone_data.get("exit_frames"),
                "turning_radius":      turning_radius_m,
                "path_length":         path_length_m,
                "curvature_avg":       curvature_avg,
                "vehicle_class":       classification.category_name if classification else class_name,
                "vehicle_category":    classification.category.name if classification else None,
                "first_zone":          zone_data.get("first_zone"),
                "last_zone":           zone_data.get("last_zone"),
                "zone_sequence":       zone_data.get("zone_sequence"),
                "time_in_roundabout_seconds": time_in_roundabout_s,
                "dwell_time_seconds":  time_in_roundabout_s,
                "track_quality":       float(np.mean(conf_arr)),
                "confidence_score":    float(np.mean(conf_arr)),
            }

            db.insert_track_parameters(track_db_id, params)

            if self.config.logging.log_tracks:
                logger.debug(
                    f"Track {ext_id} ({class_name}): "
                    f"{len(positions)} pts, "
                    f"speed={params['speed_avg']:.1f} km/h, "
                    f"class={params['vehicle_class']}"
                )

            num_processed += 1

        logger.info(f"Parameter extraction: {num_processed}/{len(track_histories)} tracks processed")
        return {"num_processed": num_processed}

    # ------------------------------------------------------------------
    # Parameter computation helpers
    # ------------------------------------------------------------------

    def _compute_speed_profile(
        self,
        trajectory: np.ndarray,
        fps: float,
        pixel_to_meter: float,
    ) -> np.ndarray:
        """
        Compute instantaneous speed (km/h) for each trajectory segment.

        Returns an array of length len(trajectory)-1.
        """
        if len(trajectory) < 2:
            return np.array([0.0])

        dt = 1.0 / fps  # seconds per frame
        diffs = np.diff(trajectory, axis=0)
        displacements_m = np.sqrt(np.sum(diffs ** 2, axis=1)) * pixel_to_meter
        speeds_mps = displacements_m / dt
        speeds_kmh = speeds_mps * 3.6

        # Smoothing
        w = self.config.parameter_extraction.speed_filter_window
        if w > 1 and len(speeds_kmh) >= w:
            kernel = np.ones(w) / w
            speeds_kmh = np.convolve(speeds_kmh, kernel, mode="valid")

        # Clip unrealistically high values (noise from rapid bbox jumps)
        max_speed = self.config.parameter_extraction.max_speed_kmh
        speeds_kmh = np.clip(speeds_kmh, 0.0, max_speed)

        return speeds_kmh

    def _classify_from_bboxes(
        self,
        bboxes: List[np.ndarray],
        yolo_class_name: str,
        pixel_to_meter: float,
    ) -> Optional[Any]:
        """
        Use the median bounding-box size across the track to classify the vehicle.

        In top-down UAV view:
          max(bbox_w, bbox_h) ≈ vehicle length
          min(bbox_w, bbox_h) ≈ vehicle width
        """
        if yolo_class_name == "person":
            return self.classifier.classify_pedestrian()

        if not bboxes:
            return None

        lengths_px = []
        widths_px  = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            lengths_px.append(max(w, h))
            widths_px.append(min(w, h))

        length_m = float(np.median(lengths_px)) * pixel_to_meter
        width_m  = float(np.median(widths_px))  * pixel_to_meter

        if length_m <= 0 or width_m <= 0:
            return None

        try:
            return self.classifier.classify_vehicle(length_m=length_m, width_m=width_m)
        except Exception as exc:
            logger.debug(f"Classification error: {exc}")
            return None

    def _analyze_zones(self, trajectory: np.ndarray) -> Dict[str, Any]:
        """Classify each trajectory point into entry/circulatory/exit zone."""
        if len(trajectory) == 0:
            return {}

        try:
            zone_map = self.zone_processor.classify_trajectory(trajectory)
        except Exception as exc:
            logger.debug(f"Zone analysis error: {exc}")
            return {}

        transitions = []
        try:
            transitions = self.zone_processor.get_zone_transitions(trajectory)
        except Exception:
            pass

        # Build zone sequence string
        zone_seq_parts = []
        prev_z = None
        for pt in trajectory:
            try:
                z = self.zone_processor.classify_point(tuple(pt))
            except Exception:
                z = "outside"
            if z != prev_z:
                zone_seq_parts.append(z)
                prev_z = z
        zone_sequence = "→".join(zone_seq_parts)

        # First / last zone
        first_zone = zone_seq_parts[0] if zone_seq_parts else None
        last_zone  = zone_seq_parts[-1] if zone_seq_parts else None

        return {
            "entry_frames":       len(zone_map.get("entry", [])),
            "circulatory_frames": len(zone_map.get("circulatory", [])),
            "exit_frames":        len(zone_map.get("exit", [])),
            "first_zone":         first_zone,
            "last_zone":          last_zone,
            "zone_sequence":      zone_sequence,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _draw_annotations(
        self,
        frame: np.ndarray,
        active_tracks: List[Any],
    ) -> np.ndarray:
        """Draw bounding boxes, IDs, speeds and zone circles onto a frame copy."""
        out = frame.copy()
        gc = self.config.geometry

        # Draw circulatory-zone rings
        if self.config.output.annotate_zones:
            cx, cy = int(gc.roundabout_center[0]), int(gc.roundabout_center[1])
            cv2.circle(out, (cx, cy), int(gc.outer_radius), (200, 200,  50), 2)
            cv2.circle(out, (cx, cy), int(gc.inner_radius), (200, 200,  50), 1)
            # Entry/exit zone outlines
            for zone in gc.entry_zones:
                pts = np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], True, (50, 220, 50), 1)
            for zone in gc.exit_zones:
                pts = np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], True, (50, 100, 255), 1)

        if not self.config.output.annotate_tracks:
            return out

        for trk in active_tracks:
            bbox  = trk.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            colour = _CLASS_COLOURS.get(trk.class_name, _DEFAULT_COLOUR)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            # Label: "ID:42 car 0.91"
            label = f"ID:{trk.track_id} {trk.class_name} {trk.confidence:.2f}"
            label_y = max(y1 - 6, 12)
            cv2.putText(out, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

            # Trajectory tail (last 20 positions in history)
            if hasattr(trk, "history") and len(trk.history) > 1:
                tail = trk.history[-20:]
                for i in range(1, len(tail)):
                    p1 = (int((tail[i-1][0] + tail[i-1][2]) / 2),
                          int((tail[i-1][1] + tail[i-1][3]) / 2))
                    p2 = (int((tail[i][0] + tail[i][2]) / 2),
                          int((tail[i][1] + tail[i][3]) / 2))
                    cv2.line(out, p1, p2, colour, 1)

        return out

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_or_create_facility(db: Any, facility_name: str) -> int:
        """Return existing facility_id or create a new one."""
        try:
            db.cursor.execute(
                "SELECT facility_id FROM facilities WHERE name = ?",
                (facility_name,),
            )
            row = db.cursor.fetchone()
            if row:
                return int(row[0])
        except Exception:
            pass

        return db.insert_facility(name=facility_name, location=None)
