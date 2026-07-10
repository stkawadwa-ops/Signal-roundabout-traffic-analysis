#!/usr/bin/env python3
"""
Signal Roundabout Traffic Analysis — Main Entry Point
=====================================================

Process a drone / CCTV video of a signalised roundabout:
  • stabilise footage (optical-flow homography)
  • detect vehicles / pedestrians (YOLOv8 ± SAHI)
  • track objects across frames (ByteTrack / Hungarian)
  • extract parameters  (speed, dimensions, zone, turning radius)
  • persist results to SQLite database
  • export CSV + JSON for downstream analysis

Quick start (CPU, smallest model, no stabilisation — fastest for testing)
---------------------------------------------------------------------------
    python main.py \\
        --video  data/raw/roundabout_1/segment_1.mp4 \\
        --facility roundabout_1 \\
        --device cpu \\
        --model  yolov8n \\
        --no-stabilize

Production run (GPU, large model, SAHI)
---------------------------------------
    python main.py \\
        --video  data/raw/roundabout_1/segment_1.mp4 \\
        --facility roundabout_1 \\
        --config configs/production.yaml \\
        --device cuda:0 \\
        --model  yolov8x \\
        --sahi

Quick smoke-test (first 200 frames only)
-----------------------------------------
    python main.py \\
        --video  data/raw/roundabout_1/segment_1.mp4 \\
        --facility test \\
        --max-frames 200 \\
        --device cpu

Geometry calibration note
--------------------------
Accuracy depends on the pixel-to-metre ratio calibrated for your roundabout.
Pass --pixel-to-meter <value> or set geometry.pixel_to_meter_ratio in your YAML.
Example: if 1000 px corresponds to 50 m on the ground → 50/1000 = 0.05 m/px.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is importable even when run directly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Signal Roundabout Traffic Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Required ----
    parser.add_argument(
        "--video",
        required=True,
        metavar="PATH",
        help="Path to input video file (.mp4, .avi, etc.)",
    )
    parser.add_argument(
        "--facility",
        required=True,
        metavar="NAME",
        help="Facility / roundabout name (used as folder key)",
    )

    # ---- Config ----
    parser.add_argument(
        "--config",
        default="configs/production.yaml",
        metavar="YAML",
        help="Path to YAML config file (default: configs/production.yaml)",
    )

    # ---- Output ----
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory for CSV / JSON / annotated video "
             "(default from config: data/analysis/<facility>)",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="SQLite database path "
             "(default from config: data/processed/<facility>/database.db)",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=1,
        metavar="N",
        help="Video segment number within the facility (default: 1)",
    )

    # ---- Detection ----
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="YOLO model name, e.g. yolov8n / yolov8s / yolov8m / yolov8l / yolov8x "
             "(overrides config)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Detection confidence threshold (overrides config, e.g. 0.5)",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEV",
        help="Inference device: 'cpu', 'cuda:0', 'cuda:1', … (overrides config)",
    )
    parser.add_argument(
        "--sahi",
        action="store_true",
        default=None,
        help="Enable SAHI sliced detection (better for small vehicles, requires GPU)",
    )
    parser.add_argument(
        "--no-sahi",
        dest="sahi",
        action="store_false",
        help="Disable SAHI (default for CPU)",
    )

    # ---- Stabilisation ----
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        default=False,
        help="Skip video stabilisation (faster; use if footage is already stable)",
    )

    # ---- Geometry / calibration ----
    parser.add_argument(
        "--pixel-to-meter",
        type=float,
        default=None,
        metavar="RATIO",
        help="Metres per pixel calibration ratio (overrides config geometry section). "
             "Example: 0.05 means 1 pixel = 5 cm.",
    )
    parser.add_argument(
        "--center",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="Roundabout centre in pixels: --center 960 540",
    )
    parser.add_argument(
        "--outer-radius",
        type=float,
        default=None,
        metavar="PX",
        help="Outer roundabout radius in pixels (overrides config)",
    )
    parser.add_argument(
        "--inner-radius",
        type=float,
        default=None,
        metavar="PX",
        help="Inner island radius in pixels (overrides config)",
    )

    # ---- Processing limits ----
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing N frames (useful for quick smoke-tests)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        metavar="N",
        help="Process every Nth frame (1 = every frame, 2 = every other, …)",
    )

    # ---- Output options ----
    parser.add_argument(
        "--annotated-video",
        action="store_true",
        default=False,
        help="Write an annotated video with bounding boxes and tracks",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        default=False,
        help="Skip CSV export",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        default=False,
        help="Skip JSON export",
    )

    # ---- Logging ----
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Write log to file as well as console",
    )

    return parser


def _setup_logging(level: str, log_file: Optional[str] = None) -> None:  # type: ignore[name-defined]
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=fmt,
        handlers=handlers,
    )
    # Suppress noisy third-party loggers at DEBUG level
    for noisy in ("ultralytics", "torch", "PIL", "sahi"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- logging ------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file
    if log_file is None and args.output_dir:
        log_file = str(Path(args.output_dir) / f"{args.facility}_{ts}.log")
    _setup_logging(args.log_level, log_file)

    logger = logging.getLogger("main")
    logger.info(f"Signal Roundabout Traffic Analysis  —  {ts}")
    logger.info(f"Video    : {args.video}")
    logger.info(f"Facility : {args.facility}")

    # ---- config -------------------------------------------------------------
    from src.config_loader import ConfigLoader

    config = ConfigLoader.load(args.config)

    # --- CLI overrides (config is overridden by explicit CLI flags) ---
    if args.model is not None:
        config.detection.model_name = args.model
    if args.conf is not None:
        config.detection.confidence_threshold = args.conf
    if args.device is not None:
        config.detection.device = args.device
    if args.sahi is True:
        config.detection.use_sahi = True
    elif args.sahi is False:
        config.detection.use_sahi = False
    if args.no_stabilize:
        config.video.enable_stabilization = False
    if args.pixel_to_meter is not None:
        config.geometry.pixel_to_meter_ratio = args.pixel_to_meter
    if args.center is not None:
        config.geometry.roundabout_center = list(args.center)
    if args.outer_radius is not None:
        config.geometry.outer_radius = args.outer_radius
    if args.inner_radius is not None:
        config.geometry.inner_radius = args.inner_radius
    if args.annotated_video:
        config.output.generate_annotated_video = True
    if args.no_csv and "csv" in config.output.export_formats:
        config.output.export_formats.remove("csv")
    if args.no_json and "json" in config.output.export_formats:
        config.output.export_formats.remove("json")

    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1

    # ---- pipeline -----------------------------------------------------------
    from src.pipeline import PipelineOrchestrator

    try:
        orchestrator = PipelineOrchestrator(config)
    except Exception as exc:
        logger.error(f"Failed to initialise pipeline: {exc}", exc_info=True)
        logger.error(
            "Common causes:\n"
            "  • ultralytics / torch not installed: pip install ultralytics\n"
            "  • Model file download failed — check internet connection\n"
            "  • CUDA device not available — pass --device cpu"
        )
        return 1

    try:
        results = orchestrator.run(
            video_path=str(video_path),
            facility_name=args.facility,
            segment_number=args.segment,
            output_dir=args.output_dir,
            db_path=args.db,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 130
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}", exc_info=True)
        return 1

    # ---- print summary ------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for key, val in results.items():
        print(f"  {key:<35} {val}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    # Allow `from typing import Optional` used inside _setup_logging above
    from typing import Optional  # noqa: F401
    raise SystemExit(main())
