#!/usr/bin/env python3
"""Run the traffic analysis pipeline end-to-end."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import PipelineConfig, PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run roundabout analysis pipeline")
    parser.add_argument("--input-video", type=Path, default=None, help="Path to input video for non-smoke mode")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/smoke_run"), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum frames to process")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, cuda:0, ...")
    parser.add_argument("--smoke", action="store_true", help="Run CPU-safe synthetic smoke mode")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config = PipelineConfig(
        output_dir=args.output_dir,
        smoke=bool(args.smoke),
        device=args.device,
        max_frames=int(args.max_frames),
        input_video=args.input_video,
    )

    runner = PipelineRunner(config)
    summary = runner.run()
    print(f"Pipeline completed: {summary['status']}")
    print(f"Mode: {summary['mode']}, frames: {summary['frames_processed']}, tracks: {summary['tracks_total']}")
    print(f"Summary: {Path(args.output_dir) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
