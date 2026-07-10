#!/usr/bin/env python3
"""Validate that the repository is ready for an initial trial run."""

from __future__ import annotations

import argparse
import importlib
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CheckResult:
    """Single readiness check result."""

    name: str
    ok: bool
    detail: str


def check_file_exists(path: Path, label: str) -> CheckResult:
    """Verify a required file exists."""
    return CheckResult(label, path.exists(), str(path) if path.exists() else f"Missing: {path}")


def check_yaml_config(path: Path) -> CheckResult:
    """Validate the production config can be parsed."""
    try:
        import yaml
    except ImportError as exc:
        return CheckResult("Config parsing", False, f"PyYAML unavailable: {exc}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except Exception as exc:
        return CheckResult("Config parsing", False, f"Invalid YAML: {exc}")

    required_sections = {"video", "detection", "tracking", "geometry", "parameter_extraction", "database", "output", "logging"}
    missing_sections = sorted(required_sections.difference(config or {}))
    if missing_sections:
        return CheckResult("Config parsing", False, f"Missing sections: {', '.join(missing_sections)}")

    return CheckResult("Config parsing", True, "production config parsed successfully")


def check_import(module_name: str) -> CheckResult:
    """Try importing a module and report the result."""
    try:
        importlib.import_module(module_name)
        return CheckResult(f"Import {module_name}", True, "ok")
    except Exception as exc:
        return CheckResult(f"Import {module_name}", False, f"{type(exc).__name__}: {exc}")


def check_database_init() -> CheckResult:
    """Create a temporary SQLite database using the packaged schema."""
    try:
        from src.data_management import DatabaseManager
    except Exception as exc:
        return CheckResult("Database schema init", False, f"{type(exc).__name__}: {exc}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trial_ready.db"
            with DatabaseManager(str(db_path), auto_init=True):
                with sqlite3.connect(db_path) as connection:
                    tables = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    }
            expected = {
                "facilities",
                "videos",
                "detections",
                "tracks",
                "track_points",
                "track_parameters",
                "aggregated_metrics",
                "optimization_results",
                "validation_log",
            }
            missing = sorted(expected.difference(tables))
            if missing:
                return CheckResult("Database schema init", False, f"Missing tables: {', '.join(missing)}")
    except Exception as exc:
        return CheckResult("Database schema init", False, f"{type(exc).__name__}: {exc}")

    return CheckResult("Database schema init", True, "all expected tables created")


def run_checks(config_path: Path) -> list[CheckResult]:
    """Run the full trial-readiness checklist."""
    checks: list[Callable[[], CheckResult]] = [
        lambda: check_file_exists(REPO_ROOT / "README.md", "Repository README"),
        lambda: check_file_exists(config_path, "Trial config"),
        lambda: check_yaml_config(config_path),
        lambda: check_import("src"),
        lambda: check_import("src.core"),
        lambda: check_import("src.video_processing"),
        lambda: check_import("src.detection"),
        lambda: check_import("src.tracking"),
        lambda: check_import("src.parameter_extraction"),
        lambda: check_import("src.data_management"),
        check_database_init,
    ]
    return [check() for check in checks]


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Validate repository readiness for an initial trial.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "production.yaml"),
        help="Path to the YAML configuration file to validate.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    results = run_checks(config_path)

    print("Signal Roundabout trial readiness report")
    print("=" * 40)

    failures = 0
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        if not result.ok:
            failures += 1

    print("=" * 40)
    if failures:
        print(f"Trial readiness failed with {failures} issue(s).")
        return 1

    print("Trial readiness passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
