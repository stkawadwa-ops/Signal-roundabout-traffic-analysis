import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_smoke_pipeline_execution(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "smoke_output"
    script_path = repo_root / "scripts" / "run_pipeline.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--smoke",
            "--device",
            "cpu",
            "--max-frames",
            "5",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    summary_path = output_dir / "summary.json"
    db_path = output_dir / "database.db"
    assert summary_path.exists()
    assert db_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert summary["mode"] == "smoke"
    assert summary["device"] == "cpu"
    assert summary["frames_processed"] == 5

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM tracks")
        track_count = cur.fetchone()[0]
    assert track_count >= 1
