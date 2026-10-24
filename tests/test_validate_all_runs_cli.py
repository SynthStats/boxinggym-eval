import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("output_dir", [None, "custom_reports"])
def test_validation_cli_writes_reports_to_selected_directory(tmp_path: Path, output_dir):
    results = tmp_path / "results"
    results.mkdir()
    script = Path(__file__).resolve().parents[1] / "scripts/validate_all_runs.py"
    command = [sys.executable, str(script), "--results-dir", str(results)]
    if output_dir is not None:
        command.extend(["--output-dir", output_dir])

    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)

    reports = tmp_path / (output_dir or "validation_reports")
    assert json.loads((reports / "validation_quarantine.json").read_text()) == []
    assert json.loads((reports / "validation_review.json").read_text()) == []
    summary = json.loads((reports / "validation_summary.json").read_text())
    assert summary["total_files"] == 0
    assert summary["valid"] == 0
