import json
import os
import subprocess
import sys
from pathlib import Path


def test_static_site_builds_all_charts_from_retained_snapshot(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    metadata = json.loads(
        (project_root / "scripts/streamlit_app/demo_data/canonical_metadata.json").read_text()
    )
    output_dir = tmp_path / "site"
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/generate_static_site.py"),
            "--results-dir",
            str(tmp_path / "missing-results"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name for path in output_dir.glob("*.html")} == {
        "index.html",
        "model_rankings.html",
        "heatmap.html",
        "budget_progression.html",
        "parameter_importance.html",
    }
    assert f"{metadata['rows_after_filters']:,}" in (output_dir / "index.html").read_text()
    for chart in output_dir.glob("*.html"):
        if chart.name != "index.html":
            assert "Plotly.newPlot" in chart.read_text()
