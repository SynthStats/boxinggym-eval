import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.refresh_canonical_results import (
    MATCHED_MODELS,
    PROTOCOL_FIELDS,
    apply_canonical_filters,
    compute_champions_model_only,
    compute_champions_model_plus_ppl,
    refresh_canonical_artifacts,
    render_hf_snapshot_block,
    render_metadata_block,
    render_readme_results_block,
    replace_marked_block,
)


def test_apply_canonical_filters_excludes_outliers_and_low_budget():
    df = pd.DataFrame(
        [
            {"model": "m1", "env": "e1", "budget": 10, "z_mean": 0.1, "is_outlier": False},
            {"model": "m1", "env": "e1", "budget": 5, "z_mean": 0.2, "is_outlier": False},
            {"model": "m2", "env": "e1", "budget": 10, "z_mean": 0.3, "is_outlier": True},
            {"model": "m3", "env": "e2", "budget": 10, "z_mean": None, "is_outlier": False},
        ]
    )

    filtered, stats = apply_canonical_filters(df, min_budget=10, exclude_outliers=True)

    assert len(filtered) == 1
    assert filtered.iloc[0]["model"] == "m1"
    assert stats["rows_total_before_filters"] == 4
    assert stats["rows_after_filters"] == 1


def test_champion_definitions_can_differ():
    df = pd.DataFrame(
        [
            # model m1 has one very good PPL=False run and one bad PPL=True run
            {"env": "e1", "model": "m1", "use_ppl": False, "z_mean": -0.20},
            {"env": "e1", "model": "m1", "use_ppl": True, "z_mean": +0.40},
            # model m2 is moderately good on average
            {"env": "e1", "model": "m2", "use_ppl": False, "z_mean": +0.05},
        ]
    )

    model_only = compute_champions_model_only(df)
    model_plus_ppl = compute_champions_model_plus_ppl(df)

    assert model_only[0]["model_raw"] == "m2"
    assert model_plus_ppl[0]["model_raw"] == "m1"
    assert model_plus_ppl[0]["use_ppl"] is False


def test_replace_marked_block_updates_only_marker_body(tmp_path: Path):
    src = "before\n<!-- CANONICAL_RESULTS:START -->\nold\n<!-- CANONICAL_RESULTS:END -->\nafter\n"
    path = tmp_path / "README.md"
    path.write_text(src)

    replace_marked_block(path, "CANONICAL_RESULTS", "new content")

    out = path.read_text()
    assert "before" in out
    assert "after" in out
    assert "new content" in out
    assert "old" not in out


@pytest.mark.parametrize("significantly_worse_models", [[], ["Model B", "Model C"]])
def test_rendered_readme_blocks_put_facts_in_tables(tmp_path: Path, significantly_worse_models):
    metrics = {
        "leaderboard": [
            {
                "rank": 1,
                "model_display": "Model A",
                "mean_z": 0.1,
                "ci_low": 0.0,
                "ci_high": 0.2,
                "n": 10,
                "significant_vs_best": False,
                "p_adjusted_vs_best": None,
            },
            {
                "rank": 2,
                "model_display": "Model B",
                "mean_z": 0.3,
                "ci_low": 0.2,
                "ci_high": 0.4,
                "n": 12,
                "significant_vs_best": bool(significantly_worse_models),
                "p_adjusted_vs_best": 0.012 if significantly_worse_models else 0.42,
            },
            {
                "rank": 3,
                "model_display": "Model C",
                "mean_z": 0.5,
                "ci_low": 0.4,
                "ci_high": 0.6,
                "n": 1,
                "significant_vs_best": False,
                "p_adjusted_vs_best": None,
            },
        ],
        "champions_model_only": [{"env": "e1", "model_display": "Model A", "z_mean": -0.1}],
        "champions_model_plus_ppl": [
            {"env": "e1", "model_display": "Model A", "use_ppl": True, "z_mean": -0.2}
        ],
        "key_findings": {
            "best_model": "Model A",
            "best_model_mean_z": 0.1,
            "n_env_beating_baseline_model_only": 1,
            "n_env_total": 1,
            "significantly_worse_models": significantly_worse_models,
        },
    }
    metadata = {
        "generated_at_utc": "2026-02-13T00:00:00Z",
        "snapshot_version": "v1",
        "source_path": ".boxing-gym-cache/runs.parquet",
        "snapshot_path": "scripts/streamlit_app/demo_data/canonical_runs.parquet",
        "rows_after_filters": 1,
        "models_after_filters": 1,
        "envs_after_filters": 1,
        "filters": {"min_budget": 10, "exclude_outliers": True},
    }

    for row in metrics["leaderboard"]:
        row["model_raw"] = row["model_display"]
    metrics["matched_comparison"] = {
        "budgets": [10],
        "n_cells": 1,
        "n_environments": 1,
        "models": metrics["leaderboard"].copy(),
    }
    metrics["leaderboard"].reverse()
    results = render_readme_results_block(metrics, metadata)
    snapshot = render_metadata_block(metadata)

    assert "### Matched and pooled scores in this snapshot" in results
    assert "| Model | Matched mean z | 95% environment-cluster interval | Matched rows | Pooled mean z | Pooled rows |" in results
    assert "Rank" not in results
    assert "rank 1" not in results
    assert "Reference" not in results
    assert "Adjusted p-value" not in results
    assert results.index("| Model A |") < results.index("| Model B |") < results.index("| Model C |")
    assert "### Lowest mean score in each environment" in results
    assert "### Lowest mean score by model and PyMC setting" in results
    assert "### Statistical methods" in results

    # Facts and definitions are table content; only section headings stand outside it.
    for block in (snapshot, results):
        assert all(
            not line.strip() or line.startswith(("|", "### ")) for line in block.splitlines()
        ), block

    expected_rows = [
        "| Model A | +0.100 | [+0.000, +0.200] | 10 | +0.100 | 10 |",
        "| Model B | +0.300 | [+0.200, +0.400] | 12 | +0.300 | 12 |",
        "| Model C | +0.500 | [+0.400, +0.600] | 1 | +0.500 | 1 |",
        "| e1 | Model A | -0.100 | ✓ Beats baseline |",
        "| e1 | Model A | true | -0.200 | ✓ Beats baseline |",
        "| Environments beating baseline (model only) | 1/1 |",
        "| Pooled mean score | Mean standardised prediction error across each model's filtered runs; lower is better |",
        "| Model within environment | Mean over all runs for each model within an environment, pooling PyMC settings |",
        "| Model and PyMC setting | Mean over each `(model, use_ppl)` combination within an environment; `use_ppl=true` adds PyMC model refinement (Box's Loop) |",
        "| Environment-cluster bootstrap | 95% percentile intervals from 200,000 resamples of exchangeable environments; seed 20261026 |",
        "| Snapshot version | `v1` |",
    ]
    assert "Lowest pooled mean" not in results
    assert "Welch" not in results
    assert "Benjamini" not in results
    for row in expected_rows:
        assert row in results
    for row in [
        "| Snapshot version | `v1` |",
        "| Bundled snapshot | `scripts/streamlit_app/demo_data/canonical_runs.parquet` |",
        "| Filters | `budget >= 10`, `exclude_outliers=true` |",
        "| Coverage | 1 runs across 1 models and 1 environments |",
    ]:
        assert row in snapshot

    path = tmp_path / "README.md"
    original = (
        "before\n<!-- CANONICAL_METADATA:START -->\nold metadata\n"
        "<!-- CANONICAL_METADATA:END -->\nbetween\n"
        "<!-- CANONICAL_RESULTS:START -->\nold results\n"
        "<!-- CANONICAL_RESULTS:END -->\nafter\n"
    )
    path.write_text(original)
    replace_marked_block(path, "CANONICAL_METADATA", snapshot)
    replace_marked_block(path, "CANONICAL_RESULTS", results)
    expected = original.replace("old metadata", snapshot.strip()).replace("old results", results)
    assert path.read_text() == expected

    # Refreshing the same content preserves the README exactly.
    replace_marked_block(path, "CANONICAL_METADATA", snapshot)
    replace_marked_block(path, "CANONICAL_RESULTS", results)
    assert path.read_text() == expected


def test_hf_snapshot_identifies_data_sources_and_preserves_values():
    metadata = {
        "generated_at_utc": "2026-02-13T00:00:00Z",
        "source_path": "runs.parquet",
        "snapshot_path": "outputs/canonical_runs.parquet",
        "rows_after_filters": 22,
        "models_after_filters": 2,
        "envs_after_filters": 1,
        "filters": {"min_budget": 10, "exclude_outliers": True},
    }
    metrics = {"key_findings": {"best_model": "Model A", "best_model_mean_z": 0.1}}
    snapshot = render_hf_snapshot_block(metrics, metadata)

    assert "| Generated | 2026-02-13T00:00:00Z |" in snapshot
    assert "| Coverage | 22 runs across 2 models and 1 environments |" in snapshot
    assert "| Lowest pooled mean | Model A (`z=+0.100`) |" in snapshot
    assert "| Leaderboard | Bundled snapshot |" in snapshot
    assert "| Sweep analysis | Selected Weights & Biases sweeps |" in snapshot
    assert "| Bundled snapshot | `outputs/canonical_runs.parquet` |" in snapshot


def test_committed_readmes_reproduce_from_bundled_metrics():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "scripts/streamlit_app/demo_data"
    metrics = json.loads((data_dir / "canonical_metrics.json").read_text())
    metadata = json.loads((data_dir / "canonical_metadata.json").read_text())

    for name, marker, rendered in [
        ("README.md", "CANONICAL_METADATA", render_metadata_block(metadata)),
        ("README.md", "CANONICAL_RESULTS", render_readme_results_block(metrics, metadata)),
        ("HF_SPACE_README.md", "CANONICAL_HF_SNAPSHOT", render_hf_snapshot_block(metrics, metadata)),
    ]:
        text = (root / name).read_text()
        start = f"<!-- {marker}:START -->\n"
        end = f"\n<!-- {marker}:END -->"
        assert start in text and end in text, name
        block = text.split(start, 1)[1].split(end, 1)[0]
        assert block == rendered, name


def test_metadata_json_roundtrip(tmp_path: Path):
    metadata = {
        "generated_at_utc": "2026-02-13T00:00:00Z",
        "snapshot_version": "v1",
        "filters": {"min_budget": 10, "exclude_outliers": True},
        "rows_total_before_filters": 100,
        "rows_after_filters": 80,
        "models_after_filters": 7,
        "envs_after_filters": 10,
    }
    p = tmp_path / "canonical_metadata.json"
    p.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    loaded = json.loads(p.read_text())
    assert loaded["snapshot_version"] == "v1"
    assert loaded["rows_after_filters"] == 80


def test_refresh_links_to_the_written_snapshot(tmp_path: Path):
    source = tmp_path / "input.parquet"
    pd.DataFrame(
        [
            {
                "model": model, "env": "e1", "budget": 10, "use_ppl": False,
                "z_mean": z, "experiment_type": "discovery", "include_prior": True,
                "goal": "prediction", **dict.fromkeys(PROTOCOL_FIELDS, "null"),
            }
            for model in MATCHED_MODELS for z in (-0.2, -0.1, 0.0)
        ]
    ).to_parquet(source, index=False)
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- CANONICAL_METADATA:START -->\n<!-- CANONICAL_METADATA:END -->\n"
        "<!-- CANONICAL_RESULTS:START -->\n<!-- CANONICAL_RESULTS:END -->\n"
    )
    hf_readme = tmp_path / "HF_SPACE_README.md"
    hf_readme.write_text("<!-- CANONICAL_HF_SNAPSHOT:START -->\n<!-- CANONICAL_HF_SNAPSHOT:END -->\n")
    output = tmp_path / "snapshot"

    _, metadata = refresh_canonical_artifacts(
        input_path=source,
        output_dir=output,
        readme_path=readme,
        hf_readme_path=hf_readme,
        min_budget=10,
        exclude_outliers=True,
        snapshot_version="test",
        n_bootstrap=10,
        dry_run=False,
    )

    expected = output / "canonical_runs.parquet"
    assert expected.is_file()
    assert metadata["source_path"] == str(source)
    assert metadata["snapshot_path"] == str(expected)
    for document in (readme, hf_readme):
        assert f"| Bundled snapshot | `{expected}` |" in document.read_text()


@pytest.mark.parametrize("enrich_protocol", [False, True])
def test_refresh_bundled_snapshot_preserves_original_provenance(tmp_path: Path, enrich_protocol):
    source = tmp_path / "canonical_runs.parquet"
    frame = pd.DataFrame(
        [
            {
                "run_id": str(i), "model": model, "env": "e1", "budget": 10,
                "use_ppl": False, "z_mean": 0.1, "experiment_type": "discovery",
                "include_prior": True, "goal": "prediction",
                **dict.fromkeys(PROTOCOL_FIELDS, "null"),
            }
            for i, model in enumerate(MATCHED_MODELS)
        ]
    )
    protocol_input = None
    if enrich_protocol:
        protocol_input = tmp_path / "protocol.csv"
        frame[["run_id", *PROTOCOL_FIELDS]].to_csv(protocol_input, index=False)
        frame = frame.drop(columns=PROTOCOL_FIELDS)
    frame.to_parquet(source, index=False)
    original = {
        "generated_at_utc": "2026-10-24T04:57:05Z",
        "snapshot_version": "2026-10-24",
        "source_path": ".boxing-gym-cache/runs.parquet",
        "source_git_commit": "c43ed1ad383b5f8f14a7d4cbe7b6e1d2cf800977",
        "snapshot_path": str(source),
        "filters": {"min_budget": 10, "exclude_outliers": True},
        "rows_total_before_filters": 5710,
        "rows_after_filters": 7,
        "models_after_filters": 7,
        "envs_after_filters": 1,
    }
    metadata_path = tmp_path / "canonical_metadata.json"
    metadata_path.write_text(json.dumps(original))

    _, metadata = refresh_canonical_artifacts(
        input_path=source,
        output_dir=tmp_path,
        readme_path=tmp_path / "README.md",
        hf_readme_path=tmp_path / "HF_SPACE_README.md",
        min_budget=10,
        exclude_outliers=True,
        snapshot_version="2026-10-26",
        n_bootstrap=10,
        dry_run=False,
        protocol_input=protocol_input,
    )

    written = json.loads(metadata_path.read_text())
    assert written == metadata
    assert written["generated_at_utc"] != original["generated_at_utc"]
    assert written["snapshot_version"] == "2026-10-26"
    for key in original.keys() - {"generated_at_utc", "snapshot_version"}:
        assert written[key] == original[key], key


def test_refresh_bundled_snapshot_counts_the_rows_it_writes(tmp_path: Path):
    """An in-place refresh must describe the snapshot it writes, not the one it replaces.

    The fixture holds one row for each of the seven benchmark models in a single
    environment. The previous metadata claims nine rows over six models in two
    environments, as it would after a snapshot loses rows, so a count carried
    over from that metadata reports a snapshot that no longer exists.
    """
    source = tmp_path / "canonical_runs.parquet"
    pd.DataFrame(
        [
            {
                "run_id": str(i), "model": model, "env": "e1", "budget": 10,
                "use_ppl": False, "z_mean": 0.1, "experiment_type": "discovery",
                "include_prior": True, "goal": "prediction",
                **dict.fromkeys(PROTOCOL_FIELDS, "null"),
            }
            for i, model in enumerate(MATCHED_MODELS)
        ]
    ).to_parquet(source, index=False)
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- CANONICAL_METADATA:START -->\n<!-- CANONICAL_METADATA:END -->\n"
        "<!-- CANONICAL_RESULTS:START -->\n<!-- CANONICAL_RESULTS:END -->\n"
    )
    hf_readme = tmp_path / "HF_SPACE_README.md"
    hf_readme.write_text(
        "<!-- CANONICAL_HF_SNAPSHOT:START -->\n<!-- CANONICAL_HF_SNAPSHOT:END -->\n"
    )
    original = {
        "generated_at_utc": "2026-10-24T04:57:05Z",
        "snapshot_version": "2026-10-24",
        "source_path": ".boxing-gym-cache/runs.parquet",
        "source_git_commit": "c43ed1ad383b5f8f14a7d4cbe7b6e1d2cf800977",
        "snapshot_path": str(source),
        "filters": {"min_budget": 10, "exclude_outliers": True},
        "rows_total_before_filters": 5710,
        "rows_after_filters": 9,
        "models_after_filters": 6,
        "envs_after_filters": 2,
    }
    metadata_path = tmp_path / "canonical_metadata.json"
    metadata_path.write_text(json.dumps(original))

    _, metadata = refresh_canonical_artifacts(
        input_path=source,
        output_dir=tmp_path,
        readme_path=readme,
        hf_readme_path=hf_readme,
        min_budget=10,
        exclude_outliers=True,
        snapshot_version="2026-10-28",
        n_bootstrap=10,
        dry_run=False,
    )

    written = json.loads(metadata_path.read_text())
    assert written == metadata
    assert written["rows_after_filters"] == 7
    assert written["models_after_filters"] == 7
    assert written["envs_after_filters"] == 1
    assert written["rows_total_before_filters"] == 5710
    assert written["source_path"] == ".boxing-gym-cache/runs.parquet"
    assert written["source_git_commit"] == "c43ed1ad383b5f8f14a7d4cbe7b6e1d2cf800977"
    assert written["snapshot_version"] == "2026-10-28"
    assert "| Coverage | 7 runs across 7 models and 1 environments |" in readme.read_text()
