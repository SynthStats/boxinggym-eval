import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.refresh_canonical_results import (
    apply_canonical_filters,
    compute_champions_model_only,
    compute_champions_model_plus_ppl,
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
        "rows_after_filters": 1,
        "models_after_filters": 1,
        "envs_after_filters": 1,
        "filters": {"min_budget": 10, "exclude_outliers": True},
    }

    results = render_readme_results_block(metrics, metadata)
    snapshot = render_metadata_block(metadata)

    # Facts and definitions are table content; only section headings stand outside it.
    for block in (snapshot, results):
        assert all(
            not line.strip() or line.startswith(("|", "### ")) for line in block.splitlines()
        ), block

    expected_rows = [
        "| 1 | Model A | +0.100 | [+0.000, +0.200] | Reference | 10 |",
        "| 3 | Model C | +0.500 | [+0.400, +0.600] | n/a | 1 |",
        "| e1 | Model A | -0.100 | ✓ Beats baseline |",
        "| e1 | Model A | true | -0.200 | ✓ Beats baseline |",
        "| Overall leader | Model A |",
        "| Leader mean z | +0.100 |",
        "| Environments beating baseline (model only) | 1/1 |",
        "| Overall ranking | Mean z-score across all filtered runs; lower is better |",
        "| Model-only champion | Mean over all runs for each model within environment; ignores `use_ppl` |",
        "| Model + PPL champion | Mean over each `(model, use_ppl)` combination within environment |",
        "| Welch's t-test | Group comparison with unequal variances |",
        "| Bootstrap CI | 95% confidence intervals from resampling |",
        "| Benjamini-Hochberg | FDR correction for multiple comparisons |",
        "| Snapshot version | `v1` |",
    ]
    p_text = "0.012*" if significantly_worse_models else "0.420"
    expected_rows.append(f"| 2 | Model B | +0.300 | [+0.200, +0.400] | {p_text} | 12 |")
    sig_text = "Model B, Model C" if significantly_worse_models else "None"
    expected_rows.append(f"| Models significantly worse than #1 (FDR < 0.05) | {sig_text} |")
    for row in expected_rows:
        assert row in results
    for row in [
        "| Snapshot version | `v1` |",
        "| Source cache | `.boxing-gym-cache/runs.parquet` |",
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
