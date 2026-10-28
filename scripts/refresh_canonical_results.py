#!/usr/bin/env python3
"""Refresh canonical benchmark snapshot and generated docs blocks.

This script makes README + HF Space content consistent by deriving both from a
single pinned snapshot parquet.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from boxing_gym.agents.results_io import get_model_display_name
from boxing_gym.analysis.stats import bootstrap_ci, compare_models

DEFAULT_INPUT = Path(".boxing-gym-cache/runs.parquet")
DEFAULT_OUTPUT_DIR = Path("scripts/streamlit_app/demo_data")
DEFAULT_README = Path("README.md")
DEFAULT_HF_README = Path("HF_SPACE_README.md")

CANONICAL_PARQUET = "canonical_runs.parquet"
CANONICAL_METADATA = "canonical_metadata.json"
CANONICAL_METRICS = "canonical_metrics.json"
MATCHED_MODELS = (
    "anthropic/glm-4.7",
    "bedrock/qwen.qwen3-32b-v1:0",
    "deepseek/deepseek-chat",
    "gpt-4o",
    "gpt-5.1-codex-mini",
    "kimi-for-coding",
    "openai/MiniMax-M2.1",
)
TASK_FIELDS = ["env", "experiment_type", "budget", "use_ppl", "include_prior", "goal"]
PROTOCOL_FIELDS = [
    "num_evals", "env_params", "com_limit", "effective_check_eig", "requested_budgets",
    "goal_original", "norm_factors", "temperature", "configured_seeds",
    "aggregation_mode", "aggregation_n_seeds", "observed_seed_ids",
]
MATCHED_FIELDS = TASK_FIELDS + PROTOCOL_FIELDS


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def apply_canonical_filters(
    df: pd.DataFrame,
    *,
    min_budget: int = 10,
    exclude_outliers: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply canonical publication filters."""
    required = {"model", "env", "budget", "z_mean"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input data missing required columns: {', '.join(missing)}")

    out = df.copy()
    stats: dict[str, Any] = {
        "rows_total_before_filters": int(len(out)),
    }

    if exclude_outliers and "is_outlier" in out.columns:
        out = out[~out["is_outlier"].fillna(False)]

    out["budget"] = pd.to_numeric(out["budget"], errors="coerce")
    out["z_mean"] = pd.to_numeric(out["z_mean"], errors="coerce")
    out = out[out["budget"] >= min_budget]
    out = out.dropna(subset=["model", "env", "z_mean"])

    if "use_ppl" not in out.columns:
        out["use_ppl"] = False
    out["use_ppl"] = out["use_ppl"].fillna(False).astype(bool)

    stats.update(
        {
            "rows_after_filters": int(len(out)),
            "models_after_filters": int(out["model"].nunique()),
            "envs_after_filters": int(out["env"].nunique()),
        }
    )
    return out, stats


def _build_comparison_lookup(df: pd.DataFrame, reference_model: str) -> dict[str, dict[str, Any]]:
    if df["model"].nunique() < 2:
        return {}
    try:
        comparison = compare_models(
            df[["model", "z_mean"]],
            model_col="model",
            score_col="z_mean",
            reference_model=reference_model,
        )
    except Exception:
        return {}
    return {c["model"]: c for c in comparison.get("comparisons", [])}


def compute_leaderboard(df: pd.DataFrame, *, n_bootstrap: int = 10_000) -> list[dict[str, Any]]:
    """Compute mean z leaderboard with bootstrap CIs and significance."""
    rows: list[dict[str, Any]] = []
    grouped = df.groupby("model")["z_mean"]
    means = grouped.mean().sort_values()
    if means.empty:
        return rows

    comparison_lookup = _build_comparison_lookup(df, means.index[0])

    for rank, model in enumerate(means.index.tolist(), 1):
        scores = df.loc[df["model"] == model, "z_mean"].dropna().to_numpy()
        n = int(len(scores))
        if n == 0:
            continue

        if n >= 2:
            ci = bootstrap_ci(scores, confidence=0.95, n_bootstrap=n_bootstrap)
            ci_low = float(ci.ci_low)
            ci_high = float(ci.ci_high)
            mean_val = float(ci.mean)
        else:
            mean_val = float(scores.mean())
            ci_low = mean_val
            ci_high = mean_val

        comp = comparison_lookup.get(model, {})
        rows.append(
            {
                "rank": rank,
                "model_raw": model,
                "model_display": get_model_display_name(str(model)),
                "mean_z": mean_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
                "p_adjusted_vs_best": comp.get("p_adjusted"),
                "significant_vs_best": bool(comp.get("significant_fdr", False)),
            }
        )

    return rows


def compute_champions_model_only(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Best model per environment, averaging across PPL settings."""
    if df.empty:
        return []
    agg = df.groupby(["env", "model"], as_index=False)["z_mean"].mean()
    idx = agg.groupby("env")["z_mean"].idxmin()
    champions = agg.loc[idx].sort_values("z_mean")
    return [
        {
            "env": str(r.env),
            "model_raw": str(r.model),
            "model_display": get_model_display_name(str(r.model)),
            "z_mean": float(r.z_mean),
        }
        for r in champions.itertuples(index=False)
    ]


def compute_champions_model_plus_ppl(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Best (model, use_ppl) configuration per environment."""
    if df.empty:
        return []
    agg = df.groupby(["env", "model", "use_ppl"], as_index=False)["z_mean"].mean()
    idx = agg.groupby("env")["z_mean"].idxmin()
    champions = agg.loc[idx].sort_values("z_mean")
    return [
        {
            "env": str(r.env),
            "model_raw": str(r.model),
            "model_display": get_model_display_name(str(r.model)),
            "use_ppl": bool(r.use_ppl),
            "z_mean": float(r.z_mean),
        }
        for r in champions.itertuples(index=False)
    ]


def compute_key_findings(
    leaderboard: list[dict[str, Any]],
    champions_model_only: list[dict[str, Any]],
) -> dict[str, Any]:
    if not leaderboard:
        return {}
    best = leaderboard[0]
    n_env_total = len(champions_model_only)
    n_beating = sum(1 for row in champions_model_only if row["z_mean"] < 0)
    sig_worse = [row["model_display"] for row in leaderboard if row.get("significant_vs_best")]
    return {
        "best_model": best["model_display"],
        "best_model_mean_z": best["mean_z"],
        "n_env_beating_baseline_model_only": n_beating,
        "n_env_total": n_env_total,
        "significantly_worse_models": sig_worse,
    }


def compute_metrics(df: pd.DataFrame, *, n_bootstrap: int = 10_000) -> dict[str, Any]:
    matched = compute_matched_comparison(df)
    leaderboard = compute_leaderboard(df, n_bootstrap=n_bootstrap)
    champions_model_only = compute_champions_model_only(df)
    champions_model_plus_ppl = compute_champions_model_plus_ppl(df)
    key_findings = compute_key_findings(leaderboard, champions_model_only)
    return {
        "matched_comparison": matched,
        "leaderboard": leaderboard,
        "champions_model_only": champions_model_only,
        "champions_model_plus_ppl": champions_model_plus_ppl,
        "key_findings": key_findings,
    }


def compute_matched_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Compare the seven snapshot models on shared task-and-replication cells."""
    missing = sorted(set(MATCHED_FIELDS + ["model", "z_mean"]) - set(df.columns))
    if missing:
        raise ValueError(f"Matched comparison missing fields: {', '.join(missing)}")
    selected = df.loc[df.model.isin(MATCHED_MODELS) & np.isfinite(df.z_mean)]
    if selected[MATCHED_FIELDS].isna().any().any():
        raise ValueError("Matched protocol values must be recorded; use JSON null for absent settings")
    means = selected.groupby(MATCHED_FIELDS + ["model"]).z_mean.mean().unstack("model")
    means = means.reindex(columns=MATCHED_MODELS).dropna()
    if means.empty:
        raise ValueError("No shared finite cells for the seven-model cohort")
    counts = selected.groupby(MATCHED_FIELDS + ["model"]).size().unstack("model", fill_value=0)
    counts = counts.reindex(index=means.index, columns=MATCHED_MODELS)
    environments = means.groupby(level="env").mean()
    matrix = environments.to_numpy()
    point = matrix.mean(axis=0)
    draws = np.random.default_rng(20261026).integers(0, len(environments), size=(200_000, len(environments)))
    bounds = np.quantile(matrix[draws].mean(axis=1), [0.025, 0.975], axis=0)
    models = [
        {
            "model_raw": model,
            "model_display": get_model_display_name(model),
            "mean_z": float(point[i]),
            "ci_low": float(bounds[0, i]),
            "ci_high": float(bounds[1, i]),
            "n": int(counts[model].sum()),
        }
        for i, model in enumerate(MATCHED_MODELS)
    ]
    return {
        "cell_fields": MATCHED_FIELDS,
        "n_cells": len(means),
        "n_environments": len(environments),
        "n_rows": int(counts.to_numpy().sum()),
        "budgets": sorted(int(budget) for budget in means.index.get_level_values("budget").unique()),
        "bootstrap_draws": 200_000,
        "bootstrap_seed": 20261026,
        "models": sorted(models, key=lambda row: row["mean_z"]),
    }


def build_metadata(
    *,
    source_path: Path,
    snapshot_path: Path,
    stats: dict[str, Any],
    min_budget: int,
    exclude_outliers: bool,
    snapshot_version: str,
) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "source_path": str(source_path),
        "snapshot_path": str(snapshot_path),
        "source_git_commit": _git_commit(),
        "filters": {
            "min_budget": int(min_budget),
            "exclude_outliers": bool(exclude_outliers),
        },
        "snapshot_version": snapshot_version,
        **stats,
    }


def _rows_to_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    if body:
        return "\n".join([header_line, sep_line, body])
    return "\n".join([header_line, sep_line])


def render_metadata_block(metadata: dict[str, Any]) -> str:
    filters = metadata.get("filters", {})
    min_budget = filters.get("min_budget", 10)
    exclude_outliers = filters.get("exclude_outliers", True)
    snapshot_version = metadata.get("snapshot_version", "unknown")
    return _rows_to_markdown_table(
        ["Snapshot", "Value"],
        [
            ["Snapshot version", f"`{snapshot_version}`"],
            ["Bundled snapshot", f"`{metadata['snapshot_path']}`"],
            [
                "Filters",
                f"`budget >= {min_budget}`, "
                f"`exclude_outliers={'true' if exclude_outliers else 'false'}`",
            ],
            [
                "Coverage",
                f"{metadata.get('rows_after_filters', 0):,} runs across "
                f"{metadata.get('models_after_filters', 0)} models and "
                f"{metadata.get('envs_after_filters', 0)} environments",
            ],
        ],
    )


def render_readme_results_block(metrics: dict[str, Any], metadata: dict[str, Any]) -> str:
    matched = metrics["matched_comparison"]
    budget_label = "budget" if len(matched["budgets"]) == 1 else "budgets"
    budgets = ", ".join(str(value) for value in matched["budgets"])
    pooled = {row["model_raw"]: row for row in metrics["leaderboard"]}
    leaderboard_rows = []
    for row in matched["models"]:
        leaderboard_rows.append(
            [
                row["model_display"],
                f"{row['mean_z']:+.3f}",
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
                str(row["n"]),
                f"{pooled[row['model_raw']]['mean_z']:+.3f}",
                str(pooled[row["model_raw"]]["n"]),
            ]
        )

    model_only_rows = [
        [
            row["env"],
            row["model_display"],
            f"{row['z_mean']:+.3f}",
            "✓ Beats baseline" if row["z_mean"] < 0 else "Above baseline",
        ]
        for row in metrics.get("champions_model_only", [])
    ]
    model_plus_ppl_rows = [
        [
            row["env"],
            row["model_display"],
            "true" if row["use_ppl"] else "false",
            f"{row['z_mean']:+.3f}",
            "✓ Beats baseline" if row["z_mean"] < 0 else "Above baseline",
        ]
        for row in metrics.get("champions_model_plus_ppl", [])
    ]

    findings = metrics.get("key_findings", {})

    parts = [
        "### Matched and pooled scores in this snapshot",
        "",
        _rows_to_markdown_table(
            ["Model", "Matched mean z", "95% environment-cluster interval", "Matched rows", "Pooled mean z", "Pooled rows"],
            leaderboard_rows,
        ),
        "",
        "### Lowest mean score in each environment",
        "",
        _rows_to_markdown_table(
            ["Environment", "Model", "Mean z", "Status"],
            model_only_rows,
        ),
        "",
        "### Lowest mean score by model and PyMC setting",
        "",
        _rows_to_markdown_table(
            ["Environment", "Model", "PyMC refinement", "Mean z", "Status"],
            model_plus_ppl_rows,
        ),
        "",
        "### Snapshot comparisons",
        "",
        _rows_to_markdown_table(
            ["Finding", "Value"],
            [
                [
                    "Environments beating baseline (model only)",
                    f"{findings.get('n_env_beating_baseline_model_only', 0)}/{findings.get('n_env_total', 0)}",
                ],
            ],
        ),
        "",
        "### Definitions",
        "",
        _rows_to_markdown_table(
            ["Result", "Definition"],
            [
                ["Matched comparison", f"{matched['n_cells']} shared cells across {matched['n_environments']} environments; {budget_label} {budgets}"],
                ["Pooled mean score", "Mean standardised prediction error across each model's filtered runs; lower is better"],
                ["Matched cell", "Same environment, mode, budget, PyMC setting, prior, goal, evaluation size, simulator parameters, communication limit, information-gain setting, budget schedule, original goal, normalisation, temperature, configured and observed seed lists, aggregation mode and seed count; finite scores from all seven models"],
                ["Matched mean score", "Mean over records within each cell, cells within each environment, then environments with equal weight"],
                [
                    "Model within environment",
                    "Mean over all runs for each model within an environment, pooling PyMC settings",
                ],
                [
                    "Model and PyMC setting",
                    "Mean over each `(model, use_ppl)` combination within an environment; `use_ppl=true` adds PyMC model refinement (Box's Loop)",
                ],
            ],
        ),
        "",
        "### Statistical methods",
        "",
        _rows_to_markdown_table(
            ["Method", "Definition"],
            [
                ["Environment-cluster bootstrap", "95% percentile intervals from 200,000 resamples of exchangeable environments; seed 20261026"],
            ],
        ),
        "",
        _rows_to_markdown_table(
            ["Snapshot", "Value"],
            [["Snapshot version", f"`{metadata.get('snapshot_version', 'unknown')}`"]],
        ),
    ]
    return "\n".join(parts).strip()


def render_hf_snapshot_block(metrics: dict[str, Any], metadata: dict[str, Any]) -> str:
    best = metrics.get("key_findings", {}).get("best_model", "n/a")
    best_z = metrics.get("key_findings", {}).get("best_model_mean_z")
    best_z_text = f"{best_z:+.3f}" if isinstance(best_z, (int, float)) else "n/a"
    filters = metadata.get("filters", {})
    rows = [
        ["Generated", str(metadata.get("generated_at_utc", "unknown"))],
        ["Bundled snapshot", f"`{metadata['snapshot_path']}`"],
        [
            "Filters",
            f"`budget >= {filters.get('min_budget', 10)}`, "
            f"`exclude_outliers={'true' if filters.get('exclude_outliers', True) else 'false'}`",
        ],
        [
            "Coverage",
            f"{metadata.get('rows_after_filters', 0):,} runs across "
            f"{metadata.get('models_after_filters', 0)} models and "
            f"{metadata.get('envs_after_filters', 0)} environments",
        ],
        ["Lowest pooled mean", f"{best} (`z={best_z_text}`)"],
        ["Leaderboard", "Bundled snapshot"],
        ["Sweep analysis", "Selected Weights & Biases sweeps"],
    ]
    return "## Snapshot\n\n" + _rows_to_markdown_table(["Snapshot", "Value"], rows)


def replace_marked_block(path: Path, marker_name: str, new_content: str) -> None:
    start = f"<!-- {marker_name}:START -->"
    end = f"<!-- {marker_name}:END -->"

    text = path.read_text()
    pattern = re.compile(rf"{re.escape(start)}(.*?){re.escape(end)}", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Marker block not found in {path}: {marker_name}")

    replacement = f"{start}\n{new_content.strip()}\n{end}"
    updated = pattern.sub(replacement, text, count=1)
    path.write_text(updated)


def refresh_canonical_artifacts(
    *,
    input_path: Path,
    output_dir: Path,
    readme_path: Path,
    hf_readme_path: Path,
    min_budget: int,
    exclude_outliers: bool,
    snapshot_version: str,
    n_bootstrap: int,
    dry_run: bool,
    protocol_input: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    raw_df = pd.read_parquet(input_path)
    if protocol_input is not None:
        protocol = pd.read_csv(protocol_input, dtype=str, keep_default_na=False)
        required = ["run_id"] + PROTOCOL_FIELDS
        missing = sorted(set(required) - set(protocol.columns))
        if missing:
            raise ValueError(f"Protocol input missing fields: {', '.join(missing)}")
        if set(PROTOCOL_FIELDS) & set(raw_df.columns):
            raise ValueError("Input snapshot already contains protocol fields")
        raw_df = raw_df.merge(protocol[required], on="run_id", how="left", validate="many_to_one")
    filtered_df, stats = apply_canonical_filters(
        raw_df,
        min_budget=min_budget,
        exclude_outliers=exclude_outliers,
    )
    metrics = compute_metrics(filtered_df, n_bootstrap=n_bootstrap)
    snapshot_path = output_dir / CANONICAL_PARQUET
    if input_path.resolve() == snapshot_path.resolve():
        metadata = json.loads((output_dir / CANONICAL_METADATA).read_text())
        if metadata["filters"] != {
            "min_budget": min_budget,
            "exclude_outliers": exclude_outliers,
        }:
            raise ValueError("Use the original source data to change snapshot filters")
        metadata.update(generated_at_utc=utc_now_iso(), snapshot_version=snapshot_version)
        metadata.update(
            {
                "rows_after_filters": stats["rows_after_filters"],
                "models_after_filters": stats["models_after_filters"],
                "envs_after_filters": stats["envs_after_filters"],
            }
        )
    else:
        metadata = build_metadata(
            source_path=input_path,
            snapshot_path=snapshot_path,
            stats=stats,
            min_budget=min_budget,
            exclude_outliers=exclude_outliers,
            snapshot_version=snapshot_version,
        )

    if dry_run:
        return metrics, metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(output_dir / CANONICAL_PARQUET, index=False)
    (output_dir / CANONICAL_METADATA).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / CANONICAL_METRICS).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )

    if readme_path.exists():
        replace_marked_block(readme_path, "CANONICAL_METADATA", render_metadata_block(metadata))
        replace_marked_block(
            readme_path, "CANONICAL_RESULTS", render_readme_results_block(metrics, metadata)
        )
    if hf_readme_path.exists():
        replace_marked_block(
            hf_readme_path,
            "CANONICAL_HF_SNAPSHOT",
            render_hf_snapshot_block(metrics, metadata),
        )

    return metrics, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh canonical snapshot and generated result docs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input parquet file.")
    parser.add_argument("--protocol-input", type=Path, help="CSV with run_id and recorded matching protocol fields.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where canonical artifacts are written.",
    )
    parser.add_argument("--readme", type=Path, default=DEFAULT_README, help="README path.")
    parser.add_argument(
        "--hf-readme", type=Path, default=DEFAULT_HF_README, help="HF Space README path."
    )
    parser.add_argument("--min-budget", type=int, default=10, help="Minimum budget filter.")
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include rows flagged as outliers in canonical artifacts.",
    )
    parser.add_argument(
        "--snapshot-version", default=utc_now_iso().split("T")[0], help="Snapshot label."
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10_000, help="Bootstrap resamples for CIs."
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics, metadata = refresh_canonical_artifacts(
        input_path=args.input,
        output_dir=args.output_dir,
        readme_path=args.readme,
        hf_readme_path=args.hf_readme,
        min_budget=args.min_budget,
        exclude_outliers=not args.include_outliers,
        snapshot_version=args.snapshot_version,
        n_bootstrap=args.n_bootstrap,
        dry_run=args.dry_run,
        protocol_input=args.protocol_input,
    )
    print(
        f"Canonical snapshot prepared: {metadata['rows_after_filters']:,} runs, "
        f"{metadata['models_after_filters']} models, {metadata['envs_after_filters']} envs"
    )
    matched = metrics["matched_comparison"]
    print(f"Matched comparison: {matched['n_cells']} cells, {matched['n_rows']:,} rows")
    if args.dry_run:
        print("Dry run complete. No files written.")
    else:
        print(f"Wrote artifacts to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
