import numpy as np
import pandas as pd
import pytest
from scripts.refresh_canonical_results import compute_metrics

MODELS = (
    "anthropic/glm-4.7",
    "bedrock/qwen.qwen3-32b-v1:0",
    "deepseek/deepseek-chat",
    "gpt-4o",
    "gpt-5.1-codex-mini",
    "kimi-for-coding",
    "openai/MiniMax-M2.1",
)


def runs(cells):
    return pd.DataFrame(
        [
            {
                "model": model,
                "env": env,
                "experiment_type": mode,
                "budget": 10,
                "use_ppl": False,
                "include_prior": True,
                "goal": "prediction",
                "z_mean": score,
                "num_evals": "10",
                "env_params": "{}",
                "com_limit": "200",
                "effective_check_eig": "false",
                "requested_budgets": "[10]",
                "goal_original": '"direct"',
                "norm_factors": '{"mu":0,"sigma":1}',
                "temperature": "0.0",
                "configured_seeds": "[1]",
                "aggregation_mode": "null",
                "aggregation_n_seeds": "null",
                "observed_seed_ids": "[]",
            }
            for env, mode, score in cells
            for model in MODELS
        ]
    )


@pytest.mark.parametrize("missing", ["absent", "nan", "infinite"])
def test_missing_finite_model_score_removes_the_cell_for_every_model(missing):
    frame = runs([("a", "discovery", 1.0), ("b", "discovery", 9.0)])
    affected = (frame.model == MODELS[-1]) & (frame.env == "b")
    if missing == "absent":
        frame = frame.loc[~affected]
    else:
        frame.loc[affected, "z_mean"] = np.nan if missing == "nan" else np.inf

    metrics = compute_metrics(frame, n_bootstrap=10)
    assert "matched_comparison" in metrics
    matched = metrics["matched_comparison"]
    assert matched["n_cells"] == matched["n_environments"] == 1
    assert len(matched["models"]) == 7
    for model in matched["models"]:
        assert model["mean_z"] == model["ci_low"] == model["ci_high"] == 1.0
        assert model["n"] == 1


def test_each_environment_has_equal_weight_despite_different_cell_counts():
    frame = runs([("a", "discovery", 2.0), ("a", "oed", 4.0), ("b", "discovery", 10.0)])
    metrics = compute_metrics(frame, n_bootstrap=10)
    assert "matched_comparison" in metrics
    for row in metrics["matched_comparison"]["models"]:
        assert row["mean_z"] == 6.5  # Mean of environment means 3 and 10.
        assert row["ci_low"] == 3.0
        assert row["ci_high"] == 10.0
        assert row["n"] == 3


def test_protocol_fields_cannot_be_silently_omitted():
    frame = runs([("a", "discovery", 1.0)]).drop(columns="env_params")
    with pytest.raises(ValueError, match="env_params"):
        compute_metrics(frame, n_bootstrap=10)


def test_cohort_cannot_shrink_when_a_model_has_no_records():
    frame = runs([("a", "discovery", 1.0)])
    with pytest.raises(ValueError, match="shared|cohort"):
        compute_metrics(frame.loc[frame.model != MODELS[-1]], n_bootstrap=10)
