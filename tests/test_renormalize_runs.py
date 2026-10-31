"""Tests for renormalize_runs.py goal-keyed baselines.

Each run's z-scores must be normalized with the constants of the goal the run
actually evaluated (config.envs.goal_name), not with the environment's
direct-goal constants, and a stored norm_factors pair that differs from the
goal's own constants must not be kept on the strength of its version tag.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.renormalize_runs import load_current_baselines, renormalize_file

# Real multi-seed run: results/death_process/env=death_process_goal=infection
# _model=openai_MiniMax-M2.1_exp=oed_prior=true_seed=s1-2-3-4-5_20260117_051801
# -egy46v5a.json (InfectionRate parameter-goal run), in the state the runner
# wrote it: no stored norm_factors, z-scores from InfectionRate's own constants.
INFECTION_RUN = {
    "config": {
        "envs": {
            "env_name": "death_process",
            "goal_name": "infection",
        }
    },
    "data": {
        "z_results": [
            {
                "budget": 0,
                "z_mean": 0.025309226426352572,
                "z_std": 0.29683674801230886,
                "raw_mean": 0.3347519636926922,
            },
            {
                "budget": 1,
                "z_mean": -0.14225348200674015,
                "z_std": 0.03390889215771223,
                "raw_mean": 0.04034578039746555,
            },
            {
                "budget": 3,
                "z_mean": -0.15290304328345586,
                "z_std": 0.0141453608852054,
                "raw_mean": 0.021634596276814834,
            },
            {
                "budget": 5,
                "z_mean": -0.15764492202793923,
                "z_std": 0.004090430990853177,
                "raw_mean": 0.013303157641890004,
            },
            {
                "budget": 7,
                "z_mean": -0.15627452404419967,
                "z_std": 0.004734652180615825,
                "raw_mean": 0.015710934669135514,
            },
            {
                "budget": 10,
                "z_mean": -0.1511941778591414,
                "z_std": 0.018305165759207362,
                "raw_mean": 0.0246370575764809,
            },
        ],
    },
}

# Real multi-seed run: results/survival/env=survival_goal=direct
# _model=anthropic_glm-4.7_exp=oed_prior=true_seed=s1-2-3-4-5_20260112_183346
# -61vl9bly.json, stamped with the pair (0.2604, 0.43885286828275377) that
# differs from the survival direct goal's constants (0.287, 0.446464).
SURVIVAL_RUN = {
    "config": {
        "envs": {
            "env_name": "survival",
            "goal_name": "direct",
        }
    },
    "data": {
        "norm_factors": {
            "mu": 0.2604,
            "sigma": 0.43885286828275377,
            "version": "v2_2026-01-27",
        },
        "z_results": [
            {
                "budget": 0,
                "z_mean": 0.09023525391313078,
                "z_std": 0.0,
                "raw_mean": 0.30000000000000004,
            },
            {
                "budget": 1,
                "z_mean": 0.3181020567240669,
                "z_std": 0.27907869810309255,
                "raw_mean": 0.4,
            },
            {
                "budget": 3,
                "z_mean": 0.4548221384106285,
                "z_std": 0.12480778800613135,
                "raw_mean": 0.45999999999999996,
            },
            {
                "budget": 5,
                "z_mean": 0.3181020567240669,
                "z_std": 0.22786680281093613,
                "raw_mean": 0.4,
            },
            {
                "budget": 7,
                "z_mean": 0.22695533559969244,
                "z_std": 0.30571539654047614,
                "raw_mean": 0.36,
            },
            {
                "budget": 10,
                "z_mean": 0.13580861447531795,
                "z_std": 0.40762052872063487,
                "raw_mean": 0.32,
            },
        ],
    },
}

# Real multi-seed run: results/death_process/env=death_process_goal=direct
# _model=anthropic_glm-4.7_exp=oed_prior=true_seed=s1-2-3-4-5_20260109_205358
# -oflfla1v.json (direct-goal run already normalized with its goal's constants).
DIRECT_RUN = {
    "config": {
        "envs": {
            "env_name": "death_process",
            "goal_name": "direct",
        }
    },
    "data": {
        "norm_factors": {"mu": 223.3291, "sigma": 190.646, "version": "v2_2026-01-27"},
        "z_results": [
            {"budget": 0, "z_mean": -0.5792363857620931, "raw_mean": 112.9},
            {"budget": 1, "z_mean": 3.0549337515604837, "raw_mean": 805.74},
            {"budget": 3, "z_mean": -0.8108698844979702, "raw_mean": 68.74000000000001},
            {"budget": 5, "z_mean": -0.8348934674737473, "raw_mean": 64.16},
            {"budget": 7, "z_mean": -0.8719254534582421, "raw_mean": 57.1},
            {"budget": 10, "z_mean": -0.7031309337725419, "raw_mean": 89.28},
        ],
    },
}


def _write_run(tmp_path: Path, run: dict) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    return path


def test_parameter_goal_run_normalized_with_its_own_goal_constants(tmp_path):
    """An infection-rate run is scored with InfectionRate's norm constants.

    The aggregate raw mean at budget 10 is 0.0246370575764809, so the correct
    score is (0.0246370575764809 - 0.2902838787350395) / 1.756991075450312 =
    -0.15119417785914144. Scoring it with the number-infected direct-goal
    constants (223.3291, 190.646) yields -1.171304212742064 instead.
    """
    path = _write_run(tmp_path, INFECTION_RUN)
    baselines = load_current_baselines()

    result = renormalize_file(path, baselines, dry_run=False)

    assert result is not None
    written = json.loads(path.read_text())
    by_budget = {zr["budget"]: zr for zr in written["data"]["z_results"]}
    assert by_budget[10]["z_mean"] == -0.15119417785914144
    assert written["data"]["norm_factors"] == {
        "mu": 0.2902838787350395,
        "sigma": 1.756991075450312,
        "version": "v2_2026-01-27",
    }
    # the runner's original score is preserved as v1
    assert by_budget[10]["z_mean_v1"] == -0.1511941778591414


def test_survival_run_with_stale_constants_is_renormalized(tmp_path):
    """A survival run stamped with the old pair is rewritten with the goal's.

    The stored norm_factors (0.2604, 0.43885286828275377) carry the current
    version tag but predate the survival direct goal's constants
    (0.287, 0.446464), so the version tag alone cannot skip the file. The
    aggregate raw mean at budget 10 is 0.32, so the correct score is
    (0.32 - 0.287) / 0.446464 = 0.07391413417431199.
    """
    path = _write_run(tmp_path, SURVIVAL_RUN)
    baselines = load_current_baselines()

    result = renormalize_file(path, baselines, dry_run=False)

    written = json.loads(path.read_text())
    assert written["data"]["norm_factors"] == {
        "mu": 0.287,
        "sigma": 0.446464,
        "version": "v2_2026-01-27",
    }
    by_budget = {zr["budget"]: zr for zr in written["data"]["z_results"]}
    assert by_budget[10]["z_mean"] == 0.07391413417431199
    assert by_budget[0]["z_mean"] == 0.029117689220183634
    # the previously stored score is preserved as v1
    assert by_budget[10]["z_mean_v1"] == 0.13580861447531795
    assert result is not None


def test_direct_goal_run_is_left_byte_identical(tmp_path):
    """A direct-goal run already holding its goal's constants is untouched."""
    path = _write_run(tmp_path, DIRECT_RUN)
    before = path.read_bytes()
    baselines = load_current_baselines()

    result = renormalize_file(path, baselines, dry_run=False)

    assert result is None
    assert path.read_bytes() == before
