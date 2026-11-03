"""Tests for the discount goal's normalization constants.

The discount goal is scored on squared error in k, the discount factor the
agent predicts. Its normalization constants must therefore describe the
reference predictor's error on that same quantity. A constant drawn from the
log-parameter distribution is a different quantity: the log scale and the k
scale differ by a constant offset, so the two error distributions do not
overlap.

The reference predictor is the constant at the mean of the k distribution, the
convention every direct goal uses. Its expected squared error against a fresh
k draw is Var(k), which the environment's own parameters give in closed form:
k = exp(N(k_mean, k_std^2)), so Var(k) = (e^k_std^2 - 1) e^(2 k_mean + k_std^2).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boxing_gym.envs.hyperbolic_temporal_discount import DiscountGoal, TemporalDiscount

# The constants are the median of 5 seeded runs, so a single run differs from
# them by the run-to-run spread of the estimator (a few percent on a
# log-normal tail).
MU_TOLERANCE = 0.10
SIGMA_TOLERANCE = 0.30


def _expected_var_k(env: TemporalDiscount) -> float:
    """Var(k) for k = exp(Normal(k_mean, k_std^2)), in closed form."""
    return (math.exp(env.k_std**2) - 1) * math.exp(2 * env.k_mean + env.k_std**2)


def _reference_error_on_k(n_draws: int, seed: int) -> tuple[float, float]:
    """Mean and spread of the constant-at-mean predictor's squared error on k."""
    np.random.seed(seed)
    env = TemporalDiscount()
    truths = []
    for _ in range(n_draws):
        env.reset()
        truths.append(env.truth[0])
    truths = np.array(truths)
    prediction = float(truths.mean())
    squared_error = (prediction - truths) ** 2
    return float(squared_error.mean()), float(squared_error.std())


def test_discount_goal_norm_factors_measure_error_on_k():
    """The routine's constants must be the reference error on k, not on log k."""
    np.random.seed(42)
    env = TemporalDiscount()
    goal = DiscountGoal(env)

    mu, sigma = goal.get_norm_factors()

    expected_mu, expected_sigma = _reference_error_on_k(20000, seed=7)

    assert mu == pytest.approx(expected_mu, rel=MU_TOLERANCE), (
        f"norm mean {mu:.6g} is not the constant-predictor squared error on k "
        f"({expected_mu:.6g}); a value near 0.25 means the error was measured "
        f"against a log-parameter draw instead of k"
    )
    assert sigma == pytest.approx(expected_sigma, rel=SIGMA_TOLERANCE), (
        f"norm spread {sigma:.6g} is not the spread of the constant-predictor "
        f"squared error on k ({expected_sigma:.6g})"
    )


def test_discount_goal_norm_mean_matches_var_of_k():
    """The reference mean is Var(k), the error of the best constant predictor."""
    np.random.seed(42)
    env = TemporalDiscount()
    goal = DiscountGoal(env)

    mu, _ = goal.get_norm_factors()

    assert mu == pytest.approx(_expected_var_k(env), rel=MU_TOLERANCE)


def test_discount_goal_stored_constants_are_current():
    """The hardcoded constants must be the ones the routine produces."""
    np.random.seed(42)
    env = TemporalDiscount()
    goal = DiscountGoal(env)

    mu, sigma = goal.get_norm_factors()

    assert goal.norm_mu == pytest.approx(mu, rel=MU_TOLERANCE), (
        f"stored norm_mu {goal.norm_mu:.6g} does not match the routine's {mu:.6g}"
    )
    assert goal.norm_sigma == pytest.approx(sigma, rel=SIGMA_TOLERANCE), (
        f"stored norm_sigma {goal.norm_sigma:.6g} does not match the routine's {sigma:.6g}"
    )


def test_discount_goal_reference_error_is_not_on_the_log_scale():
    """k and log k differ by a constant offset, so their error scales differ."""
    np.random.seed(11)
    env = TemporalDiscount()
    goal = DiscountGoal(env)

    mu, _ = goal.get_norm_factors()

    log_scale_error = float(env.k_std**2)
    assert mu != pytest.approx(log_scale_error, rel=0.05), (
        f"norm mean {mu:.6g} equals the log-parameter variance {log_scale_error:.6g}; "
        f"the goal is scored on k, so the constants must not be the log-scale ones"
    )
