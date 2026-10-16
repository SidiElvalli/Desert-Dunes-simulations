"""Tests for sediment transport flux relaxation and bedrock limiting."""

import numpy as np
import pytest
from dune_simulator.core.constants import EARTH_DESERT
from dune_simulator.core.transport import (
    compute_saturated_flux,
    solve_flux_1d,
)


def test_exponential_flux_relaxation_exact():
    """Verifies that Upwind Exponential Euler integration matches the analytical exponential relaxation."""
    nx = 500
    dx = 0.1
    x = np.arange(nx) * dx

    q_sat_val = 0.05
    q_sat = np.full(nx, q_sat_val, dtype=np.float64)

    # Inflow ratio = 0.0 (q[0] starts at 0.0)
    q = solve_flux_1d(
        q_sat=q_sat,
        dx=dx,
        env=EARTH_DESERT,
        periodic=False,
        inflow_ratio=0.0,
    )

    l_sat = EARTH_DESERT.saturation_length
    q_exact = q_sat_val * (1.0 - np.exp(-x / l_sat))

    # Should match to machine precision (< 1e-12)
    max_err = float(np.max(np.abs(q - q_exact)))
    assert max_err < 1e-12


def test_saturated_flux_threshold():
    """Verifies that saturated flux is strictly zero below threshold shear stress."""
    env = EARTH_DESERT
    tau_th = env.threshold_shear_stress

    taus = np.array([0.0, 0.5 * tau_th, 0.999 * tau_th, tau_th, 1.5 * tau_th, 2.0 * tau_th])
    q_sat = compute_saturated_flux(taus, env)

    # Below or at threshold, flux must be zero
    assert np.all(q_sat[:4] == 0.0)
    # Above threshold, flux must be strictly positive and monotonically increasing
    assert q_sat[4] > 0.0
    assert q_sat[5] > q_sat[4]


def test_bedrock_limiter():
    """Verifies that non-erodible bedrock limits flux pickup when sand height is zero."""
    nx = 100
    dx = 0.5
    q_sat = np.full(nx, 0.05, dtype=np.float64)
    h_bedrock = np.zeros(nx, dtype=np.float64)  # Bare bedrock

    q = solve_flux_1d(
        q_sat=q_sat,
        dx=dx,
        env=EARTH_DESERT,
        periodic=False,
        inflow_ratio=0.0,
        h=h_bedrock,
        enforce_bedrock=True,
        dt=10.0,
    )

    # Flux cannot increase on bare bedrock without sand to pick up
    assert np.all(q == 0.0)
