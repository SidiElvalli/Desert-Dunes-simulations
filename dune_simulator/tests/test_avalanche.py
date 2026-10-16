"""Tests for mass-conserving cellular avalanche slope limiter."""

import numpy as np
import pytest
from dune_simulator.core.avalanche import avalanche_1d, avalanche_2d


def test_avalanche_1d_slope_and_conservation():
    """Verifies that 1D avalanche relaxes steep slopes to <= 34 deg and conserves mass to < 1e-13."""
    nx = 200
    dx = 0.5
    x = np.arange(nx) * dx

    # Create an over-steepened triangular sand pile with slope ~ 55 deg
    h_before = np.zeros(nx, dtype=np.float64)
    peak_idx = nx // 2
    h_before[peak_idx] = 10.0  # Slope = 10 / 0.5 = 20 >> tan(34 deg) ~= 0.67
    mass_before = float(np.sum(h_before))

    h_after = avalanche_1d(
        h_before,
        dx=dx,
        theta_stat_deg=34.0,
        theta_dyn_deg=32.0,
        periodic=True,
    )

    mass_after = float(np.sum(h_after))

    # 1. Exact mass conservation to machine precision
    rel_mass_err = abs(mass_after - mass_before) / mass_before
    assert rel_mass_err < 1e-13

    # 2. Maximum slope must not exceed static repose threshold
    slopes = np.abs(np.roll(h_after, -1) - h_after) / dx
    s_stat = np.tan(np.radians(34.0))
    assert np.max(slopes) <= s_stat + 1e-5


def test_avalanche_2d_slope_and_conservation():
    """Verifies that 2D avalanche relaxes steep pyramids to <= 34 deg and conserves total volume."""
    nx, ny = 40, 40
    dx = dy = 0.5

    h_before = np.zeros((nx, ny), dtype=np.float64)
    # Steep pyramid in the center
    h_before[18:22, 18:22] = 8.0
    mass_before = float(np.sum(h_before))

    h_after = avalanche_2d(
        h_before,
        dx=dx,
        dy=dy,
        theta_stat_deg=34.0,
        theta_dyn_deg=32.0,
        periodic_x=True,
        periodic_y=True,
    )

    mass_after = float(np.sum(h_after))

    # 1. Exact volume conservation
    rel_mass_err = abs(mass_after - mass_before) / mass_before
    assert rel_mass_err < 1e-13

    # 2. Maximum slope check along x and y
    slope_x = np.abs(np.roll(h_after, -1, axis=0) - h_after) / dx
    slope_y = np.abs(np.roll(h_after, -1, axis=1) - h_after) / dy
    s_stat = np.tan(np.radians(34.0))

    assert np.max(slope_x) <= s_stat + 1e-4
    assert np.max(slope_y) <= s_stat + 1e-4


def test_avalanche_2d_isotropic_gradient():
    """Verifies that 2D avalanche limits the true Euclidean gradient sqrt(sx^2 + sy^2) along diagonals."""
    nx, ny = 48, 48
    dx = dy = 0.5
    h_before = np.zeros((nx, ny), dtype=np.float64)
    # Narrow central spike
    h_before[22:26, 22:26] = 10.0
    mass_before = float(np.sum(h_before))

    h_after = avalanche_2d(
        h_before,
        dx=dx,
        dy=dy,
        theta_stat_deg=34.0,
        theta_dyn_deg=32.0,
        periodic_x=False,
        periodic_y=False,
    )

    assert np.isclose(np.sum(h_after), mass_before, rtol=1e-13)

    # Check Euclidean gradient magnitude across interior cells
    sx = (h_after[2:, 1:-1] - h_after[:-2, 1:-1]) / (2.0 * dx)
    sy = (h_after[1:-1, 2:] - h_after[1:-1, :-2]) / (2.0 * dy)
    grad_mag = np.sqrt(sx ** 2 + sy ** 2)
    s_stat = np.tan(np.radians(34.0))
    assert float(np.max(grad_mag)) <= s_stat + 1e-2
