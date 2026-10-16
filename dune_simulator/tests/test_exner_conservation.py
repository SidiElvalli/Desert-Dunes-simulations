"""Tests for Exner mass conservation and adaptive time stepping."""

import numpy as np
import pytest
from dune_simulator.core.exner import (
    compute_flux_divergence_1d,
    compute_flux_divergence_2d,
    compute_adaptive_dt,
    exner_step_1d,
)


def test_exner_1d_divergence_conservation():
    """Verifies that 1D upwind flux divergence algebraically integrates to zero on a periodic domain."""
    nx = 200
    dx = 0.5
    rho_bulk = 1643.0

    # Arbitrary smooth flux profile
    x = np.arange(nx) * dx
    q = 0.05 + 0.02 * np.sin(2.0 * np.pi * x / (nx * dx))

    div_q, dh_dt = compute_flux_divergence_1d(q, dx=dx, rho_bulk=rho_bulk, periodic=True)

    # Net mass rate of change over the entire domain must be zero
    total_net_rate = float(np.sum(dh_dt) * dx)
    assert abs(total_net_rate) < 1e-14


def test_exner_1d_multi_step_conservation():
    """Verifies that repeated Exner time integration strictly conserves total sand mass."""
    nx = 200
    dx = 0.5
    rho_bulk = 1643.0

    x = np.arange(nx) * dx
    h = 3.0 + 0.5 * np.cos(2.0 * np.pi * x / (nx * dx))
    initial_mass = float(np.sum(h) * dx)

    dt = 100.0  # seconds
    for _ in range(500):
        # Flux dependent on elevation: q = 0.01 * h
        q = 0.01 * h
        _, dh_dt = compute_flux_divergence_1d(q, dx=dx, rho_bulk=rho_bulk, periodic=True)
        h = exner_step_1d(h, dh_dt, dt)

    final_mass = float(np.sum(h) * dx)
    rel_mass_err = abs(final_mass - initial_mass) / initial_mass
    assert rel_mass_err < 1e-13


def test_adaptive_cfl_timestep():
    """Verifies adaptive time stepping clamps dt within bounds and scales inversely with max dh/dt."""
    dx = 1.0
    dh_dt = np.array([0.0, 1e-4, -2e-4, 0.0])

    dt = compute_adaptive_dt(dh_dt, dx=dx, cfl=0.2, dt_min=10.0, dt_max=3600.0)

    # Expected dt: cfl * dx * tan(32 deg) / max(|dh_dt|)
    expected_dt = 0.2 * 1.0 * np.tan(np.radians(32.0)) / 2e-4
    assert np.isclose(dt, expected_dt, rtol=1e-5)

    # Quiescent bed returns dt_max
    dt_quiescent = compute_adaptive_dt(np.zeros(10), dx=dx, dt_max=3600.0)
    assert dt_quiescent == 3600.0
