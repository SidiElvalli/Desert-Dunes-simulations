"""Tests for flow separation streamline and wake mask calculation."""

import numpy as np
import pytest
from dune_simulator.core.separation import compute_separation_1d, compute_separation_2d
from dune_simulator.core.wind import compute_shear_stress_1d


def test_backward_step_separation():
    """Verifies that separation bubble length L_b = 6 * H_d for a flat-topped backward step."""
    nx = 500
    dx = 0.2  # Fine resolution
    x = np.arange(nx) * dx
    length_m = nx * dx

    # Step: height 3.0 m for x <= 40.0 m, dropping to 1.0 m for x > 40.0 m
    # Relative step height H_d = 2.0 m, flat upwind slope h'_d = 0.0
    h = np.where(x <= 40.0, 3.0, 1.0)

    h_tilde, in_wake = compute_separation_1d(h, dx=dx, periodic=False)

    # Brink index
    brink_idx = int(np.where(x <= 40.0)[0][-1])
    assert brink_idx is not None
    # Envelope must match bed at the brink
    assert np.isclose(h_tilde[brink_idx], h[brink_idx], atol=1e-12)

    # Reattachment: points where in_wake becomes False after the brink
    wake_indices = np.where(in_wake)[0]
    assert len(wake_indices) > 0
    reattach_idx = wake_indices[-1] + 1
    measured_lb = (reattach_idx - brink_idx) * dx

    # Theoretical L_b for H_d = 2.0, nu = 0 is 1.5 * 2.0 / 0.25 = 12.0 m = 6 * H_d
    expected_lb = 12.0
    assert np.isclose(measured_lb, expected_lb, atol=1.5 * dx)

    # Shear stress inside wake must be identically zero
    tau, _ = compute_shear_stress_1d(h_tilde, in_wake, dx=dx, tau_0=1.0, periodic=False)
    assert np.all(tau[in_wake] == 0.0)


def test_2d_separation_wake():
    """Verifies 2D separation on an isolated ridge."""
    nx, ny = 100, 30
    dx = 0.5
    h = np.ones((nx, ny), dtype=np.float64)
    # Add a ridge with a steep lee face
    h[20:30, :] = 3.0

    h_tilde, in_wake = compute_separation_2d(h, dx=dx, periodic=False)
    assert np.any(in_wake)
    # Envelope must never be lower than the topography
    assert np.all(h_tilde >= h - 1e-12)
