"""Tests for Jackson-Hunt Fourier basal shear stress perturbation solver."""

import numpy as np
import pytest
from dune_simulator.core.wind import (
    compute_shear_stress_1d,
    compute_shear_stress_2d,
    analytical_gaussian_shear_perturbation,
)


def test_analytical_gaussian_benchmark():
    """Verifies 1D FFT shear stress perturbation against exact continuous Dawson integral formula."""
    # Use L/R >= 80 to ensure the continuous Hilbert transform's algebraic 1/x^2 tail
    # is well contained within the periodic domain.
    nx = 2048
    length_m = 1000.0
    dx = length_m / nx
    x = np.arange(nx) * dx

    x0 = length_m / 2.0
    H = 1.0
    R = 10.0
    A = 3.2
    B = 1.6

    h = H * np.exp(-((x - x0) / R) ** 2)
    in_wake = np.zeros(nx, dtype=bool)

    tau, tau_pert = compute_shear_stress_1d(
        h_tilde=h,
        in_wake=in_wake,
        dx=dx,
        tau_0=1.0,
        A=A,
        B=B,
        periodic=True,
        smooth_length=0.0,
    )

    tau_exact = analytical_gaussian_shear_perturbation(x, x0, H, R, A=A, B=B)

    # 1. Bump region relative error (where bedform dynamics occur)
    bump_mask = np.abs(x - x0) <= 3.0 * R
    rel_error_bump = np.linalg.norm(tau_pert[bump_mask] - tau_exact[bump_mask]) / np.linalg.norm(
        tau_exact[bump_mask]
    )
    assert rel_error_bump < 1e-3  # < 0.1% accuracy

    # 2. Total domain relative error
    rel_error_total = np.linalg.norm(tau_pert - tau_exact) / np.linalg.norm(tau_exact)
    assert rel_error_total < 5e-3


def test_upwind_phase_shift():
    """Verifies that the peak of shear stress is shifted strictly upwind of the crest."""
    nx = 1024
    length_m = 200.0
    dx = length_m / nx
    x = np.arange(nx) * dx

    x0 = 100.0
    H = 1.0
    R = 15.0
    A = 3.2
    B = 1.6

    h = H * np.exp(-((x - x0) / R) ** 2)
    in_wake = np.zeros(nx, dtype=bool)

    tau, _ = compute_shear_stress_1d(h, in_wake, dx, tau_0=1.0, A=A, B=B, periodic=True)

    peak_idx = int(np.argmax(tau))
    x_peak = x[peak_idx]

    # Peak must be upwind of crest x0
    assert x_peak < x0

    # Test single cosine mode phase shift
    k_mode = 2.0 * np.pi / 50.0  # Wavelength 50 m
    h_cos = 0.5 * np.cos(k_mode * (x - x0))
    tau_cos, _ = compute_shear_stress_1d(h_cos, in_wake, dx, tau_0=1.0, A=A, B=B, periodic=True)

    # Analytical shift is -arctan(B / A) / k
    expected_shift = -np.arctan(B / A) / k_mode
    # Find peak near x0
    window = np.abs(x - x0) < 25.0
    x_window = x[window]
    tau_window = tau_cos[window]
    measured_shift = x_window[np.argmax(tau_window)] - x0

    assert np.isclose(measured_shift, expected_shift, atol=1.5 * dx)


def test_2d_shear_stress_perturbation():
    """Verifies 2D shear stress solver on a 2D Gaussian heap."""
    nx, ny = 128, 64
    dx = dy = 1.0
    x = np.arange(nx) * dx - (nx * dx / 2.0)
    y = np.arange(ny) * dy - (ny * dy / 2.0)
    X, Y = np.meshgrid(x, y, indexing="ij")

    H = 2.0
    R = 12.0
    h = H * np.exp(-(X ** 2 + Y ** 2) / (R ** 2))
    in_wake = np.zeros_like(h, dtype=bool)

    tau_x, tau_y, pert_x, pert_y = compute_shear_stress_2d(
        h_tilde=h,
        in_wake=in_wake,
        dx=dx,
        dy=dy,
        tau_0=1.0,
        A=3.2,
        B=1.6,
        A_y=1.6,
    )

    # Streamwise peak must be shifted upwind (x < 0) along symmetry line y = 0
    center_j = ny // 2
    peak_i = int(np.argmax(tau_x[:, center_j]))
    assert x[peak_i] < 0.0

    # Along symmetry axis y=0, lateral perturbation tau_y must be zero
    assert np.allclose(tau_y[:, center_j], 0.0, atol=1e-10)
