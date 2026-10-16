"""Tests for linear stability dispersion relation and numerical growth rate validation."""

import numpy as np
import pytest
from dune_simulator.core.constants import EARTH_DESERT
from dune_simulator.core.wind import compute_shear_stress_1d
from dune_simulator.core.transport import compute_saturated_flux, solve_flux_1d
from dune_simulator.core.exner import compute_flux_divergence_1d, exner_step_1d
from dune_simulator.analysis.dispersion import (
    linear_growth_rate,
    find_neutral_wavenumber,
    find_fastest_growing_wavenumber,
)


def test_analytical_dispersion_wavelengths():
    """Verifies theoretical neutral cut-off and fastest growing wavelengths."""
    env = EARTH_DESERT
    A = 3.2
    B = 1.6
    l_sat = env.saturation_length

    k_cut, lambda_min = find_neutral_wavenumber(env, A=A, B=B)
    # lambda_min = 2 * pi * (A / B) * L_sat = 4 * pi * L_sat ~= 12.566 * L_sat
    expected_ratio = 4.0 * np.pi
    assert np.isclose(lambda_min / l_sat, expected_ratio, rtol=1e-5)

    k_max, lambda_max, sigma_max = find_fastest_growing_wavenumber(
        env, tau_0=env.tau_0, A=A, B=B
    )
    # lambda_max / L_sat should be ~19.78
    assert 19.0 <= (lambda_max / l_sat) <= 20.5
    assert sigma_max > 0.0


def test_numerical_linear_growth_rate_benchmark():
    """Numerically evolves small sinusoidal bed perturbations to verify growth rate matches analytical dispersion relation within 2%."""
    env = EARTH_DESERT
    A = 3.2
    B = 1.6
    tau_0 = env.tau_0
    l_sat = env.saturation_length
    k_cut, _ = find_neutral_wavenumber(env, A=A, B=B)

    # Test 3 wavenumbers in the unstable regime (0 < k < k_cut)
    test_wavenumbers = [0.25 * k_cut, 0.50 * k_cut, 0.75 * k_cut]

    for k_val in test_wavenumbers:
        wavelength = 2.0 * np.pi / k_val
        # Domain with exactly 4 wavelengths
        n_periods = 4
        length_m = n_periods * wavelength
        nx = 512
        dx = length_m / nx
        x = np.arange(nx) * dx

        h0 = 5.0
        eps = 1e-4  # 0.1 mm amplitude (strictly in linear regime)
        h = h0 + eps * np.cos(k_val * x)

        dt = 5.0  # seconds
        n_steps = 10

        h_current = h.copy()
        in_wake = np.zeros(nx, dtype=bool)

        for _ in range(n_steps):
            tau, _ = compute_shear_stress_1d(
                h_tilde=h_current,
                in_wake=in_wake,
                dx=dx,
                tau_0=tau_0,
                A=A,
                B=B,
                periodic=True,
            )
            q_sat = compute_saturated_flux(tau, env)
            q = solve_flux_1d(q_sat, dx, env, periodic=True)
            _, dh_dt = compute_flux_divergence_1d(q, dx, env.bulk_density, periodic=True)
            h_current = exner_step_1d(h_current, dh_dt, dt)

        # Measure numerical amplitude via Fourier transform
        t_total = n_steps * dt
        initial_amp = np.max(h) - np.min(h)
        final_amp = np.max(h_current) - np.min(h_current)

        # Growth rate sigma = (1 / t) * ln(A(t) / A(0))
        sigma_num = (1.0 / t_total) * np.log(final_amp / initial_amp)
        sigma_analytical = linear_growth_rate(k_val, env, tau_0, A=A, B=B)

        rel_err = abs(sigma_num - sigma_analytical) / sigma_analytical
        assert rel_err < 0.02  # Less than 2% relative error!
