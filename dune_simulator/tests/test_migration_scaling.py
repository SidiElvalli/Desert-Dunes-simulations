"""Tests for dune migration velocity and inverse height scaling c ~ 1/H."""

import numpy as np
import pytest
from dune_simulator.core.constants import EARTH_DESERT
from dune_simulator.simulation.config import SimulationConfig1D
from dune_simulator.simulation.simulator_1d import DuneSimulator1D
from dune_simulator.analysis.morphometry import (
    find_crest_1d,
    find_brink_1d,
    compute_migration_speed,
    compute_kinematic_brink_speed,
)


def test_kinematic_brink_speed_formula():
    """Verifies that the kinematic brink relation c = q_brink / (rho_bulk * H) holds."""
    rho_bulk = EARTH_DESERT.bulk_density
    q_brink = 0.05  # kg / (m * s)
    H = 2.5         # meters

    c_s, c_yr = compute_kinematic_brink_speed(q_brink, H, rho_bulk)

    expected_c_s = q_brink / (rho_bulk * H)
    expected_c_yr = expected_c_s * (365.25 * 86400.0)

    assert np.isclose(c_s, expected_c_s, rtol=1e-10)
    assert np.isclose(c_yr, expected_c_yr, rtol=1e-10)


def test_inverse_height_migration_scaling():
    """Verifies that smaller slipfaced dunes migrate faster than larger dunes (c1 > c2)."""
    env = EARTH_DESERT
    length_m = 300.0
    nx = 600
    dx = length_m / nx

    cfg = SimulationConfig1D(
        nx=nx,
        length_m=length_m,
        env=env,
        u_star_0=0.50,
        periodic=True,
        cfl=0.20,
    )

    def create_slipfaced_dune(H: float, L_stoss: float, x0: float) -> np.ndarray:
        x = np.arange(nx) * dx
        h = np.zeros_like(x)
        # Stoss slope (curved cosine)
        upwind = (x >= x0 - L_stoss) & (x <= x0)
        h[upwind] = H * (np.cos(0.5 * np.pi * (x[upwind] - x0) / L_stoss) ** 2)
        # Lee face: angle of repose (~32 deg)
        s_lee = np.tan(np.radians(32.0))
        L_lee = H / s_lee
        downwind = (x > x0) & (x <= x0 + L_lee)
        h[downwind] = H - s_lee * (x[downwind] - x0)
        return h

    # Dune 1: H1 = 2.0 m
    # Dune 2: H2 = 4.0 m
    h_init1 = create_slipfaced_dune(H=2.0, L_stoss=25.0, x0=100.0)
    h_init2 = create_slipfaced_dune(H=4.0, L_stoss=50.0, x0=100.0)

    sim1 = DuneSimulator1D(config=cfg, h_init=h_init1)
    sim2 = DuneSimulator1D(config=cfg, h_init=h_init2)

    # Evolve each dune for ~30 days and record brink migration
    t_rec1, x_rec1 = [], []
    for state in sim1.run(total_time_years=0.08, record_interval_days=1.0):
        b_idx, bx, _ = find_brink_1d(state.h, dx)
        if b_idx is not None:
            t_rec1.append(state.time_s)
            x_rec1.append(bx)

    t_rec2, x_rec2 = [], []
    for state in sim2.run(total_time_years=0.08, record_interval_days=1.0):
        b_idx, bx, _ = find_brink_1d(state.h, dx)
        if b_idx is not None:
            t_rec2.append(state.time_s)
            x_rec2.append(bx)

    # Measure speed over the steady migration window
    half1 = len(t_rec1) // 2
    c1_s, c1_yr = compute_migration_speed(t_rec1[half1:], x_rec1[half1:], domain_length_m=length_m)

    half2 = len(t_rec2) // 2
    c2_s, c2_yr = compute_migration_speed(t_rec2[half2:], x_rec2[half2:], domain_length_m=length_m)

    # 1. Both dunes must migrate downwind
    assert c1_s > 0.0
    assert c2_s > 0.0

    # 2. The smaller dune (H=2.0m) must migrate substantially faster than the larger dune (H=4.0m)
    assert c1_s > c2_s
    speed_ratio = c1_s / c2_s
    # Inverse scaling: speed ratio is greater than 1.5
    assert speed_ratio > 1.5
