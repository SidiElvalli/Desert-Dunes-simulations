"""Tests for DuneSimulator1D, DuneSimulator2D, and initial condition presets."""

import numpy as np
import pytest
from dune_simulator.core.constants import EARTH_DESERT, MARS_DUNES
from dune_simulator.simulation.config import SimulationConfig1D, SimulationConfig2D
from dune_simulator.simulation.simulator_1d import DuneSimulator1D
from dune_simulator.simulation.simulator_2d import DuneSimulator2D
from dune_simulator.presets.initial_surfaces import (
    flat_bed_with_noise,
    gaussian_heap_1d,
    forward_step_1d,
    backward_step_1d,
    two_dunes_1d,
    barchan_seed_2d,
    ridge_2d,
)
from dune_simulator.analysis.morphometry import (
    find_crest_1d,
    find_brink_1d,
    find_slipface_1d,
    compute_aspect_ratio_1d,
)


def test_initial_surface_presets():
    """Verifies that all 1D and 2D preset generators output correct shapes and valid elevations."""
    nx, ny = 128, 64
    length = 200.0

    h_noise = flat_bed_with_noise(length, nx, mean_height=3.0, noise_amplitude=0.05)
    assert h_noise.shape == (nx,)
    assert np.isclose(np.mean(h_noise), 3.0, atol=0.05)

    h_gauss = gaussian_heap_1d(length, nx, height=4.0, sigma=20.0)
    assert h_gauss.shape == (nx,)
    assert np.isclose(np.max(h_gauss), 4.0, atol=1e-2)

    h_fwd = forward_step_1d(length, nx, step_height=2.0)
    assert h_fwd.shape == (nx,)
    assert np.min(h_fwd) == 0.0
    assert np.max(h_fwd) == 2.0

    h_bwd = backward_step_1d(length, nx, step_height=2.0, base_height=1.0)
    assert h_bwd.shape == (nx,)
    assert np.min(h_bwd) == 1.0
    assert np.max(h_bwd) == 3.0

    h_two = two_dunes_1d(length, nx, h1=2.0, h2=4.0)
    assert h_two.shape == (nx,)
    assert np.isclose(np.max(h_two), 4.0, atol=1e-2)

    h_seed_2d = barchan_seed_2d(length, length, nx, ny, height=3.0, radius=25.0)
    assert h_seed_2d.shape == (nx, ny)
    assert np.isclose(np.max(h_seed_2d), 3.0, atol=1e-2)

    h_ridge = ridge_2d(length, length, nx, ny, height=2.5)
    assert h_ridge.shape == (nx, ny)
    assert np.isclose(np.max(h_ridge), 2.5, atol=1e-2)


def test_dune_simulator_1d_lifecycle():
    """Verifies DuneSimulator1D step(), run(), and morphometry analysis."""
    nx = 200
    length_m = 100.0
    cfg = SimulationConfig1D(nx=nx, length_m=length_m, env=EARTH_DESERT, u_star_0=0.45)
    h_init = gaussian_heap_1d(length_m, nx, height=3.0, sigma=15.0)

    sim = DuneSimulator1D(config=cfg, h_init=h_init)
    state0 = sim.get_state()

    assert state0.time_s == 0.0
    assert state0.step_count == 0
    assert np.isclose(state0.crest_height, 3.0, atol=1e-4)

    # Step forward 10 steps
    for _ in range(10):
        state = sim.step()

    assert state.step_count == 10
    assert state.time_s > 0.0
    assert state.time_days > 0.0

    # Test morphometry functions
    c_idx, c_x, c_h = find_crest_1d(state.h, cfg.dx)
    assert c_idx is not None
    assert c_h > 0.0

    aspect, H, L = compute_aspect_ratio_1d(state.h, cfg.dx)
    assert aspect > 0.0
    assert H > 0.0
    assert L > 0.0

    # Test run generator
    frames = list(sim.run(total_time_years=0.01, record_interval_days=1.0))
    assert len(frames) >= 2


def test_dune_simulator_2d_lifecycle():
    """Verifies DuneSimulator2D step() and run()."""
    nx, ny = 40, 40
    length = 50.0
    cfg = SimulationConfig2D(
        nx=nx,
        ny=ny,
        length_x_m=length,
        length_y_m=length,
        env=EARTH_DESERT,
        u_star_0=0.45,
    )
    h_init = barchan_seed_2d(length, length, nx, ny, height=2.0, radius=10.0)

    sim = DuneSimulator2D(config=cfg, h_init=h_init)
    state0 = sim.get_state()
    assert state0.step_count == 0
    assert np.isclose(state0.max_height, 2.0, atol=1e-4)

    # Run 5 steps
    for _ in range(5):
        state = sim.step()

    assert state.step_count == 5
    assert state.time_s > 0.0
    assert state.total_volume > 0.0
