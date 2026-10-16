"""Regression tests for non-erodible bedrock handling and mass conservation.

These cover bugs that the original test-suite did not catch:
  * the bedrock floor used to *create* sand (np.maximum clip) -- this cancelled the entire
    outflow of a 2D barchan and gave +0.5% mass drift in a periodic 1D run;
  * the erosion limiter only triggered for cells already at the floor and used a stale dt;
  * find_slipface_1d merged the slipfaces of several dunes into one.
"""

import numpy as np

from dune_simulator.core.bedrock import enforce_floor_conservative
from dune_simulator.core.constants import EARTH_DESERT
from dune_simulator.analysis.morphometry import find_slipface_1d
from dune_simulator.presets.initial_surfaces import gaussian_heap_1d, barchan_seed_2d
from dune_simulator.simulation.config import SimulationConfig1D, SimulationConfig2D
from dune_simulator.simulation.simulator_1d import DuneSimulator1D
from dune_simulator.simulation.simulator_2d import DuneSimulator2D


def test_enforce_floor_conserves_mass_and_respects_floor():
    rng = np.random.default_rng(0)
    h = rng.uniform(-0.05, 2.0, size=(40, 30))
    total_before = h.sum()
    # Negative parts must be taken from the sand that exists, not created.
    h_new, unresolved = enforce_floor_conservative(h, 0.0)
    assert unresolved == 0.0
    assert h_new.min() >= 0.0
    assert np.isclose(h_new.sum(), total_before, rtol=1e-12)


def test_enforce_floor_no_sand_reports_unresolved():
    h = np.full(10, -0.1)
    h_new, unresolved = enforce_floor_conservative(h, 0.0)
    assert np.all(h_new == 0.0)
    assert np.isclose(unresolved, 1.0)


def test_1d_periodic_bedrock_run_conserves_mass():
    L, nx = 250.0, 500
    dx = L / nx
    cfg = SimulationConfig1D(nx=nx, length_m=L, u_star_0=0.45, periodic=True, enforce_bedrock=True, cfl=0.2)
    h0 = gaussian_heap_1d(L, nx, height=3.0, sigma=20.0, center=80.0, base_height=0.0)
    sim = DuneSimulator1D(cfg, h_init=h0)
    m0 = h0.sum() * dx
    for _ in range(400):
        sim.step()
        assert sim.h.min() >= 0.0
    assert abs(sim.h.sum() * dx - m0) / m0 < 1e-10


def test_2d_bedrock_never_creates_sand_without_inflow():
    """With no upstream sand supply total volume can only decrease (outflow through x = L)."""
    cfg = SimulationConfig2D(
        nx=80, ny=64, length_x_m=80.0, length_y_m=64.0, u_star_0=0.45,
        periodic_x=False, periodic_y=True, enforce_bedrock=True, inflow_ratio=0.0,
        cfl=0.35, dt_max=3600.0,
    )
    h0 = barchan_seed_2d(80.0, 64.0, 80, 64, height=3.0, radius=12.0, center_x=25.0, center_y=32.0)
    sim = DuneSimulator2D(cfg, h_init=h0)
    prev = sim.h.sum()
    for _ in range(60):
        sim.step()
        now = sim.h.sum()
        assert now <= prev + 1e-9 * prev, "bedrock floor created sand"
        assert sim.h.min() >= 0.0
        prev = now


def test_slipface_reports_longest_contiguous_face_not_span_between_dunes():
    dx = 1.0
    n = 400
    h = np.zeros(n)
    t = np.tan(np.radians(32.0))
    # two separate 4 m high slipfaces (ramps of 32 deg) far apart
    for start in (50, 300):
        run = int(round(4.0 / t))
        h[start - 20:start] = 4.0
        h[start:start + run] = 4.0 - t * np.arange(run)
    _, _, length = find_slipface_1d(h, dx, periodic=False)
    assert length < 15.0  # one face (~6-7 m), not the ~250 m between the two
