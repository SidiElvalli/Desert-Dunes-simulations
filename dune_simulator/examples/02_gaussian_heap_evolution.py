"""Educational Example 02: Evolution of a symmetric Gaussian sand heap into an asymmetric slipfaced dune.

Demonstrates how wind shear upwind shift and non-local saturation transients reshape
a symmetric mound into a natural dune profile with a gentle stoss slope, a sharp brink,
a 32-degree slipface, and a flow separation recirculation wake.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dune_simulator.core.constants import EARTH_DESERT
from dune_simulator.simulation.config import SimulationConfig1D
from dune_simulator.simulation.simulator_1d import DuneSimulator1D
from dune_simulator.presets.initial_surfaces import gaussian_heap_1d
from dune_simulator.visualization.plots_1d import plot_simulation_state_1d, plot_profile_time_series_1d
from dune_simulator.analysis.morphometry import find_crest_1d, find_brink_1d, find_slipface_1d


def main():
    parser = argparse.ArgumentParser(description="Gaussian heap reshaping into slipfaced dune.")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    env = EARTH_DESERT
    length_m = 250.0
    nx = 500
    dx = length_m / nx

    print("=" * 70)
    print("DUNE SIMULATOR: GAUSSIAN HEAP TO ASYMMETRIC SLIPFACED DUNE")
    print(f"Domain: {length_m:.1f} m | dx = {dx:.2f} m | Wind: {env.u_star_0:.2f} m/s")
    print("=" * 70)

    cfg = SimulationConfig1D(
        nx=nx,
        length_m=length_m,
        env=env,
        u_star_0=0.36,
        periodic=True,
        enforce_bedrock=True,
        cfl=0.20,
    )

    # Initial condition: Symmetric Gaussian mound H = 3.0 m, sigma = 12.0 m on bare bedrock (base_height = 0.0)
    h_init = gaussian_heap_1d(length_m, nx, height=3.0, sigma=12.0, center=60.0, base_height=0.0)
    sim = DuneSimulator1D(config=cfg, h_init=h_init)

    recorded_states = []
    print("Simulating dune profile transformation...")
    for state in sim.run(total_time_years=0.30, record_interval_days=10.0):
        recorded_states.append(state)
        b_idx, bx, bh = find_brink_1d(state.h, dx)
        b_str = f"Brink: {bh:.2f} m at x={bx:.1f}m" if b_idx is not None else "No slipface yet"
        print(f"  Day {state.time_days:5.1f} | Crest: {state.crest_height:.2f} m at x={state.crest_x:.1f} m | {b_str}")

    # Final morphometry
    final_state = recorded_states[-1]
    _, slip_end, slip_len = find_slipface_1d(final_state.h, dx)
    print("-" * 70)
    print(f"Final Dune Height: {final_state.crest_height:.2f} m")
    print(f"Active Slipface Length: {slip_len:.2f} m")
    print(f"Flow Separation Bubble Active: {np.any(final_state.in_wake)}")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: 3-panel detailed physical diagnostic of final state
    fig_diag = plot_simulation_state_1d(
        final_state,
        title=f"Steady Dune Morphodynamics at t = {final_state.time_days:.0f} days ({final_state.time_years:.2f} yr)",
        save_path=os.path.join(out_dir, "02_gaussian_heap_diagnostic.png"),
    )

    # Figure 2: Time series migration overlay
    fig_time = plot_profile_time_series_1d(
        recorded_states[::2],
        title="Evolution of Symmetric Heap into a Migrating Asymmetric Dune",
        save_path=os.path.join(out_dir, "02_gaussian_heap_evolution.png"),
    )

    print(f"Saved diagnostic plots to: {out_dir}")

    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
