"""Educational Example 03: Topographic step response and wake sand trapping.

Demonstrates non-periodic boundary conditions and flow separation behind a backward step,
illustrating the formation of a shadow dune where sand is trapped in the recirculating wake.
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
from dune_simulator.presets.initial_surfaces import backward_step_1d
from dune_simulator.visualization.plots_1d import plot_simulation_state_1d


def main():
    parser = argparse.ArgumentParser(description="Backward step and wake trapping.")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    env = EARTH_DESERT
    length_m = 200.0
    nx = 500
    dx = length_m / nx

    print("=" * 70)
    print("DUNE SIMULATOR: BACKWARD STEP & WAKE SAND TRAPPING")
    print(f"Domain: {length_m:.1f} m | dx = {dx:.2f} m | Inflow ratio = 1.0 (saturated sand stream)")
    print("=" * 70)

    # Step: height 3.0 m dropping to 1.0 m (H_step = 2.0 m) at x = 50 m
    h_init = backward_step_1d(length_m, nx, step_height=2.0, step_pos=50.0, base_height=1.0)

    cfg = SimulationConfig1D(
        nx=nx,
        length_m=length_m,
        env=env,
        u_star_0=0.45,
        periodic=False,     # Open boundaries
        inflow_ratio=1.0,   # Constant incoming saturated sand flux
        enforce_bedrock=True,
        bedrock_floor=h_init.copy(),  # Initial topography acts as non-erodible bedrock floor
        cfl=0.20,
    )
    sim = DuneSimulator1D(config=cfg, h_init=h_init)

    # Record initial state
    st_init = sim.get_state()

    # Evolve to observe sand trapping in the wake
    print("Evolving sand deposition in wake eddy...")
    snapshots = []
    for state in sim.run(total_time_s=20.0 * 86400.0, record_interval_days=2.5):
        snapshots.append(state)
        print(f"  Day {state.time_days:5.1f} | Sand in wake: {np.sum(state.in_wake)} cells")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Panel 1: Step profiles over time
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(snapshots)))
    for i, st in enumerate(snapshots):
        lbl = f"t = {st.time_days:.0f} d"
        ax1.plot(st.x, st.h, color=colors[i], lw=1.8 if i == len(snapshots) - 1 else 1.2, label=lbl)

    # Overlay final separation streamline
    final_st = snapshots[-1]
    if np.any(final_st.in_wake):
        ax1.plot(final_st.x, final_st.h_tilde, "g--", lw=1.5, label="Separation Streamline")
        ax1.fill_between(final_st.x, final_st.h, final_st.h_tilde, where=final_st.in_wake, color="gray", alpha=0.3)

    ax1.set_ylabel("Elevation $h$ [m]", fontsize=11, fontweight="bold")
    ax1.set_title("Sand Trapping and Shadow Dune Formation in Backward-Step Wake", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Basal shear stress and wake zone
    ax2.plot(final_st.x, final_st.tau, "tab:blue", lw=2.0, label=r"Final Basal Shear Stress $\tau(x)$")
    ax2.fill_between(final_st.x, 0, np.max(final_st.tau) * 1.1, where=final_st.in_wake, color="gray", alpha=0.25, label=r"Recirculating Wake ($\tau=0$)")
    ax2.set_xlabel("Downwind Coordinate $x$ [m]", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Shear Stress $\tau$ [Pa]", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "03_step_topography_response.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_file}")

    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
