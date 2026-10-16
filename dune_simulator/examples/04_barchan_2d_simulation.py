"""Educational Example 04: 2D Barchan crescent dune evolution on non-erodible bedrock.

Simulates the 2D morphodynamics of an isolated sand mound on bare bedrock,
demonstrating how lateral wind deflection and downwind sand transport
carve out the classic crescentic barchan morphology with forward-pointing horns.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dune_simulator.core.constants import EARTH_DESERT
from dune_simulator.simulation.config import SimulationConfig2D
from dune_simulator.simulation.simulator_2d import DuneSimulator2D
from dune_simulator.presets.initial_surfaces import barchan_seed_2d
from dune_simulator.visualization.plots_2d import (
    plot_topography_2d,
    plot_barchan_evolution_diagnostics,
    plot_surface_3d,
)


def main():
    parser = argparse.ArgumentParser(description="2D Barchan dune simulation.")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    env = EARTH_DESERT
    lx, ly = 160.0, 128.0
    nx, ny = 160, 128
    dx, dy = lx / nx, ly / ny

    print("=" * 75)
    print("DUNE SIMULATOR: 2D BARCHAN CRESCENT FORMATION ON BEDROCK")
    print(f"Domain: {lx}m x {ly}m | Grid: {nx}x{ny} | dx={dx:.2f}m, dy={dy:.2f}m")
    print("=" * 75)

    cfg = SimulationConfig2D(
        nx=nx,
        ny=ny,
        length_x_m=lx,
        length_y_m=ly,
        env=env,
        u_star_0=0.38,
        A=3.2,
        B=1.6,
        A_y=1.6,
        periodic_x=False,
        periodic_y=True,
        enforce_bedrock=True,
        inflow_ratio=0.12,
        d_lat=0.35,
        cfl=0.30,
        dt_max=3600.0,
    )

    # Isolated parabolic sand mound on bedrock floor (H=3.2 m, R=18.0 m at x0=38 m, y0=64 m)
    h_init = barchan_seed_2d(lx, ly, nx, ny, height=3.2, radius=18.0, center_x=38.0, center_y=64.0)
    sim = DuneSimulator2D(config=cfg, h_init=h_init)

    t_max_days = 55.0
    margin_m = 16.0
    sand_threshold_m = 0.12

    print("Simulating 2D barchan morphodynamics...")
    states = []
    stop_reason = f"reached {t_max_days:.0f} days"
    for state in sim.run(total_time_s=t_max_days * 86400.0, record_interval_days=5.0):
        sand = state.h > sand_threshold_m
        front_x = float(state.x[np.where(sand.any(axis=1))[0].max()]) if sand.any() else 0.0
        yc = ny // 2
        center_sand = np.where(state.h[:, yc] > sand_threshold_m)[0]
        toe_c = float(state.x[center_sand[-1]]) if len(center_sand) > 0 else 0.0
        horn_ext = max(0.0, front_x - toe_c)
        if states and front_x > lx - margin_m:
            stop_reason = f"dune horn tips reached x = {front_x:.0f} m (outlet at {lx:.0f} m)"
            break
        states.append(state)
        print(
            f"  Day {state.time_days:5.1f} | Crest: {state.max_height:.2f} m | "
            f"Volume: {state.total_volume:.1f} m^3 | Horn Ext: +{horn_ext:.1f} m | Front: x={front_x:.0f} m"
        )
    print(f"Completed: {stop_reason}")

    final_state = states[-1]
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Comprehensive 2-row publication figure (4 time-lapse maps + 3 physical diagnostic panels)
    fig_2d = plot_barchan_evolution_diagnostics(
        states=states,
        dx=dx,
        dy=dy,
        save_path=os.path.join(out_dir, "04_barchan_2d_simulation.png"),
    )

    # 2. Dual-perspective 3D surface elevation plot of the mature crescentic Barchan
    fig_3d = plot_surface_3d(
        final_state.h,
        dx=dx,
        dy=dy,
        title=f"3D Equilibrium Barchan Dune Surface (t = {final_state.time_days:.0f} days, H = {final_state.max_height:.2f} m)",
        save_path=os.path.join(out_dir, "04_barchan_3d_surface.png"),
    )

    print(f"Saved 2D and 3D barchan visualizations to: {out_dir}")

    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
