"""Animation generation utilities for time-evolving sand dunes."""

from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
import numpy as np
from ..simulation.config import SimulationState1D, SimulationState2D
from .plots_2d import _desert_sand_cmap


def animate_simulation_1d(
    states: List[SimulationState1D],
    interval_ms: int = 80,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """Creates a smooth animated sequence of 1D dune profile evolution and migration.

    Parameters:
        states: List of SimulationState1D snapshots recorded during simulation.
        interval_ms: Delay between frames in milliseconds.
        save_path: Optional path to save animation (.mp4 or .gif).

    Returns:
        anim: Matplotlib FuncAnimation instance.
    """
    if len(states) == 0:
        raise ValueError("Cannot animate empty state list.")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = states[0].x
    max_h = max(3.5, float(max(st.crest_height for st in states)) * 1.3)

    (line_bed,) = ax.plot([], [], "tab:brown", lw=2.2, label=r"Bed $h(x)$")
    (line_env,) = ax.plot([], [], "g--", lw=1.5, label=r"Envelope $\tilde{h}(x)$")
    fill_bed = [ax.fill_between([], [], color="#e4c498", alpha=0.55)]

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, max_h)
    ax.set_xlabel("Downwind Coordinate $x$ [m]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Elevation [m]", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    title_text = ax.set_title("", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")

    def init():
        line_bed.set_data([], [])
        line_env.set_data([], [])
        return line_bed, line_env, title_text

    def update(frame_idx: int):
        st = states[frame_idx]
        line_bed.set_data(x, st.h)
        line_env.set_data(x, st.h_tilde)

        # Clear old fill and add updated fill
        fill_bed[0].remove()
        fill_bed[0] = ax.fill_between(x, 0, st.h, color="#e4c498", alpha=0.55)

        title_text.set_text(
            f"Dune Morphodynamics: t = {st.time_days:.1f} days ({st.time_years:.2f} yr) | "
            f"Crest: {st.crest_height:.2f} m"
        )
        return line_bed, line_env, title_text

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(states),
        interval=interval_ms,
        blit=False,
    )

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval_ms)

    return anim


def animate_simulation_2d(
    states: List[SimulationState2D],
    dx: float,
    dy: float,
    interval_ms: int = 100,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """Creates a 2D hillshaded animation of Barchan dune morphogenesis and migration."""
    if len(states) == 0:
        raise ValueError("Cannot animate empty state list.")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    h0 = states[0].h
    nx, ny = h0.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    extent = [y[0], y[-1], x[0], x[-1]]
    vmax = max(1.0, float(max(st.max_height for st in states)))

    ls = LightSource(azdeg=315, altdeg=42)
    cmap = _desert_sand_cmap()
    rgb0 = ls.shade(h0, cmap=cmap, vmin=0.0, vmax=vmax, vert_exag=1.8, blend_mode="overlay", dx=dy, dy=dx)
    im = ax.imshow(rgb0, extent=extent, origin="lower", aspect="equal")

    ax.set_xlabel("Transverse Coordinate $y$ [m]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Downwind Coordinate $x$ [m]", fontsize=11, fontweight="bold")
    title_text = ax.set_title("", fontsize=12, fontweight="bold")

    def update(frame_idx: int):
        st = states[frame_idx]
        rgb = ls.shade(st.h, cmap=cmap, vmin=0.0, vmax=vmax, vert_exag=1.8, blend_mode="overlay", dx=dy, dy=dx)
        im.set_data(rgb)
        title_text.set_text(
            f"2D Barchan Morphodynamics: t = {st.time_days:.1f} d | Crest: {st.max_height:.2f} m"
        )
        return [im, title_text]

    anim = FuncAnimation(fig, update, frames=len(states), interval=interval_ms, blit=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval_ms)

    return anim
