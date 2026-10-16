"""1D publication-grade visualization plots for dune morphodynamics."""

from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
from ..simulation.config import SimulationState1D


def plot_simulation_state_1d(
    state: SimulationState1D,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Renders a synchronized 3-panel publication-grade figure of a 1D dune state.

    Panels:
    1. Bed profile h(x), aerodynamic envelope h_tilde(x), wake recirculation zone.
    2. Basal shear stress tau(x) / tau_0 showing upwind shift.
    3. Sand transport flux: saturated q_sat(x) vs relaxed q(x).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    x = state.x
    h = state.h
    h_tilde = state.h_tilde
    in_wake = state.in_wake

    # --- Panel 1: Elevation & Aerodynamic Envelope ---
    ax1.plot(x, h, "tab:brown", lw=2.2, label=r"Bed Topography $h(x)$")
    ax1.fill_between(x, 0, h, color="#e4c498", alpha=0.55)

    # Separation envelope where distinct from bed
    has_sep = in_wake | (h_tilde > h + 1e-4)
    if np.any(has_sep):
        ax1.plot(x, h_tilde, "g--", lw=1.5, label=r"Separation Envelope $\tilde{h}(x)$")
        ax1.fill_between(
            x,
            h,
            h_tilde,
            where=has_sep,
            color="gray",
            alpha=0.35,
            label="Recirculation Wake",
        )

    # Crest marker
    c_idx = int(np.argmax(h))
    ax1.plot(x[c_idx], h[c_idx], "r*", markersize=10, label=f"Crest: {h[c_idx]:.2f} m")

    ax1.set_ylabel("Elevation [m]", fontsize=11, fontweight="bold")
    ax1.set_ylim(bottom=0.0, top=max(4.0, float(np.max(h_tilde)) * 1.35))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9)

    t_header = title or f"Dune Profile Evolution at t = {state.time_days:.1f} days ({state.time_years:.2f} yr)"
    ax1.set_title(t_header, fontsize=13, fontweight="bold", pad=10)

    # --- Panel 2: Basal Shear Stress ---
    ax2.plot(x, state.tau, "tab:blue", lw=2.0, label=r"Basal Shear Stress $\tau(x)$ [Pa]")
    ax2.axhline(0.0, color="k", linestyle="-", lw=0.5)

    # Highlight wake shutoff
    if np.any(in_wake):
        ax2.fill_between(x, 0, np.max(state.tau) * 1.2, where=in_wake, color="gray", alpha=0.2)

    ax2.set_ylabel(r"Shear Stress $\tau$ [Pa]", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9)

    # --- Panel 3: Sediment Flux ---
    ax3.plot(x, state.q_sat * 1e3, "tab:orange", linestyle="--", lw=1.8, label=r"Saturated Capacity $q_{\rm sat}(x)$")
    ax3.plot(x, state.q * 1e3, "tab:red", lw=2.2, label=r"Actual Sand Flux $q(x)$")
    ax3.fill_between(x, 0, state.q * 1e3, color="tab:red", alpha=0.2)

    ax3.set_xlabel("Downwind Position $x$ [m]", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Flux $q$ [g / (m s)]", fontsize=11, fontweight="bold")
    ax3.set_xlim(x[0], x[-1])
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_profile_time_series_1d(
    states: List[SimulationState1D],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Overlays multiple 1D profiles colored by timestamp to illustrate dune migration."""
    fig, ax = plt.subplots(figsize=(10, 5))

    n_frames = len(states)
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, n_frames))

    for i, st in enumerate(states):
        label = f"t = {st.time_days:.1f} d" if (i == 0 or i == n_frames - 1 or i == n_frames // 2) else None
        ax.plot(st.x, st.h, color=colors[i], lw=1.8, label=label)

    ax.set_xlabel("Downwind Coordinate $x$ [m]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Elevation $h$ [m]", fontsize=11, fontweight="bold")
    ax.set_title(title or "Dune Migration Time Series", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
