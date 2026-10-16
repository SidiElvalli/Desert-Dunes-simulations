"""Educational Example 05: Planetary scaling comparison (Earth, Mars, Venus, Subaqueous).

Implements the unified dimensional scaling law discovered by Claudin & Andreotti (2006),
demonstrating how planetary atmospheric density, gravity, and grain size govern the
saturation length L_sat and determine the gigantic range of dune sizes across the Solar System:
from 2-centimeter flume ripples to 1-kilometer Martian mega-dunes.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dune_simulator.core.constants import (
    EARTH_DESERT,
    MARS_DUNES,
    VENUS_DUNES,
    AQUEOUS_FLUME,
)
from dune_simulator.analysis.dispersion import (
    linear_growth_rate,
    find_neutral_wavenumber,
    find_fastest_growing_wavenumber,
)


def main():
    parser = argparse.ArgumentParser(description="Planetary scaling comparison.")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    environments = [
        (EARTH_DESERT, "Earth Desert", "tab:brown"),
        (MARS_DUNES, "Mars Dunes", "tab:red"),
        (VENUS_DUNES, "Venus Dunes", "tab:orange"),
        (AQUEOUS_FLUME, "Subaqueous Water Flume", "tab:blue"),
    ]

    print("=" * 85)
    print("PLANETARY DUNE SCALING COMPARISON (Claudin & Andreotti 2006)")
    print("=" * 85)
    print(f"{'Environment':<20} | {'L_drag [m]':<12} | {'L_sat [m]':<12} | {'lambda_min [m]':<14} | {'lambda_max [m]':<14}")
    print("-" * 85)

    data = []
    for env, label, color in environments:
        k_max, lam_max, sig_max = find_fastest_growing_wavenumber(env, env.tau_0)
        _, lam_min = find_neutral_wavenumber(env)
        print(f"{label:<20} | {env.drag_length:<12.5f} | {env.saturation_length:<12.5f} | {lam_min:<14.3f} | {lam_max:<14.3f}")
        data.append((env, label, color, lam_min, lam_max, sig_max))
    print("=" * 85)

    # 4-panel educational plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for idx, (env, label, color, lam_min, lam_max, sig_max) in enumerate(data):
        ax = axes[idx // 2, idx % 2]

        k_cut, _ = find_neutral_wavenumber(env)
        k_vals = np.linspace(1e-6, 1.35 * k_cut, 500)
        sigmas = linear_growth_rate(k_vals, env, env.tau_0)

        # Plot growth rate vs wavelength
        wavelengths = 2.0 * np.pi / k_vals
        ax.plot(wavelengths, sigmas, color=color, lw=2.2, label=r"Linear Growth Rate $\sigma(\lambda)$")
        ax.axhline(0.0, color="k", linestyle="-", lw=0.6)

        # Highlight peak wavelength
        ax.axvline(lam_max, color="k", linestyle="--", lw=1.5, label=f"$\\lambda_{{\\rm max}} = {lam_max:.2f}$ m")
        ax.axvline(lam_min, color="gray", linestyle=":", lw=1.2, label=f"$\\lambda_{{\\rm min}} = {lam_min:.2f}$ m")

        ax.set_xscale("log")
        ax.set_xlabel("Wavelength $\\lambda$ [m]", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"Growth Rate $\sigma$ [s$^{-1}$]", fontsize=10, fontweight="bold")
        ax.set_title(f"{label}\n($L_{{\\rm sat}} = {env.saturation_length:.3f}$ m, $\\rho_f = {env.fluid_density}$ kg/m$^3$)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, which="both")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Universal Dune Instability Dispersion Across Four Planetary Regimes", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "05_planetary_comparison.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved planetary comparison figure to: {out_file}")

    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
