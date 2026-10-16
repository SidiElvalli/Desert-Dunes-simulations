"""Educational Example 01: Spontaneous dune formation from flat bed with white noise.

Demonstrates the primary hydrodynamic instability predicted by Jackson & Hunt (1975)
and Fourrière, Claudin & Andreotti (2010), showing how random bed perturbations
spontaneously select a characteristic wavelength lambda_max ~= 19.8 * L_sat.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Ensure package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dune_simulator.core.constants import EARTH_DESERT
from dune_simulator.simulation.config import SimulationConfig1D
from dune_simulator.simulation.simulator_1d import DuneSimulator1D
from dune_simulator.presets.initial_surfaces import flat_bed_with_noise
from dune_simulator.analysis.dispersion import find_fastest_growing_wavenumber


def main():
    parser = argparse.ArgumentParser(description="Spontaneous dune instability from flat sand bed.")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without GUI window")
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    env = EARTH_DESERT
    length_m = 300.0
    nx = 600
    dx = length_m / nx

    print("=" * 70)
    print("DUNE SIMULATOR: SPONTANEOUS PRIMARY INSTABILITY DEMONSTRATION")
    print(f"Domain: {length_m:.1f} m | Grid cells: {nx} (dx = {dx:.2f} m)")
    print(f"Environment: {env.name} | L_sat: {env.saturation_length:.2f} m")

    k_max, lambda_max_theory, sigma_max = find_fastest_growing_wavenumber(env, env.tau_0)
    print(f"Theoretical fastest-growing wavelength: lambda_max = {lambda_max_theory:.2f} m")
    print("=" * 70)

    cfg = SimulationConfig1D(
        nx=nx,
        length_m=length_m,
        env=env,
        u_star_0=0.45,
        periodic=True,
        cfl=0.25,
    )

    # Initial condition: Flat bed with 1 cm noise
    h_init = flat_bed_with_noise(length_m, nx, mean_height=3.0, noise_amplitude=0.01, seed=123)
    sim = DuneSimulator1D(config=cfg, h_init=h_init)

    snapshots = []
    times_days = [0.0, 15.0, 45.0, 90.0]
    target_idx = 0

    print("Simulating bedform emergence...")
    for state in sim.run(total_time_years=0.30, record_interval_days=1.0):
        if target_idx < len(times_days) and state.time_days >= times_days[target_idx]:
            snapshots.append(state)
            print(f"  Captured snapshot at t = {state.time_days:.1f} days (max wave: {np.max(state.h)-np.min(state.h):.3f} m)")
            target_idx += 1
            if target_idx >= len(times_days):
                break

    # Analyze power spectrum of both early linear emergence (t=15d) and mature state (t=90d)
    h_early = snapshots[1].h - np.mean(snapshots[1].h)
    fft_early = np.abs(np.fft.rfft(h_early)) ** 2
    freqs = np.fft.rfftfreq(nx, d=dx)  # cycles per meter

    peak_early_idx = int(np.argmax(fft_early[1:])) + 1
    measured_early_wavelength = 1.0 / freqs[peak_early_idx] if freqs[peak_early_idx] > 0 else 0.0

    final_h = snapshots[-1].h - np.mean(snapshots[-1].h)
    fft_final = np.abs(np.fft.rfft(final_h)) ** 2
    peak_final_idx = int(np.argmax(fft_final[1:])) + 1
    measured_final_wavelength = 1.0 / freqs[peak_final_idx] if freqs[peak_final_idx] > 0 else 0.0

    print("-" * 70)
    print(f"Early linear emergence (t = {snapshots[1].time_days:.1f} d): Measured peak lambda = {measured_early_wavelength:.2f} m")
    print(f"  -> Theory vs Early Measurement error: {abs(measured_early_wavelength - lambda_max_theory) / lambda_max_theory * 100:.1f}%")
    print(f"Mature non-linear coarsening (t = {snapshots[-1].time_days:.1f} d): Measured peak lambda = {measured_final_wavelength:.2f} m")
    print(f"  -> Wavelength coarsening factor: {measured_final_wavelength / measured_early_wavelength:.2f}x")

    # Render educational figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2, 1.3]})

    colors = ["gray", "tab:blue", "tab:purple", "tab:red"]
    for i, st in enumerate(snapshots):
        lbl = f"t = {st.time_days:.0f} d"
        ax1.plot(st.x, st.h, color=colors[i], lw=1.8 if i == len(snapshots) - 1 else 1.2, label=lbl)

    ax1.set_xlabel("Downwind Distance $x$ [m]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Elevation $h$ [m]", fontsize=11, fontweight="bold")
    ax1.set_title("Spontaneous Emergence of Dunes from Flat Bed with White Noise", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Power spectrum plot (avoiding division by zero on DC mode freqs[0] == 0)
    wavelengths = np.zeros_like(freqs)
    pos_mask = freqs > 0
    wavelengths[pos_mask] = 1.0 / freqs[pos_mask]
    valid = (wavelengths >= 5.0) & (wavelengths <= 100.0)

    ax2.plot(wavelengths[valid], fft_early[valid] / np.max(fft_early[valid]), color="tab:blue", lw=2.0,
             label=f"Early Linear Emergence ($t={snapshots[1].time_days:.0f}$ d, peak $\\lambda={measured_early_wavelength:.1f}$ m)")
    ax2.plot(wavelengths[valid], fft_final[valid] / np.max(fft_final[valid]), color="tab:red", lw=2.0, linestyle="--",
             label=f"Mature Coarsened Dunes ($t={snapshots[-1].time_days:.0f}$ d, peak $\\lambda={measured_final_wavelength:.1f}$ m)")
    ax2.axvline(lambda_max_theory, color="k", linestyle="-.", lw=1.5, label=f"Theoretical $\\lambda_{{\\rm max}} = {lambda_max_theory:.1f}$ m")

    ax2.set_xlabel("Wavelength $\\lambda$ [m]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Normalized Spectral Power", fontsize=11, fontweight="bold")
    ax2.set_title("Spatial Power Spectrum: Linear Instability vs Non-Linear Coarsening", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "01_spontaneous_dune_formation.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_file}")

    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
