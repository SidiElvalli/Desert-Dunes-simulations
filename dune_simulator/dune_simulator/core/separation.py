"""Flow separation bubble and aerodynamic envelope calculation for sand dunes.

References:
    - Kroy, Sauermann & Herrmann (2002) Phys. Rev. E 66, 031302.
    - Durán, Parteli & Herrmann (2010) Earth Surf. Process. Landforms 35, 1591-1600.
"""

import math
from typing import Tuple
import numpy as np
from scipy import ndimage


def compute_separation_1d(
    h: np.ndarray,
    dx: float,
    periodic: bool = True,
    separation_angle_deg: float = 14.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the aerodynamic envelope and wake mask for a 1D dune profile.

    Parameters:
        h: Topography elevation array of shape (nx,).
        dx: Spatial grid spacing in meters.
        periodic: Whether to use periodic boundary conditions.
        separation_angle_deg: Critical lee slope angle for flow detachment (default 14 deg).

    Returns:
        h_tilde: Effective aerodynamic envelope profile of shape (nx,).
        in_wake: Boolean array of shape (nx,) indicating points inside the separation bubble.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    h_base = float(np.min(h))

    s_crit = math.tan(math.radians(separation_angle_deg))
    s_bar_prime = 0.25  # Reference maximum negative slope (tan(14.03 deg) ~= 0.25)

    # 1. Forward and backward slope calculations
    if periodic:
        dh_dx_fwd = (np.roll(h, -1) - h) / dx
        dh_dx_bwd = (h - np.roll(h, 1)) / dx
    else:
        dh_dx_fwd = np.zeros_like(h)
        dh_dx_fwd[:-1] = (h[1:] - h[:-1]) / dx
        dh_dx_fwd[-1] = dh_dx_fwd[-2] if n > 1 else 0.0

        dh_dx_bwd = np.zeros_like(h)
        dh_dx_bwd[1:] = (h[1:] - h[:-1]) / dx
        dh_dx_bwd[0] = dh_dx_bwd[1] if n > 1 else 0.0

    h_tilde = h.copy()
    in_wake = np.zeros(n, dtype=bool)

    # 2. Downwind search for detachment points
    i = 0
    while i < n:
        if in_wake[i]:
            i += 1
            continue

        # Check detachment condition: lee slope steeper than s_crit and above base
        if dh_dx_fwd[i] < -s_crit and (h[i] - h_base) > 1e-5:
            h_d = h[i] - h_base

            # Upwind slope at brink (launch angle)
            # Clamped to [-s_bar_prime, s_bar_prime] for C^1 launch continuity (Kroy et al. 2002)
            if dh_dx_bwd[i] < -s_crit:
                # Discontinuous / sharp backward step brink
                h_prime_d = 0.0
            else:
                h_prime_d = float(np.clip(dh_dx_bwd[i], -s_bar_prime, s_bar_prime))

            nu = h_prime_d / s_bar_prime

            # Separation bubble length (Kroy et al. 2002 Eq. 29)
            l_b = (1.5 * h_d / s_bar_prime) * (1.0 + 0.25 * nu + 0.125 * (nu ** 2))
            l_b = max(l_b, dx)

            num_steps = int(math.ceil(l_b / dx))

            # March streamline downwind
            for m in range(1, num_steps + 1):
                idx = (i + m) % n if periodic else i + m
                if not periodic and idx >= n:
                    break

                z = (m * dx) / l_b
                if z > 1.0:
                    z = 1.0

                # Third-order Hermite polynomial streamline:
                # s(0) = h_d, s'(0) = h_prime_d, s(1) = 0, s'(1) = 0
                s_rel = (
                    (2.0 * h_d + h_prime_d * l_b) * (z ** 3)
                    - (3.0 * h_d + 2.0 * h_prime_d * l_b) * (z ** 2)
                    + (h_prime_d * l_b) * z
                    + h_d
                )
                s_val = h_base + s_rel

                if s_val > h[idx]:
                    h_tilde[idx] = max(h_tilde[idx], s_val)
                    in_wake[idx] = True
                else:
                    # Streamline intersected bed (reattachment onto rising topography)
                    break

        i += 1

    # Apply light 1D Gaussian filter (sigma = 1.0 grid cell) to h_tilde when wake is active
    # to smooth the C^1 kink at reattachment and prevent FFT shear spikes, while preserving
    # exact C^0 continuity outside the wake (including at the detachment brink).
    if np.any(in_wake):
        mode = "wrap" if periodic else "nearest"
        h_tilde_smooth = ndimage.gaussian_filter1d(h_tilde, sigma=1.0, mode=mode)
        h_tilde = np.where(in_wake, np.maximum(h, h_tilde_smooth), h)

    return h_tilde, in_wake


def compute_separation_2d(
    h: np.ndarray,
    dx: float,
    periodic: bool = True,
    separation_angle_deg: float = 14.0,
    smooth_transverse: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the 2D aerodynamic envelope and wake mask along each streamwise slice.

    Parameters:
        h: Topography elevation array of shape (nx, ny). Axis 0 is wind direction (+x).
        dx: Spatial grid spacing along x in meters.
        periodic: Whether to use periodic boundary conditions along x.
        separation_angle_deg: Critical lee slope angle for flow detachment.
        smooth_transverse: Whether to apply light transverse smoothing to the envelope.

    Returns:
        h_tilde: 2D aerodynamic envelope of shape (nx, ny).
        in_wake: 2D boolean wake mask of shape (nx, ny).
    """
    h = np.asarray(h, dtype=np.float64)
    nx, ny = h.shape

    h_tilde = np.zeros_like(h)
    in_wake = np.zeros_like(h, dtype=bool)

    for j in range(ny):
        ht_col, wake_col = compute_separation_1d(
            h[:, j],
            dx=dx,
            periodic=periodic,
            separation_angle_deg=separation_angle_deg,
        )
        h_tilde[:, j] = ht_col
        in_wake[:, j] = wake_col

    if smooth_transverse and ny > 3:
        h_tilde_smooth = ndimage.gaussian_filter1d(h_tilde, sigma=0.5, axis=1)
        # Protect shallow flanks: only let h_tilde exceed h where wake is active,
        # so unseparated flanks remain with h_tilde == h and in_wake == False
        h_tilde = np.maximum(h, np.where(in_wake, h_tilde_smooth, h))

    return h_tilde, in_wake
