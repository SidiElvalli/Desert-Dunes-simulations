"""Morphometric feature detection and dune kinematic analysis."""

from typing import Tuple, Optional
import numpy as np


def find_crest_1d(h: np.ndarray, dx: float) -> Tuple[int, float, float]:
    """Finds peak crest position and height of a 1D dune profile.

    Returns:
        (crest_idx, crest_x, crest_height)
    """
    h = np.asarray(h, dtype=np.float64)
    crest_idx = int(np.argmax(h))
    crest_x = float(crest_idx * dx)
    crest_height = float(h[crest_idx])
    return crest_idx, crest_x, crest_height


def find_brink_1d(
    h: np.ndarray,
    dx: float,
    separation_angle_deg: float = 14.0,
    periodic: bool = True,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Detects primary detachment brink line on the lee face.

    Returns:
        (brink_idx, brink_x, brink_height) or (None, None, None) if no slipface exists.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    h_base = float(np.min(h))
    s_crit = np.tan(np.radians(separation_angle_deg))

    if periodic:
        dh_dx_fwd = (np.roll(h, -1) - h) / dx
    else:
        dh_dx_fwd = np.zeros_like(h)
        dh_dx_fwd[:-1] = (h[1:] - h[:-1]) / dx
        dh_dx_fwd[-1] = dh_dx_fwd[-2]

    # Candidate detachment brinks
    candidates = np.where((dh_dx_fwd < -s_crit) & ((h - h_base) > 1e-4))[0]
    if len(candidates) == 0:
        return None, None, None

    # Choose candidate with highest elevation (primary brink)
    brink_idx = int(candidates[np.argmax(h[candidates])])
    brink_x = float(brink_idx * dx)
    brink_height = float(h[brink_idx])

    return brink_idx, brink_x, brink_height


def find_slipface_1d(
    h: np.ndarray,
    dx: float,
    repose_deg: float = 32.0,
    tolerance_deg: float = 3.0,
    periodic: bool = True,
) -> Tuple[Optional[int], Optional[int], float]:
    """Finds extent and horizontal length of the avalanche slipface where slope ~= 32 deg.

    Returns:
        (start_idx, end_idx, slipface_length_m)
    """
    h = np.asarray(h, dtype=np.float64)
    if periodic:
        dh_dx = (np.roll(h, -1) - h) / dx
    else:
        dh_dx = np.zeros_like(h)
        dh_dx[:-1] = (h[1:] - h[:-1]) / dx
        dh_dx[-1] = dh_dx[-2]

    downward_slope = -dh_dx
    s_min = np.tan(np.radians(repose_deg - tolerance_deg))
    s_max = np.tan(np.radians(repose_deg + tolerance_deg))

    slip_mask = (downward_slope >= s_min) & (downward_slope <= s_max)
    if not np.any(slip_mask):
        return None, None, 0.0

    # Longest contiguous run of repose-angle cells = the dominant slipface.
    # (Taking first..last index would merge the slipfaces of several dunes into one.)
    padded = np.concatenate(([False], slip_mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    run_starts = np.where(edges == 1)[0]
    run_ends = np.where(edges == -1)[0] - 1  # inclusive
    best = int(np.argmax(run_ends - run_starts))
    start_idx = int(run_starts[best])
    end_idx = int(run_ends[best])
    slip_length = float((end_idx - start_idx + 1) * dx)

    return start_idx, end_idx, slip_length


def compute_aspect_ratio_1d(
    h: np.ndarray,
    dx: float,
    threshold_fraction: float = 0.05,
) -> Tuple[float, float, float]:
    """Computes dune height H, base width L, and aspect ratio H / L.

    Returns:
        (aspect_ratio, H, L)
    """
    h = np.asarray(h, dtype=np.float64)
    h_min = float(np.min(h))
    h_max = float(np.max(h))
    H = h_max - h_min

    if H < 1e-6:
        return 0.0, 0.0, float(len(h) * dx)

    # Base width L: region where h - h_min >= threshold_fraction * H
    above_base = (h - h_min) >= (threshold_fraction * H)
    indices = np.where(above_base)[0]

    if len(indices) < 2:
        L = dx
    else:
        L = float((indices[-1] - indices[0] + 1) * dx)

    aspect_ratio = H / L
    return aspect_ratio, H, L


def compute_migration_speed(
    time_s_array: np.ndarray,
    crest_x_array: np.ndarray,
    domain_length_m: Optional[float] = None,
) -> Tuple[float, float]:
    """Calculates migration velocity from a time series of crest positions.

    Handles periodic boundary wraps if domain_length_m is specified.

    Returns:
        (c_m_per_s, c_m_per_year)
    """
    time_s = np.asarray(time_s_array, dtype=np.float64)
    x = np.asarray(crest_x_array, dtype=np.float64)

    if len(time_s) < 2:
        return 0.0, 0.0

    # Unwrap periodic wraps
    if domain_length_m is not None and domain_length_m > 0.0:
        dx_diff = np.diff(x)
        # If jumped backwards by more than half domain, add domain_length_m
        jumps = np.zeros_like(x)
        for i in range(len(dx_diff)):
            if dx_diff[i] < -0.5 * domain_length_m:
                jumps[i + 1:] += domain_length_m
            elif dx_diff[i] > 0.5 * domain_length_m:
                jumps[i + 1:] -= domain_length_m
        x = x + jumps

    # Robust linear regression
    dt = time_s[-1] - time_s[0]
    if dt <= 0.0:
        return 0.0, 0.0

    # Fit line x(t) = c * t + x0
    c_m_per_s, _ = np.polyfit(time_s, x, deg=1)
    c_m_per_year = c_m_per_s * (365.25 * 86400.0)

    return float(c_m_per_s), float(c_m_per_year)


def compute_kinematic_brink_speed(
    q_brink: float,
    H: float,
    rho_bulk: float,
) -> Tuple[float, float]:
    """Calculates kinematic dune migration speed c = q_brink / (rho_bulk * H) (m/s, m/year)."""
    if H <= 0.0 or rho_bulk <= 0.0:
        return 0.0, 0.0
    c_m_per_s = q_brink / (rho_bulk * H)
    c_m_per_year = c_m_per_s * (365.25 * 86400.0)
    return float(c_m_per_s), float(c_m_per_year)
