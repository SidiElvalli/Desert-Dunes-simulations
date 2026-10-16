"""Mass-conserving cellular avalanche slope limiter for granular sand beds.

Relaxes slopes steeper than static angle of repose (34 deg) to dynamic angle of repose (32 deg)
via conservative edge flux redistribution, preserving total sand mass to machine precision.

References:
    - Bouchaud, Cates, Ravi Prakash & Edwards (1994) J. Phys. I France 4, 1383-1410.
    - Durán, Parteli & Herrmann (2010) Earth Surf. Process. Landforms 35, 1591-1600.
"""

from typing import Optional
import numpy as np

from .bedrock import enforce_floor_conservative


def avalanche_1d(
    h: np.ndarray,
    dx: float,
    theta_stat_deg: float = 34.0,
    theta_dyn_deg: float = 32.0,
    periodic: bool = True,
    enforce_bedrock: bool = False,
    max_iter: int = 2000,
    tolerance: float = 1e-6,
    bedrock_floor: Optional[object] = None,
) -> np.ndarray:
    """Relaxes over-steepened 1D slopes while strictly conserving total sand mass.

    Parameters:
        h: Bed elevation array of shape (nx,).
        dx: Spatial grid spacing in meters.
        theta_stat_deg: Static angle of repose (trigger threshold, default 34 deg).
        theta_dyn_deg: Dynamic angle of repose (relaxation target, default 32 deg).
        periodic: Whether to use periodic boundary conditions.
        enforce_bedrock: Whether to prevent bed elevation from dropping below bedrock floor.
        max_iter: Maximum number of relaxation iterations.
        tolerance: Excess slope tolerance for convergence.
        bedrock_floor: Non-erodible bedrock floor elevation (scalar or array). Default 0.0.

    Returns:
        h_relaxed: Slope-limited bed elevation array of shape (nx,).
    """
    h = np.asarray(h, dtype=np.float64).copy()
    n = len(h)

    s_stat = np.tan(np.radians(theta_stat_deg))
    s_dyn = np.tan(np.radians(theta_dyn_deg))
    s_dyn_dx = s_dyn * dx

    def get_floor(idx: int) -> float:
        if bedrock_floor is None:
            return 0.0
        if hasattr(bedrock_floor, "__getitem__"):
            return float(bedrock_floor[idx])
        return float(bedrock_floor)

    for _ in range(max_iter):
        if periodic:
            # diff[i] = h[(i + 1) % n] - h[i]
            diff = np.roll(h, -1) - h
        else:
            # diff has length n - 1
            diff = h[1:] - h[:-1]

        slope = np.abs(diff) / dx
        max_excess = np.max(slope) - s_stat
        if max_excess <= tolerance:
            break

        # Compute conservative edge fluxes:
        # If diff > 0, cell i+1 is higher than cell i -> flux flows right-to-left into cell i.
        # factor 0.25 ensures unconditional stability of the 1D Jacobi relaxation.
        excess = np.maximum(0.0, np.abs(diff) - s_dyn_dx)
        flux_edge = 0.25 * np.sign(diff) * excess

        if enforce_bedrock:
            if periodic:
                for i in range(n):
                    nxt = (i + 1) % n
                    if flux_edge[i] > 0.0:
                        # Moving sand from nxt to i
                        avail = max(0.0, h[nxt] - get_floor(nxt))
                        flux_edge[i] = min(flux_edge[i], avail)
                    elif flux_edge[i] < 0.0:
                        # Moving sand from i to nxt
                        avail = max(0.0, h[i] - get_floor(i))
                        flux_edge[i] = -min(-flux_edge[i], avail)
            else:
                for i in range(n - 1):
                    if flux_edge[i] > 0.0:
                        avail = max(0.0, h[i + 1] - get_floor(i + 1))
                        flux_edge[i] = min(flux_edge[i], avail)
                    elif flux_edge[i] < 0.0:
                        avail = max(0.0, h[i] - get_floor(i))
                        flux_edge[i] = -min(-flux_edge[i], avail)

        if periodic:
            h += flux_edge - np.roll(flux_edge, 1)
        else:
            h[:-1] += flux_edge
            h[1:] -= flux_edge

        if enforce_bedrock:
            h, _ = enforce_floor_conservative(h, bedrock_floor)

    return h


def avalanche_2d(
    h: np.ndarray,
    dx: float,
    dy: float,
    theta_stat_deg: float = 34.0,
    theta_dyn_deg: float = 32.0,
    periodic_x: bool = True,
    periodic_y: bool = True,
    enforce_bedrock: bool = False,
    max_iter: int = 500,
    tolerance: float = 1e-5,
    bedrock_floor: Optional[object] = None,
) -> np.ndarray:
    """Relaxes over-steepened 2D terrain while strictly conserving total sand volume.

    Parameters:
        h: 2D bed elevation array of shape (nx, ny).
        dx: Spatial grid spacing along x [m].
        dy: Spatial grid spacing along y [m].
        theta_stat_deg: Static angle of repose (default 34 deg).
        theta_dyn_deg: Dynamic angle of repose (default 32 deg).
        periodic_x: Periodic in x.
        periodic_y: Periodic in y.
        enforce_bedrock: Prevents elevation from falling below bedrock floor.
        max_iter: Maximum relaxation sweeps.
        tolerance: Convergence tolerance.
        bedrock_floor: Non-erodible bedrock floor elevation (scalar or 2D array). Default 0.0.

    Returns:
        h_relaxed: Slope-limited 2D bed elevation array of shape (nx, ny).
    """
    h = np.asarray(h, dtype=np.float64).copy()
    nx, ny = h.shape

    s_stat = np.tan(np.radians(theta_stat_deg))
    s_dyn = np.tan(np.radians(theta_dyn_deg))
    s_dyn_dx = s_dyn * dx
    s_dyn_dy = s_dyn * dy

    # Relaxation weight 0.15 guarantees 2D 4-neighbor stability (4 * 0.15 = 0.60 < 1.0)
    omega = 0.15
    floor = (
        np.zeros_like(h)
        if bedrock_floor is None
        else np.broadcast_to(np.asarray(bedrock_floor, dtype=np.float64), h.shape)
    )

    for _ in range(max_iter):
        # --- Forward edge differences in x and y (full (nx, ny) arrays) ---
        if periodic_x:
            diff_x = np.roll(h, -1, axis=0) - h
        else:
            diff_x = np.zeros_like(h)
            diff_x[:-1, :] = h[1:, :] - h[:-1, :]

        if periodic_y:
            diff_y = np.roll(h, -1, axis=1) - h
        else:
            diff_y = np.zeros_like(h)
            diff_y[:, :-1] = h[:, 1:] - h[:, :-1]

        sx = diff_x / dx
        sy = diff_y / dy

        # Cell-centered slopes interpolated to orthogonal edges for isotropic gradient magnitude
        if periodic_y:
            sy_center = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2.0 * dy)
        else:
            sy_center = np.zeros_like(h)
            sy_center[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2.0 * dy)
            sy_center[:, 0] = (h[:, 1] - h[:, 0]) / dy
            sy_center[:, -1] = (h[:, -1] - h[:, -2]) / dy

        if periodic_x:
            sx_center = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / (2.0 * dx)
            sy_at_x = 0.5 * (sy_center + np.roll(sy_center, -1, axis=0))
        else:
            sx_center = np.zeros_like(h)
            sx_center[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2.0 * dx)
            sx_center[0, :] = (h[1, :] - h[0, :]) / dx
            sx_center[-1, :] = (h[-1, :] - h[-2, :]) / dx
            sy_at_x = np.zeros_like(h)
            sy_at_x[:-1, :] = 0.5 * (sy_center[:-1, :] + sy_center[1:, :])

        if periodic_y:
            sx_at_y = 0.5 * (sx_center + np.roll(sx_center, -1, axis=1))
        else:
            sx_at_y = np.zeros_like(h)
            sx_at_y[:, :-1] = 0.5 * (sx_center[:, :-1] + sx_center[:, 1:])

        grad_x = np.sqrt(sx ** 2 + sy_at_x ** 2)
        grad_y = np.sqrt(sx_at_y ** 2 + sy ** 2)

        max_slope = max(float(np.max(grad_x)), float(np.max(grad_y)))
        if max_slope - s_stat <= tolerance:
            break

        factor_x = np.where(grad_x > s_dyn, 1.0 - s_dyn / np.maximum(grad_x, 1e-12), 0.0)
        flux_x = omega * diff_x * factor_x

        factor_y = np.where(grad_y > s_dyn, 1.0 - s_dyn / np.maximum(grad_y, 1e-12), 0.0)
        flux_y = omega * diff_y * factor_y

        if enforce_bedrock:
            avail = np.maximum(0.0, h - floor)
            avail_x_next = (
                np.roll(avail, -1, axis=0)
                if periodic_x
                else np.pad(avail[1:, :], ((0, 1), (0, 0)))
            )
            flux_x = np.where(
                flux_x > 0.0,
                np.minimum(flux_x, 0.25 * avail_x_next),
                -np.minimum(-flux_x, 0.25 * avail),
            )
            avail_y_next = (
                np.roll(avail, -1, axis=1)
                if periodic_y
                else np.pad(avail[:, 1:], ((0, 0), (0, 1)))
            )
            flux_y = np.where(
                flux_y > 0.0,
                np.minimum(flux_y, 0.25 * avail_y_next),
                -np.minimum(-flux_y, 0.25 * avail),
            )

        if periodic_x:
            h += flux_x - np.roll(flux_x, 1, axis=0)
        else:
            h[:-1, :] += flux_x[:-1, :]
            h[1:, :] -= flux_x[:-1, :]

        if periodic_y:
            h += flux_y - np.roll(flux_y, 1, axis=1)
        else:
            h[:, :-1] += flux_y[:, :-1]
            h[:, 1:] -= flux_y[:, :-1]

        if enforce_bedrock:
            h, _ = enforce_floor_conservative(h, bedrock_floor)

    return h
