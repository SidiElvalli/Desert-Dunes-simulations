"""Exner sediment mass conservation equation and adaptive time stepping.

References:
    - Exner (1925) Sitzungsber. Akad. Wiss. Wien 134, 165-180.
    - Paola & Voller (2005) J. Geophys. Res. 110, F04014.
"""

from typing import Tuple, Optional
import numpy as np

from .bedrock import enforce_floor_conservative


def compute_flux_divergence_1d(
    q: np.ndarray,
    dx: float,
    rho_bulk: float,
    periodic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes 1D upwind flux divergence and bed elevation rate of change dh/dt.

    Since wind blows along +x (q >= 0), first-order backward (upwind) differencing
    ensures numerical stability and exact finite-volume conservation.

    Parameters:
        q: Sediment mass flux array [kg / (m * s)] of shape (nx,).
        dx: Spatial grid spacing in meters.
        rho_bulk: Bulk bed sediment density [kg / m^3].
        periodic: Whether to use periodic boundary conditions.

    Returns:
        div_q: Spatial divergence dq/dx [kg / (m^2 * s)].
        dh_dt: Elevation rate of change [m / s] = - (1 / rho_bulk) * div_q.
    """
    q = np.asarray(q, dtype=np.float64)

    if periodic:
        div_q = (q - np.roll(q, 1)) / dx
    else:
        div_q = np.zeros_like(q)
        div_q[1:] = (q[1:] - q[:-1]) / dx
        div_q[0] = 0.0  # Upstream fixed boundary anchor

    dh_dt = -div_q / rho_bulk
    return div_q, dh_dt


def compute_flux_divergence_2d(
    qx: np.ndarray,
    qy: np.ndarray,
    dx: float,
    dy: float,
    rho_bulk: float,
    periodic_x: bool = True,
    periodic_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes 2D flux divergence components and elevation rate of change dh/dt.

    qx is differentiated using 1st-order upwind backward difference.
    qy is differentiated using 2nd-order central difference (diffusive lateral transport).

    Parameters:
        qx: Streamwise flux array [kg / (m * s)] of shape (nx, ny).
        qy: Transverse flux array [kg / (m * s)] of shape (nx, ny).
        dx: Grid spacing along x in meters.
        dy: Grid spacing along y in meters.
        rho_bulk: Bulk bed density [kg / m^3].
        periodic_x: Periodic boundaries in x.
        periodic_y: Periodic boundaries in y.

    Returns:
        div_qx: Streamwise divergence dqx/dx.
        div_qy: Transverse divergence dqy/dy.
        dh_dt: Total rate of elevation change [m / s].
    """
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)

    # 1. dqx / dx (Upwind along axis 0)
    if periodic_x:
        div_qx = (qx - np.roll(qx, 1, axis=0)) / dx
    else:
        div_qx = np.zeros_like(qx)
        div_qx[1:, :] = (qx[1:, :] - qx[:-1, :]) / dx
        div_qx[0, :] = 0.0

    # 2. dqy / dy (Conservative finite-volume difference of interface fluxes qy at j + 1/2)
    if periodic_y:
        div_qy = (qy - np.roll(qy, 1, axis=1)) / dy
    else:
        div_qy = np.zeros_like(qy)
        div_qy[:, 0] = qy[:, 0] / dy
        div_qy[:, 1:-1] = (qy[:, 1:-1] - qy[:, :-2]) / dy
        div_qy[:, -1] = -qy[:, -2] / dy

    dh_dt = -(div_qx + div_qy) / rho_bulk
    return div_qx, div_qy, dh_dt


def compute_adaptive_dt(
    dh_dt: np.ndarray,
    dx: float,
    cfl: float = 0.25,
    dt_min: float = 1.0,
    dt_max: float = 3600.0,
    max_slope_deg: float = 32.0,
) -> float:
    """Calculates adaptive stable time step dt satisfying the morphological CFL condition.

    Ensures that the bed elevation cannot change by more than a small fraction (cfl)
    of the cell height step in a single time iteration.

    Parameters:
        dh_dt: Bed elevation rate of change array [m / s].
        dx: Spatial grid spacing in meters.
        cfl: Courant-Friedrichs-Lewy stability number (typically 0.15 - 0.35).
        dt_min: Minimum allowable time step in seconds.
        dt_max: Maximum allowable time step in seconds.
        max_slope_deg: Characteristic maximum bed slope (default 32 deg).

    Returns:
        dt: Stable adaptive time step in seconds.
    """
    max_rate = float(np.max(np.abs(dh_dt)))
    if max_rate < 1e-12:
        return dt_max

    # Maximum allowed delta_h in a single step
    dh_max_allowed = cfl * dx * np.tan(np.radians(max_slope_deg))
    dt_ideal = dh_max_allowed / max_rate

    return float(np.clip(dt_ideal, dt_min, dt_max))


def exner_step_1d(
    h: np.ndarray,
    dh_dt: np.ndarray,
    dt: float,
    enforce_bedrock: bool = False,
    bedrock_floor: Optional[object] = None,
) -> np.ndarray:
    """Updates 1D bed elevation: h_new = h + dt * dh_dt.

    With ``enforce_bedrock`` the bed is floored *conservatively* (see
    :func:`dune_simulator.core.bedrock.enforce_floor_conservative`), so that the
    floor never creates sand.
    """
    h_new = h + dt * dh_dt
    if enforce_bedrock:
        h_new, _ = enforce_floor_conservative(h_new, bedrock_floor)
    return h_new


def exner_step_2d(
    h: np.ndarray,
    dh_dt: np.ndarray,
    dt: float,
    enforce_bedrock: bool = False,
    bedrock_floor: Optional[object] = None,
) -> np.ndarray:
    """Updates 2D bed elevation: h_new = h + dt * dh_dt (mass-conserving floor)."""
    h_new = h + dt * dh_dt
    if enforce_bedrock:
        h_new, _ = enforce_floor_conservative(h_new, bedrock_floor)
    return h_new
