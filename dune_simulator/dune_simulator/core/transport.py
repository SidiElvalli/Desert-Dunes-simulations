"""Sediment transport flux solver with non-local saturation relaxation and bedrock limiting.

References:
    - Lettau & Lettau (1978) J. Glaciol. 20, 201-208.
    - Sauermann, Kroy & Herrmann (2001) Phys. Rev. E 64, 031305.
    - Andreotti, Claudin & Pouliquen (2010) Geomorphology 123, 343-348.
    - Durán, Parteli & Herrmann (2010) Earth Surf. Process. Landforms 35, 1591-1600.
"""

from typing import Tuple, Optional
import numpy as np
from .constants import PlanetaryEnvironment


def compute_saturated_flux(
    tau: np.ndarray,
    env: PlanetaryEnvironment,
    c_flux: float = 4.2,
    dh_dx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes saturated mass transport flux q_sat [kg / (m * s)] (Lettau & Lettau 1978 / Durán 2010).

    Includes optional Bagnold-Howard-Iversen gravitational bed-slope modification of threshold
    shear stress when dh_dx is provided:
        tau_th_eff = tau_th * clip(1 + (dh/dx) / tan(theta_repose), 0.2, 2.5)

    Parameters:
        tau: Surface shear stress array [Pa].
        env: PlanetaryEnvironment defining fluid/grain properties and thresholds.
        c_flux: Dimensionless flux coefficient (typically 2.0 - 4.2).
        dh_dx: Optional streamwise bed slope array dh/dx for gravity threshold correction.

    Returns:
        q_sat: Saturated sediment flux array [kg / (m * s)].
    """
    tau = np.asarray(tau, dtype=np.float64)
    tau_th_0 = env.threshold_shear_stress
    rho_f = env.fluid_density
    g = env.gravity

    if dh_dx is not None:
        s_rep = np.tan(np.radians(env.repose_angle_deg))
        tau_th = tau_th_0 * np.clip(1.0 + np.asarray(dh_dx, dtype=np.float64) / s_rep, 0.2, 2.5)
    else:
        tau_th = tau_th_0

    u_star = np.sqrt(np.maximum(0.0, tau) / rho_f)
    mask = tau > tau_th

    q_sat = np.zeros_like(tau, dtype=np.float64)
    if np.ndim(tau_th) == 0:
        q_sat[mask] = (
            c_flux
            * (rho_f / g)
            * (u_star[mask] ** 3)
            * (1.0 - (tau_th / tau[mask]))
        )
    else:
        q_sat[mask] = (
            c_flux
            * (rho_f / g)
            * (u_star[mask] ** 3)
            * (1.0 - (tau_th[mask] / tau[mask]))
        )

    return q_sat


def solve_flux_1d(
    q_sat: np.ndarray,
    dx: float,
    env: PlanetaryEnvironment,
    periodic: bool = True,
    inflow_ratio: float = 0.0,
    h: Optional[np.ndarray] = None,
    enforce_bedrock: bool = False,
    dt: Optional[float] = None,
    bedrock_floor: Optional[object] = None,
) -> np.ndarray:
    """Solves the 1D non-local flux relaxation equation L_sat * dq/dx = q_sat - q.

    Uses the exact Upwind Exponential Euler integration scheme:
        q_i = q_{i-1} * exp(-dx / L_sat) + q_{sat, i} * (1 - exp(-dx / L_sat))

    Parameters:
        q_sat: Saturated flux array of shape (nx,).
        dx: Spatial grid spacing in meters.
        env: PlanetaryEnvironment defining saturation length L_sat and sand density.
        periodic: Whether to use periodic boundary conditions.
        inflow_ratio: Fraction of q_sat[0] entering the upwind boundary when periodic=False.
        h: Optional elevation array of shape (nx,) to enforce non-erodible bedrock.
        enforce_bedrock: If True and h[i] <= floor, limits erosion to available sand.
        dt: Time step [s] required for bedrock limiting.
        bedrock_floor: Elevation of non-erodible bedrock (scalar float or array shape (nx,)). Default 0.0.

    Returns:
        q: Actual sediment flux array [kg / (m * s)] of shape (nx,).
    """
    q_sat = np.asarray(q_sat, dtype=np.float64)
    nx = len(q_sat)
    l_sat = env.saturation_length

    # Exact decay factor across cell dx
    mu = np.exp(-dx / l_sat)
    one_minus_mu = 1.0 - mu

    q = np.zeros(nx, dtype=np.float64)
    rho_bulk = env.bulk_density

    def apply_bedrock_limit(i: int, q_in: float, q_target: float) -> float:
        """Caps erosion by the sand actually available in cell i.

        Entrainment (q_target > q_in) removes (q_target - q_in) * dt / (rho_bulk * dx)
        of bed height from the cell. This must never exceed the sand thickness above
        the bedrock floor -- for *every* cell, not only those already at the floor
        (thin sand skins would otherwise be over-eroded and then clipped, creating sand).
        """
        if enforce_bedrock and h is not None and dt is not None and dt > 0.0 and q_target > q_in:
            if bedrock_floor is not None and hasattr(bedrock_floor, "__getitem__"):
                floor_val = float(bedrock_floor[i])
            else:
                floor_val = float(bedrock_floor if bedrock_floor is not None else 0.0)

            available_sand = max(0.0, float(h[i]) - floor_val)
            max_erodible_flux = q_in + (rho_bulk * dx / dt) * available_sand
            return min(q_target, max_erodible_flux)
        return q_target

    if periodic:
        # Pass 1: Initialize with q_sat[0]
        q[0] = q_sat[0]
        for i in range(1, nx):
            q_ideal = q[i - 1] * mu + q_sat[i] * one_minus_mu
            q[i] = apply_bedrock_limit(i, q[i - 1], q_ideal)

        # Periodic boundary wrap: feed end-of-domain flux back to inlet
        q_wrap_ideal = q[-1] * mu + q_sat[0] * one_minus_mu
        q[0] = apply_bedrock_limit(0, q[-1], q_wrap_ideal)

        # Pass 2: Converge periodic boundary flux (domain L >> L_sat guarantees convergence)
        for i in range(1, nx):
            q_ideal = q[i - 1] * mu + q_sat[i] * one_minus_mu
            q[i] = apply_bedrock_limit(i, q[i - 1], q_ideal)
    else:
        # Fixed inflow at upstream boundary
        q[0] = inflow_ratio * q_sat[0]
        for i in range(1, nx):
            q_ideal = q[i - 1] * mu + q_sat[i] * one_minus_mu
            q[i] = apply_bedrock_limit(i, q[i - 1], q_ideal)

    return q


def solve_flux_2d(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    h: np.ndarray,
    dx: float,
    dy: float,
    env: PlanetaryEnvironment,
    tau_0: float,
    periodic_x: bool = True,
    periodic_y: bool = True,
    inflow_ratio: float = 0.0,
    d_lat: float = 0.15,
    enforce_bedrock: bool = False,
    dt: Optional[float] = None,
    bedrock_floor: Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solves 2D sediment transport fluxes (qx, qy) [kg / (m * s)].

    qx relaxes along the primary wind direction x.
    qy accounts for lateral wind shear deflection and gravity-driven splash diffusion.

    Parameters:
        tau_x: Streamwise shear stress array of shape (nx, ny).
        tau_y: Transverse shear stress array of shape (nx, ny).
        h: Topographic elevation array of shape (nx, ny).
        dx: Grid spacing along x [m].
        dy: Grid spacing along y [m].
        env: PlanetaryEnvironment configuration.
        tau_0: Undisturbed baseline shear stress [Pa].
        periodic_x: Periodic boundaries in x.
        periodic_y: Periodic boundaries in y.
        inflow_ratio: Inflow ratio at x=0 when not periodic.
        d_lat: Dimensionless lateral diffusion coefficient.
        enforce_bedrock: Whether to enforce non-erodible bedrock.
        dt: Time step [s] for bedrock limiting.

    Returns:
        qx: Streamwise mass flux array [kg / (m * s)] of shape (nx, ny).
        qy: Transverse mass flux array [kg / (m * s)] of shape (nx, ny).
    """
    tau_x = np.asarray(tau_x, dtype=np.float64)
    tau_y = np.asarray(tau_y, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    nx, ny = h.shape

    tau_mag = np.sqrt(tau_x ** 2 + tau_y ** 2)
    if periodic_x:
        dh_dx = (np.roll(h, -1, axis=0) - h) / dx
    else:
        dh_dx = np.zeros_like(h)
        dh_dx[:-1, :] = (h[1:, :] - h[:-1, :]) / dx
    q_sat_mag = compute_saturated_flux(tau_mag, env, dh_dx=dh_dx)

    qx = np.zeros_like(h)
    for j in range(ny):
        col_h = h[:, j] if enforce_bedrock else None
        floor_col = (
            bedrock_floor[:, j]
            if (bedrock_floor is not None and hasattr(bedrock_floor, "__getitem__") and getattr(bedrock_floor, "ndim", 1) == 2)
            else bedrock_floor
        )
        qx[:, j] = solve_flux_1d(
            q_sat=q_sat_mag[:, j],
            dx=dx,
            env=env,
            periodic=periodic_x,
            inflow_ratio=inflow_ratio,
            h=col_h,
            enforce_bedrock=enforce_bedrock,
            dt=dt,
            bedrock_floor=floor_col,
        )

    # Transverse flux qy across cell interfaces j + 1/2 using donor-cell finite-volume scheme:
    # Eliminates wide-stencil odd-even decoupling and guarantees zero outgoing flux from bare bedrock cells.
    if periodic_y:
        h_next = np.roll(h, -1, axis=1)
        tau_y_next = np.roll(tau_y, -1, axis=1)
        qx_next = np.roll(qx, -1, axis=1)
        dh_dy_face = (h_next - h) / dy
    else:
        h_next = np.pad(h[:, 1:], ((0, 0), (0, 1)), mode="edge")
        tau_y_next = np.pad(tau_y[:, 1:], ((0, 0), (0, 1)), mode="edge")
        qx_next = np.pad(qx[:, 1:], ((0, 0), (0, 1)), mode="constant")
        dh_dy_face = (h_next - h) / dy
        dh_dy_face[:, -1] = 0.0

    vy_face = 0.5 * (tau_y + tau_y_next) / max(1e-12, tau_0)
    qy_wind = np.where(vy_face >= 0.0, qx * vy_face, qx_next * vy_face)
    qx_diff_donor = np.where(dh_dy_face <= 0.0, qx, qx_next)
    qy_diff = -d_lat * qx_diff_donor * dh_dy_face
    qy = qy_wind + qy_diff

    if not periodic_y:
        qy[:, -1] = 0.0

    return qx, qy
