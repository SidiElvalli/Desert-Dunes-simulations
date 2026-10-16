"""Linear stability analysis and dispersion relation for aeolian and subaqueous sand dunes.

References:
    - Fourrière, Claudin & Andreotti (2010) J. Fluid Mech. 649, 287-328.
    - Claudin & Andreotti (2006) Earth Planet. Sci. Lett. 252, 30-44.
    - Kroy, Sauermann & Herrmann (2002) Phys. Rev. E 66, 031302.
"""

import math
from typing import Tuple, Union
import numpy as np
from scipy import optimize
from ..core.constants import PlanetaryEnvironment, EARTH_DESERT


def compute_flux_sensitivity_q1(
    env: PlanetaryEnvironment,
    tau_0: float,
    c_flux: float = 4.2,
) -> float:
    """Computes linearized sand transport capacity sensitivity Q_1 = tau_0 * (dq_sat/dtau)|_{tau_0} [kg/(m*s)].

    Parameters:
        env: PlanetaryEnvironment configuration.
        tau_0: Baseline undisturbed shear stress [Pa].
        c_flux: Dimensionless flux coefficient.

    Returns:
        Q_1: Linearized flux sensitivity in kg / (m * s).
    """
    tau_th = env.threshold_shear_stress
    if tau_0 <= tau_th:
        return 0.0

    rho_f = env.fluid_density
    g = env.gravity
    u_star_0 = math.sqrt(tau_0 / rho_f)

    # q_sat(tau) = C * (rho_f / g) * (tau / rho_f)^(1.5) * (1 - tau_th / tau)
    # dq_sat / dtau = C * (rho_f / g) * (u_* / rho_f) * (1.5 - 0.5 * tau_th / tau)
    # Q_1 = tau_0 * (dq_sat / dtau) = C * (rho_f / g) * (u_*^3) * (1.5 - 0.5 * (tau_th / tau_0))
    return c_flux * (rho_f / g) * (u_star_0 ** 3) * (1.5 - 0.5 * (tau_th / tau_0))


def linear_growth_rate(
    k: Union[float, np.ndarray],
    env: PlanetaryEnvironment,
    tau_0: float,
    A: float = 3.2,
    B: float = 1.6,
    c_flux: float = 4.2,
) -> Union[float, np.ndarray]:
    """Computes linear stability temporal growth rate sigma(k) [1/s] (Fourrière et al. 2010).

    sigma(k) = (Q_1 / rho_bulk) * (k^2 * (B - A * |k| * L_sat)) / (1 + (k * L_sat)^2)

    Positive sigma(k) indicates exponential growth of bedforms of wavenumber k.
    Negative sigma(k) indicates decay/damping (smoothing by flux relaxation).
    """
    k_arr = np.asarray(k, dtype=np.float64)
    l_sat = env.saturation_length
    q1 = compute_flux_sensitivity_q1(env, tau_0, c_flux=c_flux)
    rho_bulk = env.bulk_density

    numerator = (k_arr ** 2) * (B - A * np.abs(k_arr) * l_sat)
    denominator = 1.0 + (k_arr * l_sat) ** 2
    sigma = (q1 / rho_bulk) * (numerator / denominator)

    return float(sigma) if np.ndim(k) == 0 else sigma


def linear_migration_speed(
    k: Union[float, np.ndarray],
    env: PlanetaryEnvironment,
    tau_0: float,
    A: float = 3.2,
    B: float = 1.6,
    c_flux: float = 4.2,
) -> Union[float, np.ndarray]:
    """Computes linear wave phase migration speed c(k) [m/s].

    c(k) = (Q_1 / rho_bulk) * (|k| * (A + B * |k| * L_sat)) / (1 + (k * L_sat)^2)
    """
    k_arr = np.asarray(k, dtype=np.float64)
    l_sat = env.saturation_length
    q1 = compute_flux_sensitivity_q1(env, tau_0, c_flux=c_flux)
    rho_bulk = env.bulk_density

    numerator = np.abs(k_arr) * (A + B * np.abs(k_arr) * l_sat)
    denominator = 1.0 + (k_arr * l_sat) ** 2
    speed = (q1 / rho_bulk) * (numerator / denominator)

    return float(speed) if np.ndim(k) == 0 else speed


def find_neutral_wavenumber(
    env: PlanetaryEnvironment,
    A: float = 3.2,
    B: float = 1.6,
) -> Tuple[float, float]:
    """Finds cut-off neutral wavenumber k_cut and minimum unstable wavelength lambda_min [m].

    Returns:
        (k_cut, lambda_min) where lambda_min = 2 * pi * (A / B) * L_sat.
    """
    l_sat = env.saturation_length
    k_cut = B / (A * l_sat)
    lambda_min = 2.0 * math.pi / k_cut
    return k_cut, lambda_min


def find_fastest_growing_wavenumber(
    env: PlanetaryEnvironment,
    tau_0: float,
    A: float = 3.2,
    B: float = 1.6,
    c_flux: float = 4.2,
) -> Tuple[float, float, float]:
    """Finds the wavenumber k_max and emergent wavelength lambda_max maximizing growth rate sigma(k).

    Returns:
        (k_max, lambda_max, sigma_max) in (rad/m, m, 1/s).
    """
    k_cut, _ = find_neutral_wavenumber(env, A=A, B=B)

    def neg_sigma(k_val: float) -> float:
        return -linear_growth_rate(k_val, env, tau_0, A=A, B=B, c_flux=c_flux)

    res = optimize.minimize_scalar(neg_sigma, bounds=(1e-7, k_cut), method="bounded")
    k_max = float(res.x)
    lambda_max = 2.0 * math.pi / k_max
    sigma_max = -float(res.fun)

    return k_max, lambda_max, sigma_max
