"""Core physical modules for dune morphodynamics."""

from .constants import (
    PlanetaryEnvironment,
    EARTH_DESERT,
    MARS_DUNES,
    VENUS_DUNES,
    AQUEOUS_FLUME,
)
from .separation import compute_separation_1d, compute_separation_2d
from .wind import (
    compute_shear_stress_1d,
    compute_shear_stress_2d,
    analytical_gaussian_shear_perturbation,
)
from .transport import (
    compute_saturated_flux,
    solve_flux_1d,
    solve_flux_2d,
)
from .avalanche import avalanche_1d, avalanche_2d
from .exner import (
    compute_flux_divergence_1d,
    compute_flux_divergence_2d,
    compute_adaptive_dt,
    exner_step_1d,
    exner_step_2d,
)

__all__ = [
    "PlanetaryEnvironment",
    "EARTH_DESERT",
    "MARS_DUNES",
    "VENUS_DUNES",
    "AQUEOUS_FLUME",
    "compute_separation_1d",
    "compute_separation_2d",
    "compute_shear_stress_1d",
    "compute_shear_stress_2d",
    "analytical_gaussian_shear_perturbation",
    "compute_saturated_flux",
    "solve_flux_1d",
    "solve_flux_2d",
    "avalanche_1d",
    "avalanche_2d",
    "compute_flux_divergence_1d",
    "compute_flux_divergence_2d",
    "compute_adaptive_dt",
    "exner_step_1d",
    "exner_step_2d",
]
