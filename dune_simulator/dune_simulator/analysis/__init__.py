"""Dune morphometric and linear stability analysis tools."""

from .morphometry import (
    find_crest_1d,
    find_brink_1d,
    find_slipface_1d,
    compute_aspect_ratio_1d,
    compute_migration_speed,
    compute_kinematic_brink_speed,
)
from .dispersion import (
    compute_flux_sensitivity_q1,
    linear_growth_rate,
    linear_migration_speed,
    find_neutral_wavenumber,
    find_fastest_growing_wavenumber,
)

__all__ = [
    "find_crest_1d",
    "find_brink_1d",
    "find_slipface_1d",
    "compute_aspect_ratio_1d",
    "compute_migration_speed",
    "compute_kinematic_brink_speed",
    "compute_flux_sensitivity_q1",
    "linear_growth_rate",
    "linear_migration_speed",
    "find_neutral_wavenumber",
    "find_fastest_growing_wavenumber",
]
