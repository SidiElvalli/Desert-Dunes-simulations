"""Dune Simulator: A scientifically grounded aeolian and subaqueous sand dune morphodynamics package."""

__version__ = "0.1.0"

# Core planetary environments
from .core.constants import (
    PlanetaryEnvironment,
    EARTH_DESERT,
    MARS_DUNES,
    VENUS_DUNES,
    AQUEOUS_FLUME,
)

# Simulation engines and configs
from .simulation.config import (
    SimulationConfig1D,
    SimulationState1D,
    SimulationConfig2D,
    SimulationState2D,
)
from .simulation.simulator_1d import DuneSimulator1D
from .simulation.simulator_2d import DuneSimulator2D

# Initial surfaces
from .presets.initial_surfaces import (
    flat_bed_with_noise,
    gaussian_heap_1d,
    forward_step_1d,
    backward_step_1d,
    two_dunes_1d,
    barchan_seed_2d,
    ridge_2d,
)

# Analysis tools
from .analysis.morphometry import (
    find_crest_1d,
    find_brink_1d,
    find_slipface_1d,
    compute_aspect_ratio_1d,
    compute_migration_speed,
    compute_kinematic_brink_speed,
)
from .analysis.dispersion import (
    compute_flux_sensitivity_q1,
    linear_growth_rate,
    linear_migration_speed,
    find_neutral_wavenumber,
    find_fastest_growing_wavenumber,
)

__all__ = [
    # Constants
    "PlanetaryEnvironment",
    "EARTH_DESERT",
    "MARS_DUNES",
    "VENUS_DUNES",
    "AQUEOUS_FLUME",
    # Simulation
    "SimulationConfig1D",
    "SimulationState1D",
    "SimulationConfig2D",
    "SimulationState2D",
    "DuneSimulator1D",
    "DuneSimulator2D",
    # Presets
    "flat_bed_with_noise",
    "gaussian_heap_1d",
    "forward_step_1d",
    "backward_step_1d",
    "two_dunes_1d",
    "barchan_seed_2d",
    "ridge_2d",
    # Analysis
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
