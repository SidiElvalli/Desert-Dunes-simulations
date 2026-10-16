"""Planetary environmental parameters and physical scalings for dune morphodynamics."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PlanetaryEnvironment:
    """Immutable planetary environmental configuration and scaling laws.
    
    References:
        - Claudin & Andreotti (2006) Earth Planet. Sci. Lett. 252, 30-44.
        - Andreotti, Claudin & Pouliquen (2010) Geomorphology 123, 343-348.
        - Bagnold (1941) The Physics of Blown Sand and Desert Dunes.
    """
    name: str
    gravity: float            # Gravitational acceleration [m/s^2]
    fluid_density: float      # Fluid density rho_f [kg/m^3]
    grain_density: float      # Sediment grain density rho_s [kg/m^3]
    grain_diameter: float     # Mean grain diameter d [m]
    u_star_0: float = 0.40    # Baseline undisturbed shear velocity [m/s]
    repose_angle_deg: float = 34.0      # Static angle of repose [deg]
    dynamic_repose_deg: float = 32.0    # Dynamic angle of repose [deg]
    separation_angle_deg: float = 14.0  # Flow detachment threshold angle [deg]
    packing_fraction: float = 0.62      # Granular packing fraction (phi_p ~= 0.62)

    @property
    def bulk_density(self) -> float:
        """Bulk sand bed density rho_bulk = phi_p * rho_s [kg/m^3]."""
        return self.packing_fraction * self.grain_density

    @property
    def drag_length(self) -> float:
        """Inertial drag length L_drag = (rho_s / rho_f) * d [m]."""
        return (self.grain_density / self.fluid_density) * self.grain_diameter

    @property
    def saturation_length(self) -> float:
        """Sediment flux saturation length L_sat = 2.2 * L_drag [m] (Claudin & Andreotti 2006)."""
        return 2.2 * self.drag_length

    @property
    def threshold_friction_velocity(self) -> float:
        """Bagnold threshold shear velocity u_*th [m/s] with A_Bagnold = 0.1."""
        delta_rho = self.grain_density - self.fluid_density
        return 0.1 * math.sqrt((delta_rho / self.fluid_density) * self.gravity * self.grain_diameter)

    @property
    def threshold_shear_stress(self) -> float:
        """Threshold shear stress tau_th = rho_f * (u_*th ** 2) [Pa]."""
        return self.fluid_density * (self.threshold_friction_velocity ** 2)

    @property
    def tau_0(self) -> float:
        """Undisturbed reference shear stress tau_0 = rho_f * (u_star_0 ** 2) [Pa]."""
        return self.fluid_density * (self.u_star_0 ** 2)

    @property
    def min_dune_wavelength(self) -> float:
        """Theoretical minimum unstable wavelength lambda_min = 4 * pi * L_sat [m] (for A=3.2, B=1.6)."""
        return 4.0 * math.pi * self.saturation_length

    @property
    def fastest_growing_wavelength(self) -> float:
        """Emergent primary instability wavelength lambda_max ~= 19.78 * L_sat [m]."""
        return 19.779 * self.saturation_length


# Standard Planetary Presets (Claudin & Andreotti 2006 Table 1)
EARTH_DESERT = PlanetaryEnvironment(
    name="Earth Desert",
    gravity=9.81,
    fluid_density=1.225,
    grain_density=2650.0,
    grain_diameter=200e-6,
    u_star_0=0.40,
)

MARS_DUNES = PlanetaryEnvironment(
    name="Mars Dunes",
    gravity=3.71,
    fluid_density=0.015,
    grain_density=3000.0,
    grain_diameter=100e-6,
    u_star_0=1.80,
)

VENUS_DUNES = PlanetaryEnvironment(
    name="Venus Dunes",
    gravity=8.87,
    fluid_density=65.0,
    grain_density=2800.0,
    grain_diameter=150e-6,
    u_star_0=0.06,
)

AQUEOUS_FLUME = PlanetaryEnvironment(
    name="Subaqueous Flume",
    gravity=9.81,
    fluid_density=1000.0,
    grain_density=2650.0,
    grain_diameter=200e-6,
    u_star_0=0.035,
)
