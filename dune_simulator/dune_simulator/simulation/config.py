"""Configuration and state dataclasses for dune morphodynamics simulations."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from ..core.constants import PlanetaryEnvironment, EARTH_DESERT


@dataclass
class SimulationConfig1D:
    """Configuration parameters for 1D profile dune simulation."""
    nx: int = 512
    length_m: float = 200.0
    env: PlanetaryEnvironment = EARTH_DESERT
    u_star_0: float = 0.40
    A: float = 3.2
    B: float = 1.6
    periodic: bool = True
    inflow_ratio: float = 0.0
    enforce_bedrock: bool = False
    cfl: float = 0.25
    dt_max: float = 1800.0
    smooth_length: Optional[float] = None
    bedrock_floor: Optional[object] = None

    @property
    def dx(self) -> float:
        """Spatial resolution [m]."""
        return self.length_m / float(self.nx)

    @property
    def tau_0(self) -> float:
        """Undisturbed shear stress [Pa]."""
        return self.env.fluid_density * (self.u_star_0 ** 2)


@dataclass
class SimulationState1D:
    """Snapshot state of a 1D dune simulation."""
    time_s: float
    step_count: int
    x: np.ndarray
    h: np.ndarray
    h_tilde: np.ndarray
    in_wake: np.ndarray
    tau: np.ndarray
    q_sat: np.ndarray
    q: np.ndarray
    dt: float

    @property
    def time_days(self) -> float:
        """Elapsed time in days."""
        return self.time_s / 86400.0

    @property
    def time_years(self) -> float:
        """Elapsed time in years (365.25 days)."""
        return self.time_s / (365.25 * 86400.0)

    @property
    def crest_height(self) -> float:
        """Peak elevation [m]."""
        return float(np.max(self.h))

    @property
    def crest_x(self) -> float:
        """Position of peak elevation [m]."""
        idx = int(np.argmax(self.h))
        return float(self.x[idx])

    @property
    def mass_volume(self) -> float:
        """Cross-sectional sand volume per unit width [m^2]."""
        dx = float(self.x[1] - self.x[0]) if len(self.x) > 1 else 1.0
        return float(np.sum(self.h) * dx)


@dataclass
class SimulationConfig2D:
    """Configuration parameters for 2D surface dune simulation."""
    nx: int = 128
    ny: int = 128
    length_x_m: float = 200.0
    length_y_m: float = 200.0
    env: PlanetaryEnvironment = EARTH_DESERT
    u_star_0: float = 0.40
    A: float = 3.2
    B: float = 1.6
    A_y: float = 1.6
    periodic_x: bool = True
    periodic_y: bool = True
    inflow_ratio: float = 0.0
    enforce_bedrock: bool = False
    d_lat: float = 0.35
    cfl: float = 0.25
    dt_max: float = 1800.0
    smooth_length: Optional[float] = None
    bedrock_floor: Optional[object] = None

    @property
    def dx(self) -> float:
        """Spatial resolution in x [m]."""
        return self.length_x_m / float(self.nx)

    @property
    def dy(self) -> float:
        """Spatial resolution in y [m]."""
        return self.length_y_m / float(self.ny)

    @property
    def tau_0(self) -> float:
        """Undisturbed shear stress [Pa]."""
        return self.env.fluid_density * (self.u_star_0 ** 2)


@dataclass
class SimulationState2D:
    """Snapshot state of a 2D dune simulation."""
    time_s: float
    step_count: int
    x: np.ndarray
    y: np.ndarray
    h: np.ndarray
    h_tilde: np.ndarray
    in_wake: np.ndarray
    tau_x: np.ndarray
    tau_y: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    dt: float

    @property
    def time_days(self) -> float:
        """Elapsed time in days."""
        return self.time_s / 86400.0

    @property
    def time_years(self) -> float:
        """Elapsed time in years."""
        return self.time_s / (365.25 * 86400.0)

    @property
    def max_height(self) -> float:
        """Peak elevation [m]."""
        return float(np.max(self.h))

    @property
    def total_volume(self) -> float:
        """Total sand volume [m^3]."""
        dx = float(self.x[1] - self.x[0]) if len(self.x) > 1 else 1.0
        dy = float(self.y[1] - self.y[0]) if len(self.y) > 1 else 1.0
        return float(np.sum(self.h) * dx * dy)
