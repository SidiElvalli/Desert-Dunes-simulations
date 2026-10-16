"""Visualization tools for 1D profiles, 2D surfaces, and animations."""

from .plots_1d import plot_simulation_state_1d, plot_profile_time_series_1d
from .plots_2d import plot_topography_2d, plot_surface_3d
from .animation import animate_simulation_1d

__all__ = [
    "plot_simulation_state_1d",
    "plot_profile_time_series_1d",
    "plot_topography_2d",
    "plot_surface_3d",
    "animate_simulation_1d",
]
