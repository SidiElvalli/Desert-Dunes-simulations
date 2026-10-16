"""Topographic initial condition presets."""

from .initial_surfaces import (
    flat_bed_with_noise,
    gaussian_heap_1d,
    forward_step_1d,
    backward_step_1d,
    two_dunes_1d,
    barchan_seed_2d,
    ridge_2d,
)

__all__ = [
    "flat_bed_with_noise",
    "gaussian_heap_1d",
    "forward_step_1d",
    "backward_step_1d",
    "two_dunes_1d",
    "barchan_seed_2d",
    "ridge_2d",
]
