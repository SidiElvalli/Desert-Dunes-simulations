"""Tests for planetary scaling laws and environmental presets."""

import pytest
from dune_simulator.core.constants import (
    EARTH_DESERT,
    MARS_DUNES,
    VENUS_DUNES,
    AQUEOUS_FLUME,
)


def test_mars_vs_earth_saturation_length():
    """Mars saturation length should be ~46x larger than Earth's due to thin CO2 atmosphere."""
    ratio = MARS_DUNES.saturation_length / EARTH_DESERT.saturation_length
    assert 40.0 <= ratio <= 50.0


def test_earth_vs_venus_saturation_length():
    """Earth saturation length should be ~67x larger than Venus due to dense CO2 atmosphere."""
    ratio = EARTH_DESERT.saturation_length / VENUS_DUNES.saturation_length
    assert 60.0 <= ratio <= 75.0


def test_subaqueous_flume_saturation_length():
    """Water flume saturation length should be on the millimeter scale (< 2 mm)."""
    assert AQUEOUS_FLUME.saturation_length < 0.002
    assert AQUEOUS_FLUME.saturation_length > 0.0005


def test_threshold_friction_velocity_ordering():
    """Mars threshold u_*th should be highest, while Venus and flume should be lowest."""
    assert MARS_DUNES.threshold_friction_velocity > EARTH_DESERT.threshold_friction_velocity
    assert EARTH_DESERT.threshold_friction_velocity > VENUS_DUNES.threshold_friction_velocity
    assert VENUS_DUNES.threshold_friction_velocity > AQUEOUS_FLUME.threshold_friction_velocity
