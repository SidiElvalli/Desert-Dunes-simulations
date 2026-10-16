"""Educational initial surface presets for sand dune simulations."""

from typing import Optional, Tuple
import numpy as np


def flat_bed_with_noise(
    length: float,
    nx: int,
    mean_height: float = 3.0,
    noise_amplitude: float = 0.01,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Creates a flat sand bed with small random white noise to study spontaneous instability.

    Parameters:
        length: Domain length in meters.
        nx: Number of grid points.
        mean_height: Mean sand bed thickness [m].
        noise_amplitude: Random perturbation magnitude [m] (e.g. 0.01 = 1 cm).
        seed: Random seed for reproducibility.

    Returns:
        h: Initial elevation profile array of shape (nx,).
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-noise_amplitude, noise_amplitude, size=nx)
    return np.full(nx, mean_height, dtype=np.float64) + noise


def gaussian_heap_1d(
    length: float,
    nx: int,
    height: float = 3.0,
    sigma: float = 15.0,
    center: Optional[float] = None,
    base_height: float = 0.0,
) -> np.ndarray:
    """Creates a single symmetric 1D Gaussian sand heap.

    Parameters:
        length: Domain length in meters.
        nx: Number of grid points.
        height: Peak mound height above base [m].
        sigma: Characteristic half-width [m].
        center: Peak position [m] (defaults to length / 2).
        base_height: Surrounding flat sand bed height [m].

    Returns:
        h: Elevation profile array of shape (nx,).
    """
    dx = length / nx
    x = np.arange(nx) * dx
    x0 = center if center is not None else length / 2.0
    return base_height + height * np.exp(-(((x - x0) / sigma) ** 2))


def forward_step_1d(
    length: float,
    nx: int,
    step_height: float = 2.0,
    step_pos: Optional[float] = None,
    base_height: float = 0.0,
) -> np.ndarray:
    """Creates a forward-facing topographic step (low bed upwind, high plateau downwind)."""
    dx = length / nx
    x = np.arange(nx) * dx
    x_step = step_pos if step_pos is not None else length * 0.4
    return np.where(x < x_step, base_height, base_height + step_height)


def backward_step_1d(
    length: float,
    nx: int,
    step_height: float = 2.0,
    step_pos: Optional[float] = None,
    base_height: float = 1.0,
) -> np.ndarray:
    """Creates a backward-facing topographic step (high plateau upwind, low bed downwind)."""
    dx = length / nx
    x = np.arange(nx) * dx
    x_step = step_pos if step_pos is not None else length * 0.3
    return np.where(x <= x_step, base_height + step_height, base_height)


def two_dunes_1d(
    length: float,
    nx: int,
    h1: float = 2.5,
    h2: float = 5.0,
    w1: float = 15.0,
    w2: float = 25.0,
    x1: Optional[float] = None,
    x2: Optional[float] = None,
    base_height: float = 0.0,
) -> np.ndarray:
    """Creates two isolated dunes of unequal height to demonstrate c ~ 1/H and dune collision."""
    dx = length / nx
    x = np.arange(nx) * dx
    pos1 = x1 if x1 is not None else length * 0.25
    pos2 = x2 if x2 is not None else length * 0.65

    heap1 = h1 * np.exp(-(((x - pos1) / w1) ** 2))
    heap2 = h2 * np.exp(-(((x - pos2) / w2) ** 2))
    return base_height + np.maximum(heap1, heap2)


def barchan_seed_2d(
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    height: float = 4.0,
    radius: float = 20.0,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    base_height: float = 0.0,
) -> np.ndarray:
    """Creates an isolated 2D parabolic sand mound on bare bedrock to breed a crescent barchan."""
    dx = length_x / nx
    dy = length_y / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    x0 = center_x if center_x is not None else length_x * 0.35
    y0 = center_y if center_y is not None else length_y * 0.50

    r_sq = ((X - x0) ** 2 + (Y - y0) ** 2) / (radius ** 2)
    # Parabolic dome profile
    dome = height * np.maximum(0.0, 1.0 - r_sq)
    return base_height + dome


def ridge_2d(
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    height: float = 3.0,
    width: float = 20.0,
    center_x: Optional[float] = None,
    base_height: float = 0.0,
) -> np.ndarray:
    """Creates a 2D transverse sand ridge (uniform across y)."""
    dx = length_x / nx
    x = np.arange(nx) * dx
    x0 = center_x if center_x is not None else length_x * 0.4
    ridge_1d = height * np.exp(-(((x - x0) / width) ** 2))
    return base_height + np.repeat(ridge_1d[:, np.newaxis], ny, axis=1)


def twin_barchans_2d(
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    height_main: float = 3.4,
    radius_main: float = 18.0,
    base_height: float = 0.0,
    noise_amplitude: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Creates two offset parabolic sand mounds (smaller fast upwind mound + larger downwind mound)."""
    dx = length_x / nx
    dy = length_y / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Main downwind mound
    x1, y1 = length_x * 0.36, length_y * 0.46
    r1_sq = ((X - x1) / radius_main) ** 2 + ((Y - y1) / radius_main) ** 2
    m1 = np.where(r1_sq < 1.0, height_main * (1.0 - r1_sq), 0.0)

    # Smaller upwind offset mound (migrates faster due to c ~ 1/H)
    h2 = 0.65 * height_main
    r2 = 0.72 * radius_main
    x2, y2 = length_x * 0.16, length_y * 0.57
    r2_sq = ((X - x2) / r2) ** 2 + ((Y - y2) / r2) ** 2
    m2 = np.where(r2_sq < 1.0, h2 * (1.0 - r2_sq), 0.0)

    h = base_height + np.maximum(m1, m2)
    if noise_amplitude > 0.0:
        rng = np.random.default_rng(seed)
        mask = h > base_height + 0.05
        h[mask] = np.maximum(base_height, h[mask] + rng.uniform(-noise_amplitude, noise_amplitude, size=np.sum(mask)))
    return h


def flat_bed_noise_2d(
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    mean_height: float = 2.5,
    noise_amplitude: float = 0.08,
    seed: int = 42,
) -> np.ndarray:
    """Creates a 2D erodible sand sheet with random perturbations for spontaneous bedform instability."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-noise_amplitude, noise_amplitude, size=(nx, ny))
    noise -= np.mean(noise)
    return np.maximum(0.0, mean_height + noise)


def backward_step_2d(
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    step_height: float = 2.5,
    step_pos: Optional[float] = None,
    base_height: float = 0.5,
) -> np.ndarray:
    """Creates a 2D escarpment / backward step with a central notch for wake sand trapping."""
    dx = length_x / nx
    dy = length_y / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    x_step = step_pos if step_pos is not None else length_x * 0.25
    # Slight crescent curvature on the escarpment edge
    edge_x = x_step + 4.0 * np.cos(2.0 * np.pi * (Y - 0.5 * length_y) / length_y)
    return np.where(X <= edge_x, base_height + step_height, base_height)


def multi_mound_field_2d(
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    height: float = 3.0,
    radius: float = 14.0,
    base_height: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Creates a staggered field of 4 sand mounds on bedrock that evolve into a Barchan swarm."""
    dx = length_x / nx
    dy = length_y / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    centers = [
        (length_x * 0.20, length_y * 0.30, 0.85 * height, 0.88 * radius),
        (length_x * 0.24, length_y * 0.72, 1.00 * height, 1.00 * radius),
        (length_x * 0.44, length_y * 0.50, 1.05 * height, 1.05 * radius),
        (length_x * 0.40, length_y * 0.20, 0.75 * height, 0.80 * radius),
    ]
    h = np.full((nx, ny), base_height, dtype=np.float64)
    for cx, cy, hm, rm in centers:
        r_sq = ((X - cx) / rm) ** 2 + ((Y - cy) / rm) ** 2
        mound = np.where(r_sq < 1.0, hm * (1.0 - r_sq), 0.0)
        h = np.maximum(h, base_height + mound)
    return h


def sahara_dune_sea_2d(
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    mean_height: float = 2.8,
    amplitude: float = 2.2,
    seed: int = 42,
) -> np.ndarray:
    """Creates a rich, sinuous multi-crest Barchanoid & Transverse desert dune sea with sharp crests."""
    dx = length_x / nx
    dy = length_y / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 3 primary sinuous transverse waves with phase-locked Barchanoid lobes
    lam_x = length_x / 3.2
    kx = 2.0 * np.pi / lam_x
    ky = 2.0 * np.pi / (length_y / 2.0)

    # Sinuous crestline modulation in y
    phase_mod = 1.15 * np.sin(ky * Y) + 0.55 * np.cos(2.0 * ky * Y + 0.7)
    phi = kx * X + phase_mod

    # Asymmetric dune profile (gentle convex stoss, steep 32-deg lee face)
    wave1 = np.sin(phi) - 0.42 * np.sin(2.0 * phi) + 0.16 * np.sin(3.0 * phi)

    # Secondary oblique intersecting dune train for realistic desert crest junctions (Y-defects)
    phi2 = kx * 1.35 * (X * 0.92 + 0.32 * Y) - 0.8 * np.cos(ky * Y)
    wave2 = 0.45 * (np.sin(phi2) - 0.35 * np.sin(2.0 * phi2))

    # Superimpose 3 prominent barchanoid domes
    domes = np.zeros_like(X)
    for cx, cy, hd, rd in [
        (0.28 * length_x, 0.35 * length_y, 1.3 * amplitude, 0.20 * length_x),
        (0.58 * length_x, 0.68 * length_y, 1.4 * amplitude, 0.22 * length_x),
        (0.75 * length_x, 0.28 * length_y, 1.2 * amplitude, 0.19 * length_x),
    ]:
        r2 = ((X - cx) / rd) ** 2 + ((Y - cy) / (1.15 * rd)) ** 2
        r_lee = ((X - (cx + 0.48 * rd)) / (0.72 * rd)) ** 2 + ((Y - cy) / (0.68 * rd)) ** 2
        d_val = np.where(r2 < 1.0, hd * (1.0 - r2) - 0.55 * hd * np.maximum(0.0, 1.0 - r_lee), 0.0)
        domes = np.maximum(domes, d_val)

    h = mean_height + 0.55 * amplitude * (wave1 + wave2) + 0.65 * domes
    return np.maximum(0.25, h)


def desert_pyramid_obstacle_2d(
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    pyramid_height: float = 5.2,
    sand_height: float = 2.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a non-erodible square rocky pyramid obstacle embedded in a wind-erodible sand dune field.

    Returns:
        (h_init, bedrock_floor) where bedrock_floor defines the non-erodible rocky pyramid.
    """
    dx = length_x / nx
    dy = length_y / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    cx, cy = 0.44 * length_x, 0.50 * length_y
    half_base = 0.16 * min(length_x, length_y)
    # Square pyramid: L_infinity norm max(|x - cx|, |y - cy|)
    dist_inf = np.maximum(np.abs(X - cx), np.abs(Y - cy))
    bedrock_floor = np.where(dist_inf < half_base, pyramid_height * (1.0 - dist_inf / half_base), 0.0)

    # Surrounding sinuous sand dunes approaching and wrapping around the pyramid
    sand_bed = sahara_dune_sea_2d(length_x, length_y, nx, ny, mean_height=1.6, amplitude=sand_height * 0.75)
    h_init = np.maximum(bedrock_floor, sand_bed)
    return h_init, bedrock_floor
