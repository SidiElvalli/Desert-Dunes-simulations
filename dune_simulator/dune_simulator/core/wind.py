"""Basal shear stress perturbation solver using Jackson-Hunt boundary layer theory in Fourier space.

References:
    - Jackson & Hunt (1975) Q. J. R. Meteorol. Soc. 101, 929-955.
    - Hunt, Leibovich & Richards (1988) Q. J. R. Meteorol. Soc. 114, 1431-1470.
    - Weng et al. (1991) Acta Mech. (Suppl.) 2, 1-22.
    - Kroy, Sauermann & Herrmann (2002) Phys. Rev. E 66, 031302.
    - Fourrière, Claudin & Andreotti (2010) J. Fluid Mech. 649, 287-328.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import fft, special


def compute_shear_stress_1d(
    h_tilde: np.ndarray,
    in_wake: np.ndarray,
    dx: float,
    tau_0: float,
    A: float = 3.2,
    B: float = 1.6,
    periodic: bool = True,
    smooth_length: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates 1D basal shear stress over aerodynamic envelope using FFT.

    Parameters:
        h_tilde: Aerodynamic envelope elevation array of shape (nx,).
        in_wake: Boolean array of shape (nx,) indicating points inside the separation bubble.
        dx: Spatial grid spacing in meters.
        tau_0: Baseline undisturbed shear stress [Pa].
        A: Dimensionless in-phase acceleration coefficient (flow convergence over crest).
        B: Dimensionless out-of-phase quadrature coefficient (upwind shift).
        periodic: Whether to use periodic boundary conditions.
        smooth_length: Low-pass Gaussian filter scale [m] to suppress Nyquist noise (default 2*dx).

    Returns:
        tau: Total surface shear stress [Pa] of shape (nx,). Zero inside the wake.
        tau_pert: Relative shear stress perturbation (tau - tau_0) / tau_0.
    """
    h_tilde = np.asarray(h_tilde, dtype=np.float64)
    nx = len(h_tilde)

    if smooth_length is None:
        smooth_length = 2.0 * dx

    if not periodic:
        pad = nx // 2
        h_padded = np.pad(h_tilde, pad, mode="edge")
    else:
        pad = 0
        h_padded = h_tilde

    n_fft = len(h_padded)
    k = 2.0 * np.pi * fft.rfftfreq(n_fft, d=dx)
    h_hat = fft.rfft(h_padded)

    # Regularization filter
    if smooth_length > 0.0:
        filt = np.exp(-0.5 * (k * smooth_length) ** 2)
    else:
        filt = 1.0

    # Perturbation kernel in Fourier space:
    # A * |k| represents in-phase speed-up (Hilbert transform of dh/dx)
    # +1j * B * k represents spatial derivative dh/dx (shifting peak upwind)
    tau_pert_hat = (A * np.abs(k) + 1j * B * k) * filt * h_hat
    tau_pert_hat[0] = 0.0  # Zero mean perturbation

    tau_pert = fft.irfft(tau_pert_hat, n=n_fft)

    if not periodic:
        tau_pert = tau_pert[pad : pad + nx]

    # Total surface shear stress: tau = tau_0 * (1 + tau_pert)
    tau = tau_0 * np.maximum(0.0, 1.0 + tau_pert)
    tau[in_wake] = 0.0

    return tau, tau_pert


def analytical_gaussian_shear_perturbation(
    x: np.ndarray,
    x0: float,
    H: float,
    R: float,
    A: float = 3.2,
    B: float = 1.6,
) -> np.ndarray:
    """Exact analytical shear stress perturbation for a 1D Gaussian heap h(x) = H * exp(-((x-x0)/R)^2).

    Uses Dawson's integral for the exact continuous Hilbert transform of dh/dx.

    Parameters:
        x: Spatial coordinate array [m].
        x0: Heap center position [m].
        H: Heap peak height [m].
        R: Gaussian characteristic half-width [m].
        A: In-phase coefficient.
        B: Upwind quadrature coefficient.

    Returns:
        tau_pert: Exact analytical shear stress perturbation array.
    """
    z = (x - x0) / R
    # Continuous Hilbert transform of dh/dx for Gaussian:
    # H{d/dx exp(-z^2)} = (2 / (sqrt(pi) * R)) * (1 - 2 * z * D(z)) where D(z) is Dawson's integral
    hilbert_dh_dx = (2.0 * H / (np.sqrt(np.pi) * R)) * (1.0 - 2.0 * z * special.dawsn(z))
    dh_dx = -(2.0 * H * z / R) * np.exp(-(z ** 2))

    return A * hilbert_dh_dx + B * dh_dx


def compute_shear_stress_2d(
    h_tilde: np.ndarray,
    in_wake: np.ndarray,
    dx: float,
    dy: float,
    tau_0: float,
    A: float = 3.2,
    B: float = 1.6,
    A_y: float = 1.6,
    periodic_x: bool = True,
    smooth_length: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates 2D basal shear stress components over 2D aerodynamic envelope using 2D FFT.

    Wind blows along axis 0 (+x). Axis 1 (+y) is the transverse direction.

    Parameters:
        h_tilde: 2D aerodynamic envelope of shape (nx, ny).
        in_wake: 2D boolean wake mask of shape (nx, ny).
        dx: Grid spacing along x [m].
        dy: Grid spacing along y [m].
        tau_0: Undisturbed baseline shear stress [Pa].
        A: Streamwise in-phase acceleration coefficient.
        B: Streamwise upwind quadrature shift coefficient.
        A_y: Lateral shear perturbation coefficient (default 2.0).
        periodic_x: Whether boundary along x is periodic (if False, pads by nx//2).
        smooth_length: Low-pass Gaussian filter scale [m] (default 2*dx).

    Returns:
        tau_x: Streamwise surface shear stress component [Pa].
        tau_y: Transverse surface shear stress component [Pa].
        tau_pert_x: Streamwise relative perturbation.
        tau_pert_y: Transverse relative perturbation.
    """
    h_tilde = np.asarray(h_tilde, dtype=np.float64)
    nx, ny = h_tilde.shape

    if smooth_length is None:
        smooth_length = 2.0 * dx

    if not periodic_x:
        pad = nx // 2
        h_padded = np.pad(h_tilde, ((pad, pad), (0, 0)), mode="edge")
    else:
        pad = 0
        h_padded = h_tilde

    nx_fft, ny_fft = h_padded.shape
    kx = 2.0 * np.pi * fft.fftfreq(nx_fft, d=dx)[:, np.newaxis]
    ky = 2.0 * np.pi * fft.rfftfreq(ny_fft, d=dy)[np.newaxis, :]

    k_mag = np.sqrt(kx ** 2 + ky ** 2)
    k_mag_safe = np.where(k_mag == 0.0, 1.0, k_mag)

    h_hat = fft.rfft2(h_padded)

    # Regularization filter
    if smooth_length > 0.0:
        filt = np.exp(-0.5 * (k_mag * smooth_length) ** 2)
    else:
        filt = 1.0

    # 2D Jackson-Hunt / Weng / Durán kernels:
    # tau_x_hat = (kx^2 / |k|) * A + 1j * B * kx
    # tau_y_hat = (kx * ky / |k|) * A_y
    kernel_x = ((kx ** 2 / k_mag_safe) * A + 1j * B * kx) * filt
    kernel_y = ((kx * ky / k_mag_safe) * A_y) * filt

    kernel_x[0, 0] = 0.0
    kernel_y[0, 0] = 0.0

    tau_pert_x = fft.irfft2(kernel_x * h_hat, s=(nx_fft, ny_fft))
    tau_pert_y = fft.irfft2(kernel_y * h_hat, s=(nx_fft, ny_fft))

    if not periodic_x:
        tau_pert_x = tau_pert_x[pad : pad + nx, :]
        tau_pert_y = tau_pert_y[pad : pad + nx, :]

    tau_x = tau_0 * np.maximum(0.0, 1.0 + tau_pert_x)
    tau_y = tau_0 * tau_pert_y

    tau_x[in_wake] = 0.0
    tau_y[in_wake] = 0.0

    return tau_x, tau_y, tau_pert_x, tau_pert_y
