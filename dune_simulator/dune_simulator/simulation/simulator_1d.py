"""1D Profile Dune Morphodynamics Simulator Engine."""

from typing import Optional, Generator
import numpy as np
from .config import SimulationConfig1D, SimulationState1D
from ..core.separation import compute_separation_1d
from ..core.wind import compute_shear_stress_1d
from ..core.transport import compute_saturated_flux, solve_flux_1d
from ..core.avalanche import avalanche_1d
from ..core.exner import compute_flux_divergence_1d, compute_adaptive_dt, exner_step_1d


class DuneSimulator1D:
    """Simulates 1D profile evolution of aeolian or subaqueous sand dunes."""

    def __init__(
        self,
        config: Optional[SimulationConfig1D] = None,
        h_init: Optional[np.ndarray] = None,
    ):
        self.config = config or SimulationConfig1D()
        self.x = np.arange(self.config.nx) * self.config.dx

        if h_init is not None:
            if len(h_init) != self.config.nx:
                raise ValueError(
                    f"h_init length ({len(h_init)}) does not match config.nx ({self.config.nx})"
                )
            self.h = np.asarray(h_init, dtype=np.float64).copy()
        else:
            self.h = np.zeros(self.config.nx, dtype=np.float64)

        self.time_s = 0.0
        self.step_count = 0
        self.dt_last = 100.0

        # Initialize diagnostic fields
        self._h_tilde, self._in_wake = compute_separation_1d(
            self.h, self.config.dx, self.config.periodic, self.config.env.separation_angle_deg
        )
        self._tau, _ = compute_shear_stress_1d(
            self._h_tilde, self._in_wake, self.config.dx, self.config.tau_0,
            self.config.A, self.config.B, self.config.periodic, self.config.smooth_length
        )
        self._q_sat = compute_saturated_flux(self._tau, self.config.env)
        self._q = self._q_sat.copy()

    def set_topography(self, h: np.ndarray) -> None:
        """Sets new bed elevation profile."""
        if len(h) != self.config.nx:
            raise ValueError(f"h length ({len(h)}) does not match config.nx ({self.config.nx})")
        self.h = np.asarray(h, dtype=np.float64).copy()

    def get_state(self) -> SimulationState1D:
        """Returns the current simulation snapshot state."""
        return SimulationState1D(
            time_s=self.time_s,
            step_count=self.step_count,
            x=self.x.copy(),
            h=self.h.copy(),
            h_tilde=self._h_tilde.copy(),
            in_wake=self._in_wake.copy(),
            tau=self._tau.copy(),
            q_sat=self._q_sat.copy(),
            q=self._q.copy(),
            dt=self.dt_last,
        )

    def step(self) -> SimulationState1D:
        """Advances the simulation by one adaptive Exner time step.

        Morphodynamic pipeline:
        1. Dynamic Flow Separation Bubble Envelope
        2. Turbulent Shear Stress Perturbation (Jackson-Hunt FFT)
        3. Saturated Flux & Non-local Relaxation
        4. Exner Mass Conservation & Adaptive CFL Time Step
        5. Avalanche Cellular Slope Relaxation
        """
        cfg = self.config
        dx = cfg.dx

        # 1. Flow separation
        h_tilde, in_wake = compute_separation_1d(
            h=self.h,
            dx=dx,
            periodic=cfg.periodic,
            separation_angle_deg=cfg.env.separation_angle_deg,
        )

        # 2. Shear stress perturbation
        tau, _ = compute_shear_stress_1d(
            h_tilde=h_tilde,
            in_wake=in_wake,
            dx=dx,
            tau_0=cfg.tau_0,
            A=cfg.A,
            B=cfg.B,
            periodic=cfg.periodic,
            smooth_length=cfg.smooth_length,
        )

        # 3. Transport flux
        q_sat = compute_saturated_flux(tau=tau, env=cfg.env)
        q = solve_flux_1d(
            q_sat=q_sat,
            dx=dx,
            env=cfg.env,
            periodic=cfg.periodic,
            inflow_ratio=cfg.inflow_ratio,
            h=self.h,
            enforce_bedrock=cfg.enforce_bedrock,
            dt=self.dt_last,
            bedrock_floor=cfg.bedrock_floor,
        )

        # 4. Exner mass conservation
        _, dh_dt = compute_flux_divergence_1d(
            q=q,
            dx=dx,
            rho_bulk=cfg.env.bulk_density,
            periodic=cfg.periodic,
        )

        # 5. Adaptive stable time step
        dt = compute_adaptive_dt(
            dh_dt=dh_dt,
            dx=dx,
            cfl=cfg.cfl,
            dt_max=cfg.dt_max,
            max_slope_deg=cfg.env.dynamic_repose_deg,
        )
        dt_prev = self.dt_last
        self.dt_last = dt

        # The erosion limiter depends on dt. The first pass used the previous step's dt,
        # so re-solve the flux with the dt that is actually applied. Otherwise a larger dt
        # could erode more sand than is available (which previously created mass at the floor).
        if cfg.enforce_bedrock and abs(dt - dt_prev) > 1e-9 * max(dt, 1.0):
            q = solve_flux_1d(
                q_sat=q_sat,
                dx=dx,
                env=cfg.env,
                periodic=cfg.periodic,
                inflow_ratio=cfg.inflow_ratio,
                h=self.h,
                enforce_bedrock=cfg.enforce_bedrock,
                dt=dt,
                bedrock_floor=cfg.bedrock_floor,
            )
            _, dh_dt = compute_flux_divergence_1d(
                q=q,
                dx=dx,
                rho_bulk=cfg.env.bulk_density,
                periodic=cfg.periodic,
            )

        # Exner update
        h_next = exner_step_1d(
            h=self.h,
            dh_dt=dh_dt,
            dt=dt,
            enforce_bedrock=cfg.enforce_bedrock,
            bedrock_floor=cfg.bedrock_floor,
        )

        # 6. Avalanche slope limiter
        h_relaxed = avalanche_1d(
            h=h_next,
            dx=dx,
            theta_stat_deg=cfg.env.repose_angle_deg,
            theta_dyn_deg=cfg.env.dynamic_repose_deg,
            periodic=cfg.periodic,
            enforce_bedrock=cfg.enforce_bedrock,
            bedrock_floor=cfg.bedrock_floor,
        )

        # Update fields
        self.h = h_relaxed
        self.time_s += dt
        self.step_count += 1

        self._h_tilde = h_tilde
        self._in_wake = in_wake
        self._tau = tau
        self._q_sat = q_sat
        self._q = q

        return self.get_state()

    def run(
        self,
        total_time_s: Optional[float] = None,
        total_time_years: Optional[float] = None,
        record_interval_s: Optional[float] = None,
        record_interval_days: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> Generator[SimulationState1D, None, None]:
        """Runs the simulation forward in time and yields state snapshots at recorded intervals.

        Parameters:
            total_time_s: Total duration in seconds.
            total_time_years: Total duration in years.
            record_interval_s: Recording interval in seconds.
            record_interval_days: Recording interval in days.
            max_steps: Optional hard safety ceiling on iteration count.

        Yields:
            SimulationState1D instances at each recording checkpoint.
        """
        if total_time_years is not None:
            t_target = total_time_years * (365.25 * 86400.0)
        elif total_time_s is not None:
            t_target = total_time_s
        else:
            t_target = 365.25 * 86400.0  # Default 1 year

        if record_interval_days is not None:
            interval = record_interval_days * 86400.0
        elif record_interval_s is not None:
            interval = record_interval_s
        else:
            interval = t_target / 50.0  # 50 frames by default

        t_end = self.time_s + t_target
        next_record = self.time_s + interval

        # Initial state
        yield self.get_state()

        steps = 0
        while self.time_s < t_end:
            if max_steps is not None and steps >= max_steps:
                break

            self.step()
            steps += 1

            if self.time_s >= next_record or self.time_s >= t_end:
                yield self.get_state()
                next_record = self.time_s + interval
