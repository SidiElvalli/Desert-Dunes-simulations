"""2D Topographic Surface Dune Morphodynamics Simulator Engine."""

from typing import Optional, Generator
import numpy as np
from .config import SimulationConfig2D, SimulationState2D
from ..core.separation import compute_separation_2d
from ..core.wind import compute_shear_stress_2d
from ..core.transport import solve_flux_2d
from ..core.avalanche import avalanche_2d
from ..core.exner import compute_flux_divergence_2d, compute_adaptive_dt, exner_step_2d


class DuneSimulator2D:
    """Simulates 2D topographic evolution of dunes (barchans, horns, transverse ridges)."""

    def __init__(
        self,
        config: Optional[SimulationConfig2D] = None,
        h_init: Optional[np.ndarray] = None,
    ):
        self.config = config or SimulationConfig2D()
        self.x = np.arange(self.config.nx) * self.config.dx
        self.y = np.arange(self.config.ny) * self.config.dy

        if h_init is not None:
            if h_init.shape != (self.config.nx, self.config.ny):
                raise ValueError(
                    f"h_init shape {h_init.shape} does not match (nx={self.config.nx}, ny={self.config.ny})"
                )
            self.h = np.asarray(h_init, dtype=np.float64).copy()
        else:
            self.h = np.zeros((self.config.nx, self.config.ny), dtype=np.float64)

        self.time_s = 0.0
        self.step_count = 0
        self.dt_last = 100.0

        # Initialize diagnostic fields
        self._h_tilde, self._in_wake = compute_separation_2d(
            h=self.h,
            dx=self.config.dx,
            periodic=self.config.periodic_x,
            separation_angle_deg=self.config.env.separation_angle_deg,
            smooth_transverse=True,
        )
        self._tau_x, self._tau_y, _, _ = compute_shear_stress_2d(
            h_tilde=self._h_tilde,
            in_wake=self._in_wake,
            dx=self.config.dx,
            dy=self.config.dy,
            tau_0=self.config.tau_0,
            A=self.config.A,
            B=self.config.B,
            A_y=self.config.A_y,
            periodic_x=self.config.periodic_x,
            smooth_length=self.config.smooth_length,
        )
        self._qx, self._qy = solve_flux_2d(
            tau_x=self._tau_x,
            tau_y=self._tau_y,
            h=self.h,
            dx=self.config.dx,
            dy=self.config.dy,
            env=self.config.env,
            tau_0=self.config.tau_0,
            periodic_x=self.config.periodic_x,
            periodic_y=self.config.periodic_y,
            inflow_ratio=self.config.inflow_ratio,
            d_lat=self.config.d_lat,
            enforce_bedrock=self.config.enforce_bedrock,
            dt=self.dt_last,
            bedrock_floor=self.config.bedrock_floor,
        )

    def set_topography(self, h: np.ndarray) -> None:
        """Sets new 2D bed elevation surface."""
        if h.shape != (self.config.nx, self.config.ny):
            raise ValueError(f"h shape {h.shape} does not match (nx={self.config.nx}, ny={self.config.ny})")
        self.h = np.asarray(h, dtype=np.float64).copy()

    def get_state(self) -> SimulationState2D:
        """Returns the current 2D simulation snapshot state."""
        return SimulationState2D(
            time_s=self.time_s,
            step_count=self.step_count,
            x=self.x.copy(),
            y=self.y.copy(),
            h=self.h.copy(),
            h_tilde=self._h_tilde.copy(),
            in_wake=self._in_wake.copy(),
            tau_x=self._tau_x.copy(),
            tau_y=self._tau_y.copy(),
            qx=self._qx.copy(),
            qy=self._qy.copy(),
            dt=self.dt_last,
        )

    def step(self) -> SimulationState2D:
        """Advances the 2D simulation by one adaptive Exner time step."""
        cfg = self.config
        dx = cfg.dx
        dy = cfg.dy

        # 1. 2D Flow separation
        h_tilde, in_wake = compute_separation_2d(
            h=self.h,
            dx=dx,
            periodic=cfg.periodic_x,
            separation_angle_deg=cfg.env.separation_angle_deg,
            smooth_transverse=True,
        )

        # 2. 2D Shear stress perturbation
        tau_x, tau_y, _, _ = compute_shear_stress_2d(
            h_tilde=h_tilde,
            in_wake=in_wake,
            dx=dx,
            dy=dy,
            tau_0=cfg.tau_0,
            A=cfg.A,
            B=cfg.B,
            A_y=cfg.A_y,
            periodic_x=cfg.periodic_x,
            smooth_length=cfg.smooth_length,
        )

        # 3. 2D Transport fluxes
        qx, qy = solve_flux_2d(
            tau_x=tau_x,
            tau_y=tau_y,
            h=self.h,
            dx=dx,
            dy=dy,
            env=cfg.env,
            tau_0=cfg.tau_0,
            periodic_x=cfg.periodic_x,
            periodic_y=cfg.periodic_y,
            inflow_ratio=cfg.inflow_ratio,
            d_lat=cfg.d_lat,
            enforce_bedrock=cfg.enforce_bedrock,
            dt=self.dt_last,
            bedrock_floor=cfg.bedrock_floor,
        )

        # 4. 2D Exner divergence
        _, _, dh_dt = compute_flux_divergence_2d(
            qx=qx,
            qy=qy,
            dx=dx,
            dy=dy,
            rho_bulk=cfg.env.bulk_density,
            periodic_x=cfg.periodic_x,
            periodic_y=cfg.periodic_y,
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

        # The erosion limiter depends on dt: re-solve the flux with the dt actually applied
        # (the first pass used the previous step's dt), otherwise erosion can exceed the
        # available sand and the floor would have to create mass.
        if cfg.enforce_bedrock and abs(dt - dt_prev) > 1e-9 * max(dt, 1.0):
            qx, qy = solve_flux_2d(
                tau_x=tau_x,
                tau_y=tau_y,
                h=self.h,
                dx=dx,
                dy=dy,
                env=cfg.env,
                tau_0=cfg.tau_0,
                periodic_x=cfg.periodic_x,
                periodic_y=cfg.periodic_y,
                inflow_ratio=cfg.inflow_ratio,
                d_lat=cfg.d_lat,
                enforce_bedrock=cfg.enforce_bedrock,
                dt=dt,
                bedrock_floor=cfg.bedrock_floor,
            )
            _, _, dh_dt = compute_flux_divergence_2d(
                qx=qx,
                qy=qy,
                dx=dx,
                dy=dy,
                rho_bulk=cfg.env.bulk_density,
                periodic_x=cfg.periodic_x,
                periodic_y=cfg.periodic_y,
            )

        # Exner update
        h_next = exner_step_2d(
            h=self.h,
            dh_dt=dh_dt,
            dt=dt,
            enforce_bedrock=cfg.enforce_bedrock,
            bedrock_floor=cfg.bedrock_floor,
        )

        # 6. 2D Avalanche slope limiter
        h_relaxed = avalanche_2d(
            h=h_next,
            dx=dx,
            dy=dy,
            theta_stat_deg=cfg.env.repose_angle_deg,
            theta_dyn_deg=cfg.env.dynamic_repose_deg,
            periodic_x=cfg.periodic_x,
            periodic_y=cfg.periodic_y,
            enforce_bedrock=cfg.enforce_bedrock,
            bedrock_floor=cfg.bedrock_floor,
        )

        self.h = h_relaxed
        self.time_s += dt
        self.step_count += 1

        self._h_tilde = h_tilde
        self._in_wake = in_wake
        self._tau_x = tau_x
        self._tau_y = tau_y
        self._qx = qx
        self._qy = qy

        return self.get_state()

    def run(
        self,
        total_time_s: Optional[float] = None,
        total_time_years: Optional[float] = None,
        record_interval_s: Optional[float] = None,
        record_interval_days: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> Generator[SimulationState2D, None, None]:
        """Runs the 2D simulation forward and yields snapshot states."""
        if total_time_years is not None:
            t_target = total_time_years * (365.25 * 86400.0)
        elif total_time_s is not None:
            t_target = total_time_s
        else:
            t_target = 365.25 * 86400.0

        if record_interval_days is not None:
            interval = record_interval_days * 86400.0
        elif record_interval_s is not None:
            interval = record_interval_s
        else:
            interval = t_target / 20.0

        t_end = self.time_s + t_target
        next_record = self.time_s + interval

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
