from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .allocator import TorqueAllocator
from .attitude import (
    normalize_quaternion,
    quaternion_error_vector,
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_to_euler,
    rotate_vector,
)
from .chopsticks import (
    LandingChopsticks,
    apply_chopstick_contacts,
    catch_heading_aligned,
    chopsticks_for_pad,
    STICK_BACKSTOP_WEST_M,
    stick_geometry_faults,
)
from .controller import (
    AltitudeForceController,
    AttitudePIDConfig,
    AttitudePIDController,
    PositionPIDConfig,
    PositionPIDController,
)
from .dynamics import (
    VehicleProperties,
    VehicleState,
    WindField,
    compose_offset_mass_properties,
    compose_pivot_mass_properties,
    integrate_6dof_rk4,
    quadratic_drag_force_inertial,
    thrust_wrench_about_com,
)
from .flight_log import FlightLog, FlightLogMetadata, FlightLogSample
from .motors import thrusts_at
from .parameters import FlightTestConfig, LandingAcceptanceBox, RealLandingMission, SystemConfig, WindConfig


QuaternionRefFn = Callable[[float], np.ndarray]
PositionRefFn = Callable[[float], np.ndarray]


@dataclass
class FlightScenario:
    name: str
    motion_mode: str
    attitude_reference_fn: QuaternionRefFn | None = None
    position_reference_fn: PositionRefFn | None = None
    # Attitude-only test stand: hold world position and zero velocity each step.
    freeze_translation: bool = False
    waypoint_positions: list[np.ndarray] = field(default_factory=list)
    waypoint_times: list[float] = field(default_factory=list)
    waypoint_advance: str = "timed"
    waypoint_capture_radius_m: float = 2.0
    waypoint_index: int = 0
    # Real-landing multi-phase mission (approach gate → moving descent to pad).
    mission_phase: str = ""  # "", "approach", "landing"
    overhead_target: np.ndarray | None = None  # first waypoint / approach gate
    landing_track_start: np.ndarray | None = None
    landing_track_end: np.ndarray | None = None
    landing_duration_s: float = 7.0
    landing_phase_start_time: float | None = None
    overhead_capture_max_speed_m_s: float = 1.5
    min_approach_time_s: float = 4.0
    # Require capture conditions to hold this long before advancing (settle).
    capture_settle_s: float = 0.0
    forced_descent_time_s: float | None = None
    # When set, pad transition is burn-clock gated (see
    # ``pad_transition_burn_remaining_s``); capture / along-track progress
    # must not start landing early.
    pad_transition_burn_remaining_s: float | None = None
    # See RealLandingMission.stage_capture_radius_m.
    stage_capture_radius_m: float = 0.55
    descent_altitude_margin_m: float = 0.4
    # Soft-pad descent tilt cap (deg). None → use position controller limit.
    landing_max_tilt_deg: float | None = None
    # Optional inertial heading (rad). When set, real-landing slews from
    # approach → landing yaw over the descent track.
    heading_approach_rad: float | None = None
    heading_landing_rad: float | None = None
    # Seconds to slew approach → landing yaw. None uses landing_duration_s,
    # or vertical_hold_s when that window is active and no clock is set.
    heading_slew_s: float | None = None
    heading_slew_start_s: float | None = None
    heading_slew_end_s: float | None = None
    # See RealLandingMission.heading_flip_period_s.
    heading_flip_period_s: float | None = None
    # See RealLandingMission.heading_finish_before_burnout_s.
    heading_finish_before_burnout_s: float | None = None
    # See RealLandingMission.return_speed_m_s. Applied only on the long
    # return, while the staging reference is still more than 2 m away.
    return_speed_m_s: float | None = None
    cross_track_tilt_deg: float | None = None
    cross_track_speed_m_s: float | None = None
    return_tilt_deg: float | None = None
    return_position_error_m: float | None = None
    return_brake_speed_m_s: float | None = None
    # See RealLandingMission.overshoot_tilt_deg.
    overshoot_tilt_deg: float | None = None
    # Pull the parabola end this many metres short of the pad (along the
    # launch→pad line) so the vehicle does not fly through the site.
    path_end_short_m: float = 0.0
    # After the parabola, ease the waypoint from the path end into the pad.
    # The profile is rest-to-rest with an early peak so the last stretch
    # eases out and does not carry speed through the box.
    lateral_slide_s: float = 0.0
    # When true, the slide eases z to landing_track_end instead of holding
    # the parabola-end altitude.
    slide_to_pad_z: bool = False
    # Geometric parabola end z. 0 uses landing_track_end z. Terminal XY still
    # pins to the pad; altitude holds here so a box-center hover is not
    # commanded while T/W is already below 1.
    path_end_altitude_m: float = 0.0
    path_hold_altitude_m: float = 0.0
    # "capture" (default) or "continuous" parabolic loft launch → gate → pad.
    path_shape: str = "capture"
    path_time_law: str = "min_jerk"
    loft_duration_s: float = 2.0
    # Loft / waypoint-0 dwell (s). With a loft target this is the hover after
    # reaching that setpoint. 0 flies through the loft (leave once crested).
    # With no target it is a clock from fire (legacy open-loop loft).
    vertical_hold_s: float = 0.0
    # When set, loft setpoint (absolute world frame) — waypoint 0, typically
    # launch + vertical_hold_offset_m. None → no waypoint (open-loop full
    # axial); logs show NaN refs.
    vertical_hold_target: np.ndarray | None = None
    launch_position: np.ndarray | None = None
    approach_duration_s: float = 4.0
    # After a time latch, ease altitude to the pad hover over this many seconds.
    terminal_settle_s: float = 0.0
    # Continuous: latch the pad hover at this mission time (s). 0 disables.
    terminal_time_s: float = 0.0
    # Internal: first time approach capture conditions became true.
    _settle_ok_since: float | None = None
    # Loft (waypoint 0) capture dwell.
    _vertical_hold_since: float | None = None
    _vertical_hold_done: bool = False
    _last_hold_velocity: np.ndarray | None = None
    # Nominal first-WP used as descent-track origin (not overwritten by live pose).
    _nominal_track_start: np.ndarray | None = None
    _terminal_hold: bool = False
    _cross_track_recovery: bool = False
    _terminal_hold_time: float | None = None
    _terminal_z0: float = 0.0

    def hover_altitude_m(self) -> float | None:
        """Altitude to hold: optional short-burn floor, else parabola bottom."""
        pad_z = None
        if self.landing_track_end is not None:
            pad_z = float(np.asarray(self.landing_track_end, dtype=float).reshape(3)[2])
        if bool(self.slide_to_pad_z) and pad_z is not None:
            return pad_z
        hold = float(self.path_hold_altitude_m or 0.0)
        if hold > 0.0:
            return hold
        path_alt = float(self.path_end_altitude_m or 0.0)
        if path_alt > 0.0:
            return path_alt
        return pad_z

    def has_loft_target(self) -> bool:
        return self.vertical_hold_target is not None

    def in_vertical_hold(self, time: float) -> bool:
        if self._vertical_hold_done:
            return False
        if self.vertical_hold_target is not None:
            return True
        hold = float(self.vertical_hold_s or 0.0)
        if hold <= 1e-9:
            return False
        return float(time) < hold

    def _advance_vertical_hold(
        self,
        time: float,
        position: np.ndarray | None,
        velocity: np.ndarray | None = None,
    ) -> None:
        """End the loft hold after dwell at waypoint 0, or on the legacy clock."""
        if self._vertical_hold_done:
            return
        hold = float(self.vertical_hold_s or 0.0)
        if self.vertical_hold_target is None:
            if hold <= 1e-9 or float(time) >= hold:
                self._vertical_hold_done = True
            return
        if position is None:
            return
        target = np.asarray(self.vertical_hold_target, dtype=float).reshape(3)
        pos = np.asarray(position, dtype=float).reshape(3)
        radius = max(float(self.waypoint_capture_radius_m), 0.0)
        xy_err = float(np.linalg.norm(pos[:2] - target[:2]))
        reached_alt = float(pos[2]) >= float(target[2]) - 1.5
        if velocity is not None:
            self._last_hold_velocity = np.asarray(velocity, dtype=float).reshape(3).copy()
        vel = self._last_hold_velocity
        # Zero dwell: leave as soon as the loft is crested. Requiring a full
        # speed capture would force a hover. Missing vz must not count as
        # crested — velocity-ref updates omit it and would skip the loft.
        slow = True
        if hold > 1e-9:
            if vel is not None:
                speed = float(np.linalg.norm(vel))
                max_speed = max(float(self.overhead_capture_max_speed_m_s), 1.0)
                slow = speed <= max_speed
        else:
            slow = vel is not None and float(vel[2]) <= 1.0
        # A long return leaves the launch site during the climb. The loft
        # waypoint stays over that site, so an XY capture would hold them
        # there until the abort clock. Crest altitude is enough. The 8 m
        # law has no brake speed and still requires the XY capture.
        xy_ok = xy_err <= radius or getattr(self, "return_brake_speed_m_s", None) is not None
        # A leaned launch is still north of that disk at the crest. Waiting
        # for the disk holds the loft until the abort and the descent starts late.
        if getattr(self, "_cross_track_recovery", False):
            xy_ok = True
        if xy_ok and reached_alt and slow:
            if hold <= 1e-9:
                self._vertical_hold_since = float(time)
                self._vertical_hold_done = True
            else:
                if self._vertical_hold_since is None:
                    self._vertical_hold_since = float(time)
                if float(time) - float(self._vertical_hold_since) >= hold:
                    self._vertical_hold_done = True
        else:
            self._vertical_hold_since = None
        # Always leave the loft so a missed capture cannot burn the whole G12.
        loft_z = float(target[2])
        abort_s = max(hold, 0.0) + 4.0 + max(0.0, (loft_z - 10.0) * 0.2)
        if float(time) >= abort_s:
            self._vertical_hold_done = True

    def _vertical_hold_reference(self, position: np.ndarray | None) -> np.ndarray:
        """Loft waypoint if configured; else NaN (no setpoint drawn in viz)."""
        del position
        if self.vertical_hold_target is not None:
            return np.asarray(self.vertical_hold_target, dtype=float).reshape(3).copy()
        return np.full(3, np.nan, dtype=float)

    def reset_waypoints(self) -> None:
        self.waypoint_index = 0
        self._settle_ok_since = None
        self._vertical_hold_since = None
        self._vertical_hold_done = False
        self._last_hold_velocity = None
        self._terminal_hold = False
        self._cross_track_recovery = False
        self._terminal_hold_time = None
        self._terminal_z0 = 0.0
        if self.overhead_target is not None:
            self.mission_phase = "approach"
            self._landing_alt_reset = False
            if self.path_shape == "continuous":
                self.landing_phase_start_time = float(self.approach_duration_s)
            else:
                self.landing_phase_start_time = None
            if self._nominal_track_start is not None:
                self.landing_track_start = np.asarray(
                    self._nominal_track_start, dtype=float
                ).reshape(3).copy()

    def _landing_track_knots(self) -> tuple[list[float], list[np.ndarray]] | None:
        """Absolute-time knots for first waypoint → pad after capture."""
        if (
            self.landing_phase_start_time is None
            or self.landing_track_start is None
            or self.landing_track_end is None
        ):
            return None
        t0 = float(self.landing_phase_start_time)
        descent = max(float(self.landing_duration_s), 0.0)
        times = [t0, t0 + max(descent, 0.0)]
        start = np.asarray(self.landing_track_start, dtype=float).reshape(3).copy()
        end = np.asarray(self.landing_track_end, dtype=float).reshape(3).copy()
        # A slanted slide drops through the sticks while translating. With
        # slide_to_pad_z off, the pad move stays at the staging altitude.
        if not bool(self.slide_to_pad_z):
            end[2] = float(start[2])
        positions = [start, end]
        return times, positions

    @staticmethod
    def _ease_out_cubic(tau: float) -> tuple[float, float, float]:
        """Cubic ease-out on ``tau`` in [0, 1].

        Returns ``(s, ds/dtau, d2s/dtau2)`` so path progress starts quickly and
        velocity falls to zero as the waypoint reaches the pad
        (s→1, ds/dtau→0).
        """
        tau = float(np.clip(tau, 0.0, 1.0))
        remaining = 1.0 - tau
        s = 1.0 - remaining * remaining * remaining
        ds_dtau = 3.0 * remaining * remaining
        d2s_dtau2 = -6.0 * remaining
        return s, ds_dtau, d2s_dtau2

    def _sync_continuous_phase(self, time: float) -> None:
        """Label approach vs landing from scheduled gate time (no capture stop)."""
        if self.path_shape != "continuous":
            return
        if self._terminal_hold:
            self.mission_phase = "landing"
            return
        t_gate = max(float(self.approach_duration_s), 0.0)
        if self._path_time_law() == "projectile":
            sched = self._projectile_schedule()
            if sched is not None:
                t_gate = float(sched["t_apex"])
        if float(time) >= t_gate:
            self.mission_phase = "landing"
            if self.landing_phase_start_time is None or self._path_time_law() == "projectile":
                self.landing_phase_start_time = t_gate
        else:
            self.mission_phase = "approach"

    def _maybe_latch_terminal(
        self,
        position: np.ndarray | None,
        time: float,
        path_z: float | None = None,
    ) -> None:
        """Latch a fixed pad hover at scheduled ``terminal_time_s``."""
        if self._terminal_hold or self.path_shape != "continuous":
            return
        if self.landing_track_end is None:
            return
        t_lock = float(self.terminal_time_s)
        if t_lock <= 0.0 or float(time) < t_lock:
            return
        self._terminal_hold = True
        self._terminal_hold_time = float(time)
        if path_z is not None:
            self._terminal_z0 = float(path_z)
        elif position is not None:
            self._terminal_z0 = float(np.asarray(position, dtype=float).reshape(3)[2])
        else:
            self._terminal_z0 = float(
                np.asarray(self.landing_track_end, dtype=float).reshape(3)[2]
            )
        self.mission_phase = "landing"

    def _continuous_knots(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
        """Launch, gate, pad, and segment durations for the continuous loft."""
        if self.path_shape != "continuous":
            return None
        if self.launch_position is None or self.landing_track_end is None:
            return None
        gate = self._nominal_track_start
        if gate is None:
            gate = self.overhead_target
        if gate is None:
            return None
        t_app = max(float(self.approach_duration_s), 1e-9)
        t_des = max(float(self.landing_duration_s), 1e-9)
        p0 = np.asarray(self.launch_position, dtype=float).reshape(3)
        p2 = np.asarray(self.landing_track_end, dtype=float).reshape(3).copy()
        path_z = float(self.path_end_altitude_m or 0.0)
        if path_z > 0.0:
            p2[2] = path_z
        short = max(float(self.path_end_short_m), 0.0)
        if short > 0.0:
            delta_xy = p2[:2] - p0[:2]
            span = float(np.linalg.norm(delta_xy))
            if span > short + 1e-9:
                p2[:2] = p2[:2] - (short / span) * delta_xy
        return (
            p0,
            np.asarray(gate, dtype=float).reshape(3),
            p2,
            t_app,
            t_des,
        )

    def _path_time_law(self) -> str:
        return str(self.path_time_law or "min_jerk").strip().lower()

    def _continuous_t_end(self, t_app: float, t_des: float) -> float:
        """Total path time. Loft-then-creep is loft + backside, not gate + descent."""
        if self._path_time_law() == "projectile":
            sched = self._projectile_schedule()
            if sched is not None:
                return float(sched["t_end"])
        if self._path_time_law() in {"loft_then_creep", "loft_through"}:
            return max(float(self.loft_duration_s), 1e-9) + max(float(t_des), 1e-9)
        return max(float(t_app), 1e-9) + max(float(t_des), 1e-9)

    def parabola_end_time_s(self) -> float:
        """Clock time when the continuous parabola reaches its geometric end."""
        knots = self._continuous_knots()
        if knots is None:
            return 0.0
        return float(self._continuous_t_end(knots[3], knots[4]))

    _PROJECTILE_G = 9.81

    def _projectile_schedule(self) -> dict[str, float] | None:
        """Boost, ballistic coast to apogee, then min-jerk brake to the pad.

        Downrange speed is constant on the coast (projectile ``v_x``). Coast
        ``σ̇`` is chosen so ``z̈ = 2 a σ̇² = −g`` on the quadratic altitude fit.
        """
        knots = self._continuous_knots()
        if knots is None:
            return None
        p0, gate, p2, _t_app, t_des = knots
        delta_xy = p2[:2] - p0[:2]
        span = float(np.linalg.norm(delta_xy))
        if span <= 1e-12:
            return None
        sigma_g = float(
            np.clip(np.dot(gate[:2] - p0[:2], delta_xy) / (span * span), 0.05, 0.95)
        )
        a, b, _c = self._fit_parabola_z(
            0.0, float(p0[2]), sigma_g, float(gate[2]), 1.0, float(p2[2])
        )
        if a >= -1e-6:
            return None
        # Full-g coast ω. Cap so a rest→ω boost at ~0.4 g excess does not
        # consume the climb — otherwise the vehicle cannot track the throw.
        omega_g = float(np.sqrt(-self._PROJECTILE_G / (2.0 * a)))
        sigma_apex = float(np.clip(-b / (2.0 * a), 0.05, 0.95))
        a_up = 4.0
        omega_cap = float(
            np.sqrt(max(0.7 * sigma_apex * a_up / max(abs(b), 1e-6), 1e-12))
        )
        omega = min(omega_g, omega_cap)
        t_up = sigma_apex / max(omega, 1e-9)
        t_boost = float(min(0.5, 0.5 * t_up))
        t_boost = max(t_boost, 1e-3)
        t_cruise = max(t_up - 0.5 * t_boost, 0.0)
        t_apex = t_boost + t_cruise
        delta_down = max(1.0 - sigma_apex, 1e-6)
        t_down_mono = 3.0 * delta_down / max(omega, 1e-9)
        t_down = min(max(float(t_des), 1e-9), t_down_mono)
        # A following lateral slide needs three live motors. Cap the backside
        # so path-end + slide finish before a short G12 burn (~12 s at −5%).
        slide = max(float(self.lateral_slide_s), 0.0)
        if slide > 0.0:
            t_down = min(t_down, max(2.5, 11.0 - t_apex - slide))
        return {
            "omega": omega,
            "sigma_apex": sigma_apex,
            "t_boost": t_boost,
            "t_cruise": t_cruise,
            "t_apex": t_apex,
            "t_down": t_down,
            "t_end": t_apex + t_down,
            "az": 2.0 * a * omega * omega,
        }

    def _projectile_sigma(self, t: float, *, derivative: int = 0) -> float:
        """σ(t) for the projectile loft. Rest at t=0 and on the pad."""
        sched = self._projectile_schedule()
        if sched is None:
            return 0.0 if derivative <= 0 else 0.0
        omega = float(sched["omega"])
        sigma_apex = float(sched["sigma_apex"])
        t_boost = float(sched["t_boost"])
        t_apex = float(sched["t_apex"])
        t_down = float(sched["t_down"])
        t_end = float(sched["t_end"])
        t = float(t)
        if t <= 0.0:
            return 0.0
        if t >= t_end:
            return 1.0 if derivative <= 0 else 0.0
        if t < t_boost:
            u = t / t_boost
            if derivative <= 0:
                return 0.5 * omega * t_boost * u * u
            if derivative == 1:
                return omega * u
            return omega / t_boost
        if t < t_apex:
            dt = t - t_boost
            if derivative <= 0:
                return 0.5 * omega * t_boost + omega * dt
            if derivative == 1:
                return omega
            return 0.0
        u = float(np.clip((t - t_apex) / t_down, 0.0, 1.0))
        return self._hermite_sigma(
            u, sigma_apex, 1.0, omega, 0.0, t_down, derivative
        )

    @staticmethod
    def _hermite_sigma(
        u: float,
        sigma0: float,
        sigma1: float,
        dsigma0: float,
        dsigma1: float,
        duration: float,
        derivative: int,
    ) -> float:
        """Cubic Hermite σ(t) on u=t/T. ``dsigma*`` are dσ/dt at the ends."""
        u = float(np.clip(u, 0.0, 1.0))
        t_span = max(float(duration), 1e-9)
        m0 = float(dsigma0) * t_span
        m1 = float(dsigma1) * t_span
        s0 = float(sigma0)
        s1 = float(sigma1)
        if derivative <= 0:
            h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
            h10 = u**3 - 2.0 * u**2 + u
            h01 = -2.0 * u**3 + 3.0 * u**2
            h11 = u**3 - u**2
            return h00 * s0 + h10 * m0 + h01 * s1 + h11 * m1
        if derivative == 1:
            d00 = 6.0 * u**2 - 6.0 * u
            d10 = 3.0 * u**2 - 4.0 * u + 1.0
            d01 = -6.0 * u**2 + 6.0 * u
            d11 = 3.0 * u**2 - 2.0 * u
            return (d00 * s0 + d10 * m0 + d01 * s1 + d11 * m1) / t_span
        d200 = 12.0 * u - 6.0
        d210 = 6.0 * u - 4.0
        d201 = -12.0 * u + 6.0
        d211 = 6.0 * u - 2.0
        return (d200 * s0 + d210 * m0 + d201 * s1 + d211 * m1) / (t_span * t_span)

    @staticmethod
    def _fit_parabola_z(
        x0: float,
        z0: float,
        xg: float,
        zg: float,
        x1: float,
        z1: float,
    ) -> tuple[float, float, float]:
        """Coefficients ``z = a x² + b x + c`` through three downrange samples."""
        vand = np.array(
            [
                [x0 * x0, x0, 1.0],
                [xg * xg, xg, 1.0],
                [x1 * x1, x1, 1.0],
            ],
            dtype=float,
        )
        try:
            a, b, c = np.linalg.solve(vand, np.array([z0, zg, z1], dtype=float))
        except np.linalg.LinAlgError:
            return 0.0, 0.0, z0
        return float(a), float(b), float(c)

    @staticmethod
    def _parabola_ds_dsigma(sigma: float, a: float, b: float, span: float) -> float:
        """Arc-length density ds/dσ for z = aσ² + bσ + c, XY linear in σ."""
        return float(np.hypot(span, 2.0 * a * sigma + b))

    @staticmethod
    def _parabola_arc_length(sigma: float, a: float, b: float, span: float) -> float:
        """Closed-form arc length from σ=0 to ``sigma``."""
        sigma = float(np.clip(sigma, 0.0, 1.0))
        m = 2.0 * float(a)
        n = float(b)
        k = float(span)
        if abs(m) < 1e-12:
            return sigma * float(np.hypot(k, n))

        def antideriv(x: float) -> float:
            root = float(np.hypot(x, k))
            if k < 1e-12:
                return 0.5 * x * abs(x)
            return 0.5 * x * root + 0.5 * k * k * float(np.log(abs(x + root) + 1e-30))

        return float((antideriv(m * sigma + n) - antideriv(n)) / m)

    def _sigma_from_arc_length(
        self,
        s: float,
        a: float,
        b: float,
        span: float,
        s_total: float,
    ) -> float:
        """Invert s(σ) with Newton; σ ∈ [0, 1]."""
        s_total = max(float(s_total), 1e-12)
        target = float(np.clip(s, 0.0, s_total))
        sigma = float(np.clip(target / s_total, 0.0, 1.0))
        for _ in range(10):
            err = self._parabola_arc_length(sigma, a, b, span) - target
            dens = self._parabola_ds_dsigma(sigma, a, b, span)
            if dens < 1e-12:
                break
            sigma = float(np.clip(sigma - err / dens, 0.0, 1.0))
            if abs(err) < 1e-10:
                break
        return sigma

    def _uniform_sigma(
        self,
        t: float,
        duration: float,
        a: float,
        b: float,
        span: float,
        *,
        derivative: int = 0,
    ) -> float:
        """σ(t) for constant speed along the parabola (arc-length uniform)."""
        T = max(float(duration), 1e-9)
        s_total = self._parabola_arc_length(1.0, a, b, span)
        speed = s_total / T
        if t <= 0.0:
            if derivative <= 0:
                return 0.0
            dens0 = self._parabola_ds_dsigma(0.0, a, b, span)
            if derivative == 1:
                return speed / max(dens0, 1e-12)
            # d²σ/dt² = -v² (p'·p'') / |p'|⁴, p'' = (0,0,2a)
            return float(-(speed**2) * (2.0 * a) * b / max(dens0, 1e-12) ** 4)
        if t >= T:
            return 1.0 if derivative <= 0 else 0.0
        s = speed * float(t)
        sigma = self._sigma_from_arc_length(s, a, b, span, s_total)
        if derivative <= 0:
            return sigma
        dens = self._parabola_ds_dsigma(sigma, a, b, span)
        if derivative == 1:
            return speed / max(dens, 1e-12)
        # p'·p'' = (2aσ+b) * 2a
        return float(
            -(speed**2) * (2.0 * a) * (2.0 * a * sigma + b) / max(dens, 1e-12) ** 4
        )

    def _sigma_time_law(
        self,
        t: float,
        duration: float,
        t_gate: float,
        sigma_g: float,
        *,
        derivative: int = 0,
    ) -> float:
        """Monotonic path parameter σ(t) ∈ [0, 1].

        ``min_jerk``: rest-to-rest quintic. ``ease_out``: cubic that starts at
        peak path speed and is at rest on the pad. ``loft_then_creep``: ease
        to the gate (``sigma_g``) over ``loft_duration_s``, then min-jerk the
        backside so a tall parabola is flyable at hover T/W. ``loft_through``:
        same split, but the waypoint keeps moving through the gate. ``uniform``
        is handled in ``_eval_continuous_path`` (constant arc-length speed).
        ``projectile`` is handled there as well (boost, −g coast, brake).
        """
        del t_gate
        T = max(float(duration), 1e-9)
        law = self._path_time_law()
        if t <= 0.0:
            if law == "ease_out":
                if derivative <= 0:
                    return 0.0
                if derivative == 1:
                    return 3.0 / T
                return -6.0 / (T * T)
            return 0.0
        if t >= T:
            return 1.0 if derivative <= 0 else 0.0
        if law == "loft_then_creep":
            return self._sigma_loft_then_creep(t, T, float(sigma_g), derivative)
        if law == "loft_through":
            return self._sigma_loft_through(t, T, float(sigma_g), derivative)
        u = float(np.clip(t / T, 0.0, 1.0))
        if law == "ease_out":
            remaining = 1.0 - u
            if derivative <= 0:
                return float(1.0 - remaining * remaining * remaining)
            if derivative == 1:
                return float(3.0 * remaining * remaining / T)
            return float(-6.0 * remaining / (T * T))
        if derivative <= 0:
            return float(10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5)
        if derivative == 1:
            return float((30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / T)
        return float((60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (T * T))

    def _sigma_loft_then_creep(
        self,
        t: float,
        duration: float,
        sigma_g: float,
        derivative: int,
    ) -> float:
        """Ease to the gate on the spike, then min-jerk the backside."""
        T = max(float(duration), 1e-9)
        t_loft = float(np.clip(self.loft_duration_s, 1e-3, 0.8 * T))
        peak = float(np.clip(sigma_g, 0.05, 0.95))
        if t <= t_loft:
            # Min-jerk to the gate: climb on the spike without a huge XY yank.
            u = float(np.clip(t / t_loft, 0.0, 1.0))
            if derivative <= 0:
                s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
                return float(peak * s)
            if derivative == 1:
                ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / t_loft
                return float(peak * ds)
            d2s = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (t_loft * t_loft)
            return float(peak * d2s)
        t_creep = max(T - t_loft, 1e-9)
        u = float(np.clip((t - t_loft) / t_creep, 0.0, 1.0))
        span = 1.0 - peak
        # Min-jerk backside: stay high through the post-spike T/W trough, rest at the pad.
        if derivative <= 0:
            s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            return float(peak + span * s)
        if derivative == 1:
            ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / t_creep
            return float(span * ds)
        d2s = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (t_creep * t_creep)
        return float(span * d2s)

    def _sigma_loft_through(
        self,
        t: float,
        duration: float,
        sigma_g: float,
        derivative: int,
    ) -> float:
        """Loft through the gate at speed, then ease out to rest at path end.

        Cubics match σ and σ̇ at the gate, so the waypoint never parks on the
        spike. Rest at launch and at the parabola end (the lateral slide, if
        any, then starts from rest).
        """
        T = max(float(duration), 1e-9)
        t_loft = float(np.clip(self.loft_duration_s, 1e-3, 0.8 * T))
        t_creep = max(T - t_loft, 1e-9)
        peak = float(np.clip(sigma_g, 0.05, 0.95))
        span = 1.0 - peak
        # Stay inside the cubic overshoot cap m ≤ 3 Δσ on both segments.
        # Keep a visible through-gate speed without a rest, but do not
        # dump the backside so fast the vehicle punches the floor.
        dsigma_dt = 0.55 * min(3.0 * peak / t_loft, 3.0 * span / t_creep)
        if t <= t_loft:
            u = float(np.clip(t / t_loft, 0.0, 1.0))
            return self._hermite_sigma(
                u, 0.0, peak, 0.0, dsigma_dt * t_loft, derivative, t_loft
            )
        u = float(np.clip((t - t_loft) / t_creep, 0.0, 1.0))
        return self._hermite_sigma(
            u, peak, 1.0, dsigma_dt * t_creep, 0.0, derivative, t_creep
        )

    @staticmethod
    def _hermite_sigma(
        u: float,
        p0: float,
        p1: float,
        m0: float,
        m1: float,
        derivative: int,
        duration: float,
    ) -> float:
        """Cubic Hermite in u∈[0,1]; m0/m1 are dσ/du. ``derivative`` is wrt time."""
        u = float(np.clip(u, 0.0, 1.0))
        segment = max(float(duration), 1e-9)
        if derivative <= 0:
            h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
            h10 = u**3 - 2.0 * u**2 + u
            h01 = -2.0 * u**3 + 3.0 * u**2
            h11 = u**3 - u**2
            return float(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
        if derivative == 1:
            dh00 = 6.0 * u**2 - 6.0 * u
            dh10 = 3.0 * u**2 - 4.0 * u + 1.0
            dh01 = -6.0 * u**2 + 6.0 * u
            dh11 = 3.0 * u**2 - 2.0 * u
            return float((dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1) / segment)
        d2h00 = 12.0 * u - 6.0
        d2h10 = 6.0 * u - 4.0
        d2h01 = -12.0 * u + 6.0
        d2h11 = 6.0 * u - 2.0
        return float(
            (d2h00 * p0 + d2h10 * m0 + d2h01 * p1 + d2h11 * m1) / (segment * segment)
        )

    def _path_sigma(
        self,
        t: float,
        t_end: float,
        t_gate: float,
        sigma_g: float,
        a: float,
        b: float,
        span: float,
        *,
        derivative: int = 0,
    ) -> float:
        if self._path_time_law() == "projectile":
            return self._projectile_sigma(t, derivative=derivative)
        if self._path_time_law() == "uniform":
            return self._uniform_sigma(t, t_end, a, b, span, derivative=derivative)
        return self._sigma_time_law(t, t_end, t_gate, sigma_g, derivative=derivative)

    def _eval_continuous_path(
        self,
        time: float,
        *,
        derivative: int = 0,
        position: np.ndarray | None = None,
    ) -> np.ndarray:
        """Moving waypoint along a parabola through launch, gate, and pad.

        Downrange XY is linear in the path parameter; altitude is the unique
        quadratic through launch, gate, and pad (a vertical-axis parabola).
        Time law is ``path_time_law`` along the parabola (the gate is a
        geometric point on the arc, not a hover). ``uniform`` keeps constant
        speed along the arc. ``projectile`` boosts off the rail, coasts with
        ``z̈ ≈ −g`` (constant downrange speed) to apogee, then min-jerk brakes
        to the path end. If ``slide_to_pad_z`` is set, the waypoint then
        becomes the pad (XY + landing-end z): instantly when
        ``lateral_slide_s`` is 0, or eased over that window. Otherwise a
        positive ``lateral_slide_s`` eases XY only at path-end altitude.
        If ``terminal_time_s`` is set, XY pins to the pad at that clock
        time and altitude eases to the pad hover over ``terminal_settle_s``.
        """
        zeros = np.zeros(3, dtype=float)
        knots = self._continuous_knots()
        t = float(time)
        path_z = None
        if knots is not None:
            p0_pre, gate_pre, p2_pre, t_app_pre, t_des_pre = knots
            t_end_pre = self._continuous_t_end(t_app_pre, t_des_pre)
            delta_xy_pre = p2_pre[:2] - p0_pre[:2]
            span_pre = float(np.linalg.norm(delta_xy_pre))
            if span_pre > 1e-12:
                sigma_g_pre = float(
                    np.clip(
                        np.dot(gate_pre[:2] - p0_pre[:2], delta_xy_pre) / (span_pre * span_pre),
                        0.05,
                        0.95,
                    )
                )
                a_pre, b_pre, c_pre = self._fit_parabola_z(
                    0.0, float(p0_pre[2]), sigma_g_pre, float(gate_pre[2]), 1.0, float(p2_pre[2])
                )
                if t <= 0.0:
                    path_z = float(p0_pre[2])
                elif t >= t_end_pre:
                    path_z = float(p2_pre[2])
                else:
                    sigma_pre = self._path_sigma(
                        t,
                        t_end_pre,
                        float(t_app_pre),
                        sigma_g_pre,
                        a_pre,
                        b_pre,
                        span_pre,
                        derivative=0,
                    )
                    path_z = float(a_pre * sigma_pre**2 + b_pre * sigma_pre + c_pre)
        self._maybe_latch_terminal(position, t, path_z=path_z)
        if self._terminal_hold and self.landing_track_end is not None:
            return self._eval_terminal_hold(t, derivative=derivative)
        if knots is None:
            return np.zeros(3, dtype=float)
        p0, gate, p2, t_app, t_des = knots
        t_gate = float(t_app)
        t_end = self._continuous_t_end(t_app, t_des)
        delta_xy = p2[:2] - p0[:2]
        span = float(np.linalg.norm(delta_xy))
        if span <= 1e-12:
            return p2.copy() if derivative <= 0 else zeros.copy()
        sigma_g = float(
            np.clip(np.dot(gate[:2] - p0[:2], delta_xy) / (span * span), 0.05, 0.95)
        )
        # Parameterize downrange by a scalar u that is linear in σ; fit z(u).
        u0, ug, u1 = 0.0, sigma_g, 1.0
        a, b, c = self._fit_parabola_z(u0, float(p0[2]), ug, float(gate[2]), u1, float(p2[2]))
        if t <= 0.0 and self._path_time_law() not in {"ease_out", "uniform"}:
            return p0.copy() if derivative <= 0 else zeros.copy()
        if t >= t_end:
            return self._eval_lateral_slide(
                t, p2, t_end, derivative=derivative, position=position
            )
        sigma = self._path_sigma(t, t_end, t_gate, sigma_g, a, b, span, derivative=0)
        dsigma_dt = self._path_sigma(t, t_end, t_gate, sigma_g, a, b, span, derivative=1)
        d2sigma_dt2 = self._path_sigma(t, t_end, t_gate, sigma_g, a, b, span, derivative=2)
        xy = p0[:2] + sigma * delta_xy
        z = a * sigma**2 + b * sigma + c
        dz_ds = 2.0 * a * sigma + b
        d2z_ds2 = 2.0 * a
        if derivative <= 0:
            return np.array([xy[0], xy[1], z], dtype=float)
        if derivative == 1:
            return np.array(
                [delta_xy[0] * dsigma_dt, delta_xy[1] * dsigma_dt, dz_ds * dsigma_dt],
                dtype=float,
            )
        return np.array(
            [
                delta_xy[0] * d2sigma_dt2,
                delta_xy[1] * d2sigma_dt2,
                d2z_ds2 * (dsigma_dt**2) + dz_ds * d2sigma_dt2,
            ],
            dtype=float,
        )

    def _eval_lateral_slide(
        self,
        time: float,
        path_end: np.ndarray,
        t_end: float,
        *,
        derivative: int = 0,
        position: np.ndarray | None = None,
    ) -> np.ndarray:
        """Move the waypoint from the parabola end to the pad.

        Default: lateral ease at path-end altitude. ``slide_to_pad_z`` aims
        at ``landing_track_end`` (including z); ``lateral_slide_s`` 0 snaps
        there, otherwise the same rest-to-rest ease. The eased waypoint may
        not run more than ``_SLIDE_LEAD_M`` ahead of the vehicle.
        """
        zeros = np.zeros(3, dtype=float)
        start = np.asarray(path_end, dtype=float).reshape(3)
        if self.landing_track_end is None:
            return start.copy() if derivative <= 0 else zeros.copy()
        pad = np.asarray(self.landing_track_end, dtype=float).reshape(3)
        duration = max(float(self.lateral_slide_s), 0.0)
        end_z = float(pad[2]) if bool(self.slide_to_pad_z) else float(start[2])
        end = np.array([pad[0], pad[1], end_z], dtype=float)
        if duration <= 1e-9:
            if bool(self.slide_to_pad_z):
                return end.copy() if derivative <= 0 else zeros.copy()
            return start.copy() if derivative <= 0 else zeros.copy()
        tau = float(np.clip((float(time) - float(t_end)) / duration, 0.0, 1.0))
        s, ds_dtau, d2s_dtau2 = self._slide_ease(tau)
        delta = end - start
        if position is not None:
            pos = np.asarray(position, dtype=float).reshape(3)
            span_xy = float(np.linalg.norm(delta[:2]))
            if span_xy > 1e-9:
                u_veh = float(
                    np.dot(pos[:2] - start[:2], delta[:2]) / (span_xy * span_xy)
                )
                s_cap = min(1.0, max(0.0, u_veh + self._SLIDE_LEAD_M / span_xy))
                if s > s_cap:
                    s = s_cap
                    ds_dtau = 0.0
                    d2s_dtau2 = 0.0
        if derivative <= 0:
            return start + s * delta
        if derivative == 1:
            return (ds_dtau / duration) * delta
        return (d2s_dtau2 / (duration * duration)) * delta

    _SLIDE_LEAD_M = 1.2

    @staticmethod
    def _slide_ease(tau: float) -> tuple[float, float, float]:
        """Rest-to-rest quartic with an early peak and a long ease-out.

        s(0)=s'(0)=0, s(1)=1, s'(1)=s''(1)=0. Peak speed at τ=1/3, then
        two-thirds of the window is deceleration into the pad.
        """
        tau = float(np.clip(tau, 0.0, 1.0))
        s = tau * tau * (6.0 - 8.0 * tau + 3.0 * tau * tau)
        ds_dtau = 12.0 * tau * (1.0 - tau) * (1.0 - tau)
        d2s_dtau2 = 12.0 - 48.0 * tau + 36.0 * tau * tau
        return s, ds_dtau, d2s_dtau2

    def _eval_terminal_hold(self, time: float, *, derivative: int = 0) -> np.ndarray:
        """Pad XY hold; optional smooth descent from latch altitude to pad hover.

        The pad hover is ``landing_track_end`` (z > 0). Settling to the floor
        is a mission fail — z=0 after liftoff and before burnout.
        """
        pad = np.asarray(self.landing_track_end, dtype=float).reshape(3)
        settle = max(float(self.terminal_settle_s), 0.0)
        z0 = float(self._terminal_z0)
        z1 = float(pad[2])
        path_z = float(self.path_end_altitude_m or 0.0)
        if path_z > 0.0:
            z1 = path_z
        hold_z = float(self.path_hold_altitude_m or 0.0)
        if hold_z > 0.0:
            z1 = z0
        if settle <= 1e-9:
            if derivative <= 0:
                out = pad.copy()
                out[2] = z1
                return out
            return np.zeros(3, dtype=float)
        t0 = float(self._terminal_hold_time if self._terminal_hold_time is not None else time)
        tau = float(np.clip((float(time) - t0) / settle, 0.0, 1.0))
        # Smoothstep: rest-to-rest in z so the pad arrival is not a step.
        s = tau * tau * (3.0 - 2.0 * tau)
        ds_dtau = 6.0 * tau * (1.0 - tau)
        d2s_dtau2 = 6.0 - 12.0 * tau
        z = z0 + s * (z1 - z0)
        if derivative <= 0:
            return np.array([pad[0], pad[1], z], dtype=float)
        if derivative == 1:
            vz = (z1 - z0) * ds_dtau / settle
            return np.array([0.0, 0.0, vz], dtype=float)
        az = (z1 - z0) * d2s_dtau2 / (settle * settle)
        return np.array([0.0, 0.0, az], dtype=float)

    def _eval_landing_track(
        self,
        time: float,
        *,
        velocity: bool = False,
        acceleration: bool = False,
    ) -> np.ndarray:
        """Evaluate first-WP → pad track (position, velocity, or accel).

        ``landing_duration_s`` 0 teleports the setpoint to the pad.
        """
        knots = self._landing_track_knots()
        if knots is None:
            return np.zeros(3, dtype=float)
        times, positions = knots
        t0 = float(times[0])
        t1 = float(times[-1])
        start = np.asarray(positions[0], dtype=float)
        end = np.asarray(positions[-1], dtype=float)
        if max(float(self.landing_duration_s), 0.0) <= 1e-9:
            if acceleration or velocity:
                return np.zeros(3, dtype=float)
            return end.copy()
        duration = max(t1 - t0, 1e-9)
        delta = end - start
        if float(time) <= t0:
            if acceleration or velocity:
                return np.zeros(3, dtype=float)
            return start.copy()
        if float(time) >= t1:
            if acceleration or velocity:
                return np.zeros(3, dtype=float)
            return end.copy()
        tau = (float(time) - t0) / duration
        s, ds_dtau, d2s_dtau2 = self._ease_out_cubic(tau)
        if acceleration:
            return (d2s_dtau2 / (duration * duration)) * delta
        if velocity:
            # chain rule: d(pos)/dt = (ds/dtau) * (1/duration) * (end - start)
            return (ds_dtau / duration) * delta
        return (1.0 - s) * start + s * end

    def _begin_landing_track(
        self,
        time: float,
        position: np.ndarray | None,
        *,
        forced: bool,
    ) -> None:
        """Start the pad ease on the holding waypoint.

        The reference continues from that waypoint into the landing site.
        It does not jump up to the live altitude, and it does not step
        sideways onto the vehicle. The ease velocity still leads west, so
        a vehicle that has already passed the hold is not commanded back.
        """
        del forced
        self.mission_phase = "landing"
        self.landing_phase_start_time = float(time)
        self._settle_ok_since = None
        nominal = (
            np.asarray(self.overhead_target, dtype=float).reshape(3).copy()
            if self.overhead_target is not None
            else np.zeros(3, dtype=float)
        )
        if self._nominal_track_start is not None:
            nominal = np.asarray(self._nominal_track_start, dtype=float).reshape(3).copy()
        start = nominal.copy()
        # A level slide that begins behind a vehicle already past the
        # stage pulls it back east and spends the burn. Start at the
        # vehicle so the aim keeps leading toward the pad.
        if (
            position is not None
            and not bool(self.slide_to_pad_z)
            and self.landing_track_end is not None
        ):
            pos = np.asarray(position, dtype=float).reshape(3)
            end_x = float(np.asarray(self.landing_track_end, dtype=float).reshape(3)[0])
            span = end_x - float(start[0])
            if abs(span) > 1e-6:
                u = (float(pos[0]) - float(start[0])) / span
                if 0.0 < u < 1.0:
                    start[0] = float(pos[0])
        self.landing_track_start = start
        if (
            float(self.landing_duration_s) <= 1e-9
            and self.landing_track_end is not None
        ):
            drop = float(start[2]) - float(
                np.asarray(self.landing_track_end, dtype=float).reshape(3)[2]
            )
            if drop > 12.0:
                # ~5 m/s mean ease so a 30 m loft is not a dive onto the gate.
                self.landing_duration_s = min(8.0, max(3.0, drop / 5.0))

    @staticmethod
    def _lerp_knots(
        time: float,
        times: list[float],
        positions: list[np.ndarray],
        *,
        velocity: bool = False,
    ) -> np.ndarray:
        if time <= times[0]:
            return np.zeros(3, dtype=float) if velocity else positions[0].copy()
        if time >= times[-1]:
            return np.zeros(3, dtype=float) if velocity else positions[-1].copy()
        for index in range(len(times) - 1):
            t0 = float(times[index])
            t1 = float(times[index + 1])
            if t0 <= time <= t1:
                dt = t1 - t0
                if dt <= 1e-9:
                    return np.zeros(3, dtype=float) if velocity else positions[index].copy()
                if velocity:
                    return (positions[index + 1] - positions[index]) / dt
                alpha = float(np.clip((float(time) - t0) / dt, 0.0, 1.0))
                return (1.0 - alpha) * positions[index] + alpha * positions[index + 1]
        return np.zeros(3, dtype=float) if velocity else positions[-1].copy()

    def resolve_heading_reference(self, time: float) -> float | None:
        """Inertial yaw (rad) for thrust+heading attitude, or None if free yaw.

        Raised-cosine slew from ``heading_approach_rad`` to
        ``heading_landing_rad`` so commanded yaw rate is zero at both ends
        (a linear ramp left residual wz that inverted tilt after apogee).
        """
        progressed = self._heading_slew_progress(time)
        if progressed is None:
            return None
        heading, _rate = progressed
        if self._heading_flip_due(time):
            heading = float(heading) + np.pi
        return heading

    def resolve_heading_rate(self, time: float) -> float:
        """Commanded inertial yaw rate (rad/s) for the heading slew, else 0."""
        progressed = self._heading_slew_progress(time)
        if progressed is None:
            return 0.0
        _heading, rate = progressed
        return rate

    def _heading_flip_due(self, time: float) -> bool:
        """True on the odd intervals of a 180° heading flip, and never in landing."""
        period = self.heading_flip_period_s
        if period is None or float(period) <= 0.0:
            return False
        if self.mission_phase == "landing":
            return False
        return int(np.floor(float(time) / float(period))) % 2 == 1

    def _heading_slew_progress(self, time: float) -> tuple[float, float] | None:
        """Return (heading_rad, heading_rate_rad_s), or None if yaw is free."""
        if self.heading_approach_rad is None and self.heading_landing_rad is None:
            return None
        approach = float(
            0.0 if self.heading_approach_rad is None else self.heading_approach_rad
        )
        landing = float(
            approach if self.heading_landing_rad is None else self.heading_landing_rad
        )
        delta = landing - approach
        delta = float(((delta + np.pi) % (2.0 * np.pi)) - np.pi)

        def _smooth(tau: float, duration: float) -> tuple[float, float]:
            tau = float(np.clip(tau, 0.0, 1.0))
            s = 0.5 * (1.0 - np.cos(np.pi * tau))
            ds_dtau = 0.5 * np.pi * np.sin(np.pi * tau)
            heading = approach + s * delta
            rate = ds_dtau * delta / max(duration, 1e-9)
            return heading, rate

        t0 = self.heading_slew_start_s
        t1 = self.heading_slew_end_s
        finish_before = self.heading_finish_before_burnout_s
        if (
            t0 is None
            and finish_before is not None
            and self.mission_phase == "landing"
            and self.landing_phase_start_time is not None
        ):
            # End the turn with thrust still on. A late slide cannot wait
            # for the halfway mark, or the motor dies mid-yaw.
            duration = (
                float(self.heading_slew_s)
                if self.heading_slew_s is not None and float(self.heading_slew_s) > 0.0
                else max(float(self.landing_duration_s), 1e-9)
            )
            slide = float(self.landing_phase_start_time)
            remain = self.pad_transition_burn_remaining_s
            forced = self.forced_descent_time_s
            if remain is not None and forced is not None:
                start = float(forced) + float(remain) - float(finish_before) - duration
            else:
                start = slide + 0.5 * float(self.landing_duration_s)
            halfway = slide + 0.5 * float(self.landing_duration_s)
            start = max(slide, min(start, halfway))
            t0 = start
            t1 = start + duration
        if t0 is not None:
            start = float(t0)
            if t1 is not None and float(t1) > start:
                end = float(t1)
            elif self.heading_slew_s is not None and float(self.heading_slew_s) > 0.0:
                end = start + float(self.heading_slew_s)
            else:
                end = start + max(float(self.landing_duration_s), 1e-9)
            duration = max(end - start, 1e-9)
            if float(time) <= start:
                return approach, 0.0
            if float(time) >= end:
                return landing, 0.0
            tau = (float(time) - start) / duration
            return _smooth(tau, duration)
        if self.mission_phase != "landing" or self.landing_phase_start_time is None:
            return approach, 0.0
        if self._terminal_hold:
            return landing, 0.0
        slew = (
            float(self.heading_slew_s)
            if self.heading_slew_s is not None and float(self.heading_slew_s) > 0.0
            else max(float(self.landing_duration_s), 1e-9)
        )
        duration = max(slew, 1e-9)
        start = float(self.landing_phase_start_time)
        if float(time) <= start:
            return approach, 0.0
        if float(time) >= start + duration:
            return landing, 0.0
        tau = (float(time) - start) / duration
        return _smooth(tau, duration)

    def resolve_velocity_reference(
        self, time: float, position: np.ndarray | None = None
    ) -> np.ndarray:
        """Velocity of a moving (lerp / landing-track) position setpoint."""
        self._sync_continuous_phase(time)
        self._advance_vertical_hold(time, position)
        if self.in_vertical_hold(time):
            return np.zeros(3, dtype=float)
        if self.path_shape == "continuous":
            return self._eval_continuous_path(time, derivative=1, position=position)
        if self.mission_phase == "landing":
            return self._eval_landing_track(time, velocity=True)

        if self.waypoint_advance != "lerp" or len(self.waypoint_positions) < 2:
            return np.zeros(3, dtype=float)
        times = self.waypoint_times
        positions = self.waypoint_positions
        if time <= times[0]:
            return np.zeros(3, dtype=float)
        if time >= times[-1]:
            return np.zeros(3, dtype=float)
        for index in range(len(times) - 1):
            t0 = float(times[index])
            t1 = float(times[index + 1])
            if t0 <= time <= t1:
                dt = t1 - t0
                if dt <= 1e-9:
                    return np.zeros(3, dtype=float)
                return (positions[index + 1] - positions[index]) / dt
        return np.zeros(3, dtype=float)

    def resolve_acceleration_reference(
        self, time: float, position: np.ndarray | None = None
    ) -> np.ndarray:
        """Path acceleration for moving (landing-track) setpoints; else zero."""
        self._sync_continuous_phase(time)
        self._advance_vertical_hold(time, position)
        if self.in_vertical_hold(time):
            return np.zeros(3, dtype=float)
        if self.path_shape == "continuous":
            return self._eval_continuous_path(time, derivative=2, position=position)
        if self.mission_phase == "landing":
            return self._eval_landing_track(time, acceleration=True)
        return np.zeros(3, dtype=float)

    def resolve_position_reference(
        self,
        time: float,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the active position setpoint, advancing capture waypoints in place."""
        self._sync_continuous_phase(time)
        self._advance_vertical_hold(time, position, velocity)
        if self.in_vertical_hold(time):
            return self._vertical_hold_reference(position)
        if self.path_shape == "continuous" and self.overhead_target is not None:
            return self._eval_continuous_path(time, derivative=0, position=position)

        if self.mission_phase in {"approach", "landing"} and self.overhead_target is not None:
            if self.mission_phase == "approach":
                target = np.asarray(self.overhead_target, dtype=float).copy()
                start_landing = False
                forced = False
                burn_gated = self.pad_transition_burn_remaining_s is not None
                if (
                    self.forced_descent_time_s is not None
                    and float(time) >= float(self.forced_descent_time_s)
                ):
                    live_z = (
                        float(np.asarray(position, dtype=float).reshape(3)[2])
                        if position is not None
                        else 0.0
                    )
                    # Finish the sink to the staging waypoint before snapping
                    # to the pad — a pad z_ref from 12 m at -6 m/s is a slam.
                    # Also wait until XY is at/past that side waypoint so the
                    # last 2 m is a lateral slide, not a yank from mid-field.
                    # A ground-level stage is the slide altitude itself, so
                    # the 3 m window would open a slanted path from above it.
                    stage_z = float(target[2])
                    z_window = 0.55 if stage_z < 0.8 else 3.0
                    if live_z <= stage_z + z_window:
                        xy_ready = True
                        if (
                            position is not None
                            and self.landing_track_end is not None
                        ):
                            pos = np.asarray(position, dtype=float).reshape(3)
                            pad = np.asarray(
                                self.landing_track_end, dtype=float
                            ).reshape(3)
                            staging = target[:2]
                            # Pad slide starts on the staging disk, not the
                            # 1.2 m loft capture radius (that yanked from x≈3).
                            stage_r = max(float(self.stage_capture_radius_m), 0.05)
                            xy_err = float(np.linalg.norm(pos[:2] - staging))
                            delta = pad[:2] - staging
                            span = float(np.linalg.norm(delta))
                            past = False
                            if span > 1e-9:
                                u = float(
                                    np.dot(pos[:2] - staging, delta)
                                    / (span * span)
                                )
                                past = u >= -0.05
                            vxy = 0.0
                            if velocity is not None:
                                vxy = float(
                                    np.linalg.norm(
                                        np.asarray(velocity, dtype=float).reshape(3)[:2]
                                    )
                                )
                            xy_ready = (xy_err <= stage_r and vxy <= 1.2) or past
                            # Don't open a level slide while still falling
                            # through the catch slot.
                            if stage_z < 0.8 and velocity is not None:
                                vz_now = float(
                                    np.asarray(velocity, dtype=float).reshape(3)[2]
                                )
                                if vz_now < -1.4:
                                    xy_ready = False
                        if xy_ready:
                            start_landing = True
                            forced = True
                elif (
                    not burn_gated
                    and position is not None
                    and float(time) >= float(self.min_approach_time_s)
                ):
                    radius = max(float(self.waypoint_capture_radius_m), 0.0)
                    pos = np.asarray(position, dtype=float).reshape(3)
                    error = float(np.linalg.norm(pos - target))
                    if velocity is None:
                        speed = 0.0
                    else:
                        speed = float(np.linalg.norm(np.asarray(velocity, dtype=float).reshape(3)))
                    max_speed = max(float(self.overhead_capture_max_speed_m_s), 0.0)
                    settle = max(float(self.capture_settle_s), 0.0)
                    if error <= radius and speed <= max_speed:
                        if self._settle_ok_since is None:
                            self._settle_ok_since = float(time)
                        if float(time) - float(self._settle_ok_since) >= settle:
                            start_landing = True
                    else:
                        self._settle_ok_since = None
                    if (
                        not start_landing
                        and self.has_loft_target()
                        and self.landing_track_end is not None
                    ):
                        pad = np.asarray(self.landing_track_end, dtype=float).reshape(3)
                        delta = pad[:2] - target[:2]
                        span = float(np.linalg.norm(delta))
                        if span > 1e-9:
                            u = float(
                                np.dot(pos[:2] - target[:2], delta) / (span * span)
                            )
                            if u >= 0.0:
                                start_landing = True
                if start_landing:
                    self._begin_landing_track(time, position, forced=forced)
                if self.mission_phase == "approach":
                    return target.copy()

            # Landing mode: ease-out gate → pad, or teleport if duration is 0.
            if self.landing_phase_start_time is None:
                self._begin_landing_track(time, position, forced=False)
            return self._eval_landing_track(time, velocity=False)

        if self.waypoint_positions:
            if self.waypoint_advance == "capture":
                if position is not None:
                    radius = max(float(self.waypoint_capture_radius_m), 0.0)
                    settle = max(float(self.capture_settle_s), 0.0)
                    max_speed = max(float(self.overhead_capture_max_speed_m_s), 0.0)
                    if velocity is None:
                        speed = 0.0
                    else:
                        speed = float(
                            np.linalg.norm(np.asarray(velocity, dtype=float).reshape(3))
                        )
                    # Advance at most one waypoint per call after settle — do not
                    # chain through coincident duplicates in a single step.
                    if self.waypoint_index < len(self.waypoint_positions) - 1:
                        error = float(
                            np.linalg.norm(
                                np.asarray(position, dtype=float)
                                - self.waypoint_positions[self.waypoint_index]
                            )
                        )
                        in_gate = error <= radius and speed <= max_speed
                        if in_gate:
                            if self._settle_ok_since is None:
                                self._settle_ok_since = float(time)
                            if float(time) - float(self._settle_ok_since) >= settle:
                                self.waypoint_index += 1
                                self._settle_ok_since = None
                        else:
                            self._settle_ok_since = None
                return self.waypoint_positions[self.waypoint_index].copy()

            if self.waypoint_advance == "lerp":
                times = self.waypoint_times
                positions = self.waypoint_positions
                if time <= times[0]:
                    return positions[0].copy()
                if time >= times[-1]:
                    return positions[-1].copy()
                for index in range(len(times) - 1):
                    t0 = float(times[index])
                    t1 = float(times[index + 1])
                    if t0 <= time <= t1:
                        dt = t1 - t0
                        alpha = 0.0 if dt <= 1e-9 else (float(time) - t0) / dt
                        alpha = float(np.clip(alpha, 0.0, 1.0))
                        return (1.0 - alpha) * positions[index] + alpha * positions[index + 1]
                return positions[-1].copy()

            active = self.waypoint_positions[0]
            for switch_time, position_m in zip(self.waypoint_times, self.waypoint_positions):
                if time >= switch_time:
                    active = position_m
                else:
                    break
            return active.copy()

        if self.position_reference_fn is not None:
            return np.asarray(self.position_reference_fn(time), dtype=float)
        return np.zeros(3, dtype=float)

    @classmethod
    def hold(cls, euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> FlightScenario:
        roll, pitch, yaw = np.deg2rad(euler_deg)
        q_ref = quaternion_from_euler(roll, pitch, yaw)

        def attitude_reference_fn(_time: float) -> np.ndarray:
            return q_ref

        return cls(
            name="hold",
            motion_mode="attitude",
            attitude_reference_fn=attitude_reference_fn,
            freeze_translation=True,
        )

    @classmethod
    def step(
        cls,
        euler_deg: tuple[float, float, float],
        *,
        step_time: float = 1.0,
        initial_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> FlightScenario:
        q_initial = quaternion_from_euler(*np.deg2rad(initial_euler_deg))
        q_target = quaternion_from_euler(*np.deg2rad(euler_deg))

        def attitude_reference_fn(time: float) -> np.ndarray:
            return q_target if time >= step_time else q_initial

        return cls(
            name="step",
            motion_mode="attitude",
            attitude_reference_fn=attitude_reference_fn,
            freeze_translation=True,
        )

    @classmethod
    def doublet(
        cls,
        axis: str,
        amplitude_deg: float,
        *,
        pulse_width: float = 0.5,
        start_time: float = 1.0,
    ) -> FlightScenario:
        axis_index = {"roll": 0, "pitch": 1, "yaw": 2}[axis]
        amplitude = np.deg2rad(amplitude_deg)

        def attitude_reference_fn(time: float) -> np.ndarray:
            euler = np.zeros(3, dtype=float)
            if start_time <= time < start_time + pulse_width:
                euler[axis_index] = amplitude
            elif start_time + pulse_width <= time < start_time + 2.0 * pulse_width:
                euler[axis_index] = -amplitude
            return quaternion_from_euler(*euler)

        return cls(
            name="doublet",
            motion_mode="attitude",
            attitude_reference_fn=attitude_reference_fn,
            freeze_translation=True,
        )

    @classmethod
    def lateral(
        cls,
        euler_deg: tuple[float, float, float] = (0.0, 8.0, 0.0),
    ) -> FlightScenario:
        roll, pitch, yaw = np.deg2rad(euler_deg)
        q_ref = quaternion_from_euler(roll, pitch, yaw)

        def attitude_reference_fn(_time: float) -> np.ndarray:
            return q_ref

        return cls(
            name="lateral",
            motion_mode="lateral",
            attitude_reference_fn=attitude_reference_fn,
        )

    @classmethod
    def position_step(
        cls,
        target_m: tuple[float, float, float],
        *,
        step_time: float = 1.0,
        initial_target_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> FlightScenario:
        p_initial = np.asarray(initial_target_m, dtype=float)
        p_target = np.asarray(target_m, dtype=float)

        def position_reference_fn(time: float) -> np.ndarray:
            return p_target if time >= step_time else p_initial

        return cls(
            name="position",
            motion_mode="position",
            position_reference_fn=position_reference_fn,
        )

    @classmethod
    def hover(
        cls,
        altitude_m: float = 10.0,
        *,
        xy_m: tuple[float, float] = (0.0, 0.0),
        euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> FlightScenario:
        """Upright attitude hold with altitude closed via axial-force mix."""
        return cls.hover_waypoints(
            [(0.0, (xy_m[0], xy_m[1], altitude_m))],
            euler_deg=euler_deg,
            name="hover",
        )

    @classmethod
    def hover_waypoints(
        cls,
        waypoints: list[tuple[float, tuple[float, float, float]]],
        *,
        euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "hover_waypoints",
        advance: str = "timed",
        capture_radius_m: float = 2.0,
        capture_settle_s: float = 0.0,
        capture_max_speed_m_s: float = 1.5,
    ) -> FlightScenario:
        """Hover mission that advances through position waypoints.

        Each entry is ``(switch_time_s, (x, y, z))``.

        - ``advance="timed"`` (default): active setpoint is the latest waypoint
          whose switch time is ``<=`` the current simulation time.
        - ``advance="capture"``: hold the current waypoint until the vehicle is
          within ``capture_radius_m``, speed ``<= capture_max_speed_m_s``, and
          that condition holds for ``capture_settle_s``. Advances one waypoint
          per call (no instant chain-through of coincident duplicates).
          The final waypoint is held for the rest of the mission.
        - ``advance="lerp"``: linearly interpolate the setpoint between timed
          waypoints (moving reference) and expose path velocity for feedforward.
        """
        if not waypoints:
            raise ValueError("hover_waypoints requires at least one waypoint.")
        advance_mode = str(advance).strip().lower()
        if advance_mode not in {"timed", "capture", "lerp"}:
            raise ValueError(
                f"Unsupported waypoint advance={advance_mode!r}; "
                "expected 'timed', 'capture', or 'lerp'."
            )
        if advance_mode in {"timed", "lerp"}:
            ordered = sorted(
                (
                    (float(switch_time), np.asarray(position_m, dtype=float))
                    for switch_time, position_m in waypoints
                ),
                key=lambda item: item[0],
            )
        else:
            ordered = [
                (float(switch_time), np.asarray(position_m, dtype=float))
                for switch_time, position_m in waypoints
            ]
        roll, pitch, yaw = np.deg2rad(euler_deg)
        q_ref = quaternion_from_euler(roll, pitch, yaw)

        def attitude_reference_fn(_time: float) -> np.ndarray:
            return q_ref

        return cls(
            name=name,
            motion_mode="hover",
            attitude_reference_fn=attitude_reference_fn,
            waypoint_positions=[position_m.copy() for _, position_m in ordered],
            waypoint_times=[switch_time for switch_time, _ in ordered],
            waypoint_advance=advance_mode,
            waypoint_capture_radius_m=float(capture_radius_m),
            capture_settle_s=float(capture_settle_s),
            overhead_capture_max_speed_m_s=float(capture_max_speed_m_s),
            waypoint_index=0,
            # Lock yaw so free quaternion_from_z_to_direction cannot wander
            # under saturated tilt (common failure on large WP steps).
            heading_approach_rad=float(yaw),
            heading_landing_rad=float(yaw),
        )

    @classmethod
    def real_landing(
        cls,
        mission: RealLandingMission,
        *,
        euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "real_landing",
    ) -> FlightScenario:
        """Multi-phase real pad approach.

        1. **Approach:** fly toward (capture) or through (continuous) the first
           waypoint gate (default 5 m east, 5 m up from the pad).
        2. **Landing:** track into the pinned landing site over
           ``descent_duration_s``. Capture mode uses a cubic ease-out that
           starts at peak speed after a hover stop. Continuous mode tracks one
           moving waypoint along a parabola through launch, gate, and pad.
           If ``terminal_time_s`` is set, the reference pins XY to the pad at
           that time; ``terminal_settle_s`` eases altitude to that hover.
           When ``landing_yaw_deg`` differs from ``approach_yaw_deg``, yaw
           slews on a clock window if set; else over ``vertical_hold_s``
           when that window is active; else over the descent track.
           Prefer yawing to catch heading early (e.g. during vertical_hold)
           while deprioritizing yaw in the a2 mixer so roll/pitch + force
           keep gimbal authority under narrow a2.
        """
        roll, pitch, yaw = np.deg2rad(euler_deg)
        q_ref = quaternion_from_euler(roll, pitch, yaw)
        approach_yaw = float(np.deg2rad(mission.approach_yaw_deg))
        landing_yaw = float(np.deg2rad(mission.landing_yaw_deg))
        # Only lock heading when the mission actually requests a yaw track.
        use_heading = abs(landing_yaw - approach_yaw) > 1e-6 or abs(approach_yaw) > 1e-6
        hold_s = float(getattr(mission, "vertical_hold_s", 0.0) or 0.0)
        slew_start = (
            None
            if getattr(mission, "heading_slew_start_s", None) is None
            else float(mission.heading_slew_start_s)
        )
        slew_end = (
            None
            if getattr(mission, "heading_slew_end_s", None) is None
            else float(mission.heading_slew_end_s)
        )
        if hold_s > 1e-9 and slew_start is None:
            slew_start = 0.0
            if slew_end is None:
                slew_end = hold_s

        def attitude_reference_fn(_time: float) -> np.ndarray:
            return q_ref

        first_wp = np.asarray(mission.first_waypoint_m(), dtype=float)
        track_end = np.asarray(mission.landing_end_m(), dtype=float)
        launch = np.asarray(mission.launch_position_m(), dtype=float)
        path_shape = str(getattr(mission, "path_shape", "capture") or "capture").strip().lower()
        if path_shape not in {"capture", "continuous"}:
            raise ValueError(
                f"Unsupported path_shape={path_shape!r}; expected 'capture' or 'continuous'."
            )
        t_gate = float(mission.min_approach_time_s)
        if mission.forced_descent_time_s is not None:
            t_gate = float(mission.forced_descent_time_s)
        return cls(
            name=name,
            motion_mode="hover",
            attitude_reference_fn=attitude_reference_fn,
            waypoint_advance="real_landing",
            waypoint_capture_radius_m=float(mission.capture_radius_m),
            mission_phase="approach",
            overhead_target=first_wp,
            landing_track_start=first_wp.copy(),
            landing_track_end=track_end,
            landing_duration_s=float(mission.descent_duration_s),
            landing_phase_start_time=(t_gate if path_shape == "continuous" else None),
            overhead_capture_max_speed_m_s=float(mission.capture_max_speed_m_s),
            min_approach_time_s=float(mission.min_approach_time_s),
            capture_settle_s=float(mission.capture_settle_s),
            forced_descent_time_s=(
                None
                if mission.forced_descent_time_s is None
                else float(mission.forced_descent_time_s)
            ),
            pad_transition_burn_remaining_s=(
                None
                if getattr(mission, "pad_transition_burn_remaining_s", None) is None
                else float(mission.pad_transition_burn_remaining_s)
            ),
            stage_capture_radius_m=float(
                getattr(mission, "stage_capture_radius_m", 0.55) or 0.55
            ),
            descent_altitude_margin_m=float(mission.descent_altitude_margin_m),
            landing_max_tilt_deg=(
                None
                if mission.landing_max_tilt_deg is None
                else float(mission.landing_max_tilt_deg)
            ),
            heading_approach_rad=approach_yaw if use_heading else None,
            heading_landing_rad=landing_yaw if use_heading else None,
            heading_slew_s=(
                None
                if getattr(mission, "heading_slew_s", None) is None
                else float(mission.heading_slew_s)
            ),
            heading_slew_start_s=slew_start,
            heading_slew_end_s=slew_end,
            heading_flip_period_s=(
                None
                if getattr(mission, "heading_flip_period_s", None) is None
                else float(mission.heading_flip_period_s)
            ),
            heading_finish_before_burnout_s=(
                None
                if getattr(mission, "heading_finish_before_burnout_s", None) is None
                else float(mission.heading_finish_before_burnout_s)
            ),
            return_speed_m_s=(
                None
                if getattr(mission, "return_speed_m_s", None) is None
                else float(mission.return_speed_m_s)
            ),
            cross_track_tilt_deg=(
                None
                if getattr(mission, "cross_track_tilt_deg", None) is None
                else float(mission.cross_track_tilt_deg)
            ),
            cross_track_speed_m_s=(
                None
                if getattr(mission, "cross_track_speed_m_s", None) is None
                else float(mission.cross_track_speed_m_s)
            ),
            return_tilt_deg=(
                None
                if getattr(mission, "return_tilt_deg", None) is None
                else float(mission.return_tilt_deg)
            ),
            return_position_error_m=(
                None
                if getattr(mission, "return_position_error_m", None) is None
                else float(mission.return_position_error_m)
            ),
            return_brake_speed_m_s=(
                None
                if getattr(mission, "return_brake_speed_m_s", None) is None
                else float(mission.return_brake_speed_m_s)
            ),
            overshoot_tilt_deg=(
                None
                if getattr(mission, "overshoot_tilt_deg", None) is None
                else float(mission.overshoot_tilt_deg)
            ),
            path_end_short_m=float(getattr(mission, "path_end_short_m", 0.0) or 0.0),
            lateral_slide_s=float(getattr(mission, "lateral_slide_s", 0.0) or 0.0),
            slide_to_pad_z=bool(getattr(mission, "slide_to_pad_z", False)),
            path_end_altitude_m=float(
                getattr(mission, "path_end_altitude_m", 0.0) or 0.0
            ),
            path_hold_altitude_m=float(
                getattr(mission, "path_hold_altitude_m", 0.0) or 0.0
            ),
            path_shape=path_shape,
            path_time_law=str(getattr(mission, "path_time_law", "min_jerk") or "min_jerk"),
            loft_duration_s=float(getattr(mission, "loft_duration_s", 2.0) or 2.0),
            vertical_hold_s=float(getattr(mission, "vertical_hold_s", 0.0) or 0.0),
            vertical_hold_target=(
                None
                if getattr(mission, "vertical_hold_offset_m", None) is None
                else (
                    launch
                    + np.asarray(mission.vertical_hold_offset_m, dtype=float).reshape(3)
                )
            ),
            launch_position=launch,
            approach_duration_s=t_gate,
            terminal_settle_s=float(getattr(mission, "terminal_settle_s", 0.0) or 0.0),
            terminal_time_s=float(getattr(mission, "terminal_time_s", 0.0) or 0.0),
            _nominal_track_start=first_wp.copy(),
        )


# Backward-compatible alias.
AttitudeScenario = FlightScenario

# Total thrust above this means at least one motor has lit.
_THRUST_LIT_N = 0.5
# After motors have lit, total thrust at or below this is burnout.
_THRUST_BURNOUT_N = 0.25


def find_engine_burnout_index(
    log: FlightLog,
    *,
    lit_threshold_n: float = _THRUST_LIT_N,
    burnout_threshold_n: float = _THRUST_BURNOUT_N,
) -> int | None:
    """Return the first sample index at engine burnout, or None if not observed.

    Burnout is the first sample where total mount thrust drops to
    ``burnout_threshold_n`` or below after total thrust previously exceeded
    ``lit_threshold_n``. Constant-thrust (no curve) flights never burn out.
    """
    if not log.samples:
        return None
    lit = False
    for index, sample in enumerate(log.samples):
        thrusts = np.asarray(sample.mount_thrusts, dtype=float).reshape(-1)
        total = float(np.sum(thrusts)) if thrusts.size else 0.0
        if not lit:
            if total > float(lit_threshold_n):
                lit = True
            continue
        if total <= float(burnout_threshold_n):
            return index
    return None


# Ground contact below this altitude with faster downward speed is a crash.
_HARD_LANDING_ALTITUDE_M = 0.08
_HARD_LANDING_VZ_M_S = -1.5
# Soft floor is z=0. Anything at/below this after liftoff is pad contact.
_GROUND_CONTACT_ALTITUDE_M = 0.01


def find_hard_landing(
    log: FlightLog,
    *,
    ground_altitude_m: float = _HARD_LANDING_ALTITUDE_M,
    max_descent_rate_m_s: float = _HARD_LANDING_VZ_M_S,
) -> tuple[int, float, float] | None:
    """Return ``(index, time_s, vz)`` of the first hard ground contact, if any.

    A hard landing is altitude at/below ``ground_altitude_m`` with vertical
    velocity more negative than ``max_descent_rate_m_s`` (default −1.5 m/s).
    Launch at rest on the pad is ignored (requires engines to have produced
    thrust earlier in the log).

    The sim soft ground plane clamps ``z`` and zeros downward ``vz`` on contact,
    so a slam can appear as one airborne sample (still descending hard) followed
    by an on-pad sample with ``vz≈0``. That transition is treated as a hard
    landing using the pre-impact descent rate.
    """
    lit = False
    prev_z: float | None = None
    prev_vz: float | None = None
    ground = float(ground_altitude_m)
    limit = float(max_descent_rate_m_s)
    for index, sample in enumerate(log.samples):
        thrusts = np.asarray(sample.mount_thrusts, dtype=float).reshape(-1)
        total = float(np.sum(thrusts)) if thrusts.size else 0.0
        if total > _THRUST_LIT_N:
            lit = True
        z = float(sample.position[2])
        vz = float(sample.velocity[2])
        if not lit:
            prev_z, prev_vz = z, vz
            continue
        # Direct: still descending hard while at/near the floor.
        if z <= ground and vz < limit:
            return index, float(sample.time), vz
        # Clamped impact: previous sample was airborne and descending hard,
        # this sample is on the ground plane with downward speed removed.
        if (
            prev_z is not None
            and prev_vz is not None
            and prev_z > ground
            and z <= ground
            and prev_vz < limit
        ):
            return index, float(sample.time), float(prev_vz)
        prev_z, prev_vz = z, vz
    return None


def find_ground_contact_before_burnout(
    log: FlightLog,
    burnout_index: int,
    *,
    ground_altitude_m: float = _GROUND_CONTACT_ALTITUDE_M,
) -> tuple[int, float, float] | None:
    """Return ``(index, time_s, vz)`` of pad contact after liftoff and before burnout.

    Launch sitting on the pad is ignored until the vehicle has been airborne
    (z above ``ground_altitude_m``) with motors lit. After that, any return to
    the floor before the burnout sample fails. Never leaving the pad after
    ignition is also contact.
    """
    if burnout_index <= 0 or not log.samples:
        return None
    ground = float(ground_altitude_m)
    lit = False
    airborne = False
    first_lit: int | None = None
    last_contact: tuple[int, float, float] | None = None
    # Inclusive of the burnout sample: a soft-plane scrape that first appears
    # exactly at burnout must still fail (catch volume can freeze later).
    for index, sample in enumerate(log.samples):
        if index > int(burnout_index):
            break
        thrusts = np.asarray(sample.mount_thrusts, dtype=float).reshape(-1)
        total = float(np.sum(thrusts)) if thrusts.size else 0.0
        if total > _THRUST_LIT_N:
            if not lit:
                first_lit = index
            lit = True
        if not lit:
            continue
        z = float(sample.position[2])
        vz = float(sample.velocity[2])
        if z > ground:
            airborne = True
            continue
        last_contact = (index, float(sample.time), vz)
        if airborne:
            return last_contact
    if lit and not airborne:
        if last_contact is not None:
            return last_contact
        if first_lit is not None:
            sample = log.samples[first_lit]
            return (
                first_lit,
                float(sample.time),
                float(np.asarray(sample.velocity, dtype=float)[2]),
            )
    return None


def evaluate_landing_acceptance(
    log: FlightLog,
    box: LandingAcceptanceBox | None,
    *,
    lit_threshold_n: float = _THRUST_LIT_N,
    burnout_threshold_n: float = _THRUST_BURNOUT_N,
    chopsticks_enabled: bool = False,
    chopsticks_caught: bool = False,
    chopsticks_heading_ok: bool = False,
    stick_backstop: bool = False,
    body_stick_contact: bool = False,
) -> dict[str, object]:
    """Evaluate landing success at engine burnout and annotate log metadata.

    Success requires engine burnout and one of: a two-bar chopstick catch
    (arms rest on the pad sticks — that freezes the plant, including during
    burn), or the vehicle inside the acceptance box at burnout. Hard landing
    or any soft-plane ground contact after liftoff (through burnout) always
    fails — even if the catch volume later freezes the vehicle. A chopstick
    landing also fails when any part of the vehicle passes the stick backstop
    or when the mast or a motor touches a bar. The catch arms may rest on
    the bars.

    Returns a result dict with keys:
    ``evaluated``, ``success``, ``burnout_index``, ``burnout_time_s``,
    ``burnout_position_m``, ``burnout_speed_m_s``, ``reason``.
    """
    meta = log.metadata
    if box is not None:
        for key, value in box.to_metadata_strings().items():
            setattr(meta, key, value)

    empty = {
        "evaluated": False,
        "success": None,
        "burnout_index": None,
        "burnout_time_s": None,
        "burnout_position_m": None,
        "burnout_speed_m_s": None,
        "reason": "no_landing_box",
    }
    if box is None:
        meta.landing_success = ""
        return empty

    burnout_index = find_engine_burnout_index(
        log,
        lit_threshold_n=lit_threshold_n,
        burnout_threshold_n=burnout_threshold_n,
    )
    if burnout_index is None:
        meta.landing_success = "false"
        meta.burnout_time_s = -1.0
        meta.burnout_position_m = ""
        meta.burnout_speed_m_s = -1.0
        return {
            "evaluated": True,
            "success": False,
            "burnout_index": None,
            "burnout_time_s": None,
            "burnout_position_m": None,
            "burnout_speed_m_s": None,
            "reason": "no_engine_burnout",
        }

    sample = log.samples[burnout_index]
    position = np.asarray(sample.position, dtype=float)
    velocity = np.asarray(sample.velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))

    # Ground contact / hard slam always fails — catch must not mask a scrape
    # along the soft plane into the pad volume.
    hard = find_hard_landing(log)
    early = find_ground_contact_before_burnout(log, int(burnout_index))
    if hard is not None and int(hard[0]) <= int(burnout_index):
        success = False
        reason = "hard_landing"
    elif early is not None:
        success = False
        reason = "ground_contact"
    elif stick_backstop:
        success = False
        reason = "backstop"
    elif body_stick_contact:
        success = False
        reason = "body_contact"
    elif chopsticks_enabled and chopsticks_caught:
        success = True
        reason = "caught"
    else:
        success = bool(box.accepts(position, velocity))
        if success:
            reason = "inside_box"
        elif not box.contains_position(position):
            reason = "outside_box"
        else:
            reason = "speed_limit"
        if chopsticks_enabled and success:
            success = False
            reason = "missed_chopsticks" if chopsticks_heading_ok else "bad_heading"

    meta.landing_success = "true" if success else "false"
    meta.burnout_time_s = float(sample.time)
    meta.burnout_position_m = f"{position[0]},{position[1]},{position[2]}"
    meta.burnout_speed_m_s = speed
    if chopsticks_enabled:
        meta.chopstick_caught = "true" if chopsticks_caught else "false"
        meta.chopstick_heading_ok = "true" if chopsticks_heading_ok else "false"

    return {
        "evaluated": True,
        "success": success,
        "burnout_index": burnout_index,
        "burnout_time_s": float(sample.time),
        "burnout_position_m": position.copy(),
        "burnout_speed_m_s": speed,
        "reason": reason,
    }


# Roll/pitch may use 30° of a2. Heading's share of the final a2 is 5°.
_A2_RP_MAX_RAD = np.deg2rad(30.0)
_A2_YAW_MAX_RAD = np.deg2rad(5.0)


def limit_a2_heading_share(
    allocator,
    angles,
    tau_cmd,
    thrusts,
    force_target,
    force_weight: float,
) -> np.ndarray:
    """Cap non-heading a2 at ±30° and the yaw share of a2 at ±5°."""
    angles = np.asarray(angles, dtype=float).copy()
    tau = np.asarray(tau_cmd, dtype=float).reshape(3).copy()
    tau[2] = 0.0
    if float(force_weight) > 0.0:
        angles_rp, _tau, _dirs, _sat = allocator.allocate_force_track(
            tau,
            force_target=force_target,
            force_weight=force_weight,
            thrusts=thrusts,
        )
    else:
        angles_rp, _tau, _dirs, _sat = allocator.allocate(
            tau,
            force_target=None,
            force_weight=0.0,
            thrusts=thrusts,
        )
    angles_rp = np.asarray(angles_rp, dtype=float)
    heading = np.clip(angles[:, 0] - angles_rp[:, 0], -_A2_YAW_MAX_RAD, _A2_YAW_MAX_RAD)
    base = np.clip(angles_rp[:, 0], -_A2_RP_MAX_RAD, _A2_RP_MAX_RAD)
    angles[:, 0] = base + heading
    return np.clip(angles, allocator.min_angles, allocator.max_angles)


def apply_actuator_slew(
    current: np.ndarray,
    commanded: np.ndarray,
    dt: float,
    slew_rate_deg_s: float | None,
    *,
    min_angles: np.ndarray | None = None,
    max_angles: np.ndarray | None = None,
) -> np.ndarray:
    """Rate-limit gimbal angles toward the allocator command (per-axis)."""
    if slew_rate_deg_s is None or slew_rate_deg_s <= 0.0 or dt <= 0.0:
        limited = np.asarray(commanded, dtype=float).copy()
    else:
        max_delta = np.deg2rad(float(slew_rate_deg_s)) * float(dt)
        delta = np.clip(
            np.asarray(commanded, dtype=float) - np.asarray(current, dtype=float),
            -max_delta,
            max_delta,
        )
        limited = np.asarray(current, dtype=float) + delta
    if min_angles is not None and max_angles is not None:
        limited = np.clip(limited, min_angles, max_angles)
    return limited


class FlightSimulator:
    def __init__(
        self,
        system: SystemConfig,
        flight: FlightTestConfig,
        *,
        vehicle: VehicleProperties | None = None,
        controller_config: AttitudePIDConfig | None = None,
        position_config: PositionPIDConfig | None = None,
    ) -> None:
        self.system = system
        self.flight = flight
        src = vehicle or flight.vehicle
        self.vehicle = VehicleProperties(
            inertia=np.asarray(src.inertia, dtype=float).copy(),
            mass=float(src.mass),
            gravity=np.asarray(src.gravity, dtype=float).copy(),
        )
        self.attitude_controller = AttitudePIDController(
            controller_config or flight.controller
        )
        self.position_controller = PositionPIDController(
            position_config or flight.position_controller
        )
        self.altitude_controller = AltitudeForceController(
            flight.altitude_controller
        )
        self.allocator = TorqueAllocator(
            system.mounts,
            system.solver,
            force_axis=system.render.force_axis,
            center_of_mass=system.center_of_mass,
        )
        self.motors = [mount.motor for mount in system.mounts]
        self._uses_thrust_curves = any(motor is not None for motor in self.motors)
        self._engine_seen_lit = np.zeros(len(system.mounts), dtype=bool)
        self._engine_out_hold = False
        self._safe_land_hold = False
        self._safe_land_q_ref: np.ndarray | None = None
        self.fallback_thrusts = self._resolve_fallback_thrusts(flight)
        self._active_force_target = float(flight.force_target)
        self.chopsticks: LandingChopsticks | None = None
        self._chopstick_caught = False
        self._chopstick_heading_ok = False
        self._chopstick_frozen: VehicleState | None = None
        self._chopstick_airborne = False
        self._stick_backstop = False
        self._body_stick_contact = False
        self.wind_config: WindConfig | None = None
        self.wind_field: WindField | None = None
        profile = system.profile
        if profile is not None and profile.wind is not None and profile.wind.enabled():
            self.wind_config = profile.wind
            self.wind_field = WindField(
                profile.wind.velocity_m_s,
                gust_std_m_s=profile.wind.gust_std_m_s,
                gust_tau_s=profile.wind.gust_tau_s,
                seed=profile.wind.seed,
            )
        if (
            profile is not None
            and profile.real_landing is not None
            and system.catch_arms
        ):
            pad = profile.real_landing.landing_end_m()
            self.chopsticks = chopsticks_for_pad(
                (pad[0], pad[1]),
                list(system.catch_arms),
            )

    def _resolve_fallback_thrusts(self, flight: FlightTestConfig) -> np.ndarray:
        """Constant thrusts used only when a mount has no engine curve."""
        if self._uses_thrust_curves:
            # Curve-driven mounts ignore constant thrust and hover scaling.
            return np.array([mount.thrust for mount in self.system.mounts], dtype=float)
        if flight.thrust_per_mount is not None:
            return np.full(len(self.system.mounts), flight.thrust_per_mount, dtype=float)
        # Axial-force mixing needs full available thrust, not rest-pose hover scale.
        if flight.force_weight > 0.0:
            return np.array([mount.thrust for mount in self.system.mounts], dtype=float)
        return self._compute_hover_thrusts()

    def thrusts_at(self, time: float) -> np.ndarray:
        return thrusts_at(self.motors, time, self.fallback_thrusts)

    def _sync_live_mass(self, time: float) -> tuple[float, np.ndarray, np.ndarray]:
        """Update inertial mass / allocator CoM as propellant is spent."""
        mass, com, inertia = compose_pivot_mass_properties(
            self.system.unloaded_vehicle,
            self.system.unloaded_center_of_mass,
            self.system.mounts,
            time_s=time,
        )
        self.vehicle.mass = float(mass)
        self.vehicle.inertia = np.asarray(inertia, dtype=float)
        self.allocator.center_of_mass = np.asarray(com, dtype=float)
        return float(mass), np.asarray(com, dtype=float), np.asarray(inertia, dtype=float)

    def _compute_hover_thrusts(self) -> np.ndarray:
        mount_thrusts = np.array([mount.thrust for mount in self.system.mounts], dtype=float)
        _, force_body, _ = self.allocator.torque_force_dirs(
            self.allocator.rest_angles,
            mount_thrusts,
        )
        gravity = np.asarray(self.vehicle.gravity, dtype=float)
        gravity_mag = float(np.linalg.norm(gravity))
        if gravity_mag < 1e-6:
            return mount_thrusts
        hover_accel = -gravity
        required_force = self.vehicle.mass * hover_accel
        vertical_capacity = float(force_body[2])
        if abs(vertical_capacity) < 1e-6:
            return mount_thrusts
        scale = float(np.clip(required_force[2] / vertical_capacity, 0.05, 1.0))
        return mount_thrusts * scale

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        self.attitude_controller.reset()
        self.position_controller.reset()
        self.altitude_controller.reset()
        self.allocator.angles = self.allocator.rest_angles.copy()
        self.allocator.allocate_count = 0
        self._active_force_target = float(self.flight.force_target)
        self._chopstick_caught = False
        self._chopstick_heading_ok = False
        self._chopstick_frozen = None
        self._chopstick_airborne = False
        self._stick_backstop = False
        self._body_stick_contact = False
        self._engine_seen_lit = np.zeros(len(self.system.mounts), dtype=bool)
        self._engine_out_hold = False
        self._safe_land_hold = False
        self._safe_land_q_ref = None
        self._flex_angle = np.zeros(3, dtype=float)
        self._flex_rate = np.zeros(3, dtype=float)
        self._flex_tau = np.zeros(3, dtype=float)
        if self.wind_field is not None:
            self.wind_field.reset()
        self._sync_live_mass(0.0)
        if initial_state is None:
            roll, pitch, yaw = np.deg2rad(self.flight.initial_euler_deg)
            return VehicleState(
                position=np.asarray(self.flight.initial_position_m, dtype=float),
                velocity=np.zeros(3, dtype=float),
                quaternion=quaternion_from_euler(roll, pitch, yaw),
                omega=np.zeros(3, dtype=float),
            )
        return initial_state.copy()

    def _body_between_chopsticks(self, state: VehicleState) -> bool:
        """True when the body origin is inside the bar length and the gap."""
        sticks = self.chopsticks
        if sticks is None:
            return False
        if not catch_heading_aligned(state.quaternion, max_error_deg=25.0):
            return False
        pos = np.asarray(state.position, dtype=float).reshape(3)
        cx, cy = float(sticks.center_xy[0]), float(sticks.center_xy[1])
        half_x = 0.5 * float(sticks.length_m)
        half_y = 0.5 * float(sticks.separation_m)
        return abs(float(pos[0]) - cx) <= half_x and abs(float(pos[1]) - cy) <= half_y

    def _ease_level_slide(
        self,
        scenario: FlightScenario,
        state: VehicleState,
        hover_pos_ref: np.ndarray,
        velocity_ref: np.ndarray,
    ) -> None:
        """Keep a level slide from running into the bars.

        The reference used to lead by a metre, so the pass hit the arms at
        about 1.5 m/s and the contact freeze counted as a catch. Lead a
        short distance at a walking pace, then stop sideways once the body
        is between the sticks and let the descent do the rest.
        """
        if scenario.mission_phase != "landing" or bool(scenario.slide_to_pad_z):
            return
        if self.chopsticks is None:
            return
        pos = np.asarray(state.position, dtype=float).reshape(3)
        x = float(pos[0])
        speed = 0.80
        vx = float(np.asarray(state.velocity, dtype=float).reshape(3)[0])
        if self._body_between_chopsticks(state):
            # Park in the slot. Chasing the pad center carries the
            # west speed through the bars.
            hold = getattr(self, "_slot_hold_x", None)
            if hold is None:
                hold = x
                self._slot_hold_x = hold
            hover_pos_ref[0] = float(hold)
            hover_pos_ref[1] = float(pos[1])
            velocity_ref[0] = 0.0
            velocity_ref[1] = 0.0
            return
        self._slot_hold_x = None
        # A west tilt already in the servos ignores a matched aim.
        # Put the reference behind the vehicle so the position loop
        # spends tilt on braking.
        if vx < -1.25:
            hover_pos_ref[0] = x + 0.70
            velocity_ref[0] = -0.35
            return
        aim = float(hover_pos_ref[0])
        if aim < x - 0.35:
            hover_pos_ref[0] = x - 0.35
        if float(velocity_ref[0]) < -speed:
            velocity_ref[0] = -speed

    def _chopstick_catch_sink(self, scenario: FlightScenario, state: VehicleState) -> bool:
        """True when the arms can span the bars and the pad is the sticks.

        The floor-proximity thrust boost caps descent near ``z / 1.6`` m/s, so
        the arms stay tens of centimetres above the bars until the motor dies.
        Skip that boost only for a low pad, only once heading puts the arms
        across the gap, and only inside the bar footprint. A high box
        (short-burn), a pad hold, or an arms-parallel approach stays on the boost.
        """
        sticks = self.chopsticks
        if sticks is None or scenario.mission_phase != "landing":
            return False
        if scenario.landing_track_end is None:
            return False
        pad_z = float(np.asarray(scenario.landing_track_end, dtype=float).reshape(3)[2])
        if pad_z >= 0.05:
            return False
        # Arms are body ±X. They span the world-Y gap only near yaw ±90.
        # The approach flies arms parallel to the sticks, so the body can sit
        # in this XY footprint and still be unable to cross both bars.
        if not catch_heading_aligned(state.quaternion, max_error_deg=25.0):
            return False
        pos = np.asarray(state.position, dtype=float).reshape(3)
        if float(pos[2]) > 1.2:
            return False
        half_x = 0.5 * float(sticks.length_m) + 0.025
        arm_len = 0.5
        arms = getattr(self.system, "catch_arms", None) or []
        if arms:
            arm_len = max(float(arm.length_m) for arm in arms)
        half_y = arm_len - 0.5 * float(sticks.separation_m) + 0.02
        if half_y <= 0.05:
            return False
        cx, cy = float(sticks.center_xy[0]), float(sticks.center_xy[1])
        return abs(float(pos[0]) - cx) <= half_x and abs(float(pos[1]) - cy) <= half_y

    def _resolve_force_target(
        self,
        state: VehicleState,
        scenario: FlightScenario,
        position_ref: np.ndarray,
        time: float = 0.0,
        *,
        velocity_ref: np.ndarray | None = None,
        accel_ref: np.ndarray | None = None,
    ) -> float:
        if scenario.motion_mode != "hover":
            return float(self.flight.force_target)
        if scenario.in_vertical_hold(time):
            pref = np.asarray(position_ref, dtype=float).reshape(3)
            if not np.isfinite(float(pref[2])):
                # Legacy open-loop loft: full available axial, no z waypoint.
                self.altitude_controller.reset()
                return float(np.sum(self.thrusts_at(time)))
            # Loft waypoint set (e.g. launch + 10 m Z): chase it hard while
            # below the target — 0.5 s of soft altitude PID never builds the
            # climb energy the catch needs.
            available = float(np.sum(self.thrusts_at(time)))
            captured = scenario._vertical_hold_since is not None
            pref_z = float(pref[2])
            z_now = float(state.position[2])
            vz_now = float(state.velocity[2])
            g = abs(float(np.linalg.norm(self.vehicle.gravity)))
            stop_dist = 0.0
            if vz_now > 0.2:
                # Remaining climb if we command ~0.3 T/W (0.7 g deceleration).
                stop_dist = (vz_now * vz_now) / max(2.0 * 0.7 * g, 1e-6)
            near_loft = z_now >= pref_z - 1.5
            must_brake = z_now + stop_dist >= pref_z - 0.25
            if z_now < pref_z - 0.25 and not captured and not near_loft and not must_brake:
                self.altitude_controller.reset()
                # Open-loop climb still aims at the loft altitude.
                self._guidance_z = float(pref_z)
                return available

        z = float(state.position[2])
        z_ref = float(position_ref[2])
        vz = float(state.velocity[2])
        vz_ref = (
            0.0
            if velocity_ref is None
            else float(np.asarray(velocity_ref, dtype=float).reshape(3)[2])
        )
        az_ref = (
            0.0
            if accel_ref is None
            else float(np.asarray(accel_ref, dtype=float).reshape(3)[2])
        )
        # Near the pad hover, ignore a sinking path rate. Tracking vz_ref < 0
        # at T/W ≈ 1 is what drives z to the floor before burnout.
        pad_z = None
        if scenario.landing_track_end is not None:
            pad_z = float(np.asarray(scenario.landing_track_end, dtype=float)[2])
        hover_z = scenario.hover_altitude_m()
        path_alt = float(getattr(scenario, "path_end_altitude_m", 0.0) or 0.0)
        if (
            hover_z is not None
            and hover_z > 0.2
            and abs(z - hover_z) < 0.40
            and vz_ref < 0.0
        ):
            vz_ref = 0.0
            az_ref = max(az_ref, 0.0)

        # Cruise T/W ≈ 1 cannot brake a ballistic backside. Do not chase a
        # sinking path rate after the ignition spike; follow z_ref gently.
        continuous = scenario.path_shape == "continuous"
        weight_pre = float(self.vehicle.mass) * float(np.linalg.norm(self.vehicle.gravity))
        available_pre = float(np.sum(self.thrusts_at(time)))
        tw_pre = available_pre / max(weight_pre, 1e-6)
        if continuous and tw_pre < 1.12 and vz < 0.10:
            vz_ref = max(vz_ref, -0.45)
            az_ref = max(az_ref, -0.3)
        # Short-burn loft (F10): ``terminal_time_s`` defers the ease to the
        # parabola bottom so T/W < 1 does not track a dive into the floor.
        # Short-burn loft (F10): hold above the parabola bottom so T/W < 1
        # does not track a dive into the floor before burnout.
        hold_z = float(getattr(scenario, "path_hold_altitude_m", 0.0) or 0.0)
        if (
            continuous
            and hold_z > 0.0
            and tw_pre < 1.12
            and vz < 0.10
            and not scenario._terminal_hold
        ):
            z_ref = max(z, hold_z)
            vz_ref = 0.0
            az_ref = 0.0

        gate_z: float | None = None
        approach_gate_z: float | None = None
        if scenario.overhead_target is not None:
            gate_z = float(np.asarray(scenario.overhead_target, dtype=float)[2])
            approach_gate_z = gate_z
        if (
            scenario.mission_phase == "landing"
            and scenario.landing_track_start is not None
        ):
            start_z = float(np.asarray(scenario.landing_track_start, dtype=float)[2])
            # Live loft pose must not replace the 1.5 m approach gate — that
            # made t_rem≈0 and held ~0.98 T, i.e. a 1 m/s sink from 30 m.
            if approach_gate_z is None or start_z <= approach_gate_z + 3.0:
                gate_z = start_z
                # Capture ease-out: never command altitude above the gate.
                # Continuous loft may peak after the gate — do not clip it.
                if not continuous:
                    z_ref = min(z_ref, gate_z)
            if not continuous and not getattr(scenario, "_landing_alt_reset", False):
                self.altitude_controller.integral = 0.0
                scenario._landing_alt_reset = True
            if z > z_ref + 0.25 and self.altitude_controller.integral > 0.0:
                self.altitude_controller.integral *= 0.5
        elif (
            scenario.mission_phase == "approach"
            and gate_z is not None
            and not scenario.in_vertical_hold(time)
        ):
            # Also don't ask to climb above the first waypoint on the way in.
            # Skip during vertical_hold loft (e.g. launch+10 m Z setpoint).
            if not continuous:
                z_ref = min(z_ref, gate_z)

        # After loft hover, command a sink toward the second waypoint. A step
        # z_ref=1.5 m saturates P at force_min (17 m/s slam). Hold z_ref≈z and
        # track vz_ref so D sets the rate, then capture the gate.
        if (
            scenario.has_loft_target()
            and not scenario.in_vertical_hold(time)
            and approach_gate_z is not None
            and z > approach_gate_z + (0.35 if float(approach_gate_z) < 0.8 else 1.0)
        ):
            drop = z - float(approach_gate_z)
            # The crest used to hover while T/W sat near 1, so the pad slide
            # opened as the motor died. Come down through the middle of the
            # return so the level slide still has burn left. 7 m/s cannot be
            # braked at cruise T/W ≈ 1.15, so the rate steps down with
            # altitude still left to stop.
            low_gate = float(approach_gate_z) < 0.8
            loft_z = 30.0
            if scenario.vertical_hold_target is not None:
                loft_z = float(
                    np.asarray(scenario.vertical_hold_target, dtype=float).reshape(3)[2]
                )
            if low_gate and loft_z < 25.0:
                # A 20 m crest has no room for the 6.5 m/s sink. That rate
                # reaches the floor before the slide altitude.
                if drop > 8.0:
                    vz_cmd = -3.6
                elif drop > 1.6:
                    vz_cmd = -2.4
                else:
                    vz_cmd = -1.0
            elif low_gate:
                # Ground-level stage: stay quick through the middle so the
                # level slide opens with burn still left.
                if drop > 14.0:
                    vz_cmd = -6.5
                elif drop > 3.5:
                    vz_cmd = -3.6
                elif drop > 0.7:
                    vz_cmd = -1.2
                else:
                    vz_cmd = -0.45
            elif drop > 14.0:
                vz_cmd = -6.5
            elif drop > 6.0:
                vz_cmd = -3.0
            elif drop > 3.0:
                vz_cmd = -1.5
            else:
                vz_cmd = -1.0
            vz_ref = min(vz_ref, vz_cmd)
            # Hold the altitude aim on the vehicle until the slide height.
            # A step down to a ground-level gate from a few metres up is a dive.
            glue = 0.5 if float(approach_gate_z) < 0.8 else 2.0
            # Once the slide is open its altitude stays put. Gluing z to
            # the vehicle turns that level segment back into a slope.
            if drop > glue and scenario.mission_phase != "landing":
                z_ref = float(z)
            az_ref = 0.0

        # 30 m return: a saturated altitude P term dives into the ground
        # while the stage is still tens of metres away, and folded nozzles
        # throw away the tilt. Bleed extra height gently until the handoff.
        if (
            getattr(scenario, "return_brake_speed_m_s", None) is not None
            and scenario.mission_phase == "approach"
            and not scenario.in_vertical_hold(time)
            and scenario.overhead_target is not None
        ):
            stage = np.asarray(scenario.overhead_target, dtype=float).reshape(3)
            east = float(state.position[0] - stage[0])
            # Match the sink to the time left before the stage, so a 30 m
            # loft comes down with the translate. A fixed 1.5 m/s hangs
            # until burnout; the loft's 7 m/s step reaches the ground while
            # the vehicle is still far east.
            if east > 3.0 and z > float(stage[2]) + 2.0:
                cruise = float(getattr(scenario, "return_speed_m_s", None) or 4.0)
                v_h = max(abs(float(state.velocity[0])), 0.45 * cruise, 1.5)
                t_go = max(east / v_h, 0.8)
                drop = z - float(stage[2])
                sink = float(np.clip(drop / t_go, 1.2, 6.0))
                z_ref = float(z)
                vz_ref = min(float(vz_ref), -sink)
                az_ref = 0.0

        # Inside the bars, with the pad below the sticks: sink onto them while
        # the motor can still hold attitude. 0.6 m/s is under the 1.5 m/s
        # hard-landing line once the altitude loop closes the last few decimetres.
        # The level slide holds half a metre up. Only after the body is
        # between the sticks does the aim drop, and then slowly.
        catch_sink = self._chopstick_catch_sink(scenario, state)
        slot_vx = float(np.asarray(state.velocity, dtype=float).reshape(3)[0])
        if (
            scenario.mission_phase == "landing"
            and self._body_between_chopsticks(state)
            and z > 0.02
            and slot_vx > -1.10
        ):
            z_ref = min(float(z_ref), 0.06)
            vz_ref = -0.50
            catch_sink = True
        elif catch_sink and z > 0.02:
            vz_ref = min(float(vz_ref), -0.60)

        # Altitude the loop is actually tracking, after the long-return
        # sink replaces a gate that is still tens of metres below.
        self._guidance_z = float(z_ref)
        force = self.altitude_controller.update(
            z,
            vz,
            z_ref,
            self.flight.dt,
            mass=self.vehicle.mass,
            gravity=self.vehicle.gravity,
            vertical_velocity_reference=vz_ref,
            vertical_accel_reference=az_ref,
        )

        if (
            gate_z is not None
            and scenario.mission_phase in {"approach", "landing"}
            and not scenario.in_vertical_hold(time)
        ):
            # Soft anti-climb only when actually above the gate altitude.
            # Do NOT cap force merely because the vehicle lags a fast-falling
            # path reference — that caused free-fall crashes onto z=0.
            # Continuous loft is allowed to peak above the gate; skip the bleed.
            # Also skip during vertical_hold loft toward an above-gate target.
            if scenario.path_shape != "continuous":
                weight = float(self.vehicle.mass) * float(
                    np.linalg.norm(self.vehicle.gravity)
                )
                margin = max(float(scenario.descent_altitude_margin_m), 0.0)
                ceiling = gate_z + margin
                # A ground-level slide sits under 1 m. The old 1.5 m
                # "still lofted" test then caps thrust and the sink
                # cannot be braked before the floor.
                loft_margin = 0.35 if float(gate_z) < 0.8 else 1.5
                lofted = scenario.has_loft_target() and z > gate_z + loft_margin
                if not lofted:
                    force = max(float(self.altitude_controller.config.force_min), force)
                if (
                    z > ceiling or (z > gate_z + 0.1 and vz > 0.2)
                ) and not lofted:
                    # Below hover to bleed excess altitude without free-fall.
                    # Skip during post-loft sink — vz_ref already commands it.
                    force = min(force, 0.72 * weight)
                if self.altitude_controller.config.force_max is not None:
                    force = min(float(self.altitude_controller.config.force_max), force)

        # z=0 after liftoff is a fail. Never command below hover when already
        # under the altitude reference, below the pad hover, or at marginal
        # T/W — a voluntary sink at T/W ≈ 1 is a floor hit before burnout.
        weight = float(self.vehicle.mass) * float(np.linalg.norm(self.vehicle.gravity))
        available = float(np.sum(self.thrusts_at(time)))
        tw = available / max(weight, 1e-6)
        if z + 0.12 < z_ref and vz < 0.20:
            force = max(force, 0.99 * weight)
        elif (
            not catch_sink
            and hover_z is not None
            and z < hover_z - 0.08
            and vz < 0.0
        ):
            force = max(force, 0.99 * weight)
        if (
            not catch_sink
            and hover_z is not None
            and z < hover_z
            and vz < 0.0
            and tw > 1.08
        ):
            force = max(force, available)
        if continuous and vz < 0.10:
            # Hold hover only near the commanded end altitude. Higher up,
            # 0.97*W kills the sink needed to reach a 0.7 m pad from a 3 m
            # coast (G12). F10's geometric end is above the box; use that.
            if hover_z is None or z <= hover_z + 0.50:
                force = max(force, 0.97 * weight)
        # The half-metre band is for a pad hover near 1.5 m. On a
        # ground-level slide it cancels the last of the descent.
        near_hover = 0.15 if hover_z is not None and hover_z < 0.8 else 0.50
        if (
            not catch_sink
            and hover_z is not None
            and z < hover_z + near_hover
            and vz < -0.35
        ):
            force = max(force, available)
        if continuous and z + 0.12 < z_ref and vz < -0.3:
            force = max(force, available)
        # Marginal T/W cannot be asked to sink near the pad: full thrust.
        # On the loft return, more than 8 m above the gate, that same clamp
        # holds a hover at the crest and spends the burn before the slide.
        high_return = (
            scenario.has_loft_target()
            and not scenario.in_vertical_hold(time)
            and approach_gate_z is not None
            and z > float(approach_gate_z) + 8.0
        )
        if not high_return and tw < 1.08 and vz < 0.15 and z <= z_ref + 0.15:
            force = max(force, available)

        # After loft, brake into the second waypoint. Do not splay to 0.2 T —
        # that overrode vz_ref and dove through the gate.
        if (
            scenario.has_loft_target()
            and not scenario.in_vertical_hold(time)
            and gate_z is not None
        ):
            if vz < -2.0 and z < float(gate_z) + 8.0:
                t_rem = (z - float(gate_z)) / max(-vz, 1e-6)
                # A ground-level slide needs the fast sink to continue
                # through a few metres. The 1.5 s brake holds that off.
                t_lim = 0.55 if float(gate_z) < 0.8 else 1.5
                if t_rem < t_lim:
                    force = max(float(force), 0.98 * available)

        # If time-to-pad is short, demand max lift (clamped to available later).
        # On a continuous loft this must not fire at altitude — projectile
        # coast from ~8 m already has t_hit < 1.6 s, and a max-thrust override
        # would pin the vehicle at a hover instead of following the parabola.
        if not catch_sink and z > 0.02 and vz < -0.2:
            t_hit = z / max(-vz, 1e-6)
            # A ground-level slide is the target. The 1.8 m band treats
            # that descent as a floor impact and cancels it.
            low_slide = (
                hover_z is not None
                and hover_z < 0.8
            ) or (
                approach_gate_z is not None and float(approach_gate_z) < 0.8
            )
            band = 0.40 if low_slide else 1.8
            near_floor = hover_z is not None and z < float(hover_z) + band
            gate_floor = float(approach_gate_z) if approach_gate_z is not None else 0.0
            near_gate = z < (gate_floor + band if low_slide else max(2.5, gate_floor + 1.8))
            if t_hit < 1.6 and (near_floor or near_gate):
                ceiling = self.altitude_controller.config.force_max
                boost = float(ceiling) if ceiling is not None else force + 20.0
                force = max(force, boost)

        throttle_min = float(getattr(self.altitude_controller.config, "throttle_min", 0.0) or 0.0)
        if throttle_min > 0.0 and available > 0.0:
            force = max(float(force), throttle_min * available)
        return float(force)

    # Solid-motor cruise T/W is often ~1.2–1.5 after the ignition spike.
    # A 1.8× gate never arms under that regime (no XY tilt, no axial mix →
    # rest-pose climb-out). 1.15× is enough excess for modest tilt (cos30°≈0.87
    # would need ~1.15) while still blocking soft-ignition chatter. Profiles
    # may lower ``flight.hover_thrust_margin`` when cruise T/W is marginal.
    _HOVER_THRUST_MARGIN = 1.15

    def _thrust_margin(self) -> float:
        margin = float(getattr(self.flight, "hover_thrust_margin", self._HOVER_THRUST_MARGIN))
        return margin if margin > 0.0 else self._HOVER_THRUST_MARGIN

    def _heading_settled(self, state: VehicleState, heading_rad: float | None) -> bool:
        """True when yaw is close to the command and yaw rate is small.

        Starting XY tilt while still spinning through the landing heading
        (narrow a2, leftover wz) pointed thrust away from the pad after apogee.
        """
        if heading_rad is None:
            return True
        _roll, _pitch, yaw = quaternion_to_euler(state.quaternion)
        err = float(heading_rad) - float(yaw)
        err = float(((err + np.pi) % (2.0 * np.pi)) - np.pi)
        return abs(err) < np.deg2rad(10.0) and abs(float(state.omega[2])) < 0.5

    def _hover_lift_authorized(
        self,
        thrusts: np.ndarray,
        state: VehicleState,
        position_ref: np.ndarray,
    ) -> bool:
        """True once motors have enough thrust margin to authorize XY tilt."""
        del state, position_ref
        weight = float(self.vehicle.mass) * float(np.linalg.norm(self.vehicle.gravity))
        available = float(np.sum(thrusts))
        return available >= self._thrust_margin() * max(weight, 1e-6)

    def _hover_axial_mix_authorized(self, time: float, thrusts: np.ndarray) -> bool:
        """Gate axial-force mix residual; torque-only until armed.

        If ``flight.axial_mix_arm_s`` is set, arm at that mission time. Otherwise
        fall back to the same thrust-margin gate used for XY tilt.
        """
        arm_s = self.flight.axial_mix_arm_s
        if arm_s is not None:
            return float(time) >= float(arm_s)
        weight = float(self.vehicle.mass) * float(np.linalg.norm(self.vehicle.gravity))
        return float(np.sum(thrusts)) >= self._thrust_margin() * max(weight, 1e-6)

    def _note_engine_thrusts(self, thrusts: np.ndarray, scenario: FlightScenario) -> None:
        """Latch open-loop coast if any previously lit motor dies during landing."""
        values = np.asarray(thrusts, dtype=float).reshape(-1)
        if self._engine_seen_lit.size != values.size:
            self._engine_seen_lit = np.zeros(values.size, dtype=bool)
        self._engine_seen_lit = self._engine_seen_lit | (values > _THRUST_LIT_N)
        shutdown = self._engine_seen_lit & (values <= _THRUST_BURNOUT_N)
        alive = int(np.sum(values > _THRUST_BURNOUT_N))
        # One dead G12 still leaves two motors; freezing then dumps T/W below 1
        # and the vehicle hits the floor before burnout. Coast only when at
        # most one motor is still producing thrust.
        if (
            np.any(shutdown)
            and alive <= 1
            and str(scenario.mission_phase) == "landing"
        ):
            self._engine_out_hold = True

    def _maybe_latch_safe_land(
        self,
        state: VehicleState,
        time: float,
        thrusts: np.ndarray,
    ) -> None:
        """Latch 0.75 T/W axial mix once landing is safe.

        Inside the stick box, slow, and heading-ok: command axial force at
        0.75 of weight so it falls onto the sticks. Attitude hold stays on
        at the latch pose (heading/level) so scatter does not tumble it.
        """
        del thrusts
        if self._safe_land_hold:
            return
        profile = self.system.profile
        box = None if profile is None else profile.landing_acceptance
        if box is None or not box.contains_position(state.position):
            return
        speed = float(np.linalg.norm(np.asarray(state.velocity, dtype=float)))
        v_lim = 1.0
        if box.max_speed_m_s is not None:
            v_lim = min(v_lim, float(box.max_speed_m_s))
        if speed > v_lim:
            return
        if not catch_heading_aligned(state.quaternion, max_error_deg=25.0):
            return
        self._safe_land_q_ref = normalize_quaternion(state.quaternion)
        self._safe_land_hold = True

    def _recover_cross_track(
        self,
        scenario: FlightScenario,
        state: VehicleState,
        time: float,
        hover_pos_ref: np.ndarray,
        cfg,
    ) -> None:
        """Open north-south authority after a leaned launch.

        The profile Y cap is a few degrees and under 1 m/s, which cannot
        stop a 30° north lean. While the miss is large, raise the Y tilt
        and speed, let the position error through, and keep the downrange
        axis alive. Drop back to the profile law once the body is near the
        centerline and slow, so the pad catch stays on the gentle gains.
        """
        tilt = getattr(scenario, "cross_track_tilt_deg", None)
        speed = getattr(scenario, "cross_track_speed_m_s", None)
        if tilt is None or speed is None:
            return
        pos = np.asarray(state.position, dtype=float).reshape(3)
        vel = np.asarray(state.velocity, dtype=float).reshape(3)
        y = float(pos[1])
        vy = float(vel[1])
        pad_xy = 1.0e9
        if scenario.landing_track_end is not None:
            pad = np.asarray(scenario.landing_track_end, dtype=float).reshape(3)
            pad_xy = float(np.hypot(float(pos[0]) - float(pad[0]), y - float(pad[1])))
        vxy = float(np.hypot(float(vel[0]), vy))
        body_z = rotate_vector(
            state.quaternion, np.array([0.0, 0.0, 1.0], dtype=float)
        )
        north_deg = float(
            np.rad2deg(np.arctan2(float(body_z[1]), max(float(body_z[2]), 1e-3)))
        )
        upright = abs(north_deg) < 8.0
        # The 5° pad law owns the last metre. Release only once the body
        # is nearly upright — handing off while still leaned restarts the miss.
        # Re-engage only for a fresh large miss, not the residual wobble.
        if pad_xy < 0.70 and vxy < 0.70 and upright:
            scenario._cross_track_recovery = False
        elif scenario._cross_track_recovery:
            if abs(y) < 0.80 and abs(vy) < 0.55 and upright:
                scenario._cross_track_recovery = False
        elif abs(y) > 2.0 or abs(vy) > 2.0:
            scenario._cross_track_recovery = True
        if not scenario._cross_track_recovery:
            return
        # Full tilt only while still running away. The ride back uses a
        # small cone so the attitude loop can follow it without a 20° reversal.
        outbound = y * vy > 0.0
        hard = outbound and (abs(vy) > 3.5 or (abs(y) > 2.5 and abs(vy) > 1.0))
        # Hard brake: Y takes the whole cone so the vehicle stays over the
        # launch site, same as the upright loft. The ride back keeps the
        # profile X speed and damper, with a slightly wider cone so that
        # 8° of Y still leaves X its usual acceleration.
        if hard:
            cfg.max_tilt_y_deg = float(tilt)
            speed_now = float(speed)
            kd_y = 4.0
        else:
            cfg.max_tilt_y_deg = min(float(tilt), 8.0)
            cfg.max_tilt_deg = max(float(cfg.max_tilt_deg), 14.0)
            speed_now = min(float(speed), 1.3)
            kd_y = 2.5
        cfg.prefer_lateral_y = True
        raw_speed = cfg.max_speed_m_s
        if isinstance(raw_speed, (list, tuple, np.ndarray)) and np.size(raw_speed) > 1:
            cfg.max_speed_m_s = (float(raw_speed[0]), float(speed_now))
        else:
            cfg.max_speed_m_s = (float(raw_speed), float(speed_now))
        cfg.max_position_error_m = max(float(cfg.max_position_error_m or 0.0), 12.0)
        cfg.max_position_error_y_m = max(float(cfg.max_position_error_y_m or 0.0), 12.0)
        cfg.lateral_priority_m = 0.0
        kd = np.asarray(cfg.kd, dtype=float).reshape(3).copy()
        kd[1] = max(float(kd[1]), kd_y)
        cfg.kd = kd

    def _guard_stick_backstop(
        self,
        scenario: FlightScenario,
        state: VehicleState,
        heading_rad: float | None,
        hover_pos_ref: np.ndarray,
        velocity_ref: np.ndarray,
        cfg,
        profile_tilt_deg: float,
    ) -> None:
        """Stop a hot slide before the west arm reaches the stick backstop.

        The pad reference runs to the bar center. With the arms still along
        world X, that puts the west tip on the backstop as the origin
        arrives. Hold the chase east of the tip, and spend the overshoot
        tilt on braking when the profile cone cannot stop in the room left.
        """
        overshoot = scenario.overshoot_tilt_deg
        if (
            overshoot is None
            or scenario.mission_phase != "landing"
            or heading_rad is None
            or self.chopsticks is None
        ):
            return
        pos = np.asarray(state.position, dtype=float).reshape(3)
        vel = np.asarray(state.velocity, dtype=float).reshape(3)
        arms = getattr(self.system, "catch_arms", None) or []
        arm = max((float(item.length_m) for item in arms), default=0.5)
        # Mount ring, so a yaw-90 rest does not put a motor through the stop.
        ring = 0.11
        reach = max(arm * abs(float(np.cos(float(heading_rad)))), ring)
        floor = (
            float(self.chopsticks.center_xy[0])
            - float(STICK_BACKSTOP_WEST_M)
            + reach
            + 0.08
        )
        x = float(pos[0])
        vx = float(vel[0])
        room = x - floor
        a_profile = 9.81 * float(
            np.tan(np.deg2rad(max(float(profile_tilt_deg), 1.0)))
        )
        if room > 0.02:
            v_stop = -float(np.sqrt(2.0 * a_profile * room))
            hot = vx < v_stop - 0.12
        else:
            v_stop = 0.0
            hot = vx < -0.20 or x < floor - 0.01
        # Arms still parallel and already inside the last metre: the tilt
        # loop cannot shed a hot slide before the west tip meets the stop.
        # Creep until the yaw has started. The nominal arrival is still
        # east of this line when its yaw leaves 0.
        parallel = abs(float(np.cos(float(heading_rad)))) > float(
            np.cos(np.deg2rad(4.0))
        )
        early = parallel and x < floor + 1.05 and vx < -0.40
        if not hot and not early and x >= floor - 0.01:
            return
        hover_pos_ref[0] = max(float(hover_pos_ref[0]), floor)
        if x < floor:
            v_cmd = 0.30
        elif hot:
            # Too fast for the profile cone. Command a stop and use the
            # extra tilt to brake, instead of chasing the ease west.
            v_cmd = 0.0
        elif early:
            v_cmd = -0.35
        else:
            v_cmd = float(v_stop)
        kp_x = float(np.asarray(cfg.kp, dtype=float).reshape(3)[0])
        pos_err = float(hover_pos_ref[0]) - x
        velocity_ref[0] = float(v_cmd - kp_x * pos_err)
        cfg.max_tilt_deg = max(float(cfg.max_tilt_deg), float(overshoot))

    def _resolve_references(
        self,
        state: VehicleState,
        scenario: FlightScenario,
        time: float,
        *,
        thrusts: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        str,
        bool,
        bool,
        float | None,
        np.ndarray,
        np.ndarray,
    ]:
        """Return (q_ref, pos_ref, att_mode, lift_auth, mix_auth, heading, vel_ref, accel_ref)."""
        position_ref = scenario.resolve_position_reference(
            time,
            state.position,
            state.velocity,
        )
        velocity_ref = scenario.resolve_velocity_reference(
            time, position=state.position
        ).copy()
        accel_ref = scenario.resolve_acceleration_reference(
            time, position=state.position
        ).copy()
        heading_rad = scenario.resolve_heading_reference(time)
        lift_auth = False
        mix_auth = False
        att_mode = "idle"

        if self._safe_land_hold and scenario.motion_mode == "hover":
            att_mode = "safe_land"
            lift_auth = True
            mix_auth = True
            q_ref = (
                self._safe_land_q_ref
                if self._safe_land_q_ref is not None
                else normalize_quaternion(state.quaternion)
            )
            velocity_ref[:2] = 0.0
            accel_ref[:2] = 0.0
            return (
                q_ref,
                position_ref,
                att_mode,
                lift_auth,
                mix_auth,
                heading_rad,
                velocity_ref,
                accel_ref,
            )

        if self._engine_out_hold and scenario.motion_mode == "hover":
            att_mode = "engine_out"
            lift_auth = False
            mix_auth = True
            q_ref = normalize_quaternion(state.quaternion)
            velocity_ref[:2] = 0.0
            accel_ref[:2] = 0.0
            return (
                q_ref,
                position_ref,
                att_mode,
                lift_auth,
                mix_auth,
                heading_rad,
                velocity_ref,
                accel_ref,
            )

        if scenario.motion_mode == "position":
            att_mode = "pos_hold"
            q_ref = self.position_controller.attitude_reference(
                state.position,
                state.velocity,
                position_ref,
                gravity=self.vehicle.gravity,
                dt=self.flight.dt,
            )
        elif scenario.motion_mode == "hover":
            # Capture / flythrough: aim velocity along LOS to the waypoint.
            # Timed / lerp / real-landing: XY via attitude tilt; altitude via
            # axial-force mixer. Real-landing uses position hold (not velocity
            # aim) so long lateral approaches stay stable under G8 dynamics.
            mix_auth = self._hover_axial_mix_authorized(time, thrusts if thrusts is not None else np.zeros(0))
            # Dying solids drop below the hover-margin gate before true burnout.
            # Keep axial mix armed in landing so remaining thrust still fights z=0.
            if (
                str(scenario.mission_phase) == "landing"
                and thrusts is not None
                and float(np.sum(thrusts)) > _THRUST_BURNOUT_N
            ):
                mix_auth = True
            if scenario.in_vertical_hold(time):
                mix_auth = True
            if scenario.in_vertical_hold(time) and scenario.vertical_hold_target is None:
                lift_auth = False
                mix_auth = True
                att_mode = "vertical"
                velocity_ref[:] = 0.0
                accel_ref[:] = 0.0
                # Legacy open-loop loft (no waypoint): upright, no XY tilt.
                # A loft waypoint uses pos_hold below so the vehicle stays on
                # the capture disk long enough to dwell.
                if heading_rad is not None:
                    q_ref = quaternion_from_euler(0.0, 0.0, float(heading_rad))
                else:
                    q_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            elif thrusts is None or not self._hover_lift_authorized(
                thrusts, state, position_ref
            ):
                lift_auth = False
                att_mode = "level"
                # Hold commanded heading (or level) before lift arms so the
                # approach attitude is defined once thrusters come up.
                if heading_rad is not None:
                    q_ref = quaternion_from_euler(0.0, 0.0, float(heading_rad))
                else:
                    q_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                # Single outer law: cascaded position → tilt. No plane_hold /
                # vel_aim / mid-flight max_tilt mutation — one tilt budget from
                # the profile (optionally landing_max_tilt when settled on pad).
                lift_auth = True
                att_mode = "pos_hold"
                hover_pos_ref = position_ref.copy()
                hover_accel_ref = accel_ref.copy()
                hover_pos_ref[2] = float(state.position[2])
                on_moving_path = (
                    scenario.mission_phase == "landing"
                    or scenario.path_shape == "continuous"
                )
                if not on_moving_path:
                    velocity_ref[2] = 0.0
                    hover_accel_ref[:] = 0.0
                else:
                    hover_accel_ref[2] = 0.0

                cfg = self.position_controller.config
                saved_tilt = float(cfg.max_tilt_deg)
                saved_speed = cfg.max_speed_m_s
                saved_err = getattr(cfg, "max_position_error_m", 3.0)
                saved_kd = np.array(cfg.kd, dtype=float).reshape(3).copy()
                saved_tilt_y = getattr(cfg, "max_tilt_y_deg", None)
                saved_err_y = float(getattr(cfg, "max_position_error_y_m", 1.0))
                saved_priority = float(getattr(cfg, "lateral_priority_m", 0.0) or 0.0)
                saved_prefer_y = bool(getattr(cfg, "prefer_lateral_y", False))
                try:
                    tilt_cap = scenario.landing_max_tilt_deg
                    if (
                        tilt_cap is not None
                        and scenario.mission_phase == "landing"
                        and scenario.landing_track_end is not None
                    ):
                        pad = np.asarray(
                            scenario.landing_track_end, dtype=float
                        ).reshape(3)
                        xy_err = float(
                            np.linalg.norm(state.position[:2] - pad[:2])
                        )
                        vxy = float(np.linalg.norm(state.velocity[:2]))
                        # Only soft-cap once over the pad and slow — do not
                        # rewrite the tilt budget mid-approach.
                        if xy_err < 0.55 and vxy < 0.70:
                            cfg.max_tilt_deg = min(saved_tilt, float(tilt_cap))
                    boost = getattr(scenario, "return_speed_m_s", None)
                    if boost is not None:
                        pos = np.asarray(state.position, dtype=float).reshape(3)
                        brake_speed = getattr(scenario, "return_brake_speed_m_s", None)
                        extra_tilt = getattr(scenario, "return_tilt_deg", None)
                        # Brake speed is the 30 m law. The 8 m profile leaves
                        # it unset and keeps a single return speed.
                        long_return = brake_speed is not None and extra_tilt is not None
                        if (
                            long_return
                            and scenario.in_vertical_hold(time)
                            and scenario.overhead_target is not None
                            and float(pos[2]) >= 1.2
                        ):
                            stage_x = float(
                                np.asarray(
                                    scenario.overhead_target, dtype=float
                                ).reshape(3)[0]
                            )
                            # The loft waypoint is over the launch site. On a
                            # 30 m offset that hold spends the burn climbing
                            # in place. Aim the tilt loop at the stage so the
                            # climb also covers ground. Altitude stays on
                            # position_ref.
                            hover_pos_ref[0] = stage_x
                        ref = np.asarray(hover_pos_ref, dtype=float).reshape(3)
                        vx_now = float(np.asarray(state.velocity, dtype=float).reshape(3)[0])
                        if long_return:
                            # Range to the stage, not height. Keep the extra
                            # tilt inside the handoff while the arrival is
                            # still fast — dropping to the 6 deg cone there
                            # is what let a hot pass coast through the pad.
                            east_now = float(pos[0] - ref[0])
                            far = (
                                float(pos[2]) >= 0.6
                                and float(ref[0]) > 1.0
                                and (east_now > 2.0 or (east_now > -0.5 and vx_now < -1.5))
                            )
                        else:
                            far = (
                                float(ref[2]) >= 1.2
                                and float(ref[0]) > 1.0
                                and abs(float(pos[0] - ref[0])) > 2.0
                            )
                        if far:
                            speed_cap = float(boost)
                            # Long return: ride the cruise cap, then taper so
                            # the handoff 2 m east of the stage is already slow.
                            # Half the tilt is the brake budget so a little
                            # speed-loop lag still stops. Unset on the 8 m law.
                            if long_return:
                                east = float(pos[0] - ref[0])
                                if east > 2.0:
                                    room = east - 2.0
                                    a_brake = (
                                        0.50
                                        * 9.81
                                        * float(np.tan(np.deg2rad(float(extra_tilt))))
                                    )
                                    v_arrive = max(float(brake_speed), 0.3)
                                    v_allow = float(
                                        np.sqrt(v_arrive * v_arrive + 2.0 * a_brake * room)
                                    )
                                    speed_cap = min(speed_cap, v_allow)
                                else:
                                    # Inside the handoff and still fast: creep,
                                    # do not command the cruise in reverse.
                                    speed_cap = 0.5
                            speed_fast = isinstance(saved_speed, (list, tuple, np.ndarray)) and np.size(saved_speed) > 1
                            if speed_fast:
                                cfg.max_speed_m_s = [float(speed_cap), float(saved_speed[1])]
                            else:
                                cfg.max_speed_m_s = float(speed_cap)
                            err_floor = getattr(scenario, "return_position_error_m", None)
                            if err_floor is None:
                                err_floor = 8.0
                            cfg.max_position_error_m = max(float(saved_err or 0.0), float(err_floor))
                            if extra_tilt is not None:
                                cfg.max_tilt_deg = max(float(cfg.max_tilt_deg), float(extra_tilt))
                            if long_return:
                                # kp * error falls below the brake schedule
                                # once the stage is inside the error clip, so
                                # a long return stalls a few metres short.
                                # Cancel that term and command the schedule.
                                # Leave X damping at the profile value: raising
                                # it to spend the whole tilt on a small speed
                                # error rang the cruise and arrived hot.
                                kp_x = float(np.asarray(cfg.kp, dtype=float).reshape(3)[0])
                                err_x = float(ref[0] - pos[0])
                                err_y = float(ref[1] - pos[1])
                                clip = float(cfg.max_position_error_m or 0.0)
                                xy = float(np.hypot(err_x, err_y))
                                if clip > 1e-6 and xy > clip:
                                    err_x *= clip / xy
                                if float(pos[0] - ref[0]) > 2.0:
                                    desired = -float(speed_cap)
                                elif vx_now < -0.4:
                                    desired = -0.4
                                else:
                                    desired = 0.0
                                velocity_ref[0] = float(desired - kp_x * err_x)
                    # The light −X nose walks south after the yaw. Brake that
                    # direction only. 5 deg is already saturated, so open the
                    # Y cap. A north brake pulls the settled catches off the bars.
                    if scenario.mission_phase == "landing" and scenario.overshoot_tilt_deg is not None:
                        drift_pos = np.asarray(state.position, dtype=float).reshape(3)
                        drift_vel = np.asarray(state.velocity, dtype=float).reshape(3)
                        if (
                            float(drift_pos[2]) < 0.60
                            and float(drift_pos[1]) < -0.02
                            and float(drift_vel[1]) < -0.12
                        ):
                            cfg.kd = [
                                float(saved_kd[0]),
                                max(float(saved_kd[1]), 4.5),
                                float(saved_kd[2]),
                            ]
                            cfg.max_tilt_y_deg = max(
                                float(getattr(cfg, "max_tilt_y_deg", 0.0) or 0.0),
                                9.0,
                            )
                            cfg.max_tilt_deg = max(float(cfg.max_tilt_deg), 11.0)
                    self._recover_cross_track(
                        scenario,
                        state,
                        time,
                        hover_pos_ref,
                        cfg,
                    )
                    self._ease_level_slide(
                        scenario,
                        state,
                        hover_pos_ref,
                        velocity_ref,
                    )
                    self._guard_stick_backstop(
                        scenario,
                        state,
                        heading_rad,
                        hover_pos_ref,
                        velocity_ref,
                        cfg,
                        saved_tilt,
                    )
                    # Horizontal aim after the long-return retarget and the
                    # backstop hold. The mission gate can still be the loft
                    # point over the launch site.
                    self._guidance_xy = np.asarray(hover_pos_ref[:2], dtype=float).copy()
                    q_ref = self.position_controller.attitude_reference(
                        state.position,
                        state.velocity,
                        hover_pos_ref,
                        velocity_ref,
                        accel_reference=hover_accel_ref,
                        gravity=self.vehicle.gravity,
                        heading_rad=heading_rad,
                        dt=self.flight.dt,
                    )
                finally:
                    cfg.max_tilt_deg = saved_tilt
                    cfg.max_speed_m_s = saved_speed
                    cfg.max_position_error_m = saved_err
                    cfg.kd = saved_kd
                    cfg.max_tilt_y_deg = saved_tilt_y
                    cfg.max_position_error_y_m = saved_err_y
                    cfg.lateral_priority_m = saved_priority
                    cfg.prefer_lateral_y = saved_prefer_y
        elif scenario.attitude_reference_fn is not None:
            att_mode = "att_ref"
            q_ref = normalize_quaternion(scenario.attitude_reference_fn(time))
        else:
            att_mode = "idle"
            q_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        return (
            normalize_quaternion(q_ref),
            position_ref,
            att_mode,
            lift_auth,
            mix_auth,
            heading_rad,
            velocity_ref,
            accel_ref,
        )

    def _active_waypoint_index(self, scenario: FlightScenario, time: float) -> int:
        """Best-effort active waypoint index for telemetry (capture / timed / lerp)."""
        n = len(scenario.waypoint_positions)
        if n <= 0:
            return 0
        if scenario.waypoint_advance == "capture":
            return int(np.clip(scenario.waypoint_index, 0, n - 1))
        if scenario.waypoint_advance in {"timed", "lerp"} and scenario.waypoint_times:
            active = 0
            for index, switch_time in enumerate(scenario.waypoint_times):
                if float(time) >= float(switch_time):
                    active = index
                else:
                    break
            return int(np.clip(active, 0, n - 1))
        return int(np.clip(scenario.waypoint_index, 0, n - 1))

    def _sample_telemetry(
        self,
        *,
        scenario: FlightScenario,
        state: VehicleState,
        time: float,
        q_ref: np.ndarray,
        att_mode: str,
        lift_auth: bool,
        mix_auth: bool,
        heading_rad: float | None,
        velocity_ref: np.ndarray,
        force_target: float,
        force_weight: float,
    ) -> dict:
        heading_val = float("nan") if heading_rad is None else float(heading_rad)
        return {
            "mission_phase": str(scenario.mission_phase or ""),
            "waypoint_index": self._active_waypoint_index(scenario, time),
            "waypoint_count": int(len(scenario.waypoint_positions)),
            "waypoint_advance": str(scenario.waypoint_advance or ""),
            "att_mode": str(att_mode),
            "lift_authorized": bool(lift_auth),
            "axial_mix_authorized": bool(mix_auth),
            "force_target_axial": float(force_target),
            "force_weight": float(force_weight),
            "attitude_error": quaternion_error_vector(state.quaternion, q_ref).copy(),
            "att_integral": self.attitude_controller.integral.copy(),
            "alt_integral": float(self.altitude_controller.integral),
            "velocity_reference": np.asarray(velocity_ref, dtype=float).copy(),
            "heading_ref_rad": heading_val,
        }

    def _frame_bend_measurement(self, quaternion: np.ndarray, omega: np.ndarray):
        """IMU reading with a small, well-damped frame-bend mode.

        The mode is a stand-in, not a measured airframe mode. It is a
        second-order bend about each body axis, driven by the torque the
        frame carried on the previous step, and the attitude loop closes
        on this reading. A small bend rings for a few cycles. As the
        angle grows, damping rises so the ring dies before the hard cap,
        and the angle and rate are clipped so a wrong gain cannot run away.
        """
        if not hasattr(self, "_flex_angle"):
            self._flex_angle = np.zeros(3, dtype=float)
            self._flex_rate = np.zeros(3, dtype=float)
            self._flex_tau = np.zeros(3, dtype=float)
        dt = float(self.flight.dt)
        wn = 2.0 * np.pi * 3.0
        cap = np.deg2rad(1.2)
        # A few visible cycles while the bend is small. Damping rises with
        # amplitude so a growing ring is killed before the hard cap.
        zeta = 0.10 + 0.75 * np.clip(np.abs(self._flex_angle) / cap, 0.0, 1.0)
        static_rad = np.deg2rad(0.45)
        bend = static_rad * wn * wn
        # Yaw is frame torsion, kept smaller than the roll/pitch bend.
        drive = np.array([bend, bend, 0.40 * bend], dtype=float)
        tau = np.asarray(self._flex_tau, dtype=float).reshape(3)
        acc = drive * tau - (2.0 * zeta * wn) * self._flex_rate - (wn * wn) * self._flex_angle
        rate = self._flex_rate + acc * dt
        angle = self._flex_angle + rate * dt
        rate_cap = np.deg2rad(20.0)
        hit = np.abs(angle) >= cap
        angle = np.clip(angle, -cap, cap)
        rate = np.where(hit, 0.0, rate)
        rate = np.clip(rate, -rate_cap, rate_cap)
        self._flex_angle = angle
        self._flex_rate = rate
        half = 0.5 * angle
        half_n = float(np.linalg.norm(half))
        if half_n < 1e-12:
            q_flex = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            scale = np.sin(half_n) / half_n
            q_flex = np.array(
                [np.cos(half_n), half[0] * scale, half[1] * scale, half[2] * scale],
                dtype=float,
            )
        q_meas = normalize_quaternion(quaternion_multiply(quaternion, q_flex))
        w_meas = np.asarray(omega, dtype=float).reshape(3) + rate
        return q_meas, w_meas

    def step(
        self,
        state: VehicleState,
        scenario: FlightScenario,
        time: float,
    ) -> tuple[VehicleState, FlightLogSample]:
        self._sync_live_mass(time)
        self._guidance_xy = None
        self._guidance_z = None
        thrusts = self.thrusts_at(time)
        (
            q_ref,
            position_ref,
            att_mode,
            lift_auth,
            mix_auth,
            heading_rad,
            velocity_ref,
            accel_ref,
        ) = self._resolve_references(state, scenario, time, thrusts=thrusts)
        force_target = self._resolve_force_target(
            state,
            scenario,
            position_ref,
            time,
            velocity_ref=velocity_ref,
            accel_ref=accel_ref,
        )
        self._note_engine_thrusts(thrusts, scenario)
        self._maybe_latch_safe_land(state, time, thrusts)
        if self._safe_land_hold:
            att_mode = "safe_land"
            lift_auth = True
            mix_auth = True
            if self._safe_land_q_ref is not None:
                q_ref = self._safe_land_q_ref
            velocity_ref = np.asarray(velocity_ref, dtype=float).copy()
            accel_ref = np.asarray(accel_ref, dtype=float).copy()
            velocity_ref[:2] = 0.0
            accel_ref[:2] = 0.0
        if self._engine_out_hold and not self._safe_land_hold:
            att_mode = "engine_out"
            lift_auth = False
            mix_auth = True
            q_ref = normalize_quaternion(state.quaternion)
            velocity_ref = np.asarray(velocity_ref, dtype=float).copy()
            accel_ref = np.asarray(accel_ref, dtype=float).copy()
            velocity_ref[:2] = 0.0
            accel_ref[:2] = 0.0
        if scenario.motion_mode == "hover":
            # Keep the axial target inside the live thrust budget so the mixer
            # does not abandon collective a1 while saturated on an impossible force.
            # At T/W ≈ 1 the usual 2% mix margin is a commanded sink — z=0
            # after liftoff is a fail, so use the full budget when hover-poor.
            available = float(np.sum(thrusts))
            if available > 0.0:
                weight = float(self.vehicle.mass) * float(
                    np.linalg.norm(self.vehicle.gravity)
                )
                mix_cap = 1.0 if available < 1.12 * weight else 0.98
                force_target = min(float(force_target), mix_cap * available)
                throttle_min = float(
                    getattr(self.altitude_controller.config, "throttle_min", 0.0) or 0.0
                )
                if throttle_min > 0.0:
                    force_target = max(float(force_target), throttle_min * available)
        self._active_force_target = force_target
        force_weight = float(self.flight.force_weight)
        if scenario.motion_mode == "hover" and force_weight <= 0.0:
            force_weight = float(self.system.render.force_weight) or 0.25
        # Startup / soft-ignition: torque-only mix so force tracking cannot steal
        # gimbal authority from disturbance rejection (e.g. sequential motors).
        if scenario.motion_mode == "hover" and not mix_auth:
            force_weight = 0.0

        q_meas, w_meas = self._frame_bend_measurement(state.quaternion, state.omega)

        if self._safe_land_hold:
            tau_cmd = self.attitude_controller.update(
                q_meas,
                w_meas,
                q_ref,
                self.flight.dt,
            )
            weight = float(self.vehicle.mass) * float(
                np.linalg.norm(self.vehicle.gravity)
            )
            available = float(np.sum(thrusts))
            force_target = min(0.75 * weight, max(available, 0.0))
            force_weight = 1.0
            prev_angles = self.allocator.angles.copy()
            if available > 0.0:
                angles_cmd, _tau_cmd_alloc, _dirs, saturated = self.allocator.allocate_force_track(
                    tau_cmd,
                    force_target=force_target,
                    force_weight=force_weight,
                    thrusts=thrusts,
                )
            else:
                angles_cmd = self.allocator.rest_angles.copy()
                saturated = False
        elif self._engine_out_hold:
            tau_cmd = np.zeros(3, dtype=float)
            available = float(np.sum(thrusts))
            weight = float(self.vehicle.mass) * float(
                np.linalg.norm(self.vehicle.gravity)
            )
            force_target = min(max(float(force_target), 0.99 * weight), max(available, 0.0))
            force_weight = max(float(force_weight), 1.0)
            prev_angles = self.allocator.angles.copy()
            if available > 0.0:
                angles_cmd, _tau_cmd_alloc, _dirs, saturated = self.allocator.allocate_force_track(
                    tau_cmd,
                    force_target=force_target,
                    force_weight=force_weight,
                    thrusts=thrusts,
                )
            else:
                angles_cmd = self.allocator.rest_angles.copy()
                saturated = False
        else:
            omega_ref = np.zeros(3, dtype=float)
            if heading_rad is not None:
                # Body wz ≈ inertial heading rate while nearly upright; without
                # this, yaw kd fights the slew and 90° in 1 s never completes.
                omega_ref[2] = float(scenario.resolve_heading_rate(time))
            tau_cmd = self.attitude_controller.update(
                q_meas,
                w_meas,
                q_ref,
                self.flight.dt,
                omega_reference=omega_ref,
            )
            # Soften roll/yaw before mix so pitch + force keep a2 under saturation.
            # Allocator *\_torque_weight only redistributes when axes compete;
            # solo-axis commands still need a smaller pre-scale.
            roll_w = float(getattr(self.allocator, "roll_torque_weight", 1.0))
            yaw_w = float(getattr(self.allocator, "yaw_torque_weight", 1.0))
            if roll_w < 1.0 - 1e-12 or yaw_w < 1.0 - 1e-12:
                tau_cmd = np.asarray(tau_cmd, dtype=float).copy()
                if roll_w < 1.0 - 1e-12:
                    tau_cmd[0] *= max(roll_w, 0.0)
                if yaw_w < 1.0 - 1e-12:
                    tau_cmd[2] *= max(yaw_w, 0.0)
            prev_angles = self.allocator.angles.copy()
            if force_weight > 0.0:
                angles_cmd, _tau_cmd_alloc, _dirs, saturated = self.allocator.allocate_force_track(
                    tau_cmd,
                    force_target=force_target,
                    force_weight=force_weight,
                    thrusts=thrusts,
                )
            else:
                angles_cmd, _tau_cmd_alloc, _dirs, saturated = self.allocator.allocate(
                    tau_cmd,
                    force_target=None,
                    force_weight=0.0,
                    thrusts=thrusts,
                )
        # Roll/pitch a2 stays inside ±30°. Heading may add ±5° on the final a2.
        # Zero thrust has no wrench, so leave the rest command alone.
        if float(np.sum(np.asarray(thrusts, dtype=float))) > 1e-6:
            angles_cmd = limit_a2_heading_share(
                self.allocator,
                angles_cmd,
                tau_cmd,
                thrusts,
                force_target,
                force_weight,
            )
        angles = apply_actuator_slew(
            prev_angles,
            angles_cmd,
            self.flight.dt,
            self.flight.actuator_slew_rate_deg_s,
            min_angles=self.allocator.min_angles,
            max_angles=self.allocator.max_angles,
        )
        self.allocator.angles = angles
        tau_act, force_body, _dirs = self.allocator.torque_force_dirs(angles, thrusts)
        # The bend on the next step is driven by the torque the frame just carried.
        self._flex_tau = np.asarray(tau_act, dtype=float).copy()
        if float(np.linalg.norm(tau_cmd - tau_act)) > self.allocator._torque_tolerance(
            tau_cmd
        ):
            saturated = True

        plant_mass, com_plant, plant_inertia, motor_cgs = compose_offset_mass_properties(
            self.system.unloaded_vehicle,
            self.system.unloaded_center_of_mass,
            self.system.mounts,
            angles,
            standoff_m=self.system.motor_pivot_standoff_m,
            time_s=time,
        )
        tau_thrust, force_thrust = thrust_wrench_about_com(
            self.system.mounts, angles, thrusts, com_plant
        )
        # Extra wrench the mixer does not see: thrust about the offset-CG CoM
        # versus the pivot-composed allocator CoM. Gimbal-rate d'Alembert terms
        # are omitted (finite-differencing them is stiff at the control dt).
        tau_mass = tau_thrust - tau_act
        force_mass = force_thrust - force_body
        force_plant = force_thrust
        tau_plant = tau_thrust
        plant_vehicle = VehicleProperties(
            inertia=plant_inertia,
            mass=plant_mass,
            gravity=self.vehicle.gravity.copy(),
        )
        force_aero = np.zeros(3, dtype=float)
        if self.wind_field is not None and self.wind_config is not None:
            wind_vel = self.wind_field.step(self.flight.dt)
            force_aero = quadratic_drag_force_inertial(
                state.velocity,
                wind_vel,
                drag_coeff=self.wind_config.drag_coeff,
                ref_area_m2=self.wind_config.ref_area_m2,
                air_density_kg_m3=self.wind_config.air_density_kg_m3,
            )
        force_inertial = rotate_vector(state.quaternion, force_plant) + force_aero
        force_command = np.asarray(self.allocator.force_axis, dtype=float) * float(force_target)
        if self._chopstick_frozen is not None:
            # Mechanically in the sticks: remaining thrust cannot move the plant.
            next_state = self._chopstick_frozen.copy()
        else:
            next_state = integrate_6dof_rk4(
                state,
                force_plant,
                tau_plant,
                plant_vehicle,
                self.flight.dt,
                force_inertial_extra=force_aero,
            )
            if bool(getattr(scenario, "freeze_translation", False)):
                next_state.position = np.asarray(state.position, dtype=float).copy()
                next_state.velocity = np.zeros(3, dtype=float)
            if self.chopsticks is not None and self.system.catch_arms:
                if float(next_state.position[2]) > 0.3:
                    self._chopstick_airborne = True
                next_state, n_hit, heading_ok = apply_chopstick_contacts(
                    next_state,
                    plant_vehicle,
                    list(self.system.catch_arms),
                    self.chopsticks,
                    com_body=com_plant,
                    dt=self.flight.dt,
                )
                if heading_ok or catch_heading_aligned(
                    next_state.quaternion,
                    max_error_deg=float(self.chopsticks.heading_tol_deg),
                ):
                    self._chopstick_heading_ok = True
                # Both bars, from arm contact. A padded box around the pad
                # (|y| up to gap/2 + 15 cm, body z up to 12 cm) froze the
                # vehicle outside the sticks with the arms still above them.
                caught = self._chopstick_airborne and n_hit >= 2
                if caught:
                    self._chopstick_caught = True
                    next_state.velocity = np.zeros(3, dtype=float)
                    next_state.omega = np.zeros(3, dtype=float)
                    self._chopstick_frozen = next_state.copy()
                self._note_stick_faults(
                    next_state.position,
                    next_state.quaternion,
                    motor_cgs,
                    gimbal_angles=angles,
                )
        # Soft ground plane for hover touchdowns (z=0 pad). Skip after a catch
        # so the freeze pose is not treated as a floor slam.
        if self._chopstick_frozen is None and float(next_state.position[2]) <= 0.0:
            next_state.position[2] = 0.0
            if float(next_state.velocity[2]) < 0.0:
                next_state.velocity[2] = 0.0
        telemetry = self._sample_telemetry(
            scenario=scenario,
            state=state,
            time=time,
            q_ref=q_ref,
            att_mode=att_mode,
            lift_auth=lift_auth,
            mix_auth=mix_auth,
            heading_rad=heading_rad,
            velocity_ref=velocity_ref,
            force_target=force_target,
            force_weight=force_weight,
        )
        sample = FlightLogSample(
            time=time,
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            position_reference=self._logged_position_reference(position_ref),
            quaternion=q_meas.copy(),
            omega=w_meas.copy(),
            quaternion_reference=q_ref.copy(),
            torque_command=tau_cmd.copy(),
            torque_achieved=tau_act.copy(),
            force_command=force_command.copy(),
            force_body=force_body.copy(),
            force_inertial=force_inertial.copy(),
            gimbal_angles=angles.copy(),
            mount_thrusts=thrusts.copy(),
            saturated=saturated,
            com_plant=com_plant.copy(),
            com_alloc=np.asarray(self.allocator.center_of_mass, dtype=float).copy(),
            motor_cgs=motor_cgs.copy(),
            torque_mass=tau_mass.copy(),
            force_mass=force_mass.copy(),
            **telemetry,
        )
        return next_state, sample

    def _logged_position_reference(self, position_ref: np.ndarray) -> np.ndarray:
        """Reference drawn in replay: the aim the loops used this step.

        On a long return the tilt aim leaves the loft gate over the launch
        site and the altitude aim is the loft height, then the live height
        on the way down. The mission gate stays in the controller; the log
        shows the point the vehicle is steering toward.
        """
        logged = np.asarray(position_ref, dtype=float).reshape(3).copy()
        guidance_xy = getattr(self, "_guidance_xy", None)
        guidance_z = getattr(self, "_guidance_z", None)
        if guidance_xy is not None:
            logged[0] = float(guidance_xy[0])
            logged[1] = float(guidance_xy[1])
        if guidance_z is not None and np.isfinite(float(guidance_z)):
            logged[2] = float(guidance_z)
        return logged

    def _apply_pad_transition_burn_gate(self, scenario: FlightScenario) -> None:
        """Arm pad waypoint at ``burnout − pad_transition_burn_remaining_s``.

        Fleet burnout is the latest per-motor end time (ignition delay + scaled
        curve duration). When this gate is configured it becomes the sole pad
        transition trigger (``forced_descent_time_s``).
        """
        remaining = getattr(scenario, "pad_transition_burn_remaining_s", None)
        if remaining is None:
            return
        rem = max(float(remaining), 0.0)
        ends: list[float] = []
        for motor in self.motors:
            if motor is None:
                continue
            scale = max(float(motor.time_scale), 1e-9)
            ends.append(
                float(motor.ignition_delay_s) + float(motor.times_s[-1]) * scale
            )
        if not ends:
            return
        burnout = float(max(ends))
        scenario.forced_descent_time_s = max(0.0, burnout - rem)

    def run(self, scenario: FlightScenario) -> FlightLog:
        scenario.reset_waypoints()
        self._apply_pad_transition_burn_gate(scenario)
        state = self.reset()
        engine_names = [
            motor.key if motor is not None else (mount.engine or "")
            for mount, motor in zip(self.system.mounts, self.motors)
        ]
        log = FlightLog(
            metadata=FlightLogMetadata(
                scenario=scenario.name,
                motion_mode=scenario.motion_mode,
                dt=self.flight.dt,
                duration=self.flight.duration,
                force_target=self.flight.force_target,
                force_weight=self.flight.force_weight,
                notes=self.flight.notes,
                mount_count=len(self.system.mounts),
                engines=",".join(engine_names),
            )
        )
        steps = int(round(self.flight.duration / self.flight.dt))
        for step_index in range(steps + 1):
            time = step_index * self.flight.dt
            if step_index > 0:
                state, sample = self.step(state, scenario, time - self.flight.dt)
                log.append(sample)
            else:
                self._sync_live_mass(time)
                self._guidance_xy = None
                self._guidance_z = None
                thrusts0 = self.thrusts_at(time)
                (
                    q_ref,
                    position_ref,
                    att_mode,
                    lift_auth,
                    mix_auth,
                    heading_rad,
                    velocity_ref,
                    _accel_ref,
                ) = self._resolve_references(state, scenario, time, thrusts=thrusts0)
                force_weight0 = float(self.flight.force_weight)
                if scenario.motion_mode == "hover" and force_weight0 <= 0.0:
                    force_weight0 = float(self.system.render.force_weight) or 0.25
                if scenario.motion_mode == "hover" and not mix_auth:
                    force_weight0 = 0.0
                force_target0 = float(self.flight.force_target)
                telemetry0 = self._sample_telemetry(
                    scenario=scenario,
                    state=state,
                    time=time,
                    q_ref=q_ref,
                    att_mode=att_mode,
                    lift_auth=lift_auth,
                    mix_auth=mix_auth,
                    heading_rad=heading_rad,
                    velocity_ref=velocity_ref,
                    force_target=force_target0,
                    force_weight=force_weight0,
                )
                sample = FlightLogSample(
                    time=time,
                    position=state.position.copy(),
                    velocity=state.velocity.copy(),
                    position_reference=self._logged_position_reference(position_ref),
                    quaternion=state.quaternion.copy(),
                    omega=state.omega.copy(),
                    quaternion_reference=q_ref.copy(),
                    torque_command=np.zeros(3, dtype=float),
                    torque_achieved=np.zeros(3, dtype=float),
                    force_command=np.zeros(3, dtype=float),
                    force_body=np.zeros(3, dtype=float),
                    force_inertial=np.zeros(3, dtype=float),
                    gimbal_angles=self.allocator.angles.copy(),
                    mount_thrusts=thrusts0.copy(),
                    saturated=False,
                    **telemetry0,
                )
                log.append(sample)

        landing_box = None
        if self.system.profile is not None:
            landing_box = self.system.profile.landing_acceptance
        evaluate_landing_acceptance(
            log,
            landing_box,
            chopsticks_enabled=self.chopsticks is not None,
            chopsticks_caught=self._chopstick_caught,
            chopsticks_heading_ok=self._chopstick_heading_ok,
            stick_backstop=self._stick_backstop,
            body_stick_contact=self._body_stick_contact,
        )
        return log

    def _note_stick_faults(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        motor_cgs: np.ndarray,
        gimbal_angles: np.ndarray | None = None,
    ) -> None:
        """Record a backstop crossing or a non-arm hit on the bars."""
        if self.chopsticks is None:
            return
        faults = stick_geometry_faults(
            position,
            quaternion,
            list(self.system.catch_arms),
            list(self.system.mounts),
            self.chopsticks,
            motor_cgs_body=motor_cgs,
            gimbal_angles=gimbal_angles,
            standoff_m=float(self.system.motor_pivot_standoff_m),
        )
        if "backstop" in faults:
            self._stick_backstop = True
        if "body_contact" in faults:
            self._body_stick_contact = True


def summarize_log(log: FlightLog) -> str:
    if not log.samples:
        return "Empty flight log."

    final = log.samples[-1]
    roll, pitch, yaw = quaternion_to_euler(final.quaternion)
    roll_ref, pitch_ref, yaw_ref = quaternion_to_euler(final.quaternion_reference)
    roll_err = np.rad2deg(roll - roll_ref)
    pitch_err = np.rad2deg(pitch - pitch_ref)
    yaw_err = np.rad2deg(yaw - yaw_ref)
    pos_err = final.position_reference - final.position
    saturation_count = sum(1 for sample in log.samples if sample.saturated)
    max_torque = np.max(np.linalg.norm(log.field_array("torque_command"), axis=1))
    summary = (
        f"scenario={log.metadata.scenario} | mode={log.metadata.motion_mode} | samples={len(log.samples)} | "
        f"pos m=({final.position[0]:.2f}, {final.position[1]:.2f}, {final.position[2]:.2f}) | "
        f"pos err m=({pos_err[0]:.2f}, {pos_err[1]:.2f}, {pos_err[2]:.2f}) | "
        f"attitude deg=({np.rad2deg(roll):.2f}, {np.rad2deg(pitch):.2f}, {np.rad2deg(yaw):.2f}) | "
        f"att err deg=({roll_err:.2f}, {pitch_err:.2f}, {yaw_err:.2f}) | "
        f"max |tau_cmd|={max_torque:.3f} N*m | saturated_steps={saturation_count}"
    )
    landing = log.metadata.landing_success
    if landing:
        if log.metadata.burnout_time_s >= 0.0:
            burnout_pos = log.metadata.burnout_position_m or "?"
            speed = log.metadata.burnout_speed_m_s
            speed_txt = f"{speed:.2f}" if speed >= 0.0 else "?"
            summary += (
                f" | landing={'SUCCESS' if landing == 'true' else 'FAIL'} "
                f"@ burnout t={log.metadata.burnout_time_s:.2f}s "
                f"pos=({burnout_pos}) |v|={speed_txt} m/s"
            )
            caught = str(log.metadata.chopstick_caught or "")
            heading_ok = str(log.metadata.chopstick_heading_ok or "")
            if caught or heading_ok:
                summary += (
                    f" | chopsticks={'caught' if caught == 'true' else 'missed'}"
                    f" heading={'ok' if heading_ok == 'true' else 'bad'}"
                )
        else:
            summary += (
                f" | landing={'SUCCESS' if landing == 'true' else 'FAIL'} "
                f"(no engine burnout observed)"
            )
    return summary


def plot_flight_log(log: FlightLog, *, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    times = log.times()
    position = log.field_array("position")
    position_ref = log.field_array("position_reference")
    euler = np.rad2deg(
        np.array([quaternion_to_euler(q) for q in log.field_array("quaternion")])
    )
    euler_ref = np.rad2deg(
        np.array([quaternion_to_euler(q) for q in log.field_array("quaternion_reference")])
    )
    omega = np.rad2deg(log.field_array("omega"))
    torque_cmd = log.field_array("torque_command")
    torque_act = log.field_array("torque_achieved")
    force_inertial = log.field_array("force_inertial")

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    labels = ("x", "y", "z")
    for axis_index, label in enumerate(labels):
        axes[0].plot(times, position[:, axis_index], label=f"{label} actual")
        axes[0].plot(
            times,
            position_ref[:, axis_index],
            "--",
            label=f"{label} ref",
            alpha=0.8,
        )
    axes[0].set_ylabel("m")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", ncol=3, fontsize=8)

    attitude_labels = ("roll", "pitch", "yaw")
    for axis_index, label in enumerate(attitude_labels):
        axes[1].plot(times, euler[:, axis_index], label=f"{label} actual")
        axes[1].plot(
            times,
            euler_ref[:, axis_index],
            "--",
            label=f"{label} ref",
            alpha=0.8,
        )
    axes[1].set_ylabel("deg")
    axes[1].set_title(title or f"Flight test: {log.metadata.scenario}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", ncol=3, fontsize=8)

    for axis_index, label in enumerate(("p", "q", "r")):
        axes[2].plot(times, omega[:, axis_index], label=label)
    axes[2].set_ylabel("deg/s")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    for axis_index, label in enumerate(("Tx", "Ty", "Tz")):
        axes[3].plot(times, torque_cmd[:, axis_index], label=f"{label} cmd")
        axes[3].plot(times, torque_act[:, axis_index], "--", label=f"{label} act", alpha=0.8)
    axes[3].set_ylabel("N*m")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best", ncol=3, fontsize=8)

    for axis_index, label in enumerate(("Fx", "Fy", "Fz")):
        axes[4].plot(times, force_inertial[:, axis_index], label=label)
    axes[4].set_ylabel("N")
    axes[4].set_xlabel("time (s)")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="best")

    fig.tight_layout()
    plt.show()
