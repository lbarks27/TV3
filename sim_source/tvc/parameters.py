from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .controller import AltitudeForceConfig, AttitudePIDConfig, PositionPIDConfig
from .dynamics import VehicleProperties, compose_loaded_properties
from .geometry import CatchArm, Mount, build_ring_mounts, make_mount, unit
from .motors import MotorCurve, load_motor


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


@dataclass
class SolverConfig:
    """Mixer knobs for the collective + differential allocator.

    Legacy LM fields (``max_iters``, ``tracking_iters``, ``motion_damping``,
    ``reseed_interval``, ``refine_passes``, ``step_scale``) remain so existing
    vehicle JSON continues to load; the simple mixer ignores them.
    """

    # Active simple-mixer parameters
    damping: float = 0.05  # ridge scale: λ = damping² on a2 least-squares
    tol: float = 1e-5
    finite_diff_eps: float = 1e-4
    force_search_iters: int = 16  # binary-search steps for collective a1
    a2_polish_iters: int = 2  # extra a2 re-linearizations after first solve
    # Scale roll/yaw (τx/τz) residual/rows in the a2 LS solve. <1 deprioritizes
    # that axis vs pitch so force + pitch keep gimbal under saturation.
    roll_torque_weight: float = 1.0
    yaw_torque_weight: float = 1.0
    # Legacy / ignored (JSON + CLI compatibility)
    step_scale: float = 0.8
    max_iters: int = 48
    tracking_iters: int = 2
    motion_damping: float = 0.0
    reseed_interval: int = 0
    refine_passes: int = 2


@dataclass
class RenderConfig:
    force_weight: float = 0.02
    torque_limit: float | None = None
    force_scale: float | None = None
    force_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class Waypoint:
    """Position setpoint for hover / position missions.

    ``t`` is the switch time when ``waypoint_advance`` is ``\"timed\"`` or
    ``\"lerp\"``. In ``\"capture\"`` mode waypoints advance on proximity and
    ``t`` is ignored. ``\"lerp\"`` interpolates the setpoint between timed
    waypoints (moving reference).
    """

    position_m: tuple[float, float, float]
    t: float = 0.0


@dataclass
class LandingAcceptanceBox:
    """Axis-aligned world-frame box used as the solid-motor mission success gate.

    Success is evaluated once at engine burnout: if the vehicle position lies
    inside ``[center - half_extents, center + half_extents]`` the flight is a
    success. Optional ``max_speed_m_s`` also requires ``|v|`` at burnout to be
    at or below that limit.
    """

    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_extents_m: tuple[float, float, float] = (5.0, 5.0, 5.0)
    max_speed_m_s: float | None = None

    def contains_position(self, position: np.ndarray | list[float] | tuple[float, ...]) -> bool:
        p = np.asarray(position, dtype=float).reshape(3)
        center = np.asarray(self.center_m, dtype=float)
        half = np.asarray(self.half_extents_m, dtype=float)
        return bool(np.all(np.abs(p - center) <= half + 1e-9))

    def accepts(
        self,
        position: np.ndarray | list[float] | tuple[float, ...],
        velocity: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> bool:
        if not self.contains_position(position):
            return False
        if self.max_speed_m_s is None:
            return True
        if velocity is None:
            return False
        speed = float(np.linalg.norm(np.asarray(velocity, dtype=float).reshape(3)))
        return speed <= float(self.max_speed_m_s) + 1e-9

    def corners(self) -> np.ndarray:
        """Return the eight corner points as an (8, 3) array."""
        c = np.asarray(self.center_m, dtype=float)
        h = np.asarray(self.half_extents_m, dtype=float)
        signs = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        return c + signs * h

    def to_metadata_strings(self) -> dict[str, str]:
        cx, cy, cz = self.center_m
        hx, hy, hz = self.half_extents_m
        payload = {
            "landing_box_center_m": f"{cx},{cy},{cz}",
            "landing_box_half_extents_m": f"{hx},{hy},{hz}",
        }
        if self.max_speed_m_s is not None:
            payload["landing_box_max_speed_m_s"] = f"{float(self.max_speed_m_s)}"
        return payload

    @classmethod
    def from_metadata(cls, metadata: object) -> "LandingAcceptanceBox | None":
        center_raw = getattr(metadata, "landing_box_center_m", "") or ""
        half_raw = getattr(metadata, "landing_box_half_extents_m", "") or ""
        if not center_raw or not half_raw:
            return None
        try:
            center = tuple(float(part) for part in str(center_raw).split(","))
            half = tuple(float(part) for part in str(half_raw).split(","))
        except ValueError:
            return None
        if len(center) != 3 or len(half) != 3:
            return None
        max_speed_raw = getattr(metadata, "landing_box_max_speed_m_s", "") or ""
        max_speed = float(max_speed_raw) if str(max_speed_raw).strip() else None
        return cls(
            center_m=(center[0], center[1], center[2]),
            half_extents_m=(half[0], half[1], half[2]),
            max_speed_m_s=max_speed,
        )


@dataclass
class RealLandingMission:
    """Real-life pad approach mission geometry.

    Landing site is the pinned pad origin for success evaluation. Launch is a
    fixed offset from the pad (default 10 m east / +x).

    ``path_shape`` selects the reference:

    - ``\"capture\"`` (default): hover-capture the approach gate, then track an
      ease-out moving waypoint from that gate into the pad.
    - ``\"continuous\"``: one moving waypoint along a parabola through
      launch, the first waypoint (gate), and the pad. Time law is
      ``path_time_law`` (min-jerk rest-to-rest, ease-out, uniform arc speed,
      or projectile boost / −g coast / brake). Optional ``terminal_time_s``
      pins XY to the pad hover at that clock time; ``terminal_settle_s``
      then eases altitude to that hover. Gate time is
      ``forced_descent_time_s`` if set, else ``min_approach_time_s``.
    """

    landing_site_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Relative to landing site: default 10 m east (+x), on the ground plane.
    launch_offset_m: tuple[float, float, float] = (10.0, 0.0, 0.0)
    # First waypoint / approach gate relative to pad: default 5 m east, 5 m up.
    first_waypoint_offset_m: tuple[float, float, float] = (5.0, 0.0, 5.0)
    # Soft-pad hover height at end of descent (must stay above ground crash).
    touchdown_altitude_m: float = 0.5
    # Time for the ease-out moving track from the first waypoint into the pad
    # (path velocity falls to zero at the soft pad hover point).
    descent_duration_s: float = 7.0
    capture_radius_m: float = 2.0
    # Require low speed before leaving the approach gate for the descent track.
    capture_max_speed_m_s: float = 1.5
    # Capture conditions must hold this long before the descent track starts.
    capture_settle_s: float = 0.0
    # Do not capture the approach gate before this mission time.
    min_approach_time_s: float = 4.0
    # If set, force landing-mode start at this mission time even without
    # capture (needed for short-burn motors like F10).
    forced_descent_time_s: float | None = None
    # If set, transition to the final (pad) waypoint when fleet burn time
    # remaining falls to this many seconds (max over motors of
    # ignition_delay + curve duration). Overrides capture / progress-based
    # pad starts and replaces ``forced_descent_time_s`` with
    # ``burnout − this``.
    pad_transition_burn_remaining_s: float | None = None
    # XY distance from the staging waypoint at which the pad slide may start,
    # once the burn clock has opened. 0.55 waits until the vehicle has nearly
    # stopped on the gate. A wider disk starts the slide while it is still
    # inbound, which a soft roll axis needs in order to cover the last 2 m.
    stage_capture_radius_m: float = 0.55
    # During final descent, force a sink command if altitude exceeds the
    # first-waypoint height by more than this margin (meters).
    descent_altitude_margin_m: float = 0.4
    # Cap XY tilt during the soft-pad descent so vertical thrust is preserved
    # under marginal T/W. None keeps the normal position-controller limit.
    landing_max_tilt_deg: float | None = None
    # Inertial heading (yaw about +Z) held on approach, then slewed to
    # ``landing_yaw_deg``. Chopstick catch needs yaw ≈ ±90° so the body-X
    # catch arms lie across the world-X pad bars. With ``vertical_hold_s``,
    # the slew defaults to that first window unless a clock is set.
    approach_yaw_deg: float = 0.0
    landing_yaw_deg: float = 0.0
    # Seconds to slew approach → landing yaw. None uses descent_duration_s,
    # or the vertical-hold window when that is set and no clock is given.
    heading_slew_s: float | None = None
    heading_slew_start_s: float | None = None
    heading_slew_end_s: float | None = None
    # While not in the landing phase, add 180° to the heading command on
    # every other interval of this length. None disables the flips.
    heading_flip_period_s: float | None = None
    # When set, and no absolute slew clock is given, finish the yaw this many
    # seconds before burnout. A late pad slide starts the turn immediately.
    heading_finish_before_burnout_s: float | None = None
    # While the staging reference is still more than 2 m away, raise the
    # return speed cap and tilt budget. None keeps the profile limits for
    # the whole flight. The pad slide itself is not boosted.
    return_speed_m_s: float | None = None
    # While the vehicle is more than about a metre off the pad centerline,
    # or still carrying cross-track speed, open the north-south tilt and
    # speed caps and stop derating the downrange axis. None keeps the
    # profile Y law for the whole flight. Used for a leaned launch.
    cross_track_tilt_deg: float | None = None
    cross_track_speed_m_s: float | None = None
    return_tilt_deg: float | None = None
    # Position-error clip used with the fast return. None keeps 8 m, which
    # is enough for the 2.3 m/s cap. A longer launch needs a wider clip or
    # the speed cap never gets commanded.
    return_position_error_m: float | None = None
    # Speed to be down to when the fast return hands off, 2 m east of the
    # stage. None holds return_speed_m_s the whole way. Set on a long
    # launch so the cruise bleeds off on the tilt budget before the slide.
    return_brake_speed_m_s: float | None = None
    # Extra tilt used to stop short of the stick backstop. While the arms
    # still reach west, the chase is held east of that stop and this tilt
    # is what brakes a hot arrival. None keeps the profile tilt cap.
    overshoot_tilt_deg: float | None = None
    # Pull the continuous-path end this many metres short of the pad hover,
    # along the launch→pad ground track. 0 ends on the pad.
    path_end_short_m: float = 0.0
    # After the parabola finishes, ease XY from that short end into the pad
    # at path-end altitude. Rest-to-rest with an ease-out so the waypoint
    # arrives stopped. 0 leaves the reference at the path end unless
    # ``slide_to_pad_z`` is set, in which case the pad waypoint snaps on.
    lateral_slide_s: float = 0.0
    # If true, after the parabola the pad waypoint (XY + ``landing_end_m`` z)
    # is the reference. A positive ``lateral_slide_s`` eases there; 0 snaps.
    slide_to_pad_z: bool = False
    # Geometric parabola end altitude (the bottom of the loft). 0 uses
    # ``landing_end_m`` z. The pad waypoint / box center can differ.
    path_end_altitude_m: float = 0.0
    # If > 0, do not command below this altitude until burnout (short burns
    # that cannot hover at the parabola bottom). 0 tracks the path end.
    path_hold_altitude_m: float = 0.0
    # "capture" = stop-at-gate then ease-out; "continuous" = parabolic loft.
    path_shape: str = "capture"
    # Continuous path time law: "min_jerk" (rest at both ends), "ease_out"
    # (peak speed at launch), "loft_then_creep" (spike to the gate, then a
    # slow backside), "loft_through" (same split, but the waypoint keeps
    # moving through the gate), "uniform" (constant speed along the parabola),
    # or "projectile" (boost off the rail, coast with z̈ ≈ −g to apogee, then
    # min-jerk brake to the pad).
    path_time_law: str = "min_jerk"
    # loft_then_creep / loft_through: seconds to reach the gate / peak.
    loft_duration_s: float = 2.0
    # Seconds to hover at the loft waypoint (waypoint 0 / launch + offset)
    # after arriving inside capture_radius_m. 0 disables. Without
    # ``vertical_hold_offset_m`` this is a clock from fire (legacy).
    vertical_hold_s: float = 0.0
    # Optional loft setpoint during ``vertical_hold_s``, relative to launch.
    # e.g. ``[0, 0, 10]`` → climb toward launch + 10 m Z. None → no waypoint
    # (open-loop full axial thrust, legacy behavior).
    vertical_hold_offset_m: tuple[float, float, float] | None = None
    # After a ``terminal_time_s`` latch, ease z from the latch altitude down
    # to the pad hover (``landing_end_m``) over this many seconds. 0 snaps.
    # Never eases to z=0 — pad contact before burnout is a fail.
    terminal_settle_s: float = 0.0
    # Continuous path only: latch the pad waypoint at this mission time (s).
    # 0 disables; the reference stays on the parabola.
    terminal_time_s: float = 0.0

    def launch_position_m(self) -> tuple[float, float, float]:
        sx, sy, sz = self.landing_site_m
        ox, oy, oz = self.launch_offset_m
        # Launch is a ground offset; do not inherit a raised pad hover altitude.
        return (sx + ox, sy + oy, max(0.0, float(oz)))

    def first_waypoint_m(self) -> tuple[float, float, float]:
        sx, sy, sz = self.landing_site_m
        ox, oy, oz = self.first_waypoint_offset_m
        return (sx + ox, sy + oy, sz + oz)

    def landing_end_m(self) -> tuple[float, float, float]:
        """Pad hover waypoint. Profiles with a landing box pin this to the box center.

        Touchdown is an absolute z and may sit below the landing-site / ground
        plane (catch profiles aim through the sticks).
        """
        sx, sy, sz = self.landing_site_m
        z = float(self.touchdown_altitude_m)
        if z == 0.0:
            z = float(sz)
        return (sx, sy, z)

    # Backward-compatible aliases used by older call sites / notes.
    def overhead_target_m(self) -> tuple[float, float, float]:
        return self.first_waypoint_m()

    def approach_start_m(self) -> tuple[float, float, float]:
        return self.first_waypoint_m()


@dataclass
class EngineVariance:
    """Independent per-motor catalog scatter.

    ``thrust`` / ``burn_time`` are ± fractions. ``ignition_delay_s`` is the max
    additive hang-fire delay drawn uniformly in ``[0, ignition_delay_s]`` per motor.
    """

    thrust: float = 0.0
    burn_time: float = 0.0
    ignition_delay_s: float = 0.0
    seed: int | None = None
    # (thrust_scale, time_scale, ignition_delay_s) per mount
    draws: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class WindConfig:
    """Inertial-frame mean wind + optional Gauss-Markov gusts with quadratic drag.

    Drag force is ``F = -0.5 ρ Cd A |v − v_wind| (v − v_wind)``. Gusts are an
    isotropic Ornstein–Uhlenbeck process with steady-state std ``gust_std_m_s``.
    """

    velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gust_std_m_s: float = 0.0
    gust_tau_s: float = 2.0
    drag_coeff: float = 1.0
    ref_area_m2: float = 0.12
    air_density_kg_m3: float = 1.225
    seed: int | None = None

    def enabled(self) -> bool:
        mean = np.linalg.norm(np.asarray(self.velocity_m_s, dtype=float))
        return (
            mean > 1e-9
            or float(self.gust_std_m_s) > 1e-9
        ) and float(self.drag_coeff) > 0.0 and float(self.ref_area_m2) > 0.0


@dataclass
class FlightProfile:
    name: str = ""
    motion: str = "attitude"
    scenario: str = "hold"
    target_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    waypoints: list[Waypoint] = field(default_factory=list)
    # "timed" switches on waypoint.t; "capture" advances when within radius;
    # "lerp" linearly interpolates between timed waypoints.
    waypoint_advance: str = "timed"
    waypoint_capture_radius_m: float = 2.0
    capture_settle_s: float = 0.0
    capture_max_speed_m_s: float = 1.5
    notes: str = ""
    altitude_controller: AltitudeForceConfig = field(default_factory=AltitudeForceConfig)
    landing_acceptance: LandingAcceptanceBox | None = None
    real_landing: RealLandingMission | None = None
    engine_variance: EngineVariance | None = None
    wind: WindConfig | None = None


@dataclass
class SystemConfig:
    mounts: list[Mount]
    solver: SolverConfig
    render: RenderConfig
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vehicle: VehicleProperties = field(default_factory=VehicleProperties)
    controller: AttitudePIDConfig = field(default_factory=AttitudePIDConfig)
    flight_test: "FlightTestConfig | None" = None
    profile: FlightProfile | None = None
    catch_arms: list[CatchArm] = field(default_factory=list)
    # Unloaded airframe (weigh-in) before motor point-mass compose. Allocator uses
    # ``center_of_mass`` (motors at gimbal pivots). The sim plant uses offset CGs.
    unloaded_center_of_mass: tuple[float, float, float] = (0.0, 0.0, 1.0)
    unloaded_vehicle: VehicleProperties = field(default_factory=VehicleProperties)
    motor_pivot_standoff_m: float = 0.055
    # Body members drawn in place of catch arms. Same root/direction/length schema.
    landing_legs: list[CatchArm] = field(default_factory=list)
    mast_top_m: float | None = None


@dataclass
class FlightTestConfig:
    dt: float = 0.01
    duration: float = 5.0
    force_target: float = 0.0
    force_weight: float = 0.0
    axial_mix_arm_s: float | None = None
    # ΣT / weight gate for XY tilt (and axial mix when arm_s is unset).
    # Default 1.15 ≈ cos(30°) headroom; lower for marginal solid-motor T/W.
    hover_thrust_margin: float = 1.15
    thrust_per_mount: float | None = None
    # Per-axis gimbal rate limit. None / <=0 disables (instant actuators).
    actuator_slew_rate_deg_s: float | None = None
    initial_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    notes: str = ""
    vehicle: VehicleProperties = field(default_factory=VehicleProperties)
    controller: AttitudePIDConfig = field(default_factory=AttitudePIDConfig)
    position_controller: PositionPIDConfig = field(default_factory=PositionPIDConfig)
    altitude_controller: AltitudeForceConfig = field(default_factory=AltitudeForceConfig)

def _vector3(data: list[float] | tuple[float, float, float], default: tuple[float, float, float]) -> tuple[float, float, float]:
    if data is None:
        return default
    values = tuple(float(value) for value in data)
    if len(values) != 3:
        raise ValueError("Expected a length-3 vector.")
    return values


def _build_vehicle(data: dict | None) -> VehicleProperties:
    """Build unloaded vehicle properties from JSON.

    Mass may be given as ``mass`` (kg) or ``mass_g`` (grams). Inertia may be a
    full ``inertia`` matrix, ``inertia_diagonal`` in kg·m², or
    ``inertia_diagonal_g_m2`` (g·m², divided by 1000). Values are treated as
    unloaded / dry-airframe about the unloaded CoM; motor point masses are
    composed later in ``_system_from_payload``.
    """
    if not data:
        return VehicleProperties()
    if "mass_g" in data:
        mass = float(data["mass_g"]) / 1000.0
    else:
        mass = float(data.get("mass", 1.0))

    inertia = data.get("inertia")
    if inertia is not None:
        inertia_matrix = np.asarray(inertia, dtype=float)
    elif "inertia_diagonal_g_m2" in data:
        diagonal = _vector3(
            data["inertia_diagonal_g_m2"],
            (50.0, 50.0, 80.0),
        )
        inertia_matrix = np.diag(tuple(v / 1000.0 for v in diagonal))
    else:
        diagonal = _vector3(
            data.get("inertia_diagonal", [0.05, 0.05, 0.08]),
            (0.05, 0.05, 0.08),
        )
        inertia_matrix = np.diag(diagonal)
    return VehicleProperties(
        inertia=inertia_matrix,
        mass=mass,
        gravity=np.asarray(
            data.get("gravity", [0.0, 0.0, -9.81]),
            dtype=float,
        ),
    )


def _build_catch_arms(data: list | None) -> list[CatchArm]:
    if not data:
        return []
    arms: list[CatchArm] = []
    for entry in data:
        position = np.asarray(entry["position"], dtype=float).reshape(3)
        direction = unit(np.asarray(entry.get("direction", [1.0, 0.0, 0.0]), dtype=float))
        length_m = float(entry.get("length_m", 0.5))
        if length_m <= 0.0:
            raise ValueError("catch arm length_m must be positive")
        arms.append(CatchArm(position=position, direction=direction, length_m=length_m))
    return arms


def _mount_point_mass_kg(mount: Mount) -> float:
    if mount.motor is not None and float(mount.motor.initial_mass_kg) > 0.0:
        return float(mount.motor.initial_mass_kg)
    return float(mount.mass_kg)


def _compose_vehicle_with_motors(
    unloaded: VehicleProperties,
    unloaded_com: tuple[float, float, float],
    mounts: list[Mount],
) -> tuple[VehicleProperties, tuple[float, float, float]]:
    point_masses = [
        (_mount_point_mass_kg(mount), mount.position)
        for mount in mounts
        if _mount_point_mass_kg(mount) > 0.0
    ]
    if not point_masses:
        return unloaded, unloaded_com
    mass, com, inertia = compose_loaded_properties(
        unloaded.mass,
        unloaded_com,
        unloaded.inertia,
        point_masses,
    )
    loaded = VehicleProperties(
        inertia=inertia,
        mass=mass,
        gravity=unloaded.gravity.copy(),
    )
    return loaded, (float(com[0]), float(com[1]), float(com[2]))


def _build_position_controller(data: dict | None) -> PositionPIDConfig:
    if not data:
        return PositionPIDConfig()
    defaults = PositionPIDConfig()
    return PositionPIDConfig(
        kp=np.asarray(data.get("kp", defaults.kp), dtype=float),
        kd=np.asarray(data.get("kd", defaults.kd), dtype=float),
        ki=np.asarray(data.get("ki", defaults.ki), dtype=float),
        max_tilt_deg=float(data.get("max_tilt_deg", defaults.max_tilt_deg)),
        min_upward_accel=float(data.get("min_upward_accel", defaults.min_upward_accel)),
        velocity_align_kp=float(
            data.get("velocity_align_kp", defaults.velocity_align_kp)
        ),
        velocity_aim_min_speed=float(
            data.get("velocity_aim_min_speed", defaults.velocity_aim_min_speed)
        ),
        integral_limit=(
            None
            if data.get("integral_limit", defaults.integral_limit) is None
            else float(data.get("integral_limit", defaults.integral_limit))
        ),
        velocity_cascade=bool(
            data.get("velocity_cascade", defaults.velocity_cascade)
        ),
        max_position_error_m=float(
            data.get("max_position_error_m", defaults.max_position_error_m)
        ),
        max_position_error_y_m=float(
            data.get("max_position_error_y_m", defaults.max_position_error_y_m)
        ),
        max_speed_m_s=(
            tuple(float(v) for v in data["max_speed_m_s"])
            if isinstance(data.get("max_speed_m_s"), (list, tuple))
            else float(data.get("max_speed_m_s", defaults.max_speed_m_s))
        ),
        integral_start_m=float(
            data.get("integral_start_m", defaults.integral_start_m)
        ),
        max_tilt_y_deg=(
            None
            if data.get("max_tilt_y_deg", defaults.max_tilt_y_deg) is None
            else float(data.get("max_tilt_y_deg", defaults.max_tilt_y_deg))
        ),
        lateral_accel_lpf_hz=float(
            data.get("lateral_accel_lpf_hz", defaults.lateral_accel_lpf_hz)
        ),
        lateral_priority_m=float(
            data.get("lateral_priority_m", defaults.lateral_priority_m)
        ),
        prefer_lateral_y=bool(
            data.get("prefer_lateral_y", defaults.prefer_lateral_y)
        ),
    )


def _build_controller(data: dict | None) -> AttitudePIDConfig:
    if not data:
        return AttitudePIDConfig()
    defaults = AttitudePIDConfig()
    return AttitudePIDConfig(
        kp=np.asarray(data.get("kp", defaults.kp), dtype=float),
        ki=np.asarray(data.get("ki", defaults.ki), dtype=float),
        kd=np.asarray(data.get("kd", defaults.kd), dtype=float),
        torque_limit=data.get("torque_limit", defaults.torque_limit),
        integral_limit=data.get("integral_limit", defaults.integral_limit),
    )


def _build_altitude_controller(data: dict | None) -> AltitudeForceConfig:
    if not data:
        return AltitudeForceConfig()
    defaults = AltitudeForceConfig()
    return AltitudeForceConfig(
        kp=float(data.get("kp", defaults.kp)),
        ki=float(data.get("ki", defaults.ki)),
        kd=float(data.get("kd", defaults.kd)),
        integral_limit=data.get("integral_limit", defaults.integral_limit),
        force_min=float(data.get("force_min", defaults.force_min)),
        force_max=data.get("force_max", defaults.force_max),
        throttle_min=float(data.get("throttle_min", defaults.throttle_min)),
        hover_force=data.get("hover_force", defaults.hover_force),
    )


def _build_flight_test(
    data: dict | None,
    vehicle: VehicleProperties,
    controller: AttitudePIDConfig,
    position_controller: PositionPIDConfig,
    altitude_controller: AltitudeForceConfig | None = None,
) -> FlightTestConfig | None:
    if not data:
        return None
    slew = data.get("actuator_slew_rate_deg_s")
    return FlightTestConfig(
        dt=float(data.get("dt", 0.01)),
        duration=float(data.get("duration", 5.0)),
        force_target=float(data.get("force_target", 0.0)),
        force_weight=float(data.get("force_weight", 0.0)),
        axial_mix_arm_s=(
            float(data["axial_mix_arm_s"]) if data.get("axial_mix_arm_s") is not None else None
        ),
        hover_thrust_margin=float(
            data.get("hover_thrust_margin", FlightTestConfig.hover_thrust_margin)
        ),
        thrust_per_mount=data.get("thrust_per_mount"),
        actuator_slew_rate_deg_s=float(slew) if slew is not None else None,
        initial_euler_deg=_vector3(data.get("initial_euler_deg", [0.0, 0.0, 0.0]), (0.0, 0.0, 0.0)),
        initial_position_m=_vector3(data.get("initial_position_m", [0.0, 0.0, 0.0]), (0.0, 0.0, 0.0)),
        notes=str(data.get("notes", "")),
        vehicle=_build_vehicle(data.get("vehicle")) if data.get("vehicle") else vehicle,
        controller=_build_controller(data.get("controller")) if data.get("controller") else controller,
        position_controller=(
            _build_position_controller(data.get("position_controller"))
            if data.get("position_controller")
            else position_controller
        ),
        altitude_controller=(
            _build_altitude_controller(data.get("altitude_controller"))
            if data.get("altitude_controller")
            else (altitude_controller or AltitudeForceConfig())
        ),
    )


def _build_waypoints(data: list | None, *, sort_by_time: bool = True) -> list[Waypoint]:
    if not data:
        return []
    waypoints: list[Waypoint] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Waypoint {index} must be an object with position_m.")
        position = item.get("position_m", item.get("position"))
        if position is None:
            raise ValueError(f"Waypoint {index} is missing position_m.")
        waypoints.append(
            Waypoint(
                position_m=_vector3(position, (0.0, 0.0, 0.0)),
                t=float(item.get("t", item.get("time_s", 0.0))),
            )
        )
    if sort_by_time:
        waypoints.sort(key=lambda waypoint: waypoint.t)
    return waypoints


def _build_real_landing(data: dict | None) -> RealLandingMission | None:
    if not data:
        return None
    if not isinstance(data, dict):
        raise ValueError("real_landing must be an object.")
    # Prefer explicit first_waypoint_offset; fall back to legacy approach/overhead keys.
    if "first_waypoint_offset_m" in data or "first_waypoint_offset" in data:
        first_offset = _vector3(
            data.get("first_waypoint_offset_m", data.get("first_waypoint_offset")),
            (5.0, 0.0, 5.0),
        )
    elif "approach_start_offset_m" in data or "approach_start_offset" in data:
        first_offset = _vector3(
            data.get("approach_start_offset_m", data.get("approach_start_offset")),
            (5.0, 0.0, 5.0),
        )
    elif "overhead_altitude_m" in data or "overhead_altitude" in data:
        alt = float(data.get("overhead_altitude_m", data.get("overhead_altitude", 5.0)))
        first_offset = (0.0, 0.0, alt)
    else:
        first_offset = (5.0, 0.0, 5.0)

    capture_radius = data.get(
        "capture_radius_m",
        data.get(
            "capture_radius",
            data.get("overhead_capture_radius_m", data.get("overhead_capture_radius", 2.0)),
        ),
    )
    capture_max_speed = data.get(
        "capture_max_speed_m_s",
        data.get(
            "capture_max_speed",
            data.get(
                "overhead_capture_max_speed_m_s",
                data.get("overhead_capture_max_speed", 1.5),
            ),
        ),
    )
    path_shape = str(
        data.get("path_shape", data.get("path", "capture"))
    ).strip().lower() or "capture"
    if path_shape not in {"capture", "continuous"}:
        raise ValueError(
            f"Unsupported path_shape={path_shape!r}; expected 'capture' or 'continuous'."
        )
    path_time_law = str(
        data.get("path_time_law", data.get("time_law", "min_jerk"))
    ).strip().lower() or "min_jerk"
    if path_time_law not in {
        "min_jerk",
        "ease_out",
        "loft_then_creep",
        "loft_through",
        "uniform",
        "projectile",
    }:
        raise ValueError(
            f"Unsupported path_time_law={path_time_law!r}; "
            "expected 'min_jerk', 'ease_out', 'loft_then_creep', 'loft_through', "
            "'uniform', or 'projectile'."
        )
    return RealLandingMission(
        landing_site_m=_vector3(
            data.get("landing_site_m", data.get("landing_site")),
            (0.0, 0.0, 0.0),
        ),
        launch_offset_m=_vector3(
            data.get("launch_offset_m", data.get("launch_offset", [10.0, 0.0, 0.0])),
            (10.0, 0.0, 0.0),
        ),
        first_waypoint_offset_m=first_offset,
        touchdown_altitude_m=float(
            data.get("touchdown_altitude_m", data.get("touchdown_altitude", 0.5))
        ),
        descent_duration_s=float(
            data.get("descent_duration_s", data.get("descent_duration", 7.0))
        ),
        capture_radius_m=float(capture_radius),
        capture_max_speed_m_s=float(capture_max_speed),
        capture_settle_s=float(
            data.get("capture_settle_s", data.get("capture_settle", 0.0))
        ),
        min_approach_time_s=float(
            data.get("min_approach_time_s", data.get("min_approach_time", 4.0))
        ),
        forced_descent_time_s=(
            float(data.get("forced_descent_time_s", data.get("forced_descent_time")))
            if data.get("forced_descent_time_s", data.get("forced_descent_time")) is not None
            else None
        ),
        pad_transition_burn_remaining_s=(
            float(
                data.get(
                    "pad_transition_burn_remaining_s",
                    data.get("pad_transition_burn_remaining"),
                )
            )
            if data.get(
                "pad_transition_burn_remaining_s",
                data.get("pad_transition_burn_remaining"),
            )
            is not None
            else None
        ),
        stage_capture_radius_m=float(
            data.get("stage_capture_radius_m", data.get("stage_capture_radius", 0.55))
        ),
        descent_altitude_margin_m=float(
            data.get(
                "descent_altitude_margin_m",
                data.get("descent_altitude_margin", 0.4),
            )
        ),
        landing_max_tilt_deg=(
            float(data["landing_max_tilt_deg"])
            if data.get("landing_max_tilt_deg") is not None
            else None
        ),
        approach_yaw_deg=float(
            data.get("approach_yaw_deg", data.get("approach_yaw", 0.0))
        ),
        landing_yaw_deg=float(
            data.get("landing_yaw_deg", data.get("landing_yaw", 0.0))
        ),
        heading_slew_s=(
            None
            if data.get("heading_slew_s", data.get("heading_slew")) is None
            else float(data.get("heading_slew_s", data.get("heading_slew")))
        ),
        heading_slew_start_s=(
            None
            if data.get("heading_slew_start_s") is None
            else float(data["heading_slew_start_s"])
        ),
        heading_slew_end_s=(
            None
            if data.get("heading_slew_end_s") is None
            else float(data["heading_slew_end_s"])
        ),
        heading_flip_period_s=(
            None
            if data.get("heading_flip_period_s") is None
            else float(data["heading_flip_period_s"])
        ),
        heading_finish_before_burnout_s=(
            None
            if data.get("heading_finish_before_burnout_s") is None
            else float(data["heading_finish_before_burnout_s"])
        ),
        return_speed_m_s=(
            None
            if data.get("return_speed_m_s") is None
            else float(data["return_speed_m_s"])
        ),
        cross_track_tilt_deg=(
            None
            if data.get("cross_track_tilt_deg") is None
            else float(data["cross_track_tilt_deg"])
        ),
        cross_track_speed_m_s=(
            None
            if data.get("cross_track_speed_m_s") is None
            else float(data["cross_track_speed_m_s"])
        ),
        return_tilt_deg=(
            None
            if data.get("return_tilt_deg") is None
            else float(data["return_tilt_deg"])
        ),
        return_position_error_m=(
            None
            if data.get("return_position_error_m") is None
            else float(data["return_position_error_m"])
        ),
        return_brake_speed_m_s=(
            None
            if data.get("return_brake_speed_m_s") is None
            else float(data["return_brake_speed_m_s"])
        ),
        overshoot_tilt_deg=(
            None
            if data.get("overshoot_tilt_deg") is None
            else float(data["overshoot_tilt_deg"])
        ),
        path_end_short_m=float(data.get("path_end_short_m", 0.0)),
        lateral_slide_s=float(
            data.get("lateral_slide_s", data.get("lateral_slide", 0.0))
        ),
        slide_to_pad_z=bool(data.get("slide_to_pad_z", False)),
        path_end_altitude_m=float(data.get("path_end_altitude_m", 0.0)),
        path_hold_altitude_m=float(data.get("path_hold_altitude_m", 0.0)),
        path_shape=path_shape,
        path_time_law=path_time_law,
        loft_duration_s=float(
            data.get("loft_duration_s", data.get("loft_duration", 2.0))
        ),
        vertical_hold_s=float(
            data.get("vertical_hold_s", data.get("vertical_hold", 0.0))
        ),
        vertical_hold_offset_m=(
            None
            if (
                data.get("vertical_hold_offset_m", data.get("vertical_hold_offset"))
                is None
            )
            else _vector3(
                data.get(
                    "vertical_hold_offset_m",
                    data.get("vertical_hold_offset"),
                ),
                (0.0, 0.0, 0.0),
            )
        ),
        terminal_settle_s=float(
            data.get("terminal_settle_s", data.get("terminal_settle", 0.0))
        ),
        terminal_time_s=float(
            data.get("terminal_time_s", data.get("terminal_time", 0.0))
        ),
    )


def _build_engine_variance(data: dict | None) -> EngineVariance | None:
    if not data or not isinstance(data, dict):
        return None
    thrust = float(data.get("thrust", data.get("thrust_fraction", 0.0)))
    burn = float(data.get("burn_time", data.get("burn_time_fraction", 0.0)))
    ign = float(
        data.get(
            "ignition_delay_s",
            data.get("ignition_delay", data.get("ignition_time_s", 0.0)),
        )
    )
    if thrust <= 0.0 and burn <= 0.0 and ign <= 0.0:
        return None
    seed_raw = data.get("seed")
    seed = None if seed_raw is None else int(seed_raw)
    return EngineVariance(
        thrust=thrust, burn_time=burn, ignition_delay_s=ign, seed=seed
    )


def _build_wind(data: dict | None) -> WindConfig | None:
    if not data or not isinstance(data, dict):
        return None
    velocity = _vector3(
        data.get("velocity_m_s", data.get("velocity", data.get("mean_m_s"))),
        (0.0, 0.0, 0.0),
    )
    gust_std = float(data.get("gust_std_m_s", data.get("gust_std", 0.0)))
    gust_tau = float(data.get("gust_tau_s", data.get("gust_tau", 2.0)))
    drag_coeff = float(data.get("drag_coeff", data.get("cd", 1.0)))
    ref_area = float(
        data.get("ref_area_m2", data.get("ref_area", data.get("area_m2", 0.12)))
    )
    density = float(
        data.get(
            "air_density_kg_m3",
            data.get("air_density", data.get("rho", 1.225)),
        )
    )
    seed_raw = data.get("seed")
    seed = None if seed_raw is None else int(seed_raw)
    wind = WindConfig(
        velocity_m_s=velocity,
        gust_std_m_s=gust_std,
        gust_tau_s=gust_tau,
        drag_coeff=drag_coeff,
        ref_area_m2=ref_area,
        air_density_kg_m3=density,
        seed=seed,
    )
    return wind if wind.enabled() else None


def apply_engine_variance(
    mounts: list[Mount],
    variance: EngineVariance,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[list[Mount], list[tuple[float, float, float]]]:
    """Draw independent thrust/burn scales and hang-fire delays per curve motor."""
    generator = np.random.default_rng(variance.seed) if rng is None else rng
    updated: list[Mount] = []
    draws: list[tuple[float, float, float]] = []
    for mount in mounts:
        if mount.motor is None:
            updated.append(mount)
            draws.append((1.0, 1.0, 0.0))
            continue
        thrust_frac = max(float(variance.thrust), 0.0)
        burn_frac = max(float(variance.burn_time), 0.0)
        ign_max = max(float(variance.ignition_delay_s), 0.0)
        thrust_scale = 1.0 + float(generator.uniform(-thrust_frac, thrust_frac))
        time_scale = 1.0 + float(generator.uniform(-burn_frac, burn_frac))
        ign_delay = float(generator.uniform(0.0, ign_max)) if ign_max > 0.0 else 0.0
        motor = mount.motor.with_performance(
            thrust_scale=thrust_scale, time_scale=time_scale
        )
        if ign_delay > 0.0:
            motor = motor.with_ignition_delay(
                float(motor.ignition_delay_s) + ign_delay
            )
        updated.append(replace(mount, motor=motor, thrust=float(motor.max_thrust_n)))
        draws.append((thrust_scale, time_scale, ign_delay))
    return updated, draws


def _build_landing_acceptance(data: dict | None) -> LandingAcceptanceBox | None:
    if not data:
        return None
    if not isinstance(data, dict):
        raise ValueError("landing_acceptance must be an object.")

    if "min_m" in data or "max_m" in data:
        mins = _vector3(data.get("min_m"), (0.0, 0.0, 0.0))
        maxs = _vector3(data.get("max_m"), (0.0, 0.0, 0.0))
        center = tuple(0.5 * (np.asarray(mins) + np.asarray(maxs)))
        half = tuple(0.5 * np.abs(np.asarray(maxs) - np.asarray(mins)))
        center_m = (float(center[0]), float(center[1]), float(center[2]))
        half_extents_m = (float(half[0]), float(half[1]), float(half[2]))
    else:
        center_m = _vector3(data.get("center_m", data.get("center")), (0.0, 0.0, 0.0))
        if "half_extents_m" in data or "half_extents" in data:
            half_extents_m = _vector3(
                data.get("half_extents_m", data.get("half_extents")),
                (5.0, 5.0, 5.0),
            )
        elif "size_m" in data or "size" in data:
            size = _vector3(data.get("size_m", data.get("size")), (10.0, 10.0, 10.0))
            half_extents_m = (0.5 * size[0], 0.5 * size[1], 0.5 * size[2])
        else:
            half_extents_m = (5.0, 5.0, 5.0)

    max_speed = data.get("max_speed_m_s", data.get("max_speed"))
    return LandingAcceptanceBox(
        center_m=center_m,
        half_extents_m=half_extents_m,
        max_speed_m_s=float(max_speed) if max_speed is not None else None,
    )


def _build_profile(data: dict | None) -> FlightProfile | None:
    if not data:
        return None
    real_landing = _build_real_landing(data.get("real_landing"))
    # Also accept mission: { "type": "real_landing", ... } blocks.
    if real_landing is None and isinstance(data.get("mission"), dict):
        mission_block = data["mission"]
        mission_type = str(mission_block.get("type", "real_landing")).strip().lower()
        if mission_type == "real_landing":
            real_landing = _build_real_landing(mission_block)

    advance = str(data.get("waypoint_advance", "timed")).strip().lower()
    if advance not in {"timed", "capture", "lerp"}:
        raise ValueError(
            f"Unsupported waypoint_advance={advance!r}; "
            "expected 'timed', 'capture', or 'lerp'."
        )
    waypoints = _build_waypoints(
        data.get("waypoints"),
        sort_by_time=(advance in {"timed", "lerp"}),
    )
    if "target_position_m" in data:
        target_position_m = _vector3(data.get("target_position_m"), (0.0, 0.0, 0.0))
    elif real_landing is not None:
        target_position_m = real_landing.landing_site_m
    elif waypoints:
        target_position_m = waypoints[-1].position_m
    else:
        target_position_m = (0.0, 0.0, 0.0)
    landing_acceptance = _build_landing_acceptance(data.get("landing_acceptance"))
    if landing_acceptance is not None and real_landing is not None:
        cx, cy, _cz = (float(v) for v in landing_acceptance.center_m)
        _sx, _sy, sz = real_landing.landing_site_m
        real_landing.landing_site_m = (cx, cy, sz)
        real_landing.touchdown_altitude_m = float(landing_acceptance.center_m[2])
    return FlightProfile(
        name=str(data.get("name", "")),
        motion=str(data.get("motion", "attitude")),
        scenario=str(data.get("scenario", "hold")),
        target_position_m=target_position_m,
        waypoints=waypoints,
        waypoint_advance=advance,
        waypoint_capture_radius_m=float(data.get("waypoint_capture_radius_m", 2.0)),
        capture_settle_s=float(
            data.get(
                "capture_settle_s",
                data.get("capture_settle", 0.0),
            )
        ),
        capture_max_speed_m_s=float(
            data.get(
                "capture_max_speed_m_s",
                data.get(
                    "overhead_capture_max_speed_m_s",
                    data.get("overhead_capture_max_speed", 1.5),
                ),
            )
        ),
        notes=str(data.get("notes", "")),
        altitude_controller=_build_altitude_controller(data.get("altitude_controller")),
        landing_acceptance=landing_acceptance,
        real_landing=real_landing,
        engine_variance=_build_engine_variance(data.get("engine_variance")),
        wind=_build_wind(data.get("wind")),
    )


def _resolve_config_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    search_roots = []
    if base is not None:
        search_roots.append(base)
    search_roots.extend([Path.cwd(), CONFIG_ROOT, CONFIG_ROOT.parent])
    for root in search_roots:
        nested = root / path
        if nested.is_file():
            return nested
        under_config = root / "config" / path
        if under_config.is_file():
            return under_config
    raise FileNotFoundError(f"Config file not found: {path}")


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _payload_from_path(path: str | Path, *, base: Path | None = None) -> tuple[dict, Path]:
    resolved = _resolve_config_path(path, base=base)
    return json.loads(resolved.read_text()), resolved.parent


def load_config(path: str | Path) -> SystemConfig:
    payload, base = _payload_from_path(path)
    if "mounts" not in payload and payload.get("vehicle"):
        return load_vehicle_profile(vehicle_path=payload["vehicle"], profile_path=path)

    profile = None
    if "mounts" in payload and (
        "motion" in payload or "target_position_m" in payload or "altitude_controller" in payload
    ):
        profile = _build_profile(
            {
                "name": payload.get("name", ""),
                "motion": payload.get("motion", "attitude"),
                "scenario": payload.get("scenario", "hold"),
                "target_position_m": payload.get("target_position_m", [0.0, 0.0, 0.0]),
                "notes": payload.get("notes", ""),
                "altitude_controller": payload.get("altitude_controller"),
            }
        )
    return _system_from_payload(payload, profile=profile)


def load_vehicle_profile(
    vehicle_path: str | Path | None = None,
    profile_path: str | Path | None = None,
) -> SystemConfig:
    """Load a vehicle config and optional flight profile, merging profile overrides."""
    if vehicle_path is None and profile_path is None:
        raise ValueError("Provide a vehicle and/or profile path.")

    profile_payload: dict = {}
    profile_base: Path | None = None
    if profile_path is not None:
        profile_payload, profile_base = _payload_from_path(profile_path)

    resolved_vehicle = vehicle_path
    if resolved_vehicle is None:
        resolved_vehicle = profile_payload.get("vehicle")
        if not resolved_vehicle:
            raise ValueError(
                f"Profile {profile_path} does not reference a vehicle and no --vehicle was given."
            )

    try:
        vehicle_payload, _ = _payload_from_path(resolved_vehicle, base=profile_base)
    except FileNotFoundError:
        vehicle_payload, _ = _payload_from_path(resolved_vehicle, base=CONFIG_ROOT)

    # Profile flight_test / controllers overlay the vehicle defaults.
    overlay = {
        key: value
        for key, value in profile_payload.items()
        if key
        in {
            "flight_test",
            "controller",
            "position_controller",
            "altitude_controller",
            "render",
            "solver",
            "notes",
        }
    }
    payload = _deep_merge(vehicle_payload, overlay)
    if "notes" in profile_payload and "flight_test" in payload:
        flight_block = dict(payload.get("flight_test") or {})
        flight_block.setdefault("notes", profile_payload["notes"])
        payload["flight_test"] = flight_block

    profile = _build_profile(profile_payload) if profile_payload else None
    return _system_from_payload(payload, profile=profile)


def build_default_config(
    count: int,
    radius: float,
    thrust: float,
    max_angle_deg: float | None = None,
) -> SystemConfig:
    mount_kwargs: dict = {
        "count": count,
        "radius": radius,
        "thrust": thrust,
    }
    if max_angle_deg is not None:
        mount_kwargs["max_angle_deg"] = max_angle_deg
    else:
        mount_kwargs["angle_limits_deg"] = {"a1": [0.0, 90.0], "a2": [-45.0, 45.0]}
        mount_kwargs["rest_angles_deg"] = {"a1": 0.0, "a2": 0.0}
    mounts = build_ring_mounts(**mount_kwargs)
    solver = SolverConfig()
    render = RenderConfig()
    return SystemConfig(mounts=mounts, solver=solver, render=render)


def _system_from_payload(payload: dict, *, profile: FlightProfile | None = None) -> SystemConfig:
    mounts_data = payload.get("mounts", [])
    if not mounts_data:
        raise ValueError("Config must include a non-empty mounts list.")
    default_engine = payload.get("engine")
    shared_limits = payload.get("angle_limits_deg")
    shared_rest = payload.get("rest_angles_deg")
    mounts = []
    for mount in mounts_data:
        data = dict(mount)
        if shared_limits is not None:
            data["angle_limits_deg"] = shared_limits
        if shared_rest is not None:
            data["rest_angles_deg"] = shared_rest
        mounts.append(_build_mount(data, default_engine=default_engine))
    if profile is not None and profile.engine_variance is not None:
        mounts, draws = apply_engine_variance(mounts, profile.engine_variance)
        profile.engine_variance.draws = draws

    solver_data = payload.get("solver", {})
    solver_defaults = SolverConfig()
    solver = SolverConfig(
        damping=solver_data.get("damping", solver_defaults.damping),
        step_scale=solver_data.get("step_scale", solver_defaults.step_scale),
        max_iters=solver_data.get("max_iters", solver_defaults.max_iters),
        tol=solver_data.get("tol", solver_defaults.tol),
        finite_diff_eps=solver_data.get(
            "finite_diff_eps", solver_defaults.finite_diff_eps
        ),
        force_search_iters=int(
            solver_data.get(
                "force_search_iters", solver_defaults.force_search_iters
            )
        ),
        a2_polish_iters=int(
            solver_data.get("a2_polish_iters", solver_defaults.a2_polish_iters)
        ),
        roll_torque_weight=float(
            solver_data.get(
                "roll_torque_weight", solver_defaults.roll_torque_weight
            )
        ),
        yaw_torque_weight=float(
            solver_data.get(
                "yaw_torque_weight", solver_defaults.yaw_torque_weight
            )
        ),
        tracking_iters=solver_data.get(
            "tracking_iters", solver_defaults.tracking_iters
        ),
        motion_damping=solver_data.get(
            "motion_damping", solver_defaults.motion_damping
        ),
        reseed_interval=solver_data.get(
            "reseed_interval", solver_defaults.reseed_interval
        ),
        refine_passes=solver_data.get(
            "refine_passes", solver_defaults.refine_passes
        ),
    )

    render_data = payload.get("render", {})
    render_defaults = RenderConfig()
    render = RenderConfig(
        force_weight=render_data.get("force_weight", render_defaults.force_weight),
        torque_limit=render_data.get("torque_limit"),
        force_scale=render_data.get("force_scale"),
        force_axis=tuple(render_data.get("force_axis", render_defaults.force_axis)),
    )
    center_of_mass = tuple(
        payload.get("center_of_mass", (0.0, 0.0, 1.0))
    )
    if len(center_of_mass) != 3:
        raise ValueError("center_of_mass must be a length-3 sequence.")
    center_of_mass = (
        float(center_of_mass[0]),
        float(center_of_mass[1]),
        float(center_of_mass[2]),
    )
    unloaded_com = center_of_mass
    unloaded_vehicle = _build_vehicle(payload.get("vehicle"))
    vehicle, center_of_mass = _compose_vehicle_with_motors(
        unloaded_vehicle, center_of_mass, mounts
    )
    motor_cg = payload.get("motor_cg") or {}
    standoff = float(motor_cg.get("pivot_standoff_m", 0.055))
    catch_arms = _build_catch_arms(payload.get("catch_arms"))
    landing_legs = _build_catch_arms(payload.get("landing_legs"))
    raw_mast = payload.get("mast_top_m")
    mast_top_m = None if raw_mast is None else float(raw_mast)
    controller = _build_controller(payload.get("controller"))
    position_controller = _build_position_controller(payload.get("position_controller"))
    altitude_controller = (
        profile.altitude_controller
        if profile is not None
        else _build_altitude_controller(payload.get("altitude_controller"))
    )
    flight_test = _build_flight_test(
        payload.get("flight_test"),
        vehicle,
        controller,
        position_controller,
        altitude_controller,
    )
    if flight_test is not None and profile is not None:
        flight_test.altitude_controller = profile.altitude_controller
        if profile.notes and not flight_test.notes:
            flight_test.notes = profile.notes
    return SystemConfig(
        mounts=mounts,
        solver=solver,
        render=render,
        center_of_mass=center_of_mass,
        vehicle=vehicle,
        controller=controller,
        flight_test=flight_test,
        profile=profile,
        catch_arms=catch_arms,
        unloaded_center_of_mass=unloaded_com,
        unloaded_vehicle=unloaded_vehicle,
        motor_pivot_standoff_m=standoff,
        landing_legs=landing_legs,
        mast_top_m=mast_top_m,
    )


def _engine_label(engine: str | dict | None) -> str | None:
    if engine is None:
        return None
    if isinstance(engine, str):
        return engine
    manufacturer = engine.get("manufacturer")
    designation = engine.get("designation") or engine.get("motor") or engine.get("name")
    if manufacturer and designation:
        return f"{manufacturer}/{designation}"
    if designation:
        return str(designation)
    path = engine.get("dynamics_file") or engine.get("path")
    return str(path) if path else None


def _attach_motor(
    mount_data: dict,
    *,
    default_engine: str | dict | None = None,
) -> tuple[float, str | None, MotorCurve | None]:
    engine = mount_data.get("engine", default_engine)
    motor: MotorCurve | None = None
    if engine is not None:
        motor = load_motor(engine)
        # Flight sim samples the curve directly. Mount.thrust stores peak for UI/bounds.
        thrust = float(motor.max_thrust_n)
    elif "thrust" in mount_data and mount_data["thrust"] is not None:
        thrust = float(mount_data["thrust"])
    else:
        raise ValueError("Each mount needs a thrust value or an engine thrust curve.")
    return thrust, _engine_label(engine), motor


def _build_mount(
    data: dict,
    *,
    default_engine: str | dict | None = None,
) -> Mount:
    position = data["position"]
    thrust, engine_label, motor = _attach_motor(data, default_engine=default_engine)
    base_dir = data.get("base_dir", [0.0, 0.0, 1.0])
    axis1 = data.get("axis1", [1.0, 0.0, 0.0])
    axis2 = data.get("axis2", [0.0, 1.0, 0.0])
    angle_limits_deg = data.get("angle_limits_deg")
    max_angle_deg = data.get("max_angle_deg")
    rest_angles_deg = data.get("rest_angles_deg")
    if "mass_g" in data:
        mass_kg = float(data["mass_g"]) / 1000.0
    elif "mass_kg" in data:
        mass_kg = float(data["mass_kg"])
    else:
        mass_kg = 0.0
    if "length_mm" in data:
        length_m = float(data["length_mm"]) / 1000.0
    elif "length_m" in data:
        length_m = float(data["length_m"])
    elif motor is not None:
        length_m = float(motor.length_m)
    else:
        length_m = 0.0
    return make_mount(
        position=position,
        thrust=thrust,
        base_dir=base_dir,
        axis1=axis1,
        axis2=axis2,
        angle_limits_deg=angle_limits_deg,
        max_angle_deg=max_angle_deg,
        rest_angles_deg=rest_angles_deg,
        engine=engine_label,
        motor=motor,
        mass_kg=mass_kg,
        motor_length_m=length_m,
    )
