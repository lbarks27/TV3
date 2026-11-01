"""Landing-zone chopsticks: two X-parallel bars that catch vehicle catch-arms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attitude import quaternion_conjugate, rotate_vector
from .dynamics import VehicleProperties, VehicleState
from .geometry import CatchArm, motor_aft_position, motor_radius_m


@dataclass(frozen=True)
class LandingChopsticks:
    """Two parallel catch bars at a pad.

    Bars run along world +X (the +x/−x line), ``separation_m`` apart in Y,
    each ``length_m`` long, at world height ``height_m``. Catch arms lie along
    body ±X, so the vehicle must yaw ≈ ±90° for the arms to rest across both
    bars.
    """

    center_xy: tuple[float, float] = (0.0, 0.0)
    length_m: float = 0.75
    separation_m: float = 0.6
    height_m: float = 1.3081
    radius_m: float = 0.03
    restitution: float = 0.0
    friction: float = 1.0
    heading_tol_deg: float = 25.0
    solver_iterations: int = 4

    def segments(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        cx, cy = float(self.center_xy[0]), float(self.center_xy[1])
        half = 0.5 * float(self.length_m)
        off = 0.5 * float(self.separation_m)
        z = float(self.height_m)
        a0 = np.array([cx - half, cy - off, z], dtype=float)
        a1 = np.array([cx + half, cy - off, z], dtype=float)
        b0 = np.array([cx - half, cy + off, z], dtype=float)
        b1 = np.array([cx + half, cy + off, z], dtype=float)
        return (a0, a1), (b0, b1)

    def points(self) -> np.ndarray:
        segs = self.segments()
        return np.vstack([segs[0][0], segs[0][1], segs[1][0], segs[1][1]])


def chopsticks_for_pad(
    site_xy: tuple[float, float] | np.ndarray,
    catch_arms: list[CatchArm],
    *,
    length_m: float = 0.75,
    separation_m: float = 0.6,
) -> LandingChopsticks | None:
    """Place chopsticks at the pad, height matched to the catch-arm roots."""
    if not catch_arms:
        return None
    height = float(
        np.mean([float(np.asarray(arm.position, dtype=float)[2]) for arm in catch_arms])
    )
    xy = np.asarray(site_xy, dtype=float).reshape(-1)
    return LandingChopsticks(
        center_xy=(float(xy[0]), float(xy[1])),
        length_m=float(length_m),
        separation_m=float(separation_m),
        height_m=height,
    )


def catch_heading_aligned(quaternion: np.ndarray, *, max_error_deg: float = 25.0) -> bool:
    """True when body +X is within ``max_error_deg`` of world ±Y (yaw ±90°)."""
    arm_world = rotate_vector(quaternion, np.array([1.0, 0.0, 0.0], dtype=float))
    xy = arm_world[:2]
    n = float(np.linalg.norm(xy))
    if n < 1e-9:
        return False
    y_align = abs(float(xy[1])) / n
    return y_align >= float(np.cos(np.deg2rad(max_error_deg)))


def in_chopstick_catch_volume(
    state: VehicleState,
    chopsticks: LandingChopsticks,
    *,
    z_min_m: float = -0.05,
    z_max_m: float = 0.12,
    xy_margin_m: float = 0.0,
) -> bool:
    """True when the body origin is between the two bars, near stick height.

    The rectangle is the bar length by the bar gap. A margin used to extend
    past the bars, which counted a body south of the south stick as a catch.
    This is not the sim freeze: that requires the arms to hit both bars.
    """
    pos = np.asarray(state.position, dtype=float).reshape(3)
    z = float(pos[2])
    if z < float(z_min_m) or z > float(z_max_m):
        return False
    cx, cy = float(chopsticks.center_xy[0]), float(chopsticks.center_xy[1])
    half_x = 0.5 * float(chopsticks.length_m) + float(xy_margin_m)
    half_y = 0.5 * float(chopsticks.separation_m) + float(xy_margin_m)
    return abs(float(pos[0]) - cx) <= half_x and abs(float(pos[1]) - cy) <= half_y


# The stick frame has a physical backstop this far west of the bar center.
# Any vehicle point past that plane cannot be a landing.
STICK_BACKSTOP_WEST_M = 0.5
# Mast contact ignores the last bit of the shaft, which is the arm joint.
_MAST_BELOW_ARM_M = 0.02
_BODY_TOUCH_MARGIN_M = 0.005


def stick_geometry_faults(
    position: np.ndarray,
    quaternion: np.ndarray,
    catch_arms: list[CatchArm],
    mounts: list,
    chopsticks: LandingChopsticks,
    *,
    motor_cgs_body: np.ndarray | None = None,
    gimbal_angles: np.ndarray | None = None,
    standoff_m: float = 0.055,
    backstop_west_m: float = STICK_BACKSTOP_WEST_M,
) -> set[str]:
    """Faults for one pose against the stick backstop and the bar cylinders.

    ``backstop`` means some part of the vehicle, including the catch arms and
    the motor cases, is west of the backstop. ``body_contact`` means the mast
    or a motor case meets a bar. The catch arms are allowed to rest on the bars.

    With gimbal angles, each motor is the case extruded from its pivot: one
    standoff plus the catalog length, opposite thrust, and the catalog
    diameter. Without angles, a logged center of mass is still treated as the
    midpoint of a zero-radius segment.
    """
    origin = np.asarray(position, dtype=float).reshape(3)
    q = np.asarray(quaternion, dtype=float).reshape(4)
    cgs = (
        None
        if motor_cgs_body is None
        else np.asarray(motor_cgs_body, dtype=float).reshape(-1, 3)
    )
    angles = (
        None
        if gimbal_angles is None
        else np.asarray(gimbal_angles, dtype=float).reshape(-1, 2)
    )

    def _world(point_body: np.ndarray) -> np.ndarray:
        return origin + rotate_vector(q, np.asarray(point_body, dtype=float).reshape(3))

    # (is_arm, body_a, body_b, case_radius)
    segments: list[tuple[bool, np.ndarray, np.ndarray, float]] = []
    arms = list(catch_arms)
    if arms:
        root = np.asarray(arms[0].position, dtype=float).reshape(3)
        mast_top = root.copy()
        mast_top[2] = max(0.0, float(root[2]) - _MAST_BELOW_ARM_M)
        segments.append((False, np.zeros(3, dtype=float), mast_top, 0.0))
    for arm in arms:
        segments.append(
            (
                True,
                np.asarray(arm.position, dtype=float).reshape(3),
                np.asarray(arm.tip(), dtype=float),
                0.0,
            )
        )
    for index, mount in enumerate(mounts):
        pivot = np.asarray(mount.position, dtype=float).reshape(3)
        radius = float(motor_radius_m(mount))
        if angles is not None and index < len(angles):
            aft = motor_aft_position(
                mount,
                float(angles[index, 0]),
                float(angles[index, 1]),
                standoff_m=float(standoff_m),
            )
            segments.append((False, pivot, aft, radius))
        elif cgs is not None and index < len(cgs):
            nozzle = pivot + 2.0 * (cgs[index] - pivot)
            segments.append((False, pivot, nozzle, radius))
        else:
            segments.append((False, pivot, pivot.copy(), radius))
    if not segments:
        segments.append((False, np.zeros(3, dtype=float), np.zeros(3, dtype=float), 0.0))

    faults: set[str] = set()
    limit = float(chopsticks.center_xy[0]) - float(backstop_west_m)
    bar_touch = float(chopsticks.radius_m) + _BODY_TOUCH_MARGIN_M
    bars = chopsticks.segments()
    for is_arm, body_a, body_b, radius in segments:
        world_a = _world(body_a)
        world_b = _world(body_b)
        west = min(float(world_a[0]), float(world_b[0])) - float(radius)
        if west < limit or float(world_a[0]) < limit or float(world_b[0]) < limit:
            faults.add("backstop")
        if is_arm:
            continue
        touch = bar_touch + float(radius)
        for bar_a, bar_b in bars:
            _pa, _pb, dist = closest_points_on_segments(world_a, world_b, bar_a, bar_b)
            if dist <= touch:
                faults.add("body_contact")
                break
    return faults


def closest_points_on_segments(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(pa, pb, distance)`` for the closest points on two segments."""
    a0 = np.asarray(a0, dtype=float).reshape(3)
    a1 = np.asarray(a1, dtype=float).reshape(3)
    b0 = np.asarray(b0, dtype=float).reshape(3)
    b1 = np.asarray(b1, dtype=float).reshape(3)
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    aa = float(np.dot(d1, d1))
    ee = float(np.dot(d2, d2))
    ff = float(np.dot(d2, r))
    eps = 1e-16
    if aa <= eps and ee <= eps:
        return a0, b0, float(np.linalg.norm(a0 - b0))
    if aa <= eps:
        s = 0.0
        t = float(np.clip(ff / ee, 0.0, 1.0))
    else:
        cc = float(np.dot(d1, r))
        if ee <= eps:
            t = 0.0
            s = float(np.clip(-cc / aa, 0.0, 1.0))
        else:
            bb = float(np.dot(d1, d2))
            denom = aa * ee - bb * bb
            if abs(denom) > eps:
                s = float(np.clip((bb * ff - cc * ee) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (bb * s + ff) / ee
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-cc / aa, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((bb - cc) / aa, 0.0, 1.0))
            else:
                t = float(t)
    pa = a0 + s * d1
    pb = b0 + t * d2
    return pa, pb, float(np.linalg.norm(pa - pb))


def apply_chopstick_contacts(
    state: VehicleState,
    vehicle: VehicleProperties,
    catch_arms: list[CatchArm],
    chopsticks: LandingChopsticks,
    *,
    com_body: np.ndarray | None = None,
    dt: float = 0.01,
) -> tuple[VehicleState, int, bool]:
    """Impulse contacts between catch-arms and chopsticks. Body does not collide.

    Returns ``(state, n_distinct_sticks_hit, heading_ok)``.
    """
    heading_ok = catch_heading_aligned(
        state.quaternion, max_error_deg=float(chopsticks.heading_tol_deg)
    )
    if not catch_arms:
        return state, 0, heading_ok

    origin = np.asarray(state.position, dtype=float).reshape(3).copy()
    vel = np.asarray(state.velocity, dtype=float).reshape(3).copy()
    omega = np.asarray(state.omega, dtype=float).reshape(3).copy()
    q = np.asarray(state.quaternion, dtype=float)
    mass = max(float(vehicle.mass), 1e-6)
    inertia = np.asarray(vehicle.inertia, dtype=float)
    com = (
        np.zeros(3, dtype=float)
        if com_body is None
        else np.asarray(com_body, dtype=float).reshape(3)
    )
    radius = float(chopsticks.radius_m)
    q_conj = quaternion_conjugate(q)
    inv_i = np.linalg.inv(inertia)
    n_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    n_body = rotate_vector(q_conj, n_hat)
    # A line rests on the cylinder one radius above the centerline.
    # The old diameter target pushed the body back up and froze the arms
    # about 3 cm above the bars. A few millimetres covers the timestep.
    xy_tol = radius + 0.01
    on_bar = radius + 0.005
    sticks_hit: set[int] = set()
    iterations = max(1, int(chopsticks.solver_iterations))

    def _world(point_body: np.ndarray) -> np.ndarray:
        return origin + rotate_vector(q, np.asarray(point_body, dtype=float).reshape(3))

    for _ in range(iterations):
        com_world = origin + rotate_vector(q, com)
        for arm in catch_arms:
            arm_a = _world(arm.position)
            arm_b = _world(arm.tip())
            for stick_index, (s0, s1) in enumerate(chopsticks.segments()):
                pa, pb, _dist = closest_points_on_segments(arm_a, arm_b, s0, s1)
                dxy = float(np.linalg.norm((pa - pb)[:2]))
                dz = float(pa[2] - pb[2])
                if dxy > xy_tol or dz > on_bar:
                    continue
                sticks_hit.add(stick_index)
                penetration = max(0.0, radius - dz)
                r = pa - com_world
                r_body = rotate_vector(q_conj, r)
                w_world = rotate_vector(q, omega)
                v_contact = vel + np.cross(w_world, r)
                vn = float(np.dot(v_contact, n_hat))
                lam = 0.0
                if penetration > 1e-9 or vn < 0.0:
                    rn = np.cross(r_body, n_body)
                    kn = (1.0 / mass) + float(np.dot(rn, inv_i @ rn))
                    kn = max(kn, 1e-8)
                    baumgarte = 0.35 * penetration / max(float(dt), 1e-4)
                    lam = -(vn * (1.0 + float(chopsticks.restitution)) + baumgarte) / kn
                    lam = max(float(lam), 0.0)
                    if lam > 0.0:
                        vel = vel + (lam * n_hat) / mass
                        omega = omega + inv_i @ np.cross(r_body, lam * n_body)
                v_contact = vel + np.cross(rotate_vector(q, omega), r)
                vt = v_contact - float(np.dot(v_contact, n_hat)) * n_hat
                vt_n = float(np.linalg.norm(vt))
                if vt_n > 1e-6:
                    t_hat = vt / vt_n
                    t_body = rotate_vector(q_conj, t_hat)
                    rt = np.cross(r_body, t_body)
                    kt = (1.0 / mass) + float(np.dot(rt, inv_i @ rt))
                    kt = max(kt, 1e-8)
                    friction_cap = max(
                        float(chopsticks.friction) * max(lam, mass * 9.81 * dt),
                        vt_n / kt,
                    )
                    lt = min(vt_n / kt, friction_cap)
                    vel = vel - (lt * t_hat) / mass
                    omega = omega + inv_i @ np.cross(r_body, -lt * t_body)
                if penetration > 1e-6:
                    origin = origin + n_hat * (0.4 * penetration)

    if len(sticks_hit) >= 2:
        # Arms across both bars: the vehicle is in the chopsticks. Freeze it.
        vel = np.zeros(3, dtype=float)
        omega = np.zeros(3, dtype=float)

    new_state = VehicleState(
        position=origin,
        velocity=vel,
        quaternion=q.copy(),
        omega=omega,
    )
    return new_state, len(sticks_hit), heading_ok
