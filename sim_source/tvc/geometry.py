from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .motors import MotorCurve


# Gimbal structure from pivot to motor face. Motor CG is this plus half motor length,
# opposite the thrust vector (at rest, thrust is +Z so CG is below the ring).
GIMBAL_MOTOR_STANDOFF_M = 0.055

# Nested a1 is commanded from vertical rest (0 = +Z). Hardware base_dir is inward,
# so the rotation applied about axis1 is π/2 − a1.
A1_VERTICAL_OFFSET_RAD = 0.5 * np.pi


@dataclass
class Mount:
    position: np.ndarray
    thrust: float
    base_dir: np.ndarray
    axis1: np.ndarray
    axis2: np.ndarray
    min_angles: np.ndarray
    max_angles: np.ndarray
    rest_angles: np.ndarray
    engine: str | None = None
    motor: MotorCurve | None = None
    # Point-mass used when no motor curve mass is available (kg).
    mass_kg: float = 0.0
    # Used when no motor-curve length is available (constant-thrust mounts).
    motor_length_m: float = 0.0


@dataclass(frozen=True)
class CatchArm:
    """Body-frame catch arm: root at ``position``, extent ``length_m`` along ``direction``."""

    position: np.ndarray
    direction: np.ndarray
    length_m: float = 0.5

    def tip(self) -> np.ndarray:
        return np.asarray(self.position, dtype=float) + (
            unit(np.asarray(self.direction, dtype=float)) * float(self.length_m)
        )


def unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec.copy()
    return vec / norm


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit(axis)
    if np.linalg.norm(axis) == 0.0:
        return np.eye(3)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def a1_from_base_dir(a1_from_rest: float) -> float:
    """Map command a1 (0 = vertical rest) to the rotation from inward ``base_dir``."""
    return A1_VERTICAL_OFFSET_RAD - float(a1_from_rest)


def gimbal_aim_direction(mount: Mount, desired: np.ndarray) -> np.ndarray:
    """Gimbal angles (a2, a1) that point thrust along ``desired``.

    Nested grid search, refined in place. Used to send each nozzle through
    the CoM so live thrust produces no torque if a motor dies first.
    """
    desired_u = unit(np.asarray(desired, dtype=float).reshape(3))
    if float(np.linalg.norm(desired_u)) < 1e-12:
        return np.asarray(mount.rest_angles, dtype=float).copy()
    a2_min, a1_min = (float(mount.min_angles[0]), float(mount.min_angles[1]))
    a2_max, a1_max = (float(mount.max_angles[0]), float(mount.max_angles[1]))
    best = np.asarray(mount.rest_angles, dtype=float).copy()
    best_dot = float(np.dot(thrust_direction(float(best[0]), float(best[1]), mount), desired_u))
    a2_lo, a2_hi = a2_min, a2_max
    a1_lo, a1_hi = a1_min, a1_max
    for _ in range(4):
        for a2 in np.linspace(a2_lo, a2_hi, 9):
            for a1 in np.linspace(a1_lo, a1_hi, 9):
                dot = float(
                    np.dot(thrust_direction(float(a2), float(a1), mount), desired_u)
                )
                if dot > best_dot:
                    best_dot = dot
                    best = np.array([a2, a1], dtype=float)
        span2 = 0.35 * max(a2_hi - a2_lo, 1e-6)
        span1 = 0.35 * max(a1_hi - a1_lo, 1e-6)
        a2_lo = max(a2_min, float(best[0]) - span2)
        a2_hi = min(a2_max, float(best[0]) + span2)
        a1_lo = max(a1_min, float(best[1]) - span1)
        a1_hi = min(a1_max, float(best[1]) + span1)
    return best


def thrust_direction(ax: float, ay: float, mount: Mount) -> np.ndarray:
    # Primary gimbal about axis2, then nested gimbal about axis1 carried by axis2.
    # ``ay`` is commanded from vertical rest (0 = +Z); convert to base_dir rotation.
    a1_kin = a1_from_base_dir(ay)
    rot_primary = axis_angle_rotation(mount.axis2, ax)
    nested_axis1 = unit(rot_primary @ mount.axis1)
    rot_nested = axis_angle_rotation(nested_axis1, a1_kin)
    return unit(rot_nested @ rot_primary @ mount.base_dir)


def motor_length_m(mount: Mount) -> float:
    if mount.motor is not None and float(getattr(mount.motor, "length_m", 0.0)) > 0.0:
        return float(mount.motor.length_m)
    return float(mount.motor_length_m)


def motor_aft_position(
    mount: Mount,
    a2: float,
    a1: float,
    *,
    standoff_m: float = GIMBAL_MOTOR_STANDOFF_M,
) -> np.ndarray:
    """Body-frame aft end of the motor case.

    The case runs from the gimbal face (one standoff out from the pivot,
    opposite thrust) through the catalog motor length. The center of mass
    sits at mid-case, so doubling the CG offset overshoots the nozzle.
    """
    reach = float(standoff_m) + motor_length_m(mount)
    direction = thrust_direction(float(a2), float(a1), mount)
    return np.asarray(mount.position, dtype=float) - direction * reach


def motor_radius_m(mount: Mount) -> float:
    """Outer radius of the motor case. 0 when the catalog has no diameter."""
    motor = getattr(mount, "motor", None)
    if motor is None:
        return 0.0
    return 0.5 * max(float(getattr(motor, "diameter_m", 0.0) or 0.0), 0.0)


def motor_cg_position(
    mount: Mount,
    a2: float,
    a1: float,
    *,
    standoff_m: float = GIMBAL_MOTOR_STANDOFF_M,
) -> np.ndarray:
    """Body-frame motor CG: pivot minus thrust_dir * (standoff + half motor length)."""
    offset = float(standoff_m) + 0.5 * motor_length_m(mount)
    direction = thrust_direction(float(a2), float(a1), mount)
    return np.asarray(mount.position, dtype=float) - direction * offset


def thrust_envelope_mesh(
    mount: Mount,
    radius: float,
    *,
    samples_per_axis: int = 20,
    invert: bool = False,
) -> list[np.ndarray]:
    """Triangular faces tracing reachable thrust directions on a sphere shell.

    Samples the rectangular actuator limit box in (a2, a1) and maps each pose to
    mount.position +/- thrust_direction * radius, producing a spherical patch.
    """
    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be >= 2")
    sign = -1.0 if invert else 1.0
    a2_vals = np.linspace(mount.min_angles[0], mount.max_angles[0], samples_per_axis)
    a1_vals = np.linspace(mount.min_angles[1], mount.max_angles[1], samples_per_axis)
    grid = np.empty((samples_per_axis, samples_per_axis, 3), dtype=float)
    for i, a2 in enumerate(a2_vals):
        for j, a1 in enumerate(a1_vals):
            direction = thrust_direction(float(a2), float(a1), mount)
            grid[i, j] = mount.position + sign * direction * radius

    faces: list[np.ndarray] = []
    for i in range(samples_per_axis - 1):
        for j in range(samples_per_axis - 1):
            p00 = grid[i, j]
            p10 = grid[i + 1, j]
            p01 = grid[i, j + 1]
            p11 = grid[i + 1, j + 1]
            faces.append(np.array([p00, p10, p11]))
            faces.append(np.array([p00, p11, p01]))
    return faces


def mount_local_frame(position: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an inward-facing mount frame on the XY ring.

    base_dir points toward the origin, axis2 is the primary gimbal axis along
    that inward radial line, and axis1 is the nested ring-tangent axis.
    """
    pos = np.asarray(position, dtype=float)
    radial = unit(pos)
    base_dir = -radial
    axis2 = base_dir
    axis1 = unit(np.array([-radial[1], radial[0], 0.0], dtype=float))
    return base_dir, axis1, axis2


def _angle_limits_from_deg(
    angle_limits_deg: dict[str, list[float]]
    | list[list[float]]
    | float
    | list[float]
    | np.ndarray
    | None = None,
    *,
    max_angle_deg: float | list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-axis limits in radians as [a2, a1] min/max pairs."""
    default_a2 = (-90.0, 90.0)
    default_a1 = (-15.0, 90.0)

    if angle_limits_deg is not None:
        if isinstance(angle_limits_deg, dict):
            a2 = tuple(angle_limits_deg.get("a2", default_a2))
            a1 = tuple(angle_limits_deg.get("a1", default_a1))
        else:
            arr = np.asarray(angle_limits_deg, dtype=float)
            if arr.shape == (2, 2):
                a2 = tuple(arr[0].tolist())
                a1 = tuple(arr[1].tolist())
            elif arr.shape == (4,):
                a2 = (arr[0], arr[1])
                a1 = (arr[2], arr[3])
            elif arr.shape == (2,):
                a2 = (-arr[0], arr[0])
                a1 = (-arr[1], arr[1])
            elif arr.shape == ():
                a2 = (-arr.item(), arr.item())
                a1 = a2
            else:
                raise ValueError(
                    "angle_limits_deg must be a dict, scalar, length-2, 2x2, or length-4 sequence."
                )
    elif max_angle_deg is not None:
        arr = np.asarray(max_angle_deg, dtype=float)
        if arr.shape == ():
            a2 = (-arr.item(), arr.item())
            a1 = a2
        elif arr.shape == (1,):
            a2 = (-arr.item(), arr.item())
            a1 = a2
        elif arr.shape == (2,):
            a2 = (-arr[0], arr[0])
            a1 = (-arr[1], arr[1])
        else:
            raise ValueError("max_angle_deg must be a scalar or length-2 sequence.")
    else:
        a2 = default_a2
        a1 = default_a1

    mins = np.deg2rad(np.array([a2[0], a1[0]], dtype=float))
    maxs = np.deg2rad(np.array([a2[1], a1[1]], dtype=float))
    if np.any(mins > maxs):
        raise ValueError("Angle limits must satisfy min <= max for each axis.")
    return mins, maxs


def _rest_angles_from_deg(
    rest_angles_deg: dict[str, float] | list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return resting pose in radians as [a2, a1]."""
    default_a2 = 0.0
    default_a1 = 0.0
    if rest_angles_deg is None:
        a2 = default_a2
        a1 = default_a1
    elif isinstance(rest_angles_deg, dict):
        a2 = float(rest_angles_deg.get("a2", default_a2))
        a1 = float(rest_angles_deg.get("a1", default_a1))
    else:
        arr = np.asarray(rest_angles_deg, dtype=float)
        if arr.shape == (2,):
            a2, a1 = arr.tolist()
        else:
            raise ValueError("rest_angles_deg must be a dict or length-2 sequence.")
    return np.deg2rad(np.array([a2, a1], dtype=float))


def make_mount(
    position: np.ndarray | list[float],
    thrust: float,
    base_dir: np.ndarray | list[float],
    axis1: np.ndarray | list[float],
    axis2: np.ndarray | list[float],
    angle_limits_deg: dict[str, list[float]]
    | list[list[float]]
    | float
    | list[float]
    | np.ndarray
    | None = None,
    max_angle_deg: float | list[float] | np.ndarray | None = None,
    rest_angles_deg: dict[str, float] | list[float] | np.ndarray | None = None,
    engine: str | None = None,
    motor: MotorCurve | None = None,
    mass_kg: float = 0.0,
    motor_length_m: float = 0.0,
) -> Mount:
    pos = np.asarray(position, dtype=float)
    base = unit(np.asarray(base_dir, dtype=float))
    axis1 = unit(np.asarray(axis1, dtype=float))
    axis2 = unit(np.asarray(axis2, dtype=float))
    min_angles, max_angles = _angle_limits_from_deg(
        angle_limits_deg,
        max_angle_deg=max_angle_deg,
    )
    rest_angles = _rest_angles_from_deg(rest_angles_deg)
    if np.any(rest_angles < min_angles) or np.any(rest_angles > max_angles):
        raise ValueError("rest_angles_deg must lie within angle_limits_deg.")
    return Mount(
        position=pos,
        thrust=float(thrust),
        base_dir=base,
        axis1=axis1,
        axis2=axis2,
        min_angles=min_angles,
        max_angles=max_angles,
        rest_angles=rest_angles,
        engine=engine,
        motor=motor,
        mass_kg=float(mass_kg),
        motor_length_m=float(motor_length_m),
    )


def build_ring_mounts(
    count: int,
    radius: float,
    thrust: float,
    base_dir: np.ndarray | list[float] | None = None,
    axis1: np.ndarray | list[float] | None = None,
    axis2: np.ndarray | list[float] | None = None,
    angle_limits_deg: dict[str, list[float]]
    | list[list[float]]
    | float
    | list[float]
    | np.ndarray
    | None = None,
    max_angle_deg: float | list[float] | np.ndarray | None = None,
    rest_angles_deg: dict[str, float] | list[float] | np.ndarray | None = None,
    face_origin: bool = True,
) -> list[Mount]:
    if count < 1:
        raise ValueError("count must be >= 1")
    use_global_axes = base_dir is not None or axis1 is not None or axis2 is not None
    if use_global_axes and face_origin:
        raise ValueError("face_origin cannot be used with explicit mount axes.")
    global_base = unit(np.array([0.0, 0.0, 1.0]) if base_dir is None else base_dir)
    global_axis1 = unit(np.array([1.0, 0.0, 0.0]) if axis1 is None else axis1)
    global_axis2 = unit(np.array([0.0, 1.0, 0.0]) if axis2 is None else axis2)
    mounts = []
    for idx in range(count):
        angle = 2.0 * np.pi * idx / count
        position = np.array(
            [radius * np.cos(angle), radius * np.sin(angle), 0.0], dtype=float
        )
        if face_origin and not use_global_axes:
            mount_base, mount_axis1, mount_axis2 = mount_local_frame(position)
        else:
            mount_base, mount_axis1, mount_axis2 = (
                global_base,
                global_axis1,
                global_axis2,
            )
        mounts.append(
            make_mount(
                position=position,
                thrust=thrust,
                base_dir=mount_base,
                axis1=mount_axis1,
                axis2=mount_axis2,
                angle_limits_deg=angle_limits_deg,
                max_angle_deg=max_angle_deg,
                rest_angles_deg=rest_angles_deg,
            )
        )
    return mounts
