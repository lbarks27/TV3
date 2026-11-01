from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attitude import normalize_quaternion, quaternion_derivative, rotate_vector
from .geometry import GIMBAL_MOTOR_STANDOFF_M, Mount, motor_cg_position, thrust_direction


@dataclass
class VehicleProperties:
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([0.05, 0.05, 0.08]).astype(float)
    )
    mass: float = 1.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=float))

    def __post_init__(self) -> None:
        self.inertia = np.asarray(self.inertia, dtype=float)
        self.gravity = np.asarray(self.gravity, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError("inertia must be a 3x3 matrix")
        if self.gravity.shape != (3,):
            raise ValueError("gravity must be a length-3 vector")


def point_mass_inertia(mass: float, offset: np.ndarray) -> np.ndarray:
    """Inertia tensor of a point mass about the origin of ``offset`` (kg·m²)."""
    r = np.asarray(offset, dtype=float).reshape(3)
    return float(mass) * (float(np.dot(r, r)) * np.eye(3) - np.outer(r, r))


def compose_loaded_properties(
    unloaded_mass: float,
    unloaded_com: np.ndarray | list[float] | tuple[float, float, float],
    unloaded_inertia: np.ndarray,
    point_masses: list[tuple[float, np.ndarray | list[float]]],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Combine unloaded body + point masses into net mass, CoM, and inertia-about-CoM.

    ``unloaded_inertia`` is the body tensor about ``unloaded_com``. Each entry in
    ``point_masses`` is ``(mass_kg, position_body)``. Returns
    ``(mass_kg, com_body, inertia_about_com)``.
    """
    m0 = float(unloaded_mass)
    if m0 <= 0.0:
        raise ValueError("unloaded_mass must be positive")
    r0 = np.asarray(unloaded_com, dtype=float).reshape(3)
    i0 = np.asarray(unloaded_inertia, dtype=float)
    if i0.shape != (3, 3):
        raise ValueError("unloaded_inertia must be a 3x3 matrix")

    masses = [m0]
    positions = [r0]
    for mass, position in point_masses:
        m_i = float(mass)
        if m_i <= 0.0:
            continue
        masses.append(m_i)
        positions.append(np.asarray(position, dtype=float).reshape(3))

    total_mass = float(sum(masses))
    com = sum(m * p for m, p in zip(masses, positions)) / total_mass

    # Shift unloaded inertia from its CoM to the loaded CoM, then add point masses.
    inertia = i0 + point_mass_inertia(m0, r0 - com)
    for mass, position in zip(masses[1:], positions[1:]):
        inertia = inertia + point_mass_inertia(mass, position - com)
    return total_mass, com, inertia


def mount_point_mass_kg(mount: Mount, time_s: float = 0.0) -> float:
    """Current mount mass (kg). Curve motors lose propellant with delivered impulse."""
    if mount.motor is not None and float(mount.motor.initial_mass_kg) > 0.0:
        return float(mount.motor.mass_at(time_s))
    return float(mount.mass_kg)


def offset_motor_cgs(
    mounts: list[Mount],
    angles: np.ndarray,
    *,
    standoff_m: float = GIMBAL_MOTOR_STANDOFF_M,
) -> np.ndarray:
    """Body-frame motor CG for each mount at the given (a2, a1) pose."""
    pose = np.asarray(angles, dtype=float).reshape(len(mounts), 2)
    return np.vstack(
        [
            motor_cg_position(mount, pose[i, 0], pose[i, 1], standoff_m=standoff_m)
            for i, mount in enumerate(mounts)
        ]
    )


def compose_pivot_mass_properties(
    unloaded: VehicleProperties,
    unloaded_com: np.ndarray | list[float] | tuple[float, float, float],
    mounts: list[Mount],
    *,
    time_s: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Loaded mass, CoM, and inertia with motor mass at the gimbal pivots."""
    point_masses = [
        (mount_point_mass_kg(mount, time_s), mount.position)
        for mount in mounts
        if mount_point_mass_kg(mount, time_s) > 0.0
    ]
    if not point_masses:
        return (
            float(unloaded.mass),
            np.asarray(unloaded_com, dtype=float).reshape(3),
            np.asarray(unloaded.inertia, dtype=float),
        )
    return compose_loaded_properties(
        unloaded.mass,
        unloaded_com,
        unloaded.inertia,
        point_masses,
    )


def compose_offset_mass_properties(
    unloaded: VehicleProperties,
    unloaded_com: np.ndarray | list[float] | tuple[float, float, float],
    mounts: list[Mount],
    angles: np.ndarray,
    *,
    standoff_m: float = GIMBAL_MOTOR_STANDOFF_M,
    time_s: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Loaded mass, CoM, inertia-about-CoM, and motor CGs at the current gimbal pose."""
    cgs = offset_motor_cgs(mounts, angles, standoff_m=standoff_m)
    point_masses = []
    for mount, cg in zip(mounts, cgs):
        mass = mount_point_mass_kg(mount, time_s)
        if mass > 0.0:
            point_masses.append((mass, cg))
    if not point_masses:
        com = np.asarray(unloaded_com, dtype=float).reshape(3)
        return float(unloaded.mass), com, np.asarray(unloaded.inertia, dtype=float), cgs
    mass, com, inertia = compose_loaded_properties(
        unloaded.mass,
        unloaded_com,
        unloaded.inertia,
        point_masses,
    )
    return mass, com, inertia, cgs


def thrust_wrench_about_com(
    mounts: list[Mount],
    angles: np.ndarray,
    thrusts: np.ndarray,
    com: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Thrust force and torque about ``com`` (body frame). Force is applied at the pivot."""
    pose = np.asarray(angles, dtype=float).reshape(len(mounts), 2)
    thrusts = np.asarray(thrusts, dtype=float).reshape(len(mounts))
    com = np.asarray(com, dtype=float).reshape(3)
    force = np.zeros(3, dtype=float)
    torque = np.zeros(3, dtype=float)
    for i, mount in enumerate(mounts):
        direction = thrust_direction(pose[i, 0], pose[i, 1], mount)
        fi = direction * float(thrusts[i])
        force = force + fi
        torque = torque + np.cross(np.asarray(mount.position, dtype=float) - com, fi)
    return torque, force


def motor_inertial_wrench(
    masses: np.ndarray,
    cgs: np.ndarray,
    prev_cgs: np.ndarray | None,
    prev_vels: np.ndarray | None,
    com: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D'Alembert reaction of moving motor point masses in the body frame.

    Returns ``(torque, force, cg_velocities)``. First step (no history) is zero.
    """
    n = int(np.asarray(cgs).reshape(-1, 3).shape[0])
    cgs = np.asarray(cgs, dtype=float).reshape(n, 3)
    masses = np.asarray(masses, dtype=float).reshape(n)
    com = np.asarray(com, dtype=float).reshape(3)
    vels = np.zeros((n, 3), dtype=float)
    force = np.zeros(3, dtype=float)
    torque = np.zeros(3, dtype=float)
    if prev_cgs is None or dt <= 1e-9:
        return torque, force, vels
    prev_cgs = np.asarray(prev_cgs, dtype=float).reshape(n, 3)
    vels = (cgs - prev_cgs) / float(dt)
    if prev_vels is None:
        return torque, force, vels
    prev_vels = np.asarray(prev_vels, dtype=float).reshape(n, 3)
    acc = (vels - prev_vels) / float(dt)
    for i in range(n):
        m_i = float(masses[i])
        if m_i <= 0.0:
            continue
        fi = -m_i * acc[i]
        force = force + fi
        torque = torque + np.cross(cgs[i] - com, fi)
    return torque, force, vels


@dataclass
class VehicleState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def copy(self) -> VehicleState:
        return VehicleState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            omega=self.omega.copy(),
        )


# Backward-compatible alias used by older attitude-only code paths.
AttitudeState = VehicleState


def rotational_acceleration(
    omega: np.ndarray,
    torque: np.ndarray,
    inertia: np.ndarray,
) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    torque = np.asarray(torque, dtype=float)
    return np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))


def translational_acceleration(
    force_body: np.ndarray,
    quaternion: np.ndarray,
    vehicle: VehicleProperties,
    *,
    force_inertial_extra: np.ndarray | None = None,
) -> np.ndarray:
    force_body = np.asarray(force_body, dtype=float)
    force_inertial = rotate_vector(quaternion, force_body)
    if force_inertial_extra is not None:
        force_inertial = force_inertial + np.asarray(force_inertial_extra, dtype=float)
    return force_inertial / vehicle.mass + vehicle.gravity


def quadratic_drag_force_inertial(
    velocity_inertial: np.ndarray,
    wind_inertial: np.ndarray,
    *,
    drag_coeff: float,
    ref_area_m2: float,
    air_density_kg_m3: float = 1.225,
) -> np.ndarray:
    """Quadratic aero drag in the inertial frame from air-relative velocity.

    ``F = -0.5 ρ Cd A |v_rel| v_rel`` with ``v_rel = v − v_wind``.
    """
    if drag_coeff <= 0.0 or ref_area_m2 <= 0.0 or air_density_kg_m3 <= 0.0:
        return np.zeros(3, dtype=float)
    v_rel = np.asarray(velocity_inertial, dtype=float) - np.asarray(wind_inertial, dtype=float)
    speed = float(np.linalg.norm(v_rel))
    if speed < 1e-9:
        return np.zeros(3, dtype=float)
    scale = -0.5 * float(air_density_kg_m3) * float(drag_coeff) * float(ref_area_m2) * speed
    return scale * v_rel


class WindField:
    """Mean wind + isotropic Gauss-Markov gusts in the inertial frame."""

    def __init__(
        self,
        mean_m_s: np.ndarray | list[float] | tuple[float, float, float],
        *,
        gust_std_m_s: float = 0.0,
        gust_tau_s: float = 2.0,
        seed: int | None = None,
    ) -> None:
        self.mean_m_s = np.asarray(mean_m_s, dtype=float).reshape(3).copy()
        self.gust_std_m_s = max(float(gust_std_m_s), 0.0)
        self.gust_tau_s = max(float(gust_tau_s), 1e-3)
        self._gust = np.zeros(3, dtype=float)
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        if self.gust_std_m_s <= 0.0:
            self._gust[:] = 0.0
            return
        self._gust = self._rng.normal(0.0, self.gust_std_m_s, size=3)

    def step(self, dt: float) -> np.ndarray:
        """Advance gust state by ``dt`` and return total wind (mean + gust)."""
        dt = float(dt)
        if self.gust_std_m_s > 0.0 and dt > 0.0:
            tau = self.gust_tau_s
            decay = float(np.exp(-dt / tau))
            sigma = self.gust_std_m_s * float(np.sqrt(max(1.0 - decay * decay, 0.0)))
            self._gust = decay * self._gust + self._rng.normal(0.0, sigma, size=3)
        return self.mean_m_s + self._gust

    @property
    def velocity_m_s(self) -> np.ndarray:
        return self.mean_m_s + self._gust


def integrate_6dof_rk4(
    state: VehicleState,
    force_body: np.ndarray,
    torque: np.ndarray,
    vehicle: VehicleProperties,
    dt: float,
    *,
    force_inertial_extra: np.ndarray | None = None,
) -> VehicleState:
    extra = (
        None
        if force_inertial_extra is None
        else np.asarray(force_inertial_extra, dtype=float).reshape(3)
    )

    def derivatives(
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        omega: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = normalize_quaternion(quaternion)
        p_dot = velocity
        v_dot = translational_acceleration(
            force_body, q, vehicle, force_inertial_extra=extra
        )
        q_dot = quaternion_derivative(q, omega)
        w_dot = rotational_acceleration(omega, torque, vehicle.inertia)
        return p_dot, v_dot, q_dot, w_dot

    p0 = state.position.copy()
    v0 = state.velocity.copy()
    q0 = normalize_quaternion(state.quaternion)
    w0 = state.omega.copy()

    p1_dot, v1_dot, q1_dot, w1_dot = derivatives(p0, v0, q0, w0)
    p2 = p0 + 0.5 * dt * p1_dot
    v2 = v0 + 0.5 * dt * v1_dot
    q2 = normalize_quaternion(q0 + 0.5 * dt * q1_dot)
    w2 = w0 + 0.5 * dt * w1_dot
    p2_dot, v2_dot, q2_dot, w2_dot = derivatives(p2, v2, q2, w2)

    p3 = p0 + 0.5 * dt * p2_dot
    v3 = v0 + 0.5 * dt * v2_dot
    q3 = normalize_quaternion(q0 + 0.5 * dt * q2_dot)
    w3 = w0 + 0.5 * dt * w2_dot
    p3_dot, v3_dot, q3_dot, w3_dot = derivatives(p3, v3, q3, w3)

    p4 = p0 + dt * p3_dot
    v4 = v0 + dt * v3_dot
    q4 = normalize_quaternion(q0 + dt * q3_dot)
    w4 = w0 + dt * w3_dot
    p4_dot, v4_dot, q4_dot, w4_dot = derivatives(p4, v4, q4, w4)

    return VehicleState(
        position=p0 + (dt / 6.0) * (p1_dot + 2 * p2_dot + 2 * p3_dot + p4_dot),
        velocity=v0 + (dt / 6.0) * (v1_dot + 2 * v2_dot + 2 * v3_dot + v4_dot),
        quaternion=normalize_quaternion(
            q0 + (dt / 6.0) * (q1_dot + 2 * q2_dot + 2 * q3_dot + q4_dot)
        ),
        omega=w0 + (dt / 6.0) * (w1_dot + 2 * w2_dot + 2 * w3_dot + w4_dot),
    )


def integrate_attitude_rk4(
    state: VehicleState,
    torque: np.ndarray,
    vehicle: VehicleProperties,
    dt: float,
) -> VehicleState:
    return integrate_6dof_rk4(state, np.zeros(3, dtype=float), torque, vehicle, dt)
