from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attitude import (
    normalize_quaternion,
    quaternion_error_vector,
    quaternion_from_thrust_and_heading,
    quaternion_from_z_to_direction,
)


@dataclass
class PositionPIDConfig:
    """Position → attitude gains.

    Prefer ``velocity_cascade=True`` (outer velocity loop):
    ``v_cmd = sat(v_ref + kp·e + ki·I, max_speed)``,
    ``accel = kd·(v_cmd - v) + a_ff``
    with ``kp`` in 1/s and ``kd`` in 1/s.

    When cascade is False, classical ``accel ≈ kp·e + kd·ė`` (legacy profiles).
    """

    kp: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 1.2], dtype=float))
    kd: np.ndarray = field(default_factory=lambda: np.array([1.4, 1.4, 2.0], dtype=float))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    max_tilt_deg: float = 20.0
    min_upward_accel: float = 2.0
    # Flythrough / capture: steer velocity onto the line-of-sight to the waypoint.
    velocity_align_kp: float = 2.0
    velocity_aim_min_speed: float = 2.0
    # XY position-integral limit (m·s) and cascade switch for tight path tracking.
    integral_limit: float | None = 1.0
    velocity_cascade: bool = True
    # Soft-clamp |e_xy| before the cascade so step waypoints do not demand
    # impossible speed/accel under a2 limits. 0 disables.
    max_position_error_m: float = 3.0
    # Tighter soft-clamp on |e_y| alone (plane-hold). 0 → use XY clamp only.
    max_position_error_y_m: float = 1.0
    # Saturate cascade |v_cmd_xy|. 0 disables. Scalar or [vx_max, vy_max].
    max_speed_m_s: float | tuple[float, float] = 4.0
    # Do not integrate XY until |e_xy| is inside this radius (m). 0 disables.
    integral_start_m: float = 1.5
    # Optional Y-only tilt budget (deg). None → share max_tilt_deg. Keeps the
    # cross-track loop from bang-banging the full cone and oscillating.
    max_tilt_y_deg: float | None = None
    # First-order LPF on lateral accel command (Hz). 0 disables. Kills the
    # ~2–3 s Y chatter that attitude lag + tilt sat otherwise sustain.
    lateral_accel_lpf_hz: float = 1.2
    # Soft xz-plane priority: when |y| error exceeds this (m), derate the
    # X accel command so roll authority is available to return to the plane.
    # 0 disables. Does not use a hard shared tilt cone.
    lateral_priority_m: float = 0.0
    prefer_lateral_y: bool = False

    def __post_init__(self) -> None:
        self.kp = np.asarray(self.kp, dtype=float)
        self.kd = np.asarray(self.kd, dtype=float)
        self.ki = np.asarray(self.ki, dtype=float)
        if self.kp.shape != (3,) or self.kd.shape != (3,) or self.ki.shape != (3,):
            raise ValueError("Position PID gains must be length-3 vectors")
        self.velocity_align_kp = float(self.velocity_align_kp)
        self.velocity_aim_min_speed = float(self.velocity_aim_min_speed)
        self.velocity_cascade = bool(self.velocity_cascade)
        self.max_position_error_m = float(self.max_position_error_m)
        self.max_position_error_y_m = float(self.max_position_error_y_m)
        if isinstance(self.max_speed_m_s, (list, tuple, np.ndarray)):
            vals = [float(v) for v in self.max_speed_m_s]
            if len(vals) < 2:
                raise ValueError("max_speed_m_s vector needs at least [vx, vy]")
            self.max_speed_m_s = (vals[0], vals[1])
        else:
            self.max_speed_m_s = float(self.max_speed_m_s)
        self.integral_start_m = float(self.integral_start_m)
        self.max_tilt_y_deg = (
            None
            if self.max_tilt_y_deg is None
            else float(self.max_tilt_y_deg)
        )
        self.lateral_accel_lpf_hz = float(self.lateral_accel_lpf_hz)
        self.lateral_priority_m = float(self.lateral_priority_m)
        self.prefer_lateral_y = bool(self.prefer_lateral_y)


class PositionPIDController:
    """Map position/velocity error to a thrust-pointing attitude reference."""

    def __init__(self, config: PositionPIDConfig | None = None) -> None:
        self.config = config or PositionPIDConfig()
        self.integral = np.zeros(3, dtype=float)
        self._accel_lpf = np.zeros(3, dtype=float)
        self._accel_lpf_init = False

    def reset(self) -> None:
        self.integral[:] = 0.0
        self._accel_lpf[:] = 0.0
        self._accel_lpf_init = False

    def _shaped_position_error(self, position_error: np.ndarray) -> np.ndarray:
        """Soft-clamp XY error so large step refs do not explode v_cmd."""
        err = np.asarray(position_error, dtype=float).copy()
        e_max = float(self.config.max_position_error_m)
        if e_max > 1e-9:
            xy = float(np.linalg.norm(err[:2]))
            if xy > e_max:
                err[:2] *= e_max / xy
        e_y = float(self.config.max_position_error_y_m)
        if e_y > 1e-9 and abs(float(err[1])) > e_y:
            err[1] = float(np.sign(err[1]) * e_y)
        return err

    def _update_integral(self, position_error: np.ndarray, dt: float) -> None:
        if dt <= 0.0:
            return
        # Integrate XY only; altitude is closed by the axial-force loop.
        # Hold integral at zero while far from the setpoint (avoid windup into
        # the tilt cone on long approaches).
        start = float(self.config.integral_start_m)
        xy = float(np.linalg.norm(position_error[:2]))
        if start > 1e-9 and xy > start:
            self.integral[:2] *= 0.90
            self.integral[2] = 0.0
            return
        self.integral[0] += float(position_error[0]) * dt
        self.integral[1] += float(position_error[1]) * dt
        self.integral[2] = 0.0
        if self.config.integral_limit is not None:
            limit = float(self.config.integral_limit)
            self.integral[:2] = np.clip(self.integral[:2], -limit, limit)

    def _speed_limits(self) -> tuple[float, float]:
        raw = self.config.max_speed_m_s
        if isinstance(raw, tuple):
            return float(raw[0]), float(raw[1])
        v = float(raw)
        return v, v

    def _accel_command(
        self,
        position_error: np.ndarray,
        velocity: np.ndarray,
        velocity_reference: np.ndarray,
        accel_reference: np.ndarray,
    ) -> np.ndarray:
        if self.config.velocity_cascade:
            v_cmd = (
                velocity_reference
                + self.config.kp * position_error
                + self.config.ki * self.integral
            ).copy()
            vx_max, vy_max = self._speed_limits()
            if vx_max > 1e-9:
                v_cmd[0] = float(np.clip(v_cmd[0], -vx_max, vx_max))
            if vy_max > 1e-9:
                v_cmd[1] = float(np.clip(v_cmd[1], -vy_max, vy_max))
            return self.config.kd * (v_cmd - velocity) + accel_reference
        velocity_error = velocity_reference - velocity
        return (
            self.config.kp * position_error
            + self.config.kd * velocity_error
            + self.config.ki * self.integral
            + accel_reference
        )

    def _filter_lateral_accel(self, accel_command: np.ndarray, dt: float) -> np.ndarray:
        """LPF Y accel only — X stays raw so approach track is not phase-lagged."""
        hz = float(self.config.lateral_accel_lpf_hz)
        if hz <= 1e-9 or dt <= 0.0:
            return accel_command
        alpha = float(dt * (2.0 * np.pi * hz))
        alpha = float(np.clip(alpha / (1.0 + alpha), 0.0, 1.0))
        out = accel_command.copy()
        if not self._accel_lpf_init:
            self._accel_lpf[1] = out[1]
            self._accel_lpf_init = True
        else:
            self._accel_lpf[1] = (1.0 - alpha) * self._accel_lpf[1] + alpha * out[1]
        out[1] = self._accel_lpf[1]
        return out

    def _apply_tilt_limit(
        self,
        accel_command: np.ndarray,
        gravity: np.ndarray,
        *,
        position_error: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        g = max(abs(float(gravity[2])), 1e-3)
        max_horiz = np.tan(np.deg2rad(self.config.max_tilt_deg)) * g
        tilt_y = self.config.max_tilt_y_deg
        max_y = (
            max_horiz
            if tilt_y is None
            else np.tan(np.deg2rad(float(tilt_y))) * g
        )
        clipped = accel_command.copy()
        if bool(getattr(self.config, "prefer_lateral_y", False)):
            # Shared tilt cone: Y first (capped by max_y), X gets leftover.
            ay = float(np.clip(accel_command[1], -max_y, max_y))
            remain = float(np.sqrt(max(max_horiz * max_horiz - ay * ay, 0.0)))
            ax = float(np.clip(accel_command[0], -remain, remain))
            clipped[0] = ax
            clipped[1] = ay
        else:
            clipped[0] = float(np.clip(accel_command[0], -max_horiz, max_horiz))
            clipped[1] = float(np.clip(accel_command[1], -max_y, max_y))
        # Anti-windup: freeze / bleed XY integral when horizontal accel saturates
        # in the same direction as the position error.
        if dt > 0.0:
            for axis in (0, 1):
                if abs(float(clipped[axis]) - float(accel_command[axis])) > 1e-9:
                    if position_error[axis] * self.integral[axis] > 0.0:
                        self.integral[axis] *= 0.85
        return clipped

    def attitude_reference(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        position_reference: np.ndarray,
        velocity_reference: np.ndarray | None = None,
        *,
        accel_reference: np.ndarray | None = None,
        gravity: np.ndarray | None = None,
        heading_rad: float | None = None,
        dt: float = 0.0,
    ) -> np.ndarray:
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        position_reference = np.asarray(position_reference, dtype=float)
        velocity_reference = (
            np.zeros(3, dtype=float)
            if velocity_reference is None
            else np.asarray(velocity_reference, dtype=float)
        )
        accel_reference = (
            np.zeros(3, dtype=float)
            if accel_reference is None
            else np.asarray(accel_reference, dtype=float)
        )
        gravity = (
            np.array([0.0, 0.0, -9.81], dtype=float)
            if gravity is None
            else np.asarray(gravity, dtype=float)
        )

        raw_error = position_reference - position
        position_error = self._shaped_position_error(raw_error)
        self._update_integral(position_error, float(dt))
        accel_command = self._accel_command(
            position_error,
            velocity,
            velocity_reference,
            accel_reference,
        )
        # Soft plane priority: when off the xz plane, give roll room by
        # dialing back the along-track (X) accel demand.
        y_gate = float(getattr(self.config, "lateral_priority_m", 0.0) or 0.0)
        if y_gate > 0.0:
            y_err = abs(float(position_error[1]))
            if y_err > y_gate:
                # Full X derate by |y| = y_gate + 0.25 m on the loft and the
                # staging hold. Once the reference is the pad, keep enough X
                # to close a still-open downrange gap before the ground.
                scale = float(np.clip(1.0 - (y_err - y_gate) / 0.25, 0.15, 1.0))
                ref = np.asarray(position_reference, dtype=float).reshape(3)
                pad_slide = (
                    abs(float(ref[0])) < 0.25
                    and abs(float(ref[1])) < 0.25
                    and float(ref[2]) < 0.5
                )
                applied = scale
                if pad_slide:
                    x_err = abs(float(position_error[0]))
                    x_floor = float(
                        np.clip(0.15 + max(x_err - 0.5, 0.0) / 1.5, 0.15, 0.90)
                    )
                    applied = max(scale, x_floor)
                accel_command = accel_command.copy()
                accel_command[0] *= applied
        accel_command = self._filter_lateral_accel(accel_command, float(dt))
        accel_command = self._apply_tilt_limit(
            accel_command,
            gravity,
            position_error=position_error,
            dt=float(dt),
        )

        thrust_accel = accel_command - gravity
        if thrust_accel[2] < self.config.min_upward_accel:
            thrust_accel[2] = self.config.min_upward_accel

        if heading_rad is None:
            return quaternion_from_z_to_direction(thrust_accel)
        return quaternion_from_thrust_and_heading(thrust_accel, float(heading_rad))

    def attitude_reference_velocity_aim(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        position_reference: np.ndarray,
        *,
        gravity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Point thrust so velocity aligns with the waypoint line-of-sight.

        Does not use planar XY position error. Desired velocity is the current
        (or minimum) speed along ``position_reference - position``; acceleration
        closes that velocity error with a tilt limit for authority.
        """
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        position_reference = np.asarray(position_reference, dtype=float)
        gravity = (
            np.array([0.0, 0.0, -9.81], dtype=float)
            if gravity is None
            else np.asarray(gravity, dtype=float)
        )

        los = position_reference - position
        distance = float(np.linalg.norm(los))
        if distance < 1e-9:
            thrust_accel = -gravity
            if thrust_accel[2] < self.config.min_upward_accel:
                thrust_accel[2] = self.config.min_upward_accel
            return quaternion_from_z_to_direction(thrust_accel)

        los_hat = los / distance
        speed = float(np.linalg.norm(velocity))
        v_des = los_hat * max(speed, self.config.velocity_aim_min_speed)
        accel_command = self.config.velocity_align_kp * (v_des - velocity)

        max_horiz = np.tan(np.deg2rad(self.config.max_tilt_deg)) * max(
            abs(float(gravity[2])), 1e-3
        )
        accel_command[:2] = np.clip(accel_command[:2], -max_horiz, max_horiz)

        thrust_accel = accel_command - gravity
        if thrust_accel[2] < self.config.min_upward_accel:
            thrust_accel[2] = self.config.min_upward_accel

        return quaternion_from_z_to_direction(thrust_accel)


@dataclass
class AttitudePIDConfig:
    kp: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 1.5], dtype=float))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    kd: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.25], dtype=float))
    torque_limit: float | None = 5.0
    integral_limit: float | None = 2.0

    def __post_init__(self) -> None:
        self.kp = np.asarray(self.kp, dtype=float)
        self.ki = np.asarray(self.ki, dtype=float)
        self.kd = np.asarray(self.kd, dtype=float)
        if self.kp.shape != (3,) or self.ki.shape != (3,) or self.kd.shape != (3,):
            raise ValueError("PID gains must be length-3 vectors")


class AttitudePIDController:
    def __init__(self, config: AttitudePIDConfig | None = None) -> None:
        self.config = config or AttitudePIDConfig()
        self.integral = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self.integral[:] = 0.0

    def update(
        self,
        quaternion: np.ndarray,
        omega: np.ndarray,
        quaternion_reference: np.ndarray,
        dt: float,
        *,
        omega_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        q = normalize_quaternion(quaternion)
        q_ref = normalize_quaternion(quaternion_reference)
        omega = np.asarray(omega, dtype=float)
        omega_ref = (
            np.zeros(3, dtype=float)
            if omega_reference is None
            else np.asarray(omega_reference, dtype=float)
        )

        attitude_error = quaternion_error_vector(q, q_ref)
        rate_error = omega - omega_ref

        self.integral += attitude_error * dt
        if self.config.integral_limit is not None:
            limit = float(self.config.integral_limit)
            self.integral = np.clip(self.integral, -limit, limit)

        torque = (
            -self.config.kp * attitude_error
            -self.config.kd * rate_error
            -self.config.ki * self.integral
        )

        if self.config.torque_limit is not None:
            limit = float(self.config.torque_limit)
            raw = torque.copy()
            torque = np.clip(torque, -limit, limit)
            # Bleed integral on axes that hit the torque rail.
            for axis in range(3):
                if abs(float(torque[axis]) - float(raw[axis])) > 1e-9:
                    if attitude_error[axis] * self.integral[axis] > 0.0:
                        self.integral[axis] *= 0.90
        return torque


@dataclass
class AltitudeForceConfig:
    """Altitude → axial force target for the axial-force mixer."""

    kp: float = 12.0
    ki: float = 1.5
    kd: float = 8.0
    integral_limit: float | None = 20.0
    force_min: float = 0.0
    force_max: float | None = None
    # Floor on axial command as a fraction of live available thrust (0–1).
    # 0 disables. Applied after the altitude loop so a 0.25 throttle hold
    # still leaves room to climb or brake.
    throttle_min: float = 0.0
    # Optional feedforward override; None → mass * |g| at runtime.
    hover_force: float | None = None

    def __post_init__(self) -> None:
        self.kp = float(self.kp)
        self.ki = float(self.ki)
        self.kd = float(self.kd)
        self.throttle_min = float(np.clip(float(self.throttle_min), 0.0, 1.0))


class AltitudeForceController:
    """Close altitude with a body-axial force command for the mixer."""

    def __init__(self, config: AltitudeForceConfig | None = None) -> None:
        self.config = config or AltitudeForceConfig()
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def update(
        self,
        altitude: float,
        vertical_velocity: float,
        altitude_reference: float,
        dt: float,
        *,
        mass: float,
        gravity: np.ndarray | None = None,
        vertical_velocity_reference: float | None = None,
        vertical_accel_reference: float | None = None,
    ) -> float:
        gravity = (
            np.array([0.0, 0.0, -9.81], dtype=float)
            if gravity is None
            else np.asarray(gravity, dtype=float)
        )
        hover_force = (
            float(self.config.hover_force)
            if self.config.hover_force is not None
            else float(mass) * float(np.linalg.norm(gravity))
        )

        altitude_error = float(altitude_reference) - float(altitude)
        self.integral += altitude_error * dt
        if self.config.integral_limit is not None:
            limit = float(self.config.integral_limit)
            self.integral = float(np.clip(self.integral, -limit, limit))

        vertical_velocity_reference = (
            0.0
            if vertical_velocity_reference is None
            else float(vertical_velocity_reference)
        )
        vertical_rate_error = float(vertical_velocity) - vertical_velocity_reference
        accel_ff = 0.0 if vertical_accel_reference is None else float(vertical_accel_reference)

        force_cmd = (
            hover_force
            + self.config.kp * altitude_error
            - self.config.kd * vertical_rate_error
            + self.config.ki * self.integral
            + float(mass) * accel_ff
        )
        force_cmd = max(float(self.config.force_min), force_cmd)
        if self.config.force_max is not None:
            force_cmd = min(float(self.config.force_max), force_cmd)
        return float(force_cmd)
