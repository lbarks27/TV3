from __future__ import annotations

import numpy as np

Quaternion = np.ndarray


def normalize_quaternion(q: Quaternion) -> Quaternion:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_multiply(q_left: Quaternion, q_right: Quaternion) -> Quaternion:
    w1, x1, y1, z1 = q_left
    w2, x2, y2, z2 = q_right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return normalize_quaternion(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )
    )


def quaternion_to_euler(q: Quaternion) -> tuple[float, float, float]:
    q = normalize_quaternion(q)
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def quaternion_error_vector(q_current: Quaternion, q_reference: Quaternion) -> np.ndarray:
    """Body-frame attitude error vector from q_ref to q_current."""
    q_err = quaternion_multiply(quaternion_conjugate(q_reference), q_current)
    q_err = normalize_quaternion(q_err)
    sign = 1.0 if q_err[0] >= 0.0 else -1.0
    return 2.0 * sign * q_err[1:4]


def rotate_vector(q: Quaternion, vector: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector into the inertial frame."""
    q = normalize_quaternion(q)
    v = np.asarray(vector, dtype=float)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    rotated = quaternion_multiply(
        quaternion_multiply(q, qv),
        quaternion_conjugate(q),
    )
    return rotated[1:4]


def world_to_body(q: Quaternion, vector: np.ndarray) -> np.ndarray:
    """Rotate an inertial-frame vector into the body frame."""
    return rotate_vector(quaternion_conjugate(q), vector)


def quaternion_derivative(q: Quaternion, omega_body: np.ndarray) -> Quaternion:
    """Time derivative of q given body angular rate omega."""
    q = normalize_quaternion(q)
    wx, wy, wz = omega_body
    omega_quat = np.array([0.0, wx, wy, wz], dtype=float)
    return 0.5 * quaternion_multiply(q, omega_quat)


def quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> Quaternion:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / norm
    half = 0.5 * angle_rad
    return normalize_quaternion(
        np.array([np.cos(half), *(axis * np.sin(half))], dtype=float)
    )


def quaternion_from_z_to_direction(direction: np.ndarray) -> Quaternion:
    """Quaternion rotating body +Z to align with direction (inertial frame)."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    target = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(target)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    target = target / norm
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if dot < -1.0 + 1e-9:
        return quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)
    axis = np.cross(z_axis, target)
    angle = np.arccos(dot)
    return quaternion_from_axis_angle(axis, angle)


def quaternion_from_thrust_and_heading(
    thrust_direction: np.ndarray,
    heading_rad: float,
) -> Quaternion:
    """Align body +Z with thrust, then set yaw about that thrust axis.

    Uses the minimum-tilt quaternion to the thrust direction, then a body-Z
    yaw correction so the vehicle's projected heading matches ``heading_rad``.
    This preserves the tilt (translation) command while tracking yaw.
    """
    q_tilt = quaternion_from_z_to_direction(thrust_direction)
    _, _, yaw_tilt = quaternion_to_euler(q_tilt)
    yaw_error = float(heading_rad) - float(yaw_tilt)
    # Choose the shortest yaw correction in [-pi, pi).
    yaw_error = float(((yaw_error + np.pi) % (2.0 * np.pi)) - np.pi)
    q_yaw = quaternion_from_euler(0.0, 0.0, yaw_error)
    return normalize_quaternion(quaternion_multiply(q_tilt, q_yaw))


def quaternion_from_rotation_matrix(rot: np.ndarray) -> Quaternion:
    """Convert a 3×3 body-to-world rotation matrix to a unit quaternion."""
    m = np.asarray(rot, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion(np.array([w, x, y, z], dtype=float))
