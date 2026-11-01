#pragma once

#include "tvc/types.hpp"

namespace tvc {

Quat q_normalize(const Quat &q);
Quat q_conj(const Quat &q);
Quat q_mul(const Quat &a, const Quat &b);
Quat q_from_euler(float roll, float pitch, float yaw);
void q_to_euler(const Quat &q, float &roll, float &pitch, float &yaw);

/** Body-frame attitude error vector from q_ref to q_current (Prototype). */
Vec3 q_error_vector(const Quat &q_current, const Quat &q_reference);

Quat q_from_axis_angle(const Vec3 &axis, float angle_rad);
Quat q_from_z_to_direction(const Vec3 &direction);
Quat q_from_thrust_and_heading(const Vec3 &thrust_direction, float heading_rad);

/**
 * Attitude that maps body_up → world thrust_direction (and optional heading).
 * When body_up ≈ +Z this matches q_from_thrust_and_heading / q_from_z_to_direction.
 */
Quat q_from_body_up_and_thrust(const Vec3 &thrust_direction, const Vec3 &body_up,
			       float heading_rad, bool heading_valid);

/** DCM (row-major 3x3 body→world) → quaternion wxyz. */
Quat q_from_dcm(const float R[9]);
void q_to_dcm(const Quat &q, float R[9]);

Vec3 q_rotate(const Quat &q, const Vec3 &v);
Vec3 q_world_to_body(const Quat &q, const Vec3 &v);

} // namespace tvc
