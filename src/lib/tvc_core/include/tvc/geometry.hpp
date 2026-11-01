#pragma once

#include "tvc/types.hpp"

namespace tvc {

/** Rodrigues rotation of vector v about unit axis by angle_rad. */
Vec3 rotate_axis_angle(const Vec3 &axis, float angle_rad, const Vec3 &v);

/**
 * Thrust direction for dual-axis gimbal (matches Prototype geometry.py):
 * primary about axis2 (a2), nested about rotated axis1.
 * a1 is commanded from vertical rest (0 = +Z); internally rotated from inward base_dir.
 */
Vec3 thrust_direction(float a2, float a1, const Mount &mount);

/** Gimbal (a2, a1) that aims thrust along ``desired`` (body frame). */
void gimbal_aim_direction(const Mount &mount, const Vec3 &desired, float out_a2_a1[2]);

} // namespace tvc
