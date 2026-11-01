#include "tvc/geometry.hpp"


namespace tvc {

Vec3 rotate_axis_angle(const Vec3 &axis_in, float angle_rad, const Vec3 &v)
{
	const Vec3 axis = vunit(axis_in);
	if (vnorm(axis) < 1e-12f) {
		return v;
	}

	const float c = cosf(angle_rad);
	const float s = sinf(angle_rad);
	const float c1 = 1.f - c;
	const float x = axis.x;
	const float y = axis.y;
	const float z = axis.z;

	// R * v
	return Vec3{
		(c + x * x * c1) * v.x + (x * y * c1 - z * s) * v.y + (x * z * c1 + y * s) * v.z,
		(y * x * c1 + z * s) * v.x + (c + y * y * c1) * v.y + (y * z * c1 - x * s) * v.z,
		(z * x * c1 - y * s) * v.x + (z * y * c1 + x * s) * v.y + (c + z * z * c1) * v.z
	};
}

Vec3 thrust_direction(float a2, float a1, const Mount &mount)
{
	// a1 is from vertical rest (0 = +Z). base_dir is inward, so kinematic
	// nested rotation is π/2 − a1.
	const float a1_kin = 1.5707963267948966f - a1;
	const Vec3 after_primary = rotate_axis_angle(mount.axis2, a2, mount.base_dir);
	const Vec3 nested_axis1 = vunit(rotate_axis_angle(mount.axis2, a2, mount.axis1));
	return vunit(rotate_axis_angle(nested_axis1, a1_kin, after_primary));
}

void gimbal_aim_direction(const Mount &mount, const Vec3 &desired, float out_a2_a1[2])
{
	const Vec3 d = vunit(desired);
	float best_a2 = mount.rest_angles[0];
	float best_a1 = mount.rest_angles[1];
	float best_dot = vdot(thrust_direction(best_a2, best_a1, mount), d);
	float a2_lo = mount.min_angles[0];
	float a2_hi = mount.max_angles[0];
	float a1_lo = mount.min_angles[1];
	float a1_hi = mount.max_angles[1];
	for (int pass = 0; pass < 4; pass++) {
		for (int i = 0; i < 9; i++) {
			const float a2 = a2_lo + (a2_hi - a2_lo) * (float)i / 8.f;
			for (int j = 0; j < 9; j++) {
				const float a1 = a1_lo + (a1_hi - a1_lo) * (float)j / 8.f;
				const float dot = vdot(thrust_direction(a2, a1, mount), d);
				if (dot > best_dot) {
					best_dot = dot;
					best_a2 = a2;
					best_a1 = a1;
				}
			}
		}
		const float s2 = 0.35f * fmaxf2(a2_hi - a2_lo, 1e-6f);
		const float s1 = 0.35f * fmaxf2(a1_hi - a1_lo, 1e-6f);
		a2_lo = fmaxf2(mount.min_angles[0], best_a2 - s2);
		a2_hi = fminf2(mount.max_angles[0], best_a2 + s2);
		a1_lo = fmaxf2(mount.min_angles[1], best_a1 - s1);
		a1_hi = fminf2(mount.max_angles[1], best_a1 + s1);
	}
	out_a2_a1[0] = best_a2;
	out_a2_a1[1] = best_a1;
}

} // namespace tvc
