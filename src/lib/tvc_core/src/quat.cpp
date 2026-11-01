#include "tvc/quat.hpp"

#include "tvc/math.hpp"

namespace tvc {

Quat q_normalize(const Quat &q)
{
	const float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
	if (n < 1e-12f) {
		return Quat{1.f, 0.f, 0.f, 0.f};
	}
	return Quat{q.w / n, q.x / n, q.y / n, q.z / n};
}

Quat q_conj(const Quat &q)
{
	return Quat{q.w, -q.x, -q.y, -q.z};
}

Quat q_mul(const Quat &a, const Quat &b)
{
	return Quat{
		a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
		a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
		a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
		a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
	};
}

Quat q_from_euler(float roll, float pitch, float yaw)
{
	const float cr = cosf(roll * 0.5f);
	const float sr = sinf(roll * 0.5f);
	const float cp = cosf(pitch * 0.5f);
	const float sp = sinf(pitch * 0.5f);
	const float cy = cosf(yaw * 0.5f);
	const float sy = sinf(yaw * 0.5f);
	return q_normalize(Quat{
		cr * cp * cy + sr * sp * sy,
		sr * cp * cy - cr * sp * sy,
		cr * sp * cy + sr * cp * sy,
		cr * cp * sy - sr * sp * cy
	});
}

void q_to_euler(const Quat &qin, float &roll, float &pitch, float &yaw)
{
	const Quat q = q_normalize(qin);
	const float sinr_cosp = 2.f * (q.w * q.x + q.y * q.z);
	const float cosr_cosp = 1.f - 2.f * (q.x * q.x + q.y * q.y);
	roll = atan2f(sinr_cosp, cosr_cosp);

	const float sinp = 2.f * (q.w * q.y - q.z * q.x);
	if (fabsf2(sinp) >= 1.f) {
		pitch = copysignf2(1.57079632679f, sinp);
	} else {
		pitch = asinf(sinp);
	}

	const float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
	const float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
	yaw = atan2f(siny_cosp, cosy_cosp);
}

Vec3 q_error_vector(const Quat &q_current, const Quat &q_reference)
{
	Quat q_err = q_mul(q_conj(q_reference), q_current);
	q_err = q_normalize(q_err);
	const float sign = q_err.w >= 0.f ? 1.f : -1.f;
	return Vec3{2.f * sign * q_err.x, 2.f * sign * q_err.y, 2.f * sign * q_err.z};
}

Quat q_from_axis_angle(const Vec3 &axis_in, float angle_rad)
{
	const float n = vnorm(axis_in);
	if (n < 1e-12f) {
		return Quat{1.f, 0.f, 0.f, 0.f};
	}
	const Vec3 axis = vscale(axis_in, 1.f / n);
	const float half = 0.5f * angle_rad;
	const float s = sinf(half);
	return q_normalize(Quat{cosf(half), axis.x * s, axis.y * s, axis.z * s});
}

Quat q_from_z_to_direction(const Vec3 &direction)
{
	const Vec3 z{0.f, 0.f, 1.f};
	const float n = vnorm(direction);
	if (n < 1e-12f) {
		return Quat{1.f, 0.f, 0.f, 0.f};
	}
	const Vec3 target = vscale(direction, 1.f / n);
	const float d = clampf(vdot(z, target), -1.f, 1.f);
	if (d > 1.f - 1e-9f) {
		return Quat{1.f, 0.f, 0.f, 0.f};
	}
	if (d < -1.f + 1e-9f) {
		return q_from_axis_angle(Vec3{1.f, 0.f, 0.f}, 3.14159265359f);
	}
	const Vec3 axis = vcross(z, target);
	return q_from_axis_angle(axis, acosf(d));
}

Quat q_from_thrust_and_heading(const Vec3 &thrust_direction, float heading_rad)
{
	const Quat q_tilt = q_from_z_to_direction(thrust_direction);
	float r, p, yaw_tilt;
	q_to_euler(q_tilt, r, p, yaw_tilt);
	float yaw_error = heading_rad - yaw_tilt;
	// wrap to (-pi, pi]
	yaw_error = fmodf(yaw_error + 3.14159265359f, 6.28318530718f);
	if (yaw_error < 0.f) {
		yaw_error += 6.28318530718f;
	}
	yaw_error -= 3.14159265359f;
	const Quat q_yaw = q_from_euler(0.f, 0.f, yaw_error);
	return q_normalize(q_mul(q_tilt, q_yaw));
}

Quat q_from_body_up_and_thrust(const Vec3 &thrust_direction, const Vec3 &body_up,
			       float heading_rad, bool heading_valid)
{
	const Quat q_z = (heading_valid && isfinitef2(heading_rad))
				 ? q_from_thrust_and_heading(thrust_direction, heading_rad)
				 : q_from_z_to_direction(thrust_direction);
	const float n = vnorm(body_up);
	if (n < 1e-12f) {
		return q_z;
	}
	const Vec3 up = vscale(body_up, 1.f / n);
	// body_up ≈ +Z → no extra body rotation
	if (up.x * up.x + up.y * up.y < 1e-8f && up.z > 0.f) {
		return q_z;
	}
	const Quat q_up = q_from_z_to_direction(up);
	return q_normalize(q_mul(q_z, q_conj(q_up)));
}

void q_to_dcm(const Quat &qin, float R[9])
{
	const Quat q = q_normalize(qin);
	const float w = q.w, x = q.x, y = q.y, z = q.z;
	R[0] = 1.f - 2.f * (y * y + z * z);
	R[1] = 2.f * (x * y - z * w);
	R[2] = 2.f * (x * z + y * w);
	R[3] = 2.f * (x * y + z * w);
	R[4] = 1.f - 2.f * (x * x + z * z);
	R[5] = 2.f * (y * z - x * w);
	R[6] = 2.f * (x * z - y * w);
	R[7] = 2.f * (y * z + x * w);
	R[8] = 1.f - 2.f * (x * x + y * y);
}

Quat q_from_dcm(const float R[9])
{
	const float m00 = R[0], m01 = R[1], m02 = R[2];
	const float m10 = R[3], m11 = R[4], m12 = R[5];
	const float m20 = R[6], m21 = R[7], m22 = R[8];
	const float trace = m00 + m11 + m22;
	Quat q;

	if (trace > 0.f) {
		const float s = 0.5f / sqrtf(trace + 1.f);
		q.w = 0.25f / s;
		q.x = (m21 - m12) * s;
		q.y = (m02 - m20) * s;
		q.z = (m10 - m01) * s;
	} else if (m00 > m11 && m00 > m22) {
		const float s = 2.f * sqrtf(1.f + m00 - m11 - m22);
		q.w = (m21 - m12) / s;
		q.x = 0.25f * s;
		q.y = (m01 + m10) / s;
		q.z = (m02 + m20) / s;
	} else if (m11 > m22) {
		const float s = 2.f * sqrtf(1.f + m11 - m00 - m22);
		q.w = (m02 - m20) / s;
		q.x = (m01 + m10) / s;
		q.y = 0.25f * s;
		q.z = (m12 + m21) / s;
	} else {
		const float s = 2.f * sqrtf(1.f + m22 - m00 - m11);
		q.w = (m10 - m01) / s;
		q.x = (m02 + m20) / s;
		q.y = (m12 + m21) / s;
		q.z = 0.25f * s;
	}
	return q_normalize(q);
}

Vec3 q_rotate(const Quat &q, const Vec3 &v)
{
	const Quat qv{0.f, v.x, v.y, v.z};
	const Quat r = q_mul(q_mul(q_normalize(q), qv), q_conj(q_normalize(q)));
	return Vec3{r.x, r.y, r.z};
}

Vec3 q_world_to_body(const Quat &q, const Vec3 &v)
{
	return q_rotate(q_conj(q), v);
}

} // namespace tvc
