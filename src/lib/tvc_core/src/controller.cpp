#include "tvc/controller.hpp"

#include "tvc/math.hpp"

namespace tvc {

AttitudePIDController::AttitudePIDController(const AttitudePIDConfig &cfg)
	: _cfg(cfg)
{
	reset();
}

void AttitudePIDController::set_config(const AttitudePIDConfig &cfg) { _cfg = cfg; }

void AttitudePIDController::reset()
{
	_integral[0] = _integral[1] = _integral[2] = 0.f;
}

Vec3 AttitudePIDController::update(const Quat &q, const Vec3 &omega, const Quat &q_ref, float dt,
				   const Vec3 *omega_ref)
{
	const Vec3 att_err = q_error_vector(q_normalize(q), q_normalize(q_ref));
	const Vec3 omega_r = omega_ref ? *omega_ref : Vec3{0.f, 0.f, 0.f};
	const Vec3 rate_err = vsub(omega, omega_r);

	if (dt > 0.f) {
		_integral[0] += att_err.x * dt;
		_integral[1] += att_err.y * dt;
		_integral[2] += att_err.z * dt;
		if (_cfg.integral_limit >= 0.f) {
			_integral[0] = clampf(_integral[0], -_cfg.integral_limit, _cfg.integral_limit);
			_integral[1] = clampf(_integral[1], -_cfg.integral_limit, _cfg.integral_limit);
			_integral[2] = clampf(_integral[2], -_cfg.integral_limit, _cfg.integral_limit);
		}
	}

	Vec3 torque{
		-_cfg.kp[0] * att_err.x - _cfg.kd[0] * rate_err.x - _cfg.ki[0] * _integral[0],
		-_cfg.kp[1] * att_err.y - _cfg.kd[1] * rate_err.y - _cfg.ki[1] * _integral[1],
		-_cfg.kp[2] * att_err.z - _cfg.kd[2] * rate_err.z - _cfg.ki[2] * _integral[2]
	};

	if (_cfg.torque_limit >= 0.f) {
		const Vec3 raw = torque;
		torque.x = clampf(torque.x, -_cfg.torque_limit, _cfg.torque_limit);
		torque.y = clampf(torque.y, -_cfg.torque_limit, _cfg.torque_limit);
		torque.z = clampf(torque.z, -_cfg.torque_limit, _cfg.torque_limit);
		if (fabsf2(torque.x - raw.x) > 1e-9f && att_err.x * _integral[0] > 0.f) {
			_integral[0] *= 0.90f;
		}
		if (fabsf2(torque.y - raw.y) > 1e-9f && att_err.y * _integral[1] > 0.f) {
			_integral[1] *= 0.90f;
		}
		if (fabsf2(torque.z - raw.z) > 1e-9f && att_err.z * _integral[2] > 0.f) {
			_integral[2] *= 0.90f;
		}
	}
	return torque;
}

PositionPIDController::PositionPIDController(const PositionPIDConfig &cfg)
	: _cfg(cfg)
{
	reset();
}

void PositionPIDController::set_config(const PositionPIDConfig &cfg) { _cfg = cfg; }

void PositionPIDController::set_body_up(const Vec3 &body_up)
{
	const float n = vnorm(body_up);
	if (n < 1e-12f) {
		_body_up = Vec3{0.f, 0.f, 1.f};
		return;
	}
	_body_up = vscale(body_up, 1.f / n);
}

void PositionPIDController::reset()
{
	_integral[0] = _integral[1] = _integral[2] = 0.f;
	_accel_lpf[0] = _accel_lpf[1] = 0.f;
	_accel_lpf_init = false;
}

Quat PositionPIDController::attitude_reference(
	const Vec3 &position,
	const Vec3 &velocity,
	const Vec3 &position_reference,
	const Vec3 &velocity_reference,
	const Vec3 &gravity,
	float heading_rad,
	float dt,
	bool heading_valid)
{
	Vec3 pos_err = vsub(position_reference, position);

	// Soft-clamp XY error so step waypoints do not explode v_cmd.
	if (_cfg.max_position_error_m > 1e-6f) {
		const float xy = sqrtf(pos_err.x * pos_err.x + pos_err.y * pos_err.y);
		if (xy > _cfg.max_position_error_m) {
			const float s = _cfg.max_position_error_m / xy;
			pos_err.x *= s;
			pos_err.y *= s;
		}
	}
	if (_cfg.max_position_error_y_m > 1e-6f) {
		pos_err.y = clampf(pos_err.y, -_cfg.max_position_error_y_m, _cfg.max_position_error_y_m);
	}

	if (dt > 0.f) {
		const float xy = sqrtf(pos_err.x * pos_err.x + pos_err.y * pos_err.y);
		if (_cfg.integral_start_m > 1e-6f && xy > _cfg.integral_start_m) {
			_integral[0] *= 0.90f;
			_integral[1] *= 0.90f;
			_integral[2] = 0.f;
		} else {
			_integral[0] += pos_err.x * dt;
			_integral[1] += pos_err.y * dt;
			_integral[2] = 0.f;
			if (_cfg.integral_limit >= 0.f) {
				_integral[0] = clampf(_integral[0], -_cfg.integral_limit, _cfg.integral_limit);
				_integral[1] = clampf(_integral[1], -_cfg.integral_limit, _cfg.integral_limit);
			}
		}
	}

	Vec3 accel{};
	if (_cfg.velocity_cascade) {
		Vec3 v_cmd{
			velocity_reference.x + _cfg.kp[0] * pos_err.x + _cfg.ki[0] * _integral[0],
			velocity_reference.y + _cfg.kp[1] * pos_err.y + _cfg.ki[1] * _integral[1],
			velocity_reference.z + _cfg.kp[2] * pos_err.z + _cfg.ki[2] * _integral[2]
		};
		if (_cfg.max_speed_y_m_s >= 0.f) {
			if (_cfg.max_speed_m_s > 1e-6f) {
				v_cmd.x = clampf(v_cmd.x, -_cfg.max_speed_m_s, _cfg.max_speed_m_s);
			}
			if (_cfg.max_speed_y_m_s > 1e-6f) {
				v_cmd.y = clampf(v_cmd.y, -_cfg.max_speed_y_m_s, _cfg.max_speed_y_m_s);
			}
		} else if (_cfg.max_speed_m_s > 1e-6f) {
			const float vxy = sqrtf(v_cmd.x * v_cmd.x + v_cmd.y * v_cmd.y);
			if (vxy > _cfg.max_speed_m_s) {
				const float s = _cfg.max_speed_m_s / vxy;
				v_cmd.x *= s;
				v_cmd.y *= s;
			}
		}
		accel = Vec3{
			_cfg.kd[0] * (v_cmd.x - velocity.x),
			_cfg.kd[1] * (v_cmd.y - velocity.y),
			_cfg.kd[2] * (v_cmd.z - velocity.z)
		};
	} else {
		const Vec3 vel_err = vsub(velocity_reference, velocity);
		accel = Vec3{
			_cfg.kp[0] * pos_err.x + _cfg.kd[0] * vel_err.x + _cfg.ki[0] * _integral[0],
			_cfg.kp[1] * pos_err.y + _cfg.kd[1] * vel_err.y + _cfg.ki[1] * _integral[1],
			_cfg.kp[2] * pos_err.z + _cfg.kd[2] * vel_err.z + _cfg.ki[2] * _integral[2]
		};
	}

	if (_cfg.lateral_priority_m > 1e-6f) {
		const float y_err = fabsf2(pos_err.y);
		if (y_err > _cfg.lateral_priority_m) {
			// Full X derate by |y| = gate + 0.25 m. Keep that derate on the
			// loft and the staging hold. Once the reference is the pad, a
			// still-open downrange gap has to keep moving or the dive meets
			// the ground short of the sticks.
			const float scale = clampf(
				1.f - (y_err - _cfg.lateral_priority_m) / 0.25f, 0.15f, 1.f);
			const bool pad_slide = fabsf2(position_reference.x) < 0.25f &&
					       fabsf2(position_reference.y) < 0.25f &&
					       position_reference.z < 0.5f;
			float applied = scale;
			if (pad_slide) {
				const float x_err = fabsf2(pos_err.x);
				const float x_floor = clampf(
					0.15f + fmaxf2(x_err - 0.5f, 0.f) / 1.5f, 0.15f, 0.90f);
				applied = fmaxf2(scale, x_floor);
			}
			accel.x *= applied;
		}
	}

	// First-order LPF on Y lateral accel only (before tilt sat). X stays
	// unfiltered so the approach track is not phase-lagged; Y is the chatter axis.
	if (_cfg.lateral_accel_lpf_hz > 1e-9f && dt > 0.f) {
		float alpha = dt * (2.f * 3.14159265359f * _cfg.lateral_accel_lpf_hz);
		alpha = clampf(alpha / (1.f + alpha), 0.f, 1.f);
		if (!_accel_lpf_init) {
			_accel_lpf[0] = accel.x;
			_accel_lpf[1] = accel.y;
			_accel_lpf_init = true;
		} else {
			_accel_lpf[1] = (1.f - alpha) * _accel_lpf[1] + alpha * accel.y;
		}
		accel.y = _accel_lpf[1];
	}

	const float gabs = fmaxf2(fabsf2(gravity.z), 1e-3f);
	const float max_horiz = tanf(deg2rad(_cfg.max_tilt_deg)) * gabs;
	const float tilt_y_deg = (_cfg.max_tilt_y_deg >= 0.f) ? _cfg.max_tilt_y_deg : _cfg.max_tilt_deg;
	const float max_y = tanf(deg2rad(tilt_y_deg)) * gabs;
	const float ax0 = accel.x;
	const float ay0 = accel.y;
	if (_cfg.prefer_lateral_y) {
		accel.y = clampf(accel.y, -max_y, max_y);
		const float remain = sqrtf(fmaxf2(max_horiz * max_horiz - accel.y * accel.y, 0.f));
		accel.x = clampf(accel.x, -remain, remain);
	} else {
		accel.x = clampf(accel.x, -max_horiz, max_horiz);
		accel.y = clampf(accel.y, -max_y, max_y);
	}
	if (dt > 0.f) {
		if (fabsf2(accel.x - ax0) > 1e-9f && pos_err.x * _integral[0] > 0.f) {
			_integral[0] *= 0.85f;
		}
		if (fabsf2(accel.y - ay0) > 1e-9f && pos_err.y * _integral[1] > 0.f) {
			_integral[1] *= 0.85f;
		}
	}

	Vec3 thrust_accel = vsub(accel, gravity);
	if (thrust_accel.z < _cfg.min_upward_accel) {
		thrust_accel.z = _cfg.min_upward_accel;
	}

	return q_from_body_up_and_thrust(thrust_accel, _body_up, heading_rad, heading_valid);
}

AltitudeForceController::AltitudeForceController(const AltitudeForceConfig &cfg)
	: _cfg(cfg)
{
	reset();
}

void AltitudeForceController::set_config(const AltitudeForceConfig &cfg) { _cfg = cfg; }
void AltitudeForceController::reset() { _integral = 0.f; }

AltitudeForceResult AltitudeForceController::update(float altitude, float vz, float alt_ref,
						    float dt, float mass, float g_abs,
						    float vz_ref, float az_ff)
{
	AltitudeForceResult out{};
	const float m = (mass > 1e-3f) ? mass : 1e-3f;
	const float alt_err = alt_ref - altitude;
	if (dt > 0.f) {
		_integral += alt_err * dt;
		if (_cfg.integral_limit >= 0.f) {
			_integral = clampf(_integral, -_cfg.integral_limit, _cfg.integral_limit);
		}
	}
	const float rate_err = vz - vz_ref;
	// Force-domain outer gains → throttle-like vertical accel demand.
	const float force_corr = _cfg.kp * alt_err - _cfg.kd * rate_err + _cfg.ki * _integral;
	float az_cmd = force_corr / m + az_ff;
	az_cmd = clampf(az_cmd, _cfg.az_min, _cfg.az_max);
	float force = m * (g_abs + az_cmd);
	force = fmaxf2(force, _cfg.force_min);
	if (_cfg.force_max >= 0.f) {
		force = fminf2(force, _cfg.force_max);
	}
	out.force_n = force;
	out.az_cmd = az_cmd;
	return out;
}

} // namespace tvc
