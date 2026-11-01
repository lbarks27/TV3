#include "tvc/host_api.h"

#include "tvc/controller.hpp"

#include <new>

struct tvc_att_s {
	tvc::AttitudePIDController ctrl{tvc::AttitudePIDConfig{}};
};

struct tvc_pos_s {
	tvc::PositionPIDController ctrl{tvc::PositionPIDConfig{}};
};

struct tvc_alt_s {
	tvc::AltitudeForceController ctrl{tvc::AltitudeForceConfig{}};
};

static tvc::Vec3 v3(const float *p)
{
	return tvc::Vec3{p[0], p[1], p[2]};
}

static tvc::Quat q4(const float *p)
{
	return tvc::Quat{p[0], p[1], p[2], p[3]};
}

/* ---- Attitude ----------------------------------------------------------- */

tvc_att_t *tvc_att_create(void)
{
	return new (std::nothrow) tvc_att_t{};
}

void tvc_att_destroy(tvc_att_t *h)
{
	delete h;
}

void tvc_att_reset(tvc_att_t *h)
{
	if (h) {
		h->ctrl.reset();
	}
}

void tvc_att_set_gains(tvc_att_t *h,
		       const float kp[3], const float ki[3], const float kd[3],
		       float torque_limit, float integral_limit)
{
	if (!h) {
		return;
	}
	tvc::AttitudePIDConfig cfg;
	for (int i = 0; i < 3; i++) {
		cfg.kp[i] = kp[i];
		cfg.ki[i] = ki[i];
		cfg.kd[i] = kd[i];
	}
	cfg.torque_limit = torque_limit;
	cfg.integral_limit = integral_limit;
	h->ctrl.set_config(cfg);
}

void tvc_att_update(tvc_att_t *h,
		    const float q[4], const float omega[3],
		    const float q_ref[4], float dt,
		    const float *omega_ref_nullable,
		    float tau_out[3])
{
	if (!h || !tau_out) {
		return;
	}
	tvc::Vec3 w_ref{};
	const tvc::Vec3 *w_ptr = nullptr;
	if (omega_ref_nullable) {
		w_ref = v3(omega_ref_nullable);
		w_ptr = &w_ref;
	}
	const tvc::Vec3 tau = h->ctrl.update(q4(q), v3(omega), q4(q_ref), dt, w_ptr);
	tau_out[0] = tau.x;
	tau_out[1] = tau.y;
	tau_out[2] = tau.z;
}

void tvc_att_get_integral(const tvc_att_t *h, float out[3])
{
	if (!h || !out) {
		return;
	}
	const float *i = h->ctrl.integral();
	out[0] = i[0];
	out[1] = i[1];
	out[2] = i[2];
}

void tvc_att_set_integral(tvc_att_t *h, const float v[3])
{
	(void)h;
	(void)v;
	/* AttitudePID has no public set_integral; reset + unused for SIH. */
}

/* ---- Position ----------------------------------------------------------- */

tvc_pos_t *tvc_pos_create(void)
{
	return new (std::nothrow) tvc_pos_t{};
}

void tvc_pos_destroy(tvc_pos_t *h)
{
	delete h;
}

void tvc_pos_reset(tvc_pos_t *h)
{
	if (h) {
		h->ctrl.reset();
	}
}

void tvc_pos_set_gains(tvc_pos_t *h,
		       const float kp[3], const float ki[3], const float kd[3],
		       float max_tilt_deg, float min_upward_accel,
		       float integral_limit, int velocity_cascade,
		       float max_position_error_m, float max_speed_m_s,
		       float integral_start_m, float lateral_priority_m,
		       float max_tilt_y_deg, float max_position_error_y_m,
		       float max_speed_y_m_s, int prefer_lateral_y,
		       float lateral_accel_lpf_hz)
{
	if (!h) {
		return;
	}
	tvc::PositionPIDConfig cfg;
	for (int i = 0; i < 3; i++) {
		cfg.kp[i] = kp[i];
		cfg.ki[i] = ki[i];
		cfg.kd[i] = kd[i];
	}
	cfg.max_tilt_deg = max_tilt_deg;
	cfg.max_tilt_y_deg = max_tilt_y_deg;
	cfg.min_upward_accel = min_upward_accel;
	cfg.integral_limit = integral_limit;
	cfg.velocity_cascade = velocity_cascade != 0;
	cfg.max_position_error_m = max_position_error_m;
	cfg.max_position_error_y_m = max_position_error_y_m;
	cfg.max_speed_m_s = max_speed_m_s;
	cfg.max_speed_y_m_s = max_speed_y_m_s;
	cfg.integral_start_m = integral_start_m;
	cfg.lateral_priority_m = lateral_priority_m;
	cfg.prefer_lateral_y = prefer_lateral_y != 0;
	cfg.lateral_accel_lpf_hz = lateral_accel_lpf_hz;
	h->ctrl.set_config(cfg);
}

void tvc_pos_set_body_up(tvc_pos_t *h, const float body_up[3])
{
	if (!h || !body_up) {
		return;
	}
	h->ctrl.set_body_up(v3(body_up));
}

void tvc_pos_attitude_reference(
	tvc_pos_t *h,
	const float position[3], const float velocity[3],
	const float position_ref[3], const float velocity_ref[3],
	const float gravity[3], float heading_rad, int heading_valid,
	float dt, float q_out[4])
{
	if (!h || !q_out) {
		return;
	}
	const tvc::Quat q = h->ctrl.attitude_reference(
		v3(position), v3(velocity), v3(position_ref), v3(velocity_ref),
		v3(gravity), heading_rad, dt, heading_valid != 0);
	q_out[0] = q.w;
	q_out[1] = q.x;
	q_out[2] = q.y;
	q_out[3] = q.z;
}

void tvc_pos_get_integral(const tvc_pos_t *h, float out[3])
{
	if (!h || !out) {
		return;
	}
	out[0] = out[1] = out[2] = 0.f;
	(void)h;
}

void tvc_pos_set_integral(tvc_pos_t *h, const float v[3])
{
	(void)h;
	(void)v;
}

/* ---- Altitude ----------------------------------------------------------- */

tvc_alt_t *tvc_alt_create(void)
{
	return new (std::nothrow) tvc_alt_t{};
}

void tvc_alt_destroy(tvc_alt_t *h)
{
	delete h;
}

void tvc_alt_reset(tvc_alt_t *h)
{
	if (h) {
		h->ctrl.reset();
	}
}

void tvc_alt_set_gains(tvc_alt_t *h,
		       float kp, float ki, float kd,
		       float integral_limit, float force_min, float force_max,
		       float az_min, float az_max)
{
	if (!h) {
		return;
	}
	tvc::AltitudeForceConfig cfg;
	cfg.kp = kp;
	cfg.ki = ki;
	cfg.kd = kd;
	cfg.integral_limit = integral_limit;
	cfg.force_min = force_min;
	cfg.force_max = force_max;
	cfg.az_min = az_min;
	cfg.az_max = az_max;
	h->ctrl.set_config(cfg);
}

void tvc_alt_update(tvc_alt_t *h,
		    float altitude, float vz, float alt_ref, float dt,
		    float mass, float g_abs, float vz_ref, float az_ff,
		    float *force_n_out, float *az_cmd_out)
{
	if (!h) {
		return;
	}
	const tvc::AltitudeForceResult r =
		h->ctrl.update(altitude, vz, alt_ref, dt, mass, g_abs, vz_ref, az_ff);
	if (force_n_out) {
		*force_n_out = r.force_n;
	}
	if (az_cmd_out) {
		*az_cmd_out = r.az_cmd;
	}
}

float tvc_alt_get_integral(const tvc_alt_t *h)
{
	return h ? h->ctrl.integral() : 0.f;
}

void tvc_alt_set_integral(tvc_alt_t *h, float v)
{
	if (h) {
		h->ctrl.set_integral(v);
	}
}
