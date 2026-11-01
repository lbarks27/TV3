/**
 * @file host_api.h
 *
 * C ABI for host SIH / tools to call firmware tvc_core controllers.
 * Opaque handles; float32; no PX4 deps.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define TVC_HOST_API __declspec(dllexport)
#else
#  define TVC_HOST_API __attribute__((visibility("default")))
#endif

typedef struct tvc_att_s tvc_att_t;
typedef struct tvc_pos_s tvc_pos_t;
typedef struct tvc_alt_s tvc_alt_t;

/* ---- Attitude PID ------------------------------------------------------- */

TVC_HOST_API tvc_att_t *tvc_att_create(void);
TVC_HOST_API void tvc_att_destroy(tvc_att_t *h);
TVC_HOST_API void tvc_att_reset(tvc_att_t *h);
TVC_HOST_API void tvc_att_set_gains(tvc_att_t *h,
				    const float kp[3], const float ki[3], const float kd[3],
				    float torque_limit, float integral_limit);
TVC_HOST_API void tvc_att_update(tvc_att_t *h,
				 const float q[4], const float omega[3],
				 const float q_ref[4], float dt,
				 const float *omega_ref_nullable,
				 float tau_out[3]);
TVC_HOST_API void tvc_att_get_integral(const tvc_att_t *h, float out[3]);
TVC_HOST_API void tvc_att_set_integral(tvc_att_t *h, const float v[3]);

/* ---- Position PID ------------------------------------------------------- */

TVC_HOST_API tvc_pos_t *tvc_pos_create(void);
TVC_HOST_API void tvc_pos_destroy(tvc_pos_t *h);
TVC_HOST_API void tvc_pos_reset(tvc_pos_t *h);
TVC_HOST_API void tvc_pos_set_gains(tvc_pos_t *h,
				    const float kp[3], const float ki[3], const float kd[3],
				    float max_tilt_deg, float min_upward_accel,
				    float integral_limit, int velocity_cascade,
				    float max_position_error_m, float max_speed_m_s,
				    float integral_start_m, float lateral_priority_m,
				    float max_tilt_y_deg, float max_position_error_y_m,
				    float max_speed_y_m_s, int prefer_lateral_y,
				    float lateral_accel_lpf_hz);
TVC_HOST_API void tvc_pos_set_body_up(tvc_pos_t *h, const float body_up[3]);
TVC_HOST_API void tvc_pos_attitude_reference(
	tvc_pos_t *h,
	const float position[3], const float velocity[3],
	const float position_ref[3], const float velocity_ref[3],
	const float gravity[3], float heading_rad, int heading_valid,
	float dt, float q_out[4]);
TVC_HOST_API void tvc_pos_get_integral(const tvc_pos_t *h, float out[3]);
TVC_HOST_API void tvc_pos_set_integral(tvc_pos_t *h, const float v[3]);

/* ---- Altitude → az_cmd → force (FF) ------------------------------------- */

TVC_HOST_API tvc_alt_t *tvc_alt_create(void);
TVC_HOST_API void tvc_alt_destroy(tvc_alt_t *h);
TVC_HOST_API void tvc_alt_reset(tvc_alt_t *h);
TVC_HOST_API void tvc_alt_set_gains(tvc_alt_t *h,
				    float kp, float ki, float kd,
				    float integral_limit, float force_min, float force_max,
				    float az_min, float az_max);
TVC_HOST_API void tvc_alt_update(tvc_alt_t *h,
				 float altitude, float vz, float alt_ref, float dt,
				 float mass, float g_abs, float vz_ref, float az_ff,
				 float *force_n_out, float *az_cmd_out);
TVC_HOST_API float tvc_alt_get_integral(const tvc_alt_t *h);
TVC_HOST_API void tvc_alt_set_integral(tvc_alt_t *h, float v);

#ifdef __cplusplus
}
#endif
