#pragma once

#include "tvc/types.hpp"
#include "tvc/quat.hpp"

namespace tvc {

struct AttitudePIDConfig {
	float kp[3]{12.f, 12.f, 2.5f};
	float ki[3]{0.f, 0.f, 0.f};
	float kd[3]{6.f, 6.f, 1.4f};
	float torque_limit{15.f};    // <0 disable
	float integral_limit{2.f};   // <0 disable
};

class AttitudePIDController {
public:
	explicit AttitudePIDController(const AttitudePIDConfig &cfg = {});
	void set_config(const AttitudePIDConfig &cfg);
	void reset();

	/** Body torque command [N·m] in FLU-sim. */
	Vec3 update(const Quat &q, const Vec3 &omega, const Quat &q_ref, float dt,
		    const Vec3 *omega_ref = nullptr);

	const float *integral() const { return _integral; }

private:
	AttitudePIDConfig _cfg{};
	float _integral[3]{};
};

struct PositionPIDConfig {
	float kp[3]{0.55f, 0.70f, 0.f};
	float kd[3]{4.f, 5.f, 0.f};
	float ki[3]{0.f, 0.015f, 0.f};
	float max_tilt_deg{15.f};
	float max_tilt_y_deg{-1.f}; // <0 → same as max_tilt_deg
	float min_upward_accel{2.f};
	float integral_limit{0.08f}; // <0 disable
	bool velocity_cascade{true};
	float max_position_error_m{3.f}; // 0 disable soft clamp
	float max_position_error_y_m{0.f}; // 0 disable per-axis Y clamp
	float max_speed_m_s{3.5f};       // 0 disable; X when Y limit set
	float max_speed_y_m_s{-1.f};     // <0 → isotropic XY speed sat
	float integral_start_m{1.5f};    // 0 always integrate
	float lateral_priority_m{0.08f}; // 0 disable
	bool prefer_lateral_y{false};    // Y first in shared tilt cone
	// First-order LPF on Y lateral accel only (before tilt sat). X stays
	// unfiltered so the approach track is not phase-lagged; Y is the chatter axis.
	float lateral_accel_lpf_hz{0.f};
};

class PositionPIDController {
public:
	explicit PositionPIDController(const PositionPIDConfig &cfg = {});
	void set_config(const PositionPIDConfig &cfg);
	void reset();
	float max_tilt_deg() const { return _cfg.max_tilt_deg; }
	void set_max_tilt_deg(float deg) { _cfg.max_tilt_deg = deg; }

	/** Body axis that should track thrust_accel (default +Z; typically loaded CoM). */
	void set_body_up(const Vec3 &body_up);
	const Vec3 &body_up() const { return _body_up; }

	Quat attitude_reference(
		const Vec3 &position,
		const Vec3 &velocity,
		const Vec3 &position_reference,
		const Vec3 &velocity_reference,
		const Vec3 &gravity,
		float heading_rad, // NAN = free heading
		float dt,
		bool heading_valid);

private:
	PositionPIDConfig _cfg{};
	float _integral[3]{};
	Vec3 _body_up{0.f, 0.f, 1.f};
	float _accel_lpf[2]{};
	bool _accel_lpf_init{false};
};

struct AltitudeForceConfig {
	float kp{12.f};             // N/m (force-domain outer gains)
	float ki{1.5f};             // N/(m·s)
	float kd{8.f};              // N/(m/s)
	float integral_limit{20.f}; // <0 disable
	float force_min{0.f};
	float force_max{-1.f};      // <0 disable
	float az_min{-9.81f};       // m/s² vertical accel demand (ENU +up)
	float az_max{3.924f};       // ~+0.4 g headroom above hover
};

/** Cascade: altitude/vz → az_cmd → F = m(g + az_cmd) feedforward. */
struct AltitudeForceResult {
	float force_n{0.f};
	float az_cmd{0.f}; // m/s², throttle-like vertical accel demand
};

class AltitudeForceController {
public:
	explicit AltitudeForceController(const AltitudeForceConfig &cfg = {});
	void set_config(const AltitudeForceConfig &cfg);
	void reset();
	void set_integral(float v) { _integral = v; }
	float integral() const { return _integral; }

	AltitudeForceResult update(float altitude, float vz, float alt_ref, float dt,
				   float mass, float g_abs, float vz_ref = 0.f,
				   float az_ff = 0.f);

private:
	AltitudeForceConfig _cfg{};
	float _integral{0.f};
};

/**
 * Altitude reference the force loop should track after the loft.
 * Matches Prototype FlightSimulator: hold the reference on the vehicle and
 * command a sink toward the staging height. A step down to that waypoint
 * saturates the altitude loop and folds the nozzles to the stop.
 * `after_loft` is false while the guide is still holding the loft.
 */
struct LoftReturnAlt {
	float z_ref{0.f};
	float vz_ref{0.f};
};

inline LoftReturnAlt loft_return_altitude(float z, float z_ref, float vz_ref,
					  float gate_z, bool after_loft,
					  float loft_z = 30.f, bool landing = false)
{
	LoftReturnAlt out{z_ref, vz_ref};

	if (!after_loft) {
		return out;
	}

	// Same schedule as Prototype FlightSimulator for this profile.
	// A 20 m crest cannot brake a 7 m/s sink before the slide.
	const bool low_gate = gate_z < 0.8f;
	const float enter = low_gate ? 0.35f : 1.f;

	if (z <= gate_z + enter) {
		return out;
	}

	const float drop = z - gate_z;
	float vz_cmd = -1.f;

	if (low_gate && loft_z < 25.f) {
		vz_cmd = drop > 8.f ? -3.6f : (drop > 1.6f ? -2.4f : -1.f);
	} else if (low_gate) {
		vz_cmd = drop > 14.f ? -6.5f : (drop > 3.5f ? -3.6f : (drop > 0.7f ? -1.2f : -0.45f));
	} else {
		vz_cmd = drop > 14.f ? -6.5f : (drop > 6.f ? -3.f : (drop > 3.f ? -1.5f : -1.f));
	}

	out.vz_ref = fminf2(vz_ref, vz_cmd);
	const float glue = low_gate ? 0.5f : 2.f;

	if (drop > glue && !landing) {
		out.z_ref = z;
	}

	return out;
}

/**
 * Tilt budget once the landing phase is over the pad and slow.
 * `cap_deg` < 0 leaves the profile tilt unchanged. Matches the host sim:
 * 7° only inside 0.55 m and under 0.7 m/s, and only to tighten.
 */
inline float landing_tilt_deg(float profile_tilt_deg, float cap_deg,
			      float x, float y, float vx, float vy,
			      float pad_x, float pad_y, bool landing)
{
	if (!landing || cap_deg < 0.f) {
		return profile_tilt_deg;
	}

	const float dx = x - pad_x;
	const float dy = y - pad_y;
	const float xy = sqrtf(dx * dx + dy * dy);
	const float vxy = sqrtf(vx * vx + vy * vy);

	if (xy < 0.55f && vxy < 0.70f) {
		return fminf2(profile_tilt_deg, cap_deg);
	}

	return profile_tilt_deg;
}

/** Keep at least `throttle_min` of the thrust on hand. 0 disables the floor. */
inline float axial_thrust_floor(float force_n, float available_n, float throttle_min)
{
	if (throttle_min > 0.f && available_n > 0.f) {
		return fmaxf2(force_n, throttle_min * available_n);
	}

	return force_n;
}

} // namespace tvc
