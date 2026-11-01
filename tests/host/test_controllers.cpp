#include <tvc/controller.hpp>

#include <cmath>
#include <cstdio>

static int g_fail = 0;

static void expect(bool c, const char *m)
{
	if (!c) {
		std::fprintf(stderr, "FAIL: %s\n", m);
		g_fail++;
	}
}

int main()
{
	// Attitude: error about X produces negative torque command on roll (Prototype sign)
	{
		tvc::AttitudePIDController pid;
		tvc::Quat q = tvc::q_from_euler(0.1f, 0.f, 0.f);
		tvc::Quat q_ref{1.f, 0.f, 0.f, 0.f};
		tvc::Vec3 w{0.f, 0.f, 0.f};
		tvc::Vec3 tau = pid.update(q, w, q_ref, 0.01f);
		// attitude error positive roll → torque negative (bring back)
		expect(tau.x < 0.f, "roll restoring torque");
	}

	// Altitude cascade: below setpoint → force > hover, az_cmd > 0
	{
		tvc::AltitudeForceController alt;
		const float mass = 2.650f;
		const float hover = mass * 9.81f;
		tvc::AltitudeForceResult r = alt.update(0.f, 0.f, 10.f, 0.01f, mass, 9.81f);
		expect(r.force_n > hover, "climb force above hover");
		expect(r.az_cmd > 0.f, "climb az_cmd positive");
	}

	// Hover at setpoint with vz=0 → az_cmd ≈ 0, F ≈ mg
	{
		tvc::AltitudeForceController alt;
		const float mass = 2.650f;
		const float hover = mass * 9.81f;
		tvc::AltitudeForceResult r = alt.update(5.f, 0.f, 5.f, 0.01f, mass, 9.81f);
		expect(std::fabs(r.az_cmd) < 1e-4f, "hover az_cmd ~0");
		expect(std::fabs(r.force_n - hover) < 1e-3f, "hover force ~mg");
	}

	// Az clamp binds on huge altitude error
	{
		tvc::AltitudeForceConfig cfg;
		cfg.az_max = 2.f;
		cfg.az_min = -2.f;
		tvc::AltitudeForceController alt(cfg);
		const float mass = 2.650f;
		tvc::AltitudeForceResult r = alt.update(0.f, 0.f, 100.f, 0.01f, mass, 9.81f);
		expect(std::fabs(r.az_cmd - 2.f) < 1e-4f, "az_cmd clamped to az_max");
		expect(std::fabs(r.force_n - mass * (9.81f + 2.f)) < 1e-2f, "force from clamped az");
	}

	// Position: offset in +X should tilt thrust (q not identity)
	{
		tvc::PositionPIDController pos;
		tvc::Vec3 p{0.f, 0.f, 10.f};
		tvc::Vec3 v{};
		tvc::Vec3 pref{2.f, 0.f, 10.f};
		tvc::Vec3 vref{};
		tvc::Vec3 g{0.f, 0.f, -9.81f};
		tvc::Quat q = pos.attitude_reference(p, v, pref, vref, g, 0.f, 0.01f, false);
		// thrust direction should have +X component in world
		tvc::Vec3 body_z{0.f, 0.f, 1.f};
		tvc::Vec3 thrust_w = tvc::q_rotate(q, body_z);
		expect(thrust_w.x > 0.05f, "tilt toward +X error");
	}

	// Y off the stick plane must not drop a 2 m downrange correction to the
	// 0.15 floor. That dive hits the ground short of the catch.
	{
		tvc::PositionPIDConfig cfg;
		cfg.kp[0] = 0.42f;
		cfg.kp[1] = 0.22f;
		cfg.kd[0] = 9.5f;
		cfg.kd[1] = 12.f;
		cfg.max_tilt_deg = 11.f;
		cfg.max_tilt_y_deg = 5.f;
		cfg.max_speed_m_s = 1.3f;
		cfg.max_speed_y_m_s = 0.7f;
		cfg.max_position_error_m = 3.f;
		cfg.max_position_error_y_m = 0.5f;
		cfg.lateral_priority_m = 0.05f;
		cfg.prefer_lateral_y = true;
		cfg.velocity_cascade = true;
		tvc::PositionPIDController pos(cfg);
		// Speed is already near the velocity command, so unsaturated X accel
		// is small. The 0.15 Y-priority floor then points thrust almost up
		// (thrust_w.x ≈ -0.02). The pad-slide floor must keep it past -0.10.
		tvc::Vec3 p{2.0f, 0.40f, 1.5f};
		tvc::Vec3 v{-0.70f, 0.f, 0.f};
		tvc::Vec3 pref{0.f, 0.f, -0.19f};
		tvc::Vec3 vref{};
		tvc::Vec3 g{0.f, 0.f, -9.81f};
		tvc::Quat q = pos.attitude_reference(p, v, pref, vref, g, 0.f, 0.01f, false);
		tvc::Vec3 thrust_w = tvc::q_rotate(q, tvc::Vec3{0.f, 0.f, 1.f});
		expect(thrust_w.x < -0.10f, "downrange tilt survives Y priority");
	}

	// Loft brake: the PID asks for zero force. The host floor keeps 20% of thrust,
	// which is the 78° nozzle angle (arccos 0.2) instead of the 90° stop.
	{
		tvc::AltitudeForceConfig cfg;
		cfg.kp = 14.f;
		cfg.kd = 18.f;
		cfg.ki = 0.2f;
		cfg.integral_limit = 6.f;
		cfg.force_max = 70.f;
		tvc::AltitudeForceController alt(cfg);
		const float mass = 2.650f;
		tvc::AltitudeForceResult dumped = alt.update(20.65f, 12.33f, 30.f, 0.02f, mass, 9.81f, 0.f);
		expect(dumped.force_n < 1.f, "loft brake saturates to zero force");
		const float held = tvc::axial_thrust_floor(dumped.force_n, 31.7f, 0.2f);
		expect(std::fabs(held - 6.34f) < 0.05f, "20% floor keeps a few newtons");
		expect(tvc::axial_thrust_floor(0.f, 31.7f, 0.f) == 0.f, "floor stays off at throttle_min 0");
	}

	// After the loft the altitude target follows the vehicle. A high gate
	// sinks at 6.5 m/s. On that rate the same PID holds about one weight.
	{
		tvc::LoftReturnAlt loft = tvc::loft_return_altitude(30.8f, 30.f, 0.f, 1.5f, false);
		expect(std::fabs(loft.z_ref - 30.f) < 1e-3f, "loft target stays 30 m");
		expect(loft.vz_ref == 0.f, "loft commands no sink");

		tvc::LoftReturnAlt crest = tvc::loft_return_altitude(30.87f, 1.5f, 0.f, 1.5f, true);
		expect(std::fabs(crest.z_ref - 30.87f) < 1e-3f, "after the loft the target is the vehicle");
		expect(std::fabs(crest.vz_ref + 6.5f) < 1e-3f, "high return sinks at 6.5 m/s");
		tvc::LoftReturnAlt crest20 = tvc::loft_return_altitude(20.f, 0.5f, 0.f, 0.5f, true, 20.f, false);
		expect(std::fabs(crest20.z_ref - 20.f) < 1e-3f, "20 m crest target is the vehicle");
		expect(std::fabs(crest20.vz_ref + 3.6f) < 1e-3f, "20 m crest sinks at 3.6 m/s");

		tvc::LoftReturnAlt low = tvc::loft_return_altitude(3.18f, 1.5f, -0.4f, 1.5f, true);
		expect(std::fabs(low.z_ref - 1.5f) < 1e-3f, "near the gate the waypoint altitude returns");
		expect(std::fabs(low.vz_ref + 1.f) < 1e-3f, "the last metres sink at 1 m/s");

		tvc::AltitudeForceConfig cfg;
		cfg.kp = 14.f;
		cfg.kd = 18.f;
		cfg.ki = 0.2f;
		cfg.integral_limit = 6.f;
		tvc::AltitudeForceController alt(cfg);
		const float mass = 2.650f;
		tvc::AltitudeForceResult held = alt.update(24.86f, -6.94f, 24.86f, 0.02f, mass, 9.81f, -7.f);
		expect(held.force_n > 20.f && held.force_n < 32.f, "on the sink rate the loop holds weight");
	}

	{
		expect(std::fabs(tvc::landing_tilt_deg(11.f, 7.f, 0.1f, 0.1f, 0.2f, 0.f, 0.f, 0.f, true) - 7.f) < 1e-3f,
		       "over the pad the tilt cap is 7 deg");
		expect(std::fabs(tvc::landing_tilt_deg(11.f, 7.f, 3.f, 0.f, 0.f, 0.f, 0.f, 0.f, true) - 11.f) < 1e-3f,
		       "the approach keeps 11 deg");
		expect(std::fabs(tvc::landing_tilt_deg(11.f, 7.f, 0.1f, 0.f, 0.f, 0.f, 0.f, 0.f, false) - 11.f) < 1e-3f,
		       "the cap waits for the landing phase");
	}

	if (g_fail) {
		return 1;
	}
	std::printf("test_controllers OK\n");
	return 0;
}
