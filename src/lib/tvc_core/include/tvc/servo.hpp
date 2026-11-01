#pragma once

// Gimbal angle [rad] + rest trim [deg] → actuator_servos normalized [-1, 1].
// Logical channels: [M0a2, M0a1, M1a2, M1a1, M2a2, M2a1].

#include "tvc/types.hpp"

namespace tvc {

static constexpr float kServoTrimMaxDeg = 15.f;

// MAIN pin for logical channel (airframe PWM_MAIN_FUNC*).
inline int servo_logical_to_main(int ch)
{
	static constexpr int kMap[kNumAngles] = {1, 5, 2, 6, 3, 7};
	return (ch >= 0 && ch < kNumAngles) ? kMap[ch] : 0;
}

// PX4 maps control -1..+1 to PWM_MAIN_MIN..MAX; 0 is the midpoint.
inline int servo_norm_to_pwm_us(float n, int min_us, int max_us)
{
	n = clampf(n, -1.f, 1.f);
	const float mid = 0.5f * static_cast<float>(min_us + max_us);
	const float half = 0.5f * static_cast<float>(max_us - min_us);
	const float us = mid + n * half;
	return static_cast<int>(us + (us >= 0.f ? 0.5f : -0.5f));
}

// Trim is geometric: +TVC_TRIM on a2/a1 moves rest in the +command direction
// (before TVC_SERVO_REV). force-rest uses a2 = 0 and a1 = TVC_A1_REST (0°).
inline float gimbal_servo_norm(int ch, float ang_rad, float trim_deg, float a2_range_rad,
			       float a1_range_rad, int rev_mask)
{
	const float trim = clampf(trim_deg, -kServoTrimMaxDeg, kServoTrimMaxDeg);
	const float ang = ang_rad + deg2rad(trim);
	float n = 0.f;

	if ((ch % 2) == 0) {
		n = (a2_range_rad > 1e-6f) ? (ang / a2_range_rad) : 0.f;
	} else {
		n = (a1_range_rad > 1e-6f) ? (-ang / a1_range_rad) : 0.f;
	}

	n = clampf(n, -1.f, 1.f);

	if (rev_mask & (1 << ch)) {
		n = -n;
	}

	return n;
}

} // namespace tvc
