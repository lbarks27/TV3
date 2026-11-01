#pragma once

/**
 * Motor thrust curve helpers (portable float32).
 * Per-mount thrust is always from SD CSV curves and/or a synthetic trapezoid.
 */

#include "tvc/math.hpp"

namespace tvc {

/**
 * Simple trapezoid thrust curve for bench without CSV:
 * rise linear 0→peak over rise_s, hold to burn_s - fall_s, fall to 0.
 * time_s is post-fire (ignition_delay already subtracted by caller).
 */
inline float trapezoid_thrust_n(float burn_t, float peak_n, float burn_s, float rise_s, float fall_s)
{
	if (peak_n <= 0.f || burn_s <= 0.f || burn_t < 0.f || burn_t > burn_s) {
		return 0.f;
	}

	rise_s = fmaxf2(1e-3f, rise_s);
	fall_s = fmaxf2(1e-3f, fall_s);

	if (rise_s + fall_s > burn_s) {
		const float mid = 0.5f * burn_s;

		if (burn_t <= mid) {
			return peak_n * (burn_t / mid);
		}

		return peak_n * (1.f - (burn_t - mid) / mid);
	}

	if (burn_t < rise_s) {
		return peak_n * (burn_t / rise_s);
	}

	if (burn_t > burn_s - fall_s) {
		return peak_n * ((burn_s - burn_t) / fall_s);
	}

	return peak_n;
}

} // namespace tvc
