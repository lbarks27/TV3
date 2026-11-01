#pragma once

#include "tvc/types.hpp"

#include <cstdint>

namespace tvc {

static constexpr int kMaxMotorSamples = 256;

/**
 * Piecewise-linear solid-motor thrust curve (Prototype MotorCurve.thrust_at).
 * Host / boot loader fills tables from SD CSV; flight path is pure lookup.
 */
struct MotorCurve {
	float times_s[kMaxMotorSamples]{};
	float thrust_n[kMaxMotorSamples]{};
	int n_samples{0};
	float ignition_delay_s{0.f};
	float burn_time_s{0.f};

	float thrust_at(float time_s) const;
	bool valid() const { return n_samples >= 2; }
};

/** Evaluate three mount thrusts at mission time (post fire t=0). */
void thrusts_at(const MotorCurve curves[kMaxMounts], float time_s, float out[kMaxMounts]);

/**
 * Load a simple two-column CSV (time_s, thrust_n) into curve.
 * Returns false on parse failure. Host-only convenience (not used on FC if tables preloaded).
 */
bool load_motor_csv(const char *path, MotorCurve &curve, float ignition_delay_s = 0.f);

} // namespace tvc
