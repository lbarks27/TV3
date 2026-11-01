#include "tvc/motors.hpp"

#include "tvc/math.hpp"

// Host and NuttX both provide C stdio for SD / local CSV curves.
#include <stdio.h>

namespace tvc {

float MotorCurve::thrust_at(float time_s) const
{
	if (n_samples < 2) {
		return 0.f;
	}
	const float burn_t = time_s - ignition_delay_s;
	if (burn_t < times_s[0] || burn_t > times_s[n_samples - 1]) {
		return 0.f;
	}
	for (int i = 0; i < n_samples - 1; i++) {
		if (burn_t >= times_s[i] && burn_t <= times_s[i + 1]) {
			const float dt = times_s[i + 1] - times_s[i];
			if (dt <= 1e-12f) {
				return thrust_n[i];
			}
			const float a = (burn_t - times_s[i]) / dt;
			return thrust_n[i] * (1.f - a) + thrust_n[i + 1] * a;
		}
	}
	return 0.f;
}

void thrusts_at(const MotorCurve curves[kMaxMounts], float time_s, float out[kMaxMounts])
{
	for (int i = 0; i < kMaxMounts; i++) {
		out[i] = curves[i].valid() ? curves[i].thrust_at(time_s) : 0.f;
	}
}

bool load_motor_csv(const char *path, MotorCurve &curve, float ignition_delay_s)
{
	if (!path) {
		return false;
	}

	FILE *f = fopen(path, "r");

	if (!f) {
		return false;
	}

	curve.n_samples = 0;
	curve.ignition_delay_s = ignition_delay_s;
	char line[256];

	while (fgets(line, sizeof(line), f)) {
		float t = 0.f, th = 0.f;

		if (sscanf(line, "%f , %f", &t, &th) == 2 ||
		    sscanf(line, "%f,%f", &t, &th) == 2 ||
		    sscanf(line, "%f %f", &t, &th) == 2) {
			if (curve.n_samples < kMaxMotorSamples) {
				curve.times_s[curve.n_samples] = t;
				curve.thrust_n[curve.n_samples] = th;
				curve.n_samples++;
			}
		}
	}

	fclose(f);

	if (curve.n_samples >= 2) {
		curve.burn_time_s = curve.times_s[curve.n_samples - 1];
	}

	return curve.n_samples >= 2;
}

} // namespace tvc
