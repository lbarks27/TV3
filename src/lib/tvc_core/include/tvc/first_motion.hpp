#pragma once

// Pad liftoff detector. Cube does not ignite; a disconnected ground system
// lights the grains. Mission commit is first motion while LAUNCH_READY.

#include "tvc/math.hpp"

namespace tvc {

struct FirstMotionConfig {
	float vz_up_mps{0.4f};   // NED-up velocity threshold [m/s]
	float dz_m{0.05f};       // height rise from pad capture [m]
	float az_up_mps2{2.0f};  // NED-up acceleration threshold [m/s^2]
	int consecutive{3};      // hits in a row (safety loop ~50 Hz)
};

struct FirstMotionSample {
	bool vz_valid{false};
	bool z_valid{false};
	bool az_valid{false};
	float vz_up_mps{0.f};      // +up
	float height_agl_m{0.f};
	float az_up_mps2{0.f};     // +up (NED az negated)
};

class FirstMotionDetector {
public:
	void reset()
	{
		_pad_h = 0.f;
		_have_pad = false;
		_hits = 0;
		_latched = false;
	}

	void capture_pad(float height_agl_m)
	{
		if (isfinitef2(height_agl_m)) {
			_pad_h = height_agl_m;
			_have_pad = true;
		} else {
			_have_pad = false;
			_pad_h = 0.f;
		}

		_hits = 0;
		_latched = false;
	}

	// Returns true once motion is confirmed (stays true until reset).
	bool update(const FirstMotionSample &s, const FirstMotionConfig &cfg)
	{
		if (_latched) {
			return true;
		}

		if (!_have_pad && s.z_valid && isfinitef2(s.height_agl_m)) {
			_pad_h = s.height_agl_m;
			_have_pad = true;
		}

		const bool vz_hit = s.vz_valid && isfinitef2(s.vz_up_mps) &&
				    (s.vz_up_mps >= cfg.vz_up_mps);
		const bool dz_hit = _have_pad && s.z_valid && isfinitef2(s.height_agl_m) &&
				    ((s.height_agl_m - _pad_h) >= cfg.dz_m);
		const bool az_hit = s.az_valid && isfinitef2(s.az_up_mps2) &&
				    (s.az_up_mps2 >= cfg.az_up_mps2);

		// Acceleration alone is a pad shock. Liftoff is speed or height.
		(void)az_hit;

		if (vz_hit || dz_hit) {
			_hits++;
		} else {
			_hits = 0;
		}

		const int need = (cfg.consecutive < 1) ? 1 : cfg.consecutive;

		if (_hits >= need) {
			_latched = true;
		}

		return _latched;
	}

	bool latched() const { return _latched; }
	int hits() const { return _hits; }
	bool have_pad() const { return _have_pad; }
	float pad_height_m() const { return _pad_h; }

private:
	float _pad_h{0.f};
	bool _have_pad{false};
	int _hits{0};
	bool _latched{false};
};

} // namespace tvc
