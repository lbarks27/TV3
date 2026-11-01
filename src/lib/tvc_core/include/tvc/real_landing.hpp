#pragma once

// Capture-path real_landing, matching Prototype FlightScenario:
// loft over the launch point, hold the staging waypoint, then an ease-out
// onto the pad once the burn gate (or a capture settle) opens.

#include "tvc/types.hpp"

namespace tvc {

struct RealLandingConfig {
	bool has_loft{false};
	Vec3 loft{};
	Vec3 staging{};
	Vec3 pad{};
	float vertical_hold_s{0.f};
	float capture_radius_m{1.2f};
	float capture_max_speed_m_s{1.f};
	float capture_settle_s{0.f};
	float stage_radius_m{1.f};
	float stage_speed_m_s{1.2f};
	float min_approach_time_s{0.5f};
	float descent_duration_s{0.8f};
	float burn_end_s{0.f};
	float pad_remain_s{0.f};
	bool burn_gate{false};
};

struct RealLandingSample {
	float t_s{0.f};
	Vec3 pos{};
	Vec3 vel{};
	bool vel_valid{false};
};

struct RealLandingOutput {
	Vec3 pos{};
	Vec3 vel{};
	int phase{1}; // 1 staging, 2 landing, 3 loft
};

class RealLandingGuide {
public:
	void reset()
	{
		_loft_done = false;
		_landing = false;
		_land_t0 = 0.f;
		_settle_since = -1.f;
		_hold_since = -1.f;
		_track_start = {};
	}

	RealLandingOutput update(const RealLandingSample &s, const RealLandingConfig &cfg)
	{
		RealLandingOutput out{};

		if (!_loft_done) {
			if (!cfg.has_loft) {
				_loft_done = true;
			} else if (loft_finished(s, cfg)) {
				_loft_done = true;
			} else {
				out.pos = cfg.loft;
				out.phase = 3;
				return out;
			}
		}

		if (!_landing && should_start_landing(s, cfg)) {
			_landing = true;
			_land_t0 = s.t_s;
			_track_start = cfg.staging;
			_settle_since = -1.f;
		}

		if (!_landing) {
			out.pos = cfg.staging;
			out.phase = 1;
			return out;
		}

		const float duration = fmaxf2(cfg.descent_duration_s, 0.1f);
		const float tau = clampf((s.t_s - _land_t0) / duration, 0.f, 1.f);
		const float remaining = 1.f - tau;
		const float ease = 1.f - remaining * remaining * remaining;
		const float ds_dtau = 3.f * remaining * remaining;
		const Vec3 delta = vsub(cfg.pad, _track_start);
		out.pos = vadd(_track_start, vscale(delta, ease));
		out.vel = vscale(delta, ds_dtau / duration);
		out.phase = 2;
		return out;
	}

	bool loft_done() const { return _loft_done; }
	bool landing() const { return _landing; }

private:
	bool loft_finished(const RealLandingSample &s, const RealLandingConfig &cfg)
	{
		const float loft_z = cfg.loft.z;
		const float abort_s = fmaxf2(cfg.vertical_hold_s, 0.f) + 4.f
				      + fmaxf2(0.f, (loft_z - 10.f) * 0.2f);

		if (s.t_s >= abort_s) {
			return true;
		}

		const float xy = hypotf(s.pos.x - cfg.loft.x, s.pos.y - cfg.loft.y);
		const bool xy_ok = xy <= fmaxf2(cfg.capture_radius_m, 0.f);
		const bool z_ok = s.pos.z >= loft_z - 1.5f;
		bool slow = false;

		if (cfg.vertical_hold_s > 1e-3f) {
			const float speed = s.vel_valid ? vnorm(s.vel) : 1e6f;
			const float max_speed = fmaxf2(cfg.capture_max_speed_m_s, 1.f);
			slow = s.vel_valid && speed <= max_speed;
		} else {
			slow = s.vel_valid && s.vel.z <= 1.f;
		}

		if (!(xy_ok && z_ok && slow)) {
			_hold_since = -1.f;
			return false;
		}

		if (cfg.vertical_hold_s <= 1e-3f) {
			return true;
		}

		if (_hold_since < 0.f) {
			_hold_since = s.t_s;
		}

		return (s.t_s - _hold_since) >= cfg.vertical_hold_s;
	}

	bool should_start_landing(const RealLandingSample &s, const RealLandingConfig &cfg)
	{
		if (cfg.burn_gate && cfg.burn_end_s > 0.f) {
			const float open_t = fmaxf2(0.f, cfg.burn_end_s - fmaxf2(cfg.pad_remain_s, 0.f));

			if (s.t_s < open_t) {
				_settle_since = -1.f;
				return false;
			}

			// A ground-level stage is the slide itself. The 3 m window
			// opens a slanted path while the vehicle is still high.
			const float z_window = (cfg.staging.z < 0.8f) ? 0.55f : 3.f;

			if (s.pos.z > cfg.staging.z + z_window) {
				return false;
			}

			if (cfg.staging.z < 0.8f && s.vel_valid && s.vel.z < -1.4f) {
				return false;
			}

			const float xy = hypotf(s.pos.x - cfg.staging.x, s.pos.y - cfg.staging.y);
			const float vxy = s.vel_valid
					  ? hypotf(s.vel.x, s.vel.y)
					  : 0.f;
			const bool in_disk = xy <= fmaxf2(cfg.stage_radius_m, 0.05f)
					     && vxy <= cfg.stage_speed_m_s;
			return in_disk || past_staging(s.pos, cfg);
		}

		if (s.t_s < cfg.min_approach_time_s) {
			_settle_since = -1.f;
			return false;
		}

		const float err = vnorm(vsub(s.pos, cfg.staging));
		const float speed = s.vel_valid ? vnorm(s.vel) : 0.f;
		const bool in_gate = err <= fmaxf2(cfg.capture_radius_m, 0.f)
				     && speed <= fmaxf2(cfg.capture_max_speed_m_s, 0.f);

		if (!in_gate) {
			_settle_since = -1.f;
			return false;
		}

		if (_settle_since < 0.f) {
			_settle_since = s.t_s;
		}

		return (s.t_s - _settle_since) >= fmaxf2(cfg.capture_settle_s, 0.f);
	}

	static bool past_staging(const Vec3 &pos, const RealLandingConfig &cfg)
	{
		const float dx = cfg.pad.x - cfg.staging.x;
		const float dy = cfg.pad.y - cfg.staging.y;
		const float span2 = dx * dx + dy * dy;

		if (span2 <= 1e-6f) {
			return false;
		}

		const float u = ((pos.x - cfg.staging.x) * dx + (pos.y - cfg.staging.y) * dy) / span2;
		return u >= -0.05f;
	}

	bool _loft_done{false};
	bool _landing{false};
	float _land_t0{0.f};
	float _settle_since{-1.f};
	float _hold_since{-1.f};
	Vec3 _track_start{};
};

} // namespace tvc
