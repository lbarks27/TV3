#include "tvc/allocator.hpp"
#include "tvc/geometry.hpp"

#include "tvc/math.hpp"

namespace tvc {

namespace {

// Solve G x = b for G kxk symmetric positive definite, k<=3 (Cholesky).
bool solve_spd(const float *G, const float *b, int k, float *x)
{
	float L[9]{};
	for (int i = 0; i < k; i++) {
		for (int j = 0; j <= i; j++) {
			float s = G[i * k + j];
			for (int p = 0; p < j; p++) {
				s -= L[i * k + p] * L[j * k + p];
			}
			if (i == j) {
				if (s <= 1e-12f) {
					return false;
				}
				L[i * k + j] = sqrtf(s);
			} else {
				L[i * k + j] = s / L[j * k + j];
			}
		}
	}
	// forward
	float y[3]{};
	for (int i = 0; i < k; i++) {
		float s = b[i];
		for (int p = 0; p < i; p++) {
			s -= L[i * k + p] * y[p];
		}
		y[i] = s / L[i * k + i];
	}
	// back
	for (int i = k - 1; i >= 0; i--) {
		float s = y[i];
		for (int p = i + 1; p < k; p++) {
			s -= L[p * k + i] * x[p];
		}
		x[i] = s / L[i * k + i];
	}
	return true;
}

} // namespace

TorqueAllocator::TorqueAllocator()
{
	default_ring3_mounts(_mounts);
	_com = default_com();
	_force_axis = default_force_axis();
	set_mounts(_mounts, kMaxMounts);
	_torque_ridge = fmaxf2(_cfg.damping * _cfg.damping, 1e-12f);
}

void TorqueAllocator::set_mounts(const Mount mounts[kMaxMounts], int n)
{
	_n = fminf2(n, kMaxMounts);
	for (int i = 0; i < _n; i++) {
		_mounts[i] = mounts[i];
		_rest[i * 2 + 0] = mounts[i].rest_angles[0];
		_rest[i * 2 + 1] = mounts[i].rest_angles[1];
		_min_a[i * 2 + 0] = mounts[i].min_angles[0];
		_min_a[i * 2 + 1] = mounts[i].min_angles[1];
		_max_a[i * 2 + 0] = mounts[i].max_angles[0];
		_max_a[i * 2 + 1] = mounts[i].max_angles[1];
		_angles[i * 2 + 0] = _rest[i * 2 + 0];
		_angles[i * 2 + 1] = _rest[i * 2 + 1];
	}
}

void TorqueAllocator::set_com(const Vec3 &com) { _com = com; }
void TorqueAllocator::set_force_axis(const Vec3 &axis) { _force_axis = vunit(axis); }

void TorqueAllocator::set_solver(const SolverConfig &cfg)
{
	_cfg = cfg;
	_torque_ridge = fmaxf2(_cfg.damping * _cfg.damping, 1e-12f);
}

void TorqueAllocator::rest_angles_flat(float out[kNumAngles]) const
{
	memcpy(out, _rest, sizeof(float) * kNumAngles);
}

float TorqueAllocator::axial_at_rest(const float *thrusts_in) const
{
	float thrusts[kMaxMounts];
	resolve_thrusts(thrusts_in, thrusts);
	float tau[3], f[3];
	torque_force(_rest, thrusts, tau, f);
	return axial_force(f);
}

void TorqueAllocator::twr_lock_angles_flat(const float *thrusts_in, float weight, float twr,
					   float out[kNumAngles]) const
{
	float thrusts[kMaxMounts];
	resolve_thrusts(thrusts_in, thrusts);
	const float target = twr * fmaxf2(weight, 1e-6f);
	float a1_min = rad2deg(_min_a[1]);
	float a1_max = rad2deg(_max_a[1]);
	const float a1_rest = rad2deg(_rest[1]);
	for (int i = 0; i < _n; i++) {
		a1_min = fminf2(a1_min, rad2deg(_min_a[i * 2 + 1]));
		a1_max = fmaxf2(a1_max, rad2deg(_max_a[i * 2 + 1]));
	}
	auto fax = [this, &thrusts](float a1_deg) {
		float ang[kNumAngles];
		uniform_a1_angles(a1_deg, ang);
		float tau[3], f[3];
		torque_force(ang, thrusts, tau, f);
		return axial_force(f);
	};
	const float f_rest = fax(a1_rest);
	const float f_fold = fax(a1_max);
	if (target >= f_rest - 1e-6f) {
		uniform_a1_angles(a1_rest, out);
		return;
	}
	if (target <= f_fold + 1e-6f) {
		uniform_a1_angles(a1_max, out);
		return;
	}
	float lo = a1_rest;
	float hi = a1_max;
	for (int i = 0; i < 16; i++) {
		const float mid = 0.5f * (lo + hi);
		if (fax(mid) > target) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	uniform_a1_angles(0.5f * (lo + hi), out);
}

void TorqueAllocator::clip_angles(float a[kNumAngles]) const
{
	for (int i = 0; i < _n * 2; i++) {
		a[i] = clampf(a[i], _min_a[i], _max_a[i]);
	}
}

void TorqueAllocator::constrain_for_thrusts(float a[kNumAngles], const float thrusts[kMaxMounts]) const
{
	for (int i = 0; i < _n; i++) {
		if (thrusts[i] <= 1e-9f) {
			a[i * 2 + 0] = _rest[i * 2 + 0];
			a[i * 2 + 1] = _rest[i * 2 + 1];
		}
	}
}

void TorqueAllocator::resolve_thrusts(const float *thrusts, float out[kMaxMounts]) const
{
	for (int i = 0; i < _n; i++) {
		out[i] = thrusts ? thrusts[i] : _mounts[i].thrust;
	}
}

void TorqueAllocator::torque_force(const float a[kNumAngles], const float thrusts[kMaxMounts],
				   float torque[3], float force[3]) const
{
	torque[0] = torque[1] = torque[2] = 0.f;
	force[0] = force[1] = force[2] = 0.f;
	for (int i = 0; i < _n; i++) {
		const Vec3 dir = thrust_direction(a[i * 2 + 0], a[i * 2 + 1], _mounts[i]);
		const Vec3 F = vscale(dir, thrusts[i]);
		force[0] += F.x;
		force[1] += F.y;
		force[2] += F.z;
		const Vec3 r = vsub(_mounts[i].position, _com);
		Vec3 tau = vcross(r, F);
		if (_cfg.invert_side_roll && i != 0) {
			tau.x = -tau.x;
		}
		if (_cfg.invert_yaw) {
			tau.z = -tau.z;
		}
		torque[0] += tau.x;
		torque[1] += tau.y;
		torque[2] += tau.z;
	}
}

float TorqueAllocator::axial_force(const float force[3]) const
{
	return force[0] * _force_axis.x + force[1] * _force_axis.y + force[2] * _force_axis.z;
}

float TorqueAllocator::torque_tolerance(const float target[3]) const
{
	const float tn = sqrtf(target[0] * target[0] + target[1] * target[1] + target[2] * target[2]);
	return fmaxf2(_cfg.tol * 10.f, fmaxf2(5e-3f, 0.03f * fmaxf2(tn, 1.f)));
}

void TorqueAllocator::uniform_a1_angles(float a1_deg, float out[kNumAngles]) const
{
	memcpy(out, _rest, sizeof(float) * kNumAngles);
	const float a1 = deg2rad(a1_deg);
	for (int i = 0; i < _n; i++) {
		out[i * 2 + 0] = 0.f;
		out[i * 2 + 1] = a1;
	}
	clip_angles(out);
}

void TorqueAllocator::solve_a2_step(const float current[kNumAngles], const float target[3],
				    const float thrusts[kMaxMounts], float out[kNumAngles]) const
{
	memcpy(out, current, sizeof(float) * kNumAngles);

	int indices[kMaxMounts];
	int k = 0;
	for (int i = 0; i < _n; i++) {
		if (thrusts[i] > 1e-9f) {
			indices[k++] = i;
		}
	}
	if (k == 0) {
		constrain_for_thrusts(out, thrusts);
		return;
	}

	float tau0[3], f0[3];
	torque_force(current, thrusts, tau0, f0);

	const float eps = _cfg.finite_diff_eps;
	// B is 3 x k
	float B[3 * kMaxMounts]{};
	for (int col = 0; col < k; col++) {
		float pert[kNumAngles];
		memcpy(pert, current, sizeof(float) * kNumAngles);
		pert[indices[col] * 2 + 0] += eps;
		float tau_eps[3], f_eps[3];
		torque_force(pert, thrusts, tau_eps, f_eps);
		B[0 * k + col] = (tau_eps[0] - tau0[0]) / eps;
		B[1 * k + col] = (tau_eps[1] - tau0[1]) / eps;
		B[2 * k + col] = (tau_eps[2] - tau0[2]) / eps;
	}

	float residual[3] = {
		target[0] - tau0[0],
		target[1] - tau0[1],
		target[2] - tau0[2]
	};

	const float w[3] = {
		clampf(_cfg.roll_torque_weight, 0.f, 1e3f),
		1.f,
		clampf(_cfg.yaw_torque_weight, 0.f, 1e3f)
	};

	// Gram = B^T W B + ridge I ; rhs = B^T W residual
	float G[kMaxMounts * kMaxMounts]{};
	float rhs[kMaxMounts]{};
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			float s = (i == j) ? _torque_ridge : 0.f;
			for (int r = 0; r < 3; r++) {
				s += w[r] * B[r * k + i] * B[r * k + j];
			}
			G[i * k + j] = s;
		}
		float s = 0.f;
		for (int r = 0; r < 3; r++) {
			s += w[r] * B[r * k + i] * residual[r];
		}
		rhs[i] = s;
	}

	float delta[kMaxMounts]{};
	if (!solve_spd(G, rhs, k, delta)) {
		// leave current
		constrain_for_thrusts(out, thrusts);
		return;
	}

	for (int col = 0; col < k; col++) {
		out[indices[col] * 2 + 0] = current[indices[col] * 2 + 0] + delta[col];
	}
	clip_angles(out);
	constrain_for_thrusts(out, thrusts);
}

void TorqueAllocator::solve_differential_a2(const float seed[kNumAngles], const float target[3],
					    const float thrusts[kMaxMounts], float out[kNumAngles]) const
{
	solve_a2_step(seed, target, thrusts, out);
	for (int p = 0; p < _cfg.a2_polish_iters; p++) {
		float next[kNumAngles];
		solve_a2_step(out, target, thrusts, next);
		memcpy(out, next, sizeof(float) * kNumAngles);
	}
}

float TorqueAllocator::collective_a1_deg(float force_target_mag, const float thrusts[kMaxMounts],
					 bool force_active, const float target_torque[3]) const
{
	float a1_min = rad2deg(_min_a[1]);
	float a1_max = rad2deg(_max_a[1]);
	float a1_rest = rad2deg(_rest[1]);
	for (int i = 0; i < _n; i++) {
		a1_min = fminf2(a1_min, rad2deg(_min_a[i * 2 + 1]));
		a1_max = fmaxf2(a1_max, rad2deg(_max_a[i * 2 + 1]));
	}

	if (!force_active) {
		return a1_rest;
	}

	// Detect thrust imbalance (engine-out / weak mount)
	int n_active = 0;
	float t_min = 1e9f, t_max = 0.f;
	for (int i = 0; i < _n; i++) {
		if (thrusts[i] > 1e-9f) {
			n_active++;
			t_min = fminf2(t_min, thrusts[i]);
			t_max = fmaxf2(t_max, thrusts[i]);
		}
	}
	const bool imbalance = (n_active > 0) &&
			       (n_active < _n || (t_max > 1e-9f && t_min / t_max < 0.85f));

	auto pose_metrics = [&](float a1, float &fax, float &tau_err) {
		if (imbalance) {
			float seed[kNumAngles];
			uniform_a1_angles(a1, seed);
			constrain_for_thrusts(seed, thrusts);
			float angles[kNumAngles];
			solve_differential_a2(seed, target_torque, thrusts, angles);
			float tau[3], force[3];
			torque_force(angles, thrusts, tau, force);
			fax = axial_force(force);
			const float dx = tau[0] - target_torque[0];
			const float dy = tau[1] - target_torque[1];
			const float dz = tau[2] - target_torque[2];
			tau_err = sqrtf(dx * dx + dy * dy + dz * dz);
		} else {
			float seed[kNumAngles];
			uniform_a1_angles(a1, seed);
			float tau[3], force[3];
			torque_force(seed, thrusts, tau, force);
			fax = axial_force(force);
			tau_err = 0.f;
		}
	};

	const int n_sample = (int)fmaxf2((float)(_cfg.force_search_iters * 2), 24.f);
	const float tau_tol = torque_tolerance(target_torque);

	float best_a1 = a1_rest;
	float best_ferr = 1e9f;
	bool have_feasible = false;
	float f_lo_sample = 0.f, f_hi_sample = 0.f;

	for (int s = 0; s < n_sample; s++) {
		const float a1 = a1_min + (a1_max - a1_min) * (float)s / (float)(n_sample - 1);
		float fax, tau_err;
		pose_metrics(a1, fax, tau_err);
		if (s == 0) {
			f_lo_sample = fax;
		}
		if (s == n_sample - 1) {
			f_hi_sample = fax;
		}
		if (imbalance) {
			if (tau_err <= tau_tol) {
				const float ferr = fabsf2(fax - force_target_mag);
				if (!have_feasible || ferr < best_ferr) {
					have_feasible = true;
					best_ferr = ferr;
					best_a1 = a1;
				}
			}
		}
	}

	if (imbalance && have_feasible) {
		return best_a1;
	}

	// Classic axial binary search (symmetric)
	const float f_low = fminf2(f_lo_sample, f_hi_sample);
	const float f_high = fmaxf2(f_lo_sample, f_hi_sample);
	if (force_target_mag <= f_low + 1e-9f) {
		return f_lo_sample <= f_hi_sample ? a1_min : a1_max;
	}
	if (force_target_mag >= f_high - 1e-9f) {
		return f_hi_sample >= f_lo_sample ? a1_max : a1_min;
	}

	const float target = clampf(force_target_mag, f_low, f_high);
	float lo = a1_min;
	float hi = a1_max;
	const bool increasing = f_hi_sample >= f_lo_sample;
	for (int it = 0; it < _cfg.force_search_iters; it++) {
		const float mid = 0.5f * (lo + hi);
		float fax, tau_err;
		pose_metrics(mid, fax, tau_err);
		(void)tau_err;
		if (increasing) {
			if (fax < target) {
				lo = mid;
			} else {
				hi = mid;
			}
		} else {
			if (fax < target) {
				hi = mid;
			} else {
				lo = mid;
			}
		}
	}
	return 0.5f * (lo + hi);
}

void TorqueAllocator::retrim_collective_a1(const float angles[kNumAngles], float force_target_mag,
					   const float thrusts[kMaxMounts], float out[kNumAngles]) const
{
	memcpy(out, angles, sizeof(float) * kNumAngles);
	float a1_min = rad2deg(_min_a[1]);
	float a1_max = rad2deg(_max_a[1]);
	for (int i = 0; i < _n; i++) {
		a1_min = fminf2(a1_min, rad2deg(_min_a[i * 2 + 1]));
		a1_max = fmaxf2(a1_max, rad2deg(_max_a[i * 2 + 1]));
	}

	bool any_active = false;
	for (int i = 0; i < _n; i++) {
		if (thrusts[i] > 1e-9f) {
			any_active = true;
			break;
		}
	}
	if (!any_active) {
		return;
	}

	auto axial_at = [&](float a1_deg) {
		float trial[kNumAngles];
		memcpy(trial, angles, sizeof(float) * kNumAngles);
		const float a1 = deg2rad(a1_deg);
		for (int i = 0; i < _n; i++) {
			if (thrusts[i] > 1e-9f) {
				trial[i * 2 + 1] = a1;
			}
		}
		clip_angles(trial);
		constrain_for_thrusts(trial, thrusts);
		float tau[3], force[3];
		torque_force(trial, thrusts, tau, force);
		return axial_force(force);
	};

	const float f_lo = axial_at(a1_min);
	const float f_hi = axial_at(a1_max);
	const float f_low = fminf2(f_lo, f_hi);
	const float f_high = fmaxf2(f_lo, f_hi);
	float a1_deg;
	if (force_target_mag <= f_low + 1e-9f) {
		a1_deg = f_lo <= f_hi ? a1_min : a1_max;
	} else if (force_target_mag >= f_high - 1e-9f) {
		a1_deg = f_hi >= f_lo ? a1_max : a1_min;
	} else {
		const float target = clampf(force_target_mag, f_low, f_high);
		float lo = a1_min, hi = a1_max;
		const bool increasing = f_hi >= f_lo;
		const int iters = (int)fmaxf2(4.f, (float)(_cfg.force_search_iters / 2));
		for (int it = 0; it < iters; it++) {
			const float mid = 0.5f * (lo + hi);
			const float f_mid = axial_at(mid);
			if (increasing) {
				if (f_mid < target) {
					lo = mid;
				} else {
					hi = mid;
				}
			} else {
				if (f_mid < target) {
					hi = mid;
				} else {
					lo = mid;
				}
			}
		}
		a1_deg = 0.5f * (lo + hi);
	}

	const float a1 = deg2rad(a1_deg);
	for (int i = 0; i < _n; i++) {
		if (thrusts[i] > 1e-9f) {
			out[i * 2 + 1] = a1;
		}
	}
	clip_angles(out);
	constrain_for_thrusts(out, thrusts);
}

AllocResult TorqueAllocator::allocate_force_track(
	const float target_torque[3],
	float force_target_N,
	float force_weight,
	const float *thrusts_in)
{
	AllocResult result{};
	float thrusts[kMaxMounts]{};
	resolve_thrusts(thrusts_in, thrusts);

	const bool force_active = (force_weight > 0.f);
	const float force_target_mag = force_active ? fabsf2(force_target_N) : 0.f;

	const float a1_deg = collective_a1_deg(force_target_mag, thrusts, force_active, target_torque);
	float seed[kNumAngles];
	uniform_a1_angles(a1_deg, seed);
	constrain_for_thrusts(seed, thrusts);

	float angles[kNumAngles];
	solve_differential_a2(seed, target_torque, thrusts, angles);

	if (force_active) {
		float tau_before[3], f_before[3];
		torque_force(angles, thrusts, tau_before, f_before);
		const float err_before = sqrtf(
			(target_torque[0] - tau_before[0]) * (target_torque[0] - tau_before[0]) +
			(target_torque[1] - tau_before[1]) * (target_torque[1] - tau_before[1]) +
			(target_torque[2] - tau_before[2]) * (target_torque[2] - tau_before[2]));

		float candidate[kNumAngles];
		retrim_collective_a1(angles, force_target_mag, thrusts, candidate);

		int polish = (int)fmaxf2(1.f, (float)_cfg.a2_polish_iters);
		int n_active = 0;
		for (int i = 0; i < _n; i++) {
			if (thrusts[i] > 1e-9f) {
				n_active++;
			}
		}
		if (n_active < _n) {
			polish = (int)fmaxf2((float)polish, 3.f);
		}
		for (int p = 0; p < polish; p++) {
			float next[kNumAngles];
			solve_a2_step(candidate, target_torque, thrusts, next);
			memcpy(candidate, next, sizeof(float) * kNumAngles);
		}

		float tau_after[3], f_after[3];
		torque_force(candidate, thrusts, tau_after, f_after);
		const float err_after = sqrtf(
			(target_torque[0] - tau_after[0]) * (target_torque[0] - tau_after[0]) +
			(target_torque[1] - tau_after[1]) * (target_torque[1] - tau_after[1]) +
			(target_torque[2] - tau_after[2]) * (target_torque[2] - tau_after[2]));

		if (err_after <= err_before * 1.15f + 1e-3f) {
			memcpy(angles, candidate, sizeof(float) * kNumAngles);
		}
	}

	float torque[3], force[3];
	torque_force(angles, thrusts, torque, force);
	const float residual = sqrtf(
		(target_torque[0] - torque[0]) * (target_torque[0] - torque[0]) +
		(target_torque[1] - torque[1]) * (target_torque[1] - torque[1]) +
		(target_torque[2] - torque[2]) * (target_torque[2] - torque[2]));

	bool saturated = residual > torque_tolerance(target_torque);
	if (force_active) {
		const float force_error = fabsf2(force_target_mag - axial_force(force));
		const float force_tol = fmaxf2(0.08f, 0.02f * fmaxf2(force_target_mag, 1.f));
		if (force_error > force_tol) {
			saturated = true;
		}
	}

	memcpy(_angles, angles, sizeof(float) * kNumAngles);
	memcpy(result.angles, angles, sizeof(float) * kNumAngles);
	result.torque[0] = torque[0];
	result.torque[1] = torque[1];
	result.torque[2] = torque[2];
	result.force[0] = force[0];
	result.force[1] = force[1];
	result.force[2] = force[2];
	result.saturated = saturated;
	result.residual = residual;
	return result;
}

void TorqueAllocator::apply_actuator_slew(
	float angles_io[kNumAngles],
	const float target[kNumAngles],
	float slew_deg_s,
	float dt)
{
	if (dt <= 0.f || slew_deg_s <= 0.f) {
		memcpy(angles_io, target, sizeof(float) * kNumAngles);
		return;
	}
	const float max_step = deg2rad(slew_deg_s) * dt;
	for (int i = 0; i < kNumAngles; i++) {
		const float err = target[i] - angles_io[i];
		angles_io[i] += clampf(err, -max_step, max_step);
	}
}

} // namespace tvc
