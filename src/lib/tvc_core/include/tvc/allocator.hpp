#pragma once

#include "tvc/types.hpp"

namespace tvc {

/**
 * Collective a1 + differential a2 mixer (Prototype tvc/allocator.py).
 *
 * Flight path always uses allocate_force_track / mix — no multi-seed LM.
 * Angles stored as flat [M0a2, M0a1, M1a2, M1a1, M2a2, M2a1].
 */
class TorqueAllocator {
public:
	TorqueAllocator();

	void set_mounts(const Mount mounts[kMaxMounts], int n = kMaxMounts);
	void set_com(const Vec3 &com);
	void set_force_axis(const Vec3 &axis);
	void set_solver(const SolverConfig &cfg);

	/** Rest pose flat angles. */
	void rest_angles_flat(float out[kNumAngles]) const;

	/** Axial force along force_axis at rest pose [N]. */
	float axial_at_rest(const float *thrusts) const;

	/** Uniform a1, a2=0 so axial force is twr * weight. Attitude off. */
	void twr_lock_angles_flat(const float *thrusts, float weight, float twr,
				  float out[kNumAngles]) const;

	/**
	 * Flight mixer: collective a1 for axial force + differential a2 for torque.
	 * thrusts[i] in N; if nullptr uses mount.thrust.
	 */
	AllocResult allocate_force_track(
		const float target_torque[3],
		float force_target_N,
		float force_weight,
		const float *thrusts);

	/** Apply per-axis slew toward target (deg/s rate), dt in seconds. */
	static void apply_actuator_slew(
		float angles_io[kNumAngles],
		const float target[kNumAngles],
		float slew_deg_s,
		float dt);

private:
	int _n{kMaxMounts};
	Mount _mounts[kMaxMounts]{};
	Vec3 _com{};
	Vec3 _force_axis{0.f, 0.f, 1.f};
	SolverConfig _cfg{};
	float _angles[kNumAngles]{};
	float _rest[kNumAngles]{};
	float _min_a[kNumAngles]{};
	float _max_a[kNumAngles]{};
	float _torque_ridge{0.0225f};

	void clip_angles(float a[kNumAngles]) const;
	void constrain_for_thrusts(float a[kNumAngles], const float thrusts[kMaxMounts]) const;
	void resolve_thrusts(const float *thrusts, float out[kMaxMounts]) const;
	void torque_force(const float a[kNumAngles], const float thrusts[kMaxMounts],
			  float torque[3], float force[3]) const;
	float axial_force(const float force[3]) const;
	float torque_tolerance(const float target[3]) const;

	void uniform_a1_angles(float a1_deg, float out[kNumAngles]) const;
	float collective_a1_deg(float force_target_mag, const float thrusts[kMaxMounts],
				bool force_active, const float target_torque[3]) const;
	void solve_differential_a2(const float seed[kNumAngles], const float target[3],
				   const float thrusts[kMaxMounts], float out[kNumAngles]) const;
	void solve_a2_step(const float current[kNumAngles], const float target[3],
			   const float thrusts[kMaxMounts], float out[kNumAngles]) const;
	void retrim_collective_a1(const float angles[kNumAngles], float force_target_mag,
				  const float thrusts[kMaxMounts], float out[kNumAngles]) const;
};

} // namespace tvc
