#include <tvc/allocator.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int g_fail = 0;

static void expect(bool cond, const char *msg)
{
	if (!cond) {
		std::fprintf(stderr, "FAIL: %s\n", msg);
		g_fail++;
	}
}

int main()
{
	tvc::TorqueAllocator alloc;
	float thrusts[3] = {12.f, 12.f, 12.f};

	// Rest pose: torque ~0, axial force ~ sum thrust
	{
		float tau[3] = {0.f, 0.f, 0.f};
		auto r = alloc.allocate_force_track(tau, 36.f, 0.25f, thrusts);
		const float tn = std::sqrt(r.torque[0] * r.torque[0] + r.torque[1] * r.torque[1] + r.torque[2] * r.torque[2]);
		expect(tn < 0.5f, "rest torque near zero");
		expect(r.force[2] > 30.f, "axial force near total thrust at rest a1");
		const float a1_rest = tvc::deg2rad(0.f);
		for (int i = 0; i < 3; i++) {
			expect(std::fabs(r.angles[i * 2 + 1] - a1_rest) < 0.05f, "a1 near rest 0 deg");
		}
	}

	// Torque about X should produce differential a2
	{
		float tau[3] = {2.f, 0.f, 0.f};
		auto r = alloc.allocate_force_track(tau, 0.f, 0.f, thrusts);
		expect(std::fabs(r.torque[0] - 2.f) < 0.8f, "track roll torque");
		bool any_a2 = false;
		for (int i = 0; i < 3; i++) {
			if (std::fabs(r.angles[i * 2 + 0]) > 1e-3f) {
				any_a2 = true;
			}
		}
		expect(any_a2, "a2 deflects for torque");
	}

	// Engine-out: dead M0 should rest that mount
	{
		float t_eo[3] = {0.f, 12.f, 12.f};
		float tau[3] = {0.f, 0.f, 0.f};
		auto r = alloc.allocate_force_track(tau, 20.f, 0.25f, t_eo);
		expect(std::fabs(r.angles[0] - 0.f) < 1e-4f, "M0 a2 rest when dead");
		expect(std::fabs(r.angles[1] - tvc::deg2rad(0.f)) < 1e-3f, "M0 a1 rest when dead");
	}

	// 0.75 T/W lock: uniform a1, a2=0, axial ≈ 0.75 * weight
	{
		float thrusts_eq[3] = {12.f, 12.f, 12.f};
		float angles[6]{};
		alloc.twr_lock_angles_flat(thrusts_eq, 36.f, 0.75f, angles);
		for (int i = 0; i < 3; i++) {
			expect(std::fabs(angles[i * 2 + 0]) < 1e-4f, "twr-lock a2 is 0");
			expect(angles[i * 2 + 1] > tvc::deg2rad(0.f) + 0.05f, "twr-lock a1 folded past rest");
		}
	}

	{
		const float fax = alloc.axial_at_rest(thrusts);
		expect(fax > 35.5f && fax <= 36.01f, "rest axial ~ sum_T at a1=0");
	}

	if (g_fail) {
		std::fprintf(stderr, "%d failures\n", g_fail);
		return 1;
	}
	std::printf("test_allocator OK\n");
	return 0;
}
