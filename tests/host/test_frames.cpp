#include <tvc/frames.hpp>

#include <cmath>
#include <cstdio>

static int g_fail = 0;

static void expect_near(float a, float b, float eps, const char *msg)
{
	if (std::fabs(a - b) > eps) {
		std::fprintf(stderr, "FAIL: %s (%.6f vs %.6f)\n", msg, a, b);
		g_fail++;
	}
}

int main()
{
	// ENU/NED round-trip
	{
		tvc::Vec3 enu{1.f, 2.f, 3.f};
		tvc::Vec3 ned = tvc::enu_to_ned(enu);
		expect_near(ned.x, 2.f, 1e-6f, "ned.x");
		expect_near(ned.y, 1.f, 1e-6f, "ned.y");
		expect_near(ned.z, -3.f, 1e-6f, "ned.z");
		tvc::Vec3 back = tvc::ned_to_enu(ned);
		expect_near(back.x, 1.f, 1e-6f, "enu roundtrip x");
		expect_near(back.y, 2.f, 1e-6f, "enu roundtrip y");
		expect_near(back.z, 3.f, 1e-6f, "enu roundtrip z");
	}

	// Identity hover: level FLU +Z up → FRD +Z down world NED
	{
		tvc::Quat q_sim{1.f, 0.f, 0.f, 0.f};
		tvc::Quat q_px4 = tvc::q_sim_to_px4(q_sim);
		tvc::Quat q_back = tvc::q_px4_to_sim(q_px4);
		expect_near(std::fabs(q_back.w), 1.f, 1e-4f, "identity |w|");
		expect_near(q_back.x, 0.f, 1e-4f, "identity x");
		expect_near(q_back.y, 0.f, 1e-4f, "identity y");
		expect_near(q_back.z, 0.f, 1e-4f, "identity z");
	}

	// 90 deg yaw about sim Z
	{
		tvc::Quat q_sim = tvc::q_from_euler(0.f, 0.f, 1.5707963f);
		tvc::Quat q_px4 = tvc::q_sim_to_px4(q_sim);
		tvc::Quat q_back = tvc::q_px4_to_sim(q_px4);
		// compare as rotation of e1
		tvc::Vec3 e1{1.f, 0.f, 0.f};
		tvc::Vec3 a = tvc::q_rotate(q_sim, e1);
		tvc::Vec3 b = tvc::q_rotate(q_back, e1);
		expect_near(a.x, b.x, 1e-3f, "yaw e1.x");
		expect_near(a.y, b.y, 1e-3f, "yaw e1.y");
		expect_near(a.z, b.z, 1e-3f, "yaw e1.z");
	}

	// FLU torque sign: body roll rate FRD ↔ FLU
	{
		tvc::Vec3 w_flu{1.f, 0.f, 0.f};
		tvc::Vec3 w_frd = tvc::flu_to_frd(w_flu);
		expect_near(w_frd.x, 1.f, 1e-6f, "roll same");
		tvc::Vec3 w_flu2{0.f, 1.f, 0.f};
		tvc::Vec3 w_frd2 = tvc::flu_to_frd(w_flu2);
		expect_near(w_frd2.y, -1.f, 1e-6f, "pitch left→right");
	}

	if (g_fail) {
		return 1;
	}
	std::printf("test_frames OK\n");
	return 0;
}
