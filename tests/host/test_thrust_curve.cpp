#include <tvc/thrust_curve.hpp>
#include <tvc/motors.hpp>

#include <cstdio>
#include <cmath>
#include <cstdlib>

static int g_fail = 0;

static void expect(bool cond, const char *msg)
{
	if (!cond) {
		std::fprintf(stderr, "FAIL: %s\n", msg);
		++g_fail;
	}
}

int main()
{
	using namespace tvc;

	expect(std::fabs(trapezoid_thrust_n(0.5f, 12.f, 4.f, 0.5f, 0.5f) - 12.f) < 1e-3f, "peak hold");
	expect(trapezoid_thrust_n(-0.1f, 12.f, 4.f, 0.5f, 0.5f) == 0.f, "pre-burn");
	expect(trapezoid_thrust_n(5.f, 12.f, 4.f, 0.5f, 0.5f) == 0.f, "post-burn");
	expect(trapezoid_thrust_n(0.25f, 12.f, 4.f, 0.5f, 0.5f) > 5.f, "rise mid");

	MotorCurve c{};
	c.n_samples = 3;
	c.times_s[0] = 0.f; c.thrust_n[0] = 0.f;
	c.times_s[1] = 1.f; c.thrust_n[1] = 10.f;
	c.times_s[2] = 2.f; c.thrust_n[2] = 0.f;
	expect(std::fabs(c.thrust_at(0.5f) - 5.f) < 1e-3f, "curve mid");
	c.ignition_delay_s = 1.f;
	expect(std::fabs(c.thrust_at(1.5f) - 5.f) < 1e-3f, "curve with delay");

	if (g_fail) {
		std::fprintf(stderr, "%d thrust_curve tests failed\n", g_fail);
		return 1;
	}

	std::printf("test_thrust_curve: OK\n");
	return 0;
}
