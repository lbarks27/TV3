#include <tvc/servo.hpp>

#include <cstdio>
#include <cmath>

static int g_fail = 0;

static void expect(bool ok, const char *msg)
{
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", msg);
		g_fail++;
	}
}

static void expect_near(float got, float want, float tol, const char *msg)
{
	if (std::fabs(got - want) > tol) {
		std::fprintf(stderr, "FAIL: %s got=%f want=%f\n", msg, (double)got, (double)want);
		g_fail++;
	}
}

int main()
{
	const float a2 = tvc::deg2rad(45.f);
	const float a1 = tvc::deg2rad(90.f);

	expect(tvc::servo_logical_to_main(0) == 1, "M0a2 MAIN 1");
	expect(tvc::servo_logical_to_main(1) == 5, "M0a1 MAIN 5");
	expect(tvc::servo_logical_to_main(5) == 7, "M2a1 MAIN 7");

	expect_near(tvc::gimbal_servo_norm(0, 0.f, 0.f, a2, a1, 0), 0.f, 1e-6f, "a2 rest");
	expect_near(tvc::gimbal_servo_norm(0, a2, 0.f, a2, a1, 0), 1.f, 1e-6f, "a2 +range");
	expect_near(tvc::gimbal_servo_norm(0, -a2, 0.f, a2, a1, 0), -1.f, 1e-6f, "a2 -range");
	expect_near(tvc::gimbal_servo_norm(1, 0.f, 0.f, a2, a1, 0), 0.f, 1e-6f, "a1 geometric 0");
	expect_near(tvc::gimbal_servo_norm(1, a1, 0.f, a2, a1, 0), -1.f, 1e-6f, "a1 fold");

	expect_near(tvc::gimbal_servo_norm(0, 0.f, 4.5f, a2, a1, 0), 0.1f, 1e-5f, "a2 +trim");
	expect_near(tvc::gimbal_servo_norm(0, 0.f, 4.5f, a2, a1, 1), -0.1f, 1e-5f, "a2 +trim rev");
	expect_near(tvc::gimbal_servo_norm(1, 0.f, 9.f, a2, a1, 0), -0.1f, 1e-5f, "a1 +trim fold sense");

	expect_near(tvc::gimbal_servo_norm(0, 0.f, 45.f, a2, a1, 0),
		    15.f / 45.f, 1e-5f, "trim clamped to 15");

	expect(tvc::servo_norm_to_pwm_us(0.f, 1125, 1875) == 1500, "a2 mid 1500");
	expect(tvc::servo_norm_to_pwm_us(1.f, 1125, 1875) == 1875, "a2 max");
	expect(tvc::servo_norm_to_pwm_us(-1.f, 1125, 1875) == 1125, "a2 min");

	const float n_rest_a1 = tvc::gimbal_servo_norm(1, tvc::deg2rad(15.f), 0.f, a2, a1, 0);
	expect(tvc::servo_norm_to_pwm_us(n_rest_a1, 800, 2200) == 1383, "a1 15deg maps to 1383 us");

	const float n = tvc::gimbal_servo_norm(0, 0.f, 4.5f, a2, a1, 0);
	expect(tvc::servo_norm_to_pwm_us(n, 1125, 1875) == 1538, "a2 +4.5deg trim pwm");

	if (g_fail) {
		std::fprintf(stderr, "%d servo checks failed\n", g_fail);
		return 1;
	}

	return 0;
}
