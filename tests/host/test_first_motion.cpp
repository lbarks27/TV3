#include <tvc/first_motion.hpp>

#include <cstdio>

static int g_fail = 0;

static void expect(bool ok, const char *msg)
{
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", msg);
		g_fail++;
	}
}

int main()
{
	tvc::FirstMotionConfig cfg{};
	cfg.vz_up_mps = 0.4f;
	cfg.dz_m = 0.05f;
	cfg.az_up_mps2 = 2.0f;
	cfg.consecutive = 3;

	tvc::FirstMotionDetector d;
	d.capture_pad(0.02f);

	tvc::FirstMotionSample rest{};
	rest.vz_valid = true;
	rest.z_valid = true;
	rest.az_valid = true;
	rest.vz_up_mps = 0.02f;
	rest.height_agl_m = 0.02f;
	rest.az_up_mps2 = 0.1f;

	for (int i = 0; i < 10; i++) {
		expect(!d.update(rest, cfg), "rest must not trip");
	}
	expect(d.hits() == 0, "rest hits stay 0");

	tvc::FirstMotionSample spike = rest;
	spike.az_up_mps2 = 5.f;
	expect(!d.update(spike, cfg), "accel spike is not motion");
	expect(!d.update(spike, cfg), "accel spike 2 is not motion");
	expect(!d.update(spike, cfg), "accel spike 3 is not motion");
	expect(!d.latched(), "pad shock does not latch");
	expect(d.hits() == 0, "accel does not count hits");

	d.reset();
	d.capture_pad(0.03f);
	tvc::FirstMotionSample climb = rest;
	climb.vz_up_mps = 0.5f;
	expect(!d.update(climb, cfg), "vz 1/3");
	expect(!d.update(climb, cfg), "vz 2/3");
	expect(d.update(climb, cfg), "vz 3/3 latches");

	d.reset();
	d.capture_pad(0.10f);
	tvc::FirstMotionSample rise = rest;
	rise.height_agl_m = 0.16f; // +6 cm
	expect(!d.update(rise, cfg), "dz 1/3");
	expect(!d.update(rise, cfg), "dz 2/3");
	expect(d.update(rise, cfg), "dz 3/3 latches");

	d.reset();
	tvc::FirstMotionSample no_pad = rest;
	no_pad.z_valid = false;
	no_pad.az_valid = false;
	no_pad.vz_valid = false;
	expect(!d.update(no_pad, cfg), "invalid sample is not motion");

	if (g_fail) {
		std::fprintf(stderr, "%d first_motion checks failed\n", g_fail);
		return 1;
	}

	return 0;
}
