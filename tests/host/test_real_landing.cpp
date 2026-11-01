#include <tvc/real_landing.hpp>

#include <cstdio>

static int g_fail = 0;

static void expect(bool ok, const char *msg)
{
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", msg);
		g_fail++;
	}
}

static tvc::RealLandingConfig g12()
{
	tvc::RealLandingConfig cfg{};
	cfg.has_loft = true;
	cfg.loft = {8.f, 0.f, 30.f};
	cfg.staging = {2.f, 0.f, 1.5f};
	cfg.pad = {0.f, 0.f, -0.1919f};
	cfg.vertical_hold_s = 0.f;
	cfg.capture_radius_m = 1.2f;
	cfg.capture_max_speed_m_s = 1.f;
	cfg.capture_settle_s = 2.f;
	cfg.stage_radius_m = 1.f;
	cfg.stage_speed_m_s = 1.2f;
	cfg.min_approach_time_s = 0.5f;
	cfg.descent_duration_s = 0.8f;
	cfg.burn_end_s = 12.6f;
	cfg.pad_remain_s = 3.f;
	cfg.burn_gate = true;
	return cfg;
}

int main()
{
	tvc::RealLandingGuide guide;
	const tvc::RealLandingConfig cfg = g12();

	tvc::RealLandingSample s{};
	s.t_s = 1.f;
	s.pos = {8.f, 0.f, 4.f};
	s.vel = {0.f, 0.f, 8.f};
	s.vel_valid = true;
	tvc::RealLandingOutput out = guide.update(s, cfg);
	expect(out.phase == 3, "climb is still the loft");
	expect(out.pos.z > 29.f, "loft target is 30 m");

	s.t_s = 6.f;
	s.pos = {8.f, 0.f, 29.f};
	s.vel = {0.f, 0.f, 0.2f};
	out = guide.update(s, cfg);
	expect(out.phase == 1, "crest leaves the loft for the staging waypoint");
	expect(out.pos.x > 1.9f && out.pos.x < 2.1f, "staging is 2 m east");

	s.t_s = 8.f;
	s.pos = {2.f, 0.f, 1.5f};
	s.vel = {};
	out = guide.update(s, cfg);
	expect(out.phase == 1, "before burnout-3 s the pad slide stays shut");

	s.t_s = 9.7f;
	s.pos = {2.1f, 0.f, 2.f};
	s.vel = {0.2f, 0.f, -0.4f};
	out = guide.update(s, cfg);
	expect(out.phase == 2, "burn gate opens the pad ease");
	expect(out.pos.z > 1.f, "ease starts at the staging waypoint, not the live altitude");

	s.t_s = 10.5f;
	out = guide.update(s, cfg);
	expect(out.phase == 2, "still landing");
	expect(out.pos.z < 0.f, "ease reaches the pad altitude");

	tvc::RealLandingGuide shock;
	s.t_s = 0.2f;
	s.pos = {8.f, 0.f, 0.f};
	s.vel = {0.f, 0.f, 0.f};
	out = shock.update(s, cfg);
	expect(out.phase == 3, "still on the pad is not a crest");

	s.t_s = 8.1f;
	s.pos = {8.f, 0.f, 12.f};
	s.vel = {0.f, 0.f, 2.f};
	out = shock.update(s, cfg);
	expect(out.phase == 1, "loft abort clock still hands off to staging");

	if (g_fail) {
		std::fprintf(stderr, "%d real_landing checks failed\n", g_fail);
		return 1;
	}

	return 0;
}
