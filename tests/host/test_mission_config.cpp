#include <tvc/mission_config.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>

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
	const char *json =
		"{\n"
		"  \"name\": \"real_landing_g12\",\n"
		"  \"mode\": 2,\n"
		"  \"hover_z\": 2.0,\n"
		"  \"wp_e\": 5.0,\n"
		"  \"wp_u\": 2.0,\n"
		"  \"capture_radius_m\": 1.5,\n"
		"  \"capture_settle_s\": 2.0,\n"
		"  \"descent_duration_s\": 9.0,\n"
		"  \"min_approach_time_s\": 2.5,\n"
		"  \"land_z\": -0.1919,\n"
		"  \"pos_kp\": 0.4,\n"
		"  \"pos_kd\": 8.5,\n"
		"  \"pos_kp_y\": 0.28,\n"
		"  \"pos_kd_y\": 10.0,\n"
		"  \"pos_ki\": 0.0,\n"
		"  \"pos_ki_y\": 0.02,\n"
		"  \"pos_i_limit\": 0.04,\n"
		"  \"pos_i_start\": 2.5,\n"
		"  \"lat_priority\": 0.05,\n"
		"  \"lat_lpf_hz\": 3.0,\n"
		"  \"pos_tilt_deg\": 10.0,\n"
		"  \"pos_tilt_y_deg\": 5.5,\n"
		"  \"max_speed_m_s\": 2.8,\n"
		"  \"max_speed_y_m_s\": 0.85,\n"
		"  \"prefer_lateral_y\": 1,\n"
		"  \"alt_kp\": 14.0,\n"
		"  \"alt_kd\": 28.0,\n"
		"  \"alt_ki\": 0.3,\n"
		"  \"force_weight\": 1.0\n"
		"}\n";

	tvc::MissionConfig cfg;
	expect(tvc::parse_mission_json(json, (unsigned)std::strlen(json), cfg), "parse mission json");
	expect(cfg.valid, "valid");
	expect(cfg.has_gains, "has_gains");
	expect(cfg.mode == 2, "mode 2");
	expect(std::strcmp(cfg.name, "real_landing_g12") == 0, "name");
	expect(std::fabs(cfg.wp_e - 5.f) < 1e-4f, "wp_e");
	expect(std::fabs(cfg.wp_u - 2.f) < 1e-4f, "wp_u");
	expect(std::fabs(cfg.capture_settle_s - 2.f) < 1e-4f, "capture_settle_s");
	expect(std::fabs(cfg.descent_duration_s - 9.f) < 1e-4f, "land_t");
	expect(std::fabs(cfg.land_z + 0.1919f) < 1e-4f, "land_z");
	expect(std::fabs(cfg.pos_kp - 0.4f) < 1e-4f, "pos_kp");
	expect(std::fabs(cfg.pos_kp_y - 0.28f) < 1e-4f, "pos_kp_y");
	expect(std::fabs(cfg.pos_kd_y - 10.f) < 1e-4f, "pos_kd_y");
	expect(cfg.has_pos_ki, "has_pos_ki");
	expect(std::fabs(cfg.pos_ki_y - 0.02f) < 1e-4f, "pos_ki_y");
	expect(cfg.has_pos_i_limit && std::fabs(cfg.pos_i_limit - 0.04f) < 1e-4f, "pos_i_limit");
	expect(cfg.has_pos_i_start && std::fabs(cfg.pos_i_start - 2.5f) < 1e-4f, "pos_i_start");
	expect(cfg.has_lat_priority && std::fabs(cfg.lat_priority - 0.05f) < 1e-4f, "lat_priority");
	expect(cfg.has_lat_lpf && std::fabs(cfg.lat_lpf_hz - 3.f) < 1e-4f, "lat_lpf");
	expect(std::fabs(cfg.pos_tilt_y_deg - 5.5f) < 1e-4f, "pos_tilt_y");
	expect(cfg.prefer_lateral_y, "prefer_lateral_y");
	expect(std::fabs(cfg.alt_kd - 28.f) < 1e-3f, "alt_kd");
	expect(std::fabs(cfg.force_weight - 1.f) < 1e-4f, "force_weight");
	expect(cfg.throttle_min == 0.f, "throttle floor off unless packed");
	expect(!cfg.has_alt_i_limit, "integral limit stays the controller default");

	const char *g12 =
		"{ \"mode\": 2, \"throttle_min\": 0.2, \"alt_i_limit\": 6.0 }";
	tvc::MissionConfig floor;
	expect(tvc::parse_mission_json(g12, (unsigned)std::strlen(g12), floor), "parse g12 floor");
	expect(std::fabs(floor.throttle_min - 0.2f) < 1e-4f, "throttle_min");
	expect(floor.has_alt_i_limit && std::fabs(floor.alt_i_limit - 6.f) < 1e-4f, "alt_i_limit");

	tvc::MissionConfig bad;
	expect(!tvc::parse_mission_json("{}", 2, bad), "reject missing mode");
	expect(bad.status == tvc::MissionLoadStatus::MissingField, "missing field status");

	tvc::MissionConfig missing;
	expect(!tvc::load_mission_config_file("/tmp/tvc_mission_does_not_exist.json", missing),
	       "missing file");
	expect(missing.status == tvc::MissionLoadStatus::FileMissing, "file missing status");

	if (g_fail) {
		std::fprintf(stderr, "%d failures\n", g_fail);
		return 1;
	}
	std::printf("test_mission_config OK\n");
	return 0;
}
