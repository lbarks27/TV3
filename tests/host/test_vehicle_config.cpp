#include <tvc/geometry.hpp>
#include <tvc/vehicle_config.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static int g_fail = 0;

static void expect(bool cond, const char *msg)
{
	if (!cond) {
		std::fprintf(stderr, "FAIL: %s\n", msg);
		g_fail++;
	}
}

static std::string g12_path()
{
	const char *from_env = std::getenv("TVC_G12_JSON");
	if (from_env && from_env[0]) {
		return from_env;
	}
	return std::string(TEST_VEHICLE_JSON);
}

static bool read_file(const std::string &path, std::string &out)
{
	FILE *f = fopen(path.c_str(), "rb");
	if (!f) {
		return false;
	}
	out.clear();
	char buf[4096];
	size_t n = 0;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
		out.append(buf, n);
	}
	fclose(f);
	return !out.empty();
}

int main()
{
	{
		tvc::Mount mounts[tvc::kMaxMounts];
		tvc::default_ring3_mounts(mounts);
		const tvc::Vec3 dir = tvc::thrust_direction(0.f, 0.f, mounts[0]);
		expect(std::fabs(dir.x) < 1e-5f && std::fabs(dir.y) < 1e-5f && dir.z > 0.99f,
		       "geometric a1=0 thrust is +Z");
		const tvc::Vec3 mid = tvc::thrust_direction(0.f, tvc::deg2rad(15.f), mounts[0]);
		expect(mid.z > 0.96f, "a1=15 still mostly +Z");
		const tvc::Vec3 folded = tvc::thrust_direction(0.f, tvc::deg2rad(90.f), mounts[0]);
		expect(folded.x < -0.9f && std::fabs(folded.z) < 0.1f, "a1=90 folds inward");
	}

	std::string json;
	const std::string path = g12_path();
	expect(read_file(path, json), "read tvc_ring3_g12.json");
	if (json.empty()) {
		std::fprintf(stderr, "missing %s\n", path.c_str());
		return 1;
	}

	tvc::VehicleConfig cfg;
	expect(tvc::parse_vehicle_json(json.c_str(), (unsigned)json.size(), cfg), "parse g12 json");
	expect(cfg.valid, "g12 valid");
	expect(cfg.n_mounts == 3, "3 mounts");
	expect(std::fabs(cfg.unloaded_mass_kg - 2.6056f) < 1e-4f, "unloaded mass");
	expect(std::fabs(cfg.motor_mass_kg[0] - 0.150f) < 1e-4f, "G12 motor mass");
	expect(std::fabs(cfg.loaded_mass_kg - (2.6056f + 3.f * 0.150f)) < 1e-3f, "loaded mass");
	expect(std::fabs(cfg.a1_rest_deg - 0.f) < 1e-3f, "a1 rest 0");
	expect(std::fabs(cfg.a1_min_deg - 0.f) < 1e-3f, "a1 min 0");
	expect(std::fabs(cfg.a2_max_deg - 35.f) < 1e-3f, "a2 limit 35");
	expect(std::fabs(cfg.mounts[0].rest_angles[1] - tvc::deg2rad(0.f)) < 1e-4f,
	       "mount rest a1 0 deg");
	expect(cfg.loaded_com.z < cfg.unloaded_com.z, "motors at ring pull CoM down");
	expect(std::fabs(cfg.unloaded_com.x) < 1e-5f, "unloaded CoM X");
	expect(std::fabs(cfg.unloaded_com.y) < 1e-5f, "unloaded CoM Y");
	expect(std::fabs(cfg.unloaded_com.z - 0.4699f) < 1e-5f, "unloaded CoM Z (18.5 in)");
	expect(std::fabs(cfg.inertia_diag_kgm2[0] - 1.012464f) < 1e-5f, "Ixx 1012.464 g·m²");
	expect(std::fabs(cfg.inertia_diag_kgm2[1] - 1.049836f) < 1e-5f, "Iyy 1049.836 g·m²");
	expect(std::fabs(cfg.inertia_diag_kgm2[2] - 0.062094f) < 1e-5f, "Izz 62.094 g·m²");
	// CoM-up unit matches force_axis on this geometry
	const float ncom = std::sqrt(cfg.loaded_com.x * cfg.loaded_com.x +
				     cfg.loaded_com.y * cfg.loaded_com.y +
				     cfg.loaded_com.z * cfg.loaded_com.z);
	expect(ncom > 0.3f, "loaded CoM norm");
	expect(std::fabs(cfg.loaded_com.z - 0.400698f) < 1e-4f, "loaded CoM Z");
	expect(std::fabs(cfg.loaded_com.x / ncom) < 1e-4f, "loaded CoM unit X");
	expect(std::fabs(cfg.loaded_com.y / ncom) < 1e-4f, "loaded CoM unit Y");
	expect(std::fabs(cfg.loaded_com.z / ncom - 1.f) < 1e-4f, "loaded CoM unit Z");

	const uint32_t crc = tvc::crc32_ieee(json.data(), (unsigned)json.size());
	expect(crc != 0, "crc nonzero");

	const char *json_tmp = "/tmp/tvc_vehicle_test.json";
	const char *crc_tmp = "/tmp/tvc_vehicle_test.crc";
	{
		FILE *jf = fopen(json_tmp, "wb");
		expect(jf != nullptr, "write tmp json");
		if (jf) {
			fwrite(json.data(), 1, json.size(), jf);
			fclose(jf);
		}
		FILE *cf = fopen(crc_tmp, "w");
		expect(cf != nullptr, "write tmp crc");
		if (cf) {
			std::fprintf(cf, "crc32=%08x\n", (unsigned)crc);
			fclose(cf);
		}
	}

	tvc::VehicleConfig loaded;
	expect(tvc::load_vehicle_config_file(json_tmp, crc_tmp, loaded), "load with matching crc");
	expect(loaded.valid, "loaded valid");
	expect(loaded.crc == crc, "crc stored");

	{
		FILE *cf = fopen(crc_tmp, "w");
		if (cf) {
			std::fprintf(cf, "crc32=%08x\n", (unsigned)(crc ^ 0xffffffffu));
			fclose(cf);
		}
		tvc::VehicleConfig bad;
		expect(!tvc::load_vehicle_config_file(json_tmp, crc_tmp, bad), "reject crc mismatch");
		expect(bad.status == tvc::VehicleLoadStatus::CrcMismatch, "crc mismatch status");
	}

	if (g_fail) {
		std::fprintf(stderr, "%d failures\n", g_fail);
		return 1;
	}
	std::printf("test_vehicle_config OK\n");
	return 0;
}
