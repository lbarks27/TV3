#pragma once

#include "tvc/types.hpp"

#include <stdint.h>

namespace tvc {

static constexpr int kVehicleJsonMaxBytes = 24576;
static constexpr const char *kVehicleJsonPath = "/fs/microsd/tvc/vehicle.json";
static constexpr const char *kVehicleCrcPath = "/fs/microsd/tvc/vehicle.crc";

enum class VehicleLoadStatus : uint8_t {
	Ok = 0,
	FileMissing = 1,
	CrcMissing = 2,
	CrcMismatch = 3,
	ParseError = 4,
	MissingField = 5,
	InvalidValue = 6,
};

struct VehicleConfig {
	bool valid{false};
	VehicleLoadStatus status{VehicleLoadStatus::FileMissing};
	uint32_t crc{0};
	int n_mounts{0};
	Mount mounts[kMaxMounts]{};
	Vec3 unloaded_com{};
	Vec3 loaded_com{};          // unloaded + motor masses at gimbal pivots
	float unloaded_mass_kg{0.f};
	float loaded_mass_kg{0.f};
	float motor_mass_kg[kMaxMounts]{};
	float inertia_diag_kgm2[3]{};
	float a2_min_deg{0.f};
	float a2_max_deg{0.f};
	float a1_min_deg{0.f};
	float a1_max_deg{0.f};
	float a2_rest_deg{0.f};
	float a1_rest_deg{0.f};
};

uint32_t crc32_ieee(const void *data, unsigned len);

/** Parse vehicle JSON from a memory buffer (no CRC). */
bool parse_vehicle_json(const char *json, unsigned len, VehicleConfig &out);

/**
 * Read JSON + sidecar CRC from disk. Fail-closed: out.valid is true only when
 * every required geometry / mass / CoM / limit field is present and finite.
 */
bool load_vehicle_config_file(const char *json_path, const char *crc_path, VehicleConfig &out);

/** Last successful boot load (invalid until load_vehicle_config_file succeeds). */
const VehicleConfig &loaded_vehicle_config();

/** Compose loaded CoM = unloaded CoM + point masses at mount pivots. */
Vec3 compose_com_at_pivots(const Vec3 &unloaded_com, float unloaded_mass_kg,
			   const Mount mounts[kMaxMounts], const float motor_mass_kg[kMaxMounts],
			   int n_mounts, float *loaded_mass_kg);

} // namespace tvc
