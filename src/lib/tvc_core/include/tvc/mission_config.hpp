#pragma once

#include "tvc/types.hpp"

#include <stdint.h>

namespace tvc {

static constexpr int kMissionJsonMaxBytes = 8192;
static constexpr int kMissionNameMax = 40;
static constexpr const char *kMissionJsonPath = "/fs/microsd/tvc/mission.json";

enum class MissionLoadStatus : uint8_t {
	Ok = 0,
	FileMissing = 1,
	ParseError = 2,
	InvalidValue = 3,
	MissingField = 4,
};

/**
 * Flattened pad mission written by tools/pack_vehicle_sd.py.
 * Missing file is not fail-closed (params remain the fallback).
 */
struct MissionConfig {
	bool valid{false};
	bool has_gains{false};
	MissionLoadStatus status{MissionLoadStatus::FileMissing};
	char name[kMissionNameMax]{};
	int mode{0}; // 0=hold, 1=hover, 2=real_landing
	float hover_z{10.f};
	float wp_e{5.f};
	float wp_n{0.f};
	float wp_u{5.f};
	float capture_radius_m{1.5f};
	float capture_max_speed_m_s{2.f};
	float capture_settle_s{0.f};
	float descent_duration_s{7.f};
	float min_approach_time_s{2.f};
	bool has_loft{false};
	float loft_e{0.f};
	float loft_n{0.f};
	float loft_u{0.f};
	float vertical_hold_s{0.f};
	float stage_radius_m{1.f};
	float stage_speed_m_s{1.2f};
	bool has_burn_gate{false};
	float burn_end_s{0.f};
	float pad_remain_s{0.f};
	float safe_z{1.3081f};
	float land_z{0.f};
	float safe_half_m{0.40f};
	float safe_vmax{1.0f};
	float pos_kp{0.8f};
	float pos_kd{1.4f};
	float pos_kp_y{0.8f};
	float pos_kd_y{1.4f};
	float pos_ki{0.f};
	float pos_ki_y{0.015f};
	bool has_pos_ki{false};
	float pos_i_limit{0.08f};
	bool has_pos_i_limit{false};
	float pos_i_start{1.5f};
	bool has_pos_i_start{false};
	float lat_priority{0.05f};
	bool has_lat_priority{false};
	float lat_lpf_hz{0.f};
	bool has_lat_lpf{false};
	float pos_tilt_deg{20.f};
	float pos_tilt_y_deg{-1.f}; // <0 → same as pos_tilt_deg
	float max_speed_m_s{0.f};   // 0 → module default
	float max_speed_y_m_s{-1.f};
	float max_position_error_m{3.f};
	float max_position_error_y_m{0.f};
	bool prefer_lateral_y{false};
	float alt_kp{12.f};
	float alt_kd{8.f};
	float alt_ki{1.5f};
	float alt_i_limit{20.f};
	bool has_alt_i_limit{false};
	float throttle_min{0.f}; // fraction of available thrust; 0 disables the floor
	float land_tilt_deg{-1.f}; // <0 keeps the approach tilt through the pad ease
	float force_weight{0.25f};
	float heading_rad{NAN}; // NAN = free heading
};

bool parse_mission_json(const char *json, unsigned len, MissionConfig &out);
bool load_mission_config_file(const char *json_path, MissionConfig &out);
const MissionConfig &loaded_mission_config();

} // namespace tvc
