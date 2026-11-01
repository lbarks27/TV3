#pragma once

// QGC / MAVLink contract for TVC pad commands (no PX4 types).
// Fly View custom actions send MAV_CMD_USER_1 (31010) with param1 = GcsAction.
// Fire (HIL software liftoff) also requires param2 = kFireMagic.

#include "tvc/math.hpp"

namespace tvc {

static constexpr uint16_t kMavCmdTvc = 31010; // MAV_CMD_USER_1
static constexpr uint16_t kMavCmdFlightTermination = 185;
static constexpr uint16_t kMavCmdArmDisarm = 400;

// Same magic PX4 uses for force-arm (COMMAND_LONG param2).
static constexpr float kFireMagic = 21196.f;
static constexpr float kFireMagicTol = 0.5f;

enum class GcsAction : uint8_t {
	None = 0,
	Arm = 1,
	Disarm = 2,
	IgnArm = 3,
	Fire = 4,
	Abort = 5,
	Reset = 6,
	ReloadThrust = 8,
	ReloadMission = 9,
	Bench = 10,
	NoBench = 11,
	Zero = 12,
	NoZero = 13,
};

inline bool fire_magic_ok(float param2)
{
	return fabsf2(param2 - kFireMagic) <= kFireMagicTol;
}

inline GcsAction gcs_action_from_param1(float param1)
{
	const int v = static_cast<int>(param1 + (param1 >= 0.f ? 0.5f : -0.5f));

	if (v < static_cast<int>(GcsAction::Arm) || v > static_cast<int>(GcsAction::NoZero)) {
		return GcsAction::None;
	}

	return static_cast<GcsAction>(v);
}

inline bool gcs_action_is_fsm(GcsAction a)
{
	return a == GcsAction::Arm || a == GcsAction::Disarm || a == GcsAction::IgnArm ||
	       a == GcsAction::Fire || a == GcsAction::Abort || a == GcsAction::Reset;
}

inline const char *gcs_action_name(GcsAction a)
{
	switch (a) {
	case GcsAction::Arm: return "arm";
	case GcsAction::Disarm: return "disarm";
	case GcsAction::IgnArm: return "launch_ready";
	case GcsAction::Fire: return "fire";
	case GcsAction::Abort: return "abort";
	case GcsAction::Reset: return "reset";
	case GcsAction::ReloadThrust: return "reload_thrust";
	case GcsAction::ReloadMission: return "reload_mission";
	case GcsAction::Bench: return "bench";
	case GcsAction::NoBench: return "nobench";
	case GcsAction::Zero: return "zero";
	case GcsAction::NoZero: return "nozero";
	case GcsAction::None:
	default: return "none";
	}
}

// Matches tvc_vehicle_state.msg STATE_* (short enough for STATUSTEXT).
inline const char *vehicle_state_name(uint8_t state)
{
	switch (state) {
	case 0: return "DISARM";
	case 1: return "PREFLT";
	case 2: return "ARMED";
	case 3: return "LAUNCH";
	case 4: return "BURN";
	case 5: return "COAST";
	case 6: return "LAND";
	case 7: return "ABORT";
	case 8: return "SAFE";
	default: return "?";
	}
}

} // namespace tvc
