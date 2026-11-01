#include <tvc/gcs_cmd.hpp>

#include <cstdio>
#include <cstring>

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
	expect(tvc::kMavCmdTvc == 31010, "USER_1 id");
	expect(tvc::gcs_action_from_param1(1.f) == tvc::GcsAction::Arm, "arm");
	expect(tvc::gcs_action_from_param1(4.f) == tvc::GcsAction::Fire, "fire");
	expect(tvc::gcs_action_from_param1(4.2f) == tvc::GcsAction::Fire, "fire round");
	expect(tvc::gcs_action_from_param1(11.f) == tvc::GcsAction::NoBench, "nobench");
	expect(tvc::gcs_action_from_param1(12.f) == tvc::GcsAction::Zero, "zero");
	expect(tvc::gcs_action_from_param1(13.f) == tvc::GcsAction::NoZero, "nozero");
	expect(std::strcmp(tvc::gcs_action_name(tvc::GcsAction::Zero), "zero") == 0, "zero name");
	expect(tvc::gcs_action_from_param1(0.f) == tvc::GcsAction::None, "zero");
	expect(tvc::gcs_action_from_param1(99.f) == tvc::GcsAction::None, "oob");
	expect(tvc::fire_magic_ok(21196.f), "magic exact");
	expect(tvc::fire_magic_ok(21196.1f), "magic tol");
	expect(!tvc::fire_magic_ok(0.f), "magic missing");
	expect(tvc::gcs_action_is_fsm(tvc::GcsAction::Fire), "fire is fsm");
	expect(!tvc::gcs_action_is_fsm(tvc::GcsAction::ReloadThrust), "reload not fsm");
	expect(std::strcmp(tvc::vehicle_state_name(4), "BURN") == 0, "burn name");
	expect(std::strcmp(tvc::gcs_action_name(tvc::GcsAction::IgnArm), "launch_ready") == 0,
	       "launch_ready name");
	expect(std::strcmp(tvc::vehicle_state_name(3), "LAUNCH") == 0, "launch name");

	if (g_fail) {
		std::fprintf(stderr, "%d gcs_cmd checks failed\n", g_fail);
		return 1;
	}

	return 0;
}
