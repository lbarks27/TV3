/**
 * @file TvcActuators.cpp
 * Gimbal angle [rad] → actuator_servos normalized [-1,1].
 *
 * Logical channels (control[0..5], TVC_SERVO_REV bits 0..5):
 *   [M0a2, M0a1, M1a2, M1a1, M2a2, M2a1]
 * Pin map is airframe PWM_MAIN_FUNC* (MAIN 1/2/3 = a2, MAIN 5/6/7 = a1).
 *
 * Mapping (default):
 *   a2: rest=0 → 0, +range → +1, -range → -1
 *   a1: geometric 0 → 0, +fold (toward a1 max) → -1 (same PWM sense as the old rest=90 mapping)
 *       Mechanical rest is TVC_A1_REST (0°, vertical). Force-rest / DIS use that angle.
 *
 * TVC_TRIM_0..5 [deg] offset rest in geometric angle space (before reverse).
 * TVC_ZERO holds trimmed rest with COM_PREARM_MODE=2 so the pad can center
 * horns while disarmed. IO failsafe/disarmed PWM is kept in sync with trim.
 */

#include <parameters/param.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>
#include <drivers/drv_hrt.h>
#include <mathlib/mathlib.h>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/topics/actuator_servos.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/tvc_gimbal_cmd.h>
#include <uORB/topics/tvc_vehicle_state.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_command_ack.h>
#include <uORB/topics/vehicle_status.h>

#include <tvc/gcs_cmd.hpp>
#include <tvc/px4_gcs.hpp>
#include <tvc/servo.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace time_literals;

namespace
{
constexpr const char *kTrimNames[6] = {
	"TVC_TRIM_0", "TVC_TRIM_1", "TVC_TRIM_2",
	"TVC_TRIM_3", "TVC_TRIM_4", "TVC_TRIM_5"
};
constexpr const char *kChNames[6] = {
	"M0a2", "M0a1", "M1a2", "M1a1", "M2a2", "M2a1"
};
} // namespace

class TvcActuators : public ModuleBase<TvcActuators>, public ModuleParams, public px4::ScheduledWorkItem
{
public:
	TvcActuators() : ModuleParams(nullptr), ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::hp_default) {}
	~TvcActuators() override = default;

	static int task_spawn(int argc, char *argv[])
	{
		TvcActuators *inst = new TvcActuators();
		if (!inst) { return PX4_ERROR; }
		_object.store(inst);
		_task_id = task_id_is_work_queue;
		if (!inst->init()) {
			delete inst; _object.store(nullptr); _task_id = -1; return PX4_ERROR;
		}
		return PX4_OK;
	}

	static int custom_command(int argc, char *argv[])
	{
		if (argc >= 1 && argv[0] != nullptr) {
			if (!strcmp(argv[0], "zero")) {
				set_zero_mode(true);
				return 0;
			}

			if (!strcmp(argv[0], "nozero")) {
				set_zero_mode(false);
				return 0;
			}

			if (!strcmp(argv[0], "trim")) {
				return trim_command(argc, argv);
			}
		}

		return print_usage("unknown command");
	}

	static int print_usage(const char *reason = nullptr)
	{
		if (reason) { PX4_WARN("%s", reason); }
		PRINT_MODULE_DESCRIPTION(R"DESCR_STR(
### Description
Publishes actuator_servos from tvc_gimbal_cmd for 6 gimbal servos.
TVC_TRIM_0..5 offset rest. tvc_actuators zero holds rest for pad centering.
)DESCR_STR");
		PRINT_MODULE_USAGE_NAME("tvc_actuators", "driver");
		PRINT_MODULE_USAGE_COMMAND("start");
		PRINT_MODULE_USAGE_COMMAND_DESCR("zero", "Hold trimmed rest, PWM live disarmed, no fire");
		PRINT_MODULE_USAGE_COMMAND_DESCR("nozero", "Leave zero-gimbal mode");
		PRINT_MODULE_USAGE_COMMAND_DESCR("trim", "Print trims, or: trim <0-5> <deg>");
		PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
		return 0;
	}

	bool init()
	{
		// Claim actuator_servos instance 0 before any leftover
		// control_allocator can advertise and steal PWM.
		_servos_pub.advertise();
		ScheduleOnInterval(10_ms); // 100 Hz
		return true;
	}

	int print_status() override
	{
		PX4_INFO("zero=%d rev=%d a2=%.1f a1=%.1f",
			 (int)_param_zero.get(), (int)_param_rev.get(),
			 (double)_param_a2_range.get(), (double)_param_a1_range.get());
		PX4_INFO("trim deg: M0a2=%.1f M0a1=%.1f M1a2=%.1f M1a1=%.1f M2a2=%.1f M2a1=%.1f",
			 (double)_param_trim0.get(), (double)_param_trim1.get(),
			 (double)_param_trim2.get(), (double)_param_trim3.get(),
			 (double)_param_trim4.get(), (double)_param_trim5.get());
		return 0;
	}

private:
	static void set_zero_mode(bool on)
	{
		int32_t one = 1;
		int32_t two = 2;
		int32_t zero = 0;

		if (on) {
			param_set(param_find("TVC_ZERO"), &one);
			param_set(param_find("COM_PREARM_MODE"), &two);
			param_set(param_find("TVC_BENCH"), &zero);
			param_set(param_find("TVC_ALLOW_FIRE"), &zero);
			param_set(param_find("TVC_IGN_GPIO_EN"), &zero);
			PX4_WARN("tvc_actuators: ZERO on (rest+trim, PWM live, launch-ready denied). Adjust TVC_TRIM_0..5, then nozero + param save");
		} else {
			param_set(param_find("TVC_ZERO"), &zero);
			int32_t bench = 0;
			param_get(param_find("TVC_BENCH"), &bench);

			if (bench == 0) {
				param_set(param_find("COM_PREARM_MODE"), &zero);
			}

			PX4_INFO("tvc_actuators: ZERO off. param save to keep TVC_TRIM_* across reboot");
		}

		TvcActuators *inst = _object.load();

		if (inst != nullptr) {
			inst->updateParams();
			inst->sync_disarmed_pwm();
			inst->_zero_side_applied = on;
		}
	}

	static int trim_command(int argc, char *argv[])
	{
		if (argc < 3) {
			for (int ch = 0; ch < 6; ch++) {
				float deg = 0.f;
				param_get(param_find(kTrimNames[ch]), &deg);
				PX4_INFO("%s %s = %.1f deg", kTrimNames[ch], kChNames[ch], (double)deg);
			}

			return 0;
		}

		const int ch = atoi(argv[1]);

		if (ch < 0 || ch > 5) {
			PX4_ERR("trim channel must be 0..5 (M0a2..M2a1)");
			return PX4_ERROR;
		}

		float deg = strtof(argv[2], nullptr);
		deg = tvc::clampf(deg, -tvc::kServoTrimMaxDeg, tvc::kServoTrimMaxDeg);
		param_set(param_find(kTrimNames[ch]), &deg);
		PX4_INFO("%s (%s) = %.1f deg", kTrimNames[ch], kChNames[ch], (double)deg);
		TvcActuators *inst = _object.load();

		if (inst != nullptr) {
			inst->updateParams();
			inst->sync_disarmed_pwm();
		}

		return 0;
	}

	static int pwm_param_int(const char *prefix, int main_ch, int fallback)
	{
		char name[20];
		snprintf(name, sizeof(name), "%s%d", prefix, main_ch);
		param_t p = param_find(name);

		if (p == PARAM_INVALID) {
			return fallback;
		}

		int32_t v = fallback;
		param_get(p, &v);
		return static_cast<int>(v);
	}

	static void set_pwm_param_int(const char *prefix, int main_ch, int value)
	{
		char name[20];
		snprintf(name, sizeof(name), "%s%d", prefix, main_ch);
		param_t p = param_find(name);

		if (p == PARAM_INVALID) {
			return;
		}

		int32_t cur = 0;
		param_get(p, &cur);

		if (cur != value) {
			int32_t v = value;
			param_set(p, &v);
		}
	}

	float trim_deg(int ch) const
	{
		switch (ch) {
		case 0: return _param_trim0.get();
		case 1: return _param_trim1.get();
		case 2: return _param_trim2.get();
		case 3: return _param_trim3.get();
		case 4: return _param_trim4.get();
		case 5: return _param_trim5.get();
		default: return 0.f;
		}
	}

	void sync_disarmed_pwm()
	{
		const float a2_range = math::radians(_param_a2_range.get());
		const float a1_range = math::radians(_param_a1_range.get());
		const int rev = _param_rev.get();

		for (int ch = 0; ch < 6; ch++) {
			const float ang = (ch % 2) == 0 ? 0.f : math::radians(_param_a1_rest.get());
			const float n = tvc::gimbal_servo_norm(ch, ang, trim_deg(ch), a2_range, a1_range, rev);
			const int main_ch = tvc::servo_logical_to_main(ch);
			const int min_us = pwm_param_int("PWM_MAIN_MIN", main_ch, 800);
			const int max_us = pwm_param_int("PWM_MAIN_MAX", main_ch, 2200);
			const int us = tvc::servo_norm_to_pwm_us(n, min_us, max_us);
			set_pwm_param_int("PWM_MAIN_DIS", main_ch, us);
			set_pwm_param_int("PWM_MAIN_FAIL", main_ch, us);
		}
	}

	void apply_zero_side_effects(bool on)
	{
		int32_t two = 2;
		int32_t zero = 0;

		if (on) {
			param_set(param_find("COM_PREARM_MODE"), &two);
			param_set(param_find("TVC_BENCH"), &zero);
			param_set(param_find("TVC_ALLOW_FIRE"), &zero);
			param_set(param_find("TVC_IGN_GPIO_EN"), &zero);
		} else {
			int32_t bench = 0;
			param_get(param_find("TVC_BENCH"), &bench);

			if (bench == 0) {
				param_set(param_find("COM_PREARM_MODE"), &zero);
			}
		}
	}

	void handle_gcs_commands()
	{
		vehicle_command_s cmd{};

		while (_vehicle_command_sub.update(&cmd)) {
			if (!tvc::is_external_gcs_command(cmd) ||
			    cmd.command != static_cast<uint32_t>(tvc::kMavCmdTvc)) {
				continue;
			}

			const tvc::GcsAction action = tvc::gcs_action_from_param1(cmd.param1);

			if (action == tvc::GcsAction::Zero) {
				set_zero_mode(true);
				tvc::ack_vehicle_command(_vehicle_command_ack_pub, cmd,
						       vehicle_command_ack_s::VEHICLE_CMD_RESULT_ACCEPTED);
			} else if (action == tvc::GcsAction::NoZero) {
				set_zero_mode(false);
				tvc::ack_vehicle_command(_vehicle_command_ack_pub, cmd,
						       vehicle_command_ack_s::VEHICLE_CMD_RESULT_ACCEPTED);
			}
		}
	}

	void Run() override
	{
		if (should_exit()) {
			ScheduleClear();
			exit_and_cleanup();
			return;
		}

		if (_parameter_update_sub.updated()) {
			parameter_update_s p; _parameter_update_sub.copy(&p);
			updateParams();
			sync_disarmed_pwm();
		}

		if (!_dis_synced) {
			updateParams();
			sync_disarmed_pwm();
			_dis_synced = true;
		}

		handle_gcs_commands();

		const bool zero = _param_zero.get() != 0;

		if (zero != _zero_side_applied) {
			apply_zero_side_effects(zero);
			_zero_side_applied = zero;
		}

		tvc_gimbal_cmd_s cmd{};
		if (!_gimbal_sub.updated() && !_gimbal_sub.copy(&cmd)) {
			// hold last / NaN = disarmed mapping when never received
			if (!zero) {
				return;
			}
		} else {
			_gimbal_sub.copy(&cmd);
		}

		// Force rest when disarmed or in SAFE / DISARMED / PREFLIGHT.
		// LAND stays on the gimbal command: the loops keep running after touchdown.
		// TVC_BENCH: keep following tvc_gimbal_cmd (COM_PREARM_MODE=2 for pin output).
		// TVC_ZERO: always trimmed rest (pad centering).
		const bool bench = (_param_bench.get() != 0) && !zero;
		bool force_rest = zero;

		if (!bench && !zero) {
			vehicle_status_s vstatus{};
			if (_vehicle_status_sub.copy(&vstatus)) {
				if (vstatus.arming_state != vehicle_status_s::ARMING_STATE_ARMED) {
					force_rest = true;
				}
			}
			tvc_vehicle_state_s st{};
			if (_tvc_state_sub.copy(&st)) {
				if (st.state == tvc_vehicle_state_s::STATE_DISARMED ||
				    st.state == tvc_vehicle_state_s::STATE_PREFLIGHT ||
				    st.state == tvc_vehicle_state_s::STATE_SAFE) {
					force_rest = true;
				}
			}
		}

		const float a2_range = math::radians(_param_a2_range.get());
		const float a1_range = math::radians(_param_a1_range.get());
		const int rev = _param_rev.get();

		actuator_servos_s servos{};
		servos.timestamp = hrt_absolute_time();
		servos.timestamp_sample = cmd.timestamp;

		for (int i = 0; i < 8; i++) {
			servos.control[i] = NAN;
		}

		for (int ch = 0; ch < 6; ch++) {
			float ang = cmd.angles_rad[ch];
			if (force_rest) {
				ang = (ch % 2) == 0 ? 0.f : math::radians(_param_a1_rest.get());
			}
			servos.control[ch] = tvc::gimbal_servo_norm(ch, ang, trim_deg(ch),
					     a2_range, a1_range, rev);
		}

		_servos_pub.publish(servos);
	}

	uORB::Subscription _parameter_update_sub{ORB_ID(parameter_update)};
	uORB::Subscription _gimbal_sub{ORB_ID(tvc_gimbal_cmd)};
	uORB::Subscription _tvc_state_sub{ORB_ID(tvc_vehicle_state)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _vehicle_command_sub{ORB_ID(vehicle_command)};
	uORB::Publication<actuator_servos_s> _servos_pub{ORB_ID(actuator_servos)};
	uORB::Publication<vehicle_command_ack_s> _vehicle_command_ack_pub{ORB_ID(vehicle_command_ack)};

	bool _dis_synced{false};
	bool _zero_side_applied{false};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::TVC_A2_RANGE>) _param_a2_range,
		(ParamFloat<px4::params::TVC_A1_RANGE>) _param_a1_range,
		(ParamFloat<px4::params::TVC_A1_REST>) _param_a1_rest,
		(ParamInt<px4::params::TVC_SERVO_REV>) _param_rev,
		(ParamInt<px4::params::TVC_BENCH>) _param_bench,
		(ParamInt<px4::params::TVC_ZERO>) _param_zero,
		(ParamFloat<px4::params::TVC_TRIM_0>) _param_trim0,
		(ParamFloat<px4::params::TVC_TRIM_1>) _param_trim1,
		(ParamFloat<px4::params::TVC_TRIM_2>) _param_trim2,
		(ParamFloat<px4::params::TVC_TRIM_3>) _param_trim3,
		(ParamFloat<px4::params::TVC_TRIM_4>) _param_trim4,
		(ParamFloat<px4::params::TVC_TRIM_5>) _param_trim5
	)
};

extern "C" __EXPORT int tvc_actuators_main(int argc, char *argv[])
{
	return TvcActuators::main(argc, argv);
}
