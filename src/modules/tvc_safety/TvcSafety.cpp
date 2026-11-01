/**
 * @file TvcSafety.cpp
 *
 * TVC vehicle state machine + inhibit gates.
 * Cube does not ignite — a disconnected ground system lights the grains.
 * LAUNCH_READY waits for first motion (or HIL software fire) to start BURN.
 * Requests commander arm for PWM unlock (pattern A).
 *
 * Abort reasons (tvc_vehicle_state.abort_reason):
 *  0 none / user
 *  1 max rate
 *  2 geofence
 *  3 hard landing
 *  4 hang-fire / partial ignition
 *  5 max tilt
 *  6 allocator timeouts
 *  7 (reserved link)
 *  8 rail voltage
 *
 * Automatic reasons 1–6 run only when TVC_AUTO_ABORT=1 (default 0).
 * Operator abort (reason 9) is always available.
 */

#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>
#include <drivers/drv_hrt.h>
#include <lib/mathlib/mathlib.h>
#include <matrix/math.hpp>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionInterval.hpp>
#include <uORB/SubscriptionMultiArray.hpp>
#include <uORB/topics/estimator_status_flags.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/sensor_gps.h>
#include <uORB/topics/system_power.h>
#include <systemlib/mavlink_log.h>
#include <uORB/topics/debug_key_value.h>
#include <uORB/topics/input_rc.h>
#include <uORB/topics/telemetry_status.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_command_ack.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/tvc_gimbal_cmd.h>
#include <uORB/topics/tvc_inhibit_flags.h>
#include <uORB/topics/tvc_mount_thrust.h>
#include <uORB/topics/tvc_vehicle_state.h>

#include <tvc/first_motion.hpp>
#include <tvc/mission_config.hpp>
#include <tvc/gcs_cmd.hpp>
#include <tvc/px4_gcs.hpp>
#include <tvc/types.hpp>
#include <tvc/vehicle_config.hpp>

#include <cstring>

using namespace time_literals;
using matrix::Eulerf;
using matrix::Quatf;
using matrix::Dcmf;

class TvcSafety : public ModuleBase<TvcSafety>, public ModuleParams, public px4::ScheduledWorkItem
{
public:
	TvcSafety() :
		ModuleParams(nullptr),
		ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::hp_default)
	{
		_state.state = tvc_vehicle_state_s::STATE_DISARMED;
	}

	static int task_spawn(int argc, char *argv[])
	{
		TvcSafety *inst = new TvcSafety();

		if (!inst) {
			return PX4_ERROR;
		}

		_object.store(inst);
		_task_id = task_id_is_work_queue;

		if (!inst->init()) {
			delete inst;
			_object.store(nullptr);
			_task_id = -1;
			return PX4_ERROR;
		}

		return PX4_OK;
	}

	static int custom_command(int argc, char *argv[])
	{
		if (argc >= 1) {
			TvcSafety *inst = get_instance();

			if (!inst) {
				return 1;
			}

			if (!strcmp(argv[0], "arm")) { inst->_cmd_arm = true; return 0; }
			if (!strcmp(argv[0], "disarm")) { inst->_cmd_disarm = true; return 0; }
			if (!strcmp(argv[0], "ign_arm") || !strcmp(argv[0], "launch_ready")) {
				inst->_cmd_ign_arm = true;
				return 0;
			}
			if (!strcmp(argv[0], "fire")) { inst->_cmd_fire = true; return 0; }
			if (!strcmp(argv[0], "abort")) { inst->_cmd_abort = true; return 0; }
			if (!strcmp(argv[0], "reset")) { inst->_cmd_reset = true; return 0; }
			if (!strcmp(argv[0], "origin")) { inst->_cmd_origin = true; return 0; }
		}

		return print_usage("unknown");
	}

	static int print_usage(const char *reason = nullptr)
	{
		if (reason) {
			PX4_WARN("%s", reason);
		}

		PRINT_MODULE_DESCRIPTION(R"DESCR_STR(
### Description
TVC safety FSM with arm gates, hang-fire, hard-landing, tilt/rate aborts.
Cube does not ignite. launch_ready waits for first motion to start BURN.
NSH: arm, disarm, ign_arm/launch_ready, fire (HIL only), abort, reset.
QGC: MAV_CMD_USER_1 (31010) param1=1..6 (HIL fire needs param2=21196).
QGC Arm/Disarm and flight-termination also map onto the FSM.
)DESCR_STR");
		PRINT_MODULE_USAGE_NAME("tvc_safety", "system");
		PRINT_MODULE_USAGE_COMMAND("start");
		PRINT_MODULE_USAGE_COMMAND_DESCR("arm", "TVC arm + commander force-arm (PWM unlock)");
		PRINT_MODULE_USAGE_COMMAND_DESCR("disarm", "Disarm if pre-liftoff");
		PRINT_MODULE_USAGE_COMMAND_DESCR("ign_arm", "Enter LAUNCH_READY (alias: launch_ready)");
		PRINT_MODULE_USAGE_COMMAND_DESCR("launch_ready", "Enter LAUNCH_READY; wait for first motion");
		PRINT_MODULE_USAGE_COMMAND_DESCR("fire", "HIL software liftoff (no onboard ignition)");
		PRINT_MODULE_USAGE_COMMAND_DESCR("abort", "ABORT / rest latch");
		PRINT_MODULE_USAGE_COMMAND_DESCR("reset", "SAFE → DISARMED (crew)");
		PRINT_MODULE_USAGE_COMMAND_DESCR("origin", "Recapture the landing pad in PREFLIGHT");
		PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
		return 0;
	}

	int print_status() override
	{
		const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
		PX4_INFO("state=%u abort=%u arm_ok=%d ignite_ok=%d fire=%d",
			 (unsigned)_state.state, (unsigned)_state.abort_reason,
			 (int)_state.arm_ok, (int)_state.ignite_ok, (int)_state.fire_latched);
		PX4_INFO("gates rtk=%d ekf=%d gim=%d geo=%d link=%d alloc=%d thr=%d rail=%d mass=%d",
			 (int)_last_flags.rtk_ok, (int)_last_flags.ekf_ok, (int)_last_flags.gimbal_rest_ok,
			 (int)_last_flags.geofence_ok, (int)_last_flags.link_ok, (int)_last_flags.alloc_ok,
			 (int)_last_flags.thrust_ok, (int)_last_flags.rail_v_ok, (int)_last_flags.mass_cfg_ok);
		PX4_INFO("vehicle valid=%d status=%u crc=0x%08x loaded_kg=%.3f TVC_MASS=%.3f",
			 (int)vcfg.valid, (unsigned)vcfg.status, (unsigned)vcfg.crc,
			 (double)vcfg.loaded_mass_kg, (double)_param_mass.get());
		return 0;
	}

	bool init()
	{
		tvc::VehicleConfig vcfg;
		if (!tvc::load_vehicle_config_file(tvc::kVehicleJsonPath, tvc::kVehicleCrcPath, vcfg) ||
		    !vcfg.valid) {
			PX4_ERR("vehicle.json missing/invalid status=%u — arm blocked",
				(unsigned)vcfg.status);
		}
		ScheduleOnInterval(20_ms); // 50 Hz
		return true;
	}

private:
	// abort_reason codes
	static constexpr uint8_t AR_NONE = 0;
	static constexpr uint8_t AR_MAX_RATE = 1;
	static constexpr uint8_t AR_GEOFENCE = 2;
	static constexpr uint8_t AR_HARD_LAND = 3;
	static constexpr uint8_t AR_HANGFIRE = 4;
	static constexpr uint8_t AR_MAX_TILT = 5;
	static constexpr uint8_t AR_ALLOC = 6;
	static constexpr uint8_t AR_RAIL = 8;
	static constexpr uint8_t AR_USER = 9;

	void request_commander_arm(bool arm)
	{
		vehicle_command_s cmd{};
		cmd.timestamp = hrt_absolute_time();
		cmd.command = vehicle_command_s::VEHICLE_CMD_COMPONENT_ARM_DISARM;
		cmd.param1 = arm ? 1.f : 0.f;

		// Magic 21196: force arm/disarm, skip stock MC preflight (servo-only TVC)
		if (arm && _param_force_arm.get() != 0) {
			cmd.param2 = 21196.f;
		}

		cmd.target_system = 1;
		cmd.target_component = 1;
		cmd.source_system = 1;
		cmd.source_component = 1;
		cmd.from_external = false;
		_vehicle_command_pub.publish(cmd);
	}

	void request_abort(uint8_t reason)
	{
		if (_state.state >= tvc_vehicle_state_s::STATE_ARMED &&
		    _state.state != tvc_vehicle_state_s::STATE_SAFE) {
			_cmd_abort = true;
			_state.abort_reason = reason;
		}
	}

	static uint16_t inhibit_bits(const tvc_inhibit_flags_s &flags)
	{
		uint16_t b = 0;
		if (flags.rtk_ok) { b |= 1u << 0; }
		if (flags.ekf_ok) { b |= 1u << 1; }
		if (flags.gimbal_rest_ok) { b |= 1u << 2; }
		if (flags.geofence_ok) { b |= 1u << 3; }
		if (flags.link_ok) { b |= 1u << 4; }
		if (flags.alloc_ok) { b |= 1u << 5; }
		if (flags.thrust_ok) { b |= 1u << 6; }
		if (flags.rail_v_ok) { b |= 1u << 7; }
		if (flags.mass_cfg_ok) { b |= 1u << 8; }
		if (flags.mission_ok) { b |= 1u << 9; }
		if (flags.origin_ok) { b |= 1u << 10; }
		return b;
	}

	void ack_cmd(const vehicle_command_s &cmd, uint8_t result)
	{
		tvc::ack_vehicle_command(_vehicle_command_ack_pub, cmd, result);
	}

	void handle_vehicle_commands()
	{
		vehicle_command_s cmd{};

		while (_vehicle_command_sub.update(&cmd)) {
			if (!tvc::is_external_gcs_command(cmd)) {
				continue;
			}

			if (cmd.command == vehicle_command_s::VEHICLE_CMD_COMPONENT_ARM_DISARM) {
				_gcs_arm_grace_until = hrt_absolute_time() + 500_ms;

				if (cmd.param1 >= 0.5f) {
					_cmd_arm = true;
				} else {
					_cmd_disarm = true;
				}

				continue;
			}

			if (cmd.command == static_cast<uint32_t>(tvc::kMavCmdFlightTermination) &&
			    cmd.param1 >= 0.5f) {
				_cmd_abort = true;
				if (_state.abort_reason == AR_NONE) {
					_state.abort_reason = AR_USER;
				}
				ack_cmd(cmd, vehicle_command_ack_s::VEHICLE_CMD_RESULT_ACCEPTED);
				continue;
			}

			if (cmd.command != static_cast<uint32_t>(tvc::kMavCmdTvc)) {
				continue;
			}

			const tvc::GcsAction action = tvc::gcs_action_from_param1(cmd.param1);

			if (!tvc::gcs_action_is_fsm(action)) {
				// bench / reload / zero are owned by other tvc_* modules
				continue;
			}

			uint8_t result = vehicle_command_ack_s::VEHICLE_CMD_RESULT_ACCEPTED;

			switch (action) {
			case tvc::GcsAction::Arm:
				_gcs_arm_grace_until = hrt_absolute_time() + 500_ms;
				_cmd_arm = true;
				break;

			case tvc::GcsAction::Disarm:
				_cmd_disarm = true;
				break;

			case tvc::GcsAction::IgnArm:
				_cmd_ign_arm = true;
				break;

			case tvc::GcsAction::Fire:
				if (!tvc::fire_magic_ok(cmd.param2)) {
					result = vehicle_command_ack_s::VEHICLE_CMD_RESULT_DENIED;
					mavlink_log_critical(&_mavlink_log_pub,
							     "TVC: fire denied (need param2=21196)");
					PX4_WARN("tvc_safety: QGC fire missing magic param2");
				} else {
					_cmd_fire = true;
				}
				break;

			case tvc::GcsAction::Abort:
				_cmd_abort = true;
				if (_state.abort_reason == AR_NONE) {
					_state.abort_reason = AR_USER;
				}
				break;

			case tvc::GcsAction::Reset:
				_cmd_reset = true;
				break;

			default:
				result = vehicle_command_ack_s::VEHICLE_CMD_RESULT_UNSUPPORTED;
				break;
			}

			ack_cmd(cmd, result);
		}
	}

	void check_rc_abort()
	{
		const int ch = _param_abort_ch.get();

		if (ch < 1 || ch > 18) {
			return;
		}

		input_rc_s rc{};

		if (!_rc_sub.copy(&rc)) {
			return;
		}

		const unsigned nmax = sizeof(rc.values) / sizeof(rc.values[0]);
		const unsigned n = math::min(static_cast<unsigned>(rc.channel_count), nmax);

		if ((unsigned)ch > n) {
			return;
		}

		// Rising edge on a high PWM switch. COM_RC_IN_MODE=1 still publishes input_rc.
		const bool high = rc.values[ch - 1] > 1800;
		const bool rise = high && !_rc_abort_high;
		_rc_abort_high = high;

		if (rise) {
			request_abort(AR_USER);
			mavlink_log_emergency(&_mavlink_log_pub, "TVC: RC abort CH%u", (unsigned)ch);
			PX4_WARN("tvc_safety: RC abort channel %d", ch);
		}
	}

	void hold_commander_while_burning()
	{
		if (_state.state != tvc_vehicle_state_s::STATE_BURN &&
		    _state.state != tvc_vehicle_state_s::STATE_COAST &&
		    _state.state != tvc_vehicle_state_s::STATE_LAND &&
		    _state.state != tvc_vehicle_state_s::STATE_ABORT) {
			return;
		}

		vehicle_status_s vs{};

		if (!_status_sub.copy(&vs)) {
			return;
		}

		if (vs.arming_state != vehicle_status_s::ARMING_STATE_ARMED) {
			request_commander_arm(true);
			_gcs_arm_grace_until = hrt_absolute_time() + 2_s;
		}
	}

	void block_stock_arm_without_tvc()
	{
		if (_state.state >= tvc_vehicle_state_s::STATE_ARMED) {
			return;
		}

		if (hrt_absolute_time() < _gcs_arm_grace_until) {
			return;
		}

		vehicle_status_s vs{};

		if (!_status_sub.copy(&vs)) {
			return;
		}

		if (vs.arming_state == vehicle_status_s::ARMING_STATE_ARMED) {
			request_commander_arm(false);
			mavlink_log_critical(&_mavlink_log_pub, "TVC: blocked stock arm — use TVC Arm");
			PX4_WARN("tvc_safety: commander armed without TVC ARMED — disarmed");
		}
	}

	void publish_named_values(const tvc_inhibit_flags_s &flags)
	{
		if (_last_named != 0 && hrt_elapsed_time(&_last_named) < 100_ms) {
			return;
		}

		_last_named = hrt_absolute_time();

		float a2max = 0.f;
		tvc_gimbal_cmd_s gim{};

		if (_gimbal_sub.copy(&gim)) {
			for (int m = 0; m < 3; m++) {
				a2max = math::max(a2max, fabsf(math::degrees(gim.angles_rad[m * 2])));
			}
		}

		float t0 = 0.f, t1 = 0.f, t2 = 0.f;
		tvc_mount_thrust_s thr{};

		if (_thrust_sub.copy(&thr)) {
			t0 = thr.thrust[0];
			t1 = thr.thrust[1];
			t2 = thr.thrust[2];
		}

		const float values[11] = {
			static_cast<float>(_state.state),
			static_cast<float>(_state.abort_reason),
			_state.arm_ok ? 1.f : 0.f,
			_state.ignite_ok ? 1.f : 0.f,
			_state.fire_latched ? 1.f : 0.f,
			static_cast<float>(inhibit_bits(flags)),
			body_tilt_deg(),
			t0, t1, t2, a2max
		};
		static constexpr const char *keys[11] = {
			"tvc_st", "tvc_ab", "tvc_ok", "tvc_ig", "tvc_fr",
			"tvc_gt", "tvc_tl", "tvc_t0", "tvc_t1", "tvc_t2", "tvc_a2"
		};

		debug_key_value_s dbg{};
		dbg.timestamp = hrt_absolute_time();
		strncpy(dbg.key, keys[_named_i], sizeof(dbg.key) - 1);
		dbg.key[sizeof(dbg.key) - 1] = '\0';
		dbg.value = values[_named_i];
		_named_pub.publish(dbg);
		_named_i = static_cast<uint8_t>((_named_i + 1) % 11);
	}

	void publish_gcs_status(const tvc_inhibit_flags_s &flags)
	{
		if (_state.state != _last_announced_state) {
			_last_announced_state = _state.state;

			if (_state.state == tvc_vehicle_state_s::STATE_BURN) {
				mavlink_log_emergency(&_mavlink_log_pub, "TVC: LIFTOFF → BURN");
			} else if (_state.state == tvc_vehicle_state_s::STATE_ABORT) {
				mavlink_log_emergency(&_mavlink_log_pub, "TVC: ABORT reason=%u",
						      (unsigned)_state.abort_reason);
			} else {
				mavlink_log_critical(&_mavlink_log_pub, "TVC: %s abort=%u",
						     tvc::vehicle_state_name(_state.state),
						     (unsigned)_state.abort_reason);
			}
		}

		const hrt_abstime period =
			(_state.state >= tvc_vehicle_state_s::STATE_LAUNCH_READY &&
			 _state.state <= tvc_vehicle_state_s::STATE_ABORT) ? 1_s : 2_s;

		if (_last_gcs_status != 0 && hrt_elapsed_time(&_last_gcs_status) < period) {
			return;
		}

		_last_gcs_status = hrt_absolute_time();
		mavlink_log_info(&_mavlink_log_pub,
				 "TVC %s a=%d i=%d f=%d g=%03x tl=%.1f",
				 tvc::vehicle_state_name(_state.state),
				 (int)_state.arm_ok, (int)_state.ignite_ok, (int)_state.fire_latched,
				 (unsigned)inhibit_bits(flags), (double)body_tilt_deg());
	}

	float height_agl_m(const vehicle_local_position_s &lpos) const
	{
		if (lpos.dist_bottom_valid) {
			return lpos.dist_bottom;
		}

		// NED down. Height above the captured pad, else above the EKF origin.
		if (_origin_set && lpos.z_valid) {
			return _pad_z - lpos.z;
		}

		if (lpos.z_valid) {
			return -lpos.z;
		}

		return 1e6f;
	}

	float body_tilt_deg()
	{
		vehicle_attitude_s att{};

		if (!_att_sub.copy(&att)) {
			return 0.f;
		}

		// Body +Z (down FRD) vs world +Z (down NED): tilt of thrust axis from vertical
		const Quatf q(att.q);
		const Dcmf R(q);
		// Body z-axis in NED
		const float bz_x = R(0, 2);
		const float bz_y = R(1, 2);
		const float bz_z = R(2, 2);
		const float cos_t = math::constrain(bz_z, -1.f, 1.f);
		// When upright, body +Z aligns with NED +Z → cos≈1, tilt=0
		const float tilt_rad = acosf(cos_t);
		(void)bz_x;
		(void)bz_y;
		return math::degrees(tilt_rad);
	}

	bool link_heartbeat_ok()
	{
		bool any_gcs = false;
		hrt_abstime newest = 0;

		for (auto &sub : _telem_subs) {
			telemetry_status_s ts{};

			if (!sub.copy(&ts)) {
				continue;
			}

			if (ts.heartbeat_type_gcs || ts.heartbeat_component_telemetry_radio) {
				any_gcs = true;

				if (ts.timestamp > newest) {
					newest = ts.timestamp;
				}
			}
		}

		if (!any_gcs || newest == 0) {
			return false;
		}

		// Heartbeat path updates telemetry_status regularly; treat >2 s as dead
		return hrt_elapsed_time(&newest) < 2_s;
	}

	void evaluate_gates(tvc_inhibit_flags_s &flags)
	{
		flags.timestamp = hrt_absolute_time();
		flags.rtk_ok = true;
		flags.ekf_ok = true;
		flags.gimbal_rest_ok = true;
		flags.geofence_ok = true;
		flags.link_ok = true;
		flags.alloc_ok = true;
		flags.thrust_ok = true;
		flags.rail_v_ok = true;
		flags.mass_cfg_ok = false;
		{
			const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
			flags.mass_cfg_ok = vcfg.valid && vcfg.loaded_mass_kg > 0.f;
		}
		flags.mission_ok = tvc::loaded_mission_config().valid;
		flags.origin_ok = (_param_req_rtk.get() == 0) || _origin_set;

		// --- mass ---
		// already set

		// --- RTK ---
		if (_param_req_rtk.get() != 0) {
			sensor_gps_s gps{};

			if (_gps_sub.copy(&gps)) {
				flags.rtk_ok = (gps.fix_type >= 6); // RTK fixed
			} else {
				flags.rtk_ok = false;
			}
		}

		// --- EKF ---
		estimator_status_flags_s est{};

		if (_est_flags_sub.copy(&est)) {
			flags.ekf_ok = est.cs_tilt_align && est.cs_yaw_align &&
				       est.cs_gps && !est.cs_inertial_dead_reckoning;
		} else {
			flags.ekf_ok = false;
		}

		vehicle_local_position_s lpos{};
		const bool have_lpos = _lpos_sub.copy(&lpos);

		if (have_lpos) {
			flags.ekf_ok = flags.ekf_ok && lpos.xy_valid && lpos.z_valid && lpos.v_xy_valid;
		} else {
			flags.ekf_ok = false;
		}

		// --- Geofence (optionally tighten on RTK loss in BURN) ---
		float R = _param_geofence_r.get();

		if (R > 0.f) {
			if (_state.state == tvc_vehicle_state_s::STATE_BURN && !flags.rtk_ok) {
				R *= 0.7f;
			}

			if (have_lpos && lpos.xy_valid) {
				const float dx = lpos.x - (_origin_set ? _pad_x : 0.f);
				const float dy = lpos.y - (_origin_set ? _pad_y : 0.f);
				const float r = sqrtf(dx * dx + dy * dy);
				flags.geofence_ok = (r <= R);
			} else {
				// no position → not OK when fence enabled
				flags.geofence_ok = false;
			}
		}

		// --- Link ---
		if (_param_req_link.get() != 0) {
			flags.link_ok = link_heartbeat_ok();
		}

		// --- Gimbal rest ---
		{
			tvc_gimbal_cmd_s cmd{};
			const float tol = math::radians(_param_gim_rest.get());
			const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
			const float rest_a2 = vcfg.valid ? tvc::deg2rad(vcfg.a2_rest_deg) : 0.f;
			const float rest_a1 = vcfg.valid ? tvc::deg2rad(vcfg.a1_rest_deg) : tvc::deg2rad(0.f);

			if (_gimbal_sub.copy(&cmd) && hrt_elapsed_time(&cmd.timestamp) < 200_ms) {
				bool ok = true;

				for (int m = 0; m < 3; m++) {
					const float a2 = cmd.angles_rad[m * 2 + 0];
					const float a1 = cmd.angles_rad[m * 2 + 1];

					if (fabsf(a2 - rest_a2) > tol || fabsf(a1 - rest_a1) > tol) {
						ok = false;
						break;
					}
				}

				flags.gimbal_rest_ok = ok;
			} else {
				// No recent cmd: treat as rest OK in DISARMED/PREFLIGHT (actuators hold rest)
				// but not OK once armed if stream dies
				flags.gimbal_rest_ok = (_state.state <= tvc_vehicle_state_s::STATE_PREFLIGHT);
			}
		}

		// --- Allocator health ---
		{
			tvc_gimbal_cmd_s cmd{};

			if (_gimbal_sub.copy(&cmd) && hrt_elapsed_time(&cmd.timestamp) < 200_ms) {
				if (cmd.timed_out) {
					_alloc_fail_streak++;
				} else {
					_alloc_fail_streak = 0;
				}

				// arm gate: not continuously timing out
				flags.alloc_ok = (_alloc_fail_streak < 3) && !cmd.timed_out;
			} else if (_state.state >= tvc_vehicle_state_s::STATE_ARMED) {
				flags.alloc_ok = false;
			}
		}

		// --- Thrust topic (curve publisher must be alive) ---
		{
			tvc_mount_thrust_s thr{};

			if (_thrust_sub.copy(&thr)) {
				flags.thrust_ok = hrt_elapsed_time(&thr.timestamp) <= 500_ms;
			} else if (_state.state >= tvc_vehicle_state_s::STATE_ARMED) {
				flags.thrust_ok = false;
			}
		}

		// --- Rail 5V ---
		{
			const float vmin = _param_rail_vmin.get();

			if (vmin > 0.f) {
				system_power_s pwr{};

				if (_power_sub.copy(&pwr) && hrt_elapsed_time(&pwr.timestamp) < 1_s) {
					flags.rail_v_ok = (pwr.voltage5v_v >= vmin) && (pwr.servo_valid != 0 || pwr.voltage5v_v > 0.f);
				} else {
					// No system_power sample: fail closed only if we ever saw power
					flags.rail_v_ok = !_saw_system_power;
				}

				if (_power_sub.copy(&pwr) && pwr.voltage5v_v > 0.f) {
					_saw_system_power = true;
				}
			}
		}

		_last_flags = flags;
	}

	bool gates_for_arm(const tvc_inhibit_flags_s &flags) const
	{
		return flags.rtk_ok && flags.ekf_ok && flags.geofence_ok &&
		       flags.mass_cfg_ok && flags.mission_ok && flags.origin_ok &&
		       flags.link_ok &&
		       flags.gimbal_rest_ok && flags.alloc_ok &&
		       flags.thrust_ok && flags.rail_v_ok;
	}

	void check_burn_aborts(const tvc_inhibit_flags_s &flags)
	{
		// Default off. These trips park the nozzles while the grains are
		// still burning; they do not extinguish anything.
		if (_param_auto_abort.get() == 0) {
			return;
		}

		if (_state.state != tvc_vehicle_state_s::STATE_BURN &&
		    _state.state != tvc_vehicle_state_s::STATE_COAST) {
			return;
		}

		// Max rate (BURN + COAST)
		vehicle_angular_velocity_s ang{};

		if (_ang_sub.copy(&ang)) {
			const float rate_dps = math::degrees(sqrtf(
								     ang.xyz[0] * ang.xyz[0] +
								     ang.xyz[1] * ang.xyz[1] +
								     ang.xyz[2] * ang.xyz[2]));

			if (rate_dps > _param_max_rate.get()) {
				request_abort(AR_MAX_RATE);
				return;
			}
		}

		// Max tilt
		const float tilt = body_tilt_deg();

		if (tilt > _param_max_tilt.get()) {
			request_abort(AR_MAX_TILT);
			return;
		}

		// Geofence
		if (!flags.geofence_ok && _param_geofence_r.get() > 0.f) {
			request_abort(AR_GEOFENCE);
			return;
		}

		// Allocator consecutive timeouts (BURN only)
		if (_state.state == tvc_vehicle_state_s::STATE_BURN) {
			const int n_lim = _param_alloc_n.get();

			if (n_lim > 0 && _alloc_fail_streak >= n_lim) {
				request_abort(AR_ALLOC);
				return;
			}
		}

		// Hard landing: BURN only while any lit / not all burned out
		if (_state.state == tvc_vehicle_state_s::STATE_BURN) {
			tvc_mount_thrust_s thr{};
			bool any_active = false;

			if (_thrust_sub.copy(&thr)) {
				for (int i = 0; i < 3; i++) {
					if (thr.lit[i] || (!thr.burned_out[i] && thr.thrust[i] > 0.25f)) {
						any_active = true;
						break;
					}
				}
			} else {
				// no thrust yet early burn — still protect
				any_active = true;
			}

			if (any_active) {
				vehicle_local_position_s lpos{};

				if (_lpos_sub.copy(&lpos) && (lpos.z_valid || lpos.dist_bottom_valid)) {
					const float h = height_agl_m(lpos);
					const float vz_down = lpos.v_z_valid ? lpos.vz : 0.f;

					if (h <= _param_hl_z.get() && vz_down >= _param_hl_vz.get()) {
						request_abort(AR_HARD_LAND);
						return;
					}
				}
			}
		}

		// Hang-fire / partial ignition (BURN only)
		if (_state.state == tvc_vehicle_state_s::STATE_BURN && _fire_time != 0) {
			const float burn_t = (hrt_absolute_time() - _fire_time) * 1e-6f;
			const float hang_s = static_cast<float>(_param_hangfire_ms.get()) * 1e-3f;
			const float dly[3] = {
				_param_ign_dly0.get(), _param_ign_dly1.get(), _param_ign_dly2.get()
			};

			tvc_mount_thrust_s thr{};

			if (_thrust_sub.copy(&thr)) {
				bool any_lit = false;
				bool any_hang = false;

				for (int i = 0; i < 3; i++) {
					if (thr.lit[i]) {
						any_lit = true;
						_mount_lit_latch[i] = true;
					}

					const float due = dly[i] + hang_s;

					if (burn_t > due && !_mount_lit_latch[i] && !thr.lit[i]) {
						any_hang = true;
						_mount_hang[i] = true;
					}
				}

				if (any_hang) {
					if (_param_part_ign.get() != 0 && any_lit) {
						// Asymmetric torque risk
						request_abort(AR_HANGFIRE);
						return;
					}

					// Soft: hang mounts stay unlit → T≈0 → allocator rest angles
					if (!_hang_warned) {
						PX4_WARN("tvc_safety: hang-fire mount(s) (PART_IGN=0, no abort)");
						_hang_warned = true;
					}
				}
			}
		}
	}

	void update_burnout_coast()
	{
		if (_state.state != tvc_vehicle_state_s::STATE_BURN) {
			_burnout_since = 0;
			return;
		}

		tvc_mount_thrust_s thr{};

		if (!_thrust_sub.copy(&thr)) {
			return;
		}

		bool all_out = true;
		bool any_lit_ever = _any_lit;

		for (int i = 0; i < 3; i++) {
			if (thr.lit[i] || _mount_lit_latch[i]) {
				_any_lit = true;
				any_lit_ever = true;
			}

			if (!thr.burned_out[i] && thr.thrust[i] > 0.25f) {
				all_out = false;
			}
		}

		if (any_lit_ever && all_out) {
			if (_burnout_since == 0) {
				_burnout_since = hrt_absolute_time();
			}

			const hrt_abstime need = static_cast<hrt_abstime>(_param_bo_deb_ms.get()) * 1000ULL;

			if (hrt_elapsed_time(&_burnout_since) >= need) {
				_state.state = tvc_vehicle_state_s::STATE_COAST;
				_burnout_since = 0;
				PX4_INFO("tvc_safety: BURN → COAST (burnout debounced)");
			}
		} else {
			_burnout_since = 0;
		}
	}

	void update_coast_land()
	{
		if (_state.state != tvc_vehicle_state_s::STATE_COAST) {
			return;
		}

		vehicle_local_position_s lpos{};

		if (!_lpos_sub.copy(&lpos)) {
			return;
		}

		const float h = height_agl_m(lpos);
		const float vx = lpos.v_xy_valid ? lpos.vx : 0.f;
		const float vy = lpos.v_xy_valid ? lpos.vy : 0.f;
		const float vz = lpos.v_z_valid ? lpos.vz : 0.f;
		const float spd = sqrtf(vx * vx + vy * vy + vz * vz);

		if (h <= _param_land_z.get() && spd <= _param_land_v.get()) {
			_state.state = tvc_vehicle_state_s::STATE_LAND;
			PX4_INFO("tvc_safety: COAST → LAND");
		}
	}

	void update_link_preburn(const tvc_inhibit_flags_s &flags)
	{
		// LAUNCH_READY stays put on link loss: ground igniter is independent
		// of GCS, and first motion must still start the mission.
		if (_state.state != tvc_vehicle_state_s::STATE_ARMED) {
			_link_loss_since = 0;
			return;
		}

		if (flags.link_ok || _param_req_link.get() == 0) {
			_link_loss_since = 0;
			return;
		}

		if (_link_loss_since == 0) {
			_link_loss_since = hrt_absolute_time();
		}

		const hrt_abstime lim = static_cast<hrt_abstime>(_param_link_pre_ms.get()) * 1000ULL;

		if (hrt_elapsed_time(&_link_loss_since) >= lim) {
			_state.state = tvc_vehicle_state_s::STATE_DISARMED;
			_state.fire_latched = false;
			request_commander_arm(false);
			PX4_WARN("tvc_safety: link loss — disarm");
			_link_loss_since = 0;
		}
	}

	void enter_burn(const char *reason)
	{
		_state.state = tvc_vehicle_state_s::STATE_BURN;
		_state.fire_latched = true;
		_state.abort_reason = AR_NONE;
		_any_lit = false;
		_fire_time = hrt_absolute_time();
		_burnout_since = 0;
		_alloc_fail_streak = 0;
		_hang_warned = false;

		for (int i = 0; i < 3; i++) {
			_mount_lit_latch[i] = false;
			_mount_hang[i] = false;
		}

		PX4_WARN("tvc_safety: %s → BURN", reason);
	}

	void maybe_capture_origin(const tvc_inhibit_flags_s &flags)
	{
		if (_cmd_origin) {
			_cmd_origin = false;

			if (_state.state <= tvc_vehicle_state_s::STATE_PREFLIGHT) {
				_origin_set = false;
				_still_since = 0;
				PX4_INFO("tvc_safety: origin cleared — sit still on the pad");
			}
		}

		if (_origin_set || _state.state > tvc_vehicle_state_s::STATE_PREFLIGHT) {
			return;
		}

		if ((_param_req_rtk.get() != 0 && !flags.rtk_ok) || !flags.ekf_ok) {
			_still_since = 0;
			return;
		}

		vehicle_local_position_s lpos{};

		if (!_lpos_sub.copy(&lpos) || !lpos.xy_valid || !lpos.z_valid) {
			_still_since = 0;
			return;
		}

		const float vh = lpos.v_xy_valid ? sqrtf(lpos.vx * lpos.vx + lpos.vy * lpos.vy) : 1.f;
		const float vz = lpos.v_z_valid ? fabsf(lpos.vz) : 1.f;

		if (vh > 0.2f || vz > 0.2f) {
			_still_since = 0;
			return;
		}

		if (_still_since == 0) {
			_still_since = hrt_absolute_time();
			return;
		}

		if (hrt_elapsed_time(&_still_since) < 2_s) {
			return;
		}

		_pad_x = lpos.x;
		_pad_y = lpos.y;
		_pad_z = lpos.z;
		_origin_set = true;
		PX4_INFO("tvc_safety: origin set NED (%.2f, %.2f, %.2f)",
			 (double)_pad_x, (double)_pad_y, (double)_pad_z);
		mavlink_log_info(&_mavlink_log_pub,
				 "TVC origin set at pad. Landing is this point.");
	}

	void check_first_motion()
	{
		if (_state.state != tvc_vehicle_state_s::STATE_LAUNCH_READY) {
			return;
		}

		if (_param_bench.get() != 0 || _param_zero.get() != 0) {
			return;
		}

		vehicle_local_position_s lpos{};

		if (!_lpos_sub.copy(&lpos)) {
			return;
		}

		tvc::FirstMotionSample s{};
		s.vz_valid = lpos.v_z_valid;
		s.z_valid = lpos.z_valid || lpos.dist_bottom_valid;
		s.az_valid = lpos.v_z_valid;
		s.vz_up_mps = s.vz_valid ? -lpos.vz : 0.f;
		s.height_agl_m = height_agl_m(lpos);
		s.az_up_mps2 = s.az_valid ? -lpos.az : 0.f;

		tvc::FirstMotionConfig cfg{};
		cfg.vz_up_mps = _param_mot_vz.get();
		cfg.dz_m = _param_mot_dz.get();
		cfg.az_up_mps2 = _param_mot_az.get();
		cfg.consecutive = static_cast<int>(_param_mot_n.get());

		if (_motion.update(s, cfg)) {
			enter_burn("first motion");
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
			parameter_update_s p;
			_parameter_update_sub.copy(&p);
			updateParams();
		}

		tvc_inhibit_flags_s flags{};
		evaluate_gates(flags);
		maybe_capture_origin(flags);
		flags.origin_ok = (_param_req_rtk.get() == 0) || _origin_set;
		_inhibit_pub.publish(flags);

		const bool arm_gates = gates_for_arm(flags);

		handle_vehicle_commands();
		check_rc_abort();

		// In-flight / burn aborts
		check_burn_aborts(flags);
		update_burnout_coast();
		update_coast_land();
		update_link_preburn(flags);
		check_first_motion();

		// Stay in LAND. The old 1 s timer stepped to SAFE and disarmed,
		// which parked the nozzles. Attitude and altitude keep running.
		// Crew reset is what leaves LAND.

		// Commands
		if (_cmd_abort) {
			_cmd_abort = false;

			if (_state.state >= tvc_vehicle_state_s::STATE_ARMED &&
			    _state.state != tvc_vehicle_state_s::STATE_SAFE) {
				if (_state.abort_reason == AR_NONE) {
					_state.abort_reason = AR_USER;
				}

				_state.state = tvc_vehicle_state_s::STATE_ABORT;
				_state.fire_latched = false;
				PX4_WARN("tvc_safety: ABORT reason=%u", (unsigned)_state.abort_reason);
			}
		}

		if (_cmd_reset) {
			_cmd_reset = false;

			if (_state.state == tvc_vehicle_state_s::STATE_SAFE ||
			    _state.state == tvc_vehicle_state_s::STATE_LAND) {
				_state.state = tvc_vehicle_state_s::STATE_DISARMED;
				_state.fire_latched = false;
				_state.abort_reason = AR_NONE;
				_any_lit = false;
				_fire_time = 0;
				_alloc_fail_streak = 0;
				_motion.reset();
				_origin_set = false;
				_still_since = 0;

				for (int i = 0; i < 3; i++) {
					_mount_lit_latch[i] = false;
					_mount_hang[i] = false;
				}

				request_commander_arm(false);
				PX4_INFO("tvc_safety: crew reset → DISARMED");
			}
		}

		if (_cmd_disarm) {
			_cmd_disarm = false;

			if (_state.state <= tvc_vehicle_state_s::STATE_LAUNCH_READY) {
				_state.state = tvc_vehicle_state_s::STATE_DISARMED;
				_state.fire_latched = false;
				request_commander_arm(false);
			}
		}

		if (_cmd_arm) {
			_cmd_arm = false;

			if (_param_zero.get() != 0) {
				PX4_WARN("tvc_safety: arm denied (TVC_ZERO)");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: arm denied (ZERO)");
			} else if (arm_gates && _state.state <= tvc_vehicle_state_s::STATE_PREFLIGHT) {
				_state.state = tvc_vehicle_state_s::STATE_ARMED;
				request_commander_arm(true);
				_gcs_arm_grace_until = hrt_absolute_time() + 2_s;
				PX4_INFO("tvc_safety: ARMED (commander force=%d)",
					 _param_force_arm.get() != 0);
			} else if (!arm_gates) {
				PX4_WARN("tvc_safety: arm denied rtk=%d ekf=%d gim=%d geo=%d link=%d alloc=%d thr=%d rail=%d mass=%d mis=%d org=%d",
					 (int)flags.rtk_ok, (int)flags.ekf_ok, (int)flags.gimbal_rest_ok,
					 (int)flags.geofence_ok, (int)flags.link_ok, (int)flags.alloc_ok,
					 (int)flags.thrust_ok, (int)flags.rail_v_ok, (int)flags.mass_cfg_ok,
					 (int)flags.mission_ok, (int)flags.origin_ok);
				mavlink_log_critical(&_mavlink_log_pub,
						     "TVC: arm denied g=%03x",
						     (unsigned)inhibit_bits(flags));
			}
		}

		if (_cmd_ign_arm) {
			_cmd_ign_arm = false;

			if (_param_bench.get() != 0) {
				PX4_WARN("tvc_safety: launch_ready denied (TVC_BENCH)");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: launch_ready denied (BENCH)");
			} else if (_param_zero.get() != 0) {
				PX4_WARN("tvc_safety: launch_ready denied (TVC_ZERO)");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: launch_ready denied (ZERO)");
			} else if (_param_allow_fire.get() == 0) {
				PX4_WARN("tvc_safety: launch_ready denied (TVC_ALLOW_FIRE=0)");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: launch_ready denied (ALLOW_FIRE=0)");
			} else if (_state.state == tvc_vehicle_state_s::STATE_ARMED && arm_gates) {
				_state.state = tvc_vehicle_state_s::STATE_LAUNCH_READY;
				_motion.reset();
				{
					vehicle_local_position_s lpos{};

					if (_lpos_sub.copy(&lpos)) {
						_motion.capture_pad(height_agl_m(lpos));
					}
				}
				PX4_INFO("tvc_safety: LAUNCH_READY (waiting for first motion)");
			} else {
				PX4_WARN("tvc_safety: launch_ready denied");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: launch_ready denied");
			}
		}

		if (_cmd_fire) {
			_cmd_fire = false;

			if (_param_bench.get() != 0) {
				PX4_WARN("tvc_safety: fire denied (TVC_BENCH)");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: fire denied (BENCH)");
			} else if (_param_zero.get() != 0) {
				PX4_WARN("tvc_safety: fire denied (TVC_ZERO)");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: fire denied (ZERO)");
			} else if (_param_allow_fire.get() == 0) {
				PX4_WARN("tvc_safety: fire denied (TVC_ALLOW_FIRE=0)");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: fire denied (ALLOW_FIRE=0)");
			} else if (_state.state != tvc_vehicle_state_s::STATE_LAUNCH_READY) {
				PX4_WARN("tvc_safety: fire denied (state)");
				mavlink_log_critical(&_mavlink_log_pub, "TVC: fire denied (state %s)",
						     tvc::vehicle_state_name(_state.state));
			} else {
				// HIL / software liftoff only. Cube never pulses an igniter.
				enter_burn("HIL fire");
			}
		}

		// Power-on auto PREFLIGHT when mass configured
		if (_state.state == tvc_vehicle_state_s::STATE_DISARMED && flags.mass_cfg_ok) {
			_state.state = tvc_vehicle_state_s::STATE_PREFLIGHT;
		}

		// ABORT → SAFE after settle
		if (_state.state == tvc_vehicle_state_s::STATE_ABORT) {
			if (_abort_time == 0) {
				_abort_time = hrt_absolute_time();
			} else if (hrt_elapsed_time(&_abort_time) > 2_s) {
				_state.state = tvc_vehicle_state_s::STATE_SAFE;
				_abort_time = 0;
				request_commander_arm(false);
			}
		} else {
			_abort_time = 0;
		}

		_state.timestamp = hrt_absolute_time();
		_state.origin_set = _origin_set;
		_state.pad_x = _pad_x;
		_state.pad_y = _pad_y;
		_state.pad_z = _pad_z;
		_state.arm_ok = arm_gates;
		_state.ignite_ok = arm_gates && (_param_allow_fire.get() != 0) &&
				   (_state.state == tvc_vehicle_state_s::STATE_LAUNCH_READY);
		_state_pub.publish(_state);

		hold_commander_while_burning();
		block_stock_arm_without_tvc();
		publish_named_values(flags);
		publish_gcs_status(flags);
	}

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _gps_sub{ORB_ID(sensor_gps)};
	uORB::Subscription _est_flags_sub{ORB_ID(estimator_status_flags)};
	uORB::Subscription _lpos_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _ang_sub{ORB_ID(vehicle_angular_velocity)};
	uORB::Subscription _att_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _thrust_sub{ORB_ID(tvc_mount_thrust)};
	uORB::Subscription _gimbal_sub{ORB_ID(tvc_gimbal_cmd)};
	uORB::Subscription _power_sub{ORB_ID(system_power)};
	uORB::Subscription _vehicle_command_sub{ORB_ID(vehicle_command)};
	uORB::Subscription _status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _rc_sub{ORB_ID(input_rc)};
	uORB::SubscriptionMultiArray<telemetry_status_s> _telem_subs{ORB_ID::telemetry_status};

	uORB::Publication<tvc_vehicle_state_s> _state_pub{ORB_ID(tvc_vehicle_state)};
	uORB::Publication<tvc_inhibit_flags_s> _inhibit_pub{ORB_ID(tvc_inhibit_flags)};
	uORB::Publication<vehicle_command_s> _vehicle_command_pub{ORB_ID(vehicle_command)};
	uORB::Publication<vehicle_command_ack_s> _vehicle_command_ack_pub{ORB_ID(vehicle_command_ack)};
	uORB::Publication<debug_key_value_s> _named_pub{ORB_ID(debug_key_value)};

	tvc_vehicle_state_s _state{};
	tvc_inhibit_flags_s _last_flags{};

	bool _cmd_arm{false};
	bool _cmd_disarm{false};
	bool _cmd_ign_arm{false};
	bool _cmd_fire{false};
	bool _cmd_abort{false};
	bool _cmd_reset{false};
	bool _cmd_origin{false};
	bool _origin_set{false};
	float _pad_x{0.f};
	float _pad_y{0.f};
	float _pad_z{0.f};
	hrt_abstime _still_since{0};

	bool _any_lit{false};
	bool _mount_lit_latch[3]{};
	bool _mount_hang[3]{};
	bool _saw_system_power{false};
	bool _hang_warned{false};

	int _alloc_fail_streak{0};

	hrt_abstime _abort_time{0};
	hrt_abstime _fire_time{0};
	hrt_abstime _burnout_since{0};
	hrt_abstime _link_loss_since{0};
	hrt_abstime _gcs_arm_grace_until{0};
	hrt_abstime _last_gcs_status{0};
	hrt_abstime _last_named{0};
	uint8_t _last_announced_state{0xff};
	uint8_t _named_i{0};
	bool _rc_abort_high{false};
	orb_advert_t _mavlink_log_pub{nullptr};
	tvc::FirstMotionDetector _motion{};

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::TVC_ALLOW_FIRE>) _param_allow_fire,
		(ParamInt<px4::params::TVC_AUTO_ABORT>) _param_auto_abort,
		(ParamInt<px4::params::TVC_REQ_RTK>) _param_req_rtk,
		(ParamInt<px4::params::TVC_REQ_LINK>) _param_req_link,
		(ParamFloat<px4::params::TVC_MAX_TILT>) _param_max_tilt,
		(ParamFloat<px4::params::TVC_MAX_RATE>) _param_max_rate,
		(ParamFloat<px4::params::TVC_GEOFENCE_R>) _param_geofence_r,
		(ParamFloat<px4::params::TVC_MASS>) _param_mass,
		(ParamInt<px4::params::TVC_HANGFIRE_MS>) _param_hangfire_ms,
		(ParamInt<px4::params::TVC_PART_IGN>) _param_part_ign,
		(ParamFloat<px4::params::TVC_HL_Z>) _param_hl_z,
		(ParamFloat<px4::params::TVC_HL_VZ>) _param_hl_vz,
		(ParamInt<px4::params::TVC_BO_DEB_MS>) _param_bo_deb_ms,
		(ParamFloat<px4::params::TVC_LAND_Z>) _param_land_z,
		(ParamFloat<px4::params::TVC_LAND_V>) _param_land_v,
		(ParamInt<px4::params::TVC_LINK_PRE>) _param_link_pre_ms,
		(ParamFloat<px4::params::TVC_RAIL_VMIN>) _param_rail_vmin,
		(ParamFloat<px4::params::TVC_GIM_REST>) _param_gim_rest,
		(ParamInt<px4::params::TVC_ALLOC_N>) _param_alloc_n,
		(ParamInt<px4::params::TVC_FORCE_ARM>) _param_force_arm,
		(ParamInt<px4::params::TVC_BENCH>) _param_bench,
		(ParamInt<px4::params::TVC_ZERO>) _param_zero,
		(ParamFloat<px4::params::TVC_IGN_DLY0>) _param_ign_dly0,
		(ParamFloat<px4::params::TVC_IGN_DLY1>) _param_ign_dly1,
		(ParamFloat<px4::params::TVC_IGN_DLY2>) _param_ign_dly2,
		(ParamInt<px4::params::TVC_ABORT_CH>) _param_abort_ch,
		(ParamFloat<px4::params::TVC_MOT_VZ>) _param_mot_vz,
		(ParamFloat<px4::params::TVC_MOT_DZ>) _param_mot_dz,
		(ParamFloat<px4::params::TVC_MOT_AZ>) _param_mot_az,
		(ParamInt<px4::params::TVC_MOT_N>) _param_mot_n
	)
};

extern "C" __EXPORT int tvc_safety_main(int argc, char *argv[])
{
	return TvcSafety::main(argc, argv);
}
