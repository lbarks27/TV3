#include "TvcControl.hpp"

#include <cstring>
#include <mathlib/mathlib.h>
#include <parameters/param.h>
#include <px4_platform_common/getopt.h>
#include <px4_platform_common/log.h>
#include <tvc/quat.hpp>

TvcControl::TvcControl() :
	ModuleParams(nullptr),
	ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::hp_default)
{
	tvc::SolverConfig scfg;
	// Verified Mini Carrier bench 2026-08-24 (heading hold):
	// invert_side_roll = M1/M2 Jx, invert_yaw = all Jz, then remap M0
	// yaw a2 after allocate. Pair with TVC_SERVO_REV=47, SENS_BOARD_ROT=2.
	scfg.invert_side_roll = true;
	scfg.invert_yaw = true;
	// Match SIH g12: pitch primary; heading/roll last (yaw 0.25, roll 0.7).
	scfg.roll_torque_weight = 0.7f;
	scfg.yaw_torque_weight = 0.25f;
	_allocator.set_solver(scfg);
	_allocator.rest_angles_flat(_angles_cmd);
	_allocator.rest_angles_flat(_angles_slew);
}

TvcControl::~TvcControl()
{
	perf_free(_loop_perf);
	perf_free(_loop_interval_perf);
}

bool TvcControl::init()
{
	update_params();
	tvc::VehicleConfig vcfg;
	if (tvc::load_vehicle_config_file(tvc::kVehicleJsonPath, tvc::kVehicleCrcPath, vcfg) &&
	    vcfg.valid) {
		apply_vehicle_mounts();
		_allocator.rest_angles_flat(_angles_cmd);
		_allocator.rest_angles_flat(_angles_slew);
		PX4_INFO("vehicle.json ok crc=0x%08x mass=%.3f kg com=[%.4f %.4f %.4f]",
			 (unsigned)vcfg.crc, (double)vcfg.loaded_mass_kg,
			 (double)vcfg.loaded_com.x, (double)vcfg.loaded_com.y,
			 (double)vcfg.loaded_com.z);
	} else {
		PX4_ERR("vehicle.json load failed status=%u — arm blocked",
			(unsigned)vcfg.status);
	}
	const int hz = _param_ctrl_hz.get() > 0 ? _param_ctrl_hz.get() : 100;
	ScheduleOnInterval(1000000 / hz);
	return true;
}

void TvcControl::update_params()
{
	updateParams();

	tvc::AttitudePIDConfig cfg;
	cfg.kp[0] = _param_att_rp.get();
	cfg.kp[1] = _param_att_pp.get();
	cfg.kp[2] = _param_att_yp.get();
	cfg.kd[0] = _param_att_rd.get();
	cfg.kd[1] = _param_att_pd.get();
	cfg.kd[2] = _param_att_yd.get();
	cfg.ki[0] = _param_att_ri.get();
	cfg.ki[1] = _param_att_pi.get();
	cfg.ki[2] = _param_att_yi.get();
	cfg.torque_limit = _param_tau_lim.get();
	_att_pid.set_config(cfg);

	tvc::Vec3 com = tvc::default_com();
	com.z = _param_com_z.get();
	const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
	if (vcfg.valid) {
		apply_vehicle_mounts();
	} else {
		_allocator.set_com(com);
	}
}

void TvcControl::apply_vehicle_mounts()
{
	const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
	if (!vcfg.valid) {
		return;
	}
	_allocator.set_mounts(vcfg.mounts, vcfg.n_mounts);
	_allocator.set_com(vcfg.loaded_com);
	_allocator.set_force_axis(vcfg.loaded_com);
}

void TvcControl::publish_rest_cmd()
{
	tvc_gimbal_cmd_s cmd{};
	cmd.timestamp = hrt_absolute_time();
	_allocator.rest_angles_flat(cmd.angles_rad);
	cmd.saturated = false;
	cmd.timed_out = false;
	cmd.solve_us = 0;
	cmd.residual = 0.f;
	_tvc_gimbal_cmd_pub.publish(cmd);
}

void TvcControl::Run()
{
	if (should_exit()) {
		ScheduleClear();
		exit_and_cleanup();
		return;
	}

	perf_begin(_loop_perf);
	perf_count(_loop_interval_perf);

	if (_parameter_update_sub.updated()) {
		parameter_update_s pu;
		_parameter_update_sub.copy(&pu);
		update_params();
	}

	handle_gcs_commands();

	const hrt_abstime now = hrt_absolute_time();
	float dt = 0.01f;
	if (_last_run != 0) {
		dt = math::constrain((now - _last_run) * 1e-6f, 0.001f, 0.05f);
	}
	_last_run = now;

	// Vehicle state gates closed-loop
	tvc_vehicle_state_s vstate{};
	const bool have_state = _tvc_vehicle_state_sub.copy(&vstate);
	const uint8_t st = have_state ? vstate.state : tvc_vehicle_state_s::STATE_DISARMED;

	const bool bench = _param_bench.get() != 0;
	const bool closed_loop = bench ||
		st == tvc_vehicle_state_s::STATE_ARMED ||
		st == tvc_vehicle_state_s::STATE_LAUNCH_READY ||
		st == tvc_vehicle_state_s::STATE_BURN ||
		st == tvc_vehicle_state_s::STATE_COAST ||
		st == tvc_vehicle_state_s::STATE_LAND ||
		st == tvc_vehicle_state_s::STATE_ABORT;

	if (!closed_loop) {
		// Force rest with slew
		float rest[tvc::kNumAngles];
		_allocator.rest_angles_flat(rest);
		tvc::TorqueAllocator::apply_actuator_slew(_angles_slew, rest, _param_slew_dps.get(), dt);

		tvc_gimbal_cmd_s cmd{};
		cmd.timestamp = now;
		for (int i = 0; i < tvc::kNumAngles; i++) {
			cmd.angles_rad[i] = _angles_slew[i];
		}
		_tvc_gimbal_cmd_pub.publish(cmd);
		perf_end(_loop_perf);
		return;
	}

	vehicle_attitude_s att{};
	vehicle_angular_velocity_s ang{};
	if (!_vehicle_attitude_sub.copy(&att) || !_vehicle_angular_velocity_sub.copy(&ang)) {
		// One missed sample must not snap the nozzles to rest while burning.
		if (_att_lost_since == 0) {
			_att_lost_since = now;
		}

		tvc_gimbal_cmd_s held{};
		held.timestamp = now;

		if (hrt_elapsed_time(&_att_lost_since) < 200_ms) {
			for (int i = 0; i < tvc::kNumAngles; i++) {
				held.angles_rad[i] = _angles_slew[i];
			}
		} else {
			float rest[tvc::kNumAngles];
			_allocator.rest_angles_flat(rest);
			tvc::TorqueAllocator::apply_actuator_slew(_angles_slew, rest, _param_slew_dps.get(), dt);

			for (int i = 0; i < tvc::kNumAngles; i++) {
				held.angles_rad[i] = _angles_slew[i];
			}
		}

		_tvc_gimbal_cmd_pub.publish(held);
		perf_end(_loop_perf);
		return;
	}
	_att_lost_since = 0;

	// PX4 NED/FRD → sim ENU/FLU (FRD +X is M0 after SENS_BOARD_ROT=2)
	const tvc::Quat q_px4{att.q[0], att.q[1], att.q[2], att.q[3]};
	const tvc::Quat q_sim = tvc::q_px4_to_sim(q_px4);
	const tvc::Vec3 omega_frd{ang.xyz[0], ang.xyz[1], ang.xyz[2]};
	const tvc::Vec3 omega_sim = tvc::frd_to_flu(omega_frd);

	tvc::Quat q_ref = q_sim;
	if (bench) {
		float r = 0.f, p = 0.f, y = 0.f;
		tvc::q_to_euler(q_sim, r, p, y);
		if (!_bench_heading_valid) {
			_bench_heading_rad = y;
			_bench_heading_valid = true;
			PX4_INFO("bench heading hold %.1f deg", (double)(y * 57.2957795f));
		}
		q_ref = tvc::q_from_euler(0.f, 0.f, _bench_heading_rad);
	} else {
		_bench_heading_valid = false;
		tvc_attitude_sp_s asp{};
		if (_tvc_attitude_sp_sub.copy(&asp)) {
			q_ref = tvc::Quat{asp.q_sim[0], asp.q_sim[1], asp.q_sim[2], asp.q_sim[3]};
		}
	}

	const tvc::Vec3 tau_cmd = _att_pid.update(q_sim, omega_sim, q_ref, dt);
	// Allocator residual weights (roll 0.7 / yaw 0.25) already match SIH.
	// Do not pre-scale tau here — that double-softens vs the pass log.
	float tau[3] = {tau_cmd.x, tau_cmd.y, tau_cmd.z};

	// Force setpoint
	float force_target = 0.f;
	float force_weight = 0.f;
	tvc_force_sp_s fsp{};
	if (_tvc_force_sp_sub.copy(&fsp) && fsp.mix_armed) {
		force_target = fsp.f_axial_n;
		force_weight = fsp.force_weight > 0.f ? fsp.force_weight : _param_force_w.get();
	}

	// Abort parks the nozzles. Coast and landing keep the altitude force
	// and the attitude torque; touchdown does not drop either loop.
	if (!bench && st == tvc_vehicle_state_s::STATE_ABORT) {
		force_weight = 0.f;
		tau[0] = tau[1] = tau[2] = 0.f;
	}

	// Per-mount thrust from motor curves (no force sensing)
	float thrusts[3] = {0.f, 0.f, 0.f};
	tvc_mount_thrust_s thr{};
	if (_tvc_mount_thrust_sub.copy(&thr)) {
		for (int i = 0; i < 3; i++) {
			thrusts[i] = thr.thrust[i];
		}
	} else {
		const float nom = _param_thr_nom.get() > 1.f ? _param_thr_nom.get() : 12.f;
		for (int i = 0; i < 3; i++) {
			thrusts[i] = nom;
		}
	}
	if (bench) {
		const float nom = _param_thr_nom.get() > 1.f ? _param_thr_nom.get() : 12.f;
		for (int i = 0; i < 3; i++) {
			if (thrusts[i] < 1.f) {
				thrusts[i] = nom;
			}
		}
	}

	if (bench) {
		// Stick up = max axial (a1 rest = 0°). Stick down = fold 90°.
		// No stick and TVC_BENCH_THR < 0 → a1 rest. Param 0..1 overrides stick.
		force_weight = 0.f;
		float thr01 = -1.f;
		const float bench_thr = _param_bench_thr.get();
		if (bench_thr >= 0.f) {
			thr01 = math::constrain(bench_thr, 0.f, 1.f);
		} else {
			manual_control_setpoint_s mc{};
			if (_manual_control_setpoint_sub.copy(&mc) && mc.valid &&
			    PX4_ISFINITE(mc.throttle) &&
			    (now > mc.timestamp) && (now - mc.timestamp < 500_ms)) {
				thr01 = math::constrain(0.5f * (mc.throttle + 1.f), 0.f, 1.f);
			}
		}
		if (thr01 >= 0.f) {
			float sum_T = 0.f;
			for (int i = 0; i < 3; i++) {
				sum_T += thrusts[i];
			}
			// Stick 1.0 = max axial at a1 rest (0°, vertical).
			const float f_rest = _allocator.axial_at_rest(thrusts);
			force_target = thr01 * (f_rest > 1.f ? f_rest : sum_T);
			force_weight = 1.f;
		}
	}

	const hrt_abstime t0 = hrt_absolute_time();
	tvc::AllocResult alloc = _allocator.allocate_force_track(tau, force_target, force_weight, thrusts);
	// Roll/pitch may use TVC_A2_MAX. Heading's share of the final a2 is
	// TVC_A2_YAW, measured against the same solve with yaw torque removed.
	// invert_yaw is on all three columns. M0 heading is the remaining
	// reversed couple: isolate that mount's yaw share of a2 and flip it
	// after the solve (verified 2026-08-24). Do not change the Jacobian.
	const float tau_no_yaw[3] = {tau[0], tau[1], 0.f};
	const tvc::AllocResult no_yaw = _allocator.allocate_force_track(
		tau_no_yaw, force_target, force_weight, thrusts);
	if (fabsf(tau[2]) > 1e-3f) {
		alloc.angles[0] = 2.f * no_yaw.angles[0] - alloc.angles[0];
	}
	const float a2_rp = math::radians(math::constrain(_param_a2_max.get(), 5.f, 45.f));
	const float a2_yaw = math::radians(math::constrain(_param_a2_yaw.get(), 0.f, 15.f));
	const float a2_abs = a2_rp + a2_yaw;
	for (int m = 0; m < 3; m++) {
		const float heading = math::constrain(
			alloc.angles[m * 2] - no_yaw.angles[m * 2], -a2_yaw, a2_yaw);
		const float base = math::constrain(no_yaw.angles[m * 2], -a2_rp, a2_rp);
		alloc.angles[m * 2] = math::constrain(base + heading, -a2_abs, a2_abs);
	}
	const uint32_t solve_us = (uint32_t)(hrt_absolute_time() - t0);

	tvc::TorqueAllocator::apply_actuator_slew(_angles_slew, alloc.angles, _param_slew_dps.get(), dt);
	const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
	const float a1_min = math::radians(vcfg.valid ? vcfg.a1_min_deg : 0.f);
	const float a1_max = math::radians(vcfg.valid ? vcfg.a1_max_deg : 90.f);
	for (int m = 0; m < 3; m++) {
		_angles_slew[m * 2] = math::constrain(_angles_slew[m * 2], -a2_abs, a2_abs);
		_angles_slew[m * 2 + 1] = math::constrain(_angles_slew[m * 2 + 1], a1_min, a1_max);
	}

	tvc_gimbal_cmd_s cmd{};
	cmd.timestamp = now;
	for (int i = 0; i < tvc::kNumAngles; i++) {
		cmd.angles_rad[i] = _angles_slew[i];
	}
	cmd.saturated = alloc.saturated;
	cmd.timed_out = alloc.timed_out;
	cmd.solve_us = solve_us;
	cmd.residual = alloc.residual;
	_tvc_gimbal_cmd_pub.publish(cmd);

	tvc_control_debug_s dbg{};
	dbg.timestamp = now;
	dbg.tau_cmd[0] = tau[0];
	dbg.tau_cmd[1] = tau[1];
	dbg.tau_cmd[2] = tau[2];
	dbg.tau_act[0] = alloc.torque[0];
	dbg.tau_act[1] = alloc.torque[1];
	dbg.tau_act[2] = alloc.torque[2];
	dbg.force_body[0] = alloc.force[0];
	dbg.force_body[1] = alloc.force[1];
	dbg.force_body[2] = alloc.force[2];
	dbg.q[0] = q_sim.w;
	dbg.q[1] = q_sim.x;
	dbg.q[2] = q_sim.y;
	dbg.q[3] = q_sim.z;
	dbg.q_ref[0] = q_ref.w;
	dbg.q_ref[1] = q_ref.x;
	dbg.q_ref[2] = q_ref.y;
	dbg.q_ref[3] = q_ref.z;
	dbg.omega[0] = omega_sim.x;
	dbg.omega[1] = omega_sim.y;
	dbg.omega[2] = omega_sim.z;
	const float *integ = _att_pid.integral();
	dbg.att_integral[0] = integ[0];
	dbg.att_integral[1] = integ[1];
	dbg.att_integral[2] = integ[2];
	_tvc_control_debug_pub.publish(dbg);

	perf_end(_loop_perf);
}

int TvcControl::task_spawn(int argc, char *argv[])
{
	TvcControl *instance = new TvcControl();
	if (!instance) {
		PX4_ERR("alloc failed");
		return PX4_ERROR;
	}
	_object.store(instance);
	_task_id = task_id_is_work_queue;
	if (!instance->init()) {
		delete instance;
		_object.store(nullptr);
		_task_id = -1;
		return PX4_ERROR;
	}
	return PX4_OK;
}

int TvcControl::custom_command(int argc, char *argv[])
{
	if (argc >= 1 && argv[0] != nullptr) {
		if (!strcmp(argv[0], "bench")) {
			set_bench_mode(true);
			return 0;
		}

		if (!strcmp(argv[0], "nobench")) {
			set_bench_mode(false);
			return 0;
		}
	}

	return print_usage("unknown command");
}

void TvcControl::set_bench_mode(bool on)
{
	int32_t one = 1;
	int32_t two = 2;
	int32_t zero = 0;

	if (on) {
		param_set(param_find("TVC_BENCH"), &one);
		param_set(param_find("COM_PREARM_MODE"), &two);
		param_set(param_find("TVC_ZERO"), &zero);
		param_set(param_find("TVC_ALLOW_FIRE"), &zero);
		param_set(param_find("TVC_IGN_GPIO_EN"), &zero);
		TvcControl *inst = _object.load();
		if (inst != nullptr) {
			inst->_bench_heading_valid = false;
		}
		PX4_WARN("tvc_control: BENCH on (100 Hz, heading hold, a1 from throttle, launch-ready denied)");
	} else {
		param_set(param_find("TVC_BENCH"), &zero);
		param_set(param_find("COM_PREARM_MODE"), &zero);
		PX4_INFO("tvc_control: BENCH off");
	}
}

void TvcControl::handle_gcs_commands()
{
	vehicle_command_s cmd{};

	while (_vehicle_command_sub.update(&cmd)) {
		if (!tvc::is_external_gcs_command(cmd) ||
		    cmd.command != static_cast<uint32_t>(tvc::kMavCmdTvc)) {
			continue;
		}

		const tvc::GcsAction action = tvc::gcs_action_from_param1(cmd.param1);

		if (action == tvc::GcsAction::Bench) {
			set_bench_mode(true);
			tvc::ack_vehicle_command(_vehicle_command_ack_pub, cmd,
						 vehicle_command_ack_s::VEHICLE_CMD_RESULT_ACCEPTED);
		} else if (action == tvc::GcsAction::NoBench) {
			set_bench_mode(false);
			tvc::ack_vehicle_command(_vehicle_command_ack_pub, cmd,
						 vehicle_command_ack_s::VEHICLE_CMD_RESULT_ACCEPTED);
		}
	}
}

int TvcControl::print_status()
{
	const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
	PX4_INFO("vehicle valid=%d status=%u crc=0x%08x loaded_mass=%.3f kg bench=%d heading_hold=%d",
		 (int)vcfg.valid, (unsigned)vcfg.status, (unsigned)vcfg.crc,
		 (double)vcfg.loaded_mass_kg, (int)_param_bench.get(),
		 (int)_bench_heading_valid);
	perf_print_counter(_loop_perf);
	perf_print_counter(_loop_interval_perf);
	return 0;
}

int TvcControl::print_usage(const char *reason)
{
	if (reason) {
		PX4_WARN("%s", reason);
	}
	PRINT_MODULE_DESCRIPTION(
		R"DESCR_STR(
### Description
TVC attitude control and collective/differential gimbal allocation at 100 Hz.
Control math runs in Prototype ENU-sim / FLU-sim frames after converting EKF2 state.
)DESCR_STR");
	PRINT_MODULE_USAGE_NAME("tvc_control", "controller");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_COMMAND_DESCR("bench", "Onboard attitude hold (no fire)");
	PRINT_MODULE_USAGE_COMMAND_DESCR("nobench", "Disable bench mode");
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
	return 0;
}

extern "C" __EXPORT int tvc_control_main(int argc, char *argv[])
{
	return TvcControl::main(argc, argv);
}
