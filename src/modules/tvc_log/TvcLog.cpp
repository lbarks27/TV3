/**
 * @file TvcLog.cpp
 * 100 Hz flight_log_vl publisher for onboard ULog (flight_log_v1 schema).
 *
 * Snapshots tvc_* + EKF into ENU-sim / FLU-sim so pad logs match SIH CSV.
 * Logger extra topics: /fs/microsd/etc/logging/logger_topics.txt (packed by
 * tools/pack_vehicle_sd.py).
 */

#include <px4_platform_common/module.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>
#include <drivers/drv_hrt.h>
#include <mathlib/mathlib.h>
#include <cmath>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/topics/flight_log_vl.h>
#include <uORB/topics/tvc_control_debug.h>
#include <uORB/topics/tvc_force_sp.h>
#include <uORB/topics/tvc_gimbal_cmd.h>
#include <uORB/topics/tvc_mount_thrust.h>
#include <uORB/topics/tvc_setpoint.h>
#include <uORB/topics/tvc_vehicle_state.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_local_position.h>

#include <tvc/frames.hpp>

using namespace time_literals;

class TvcLog : public ModuleBase<TvcLog>, public px4::ScheduledWorkItem
{
public:
	TvcLog() : ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::nav_and_controllers) {}

	static int task_spawn(int argc, char *argv[])
	{
		TvcLog *inst = new TvcLog();

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

	static int custom_command(int, char **) { return print_usage("unknown"); }

	static int print_usage(const char *reason = nullptr)
	{
		if (reason) {
			PX4_WARN("%s", reason);
		}

		PRINT_MODULE_DESCRIPTION(R"DESCR_STR(
### Description
Publishes flight_log_vl at 100 Hz for ULog (flight_log_v1, ENU-sim / FLU-sim).
)DESCR_STR");
		PRINT_MODULE_USAGE_NAME("tvc_log", "system");
		PRINT_MODULE_USAGE_COMMAND("start");
		PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
		return 0;
	}

	int print_status() override
	{
		PX4_INFO("flight_log_vl 100 Hz  samples=%lu  t=%.3f s",
			 (unsigned long)_samples,
			 (double)_last_time_s);
		return 0;
	}

	bool init()
	{
		ScheduleOnInterval(10_ms); // 100 Hz
		return true;
	}

private:
	void Run() override
	{
		if (should_exit()) {
			ScheduleClear();
			exit_and_cleanup();
			return;
		}

		const hrt_abstime now = hrt_absolute_time();

		if (_t0 == 0) {
			_t0 = now;
		}

		flight_log_vl_s log{};
		log.timestamp = now;
		log.time_s = (now - _t0) * 1e-6f;
		log.heading_ref_rad = nanf("");

		tvc_control_debug_s dbg{};
		const bool have_dbg = _dbg_sub.copy(&dbg);

		if (have_dbg) {
			for (int i = 0; i < 4; i++) {
				log.q[i] = dbg.q[i];
				log.q_ref[i] = dbg.q_ref[i];
			}

			for (int i = 0; i < 3; i++) {
				log.omega[i] = dbg.omega[i];
				log.tau_cmd[i] = dbg.tau_cmd[i];
				log.tau_act[i] = dbg.tau_act[i];
				log.force_body[i] = dbg.force_body[i];
				log.att_integral[i] = dbg.att_integral[i];
			}

			log.alt_integral = dbg.alt_integral;
			log.mission_phase = dbg.mission_phase;
		} else {
			vehicle_attitude_s att{};

			if (_att_sub.copy(&att)) {
				const tvc::Quat q_px4{att.q[0], att.q[1], att.q[2], att.q[3]};
				const tvc::Quat q_sim = tvc::q_px4_to_sim(q_px4);
				log.q[0] = q_sim.w;
				log.q[1] = q_sim.x;
				log.q[2] = q_sim.y;
				log.q[3] = q_sim.z;
				log.q_ref[0] = q_sim.w;
				log.q_ref[1] = q_sim.x;
				log.q_ref[2] = q_sim.y;
				log.q_ref[3] = q_sim.z;
			} else {
				log.q[0] = 1.f;
				log.q_ref[0] = 1.f;
			}
		}

		tvc_gimbal_cmd_s gim{};

		if (_gim_sub.copy(&gim)) {
			for (int i = 0; i < 6; i++) {
				log.gimbal_angles[i] = gim.angles_rad[i];
			}

			log.saturated = gim.saturated ? 1 : 0;
		}

		tvc_mount_thrust_s thr{};

		if (_thr_sub.copy(&thr)) {
			for (int i = 0; i < 3; i++) {
				log.mount_thrusts[i] = thr.thrust[i];
			}
		}

		tvc_setpoint_s sp{};

		if (_sp_sub.copy(&sp)) {
			for (int i = 0; i < 3; i++) {
				log.position_ref[i] = sp.pos[i];
				log.velocity_ref[i] = sp.vel[i];
			}

			log.heading_ref_rad = sp.heading_rad;
			log.mission_phase = sp.phase;
		}

		tvc_force_sp_s fsp{};

		if (_fsp_sub.copy(&fsp)) {
			log.force_target_axial = fsp.f_axial_n;
			log.force_weight = fsp.force_weight;
			log.mix_armed = fsp.mix_armed ? 1 : 0;
		}

		tvc_vehicle_state_s st{};

		if (_st_sub.copy(&st)) {
			log.vehicle_state = st.state;
			log.abort_reason = st.abort_reason;
			log.fire_latched = st.fire_latched ? 1 : 0;
		}

		vehicle_local_position_s lpos{};

		if (_lpos_sub.copy(&lpos)) {
			const tvc::Vec3 p = tvc::ned_to_enu(tvc::Vec3{lpos.x, lpos.y, lpos.z});
			const tvc::Vec3 v = tvc::ned_to_enu(tvc::Vec3{lpos.vx, lpos.vy, lpos.vz});
			log.position[0] = p.x;
			log.position[1] = p.y;
			log.position[2] = p.z;
			log.velocity[0] = v.x;
			log.velocity[1] = v.y;
			log.velocity[2] = v.z;
		}

		_log_pub.publish(log);
		_samples++;
		_last_time_s = log.time_s;
	}

	uORB::Subscription _dbg_sub{ORB_ID(tvc_control_debug)};
	uORB::Subscription _gim_sub{ORB_ID(tvc_gimbal_cmd)};
	uORB::Subscription _thr_sub{ORB_ID(tvc_mount_thrust)};
	uORB::Subscription _sp_sub{ORB_ID(tvc_setpoint)};
	uORB::Subscription _fsp_sub{ORB_ID(tvc_force_sp)};
	uORB::Subscription _st_sub{ORB_ID(tvc_vehicle_state)};
	uORB::Subscription _att_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _lpos_sub{ORB_ID(vehicle_local_position)};
	uORB::Publication<flight_log_vl_s> _log_pub{ORB_ID(flight_log_vl)};

	hrt_abstime _t0{0};
	uint32_t _samples{0};
	float _last_time_s{0.f};
};

extern "C" __EXPORT int tvc_log_main(int argc, char *argv[])
{
	return TvcLog::main(argc, argv);
}
