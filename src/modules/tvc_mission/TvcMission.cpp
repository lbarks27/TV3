/**
 * @file TvcMission.cpp
 * Mission reference generator — port of Prototype FlightScenario real_landing / hover.
 */

#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>
#include <drivers/drv_hrt.h>
#include <mathlib/mathlib.h>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionInterval.hpp>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_command_ack.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/tvc_setpoint.h>
#include <uORB/topics/tvc_vehicle_state.h>
#include <systemlib/mavlink_log.h>

#include <tvc/frames.hpp>
#include <tvc/gcs_cmd.hpp>
#include <tvc/mission_config.hpp>
#include <tvc/px4_gcs.hpp>
#include <tvc/real_landing.hpp>
#include <cmath>
#include <cstring>

using namespace time_literals;

class TvcMission : public ModuleBase<TvcMission>, public ModuleParams, public px4::ScheduledWorkItem
{
public:
	TvcMission() : ModuleParams(nullptr), ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::nav_and_controllers) {}

	static int task_spawn(int argc, char *argv[])
	{
		TvcMission *inst = new TvcMission();
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
		if (argc >= 1 && !strcmp(argv[0], "reload")) {
			TvcMission *inst = get_instance();

			if (!inst) {
				return 1;
			}

			inst->load_mission();
			return 0;
		}

		return print_usage("unknown");
	}
	static int print_usage(const char *reason = nullptr)
	{
		if (reason) { PX4_WARN("%s", reason); }
		PRINT_MODULE_DESCRIPTION(R"DESCR_STR(
### Description
Publishes tvc_setpoint from hover / real_landing style profiles.
Loads /fs/microsd/tvc/mission.json when packed onto the SD card.
)DESCR_STR");
		PRINT_MODULE_USAGE_NAME("tvc_mission", "commander");
		PRINT_MODULE_USAGE_COMMAND("start");
		PRINT_MODULE_USAGE_COMMAND_DESCR("reload", "Reload /fs/microsd/tvc/mission.json");
		PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
		return 0;
	}

	int print_status() override
	{
		const tvc::MissionConfig &m = tvc::loaded_mission_config();
		PX4_INFO("sd_mission valid=%d status=%u name=%s mode=%d loft=%d (%.1f, %.1f, %.1f) burn_end=%.2f throttle=%.2f land_tilt=%.1f",
			 (int)m.valid, (unsigned)m.status, m.name, m.mode, (int)m.has_loft,
			 (double)m.loft_e, (double)m.loft_n, (double)m.loft_u, (double)m.burn_end_s,
			 (double)m.throttle_min, (double)m.land_tilt_deg);
		return 0;
	}

	bool init()
	{
		load_mission();
		ScheduleOnInterval(50_ms); // 20 Hz
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

		if (_parameter_update_sub.updated()) {
			parameter_update_s p; _parameter_update_sub.copy(&p);
			updateParams();
		}

		handle_gcs_commands();

		vehicle_local_position_s lpos{};
		if (!_lpos_sub.copy(&lpos)) {
			return;
		}

		const tvc::Vec3 p = tvc::ned_to_enu(tvc::Vec3{lpos.x, lpos.y, lpos.z});
		const tvc::Vec3 v = tvc::ned_to_enu(tvc::Vec3{lpos.vx, lpos.vy, lpos.vz});

		tvc_vehicle_state_s st{};
		_state_sub.copy(&st);

		const tvc::MissionConfig &mis = tvc::loaded_mission_config();
		const int mode = mis.valid ? mis.mode : _param_mode.get();

		// Hover can run dry from ARMED. real_landing times approach /
		// hover-hold from fire_latched so a long launch-ready does not eat
		// min_approach_time (pad 10 s hold, or loft capture).
		const bool fired = st.fire_latched ||
				   st.state == tvc_vehicle_state_s::STATE_BURN ||
				   st.state == tvc_vehicle_state_s::STATE_COAST;
		const bool clock_ok = fired ||
				      (mode == 1 && st.state == tvc_vehicle_state_s::STATE_ARMED);

		if (clock_ok && _t0 == 0) {
			_t0 = hrt_absolute_time();
		}
		if (st.state == tvc_vehicle_state_s::STATE_DISARMED) {
			_t0 = 0;
			_guide.reset();
		}

		const float t = (_t0 != 0) ? (hrt_absolute_time() - _t0) * 1e-6f : 0.f;
		const float hover_z = mis.valid ? mis.hover_z : _param_hover_z.get();
		const float wp_e = mis.valid ? mis.wp_e : _param_wp_e.get();
		const float wp_u = mis.valid ? mis.wp_u : _param_wp_u.get();
		const float cap_r = mis.valid ? mis.capture_radius_m : _param_cap_r.get();
		const float land_t = mis.valid ? mis.descent_duration_s : _param_land_t.get();
		const float min_appr = mis.valid ? mis.min_approach_time_s : _param_min_appr.get();

		tvc_setpoint_s sp{};
		sp.timestamp = hrt_absolute_time();
		sp.heading_rad = (mis.valid && PX4_ISFINITE(mis.heading_rad)) ? mis.heading_rad : NAN;

		if (mode == 0) {
			// hold current
			sp.pos[0] = p.x; sp.pos[1] = p.y; sp.pos[2] = p.z;
			sp.vel[0] = sp.vel[1] = sp.vel[2] = 0.f;
			sp.phase = 0;
		} else if (mode == 1) {
			// hover above the captured pad
			tvc::Vec3 site{};

			if (st.origin_set) {
				site = tvc::ned_to_enu(tvc::Vec3{st.pad_x, st.pad_y, st.pad_z});
			}

			sp.pos[0] = site.x;
			sp.pos[1] = site.y;
			sp.pos[2] = site.z + hover_z;
			sp.vel[0] = sp.vel[1] = sp.vel[2] = 0.f;
			sp.phase = 0;
		} else {
			// real_landing: loft, staging waypoint, then ease onto the pad.
			tvc::Vec3 site{};

			if (st.origin_set) {
				site = tvc::ned_to_enu(tvc::Vec3{st.pad_x, st.pad_y, st.pad_z});
			}

			tvc::RealLandingConfig cfg{};
			cfg.has_loft = mis.valid && mis.has_loft;
			cfg.loft = {mis.loft_e + site.x, mis.loft_n + site.y, mis.loft_u + site.z};
			cfg.staging = {wp_e + site.x, (mis.valid ? mis.wp_n : 0.f) + site.y, wp_u + site.z};
			cfg.pad = {site.x, site.y, site.z + (mis.valid ? mis.land_z : 0.f)};
			cfg.vertical_hold_s = mis.valid ? mis.vertical_hold_s : 0.f;
			cfg.capture_radius_m = cap_r;
			cfg.capture_max_speed_m_s = mis.valid ? mis.capture_max_speed_m_s : 2.f;
			cfg.capture_settle_s = mis.valid ? mis.capture_settle_s : 0.f;
			cfg.stage_radius_m = mis.valid ? mis.stage_radius_m : 1.f;
			cfg.stage_speed_m_s = mis.valid ? mis.stage_speed_m_s : 1.2f;
			cfg.min_approach_time_s = min_appr;
			cfg.descent_duration_s = land_t;
			cfg.burn_end_s = mis.valid ? mis.burn_end_s : 0.f;
			cfg.pad_remain_s = mis.valid ? mis.pad_remain_s : 0.f;
			cfg.burn_gate = mis.valid && mis.has_burn_gate;

			tvc::RealLandingSample sample{};
			sample.t_s = t;
			sample.pos = p;
			sample.vel = v;
			sample.vel_valid = lpos.v_xy_valid && lpos.v_z_valid;
			const tvc::RealLandingOutput ref = _guide.update(sample, cfg);
			sp.phase = static_cast<uint8_t>(ref.phase);
			sp.pos[0] = ref.pos.x;
			sp.pos[1] = ref.pos.y;
			sp.pos[2] = ref.pos.z;
			sp.vel[0] = ref.vel.x;
			sp.vel[1] = ref.vel.y;
			sp.vel[2] = ref.vel.z;
		}

		_sp_pub.publish(sp);
	}

	void load_mission()
	{
		tvc::MissionConfig mcfg;
		if (tvc::load_mission_config_file(tvc::kMissionJsonPath, mcfg) && mcfg.valid) {
			PX4_INFO("mission.json ok name=%s mode=%d loft=%d (%.1f, %.1f, %.1f) burn_end=%.2f wp_e=%.2f wp_u=%.2f",
				 mcfg.name, mcfg.mode, (int)mcfg.has_loft,
				 (double)mcfg.loft_e, (double)mcfg.loft_n, (double)mcfg.loft_u,
				 (double)mcfg.burn_end_s, (double)mcfg.wp_e, (double)mcfg.wp_u);
			mavlink_log_info(&_mavlink_log_pub, "TVC: mission %s", mcfg.name);
		} else {
			PX4_WARN("mission.json missing/invalid status=%u — using TVC_MIS_* params",
				 (unsigned)mcfg.status);
			mavlink_log_critical(&_mavlink_log_pub, "TVC: mission.json invalid");
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

			if (tvc::gcs_action_from_param1(cmd.param1) == tvc::GcsAction::ReloadMission) {
				load_mission();
				tvc::ack_vehicle_command(_vehicle_command_ack_pub, cmd,
							 vehicle_command_ack_s::VEHICLE_CMD_RESULT_ACCEPTED);
			}
		}
	}

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _lpos_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _state_sub{ORB_ID(tvc_vehicle_state)};
	uORB::Subscription _vehicle_command_sub{ORB_ID(vehicle_command)};
	uORB::Publication<tvc_setpoint_s> _sp_pub{ORB_ID(tvc_setpoint)};
	uORB::Publication<vehicle_command_ack_s> _vehicle_command_ack_pub{ORB_ID(vehicle_command_ack)};

	hrt_abstime _t0{0};
	tvc::RealLandingGuide _guide{};
	orb_advert_t _mavlink_log_pub{nullptr};

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::TVC_MIS_MODE>) _param_mode,
		(ParamFloat<px4::params::TVC_HOVER_Z>) _param_hover_z,
		(ParamFloat<px4::params::TVC_WP_E>) _param_wp_e,
		(ParamFloat<px4::params::TVC_WP_U>) _param_wp_u,
		(ParamFloat<px4::params::TVC_CAP_R>) _param_cap_r,
		(ParamFloat<px4::params::TVC_LAND_T>) _param_land_t,
		(ParamFloat<px4::params::TVC_MIN_APPR>) _param_min_appr
	)
};

extern "C" __EXPORT int tvc_mission_main(int argc, char *argv[])
{
	return TvcMission::main(argc, argv);
}
