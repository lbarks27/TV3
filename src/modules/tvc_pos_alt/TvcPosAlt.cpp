/**
 * @file TvcPosAlt.cpp
 * Outer position / altitude loops — Prototype PositionPID + AltitudeForce
 * cascade (z/vz → az_cmd → F = m(g+az) feedforward; no IMU accel loop).
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
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/tvc_attitude_sp.h>
#include <uORB/topics/tvc_force_sp.h>
#include <uORB/topics/tvc_mount_thrust.h>
#include <uORB/topics/tvc_setpoint.h>
#include <uORB/topics/tvc_vehicle_state.h>

#include <tvc/controller.hpp>
#include <tvc/frames.hpp>
#include <tvc/mission_config.hpp>
#include <tvc/quat.hpp>
#include <tvc/vehicle_config.hpp>

using namespace time_literals;

class TvcPosAlt : public ModuleBase<TvcPosAlt>, public ModuleParams, public px4::ScheduledWorkItem
{
public:
	TvcPosAlt() : ModuleParams(nullptr), ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::nav_and_controllers) {}

	static int task_spawn(int argc, char *argv[])
	{
		TvcPosAlt *inst = new TvcPosAlt();
		if (!inst) { return PX4_ERROR; }
		_object.store(inst);
		_task_id = task_id_is_work_queue;
		if (!inst->init()) {
			delete inst; _object.store(nullptr); _task_id = -1; return PX4_ERROR;
		}
		return PX4_OK;
	}
	static int custom_command(int, char **) { return print_usage("unknown"); }
	static int print_usage(const char *reason = nullptr)
	{
		if (reason) { PX4_WARN("%s", reason); }
		PRINT_MODULE_DESCRIPTION(R"DESCR_STR(
### Description
50 Hz position → attitude setpoint and altitude → axial force setpoint.
)DESCR_STR");
		PRINT_MODULE_USAGE_NAME("tvc_pos_alt", "controller");
		PRINT_MODULE_USAGE_COMMAND("start");
		PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
		return 0;
	}

	bool init()
	{
		tvc::VehicleConfig vcfg;
		(void)tvc::load_vehicle_config_file(tvc::kVehicleJsonPath, tvc::kVehicleCrcPath, vcfg);
		tvc::MissionConfig mcfg;
		(void)tvc::load_mission_config_file(tvc::kMissionJsonPath, mcfg);
		update_gains();
		ScheduleOnInterval(20_ms);
		return true;
	}

private:
	void update_gains()
	{
		updateParams();
		tvc::PositionPIDConfig pc;
		tvc::AltitudeForceConfig ac;
		const tvc::MissionConfig &mis = tvc::loaded_mission_config();
		// Defaults match the pre-profile law. A packed mission overrides them.
		pc.integral_start_m = 1.5f;
		pc.lateral_priority_m = 0.05f;
		pc.lateral_accel_lpf_hz = 0.f;
		pc.integral_limit = 0.08f;

		if (mis.valid && mis.has_gains) {
			pc.kp[0] = mis.pos_kp;
			pc.kp[1] = mis.pos_kp_y;
			pc.kd[0] = mis.pos_kd;
			pc.kd[1] = mis.pos_kd_y;
			pc.max_tilt_deg = mis.pos_tilt_deg;
			pc.max_tilt_y_deg = mis.pos_tilt_y_deg;
			pc.max_speed_m_s = mis.max_speed_m_s;
			pc.max_speed_y_m_s = mis.max_speed_y_m_s;
			pc.max_position_error_m = mis.max_position_error_m;
			pc.max_position_error_y_m = mis.max_position_error_y_m;
			pc.prefer_lateral_y = mis.prefer_lateral_y;
			ac.kp = mis.alt_kp;
			ac.kd = mis.alt_kd;
			ac.ki = mis.alt_ki;
			if (mis.has_alt_i_limit) {
				ac.integral_limit = mis.alt_i_limit;
			}
			if (mis.has_pos_ki) {
				pc.ki[0] = mis.pos_ki;
				pc.ki[1] = mis.pos_ki_y;
			}
			if (mis.has_pos_i_limit) {
				pc.integral_limit = mis.pos_i_limit;
			}
			if (mis.has_pos_i_start) {
				pc.integral_start_m = mis.pos_i_start;
			}
			if (mis.has_lat_priority) {
				pc.lateral_priority_m = mis.lat_priority;
			}
			if (mis.has_lat_lpf) {
				pc.lateral_accel_lpf_hz = mis.lat_lpf_hz;
			}
		} else {
			pc.kp[0] = pc.kp[1] = _param_pos_p.get();
			pc.kd[0] = pc.kd[1] = _param_pos_d.get();
			pc.max_tilt_deg = _param_pos_tilt.get();
			ac.kp = _param_alt_p.get();
			ac.kd = _param_alt_d.get();
			ac.ki = _param_alt_i.get();
		}
		// Single outer law: always cascaded position → velocity → tilt.
		pc.velocity_cascade = true;

		_pos.set_config(pc);
		_alt.set_config(ac);

		const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
		if (vcfg.valid) {
			_pos.set_body_up(vcfg.loaded_com);
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
			update_gains();
		}

		const hrt_abstime now = hrt_absolute_time();
		float dt = 0.02f;
		if (_last != 0) {
			dt = math::constrain((now - _last) * 1e-6f, 0.001f, 0.1f);
		}
		_last = now;

		vehicle_local_position_s lpos{};
		if (!_lpos_sub.copy(&lpos) || !lpos.xy_valid || !lpos.z_valid) {
			return;
		}

		// NED → ENU-sim
		const tvc::Vec3 p_ned{lpos.x, lpos.y, lpos.z};
		const tvc::Vec3 v_ned{lpos.vx, lpos.vy, lpos.vz};
		const tvc::Vec3 p = tvc::ned_to_enu(p_ned);
		const tvc::Vec3 v = tvc::ned_to_enu(v_ned);

		tvc_setpoint_s sp{};
		bool have_sp = _sp_sub.copy(&sp);
		tvc::Vec3 p_ref = have_sp ? tvc::Vec3{sp.pos[0], sp.pos[1], sp.pos[2]} : p;
		tvc::Vec3 v_ref = have_sp ? tvc::Vec3{sp.vel[0], sp.vel[1], sp.vel[2]} : tvc::Vec3{};

		// Hover: XY tilt from pos PID with z held to current (altitude via force)
		tvc::Vec3 hover_pref = p_ref;
		hover_pref.z = p.z;
		tvc::Vec3 hover_vref = v_ref;
		if (!(have_sp && sp.phase == 2)) { // not landing
			hover_vref.z = 0.f;
		}

		const tvc::Vec3 gravity{0.f, 0.f, -9.81f};
		const bool heading_valid = have_sp && PX4_ISFINITE(sp.heading_rad);
		const float heading = heading_valid ? sp.heading_rad : 0.f;

		// Lift gate matches the G12 profile hover margin (0.95 mg).
		float sum_T = 0.f;
		tvc_mount_thrust_s thr{};
		if (_thrust_sub.copy(&thr)) {
			sum_T = thr.thrust[0] + thr.thrust[1] + thr.thrust[2];
		}
		// Loaded mass comes from SD vehicle.json (unloaded + motor masses at pivots).
		const tvc::VehicleConfig &vcfg = tvc::loaded_vehicle_config();
		const float mass = vcfg.valid ? vcfg.loaded_mass_kg
					     : (_param_mass.get() > 0.f ? _param_mass.get() : 3.0556f);
		const float weight = mass * 9.81f;
		const bool lift_ok = sum_T >= 0.95f * weight;

		tvc::Quat q_ref;
		if (!lift_ok) {
			// CoM-up level / heading-only (matches attitude_reference body_up)
			const tvc::Vec3 world_up{0.f, 0.f, 1.f};
			q_ref = tvc::q_from_body_up_and_thrust(world_up, _pos.body_up(), heading,
							       heading_valid);
		} else {
			const tvc::MissionConfig &mis_now = tvc::loaded_mission_config();
			tvc_vehicle_state_s origin{};
			const bool have_origin = _state_sub.copy(&origin) && origin.origin_set;
			tvc::Vec3 site{};

			if (have_origin) {
				site = tvc::ned_to_enu(tvc::Vec3{origin.pad_x, origin.pad_y, origin.pad_z});
			}

			const float profile_tilt = _pos.max_tilt_deg();
			const float tilt = tvc::landing_tilt_deg(
				profile_tilt,
				(mis_now.valid && have_origin) ? mis_now.land_tilt_deg : -1.f,
				p.x, p.y, v.x, v.y, site.x, site.y,
				have_sp && sp.phase == 2);

			if (tilt != profile_tilt) {
				_pos.set_max_tilt_deg(tilt);
			}

			q_ref = _pos.attitude_reference(p, v, hover_pref, hover_vref, gravity,
							heading, dt, heading_valid);

			if (tilt != profile_tilt) {
				_pos.set_max_tilt_deg(profile_tilt);
			}
		}

		tvc_attitude_sp_s asp{};
		asp.timestamp = now;
		asp.q_sim[0] = q_ref.w;
		asp.q_sim[1] = q_ref.x;
		asp.q_sim[2] = q_ref.y;
		asp.q_sim[3] = q_ref.z;
		_att_sp_pub.publish(asp);

		// Axial mix arm
		tvc_vehicle_state_s st{};
		bool mix_armed = false;
		float burn_t = 0.f;
		if (_state_sub.copy(&st) && st.fire_latched) {
			if (_fire_t0 == 0) {
				_fire_t0 = now;
			}
			burn_t = (now - _fire_t0) * 1e-6f;
			const float arm_s = _param_axmix_arm.get();
			if (arm_s >= 0.f) {
				mix_armed = burn_t >= arm_s;
			} else {
				mix_armed = lift_ok;
			}

			// Loft and the pad ease mix as soon as there is thrust. The
			// caught hop does not wait out a fixed 5 s lock at a1 rest.
			if (have_sp && sp.phase == 3) {
				mix_armed = true;
			} else if (have_sp && sp.phase == 2 && sum_T > 0.25f) {
				mix_armed = true;
			}
		} else {
			_fire_t0 = 0;
			// hover without solids: allow mix when mass valid
			mix_armed = vcfg.valid || (_param_mass.get() > 0.f);
		}

		// Landing integral reset once on phase entry
		if (have_sp && sp.phase == 2 && _last_phase != 2) {
			_alt.reset();
		}
		_last_phase = have_sp ? sp.phase : 0;

		// Outer position / altitude loops — Prototype PositionPID + AltitudeForce
		// cascade (altitude → az_cmd → F = m(g+az) feedforward).
		// After the loft, the host sim does not step the altitude target down to
		// the staging waypoint. It tracks the current height at a sink rate, and
		// it will not command less than throttle_min of the thrust on hand.
		// Without those, this loop saturates to zero force and a1 sits on the stop.
		const tvc::MissionConfig &mis = tvc::loaded_mission_config();
		float z_ref = p_ref.z;
		float vz_ref = v_ref.z;
		const bool after_loft = have_sp && mis.valid && mis.mode == 2 && mis.has_loft &&
					(sp.phase == 1 || sp.phase == 2);
		const float loft_z = (mis.valid && mis.has_loft) ? mis.loft_u : 30.f;
		const bool on_slide = have_sp && sp.phase == 2;
		const tvc::LoftReturnAlt sunk = tvc::loft_return_altitude(
			p.z, z_ref, vz_ref, mis.wp_u, after_loft, loft_z, on_slide);
		z_ref = sunk.z_ref;
		vz_ref = sunk.vz_ref;
		const tvc::AltitudeForceResult alt = _alt.update(p.z, v.z, z_ref, dt, mass, 9.81f,
								 vz_ref);
		float F = alt.force_n;
		F = tvc::axial_thrust_floor(F, sum_T, mis.valid ? mis.throttle_min : 0.f);
		if (sum_T > 0.f) {
			F = math::min(F, 0.98f * sum_T);
		}

		tvc_force_sp_s fsp{};
		fsp.timestamp = now;
		fsp.f_axial_n = F;
		fsp.az_cmd = alt.az_cmd;
		const float fw = (mis.valid && mis.force_weight > 0.f) ? mis.force_weight : 0.25f;
		fsp.force_weight = mix_armed ? fw : 0.f;
		fsp.mix_armed = mix_armed;
		// The pad box used to freeze attitude and replace this force with
		// 0.75 weight. Both loops keep tracking the guide through touchdown.
		fsp.safe_land = false;
		_force_sp_pub.publish(fsp);
	}

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _lpos_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _sp_sub{ORB_ID(tvc_setpoint)};
	uORB::Subscription _thrust_sub{ORB_ID(tvc_mount_thrust)};
	uORB::Subscription _state_sub{ORB_ID(tvc_vehicle_state)};
	uORB::Publication<tvc_attitude_sp_s> _att_sp_pub{ORB_ID(tvc_attitude_sp)};
	uORB::Publication<tvc_force_sp_s> _force_sp_pub{ORB_ID(tvc_force_sp)};

	tvc::PositionPIDController _pos;
	tvc::AltitudeForceController _alt;
	hrt_abstime _last{0};
	hrt_abstime _fire_t0{0};
	uint8_t _last_phase{0};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::TVC_POS_P>) _param_pos_p,
		(ParamFloat<px4::params::TVC_POS_D>) _param_pos_d,
		(ParamFloat<px4::params::TVC_POS_TILT>) _param_pos_tilt,
		(ParamFloat<px4::params::TVC_ALT_P>) _param_alt_p,
		(ParamFloat<px4::params::TVC_ALT_D>) _param_alt_d,
		(ParamFloat<px4::params::TVC_ALT_I>) _param_alt_i,
		(ParamFloat<px4::params::TVC_AXMIX_ARM>) _param_axmix_arm,
		(ParamFloat<px4::params::TVC_MASS>) _param_mass
	)
};

extern "C" __EXPORT int tvc_pos_alt_main(int argc, char *argv[])
{
	return TvcPosAlt::main(argc, argv);
}
