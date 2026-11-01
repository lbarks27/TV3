/**
 * @file TvcThrust.cpp
 *
 * Per-mount thrust publisher from motor curves only (SD CSV or synthetic trapezoid).
 * Thrust, lit, and burnout are model-based — there is no per-mount force sensing.
 */

#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>
#include <drivers/drv_hrt.h>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionInterval.hpp>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/tvc_mount_thrust.h>
#include <uORB/topics/tvc_vehicle_state.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_command_ack.h>

#include <tvc/gcs_cmd.hpp>
#include <tvc/motors.hpp>
#include <tvc/px4_gcs.hpp>
#include <tvc/thrust_curve.hpp>

using namespace time_literals;

class TvcThrust : public ModuleBase<TvcThrust>, public ModuleParams, public px4::ScheduledWorkItem
{
public:
	TvcThrust() :
		ModuleParams(nullptr),
		ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::lp_default)
	{
	}

	static int task_spawn(int argc, char *argv[])
	{
		TvcThrust *inst = new TvcThrust();

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
		if (argc >= 1 && !strcmp(argv[0], "reload")) {
			TvcThrust *inst = get_instance();

			if (!inst) {
				return 1;
			}

			inst->_cmd_reload = true;
			return 0;
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
Per-mount thrust from motor CSV curves on SD and/or a synthetic trapezoid.
Command: reload (re-read /fs/microsd/tvc/motor*.csv).
)DESCR_STR");
		PRINT_MODULE_USAGE_NAME("tvc_thrust", "estimator");
		PRINT_MODULE_USAGE_COMMAND("start");
		PRINT_MODULE_USAGE_COMMAND_DESCR("reload", "Reload /fs/microsd/tvc/motor*.csv");
		PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
		return 0;
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

			if (action == tvc::GcsAction::ReloadThrust) {
				_cmd_reload = true;
				tvc::ack_vehicle_command(_vehicle_command_ack_pub, cmd,
							 vehicle_command_ack_s::VEHICLE_CMD_RESULT_ACCEPTED);
			}
		}
	}

	int print_status() override
	{
		PX4_INFO("burn_t=%.2f", (double)_last_burn_t);

		for (int i = 0; i < 3; i++) {
			PX4_INFO("m%d T=%.1f N lit=%d bo=%d curve=%d",
				 i,
				 (double)_thrust_out[i],
				 (int)_last_lit[i],
				 (int)_last_bo[i],
				 (int)_curves[i].valid());
		}

		return 0;
	}

	bool init()
	{
		updateParams();
		try_load_curves();
		ScheduleOnInterval(20_ms); // 50 Hz
		return true;
	}

private:
	static constexpr int kMounts = 3;

	void try_load_curves()
	{
		const float dly[3] = {
			_param_dly0.get(), _param_dly1.get(), _param_dly2.get()
		};

		for (int i = 0; i < kMounts; i++) {
			_curves[i] = tvc::MotorCurve{};
			_curves[i].ignition_delay_s = dly[i];
		}

		if (_param_sdcurv.get() == 0) {
			return;
		}

		const char *paths[3] = {
			"/fs/microsd/tvc/motor0.csv",
			"/fs/microsd/tvc/motor1.csv",
			"/fs/microsd/tvc/motor2.csv"
		};

		for (int i = 0; i < kMounts; i++) {
			if (tvc::load_motor_csv(paths[i], _curves[i], dly[i])) {
				PX4_INFO("tvc_thrust: loaded %s (%d samples, dly=%.2f)",
					 paths[i], _curves[i].n_samples, (double)dly[i]);
			}
		}
	}

	float curve_thrust(int i, float burn_t) const
	{
		if (_curves[i].valid()) {
			return _curves[i].thrust_at(burn_t);
		}

		const float peak = _param_peak.get();

		if (peak > 0.f) {
			const float dly = (i == 0) ? _param_dly0.get()
					 : (i == 1) ? _param_dly1.get() : _param_dly2.get();
			return tvc::trapezoid_thrust_n(burn_t - dly, peak,
						       _param_burn.get(),
						       _param_rise.get(),
						       _param_fall.get());
		}

		return _param_nom.get();
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

		handle_gcs_commands();

		if (_cmd_reload) {
			_cmd_reload = false;
			try_load_curves();
		}

		tvc_vehicle_state_s st{};
		const bool have_st = _state_sub.copy(&st);

		float burn_t = 0.f;

		if (have_st && st.fire_latched) {
			if (_fire_time == 0) {
				_fire_time = hrt_absolute_time();
			}

			burn_t = (hrt_absolute_time() - _fire_time) * 1e-6f;
		} else {
			_fire_time = 0;
		}

		_last_burn_t = burn_t;

		const float lit_th = _param_lit.get();
		const float bo_th = _param_bo.get();
		const bool fire = have_st && st.fire_latched;
		const bool bench = _param_bench.get() != 0;

		tvc_mount_thrust_s msg{};
		msg.timestamp = hrt_absolute_time();
		msg.burn_time_s = burn_t;

		for (int i = 0; i < kMounts; i++) {
			float T = 0.f;

			if (fire) {
				T = curve_thrust(i, burn_t);
			} else if (bench) {
				T = _param_nom.get() > 1.f ? _param_nom.get() : 12.f;
			}

			msg.thrust[i] = T;
			msg.valid[i] = true;
			msg.source[i] = (T > 0.f) ? 1 : 3; // 1=curve, 3=zero
			msg.lit[i] = T > lit_th;
			msg.burned_out[i] = (fire && burn_t > 0.5f && T < bo_th);

			_thrust_out[i] = T;
			_last_lit[i] = msg.lit[i];
			_last_bo[i] = msg.burned_out[i];
		}

		_thrust_pub.publish(msg);
	}

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _state_sub{ORB_ID(tvc_vehicle_state)};
	uORB::Subscription _vehicle_command_sub{ORB_ID(vehicle_command)};
	uORB::Publication<tvc_mount_thrust_s> _thrust_pub{ORB_ID(tvc_mount_thrust)};
	uORB::Publication<vehicle_command_ack_s> _vehicle_command_ack_pub{ORB_ID(vehicle_command_ack)};

	tvc::MotorCurve _curves[kMounts]{};

	float _thrust_out[kMounts]{};
	bool _last_lit[kMounts]{};
	bool _last_bo[kMounts]{};

	float _last_burn_t{0.f};
	bool _cmd_reload{false};
	hrt_abstime _fire_time{0};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::TVC_THR_NOM>) _param_nom,
		(ParamFloat<px4::params::TVC_THR_LIT>) _param_lit,
		(ParamFloat<px4::params::TVC_THR_BO>) _param_bo,
		(ParamFloat<px4::params::TVC_THR_DLY0>) _param_dly0,
		(ParamFloat<px4::params::TVC_THR_DLY1>) _param_dly1,
		(ParamFloat<px4::params::TVC_THR_DLY2>) _param_dly2,
		(ParamFloat<px4::params::TVC_THR_PEAK>) _param_peak,
		(ParamFloat<px4::params::TVC_THR_BURN>) _param_burn,
		(ParamFloat<px4::params::TVC_THR_RISE>) _param_rise,
		(ParamFloat<px4::params::TVC_THR_FALL>) _param_fall,
		(ParamInt<px4::params::TVC_THR_SDCURV>) _param_sdcurv,
		(ParamInt<px4::params::TVC_BENCH>) _param_bench
	)
};

extern "C" __EXPORT int tvc_thrust_main(int argc, char *argv[])
{
	return TvcThrust::main(argc, argv);
}
