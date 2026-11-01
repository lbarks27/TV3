#pragma once

#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>

#include <drivers/drv_hrt.h>
#include <lib/perf/perf_counter.h>

#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionInterval.hpp>
#include <uORB/topics/manual_control_setpoint.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_command_ack.h>
#include <uORB/topics/vehicle_status.h>

#include <uORB/topics/tvc_attitude_sp.h>
#include <uORB/topics/tvc_control_debug.h>
#include <uORB/topics/tvc_force_sp.h>
#include <uORB/topics/tvc_gimbal_cmd.h>
#include <uORB/topics/tvc_mount_thrust.h>
#include <uORB/topics/tvc_vehicle_state.h>

#include <tvc/allocator.hpp>
#include <tvc/controller.hpp>
#include <tvc/frames.hpp>
#include <tvc/gcs_cmd.hpp>
#include <tvc/px4_gcs.hpp>
#include <tvc/vehicle_config.hpp>

using namespace time_literals;

class TvcControl : public ModuleBase<TvcControl>, public ModuleParams, public px4::ScheduledWorkItem
{
public:
	TvcControl();
	~TvcControl() override;

	static int task_spawn(int argc, char *argv[]);
	static int custom_command(int argc, char *argv[]);
	static int print_usage(const char *reason = nullptr);

	bool init();
	int print_status() override;

private:
	void Run() override;
	void update_params();
	void apply_vehicle_mounts();
	void publish_rest_cmd();
	void handle_gcs_commands();
	static void set_bench_mode(bool on);

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _manual_control_setpoint_sub{ORB_ID(manual_control_setpoint)};
	uORB::Subscription _vehicle_attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _vehicle_angular_velocity_sub{ORB_ID(vehicle_angular_velocity)};
	uORB::Subscription _tvc_attitude_sp_sub{ORB_ID(tvc_attitude_sp)};
	uORB::Subscription _tvc_force_sp_sub{ORB_ID(tvc_force_sp)};
	uORB::Subscription _tvc_mount_thrust_sub{ORB_ID(tvc_mount_thrust)};
	uORB::Subscription _tvc_vehicle_state_sub{ORB_ID(tvc_vehicle_state)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _vehicle_command_sub{ORB_ID(vehicle_command)};

	uORB::Publication<tvc_gimbal_cmd_s> _tvc_gimbal_cmd_pub{ORB_ID(tvc_gimbal_cmd)};
	uORB::Publication<tvc_control_debug_s> _tvc_control_debug_pub{ORB_ID(tvc_control_debug)};
	uORB::Publication<vehicle_command_ack_s> _vehicle_command_ack_pub{ORB_ID(vehicle_command_ack)};

	perf_counter_t _loop_perf{perf_alloc(PC_ELAPSED, MODULE_NAME": cycle")};
	perf_counter_t _loop_interval_perf{perf_alloc(PC_INTERVAL, MODULE_NAME": interval")};

	tvc::TorqueAllocator _allocator;
	tvc::AttitudePIDController _att_pid;
	float _angles_cmd[tvc::kNumAngles]{};
	float _angles_slew[tvc::kNumAngles]{};
	hrt_abstime _last_run{0};
	hrt_abstime _att_lost_since{0};
	bool _bench_heading_valid{false};
	float _bench_heading_rad{0.f};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::TVC_ATT_RP>) _param_att_rp,
		(ParamFloat<px4::params::TVC_ATT_PP>) _param_att_pp,
		(ParamFloat<px4::params::TVC_ATT_YP>) _param_att_yp,
		(ParamFloat<px4::params::TVC_ATT_RD>) _param_att_rd,
		(ParamFloat<px4::params::TVC_ATT_PD>) _param_att_pd,
		(ParamFloat<px4::params::TVC_ATT_YD>) _param_att_yd,
		(ParamFloat<px4::params::TVC_ATT_RI>) _param_att_ri,
		(ParamFloat<px4::params::TVC_ATT_PI>) _param_att_pi,
		(ParamFloat<px4::params::TVC_ATT_YI>) _param_att_yi,
		(ParamFloat<px4::params::TVC_TAU_LIM>) _param_tau_lim,
		(ParamFloat<px4::params::TVC_FORCE_W>) _param_force_w,
		(ParamFloat<px4::params::TVC_SLEW_DPS>) _param_slew_dps,
		(ParamFloat<px4::params::TVC_MASS>) _param_mass,
		(ParamFloat<px4::params::TVC_COM_Z>) _param_com_z,
		(ParamInt<px4::params::TVC_CTRL_HZ>) _param_ctrl_hz,
		(ParamInt<px4::params::TVC_BENCH>) _param_bench,
		(ParamFloat<px4::params::TVC_BENCH_THR>) _param_bench_thr,
		(ParamFloat<px4::params::TVC_A2_MAX>) _param_a2_max,
		(ParamFloat<px4::params::TVC_A2_YAW>) _param_a2_yaw,
		(ParamFloat<px4::params::TVC_THR_NOM>) _param_thr_nom
	)
};
