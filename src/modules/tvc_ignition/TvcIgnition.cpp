/**
 * @file TvcIgnition.cpp
 *
 * Onboard ignition is disabled. Grains are lit by a ground system that is
 * electrically disconnected from the Cube. tvc_safety starts the mission on
 * first motion after LAUNCH_READY.
 *
 * This module only publishes tvc_ignition_status so existing listeners/logs
 * still have a topic. It never allocates or drives FMU GPIO.
 */

#include <px4_platform_common/module.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>
#include <drivers/drv_hrt.h>
#include <uORB/Publication.hpp>
#include <uORB/topics/tvc_ignition_status.h>

using namespace time_literals;

class TvcIgnition : public ModuleBase<TvcIgnition>, public px4::ScheduledWorkItem
{
public:
	TvcIgnition() :
		ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::lp_default)
	{}

	static int task_spawn(int argc, char *argv[])
	{
		TvcIgnition *inst = new TvcIgnition();

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
Onboard igniter GPIO is removed. Ground ignition is independent of the Cube.
Mission start is first motion in tvc_safety LAUNCH_READY.
)DESCR_STR");
		PRINT_MODULE_USAGE_NAME("tvc_ignition", "driver");
		PRINT_MODULE_USAGE_COMMAND("start");
		PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
		return 0;
	}

	int print_status() override
	{
		PX4_INFO("onboard ignition disabled (ground igniter + first motion)");
		PX4_INFO("gpio_en=0 fire=0 pulse=0 setup_ok=1");
		return 0;
	}

	bool init()
	{
		PX4_INFO("tvc_ignition: onboard fire GPIO disabled");
		ScheduleOnInterval(200_ms);
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

		tvc_ignition_status_s st{};
		st.timestamp = hrt_absolute_time();
		st.gpio_enabled = false;
		st.fire_asserted = false;
		st.pulse_active = false;
		st.setup_ok = true;
		st.last_error = 0;
		_status_pub.publish(st);
	}

	uORB::Publication<tvc_ignition_status_s> _status_pub{ORB_ID(tvc_ignition_status)};
};

extern "C" __EXPORT int tvc_ignition_main(int argc, char *argv[])
{
	return TvcIgnition::main(argc, argv);
}
