#pragma once

// PX4-only GCS helpers. Do not include from host tests.

#include <drivers/drv_hrt.h>
#include <uORB/Publication.hpp>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_command_ack.h>

#include <tvc/gcs_cmd.hpp>

namespace tvc {

inline void ack_vehicle_command(uORB::Publication<vehicle_command_ack_s> &pub,
				const vehicle_command_s &cmd, uint8_t result)
{
	vehicle_command_ack_s ack{};
	ack.timestamp = hrt_absolute_time();
	ack.command = cmd.command;
	ack.result = result;
	ack.target_system = cmd.source_system;
	ack.target_component = cmd.source_component;
	pub.publish(ack);
}

inline bool is_external_gcs_command(const vehicle_command_s &cmd)
{
	return cmd.from_external;
}

} // namespace tvc
