# Out-of-tree TVC modules for PX4 (EXTERNAL_MODULES_LOCATION = this repo root)

add_subdirectory(lib/tvc_core)

set(config_module_list_external
	modules/tvc_control
	modules/tvc_actuators
	modules/tvc_thrust
	modules/tvc_safety
	modules/tvc_mission
	modules/tvc_pos_alt
	modules/tvc_ignition
	modules/tvc_log
	PARENT_SCOPE
)
