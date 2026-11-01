px4_add_module(
	MODULE modules__tvc_actuators
	MAIN tvc_actuators
	INCLUDES
		${CMAKE_CURRENT_LIST_DIR}/../../lib/tvc_core/include
	SRCS
		TvcActuators.cpp
	EXTERNAL
)
