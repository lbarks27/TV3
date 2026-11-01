px4_add_module(
	MODULE modules__tvc_thrust
	MAIN tvc_thrust
	INCLUDES
		${CMAKE_CURRENT_LIST_DIR}/../../lib/tvc_core/include
	SRCS
		TvcThrust.cpp
	DEPENDS
		tvc_core
	EXTERNAL
)
