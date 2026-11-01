px4_add_module(
	MODULE modules__tvc_control
	MAIN tvc_control
	INCLUDES
		${CMAKE_CURRENT_LIST_DIR}/../../lib/tvc_core/include
	SRCS
		TvcControl.cpp
	DEPENDS
		tvc_core
	EXTERNAL
)
