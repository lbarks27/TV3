px4_add_module(
	MODULE modules__tvc_mission
	MAIN tvc_mission
	INCLUDES
		${CMAKE_CURRENT_LIST_DIR}/../../lib/tvc_core/include
	SRCS
		TvcMission.cpp
	DEPENDS
		tvc_core
	EXTERNAL
)
