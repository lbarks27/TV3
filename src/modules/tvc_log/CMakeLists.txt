px4_add_module(
	MODULE modules__tvc_log
	MAIN tvc_log
	INCLUDES
		${CMAKE_CURRENT_LIST_DIR}/../../lib/tvc_core/include
	SRCS
		TvcLog.cpp
	DEPENDS
		tvc_core
	EXTERNAL
)
