px4_add_module(
	MODULE modules__tvc_safety
	MAIN tvc_safety
	INCLUDES
		${CMAKE_CURRENT_LIST_DIR}/../../lib/tvc_core/include
	SRCS
		TvcSafety.cpp
	DEPENDS
		tvc_core
	EXTERNAL
)
