px4_add_module(
	MODULE modules__tvc_pos_alt
	MAIN tvc_pos_alt
	INCLUDES
		${CMAKE_CURRENT_LIST_DIR}/../../lib/tvc_core/include
	SRCS
		TvcPosAlt.cpp
	DEPENDS
		tvc_core
	EXTERNAL
)
