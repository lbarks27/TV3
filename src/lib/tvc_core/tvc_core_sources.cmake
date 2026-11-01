# Absolute paths to tvc_core sources for embedding into PX4 modules.
# Include from module CMakeLists: include(.../tvc_core_sources.cmake)

get_filename_component(_TVC_CORE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(TVC_CORE_SRCS
	${_TVC_CORE_DIR}/src/geometry.cpp
	${_TVC_CORE_DIR}/src/quat.cpp
	${_TVC_CORE_DIR}/src/frames.cpp
	${_TVC_CORE_DIR}/src/allocator.cpp
	${_TVC_CORE_DIR}/src/controller.cpp
	${_TVC_CORE_DIR}/src/motors.cpp
	${_TVC_CORE_DIR}/src/vehicle_config.cpp
	${_TVC_CORE_DIR}/src/mission_config.cpp
)

set(TVC_CORE_INCLUDE_DIR ${_TVC_CORE_DIR}/include)
