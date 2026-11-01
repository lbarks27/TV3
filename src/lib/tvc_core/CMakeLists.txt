# Portable TVC core — float32, no intentional PX4 includes.
#
# Host: plain static library for unit tests.
# PX4 EXTERNAL_MODULES: still a static library, but compile order must wait for
# NuttX config.h (global -include visibility.h pulls NuttX headers).

set(TVC_CORE_SRC
	src/geometry.cpp
	src/quat.cpp
	src/frames.cpp
	src/allocator.cpp
	src/controller.cpp
	src/motors.cpp
	src/vehicle_config.cpp
	src/mission_config.cpp
)

if(COMMAND px4_add_module)
	# NuttX config is generated into the *source* NuttX tree by nuttx_context.
	if(DEFINED PX4_SOURCE_DIR)
		set(_NUTTX_CONFIG_H "${PX4_SOURCE_DIR}/platforms/nuttx/NuttX/nuttx/include/nuttx/config.h")
		set_source_files_properties(${TVC_CORE_SRC} PROPERTIES
			OBJECT_DEPENDS "${_NUTTX_CONFIG_H}"
		)
	endif()

	add_library(tvc_core STATIC ${TVC_CORE_SRC})
	target_include_directories(tvc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
	target_compile_options(tvc_core PRIVATE -Wno-float-equal -Wno-error=float-equal)

	# Prefer explicit dependency when the target exists (after full configure).
	# At generate time this works if platforms was already processed; if not,
	# OBJECT_DEPENDS above still orders Ninja correctly.
	if(TARGET nuttx_context)
		add_dependencies(tvc_core nuttx_context)
	endif()
else()
	add_library(tvc_core STATIC ${TVC_CORE_SRC})
	target_include_directories(tvc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
	target_compile_options(tvc_core PRIVATE -Wno-float-equal -Wno-error=float-equal)
	target_compile_features(tvc_core PUBLIC cxx_std_17)

	# Shared library + C ABI for SIH / host tools (not linked into PX4).
	add_library(tvc_core_host SHARED ${TVC_CORE_SRC} src/host_api.cpp)
	target_include_directories(tvc_core_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
	target_compile_options(tvc_core_host PRIVATE -Wno-float-equal -Wno-error=float-equal)
	target_compile_features(tvc_core_host PUBLIC cxx_std_17)
	set_target_properties(tvc_core_host PROPERTIES
		OUTPUT_NAME tvc_core_host
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
	)
endif()
