cmake_minimum_required(VERSION 3.16)
project(tvc_host_tests LANGUAGES CXX)

set(TVC_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/lib/tvc_core)

add_subdirectory(${TVC_CORE_DIR} tvc_core_build)

add_executable(test_allocator test_allocator.cpp)
target_link_libraries(test_allocator PRIVATE tvc_core)

add_executable(test_frames test_frames.cpp)
target_link_libraries(test_frames PRIVATE tvc_core)

add_executable(test_controllers test_controllers.cpp)
target_link_libraries(test_controllers PRIVATE tvc_core)

add_executable(test_thrust_curve test_thrust_curve.cpp)
target_link_libraries(test_thrust_curve PRIVATE tvc_core)

add_executable(test_vehicle_config test_vehicle_config.cpp)
target_link_libraries(test_vehicle_config PRIVATE tvc_core)
target_compile_definitions(test_vehicle_config PRIVATE
	TEST_VEHICLE_JSON="${CMAKE_CURRENT_SOURCE_DIR}/../../config/vehicles/tvc_ring3_g12.json"
)

add_executable(test_mission_config test_mission_config.cpp)
target_link_libraries(test_mission_config PRIVATE tvc_core)

add_executable(test_gcs_cmd test_gcs_cmd.cpp)
target_link_libraries(test_gcs_cmd PRIVATE tvc_core)

add_executable(test_first_motion test_first_motion.cpp)
target_link_libraries(test_first_motion PRIVATE tvc_core)

add_executable(test_real_landing test_real_landing.cpp)
target_link_libraries(test_real_landing PRIVATE tvc_core)

add_executable(test_servo test_servo.cpp)
target_link_libraries(test_servo PRIVATE tvc_core)

enable_testing()
add_test(NAME allocator COMMAND test_allocator)
add_test(NAME frames COMMAND test_frames)
add_test(NAME controllers COMMAND test_controllers)
add_test(NAME thrust_curve COMMAND test_thrust_curve)
add_test(NAME vehicle_config COMMAND test_vehicle_config)
add_test(NAME mission_config COMMAND test_mission_config)
add_test(NAME gcs_cmd COMMAND test_gcs_cmd)
add_test(NAME first_motion COMMAND test_first_motion)
add_test(NAME real_landing COMMAND test_real_landing)
add_test(NAME servo COMMAND test_servo)
