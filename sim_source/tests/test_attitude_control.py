import unittest

import numpy as np

from tvc.attitude import (
    normalize_quaternion,
    quaternion_error_vector,
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_to_euler,
    rotate_vector,
)
from tvc.controller import AttitudePIDConfig, AttitudePIDController, PositionPIDController
from tvc.dynamics import VehicleProperties, VehicleState, integrate_6dof_rk4
from tvc.flight_log import FlightLog, FlightLogSample
from tvc.flight_sim import FlightScenario, FlightSimulator, apply_actuator_slew, summarize_log
from tvc.parameters import FlightTestConfig, build_default_config, load_vehicle_profile


class AttitudeMathTests(unittest.TestCase):
    def test_quaternion_round_trip(self) -> None:
        roll, pitch, yaw = np.deg2rad([12.0, -8.0, 45.0])
        q = quaternion_from_euler(roll, pitch, yaw)
        recovered = quaternion_to_euler(q)
        self.assertTrue(
            np.allclose(
                [roll, pitch, yaw],
                recovered,
                atol=1e-9,
            )
        )

    def test_zero_error_for_matching_attitude(self) -> None:
        q = quaternion_from_euler(0.2, -0.1, 0.4)
        error = quaternion_error_vector(q, q)
        self.assertLess(np.linalg.norm(error), 1e-12)


class AttitudePIDTests(unittest.TestCase):
    def test_controller_opposes_attitude_error(self) -> None:
        controller = AttitudePIDController(
            AttitudePIDConfig(
                kp=np.array([2.0, 2.0, 2.0]),
                ki=np.zeros(3),
                kd=np.zeros(3),
                torque_limit=10.0,
            )
        )
        q_ref = quaternion_from_euler(0.0, 0.0, 0.0)
        q = quaternion_from_euler(np.deg2rad(10.0), 0.0, 0.0)
        torque = controller.update(q, np.zeros(3), q_ref, 0.01)
        self.assertLess(torque[0], 0.0)
        self.assertAlmostEqual(torque[1], 0.0, places=6)
        self.assertAlmostEqual(torque[2], 0.0, places=6)


class VelocityAimTests(unittest.TestCase):
    def test_velocity_aim_steers_toward_line_of_sight(self) -> None:
        controller = PositionPIDController()
        # Moving +X while the waypoint is +Y → command should bank toward +Y.
        q_ref = controller.attitude_reference_velocity_aim(
            position=np.zeros(3),
            velocity=np.array([10.0, 0.0, 0.0]),
            position_reference=np.array([0.0, 20.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),
        )
        body_z_world = rotate_vector(q_ref, np.array([0.0, 0.0, 1.0]))
        self.assertGreater(body_z_world[1], 0.05)
        self.assertGreater(body_z_world[2], 0.5)


class PositionCascadeTests(unittest.TestCase):
    def test_cascade_and_accel_ff_bank_toward_offset(self) -> None:
        from tvc.controller import PositionPIDConfig

        controller = PositionPIDController(
            PositionPIDConfig(
                kp=np.array([1.2, 1.2, 0.0]),
                kd=np.array([2.5, 2.5, 0.0]),
                ki=np.array([0.3, 0.3, 0.0]),
                max_tilt_deg=30.0,
                velocity_cascade=True,
            )
        )
        q_ref = controller.attitude_reference(
            position=np.zeros(3),
            velocity=np.zeros(3),
            position_reference=np.array([2.0, 0.0, 0.0]),
            velocity_reference=np.array([1.0, 0.0, 0.0]),
            accel_reference=np.array([0.5, 0.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),
            dt=0.01,
        )
        body_z_world = rotate_vector(q_ref, np.array([0.0, 0.0, 1.0]))
        self.assertGreater(body_z_world[0], 0.02)
        self.assertGreater(float(np.linalg.norm(controller.integral[:2])), 0.0)

    def test_integral_resets(self) -> None:
        from tvc.controller import PositionPIDConfig

        controller = PositionPIDController(
            PositionPIDConfig(ki=np.array([0.5, 0.5, 0.0]), velocity_cascade=True)
        )
        controller.attitude_reference(
            position=np.zeros(3),
            velocity=np.zeros(3),
            position_reference=np.array([1.0, 0.0, 0.0]),
            dt=0.1,
        )
        self.assertGreater(abs(float(controller.integral[0])), 0.0)
        controller.reset()
        np.testing.assert_allclose(controller.integral, [0.0, 0.0, 0.0])


class DynamicsTests(unittest.TestCase):
    def test_free_rotation_integrates_without_torque(self) -> None:
        vehicle = VehicleProperties(inertia=np.diag([0.05, 0.05, 0.08]))
        state = VehicleState(
            quaternion=quaternion_from_euler(0.0, 0.0, 0.0),
            omega=np.array([0.0, 0.0, 1.0]),
        )
        next_state = integrate_6dof_rk4(state, np.zeros(3), np.zeros(3), vehicle, 0.01)
        self.assertGreater(next_state.quaternion[0], 0.999)

    def test_lateral_thrust_produces_horizontal_motion(self) -> None:
        vehicle = VehicleProperties(mass=1.0, gravity=np.array([0.0, 0.0, -9.81]))
        state = VehicleState()
        force_body = np.array([6.0, 0.0, 20.0], dtype=float)
        next_state = integrate_6dof_rk4(state, force_body, np.zeros(3), vehicle, 0.05)
        self.assertGreater(next_state.velocity[0], 0.0)
        self.assertGreater(next_state.position[0], 0.0)

    def test_actuator_slew_limits_step_per_dt(self) -> None:
        current = np.zeros((2, 2), dtype=float)
        commanded = np.deg2rad(np.full((2, 2), 90.0))
        limited = apply_actuator_slew(current, commanded, 0.01, 400.0)
        np.testing.assert_allclose(np.rad2deg(limited), 4.0)
        unlimited = apply_actuator_slew(current, commanded, 0.01, None)
        np.testing.assert_allclose(unlimited, commanded)


class FlightSimTests(unittest.TestCase):
    def test_step_scenario_converges_toward_reference(self) -> None:
        system = build_default_config(count=3, radius=0.098, thrust=10.0)
        flight = FlightTestConfig(
            dt=0.02,
            duration=4.0,
            thrust_per_mount=10.0,
            controller=AttitudePIDConfig(
                kp=np.array([4.0, 4.0, 2.0]),
                ki=np.zeros(3),
                kd=np.array([0.8, 0.8, 0.4]),
                torque_limit=8.0,
            ),
        )
        simulator = FlightSimulator(system, flight)
        log = simulator.run(
            FlightScenario.step((15.0, 0.0, 0.0), step_time=0.5)
        )
        final = log.samples[-1]
        roll, _, _ = quaternion_to_euler(final.quaternion)
        roll_ref, _, _ = quaternion_to_euler(final.quaternion_reference)
        self.assertLess(abs(np.rad2deg(roll - roll_ref)), 5.0)

    def test_lateral_scenario_moves_in_xy_plane(self) -> None:
        system = build_default_config(count=3, radius=0.098, thrust=10.0)
        flight = FlightTestConfig(
            dt=0.02,
            duration=6.0,
            thrust_per_mount=10.0,
            controller=AttitudePIDConfig(
                kp=np.array([4.0, 4.0, 2.0]),
                ki=np.zeros(3),
                kd=np.array([0.8, 0.8, 0.4]),
                torque_limit=8.0,
            ),
        )
        simulator = FlightSimulator(system, flight)
        log = simulator.run(FlightScenario.lateral((0.0, 10.0, 0.0)))
        final = log.samples[-1]
        displacement = np.linalg.norm(final.position[:2])
        self.assertGreater(displacement, 0.2)

    def test_position_scenario_moves_toward_target(self) -> None:
        # Constant-thrust mounts with hover scaling (no motor curves).
        system = build_default_config(count=3, radius=0.098, thrust=10.0)
        flight = FlightTestConfig(
            dt=0.02,
            duration=12.0,
            controller=AttitudePIDConfig(
                kp=np.array([4.0, 4.0, 2.0]),
                ki=np.zeros(3),
                kd=np.array([0.8, 0.8, 0.4]),
                torque_limit=8.0,
            ),
        )
        simulator = FlightSimulator(system, flight)
        log = simulator.run(FlightScenario.position_step((1.5, 0.5, 0.0), step_time=0.5))
        final = log.samples[-1]
        error = np.linalg.norm(final.position_reference - final.position)
        # Simple collective+differential mixer is slightly less precise than LM.
        self.assertLess(error, 1.6)

    def test_hover_climbs_toward_ten_meters(self) -> None:
        system = load_vehicle_profile(profile_path="config/profiles/hover_10m.json")
        flight = system.flight_test
        assert flight is not None
        flight.dt = 0.02
        flight.duration = 8.0
        simulator = FlightSimulator(system, flight)
        log = simulator.run(FlightScenario.hover(10.0))
        final = log.samples[-1]
        self.assertGreater(final.position[2], 6.0)
        self.assertLess(abs(final.position[2] - 10.0), 3.0)
        self.assertLess(abs(final.velocity[2]), 2.5)

    def test_hover_waypoints_advance_setpoint(self) -> None:
        system = load_vehicle_profile(profile_path="config/profiles/hover_two_wp.json")
        flight = system.flight_test
        assert flight is not None
        flight.dt = 0.02
        flight.duration = 20.0
        simulator = FlightSimulator(system, flight)
        assert system.profile is not None
        waypoints = [(wp.t, wp.position_m) for wp in system.profile.waypoints]
        log = simulator.run(FlightScenario.hover_waypoints(waypoints))
        mid = next(sample for sample in log.samples if sample.time >= 6.0)
        np.testing.assert_allclose(mid.position_reference, [0.0, 0.0, 10.0], atol=1e-9)
        final = log.samples[-1]
        np.testing.assert_allclose(final.position_reference, [5.0, 5.0, 10.0], atol=1e-9)
        self.assertGreater(final.position[2], 6.0)
        self.assertGreater(np.linalg.norm(final.position[:2]), 1.0)

    def test_hover_waypoints_capture_advance(self) -> None:
        system = load_vehicle_profile(profile_path="config/profiles/hover_two_wp.json")
        flight = system.flight_test
        assert flight is not None
        flight.dt = 0.02
        flight.duration = 20.0
        simulator = FlightSimulator(system, flight)
        waypoints = [
            (0.0, (0.0, 0.0, 10.0)),
            (0.0, (5.0, 5.0, 10.0)),
        ]
        log = simulator.run(
            FlightScenario.hover_waypoints(
                waypoints,
                advance="capture",
                capture_radius_m=2.0,
            )
        )
        refs = [tuple(np.round(sample.position_reference, 6)) for sample in log.samples]
        self.assertIn((0.0, 0.0, 10.0), refs)
        self.assertIn((5.0, 5.0, 10.0), refs)
        switch = next(
            sample
            for sample in log.samples
            if np.allclose(sample.position_reference, [5.0, 5.0, 10.0])
        )
        # Capture must wait for proximity, not fire at a fixed mission time.
        self.assertGreater(switch.time, 0.5)
        self.assertLess(
            np.linalg.norm(switch.position - np.array([0.0, 0.0, 10.0])),
            2.0 + 1e-6,
        )
        final = log.samples[-1]
        np.testing.assert_allclose(final.position_reference, [5.0, 5.0, 10.0], atol=1e-9)

    def test_hover_waypoints_lerp_moves_continuously(self) -> None:
        system = load_vehicle_profile(profile_path="config/profiles/hover_moving_wp.json")
        flight = system.flight_test
        assert flight is not None
        flight.dt = 0.05
        flight.duration = 12.0
        simulator = FlightSimulator(system, flight)
        assert system.profile is not None
        waypoints = [(wp.t, wp.position_m) for wp in system.profile.waypoints]
        log = simulator.run(
            FlightScenario.hover_waypoints(waypoints, advance="lerp")
        )
        scenario = FlightScenario.hover_waypoints(waypoints, advance="lerp")
        mid = scenario.resolve_position_reference(8.75)
        # Between (0,0,8)@5.5 and (6,0,8)@12 → halfway at t=8.75 is x=3.
        np.testing.assert_allclose(mid, [3.0, 0.0, 8.0], atol=1e-9)
        v_ref = scenario.resolve_velocity_reference(8.75)
        np.testing.assert_allclose(v_ref, [6.0 / 6.5, 0.0, 0.0], atol=1e-9)
        refs = np.array([sample.position_reference for sample in log.samples])
        # Setpoint should sweep through intermediate X values, not jump.
        self.assertTrue(np.any((refs[:, 0] > 1.0) & (refs[:, 0] < 5.0)))
        final = log.samples[-1]
        self.assertLess(abs(final.position[2] - 8.0), 0.75)
        self.assertLess(abs(final.position[0] - final.position_reference[0]), 1.5)

    def test_flight_log_round_trip(self) -> None:
        sample = FlightLogSample(
            time=0.0,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
            quaternion_reference=np.array([1.0, 0.0, 0.0, 0.0]),
            torque_command=np.array([0.1, 0.0, 0.0]),
            torque_achieved=np.array([0.09, 0.0, 0.0]),
            gimbal_angles=np.zeros((3, 2)),
            saturated=False,
        )
        log = FlightLog(samples=[sample])
        path = self._tmp_path() / "flight.csv"
        log.save(path)
        restored = FlightLog.load(path)
        self.assertEqual(len(restored.samples), 1)
        self.assertEqual(restored.metadata.mount_count, 3)
        self.assertFalse(restored.samples[0].saturated)
        np.testing.assert_allclose(restored.samples[0].torque_command, [0.1, 0.0, 0.0])
        self.assertIn("samples=1", summarize_log(restored))

    def _tmp_path(self):
        from pathlib import Path
        import tempfile

        return Path(tempfile.mkdtemp())


if __name__ == "__main__":
    unittest.main()
