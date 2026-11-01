import unittest
from pathlib import Path

import numpy as np

from tvc.attitude import quaternion_from_euler
from tvc.geometry import motor_aft_position, motor_radius_m
from tvc.chopsticks import (
    LandingChopsticks,
    apply_chopstick_contacts,
    catch_heading_aligned,
    chopsticks_for_pad,
    closest_points_on_segments,
    in_chopstick_catch_volume,
    stick_geometry_faults,
)
from tvc.dynamics import VehicleProperties, VehicleState
from tvc.flight_log import FlightLog, FlightLogMetadata, FlightLogSample
from tvc.flight_sim import evaluate_landing_acceptance
from tvc.geometry import CatchArm
from tvc.parameters import LandingAcceptanceBox


def _arms() -> list[CatchArm]:
    root = np.array([0.0, 0.0, 1.3081], dtype=float)
    return [
        CatchArm(position=root.copy(), direction=np.array([1.0, 0.0, 0.0]), length_m=0.5),
        CatchArm(position=root.copy(), direction=np.array([-1.0, 0.0, 0.0]), length_m=0.5),
    ]


def _sample(
    time: float,
    position: tuple[float, float, float],
    thrusts: list[float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FlightLogSample:
    return FlightLogSample(
        time=time,
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        position_reference=np.asarray(position, dtype=float),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        omega=np.zeros(3, dtype=float),
        quaternion_reference=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        torque_command=np.zeros(3, dtype=float),
        torque_achieved=np.zeros(3, dtype=float),
        force_command=np.zeros(3, dtype=float),
        force_body=np.zeros(3, dtype=float),
        force_inertial=np.zeros(3, dtype=float),
        gimbal_angles=np.zeros((3, 2), dtype=float),
        mount_thrusts=np.asarray(thrusts, dtype=float),
        saturated=False,
    )


class ChopstickGeometryTests(unittest.TestCase):
    def test_bars_are_x_parallel_and_separated_in_y(self) -> None:
        sticks = chopsticks_for_pad((0.0, 0.0), _arms())
        assert sticks is not None
        self.assertAlmostEqual(sticks.length_m, 0.75)
        self.assertAlmostEqual(sticks.separation_m, 0.6)
        self.assertAlmostEqual(sticks.height_m, 1.3081)
        (a0, a1), (b0, b1) = sticks.segments()
        np.testing.assert_allclose(a1 - a0, [0.75, 0.0, 0.0])
        np.testing.assert_allclose(b1 - b0, [0.75, 0.0, 0.0])
        self.assertAlmostEqual(float(a0[1]), -0.3)
        self.assertAlmostEqual(float(b0[1]), 0.3)

    def test_closest_points_on_crossing_segments(self) -> None:
        pa, pb, dist = closest_points_on_segments(
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.1]),
            np.array([0.0, 1.0, 0.1]),
        )
        np.testing.assert_allclose(pa, [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(pb, [0.0, 0.0, 0.1], atol=1e-9)
        self.assertAlmostEqual(dist, 0.1, places=9)


class CatchHeadingTests(unittest.TestCase):
    def test_yaw_90_is_aligned(self) -> None:
        q = quaternion_from_euler(0.0, 0.0, 0.5 * np.pi)
        self.assertTrue(catch_heading_aligned(q))

    def test_yaw_minus_90_is_aligned(self) -> None:
        q = quaternion_from_euler(0.0, 0.0, -0.5 * np.pi)
        self.assertTrue(catch_heading_aligned(q))

    def test_yaw_0_and_180_are_not_aligned(self) -> None:
        self.assertFalse(catch_heading_aligned(quaternion_from_euler(0.0, 0.0, 0.0)))
        self.assertFalse(catch_heading_aligned(quaternion_from_euler(0.0, 0.0, np.pi)))

    def test_on_pad_at_stick_height_is_a_catch(self) -> None:
        sticks = LandingChopsticks(center_xy=(0.0, 0.0), height_m=1.3081)
        state = VehicleState(
            position=np.array([-0.14, -0.02, 0.08], dtype=float),
            quaternion=quaternion_from_euler(0.0, 0.0, 0.0),
        )
        self.assertTrue(in_chopstick_catch_volume(state, sticks))

    def test_south_of_the_bars_is_not_between_the_sticks(self) -> None:
        # G12 zero-scatter freeze: body past the south bar (y=-0.30), north
        # arm short of the north bar, arms still ~10 cm above the sticks.
        sticks = LandingChopsticks(center_xy=(0.0, 0.0), height_m=1.3081)
        q = quaternion_from_euler(
            np.deg2rad(6.39), np.deg2rad(-1.60), np.deg2rad(89.60)
        )
        state = VehicleState(
            position=np.array([0.108, -0.351, 0.106], dtype=float),
            velocity=np.array([0.0, 0.45, -0.7], dtype=float),
            quaternion=q,
            omega=np.zeros(3, dtype=float),
        )
        self.assertFalse(in_chopstick_catch_volume(state, sticks))
        vehicle = VehicleProperties(
            mass=2.65,
            inertia=np.diag([0.05, 0.05, 0.02]),
            gravity=np.array([0.0, 0.0, -9.81], dtype=float),
        )
        _state, n_hit, _heading = apply_chopstick_contacts(
            state, vehicle, _arms(), sticks, dt=0.01
        )
        self.assertLess(n_hit, 2)

    def test_six_centimetres_above_the_bar_is_not_a_catch(self) -> None:
        # Level yaw 90. Body z equals the arm's height above the bar center.
        # The cylinder surface is one radius (3 cm) up. 6 cm is clear air.
        sticks = LandingChopsticks(center_xy=(0.0, 0.0), height_m=1.3081)
        q = quaternion_from_euler(0.0, 0.0, 0.5 * np.pi)
        vehicle = VehicleProperties(
            mass=2.65,
            inertia=np.diag([0.05, 0.05, 0.02]),
            gravity=np.array([0.0, 0.0, -9.81], dtype=float),
        )
        above = VehicleState(
            position=np.array([0.0, 0.0, 0.06], dtype=float),
            velocity=np.array([0.0, 0.0, -0.3], dtype=float),
            quaternion=q,
            omega=np.zeros(3, dtype=float),
        )
        _state, n_hit, _heading = apply_chopstick_contacts(
            above, vehicle, _arms(), sticks, dt=0.01
        )
        self.assertLess(n_hit, 2)
        on = VehicleState(
            position=np.array([0.0, 0.0, 0.02], dtype=float),
            velocity=np.array([0.0, 0.0, -0.3], dtype=float),
            quaternion=q,
            omega=np.zeros(3, dtype=float),
        )
        _state, n_hit, _heading = apply_chopstick_contacts(
            on, vehicle, _arms(), sticks, dt=0.01
        )
        self.assertGreaterEqual(n_hit, 2)

    def test_launch_offset_is_not_a_catch(self) -> None:
        sticks = LandingChopsticks(center_xy=(0.0, 0.0), height_m=1.3081)
        state = VehicleState(
            position=np.array([8.0, 0.0, 0.0], dtype=float),
            quaternion=quaternion_from_euler(0.0, 0.0, 0.0),
        )
        self.assertFalse(in_chopstick_catch_volume(state, sticks))


def _drop(
    yaw_rad: float,
    *,
    steps: int = 250,
    start_z: float = 0.35,
) -> tuple[VehicleState, int, bool]:
    q = quaternion_from_euler(0.0, 0.0, yaw_rad)
    arms = _arms()
    sticks = LandingChopsticks(center_xy=(0.0, 0.0), height_m=1.3081)
    vehicle = VehicleProperties(
        mass=2.5,
        inertia=np.diag([0.05, 0.05, 0.02]),
        gravity=np.array([0.0, 0.0, -9.81], dtype=float),
    )
    state = VehicleState(
        position=np.array([0.0, 0.0, start_z], dtype=float),
        velocity=np.zeros(3, dtype=float),
        quaternion=q,
        omega=np.zeros(3, dtype=float),
    )
    dt = 0.01
    max_hit = 0
    heading_ok = False
    for _ in range(steps):
        state.velocity = state.velocity + vehicle.gravity * dt
        state.position = state.position + state.velocity * dt
        state, n_hit, heading_ok = apply_chopstick_contacts(
            state, vehicle, arms, sticks, dt=dt
        )
        max_hit = max(max_hit, n_hit)
    return state, max_hit, heading_ok


class ChopstickCatchTests(unittest.TestCase):
    def test_yaw_90_drop_catches_both_sticks(self) -> None:
        state, max_hit, heading_ok = _drop(0.5 * np.pi)
        self.assertTrue(heading_ok)
        self.assertGreaterEqual(max_hit, 2)
        self.assertLess(float(state.position[2]), 0.12)
        self.assertGreater(float(state.position[2]), -0.05)
        self.assertLess(abs(float(state.velocity[2])), 0.05)
        self.assertLess(float(np.linalg.norm(state.velocity)), 0.05)
        self.assertLess(float(np.linalg.norm(state.position[:2])), 0.25)

    def test_two_bar_contact_zeros_velocity(self) -> None:
        q = quaternion_from_euler(0.0, 0.0, 0.5 * np.pi)
        arms = _arms()
        sticks = LandingChopsticks(center_xy=(0.0, 0.0), height_m=1.3081)
        vehicle = VehicleProperties(
            mass=2.5,
            inertia=np.diag([0.05, 0.05, 0.02]),
            gravity=np.array([0.0, 0.0, -9.81], dtype=float),
        )
        state = VehicleState(
            position=np.array([0.0, 0.0, 0.02], dtype=float),
            velocity=np.array([0.4, -0.3, -2.5], dtype=float),
            quaternion=q,
            omega=np.array([0.2, -0.1, 0.5], dtype=float),
        )
        state, n_hit, heading_ok = apply_chopstick_contacts(
            state, vehicle, arms, sticks, dt=0.01
        )
        self.assertTrue(heading_ok)
        self.assertGreaterEqual(n_hit, 2)
        np.testing.assert_allclose(state.velocity, 0.0, atol=1e-9)
        np.testing.assert_allclose(state.omega, 0.0, atol=1e-9)

    def test_yaw_0_drop_misses_sticks(self) -> None:
        state, max_hit, heading_ok = _drop(0.0)
        self.assertFalse(heading_ok)
        self.assertLess(max_hit, 2)
        self.assertLess(float(state.position[2]), -1.0)


class ChopstickAcceptanceTests(unittest.TestCase):
    def _box_log(self) -> tuple[FlightLog, LandingAcceptanceBox]:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(1.0, (0, 0, 2), [6.0, 6.0, 6.0]),
                _sample(2.0, (0.1, 0.0, 0.8), [0.0, 0.0, 0.0], velocity=(0.2, 0.0, -0.3)),
            ],
        )
        box = LandingAcceptanceBox(
            center_m=(0.0, 0.0, 0.7),
            half_extents_m=(0.5, 0.5, 0.65),
            max_speed_m_s=3.2,
        )
        return log, box

    def test_inside_box_without_chopsticks_still_succeeds(self) -> None:
        log, box = self._box_log()
        result = evaluate_landing_acceptance(log, box)
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "inside_box")

    def test_missed_chopsticks_fails_when_heading_ok(self) -> None:
        log, box = self._box_log()
        result = evaluate_landing_acceptance(
            log,
            box,
            chopsticks_enabled=True,
            chopsticks_caught=False,
            chopsticks_heading_ok=True,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "missed_chopsticks")
        self.assertEqual(log.metadata.chopstick_caught, "false")

    def test_bad_heading_fails_when_not_caught(self) -> None:
        log, box = self._box_log()
        result = evaluate_landing_acceptance(
            log,
            box,
            chopsticks_enabled=True,
            chopsticks_caught=False,
            chopsticks_heading_ok=False,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "bad_heading")

    def test_hard_landing_fails_even_when_caught(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(1.0, (0, 0, 2), [6.0, 6.0, 6.0], velocity=(0.0, 0.0, -1.0)),
                _sample(2.0, (0.05, 0.0, 0.04), [6.0, 6.0, 6.0], velocity=(0.0, 0.0, -2.8)),
                _sample(3.0, (0.05, 0.0, 0.04), [0.0, 0.0, 0.0], velocity=(0.0, 0.0, 0.0)),
            ],
        )
        box = LandingAcceptanceBox(
            center_m=(0.0, 0.0, 0.7),
            half_extents_m=(0.5, 0.5, 0.65),
            max_speed_m_s=3.2,
        )
        result = evaluate_landing_acceptance(
            log,
            box,
            chopsticks_enabled=True,
            chopsticks_caught=True,
            chopsticks_heading_ok=True,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "hard_landing")

    def test_ground_scrape_fails_even_when_caught(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(1.0, (0, 0, 2), [6.0, 6.0, 6.0], velocity=(0.0, 0.0, 1.0)),
                # Soft-plane scrape far from pad (vz already clamped).
                _sample(2.0, (2.2, 0.0, 0.0), [6.0, 6.0, 6.0], velocity=(1.0, 0.0, 0.0)),
                _sample(3.0, (0.05, 0.0, 0.05), [0.0, 0.0, 0.0], velocity=(0.0, 0.0, 0.0)),
            ],
        )
        box = LandingAcceptanceBox(
            center_m=(0.0, 0.0, 0.7),
            half_extents_m=(0.5, 0.5, 0.65),
            max_speed_m_s=3.2,
        )
        result = evaluate_landing_acceptance(
            log,
            box,
            chopsticks_enabled=True,
            chopsticks_caught=True,
            chopsticks_heading_ok=True,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "ground_contact")

    def test_catch_after_box_succeeds(self) -> None:
        log, box = self._box_log()
        result = evaluate_landing_acceptance(
            log,
            box,
            chopsticks_enabled=True,
            chopsticks_caught=True,
            chopsticks_heading_ok=True,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "caught")
        self.assertEqual(log.metadata.chopstick_caught, "true")
        self.assertEqual(log.metadata.chopstick_heading_ok, "true")

    def test_backstop_fails_even_when_caught(self) -> None:
        log, box = self._box_log()
        result = evaluate_landing_acceptance(
            log,
            box,
            chopsticks_enabled=True,
            chopsticks_caught=True,
            chopsticks_heading_ok=True,
            stick_backstop=True,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "backstop")

    def test_body_on_stick_fails_even_when_caught(self) -> None:
        log, box = self._box_log()
        result = evaluate_landing_acceptance(
            log,
            box,
            chopsticks_enabled=True,
            chopsticks_caught=True,
            chopsticks_heading_ok=True,
            body_stick_contact=True,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "body_contact")


class StickGeometryFaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sticks = chopsticks_for_pad((0.0, 0.0), _arms())
        self.mounts = [
            _Mount(np.array([0.098, 0.0, 0.0])),
            _Mount(np.array([-0.049, 0.085, 0.0])),
            _Mount(np.array([-0.049, -0.085, 0.0])),
        ]
        self.cgs = np.array(
            [
                [0.098, 0.0, -0.08],
                [-0.049, 0.085, -0.08],
                [-0.049, -0.085, -0.08],
            ],
            dtype=float,
        )

    def test_arms_on_bars_are_not_body_contact(self) -> None:
        # Yaw 90 puts the arms across the bars. The mast stays between them.
        q = quaternion_from_euler(0.0, 0.0, np.pi / 2.0)
        faults = stick_geometry_faults(
            np.array([0.0, 0.0, 0.03]),
            q,
            _arms(),
            self.mounts,
            self.sticks,
            motor_cgs_body=self.cgs,
        )
        self.assertEqual(faults, set())

    def test_point_past_backstop_fails(self) -> None:
        q = quaternion_from_euler(0.0, 0.0, np.pi / 2.0)
        faults = stick_geometry_faults(
            np.array([-0.60, 0.0, 0.03]),
            q,
            _arms(),
            self.mounts,
            self.sticks,
            motor_cgs_body=self.cgs,
        )
        self.assertIn("backstop", faults)

    def test_arm_tip_past_backstop_fails(self) -> None:
        # Yaw 0 leaves the arms along world X, so the west tip crosses x=-0.5
        # while the body origin is still east of the plane.
        q = quaternion_from_euler(0.0, 0.0, 0.0)
        faults = stick_geometry_faults(
            np.array([-0.05, 0.0, 1.0]),
            q,
            _arms(),
            self.mounts,
            self.sticks,
            motor_cgs_body=self.cgs,
        )
        self.assertIn("backstop", faults)
        self.assertNotIn("body_contact", faults)

    def test_motor_case_against_a_bar_is_body_contact(self) -> None:
        # G12 rest thrust is +Z. The case hangs 55 mm + 156 mm below the pivot.
        # Four centimetres off a bar centerline misses the bar and hits the
        # 29 mm case. Twelve centimetres clears the case.
        from tvc.parameters import load_config

        system = load_config(
            Path(__file__).resolve().parents[1] / "config/vehicles/tvc_ring3_g12.json"
        )
        mount = system.mounts[0]
        rest = np.asarray(mount.rest_angles, dtype=float)
        standoff = float(system.motor_pivot_standoff_m)
        aft = motor_aft_position(mount, float(rest[0]), float(rest[1]), standoff_m=standoff)
        reach = standoff + 0.156
        np.testing.assert_allclose(
            aft,
            np.asarray(mount.position, dtype=float) - np.array([0.0, 0.0, reach]),
            atol=1e-6,
        )
        self.assertAlmostEqual(motor_radius_m(mount), 0.0145, places=6)
        q = quaternion_from_euler(0.0, 0.0, 0.0)
        angles = rest.reshape(1, 2)
        bar_z = 1.3081
        hit = stick_geometry_faults(
            np.array([0.0, -0.26, bar_z]) - aft,
            q,
            _arms(),
            [mount],
            self.sticks,
            gimbal_angles=angles,
            standoff_m=standoff,
        )
        self.assertIn("body_contact", hit)
        clear = stick_geometry_faults(
            np.array([0.0, -0.18, bar_z]) - aft,
            q,
            _arms(),
            [mount],
            self.sticks,
            gimbal_angles=angles,
            standoff_m=standoff,
        )
        self.assertNotIn("body_contact", clear)

    def test_mast_against_a_bar_is_body_contact(self) -> None:
        q = quaternion_from_euler(0.0, 0.0, np.pi / 2.0)
        faults = stick_geometry_faults(
            np.array([0.0, -0.30, 0.03]),
            q,
            _arms(),
            self.mounts,
            self.sticks,
            motor_cgs_body=self.cgs,
        )
        self.assertIn("body_contact", faults)
        self.assertNotIn("backstop", faults)


class _Mount:
    def __init__(self, position: np.ndarray) -> None:
        self.position = position
