import unittest
from pathlib import Path

import numpy as np

from tvc.flight_log import FlightLog, FlightLogMetadata, FlightLogSample
from tvc.flight_sim import (
    FlightScenario,
    FlightSimulator,
    evaluate_landing_acceptance,
    find_engine_burnout_index,
    find_ground_contact_before_burnout,
    find_hard_landing,
    summarize_log,
)
from tvc.parameters import LandingAcceptanceBox, load_vehicle_profile


ROOT = Path(__file__).resolve().parents[1]


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


class LandingAcceptanceBoxTests(unittest.TestCase):
    def test_contains_position_axis_aligned(self) -> None:
        box = LandingAcceptanceBox(center_m=(10.0, 0.0, 2.0), half_extents_m=(5.0, 4.0, 3.0))
        self.assertTrue(box.contains_position((10.0, 0.0, 2.0)))
        self.assertTrue(box.contains_position((15.0, 4.0, 5.0)))
        self.assertFalse(box.contains_position((15.1, 0.0, 2.0)))
        self.assertFalse(box.contains_position((10.0, 0.0, 5.1)))

    def test_accepts_optional_speed_limit(self) -> None:
        box = LandingAcceptanceBox(
            center_m=(0.0, 0.0, 0.0),
            half_extents_m=(1.0, 1.0, 1.0),
            max_speed_m_s=2.0,
        )
        self.assertTrue(box.accepts((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertFalse(box.accepts((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)))

    def test_from_profile_config(self) -> None:
        system = load_vehicle_profile(
            profile_path=ROOT / "config/profiles/true_flight_landing_g8.json"
        )
        assert system.profile is not None
        box = system.profile.landing_acceptance
        assert box is not None
        np.testing.assert_allclose(box.center_m, [8.0, 0.0, 1.5])
        np.testing.assert_allclose(box.half_extents_m, [0.5, 0.5, 0.5])
        self.assertAlmostEqual(box.max_speed_m_s or 0.0, 5.0)

    def test_from_min_max_config(self) -> None:
        # min_m/max_m is an alternate authoring form for the same AABB.
        from tvc.parameters import _build_landing_acceptance

        box = _build_landing_acceptance(
            {
                "min_m": [3.0, -5.0, -1.0],
                "max_m": [13.0, 5.0, 5.0],
                "max_speed_m_s": 4.0,
            }
        )
        assert box is not None
        np.testing.assert_allclose(box.center_m, [8.0, 0.0, 2.0])
        np.testing.assert_allclose(box.half_extents_m, [5.0, 5.0, 3.0])
        self.assertAlmostEqual(box.max_speed_m_s or 0.0, 4.0)


class LandingEvaluationTests(unittest.TestCase):
    def test_find_burnout_after_lit(self) -> None:
        log = FlightLog(
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(1.0, (0, 0, 1), [5.0, 5.0, 5.0]),
                _sample(2.0, (0, 0, 2), [4.0, 4.0, 4.0]),
                _sample(3.0, (1, 0, 2), [0.0, 0.0, 0.0]),
                _sample(4.0, (1, 0, 1), [0.0, 0.0, 0.0]),
            ]
        )
        self.assertEqual(find_engine_burnout_index(log), 3)

    def test_success_when_inside_box_at_burnout(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(1.0, (5, 0, 5), [6.0, 6.0, 6.0]),
                _sample(2.0, (10, 0, 2), [0.1, 0.0, 0.0], velocity=(0.5, 0.0, -0.2)),
            ],
        )
        box = LandingAcceptanceBox(
            center_m=(10.0, 0.0, 2.0),
            half_extents_m=(2.0, 2.0, 2.0),
            max_speed_m_s=2.0,
        )
        result = evaluate_landing_acceptance(log, box)
        self.assertTrue(result["evaluated"])
        self.assertTrue(result["success"])
        self.assertEqual(log.metadata.landing_success, "true")
        self.assertAlmostEqual(log.metadata.burnout_time_s, 2.0)
        self.assertIn("SUCCESS", summarize_log(log))

    def test_fail_when_outside_box_at_burnout(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(1.0, (0, 0, 5), [6.0, 6.0, 6.0]),
                _sample(2.0, (50, 0, 2), [0.0, 0.0, 0.0]),
            ],
        )
        box = LandingAcceptanceBox(center_m=(0.0, 0.0, 2.0), half_extents_m=(5.0, 5.0, 3.0))
        result = evaluate_landing_acceptance(log, box)
        self.assertTrue(result["evaluated"])
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "outside_box")
        self.assertEqual(log.metadata.landing_success, "false")
        self.assertIn("FAIL", summarize_log(log))

    def test_fail_when_no_burnout_observed(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [5.0, 5.0, 5.0]),
                _sample(1.0, (0, 0, 1), [5.0, 5.0, 5.0]),
            ],
        )
        box = LandingAcceptanceBox(center_m=(0.0, 0.0, 0.0), half_extents_m=(10.0, 10.0, 10.0))
        result = evaluate_landing_acceptance(log, box)
        self.assertTrue(result["evaluated"])
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "no_engine_burnout")

    def test_hard_landing_detects_ground_plane_clamp_slam(self) -> None:
        """Soft floor zeros vz on contact; impact speed lives on the prior sample."""
        log = FlightLog(
            samples=[
                _sample(0.0, (0, 0, 5), [0.0, 0.0, 0.0]),
                _sample(1.0, (0, 0, 3), [10.0, 10.0, 10.0], velocity=(0.0, 0.0, -8.0)),
                # Airborne, still slamming toward the pad.
                _sample(2.00, (0, 0, 0.15), [10.0, 10.0, 10.0], velocity=(0.0, 0.0, -14.7)),
                # Ground plane has clamped z and wiped downward speed.
                _sample(2.01, (0, 0, 0.0), [10.0, 10.0, 10.0], velocity=(0.0, 0.0, 0.0)),
                _sample(3.0, (0, 0, 0.5), [0.0, 0.0, 0.0], velocity=(0.0, 0.0, 0.0)),
            ]
        )
        hard = find_hard_landing(log)
        self.assertIsNotNone(hard)
        assert hard is not None
        self.assertEqual(hard[0], 3)
        self.assertAlmostEqual(hard[2], -14.7, places=5)

        box = LandingAcceptanceBox(
            center_m=(0.0, 0.0, 0.5),
            half_extents_m=(2.0, 2.0, 1.0),
            max_speed_m_s=5.0,
        )
        result = evaluate_landing_acceptance(log, box)
        self.assertTrue(result["evaluated"])
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "hard_landing")

    def test_soft_ground_touch_is_not_hard_landing(self) -> None:
        log = FlightLog(
            samples=[
                _sample(0.0, (0, 0, 2), [5.0, 5.0, 5.0], velocity=(0.0, 0.0, -0.5)),
                _sample(1.0, (0, 0, 0.12), [5.0, 5.0, 5.0], velocity=(0.0, 0.0, -0.8)),
                _sample(1.01, (0, 0, 0.0), [5.0, 5.0, 5.0], velocity=(0.0, 0.0, 0.0)),
                _sample(2.0, (0, 0, 0.5), [0.0, 0.0, 0.0], velocity=(0.0, 0.0, 0.0)),
            ]
        )
        self.assertIsNone(find_hard_landing(log))

    def test_ground_contact_after_liftoff_fails_before_burnout(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(0.5, (0, 0, 1.0), [8.0, 8.0, 8.0]),
                _sample(1.0, (0, 0, 0.0), [8.0, 8.0, 8.0]),
                _sample(2.0, (0, 0, 0.4), [0.0, 0.0, 0.0]),
            ],
        )
        early = find_ground_contact_before_burnout(log, 3)
        self.assertIsNotNone(early)
        assert early is not None
        self.assertEqual(early[0], 2)
        box = LandingAcceptanceBox(
            center_m=(0.0, 0.0, 0.4),
            half_extents_m=(1.0, 1.0, 0.5),
            max_speed_m_s=3.0,
        )
        result = evaluate_landing_acceptance(log, box)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "ground_contact")

    def test_launch_on_pad_then_airborne_to_burnout_succeeds(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(0.2, (0, 0, 0), [8.0, 8.0, 8.0]),
                _sample(1.0, (0, 0, 1.2), [8.0, 8.0, 8.0]),
                _sample(2.0, (0, 0, 0.4), [0.0, 0.0, 0.0]),
            ],
        )
        self.assertIsNone(find_ground_contact_before_burnout(log, 3))
        box = LandingAcceptanceBox(
            center_m=(0.0, 0.0, 0.4),
            half_extents_m=(1.0, 1.0, 0.5),
        )
        result = evaluate_landing_acceptance(log, box)
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "inside_box")

    def test_never_leaving_the_pad_fails(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(1.0, (2, 0, 0), [8.0, 8.0, 8.0]),
                _sample(2.0, (0, 0, 0), [0.0, 0.0, 0.0]),
            ],
        )
        box = LandingAcceptanceBox(
            center_m=(0.0, 0.0, 0.4),
            half_extents_m=(1.0, 1.0, 0.5),
        )
        result = evaluate_landing_acceptance(log, box)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "ground_contact")

    def test_metadata_round_trip_preserves_box(self) -> None:
        log = FlightLog(
            metadata=FlightLogMetadata(scenario="test"),
            samples=[
                _sample(0.0, (0, 0, 0), [0.0, 0.0, 0.0]),
                _sample(1.0, (0, 0, 1), [5.0, 5.0, 5.0]),
                _sample(2.0, (1, 0, 1), [0.0, 0.0, 0.0]),
            ],
        )
        box = LandingAcceptanceBox(center_m=(1.0, 2.0, 3.0), half_extents_m=(4.0, 5.0, 6.0))
        evaluate_landing_acceptance(log, box)
        path = Path(self._tmp()) / "landing.csv"
        log.save(path)
        restored = FlightLog.load(path)
        self.assertEqual(restored.metadata.landing_success, "true")
        restored_box = LandingAcceptanceBox.from_metadata(restored.metadata)
        assert restored_box is not None
        np.testing.assert_allclose(restored_box.center_m, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(restored_box.half_extents_m, [4.0, 5.0, 6.0])

    def test_true_flight_profile_runs_and_evaluates(self) -> None:
        system = load_vehicle_profile(
            profile_path=ROOT / "config/profiles/true_flight_landing_g8.json"
        )
        flight = system.flight_test
        assert flight is not None
        # Coarser dt for a fast unit test; still covers G8 burnout (~20 s).
        flight.dt = 0.05
        flight.duration = 22.0
        simulator = FlightSimulator(system, flight)
        assert system.profile is not None
        waypoints = [(wp.t, wp.position_m) for wp in system.profile.waypoints]
        log = simulator.run(
            FlightScenario.hover_waypoints(
                waypoints,
                advance=system.profile.waypoint_advance,
                capture_radius_m=system.profile.waypoint_capture_radius_m,
            )
        )
        # Loaded ~3.2 kg exceeds 3×G8 average thrust (T/W≈0.63), so the
        # acceptance evaluator must still run and report failure.
        self.assertEqual(log.metadata.landing_success, "false")
        self.assertGreater(log.metadata.burnout_time_s, 15.0)
        self.assertTrue(log.metadata.landing_box_center_m)
        summary = summarize_log(log)
        self.assertIn("landing=FAIL", summary)

    def _tmp(self) -> str:
        import tempfile

        return tempfile.mkdtemp()


if __name__ == "__main__":
    unittest.main()
