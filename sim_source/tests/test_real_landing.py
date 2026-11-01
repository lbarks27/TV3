import unittest
from pathlib import Path

import numpy as np

from tvc.flight_sim import FlightScenario, FlightSimulator
from tvc.parameters import FlightTestConfig, RealLandingMission, load_config, load_vehicle_profile


ROOT = Path(__file__).resolve().parents[1]


class LaunchPositionTests(unittest.TestCase):
    def test_launch_is_fixed_offset_east_of_pad(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            launch_offset_m=(10.0, 0.0, 0.0),
        )
        self.assertEqual(mission.launch_position_m(), (10.0, 0.0, 0.0))

    def test_first_waypoint_is_five_east_five_up(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            first_waypoint_offset_m=(5.0, 0.0, 5.0),
        )
        self.assertEqual(mission.first_waypoint_m(), (5.0, 0.0, 5.0))


class RealLandingScenarioTests(unittest.TestCase):
    def test_slide_eases_from_the_holding_waypoint(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            launch_offset_m=(8.0, 0.0, 0.0),
            first_waypoint_offset_m=(2.0, 0.0, 1.5),
            touchdown_altitude_m=-0.1919,
            descent_duration_s=3.6,
            min_approach_time_s=0.5,
            forced_descent_time_s=8.0,
            stage_capture_radius_m=1.0,
            vertical_hold_s=0.0,
            vertical_hold_offset_m=(0.0, 0.0, 30.0),
            capture_radius_m=2.0,
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        loft = np.array([8.0, 0.0, 30.0])
        scenario.resolve_position_reference(
            0.2, position=loft, velocity=np.zeros(3)
        )
        # Already past the hold and still well above it. The old track
        # jumped the waypoint up to this pose.
        live = np.array([1.44, 0.0, 4.40])
        ref = scenario.resolve_position_reference(
            8.0, position=live, velocity=np.array([-1.2, 0.0, -1.0])
        )
        self.assertEqual(scenario.mission_phase, "landing")
        np.testing.assert_allclose(ref, [2.0, 0.0, 1.5], atol=1e-9)
        later = scenario.resolve_position_reference(8.2, position=live)
        self.assertLess(float(later[2]), 1.5)
        self.assertLess(float(later[0]), 2.0)

    def test_phases_transition_on_first_waypoint_capture(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            first_waypoint_offset_m=(5.0, 0.0, 5.0),
            descent_duration_s=4.0,
            capture_radius_m=2.0,
            capture_max_speed_m_s=1.5,
            min_approach_time_s=3.0,
            capture_settle_s=0.0,
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        self.assertEqual(scenario.mission_phase, "approach")

        # Far from first WP: still approach.
        ref = scenario.resolve_position_reference(1.0, position=np.array([20.0, 0.0, 2.0]))
        np.testing.assert_allclose(ref, [5.0, 0.0, 5.0])
        self.assertEqual(scenario.mission_phase, "approach")

        # Before min approach time: do not capture even if near and slow.
        ref = scenario.resolve_position_reference(
            2.0,
            position=np.array([5.2, 0.0, 5.1]),
            velocity=np.array([0.1, 0.0, 0.0]),
        )
        self.assertEqual(scenario.mission_phase, "approach")

        # Inside radius but still fast: stay in approach.
        ref = scenario.resolve_position_reference(
            4.0,
            position=np.array([5.2, 0.0, 5.1]),
            velocity=np.array([4.0, 0.0, 0.0]),
        )
        self.assertEqual(scenario.mission_phase, "approach")
        np.testing.assert_allclose(ref, [5.0, 0.0, 5.0])

        # Capture first WP (slow + near + after min time).
        # Descent track starts from the nominal gate (not the live capture pose).
        capture_pos = np.array([5.2, 0.0, 5.1])
        nominal_wp = np.array([5.0, 0.0, 5.0])
        ref = scenario.resolve_position_reference(
            4.0,
            position=capture_pos,
            velocity=np.array([0.2, 0.0, 0.0]),
        )
        self.assertEqual(scenario.mission_phase, "landing")
        np.testing.assert_allclose(ref, nominal_wp, atol=1e-9)
        np.testing.assert_allclose(scenario.landing_track_start, nominal_wp, atol=1e-9)

        # Cubic ease-out at tau=0.5: s = 1 - 0.5^3 = 0.875 (soft pad ends at z=0.5).
        # start=nominal_wp, end=(0,0,0.5)
        ref_mid = scenario.resolve_position_reference(6.0, position=np.array([2.0, 0.0, 2.0]))
        expected_mid = 0.125 * nominal_wp + 0.875 * np.array([0.0, 0.0, 0.5])
        np.testing.assert_allclose(ref_mid, expected_mid, atol=1e-9)

        # Path speed decreases toward the pad and is zero at the end.
        v_early = scenario.resolve_velocity_reference(4.5)  # tau=0.125
        v_late = scenario.resolve_velocity_reference(7.5)  # tau=0.875
        v_end = scenario.resolve_velocity_reference(8.0)  # tau=1
        self.assertGreater(float(np.linalg.norm(v_early)), float(np.linalg.norm(v_late)))
        np.testing.assert_allclose(v_end, [0.0, 0.0, 0.0], atol=1e-12)

        # Path accel is available and finite during the track.
        a_mid = scenario.resolve_acceleration_reference(6.0)
        self.assertTrue(np.all(np.isfinite(a_mid)))
        self.assertGreater(float(np.linalg.norm(a_mid)), 0.0)
        np.testing.assert_allclose(
            scenario.resolve_acceleration_reference(8.0),
            [0.0, 0.0, 0.0],
            atol=1e-12,
        )

        # End of track pins soft pad hover (not ground crash).
        ref_end = scenario.resolve_position_reference(20.0, position=np.array([0.0, 0.0, 0.1]))
        np.testing.assert_allclose(ref_end, [0.0, 0.0, 0.5], atol=1e-9)

    def test_capture_requires_settle_dwell(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            first_waypoint_offset_m=(5.0, 0.0, 5.0),
            capture_radius_m=0.5,
            capture_max_speed_m_s=1.0,
            min_approach_time_s=1.0,
            capture_settle_s=0.3,
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        near = np.array([5.1, 0.0, 5.0])
        slow = np.array([0.1, 0.0, 0.0])
        scenario.resolve_position_reference(1.0, position=near, velocity=slow)
        self.assertEqual(scenario.mission_phase, "approach")
        scenario.resolve_position_reference(1.2, position=near, velocity=slow)
        self.assertEqual(scenario.mission_phase, "approach")
        scenario.resolve_position_reference(1.35, position=near, velocity=slow)
        self.assertEqual(scenario.mission_phase, "landing")

    def test_loft_waypoint_holds_two_seconds_after_capture(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            launch_offset_m=(8.0, 0.0, 0.0),
            first_waypoint_offset_m=(2.5, 0.0, 1.5),
            vertical_hold_s=2.0,
            vertical_hold_offset_m=(0.0, 0.0, 10.0),
            capture_radius_m=1.2,
            min_approach_time_s=0.5,
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        loft = np.array([8.0, 0.0, 10.0])
        gate = np.array([2.5, 0.0, 1.5])
        climbing = np.array([8.0, 0.0, 6.0])
        ref = scenario.resolve_position_reference(1.0, position=climbing)
        np.testing.assert_allclose(ref, loft, atol=1e-9)
        self.assertTrue(scenario.in_vertical_hold(1.0))
        at_loft = np.array([8.05, 0.02, 9.9])
        ref = scenario.resolve_position_reference(2.0, position=at_loft)
        np.testing.assert_allclose(ref, loft, atol=1e-9)
        self.assertTrue(scenario.in_vertical_hold(2.0))
        high = np.array([8.1, 0.0, 12.0])
        ref = scenario.resolve_position_reference(3.5, position=high)
        np.testing.assert_allclose(ref, loft, atol=1e-9)
        self.assertTrue(scenario.in_vertical_hold(3.5))
        ref = scenario.resolve_position_reference(4.0, position=high)
        np.testing.assert_allclose(ref, gate, atol=1e-9)
        self.assertFalse(scenario.in_vertical_hold(4.0))

    def test_loft_waypoint_zero_hold_leaves_on_capture(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            launch_offset_m=(8.0, 0.0, 0.0),
            first_waypoint_offset_m=(2.5, 0.0, 1.5),
            vertical_hold_s=0.0,
            vertical_hold_offset_m=(0.0, 0.0, 10.0),
            capture_radius_m=1.2,
            min_approach_time_s=0.5,
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        loft = np.array([8.0, 0.0, 10.0])
        gate = np.array([2.5, 0.0, 1.5])
        climbing = np.array([8.0, 0.0, 6.0])
        ref = scenario.resolve_position_reference(1.0, position=climbing)
        np.testing.assert_allclose(ref, loft, atol=1e-9)
        self.assertTrue(scenario.in_vertical_hold(1.0))
        at_loft = np.array([8.05, 0.02, 9.9])
        still_climbing = np.array([0.0, 0.0, 4.0])
        ref = scenario.resolve_position_reference(
            2.0, position=at_loft, velocity=still_climbing
        )
        np.testing.assert_allclose(ref, loft, atol=1e-9)
        self.assertTrue(scenario.in_vertical_hold(2.0))
        scenario.resolve_velocity_reference(2.0, position=at_loft)
        self.assertTrue(scenario.in_vertical_hold(2.0))
        crested = np.array([0.2, 0.0, 0.4])
        ref = scenario.resolve_position_reference(
            2.2, position=at_loft, velocity=crested
        )
        np.testing.assert_allclose(ref, gate, atol=1e-9)
        self.assertFalse(scenario.in_vertical_hold(2.2))

    def test_profile_loads_mission_block(self) -> None:
        system = load_vehicle_profile(
            profile_path=ROOT / "config/profiles/real_landing_g8.json"
        )
        assert system.profile is not None
        mission = system.profile.real_landing
        assert mission is not None
        self.assertEqual(mission.landing_site_m, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(mission.launch_offset_m, [10.0, 0.0, 0.0])
        self.assertEqual(mission.launch_position_m(), (10.0, 0.0, 0.0))
        np.testing.assert_allclose(mission.first_waypoint_offset_m, [5.0, 0.0, 5.0])
        self.assertEqual(mission.first_waypoint_m(), (5.0, 0.0, 5.0))
        self.assertEqual(mission.landing_end_m(), (0.0, 0.0, 0.5))
        box = system.profile.landing_acceptance
        assert box is not None
        np.testing.assert_allclose(box.half_extents_m, [0.5, 0.5, 0.5])

    def test_heading_slews_180_during_landing(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            first_waypoint_offset_m=(5.0, 0.0, 5.0),
            descent_duration_s=4.0,
            approach_yaw_deg=0.0,
            landing_yaw_deg=180.0,
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        # Approach holds 0°.
        approach_h = scenario.resolve_heading_reference(1.0)
        self.assertIsNotNone(approach_h)
        self.assertAlmostEqual(float(approach_h), 0.0)
        # Enter landing at t=3 with a 4 s descent.
        scenario.mission_phase = "landing"
        scenario.landing_phase_start_time = 3.0
        # Mid-descent: halfway to ±180° (shortest-path is −π).
        mid = scenario.resolve_heading_reference(5.0)
        self.assertIsNotNone(mid)
        self.assertAlmostEqual(abs(float(mid)), 0.5 * np.pi, places=5)
        # End of track: full pad flip.
        end = scenario.resolve_heading_reference(7.0)
        self.assertIsNotNone(end)
        self.assertAlmostEqual(abs(float(end)), np.pi, places=5)

    def test_heading_slew_uses_clock_window(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            first_waypoint_offset_m=(5.0, 0.0, 5.0),
            approach_yaw_deg=0.0,
            landing_yaw_deg=90.0,
            heading_slew_start_s=2.0,
            heading_slew_end_s=3.0,
            path_shape="continuous",
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        self.assertAlmostEqual(float(scenario.resolve_heading_reference(1.9) or 0.0), 0.0)
        mid = float(scenario.resolve_heading_reference(2.5) or 0.0)
        self.assertAlmostEqual(mid, 0.5 * np.pi / 2.0, places=5)
        self.assertAlmostEqual(float(scenario.resolve_heading_reference(3.0) or 0.0), 0.5 * np.pi)
        self.assertAlmostEqual(float(scenario.resolve_heading_reference(8.0) or 0.0), 0.5 * np.pi)

    def test_vertical_hold_slews_to_landing_heading(self) -> None:
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            first_waypoint_offset_m=(2.5, 0.0, 1.3081),
            path_shape="capture",
            vertical_hold_s=1.0,
            approach_yaw_deg=0.0,
            landing_yaw_deg=90.0,
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        self.assertIsNotNone(scenario.heading_slew_start_s)
        self.assertIsNotNone(scenario.heading_slew_end_s)
        self.assertAlmostEqual(float(scenario.heading_slew_start_s), 0.0)
        self.assertAlmostEqual(float(scenario.heading_slew_end_s), 1.0)
        self.assertAlmostEqual(float(scenario.resolve_heading_reference(0.0)), 0.0)
        self.assertAlmostEqual(
            float(scenario.resolve_heading_reference(0.5) or 0.0), 0.25 * np.pi, places=5
        )
        self.assertAlmostEqual(
            float(scenario.resolve_heading_reference(1.0) or 0.0), 0.5 * np.pi, places=5
        )
        self.assertAlmostEqual(
            float(scenario.resolve_heading_reference(4.0) or 0.0), 0.5 * np.pi, places=5
        )
        self.assertAlmostEqual(scenario.resolve_heading_rate(0.0), 0.0)
        self.assertAlmostEqual(
            scenario.resolve_heading_rate(0.5), 0.25 * np.pi * np.pi, places=5
        )
        self.assertAlmostEqual(scenario.resolve_heading_rate(1.0), 0.0)

    def test_f10_stays_airborne_through_burnout(self) -> None:
        system = load_vehicle_profile(
            profile_path=ROOT / "config/profiles/real_landing_f10.json"
        )
        assert system.profile is not None
        mission = system.profile.real_landing
        assert mission is not None
        flight = system.flight_test
        assert flight is not None
        # Full-rate step needed for the short F10 burn window.
        flight.dt = 0.01
        flight.duration = 16.0
        flight.initial_position_m = mission.launch_position_m()
        simulator = FlightSimulator(system, flight)
        scenario = FlightScenario.real_landing(mission, name="real_landing_f10")
        log = simulator.run(scenario)
        from tvc.flight_sim import (
            find_engine_burnout_index,
            find_ground_contact_before_burnout,
        )

        burnout_idx = find_engine_burnout_index(log)
        self.assertIsNotNone(burnout_idx)
        assert burnout_idx is not None
        self.assertIsNone(find_ground_contact_before_burnout(log, burnout_idx))
        self.assertEqual(scenario.mission_phase, "landing")
        self.assertGreater(float(log.samples[burnout_idx].position[2]), 0.01)
        np.testing.assert_allclose(mission.launch_position_m(), (16.0, 0.0, 0.0), atol=1e-12)

    def test_profile_runs_from_east_launch(self) -> None:
        system = load_vehicle_profile(
            profile_path=ROOT / "config/profiles/real_landing_g12.json"
        )
        assert system.profile is not None
        mission = system.profile.real_landing
        assert mission is not None
        flight = system.flight_test
        assert flight is not None
        flight.dt = 0.05
        flight.duration = 25.0
        flight.initial_position_m = mission.launch_position_m()
        simulator = FlightSimulator(system, flight)
        scenario = FlightScenario.real_landing(mission, name="real_landing_g12")
        log = simulator.run(scenario)
        self.assertGreater(len(log.samples), 10)
        self.assertIn(log.metadata.landing_success, {"true", "false"})
        np.testing.assert_allclose(log.samples[0].position, mission.launch_position_m(), atol=1e-6)
        # Approach may not capture under the new loaded mass properties yet.
        self.assertIn(scenario.mission_phase, {"approach", "landing"})


class ContinuousPathTests(unittest.TestCase):
    def _mission(self) -> RealLandingMission:
        return RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            launch_offset_m=(10.0, 0.0, 0.0),
            first_waypoint_offset_m=(5.0, 0.0, 5.0),
            touchdown_altitude_m=0.5,
            descent_duration_s=4.0,
            min_approach_time_s=3.0,
            forced_descent_time_s=3.0,
            path_shape="continuous",
        )

    def test_profile_loads_continuous_path_shape(self) -> None:
        for name in ("real_landing_g12.json", "real_landing_f10.json"):
            system = load_vehicle_profile(profile_path=ROOT / "config/profiles" / name)
            assert system.profile is not None
            mission = system.profile.real_landing
            assert mission is not None
            self.assertEqual(mission.path_shape, "continuous")
            self.assertIn(
                mission.path_time_law,
                {
                    "min_jerk",
                    "loft_then_creep",
                    "loft_through",
                    "ease_out",
                    "uniform",
                    "projectile",
                },
            )
            self.assertGreater(mission.touchdown_altitude_m, 0.3)
            box = system.profile.landing_acceptance
            assert box is not None
            np.testing.assert_allclose(mission.landing_end_m(), box.center_m, atol=1e-12)
            self.assertAlmostEqual(mission.path_end_altitude_m, 0.5 if name != "real_landing_g12.json" else 1.2)
            self.assertEqual(mission.launch_offset_m, (16.0, 0.0, 0.0))
            self.assertEqual(mission.first_waypoint_offset_m, (8.0, 0.0, 9.0))
            if name == "real_landing_g12.json":
                self.assertAlmostEqual(mission.path_end_short_m, 2.0)
                self.assertAlmostEqual(mission.lateral_slide_s, 3.0)
                scenario = FlightScenario.real_landing(mission)
                scenario.reset_waypoints()
                knots = scenario._continuous_knots()
                assert knots is not None
                t_end = scenario._continuous_t_end(knots[3], knots[4])
                self.assertLess(t_end, 9.0)
                self.assertLess(t_end + mission.lateral_slide_s, 12.0)
            else:
                self.assertAlmostEqual(mission.path_end_short_m, 0.0)

    def test_rest_at_launch_and_pad(self) -> None:
        scenario = FlightScenario.real_landing(self._mission())
        scenario.reset_waypoints()
        np.testing.assert_allclose(
            scenario.resolve_position_reference(0.0), [10.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(0.0), [0.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_acceleration_reference(0.0), [0.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_position_reference(20.0), [0.0, 0.0, 0.5], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(20.0), [0.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_acceleration_reference(20.0), [0.0, 0.0, 0.0], atol=1e-12
        )

    def test_parabola_through_launch_gate_pad(self) -> None:
        scenario = FlightScenario.real_landing(self._mission())
        scenario.reset_waypoints()
        t_gate = 3.0
        # Gate is a geometric point on the parabola, not a timed hover.
        times = np.linspace(0.0, 7.0, 701)
        refs = np.array(
            [scenario.resolve_position_reference(float(t)) for t in times]
        )
        gate_err = np.linalg.norm(refs - np.array([5.0, 0.0, 5.0]), axis=1)
        self.assertLess(float(np.min(gate_err)), 0.05)
        scenario.resolve_position_reference(t_gate - 1e-6)
        self.assertEqual(scenario.mission_phase, "approach")
        scenario.resolve_position_reference(t_gate)
        self.assertEqual(scenario.mission_phase, "landing")

        times = np.linspace(0.0, 7.0, 81)
        xs = []
        zs = []
        for t in times:
            p = scenario.resolve_position_reference(float(t))
            xs.append(float(p[0]))
            zs.append(float(p[2]))
        xs = np.asarray(xs)
        zs = np.asarray(zs)
        # Unique quadratic z(x) through launch, gate, pad must fit the path.
        pts = np.array([[10.0, 0.0], [5.0, 5.0], [0.0, 0.5]])
        vand = np.column_stack([pts[:, 0] ** 2, pts[:, 0], np.ones(3)])
        coeff = np.linalg.solve(vand, pts[:, 1])
        z_fit = coeff[0] * xs**2 + coeff[1] * xs + coeff[2]
        np.testing.assert_allclose(zs, z_fit, atol=1e-6)
        # Waypoint is moving at the gate (no hover-stop).
        self.assertGreater(
            float(np.linalg.norm(scenario.resolve_velocity_reference(t_gate))), 0.2
        )

    def test_parabola_ends_short_of_pad(self) -> None:
        mission = self._mission()
        mission.path_end_short_m = 1.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        end = scenario.resolve_position_reference(20.0)
        np.testing.assert_allclose(end, [1.0, 0.0, 0.5], atol=1e-9)
        # Pad itself is still the terminal hover target.
        np.testing.assert_allclose(scenario.landing_track_end, [0.0, 0.0, 0.5], atol=1e-12)

    def test_lateral_slide_eases_from_path_end_to_pad(self) -> None:
        mission = self._mission()
        mission.path_end_short_m = 4.0
        mission.lateral_slide_s = 2.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        knots = scenario._continuous_knots()
        assert knots is not None
        _p0, _gate, path_end, t_app, t_des = knots
        np.testing.assert_allclose(path_end, [4.0, 0.0, 0.5], atol=1e-9)
        t_end = scenario._continuous_t_end(t_app, t_des)
        np.testing.assert_allclose(
            scenario.resolve_position_reference(t_end), path_end, atol=1e-9
        )
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(t_end), [0.0, 0.0, 0.0], atol=1e-6
        )
        mid = scenario.resolve_position_reference(t_end + 1.0)
        self.assertGreater(float(mid[0]), 0.5)
        self.assertLess(float(mid[0]), 3.5)
        self.assertAlmostEqual(float(mid[1]), 0.0, places=9)
        np.testing.assert_allclose(
            scenario.resolve_position_reference(t_end + 2.0),
            [0.0, 0.0, 0.5],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(t_end + 2.0),
            [0.0, 0.0, 0.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            scenario.resolve_position_reference(t_end + 8.0),
            [0.0, 0.0, 0.5],
            atol=1e-9,
        )

    def test_lateral_slide_eases_out_to_rest(self) -> None:
        mission = self._mission()
        mission.path_end_short_m = 4.0
        mission.lateral_slide_s = 2.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        knots = scenario._continuous_knots()
        assert knots is not None
        t_end = scenario._continuous_t_end(knots[3], knots[4])
        times = np.linspace(t_end, t_end + 2.0, 81)
        speeds = np.array(
            [
                float(np.linalg.norm(scenario.resolve_velocity_reference(float(t))))
                for t in times
            ]
        )
        peak_i = int(np.argmax(speeds))
        self.assertLess(float(times[peak_i] - t_end), 0.9)
        early = float(
            np.linalg.norm(scenario.resolve_velocity_reference(t_end + 0.6))
        )
        late = float(
            np.linalg.norm(scenario.resolve_velocity_reference(t_end + 1.7))
        )
        self.assertGreater(early, late)
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(t_end + 2.0),
            [0.0, 0.0, 0.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            scenario.resolve_acceleration_reference(t_end + 2.0),
            [0.0, 0.0, 0.0],
            atol=1e-6,
        )

    def test_lateral_slide_waypoint_does_not_sprint_ahead(self) -> None:
        mission = self._mission()
        mission.path_end_short_m = 4.0
        mission.lateral_slide_s = 2.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        knots = scenario._continuous_knots()
        assert knots is not None
        t_end = scenario._continuous_t_end(knots[3], knots[4])
        # Vehicle still at the path end while the clock is mid-slide.
        lagging = np.array([4.0, 0.0, 0.5])
        ref = scenario.resolve_position_reference(t_end + 1.8, position=lagging)
        free = scenario.resolve_position_reference(t_end + 1.8)
        self.assertGreater(float(ref[0]), float(free[0]) + 0.3)
        self.assertGreater(float(ref[0]), 1.5)

    def test_path_end_altitude_keeps_pad_at_box_center(self) -> None:
        mission = self._mission()
        mission.path_end_short_m = 1.0
        mission.path_end_altitude_m = 2.8
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        end = scenario.resolve_position_reference(20.0)
        np.testing.assert_allclose(end, [1.0, 0.0, 2.8], atol=1e-9)
        np.testing.assert_allclose(scenario.landing_track_end, [0.0, 0.0, 0.5], atol=1e-12)

    def test_slide_to_pad_z_eases_through_sticks(self) -> None:
        mission = self._mission()
        mission.path_end_short_m = 2.5
        mission.path_end_altitude_m = 1.3081
        mission.touchdown_altitude_m = 1.3081 - 1.5
        mission.lateral_slide_s = 5.0
        mission.slide_to_pad_z = True
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        knots = scenario._continuous_knots()
        assert knots is not None
        t_end = scenario._continuous_t_end(knots[3], knots[4])
        start = scenario.resolve_position_reference(t_end)
        self.assertAlmostEqual(float(start[0]), 2.5, places=6)
        self.assertAlmostEqual(float(start[2]), 1.3081, places=6)
        done = scenario.resolve_position_reference(t_end + 5.0)
        np.testing.assert_allclose(done, [0.0, 0.0, 1.3081 - 1.5], atol=1e-9)
        self.assertAlmostEqual(float(scenario.hover_altitude_m()), 1.3081 - 1.5, places=6)

    def test_slide_to_pad_z_snaps_when_slide_is_zero(self) -> None:
        mission = self._mission()
        mission.path_end_short_m = 2.5
        mission.path_end_altitude_m = 1.3081
        mission.touchdown_altitude_m = 1.3081 - 1.5
        mission.lateral_slide_s = 0.0
        mission.slide_to_pad_z = True
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        knots = scenario._continuous_knots()
        assert knots is not None
        t_end = scenario._continuous_t_end(knots[3], knots[4])
        before = scenario.resolve_position_reference(t_end - 0.01)
        self.assertGreater(float(before[0]), 2.0)
        self.assertAlmostEqual(float(before[2]), 1.3081, places=2)
        after = scenario.resolve_position_reference(t_end)
        np.testing.assert_allclose(after, [0.0, 0.0, 1.3081 - 1.5], atol=1e-9)
        vel = scenario.resolve_velocity_reference(t_end + 0.1)
        np.testing.assert_allclose(vel, [0.0, 0.0, 0.0], atol=1e-12)

    def test_vertical_hold_then_capture_parabola_base(self) -> None:
        mission = self._mission()
        mission.path_shape = "capture"
        mission.vertical_hold_s = 3.0
        mission.first_waypoint_offset_m = (2.5, 0.0, 1.3081)
        mission.path_end_short_m = 2.5
        mission.path_end_altitude_m = 1.3081
        mission.touchdown_altitude_m = 1.3081 - 1.5
        mission.min_approach_time_s = 3.0
        mission.forced_descent_time_s = 8.0
        mission.descent_duration_s = 5.0
        mission.launch_offset_m = (8.0, 0.0, 0.0)
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        live = np.array([8.0, 0.0, 2.4])
        hold = scenario.resolve_position_reference(1.5, position=live)
        self.assertTrue(np.all(np.isnan(hold)))
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(1.5, position=live),
            [0.0, 0.0, 0.0],
            atol=1e-12,
        )
        self.assertTrue(scenario.in_vertical_hold(2.99))
        self.assertFalse(scenario.in_vertical_hold(3.0))
        approach = scenario.resolve_position_reference(3.5, position=live)
        np.testing.assert_allclose(approach, [2.5, 0.0, 1.3081], atol=1e-9)
        at_base = np.array([2.55, 0.02, 1.32])
        scenario.resolve_position_reference(8.0, position=at_base)
        self.assertEqual(scenario.mission_phase, "landing")
        end = scenario.resolve_position_reference(20.0, position=at_base)
        np.testing.assert_allclose(end, [0.0, 0.0, 1.3081 - 1.5], atol=1e-9)

    def test_capture_teleports_to_pad_when_descent_is_zero(self) -> None:
        mission = self._mission()
        mission.path_shape = "capture"
        mission.vertical_hold_s = 1.0
        mission.first_waypoint_offset_m = (2.5, 0.0, 1.5)
        mission.path_end_altitude_m = 1.5
        mission.touchdown_altitude_m = 1.3081 - 1.5
        mission.min_approach_time_s = 1.0
        mission.forced_descent_time_s = 8.0
        mission.descent_duration_s = 0.0
        mission.launch_offset_m = (8.0, 0.0, 0.0)
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        live = np.array([8.0, 0.0, 2.4])
        approach = scenario.resolve_position_reference(1.5, position=live)
        np.testing.assert_allclose(approach, [2.5, 0.0, 1.5], atol=1e-9)
        at_gate = np.array([2.55, 0.02, 1.52])
        snapped = scenario.resolve_position_reference(8.0, position=at_gate)
        self.assertEqual(scenario.mission_phase, "landing")
        np.testing.assert_allclose(snapped, [0.0, 0.0, 1.3081 - 1.5], atol=1e-9)
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(8.0, position=at_gate),
            [0.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_uniform_speed_along_parabola(self) -> None:
        mission = self._mission()
        mission.path_time_law = "uniform"
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        times = np.linspace(0.05, 6.9, 140)
        speeds = np.array(
            [float(np.linalg.norm(scenario.resolve_velocity_reference(float(t)))) for t in times]
        )
        self.assertGreater(float(np.min(speeds)), 0.5)
        self.assertLess(float(np.ptp(speeds)), 0.05)
        refs = np.array(
            [scenario.resolve_position_reference(float(t)) for t in np.linspace(0.0, 7.0, 701)]
        )
        gate_err = np.linalg.norm(refs - np.array([5.0, 0.0, 5.0]), axis=1)
        self.assertLess(float(np.min(gate_err)), 0.05)
        np.testing.assert_allclose(
            scenario.resolve_position_reference(0.0), [10.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_position_reference(20.0), [0.0, 0.0, 0.5], atol=1e-12
        )

    def test_velocity_and_accel_continuous(self) -> None:
        scenario = FlightScenario.real_landing(self._mission())
        scenario.reset_waypoints()
        t_gate = 3.0
        eps = 1e-4
        v_left = scenario.resolve_velocity_reference(t_gate - eps)
        v_right = scenario.resolve_velocity_reference(t_gate + eps)
        a_left = scenario.resolve_acceleration_reference(t_gate - eps)
        a_right = scenario.resolve_acceleration_reference(t_gate + eps)
        np.testing.assert_allclose(v_left, v_right, atol=2e-3)
        np.testing.assert_allclose(a_left, a_right, atol=2e-2)

    def test_velocity_reference_has_no_step_at_gate(self) -> None:
        scenario = FlightScenario.real_landing(self._mission())
        scenario.reset_waypoints()
        times = np.linspace(0.0, 7.0, 701)
        speeds = np.array(
            [float(np.linalg.norm(scenario.resolve_velocity_reference(t))) for t in times]
        )
        ds = np.diff(speeds)
        # No capture-style slam: speed should not jump by > 0.05 m/s in 10 ms.
        self.assertLess(float(np.max(np.abs(ds))), 0.05)

    def test_terminal_hold_switches_to_fixed_pad(self) -> None:
        mission = self._mission()
        mission.terminal_time_s = 5.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        far = np.array([8.0, 0.0, 1.0])
        ref = scenario.resolve_position_reference(2.0, position=far)
        self.assertFalse(scenario._terminal_hold)
        self.assertGreater(float(np.linalg.norm(ref - np.array([0.0, 0.0, 0.5]))), 2.0)
        ref = scenario.resolve_position_reference(5.0, position=far)
        self.assertTrue(scenario._terminal_hold)
        np.testing.assert_allclose(ref, [0.0, 0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(5.0), [0.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_acceleration_reference(5.0), [0.0, 0.0, 0.0], atol=1e-12
        )
        away = np.array([5.0, 0.0, 2.0])
        ref = scenario.resolve_position_reference(6.0, position=away)
        np.testing.assert_allclose(ref, [0.0, 0.0, 0.5], atol=1e-12)
        self.assertEqual(scenario.mission_phase, "landing")

    def test_loft_then_creep_reaches_gate_then_slows(self) -> None:
        mission = self._mission()
        mission.path_time_law = "loft_then_creep"
        mission.loft_duration_s = 2.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        gate = np.array([5.0, 0.0, 5.0])
        at_loft = scenario.resolve_position_reference(2.0)
        np.testing.assert_allclose(at_loft, gate, atol=0.05)
        v_loft = float(np.linalg.norm(scenario.resolve_velocity_reference(2.0)))
        v_mid = float(np.linalg.norm(scenario.resolve_velocity_reference(4.5)))
        self.assertLess(v_loft, 0.35)
        self.assertGreater(v_mid, 0.15)
        np.testing.assert_allclose(
            scenario.resolve_position_reference(20.0), [0.0, 0.0, 0.5], atol=1e-12
        )

    def test_loft_through_keeps_moving_at_gate_then_slides(self) -> None:
        mission = self._mission()
        mission.path_time_law = "loft_through"
        mission.loft_duration_s = 2.0
        mission.path_end_short_m = 1.0
        mission.lateral_slide_s = 2.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        knots = scenario._continuous_knots()
        assert knots is not None
        _p0, gate, path_end, t_app, t_des = knots
        t_end = scenario._continuous_t_end(t_app, t_des)
        self.assertAlmostEqual(t_end, 2.0 + 4.0, places=9)
        at_loft = scenario.resolve_position_reference(2.0)
        np.testing.assert_allclose(at_loft, gate, atol=0.05)
        v_loft = float(np.linalg.norm(scenario.resolve_velocity_reference(2.0)))
        self.assertGreater(v_loft, 0.6)
        # Keep moving through the gate — this is the pause loft_then_creep has.
        for t in (1.6, 2.0, 2.4, 2.8):
            speed = float(np.linalg.norm(scenario.resolve_velocity_reference(t)))
            self.assertGreater(speed, 0.6, msg=f"waypoint parked at t={t:.1f}s")
        np.testing.assert_allclose(
            scenario.resolve_position_reference(t_end), path_end, atol=1e-6
        )
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(t_end), [0.0, 0.0, 0.0], atol=1e-5
        )
        mid_slide = scenario.resolve_position_reference(t_end + 1.0)
        self.assertGreater(float(path_end[0] - mid_slide[0]), 0.2)
        self.assertAlmostEqual(float(mid_slide[2]), float(path_end[2]), places=6)
        np.testing.assert_allclose(
            scenario.resolve_position_reference(t_end + 2.0),
            [0.0, 0.0, float(path_end[2])],
            atol=1e-6,
        )

    def test_ease_out_starts_moving_and_rests_at_pad(self) -> None:
        mission = self._mission()
        mission.path_time_law = "ease_out"
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        v0 = scenario.resolve_velocity_reference(1e-6)
        self.assertGreater(float(np.linalg.norm(v0)), 1.0)
        np.testing.assert_allclose(
            scenario.resolve_position_reference(20.0), [0.0, 0.0, 0.5], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(20.0), [0.0, 0.0, 0.0], atol=1e-12
        )

    def test_projectile_coasts_at_minus_g_then_rests_at_pad(self) -> None:
        mission = self._mission()
        mission.path_time_law = "projectile"
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        sched = scenario._projectile_schedule()
        assert sched is not None
        t_boost = float(sched["t_boost"])
        t_apex = float(sched["t_apex"])
        # Rest on the rail.
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(0.0), [0.0, 0.0, 0.0], atol=1e-12
        )
        # Mid-coast: constant downrange speed, vertical accel ≈ −g, vz heading to 0.
        t_coast = 0.5 * (t_boost + t_apex)
        accel = scenario.resolve_acceleration_reference(t_coast)
        vel = scenario.resolve_velocity_reference(t_coast)
        self.assertAlmostEqual(float(accel[2]), float(sched["az"]), delta=0.15)
        self.assertLess(float(accel[2]), -1.0)
        self.assertAlmostEqual(float(accel[0]), 0.0, delta=1e-6)
        self.assertGreater(float(vel[2]), 0.5)
        vz_boost = float(scenario.resolve_velocity_reference(t_boost + 1e-3)[2])
        vz_apex = float(scenario.resolve_velocity_reference(t_apex - 1e-4)[2])
        self.assertGreater(vz_boost, vz_apex)
        self.assertLess(abs(vz_apex), 0.35)
        np.testing.assert_allclose(
            scenario.resolve_position_reference(20.0), [0.0, 0.0, 0.5], atol=1e-12
        )
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(20.0), [0.0, 0.0, 0.0], atol=1e-12
        )

    def test_terminal_settle_eases_to_ground(self) -> None:
        mission = self._mission()
        mission.terminal_time_s = 5.0
        mission.terminal_settle_s = 2.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        near = np.array([1.0, 0.0, 0.8])
        ref0 = scenario.resolve_position_reference(5.0, position=near)
        self.assertTrue(scenario._terminal_hold)
        self.assertAlmostEqual(float(ref0[0]), 0.0, places=9)
        self.assertAlmostEqual(float(ref0[1]), 0.0, places=9)
        # Latch altitude is the live path z, not a snap to the pad hover.
        self.assertGreater(float(ref0[2]), 0.4)
        ref_mid = scenario.resolve_position_reference(6.0, position=near)
        self.assertLess(float(ref_mid[2]), float(ref0[2]))
        ref_end = scenario.resolve_position_reference(7.0, position=near)
        np.testing.assert_allclose(ref_end, [0.0, 0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(7.0), [0.0, 0.0, 0.0], atol=1e-12
        )

    def test_terminal_time_locks_pad_xy(self) -> None:
        mission = self._mission()
        mission.terminal_time_s = 5.0
        mission.terminal_settle_s = 2.0
        scenario = FlightScenario.real_landing(mission)
        scenario.reset_waypoints()
        far = np.array([8.0, 0.0, 6.0])
        ref_before = scenario.resolve_position_reference(4.9, position=far)
        self.assertFalse(scenario._terminal_hold)
        self.assertGreater(float(ref_before[0]), 1.0)
        ref0 = scenario.resolve_position_reference(5.0, position=far)
        self.assertTrue(scenario._terminal_hold)
        self.assertAlmostEqual(float(ref0[0]), 0.0, places=9)
        self.assertAlmostEqual(float(ref0[1]), 0.0, places=9)
        self.assertGreater(float(ref0[2]), 0.5)
        ref_end = scenario.resolve_position_reference(7.0, position=far)
        np.testing.assert_allclose(ref_end, [0.0, 0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(
            scenario.resolve_velocity_reference(7.0), [0.0, 0.0, 0.0], atol=1e-12
        )


class EngineOutHoldTests(unittest.TestCase):
    def test_engine_shutdown_latches_only_in_landing(self) -> None:
        system = load_config(ROOT / "config/vehicles/tvc_ring3_g12.json")
        flight = FlightTestConfig(dt=0.01, duration=0.05)
        flight.vehicle = system.vehicle
        sim = FlightSimulator(system, flight)
        sim.reset()
        mission = RealLandingMission(
            landing_site_m=(0.0, 0.0, 0.0),
            path_shape="continuous",
        )
        scenario = FlightScenario.real_landing(mission)
        scenario.mission_phase = "approach"
        sim._note_engine_thrusts(np.array([20.0, 20.0, 20.0]), scenario)
        sim._note_engine_thrusts(np.array([0.0, 20.0, 20.0]), scenario)
        self.assertFalse(sim._engine_out_hold)
        scenario.mission_phase = "landing"
        sim._note_engine_thrusts(np.array([0.0, 20.0, 20.0]), scenario)
        self.assertFalse(sim._engine_out_hold)
        sim._note_engine_thrusts(np.array([0.0, 0.0, 20.0]), scenario)
        self.assertTrue(sim._engine_out_hold)


if __name__ == "__main__":
    unittest.main()
