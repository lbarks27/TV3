import unittest
from pathlib import Path

import numpy as np

from tvc.flight_sim import FlightScenario, FlightSimulator
from tvc.motors import MotorCurve, load_motor, thrusts_at
from tvc.parameters import (
    EngineVariance,
    FlightTestConfig,
    apply_engine_variance,
    load_config,
    load_vehicle_profile,
)


ROOT = Path(__file__).resolve().parents[1]


class MotorCurveTests(unittest.TestCase):
    def test_load_estes_d12_from_inventory(self) -> None:
        motor = load_motor("Estes/D12")
        self.assertEqual(motor.manufacturer, "Estes")
        self.assertEqual(motor.designation, "D12")
        self.assertGreater(motor.burn_time_s, 1.0)
        self.assertGreater(motor.max_thrust_n, 20.0)
        self.assertAlmostEqual(motor.thrust_at(-0.1), 0.0)
        self.assertAlmostEqual(motor.thrust_at(motor.times_s[-1] + 0.1), 0.0)

    def test_thrust_follows_samples_without_scaling(self) -> None:
        motor = load_motor("Estes/D12")
        mid_index = len(motor.times_s) // 2
        t0 = float(motor.times_s[mid_index])
        expected = float(motor.thrust_n[mid_index])
        self.assertAlmostEqual(motor.thrust_at(t0), expected, places=9)

        t_lo = float(motor.times_s[mid_index])
        t_hi = float(motor.times_s[mid_index + 1])
        f_lo = float(motor.thrust_n[mid_index])
        f_hi = float(motor.thrust_n[mid_index + 1])
        mid_t = 0.5 * (t_lo + t_hi)
        expected_mid = 0.5 * (f_lo + f_hi)
        self.assertAlmostEqual(motor.thrust_at(mid_t), expected_mid, places=9)

    def test_ignition_delay_shifts_curve(self) -> None:
        motor = load_motor({"manufacturer": "Estes", "designation": "D12", "ignition_delay_s": 0.5})
        bare = load_motor("Estes/D12")
        sample_t = float(bare.times_s[3])
        self.assertAlmostEqual(motor.thrust_at(0.5 + sample_t), bare.thrust_at(sample_t), places=9)
        self.assertAlmostEqual(motor.thrust_at(0.0), 0.0)

    def test_config_attaches_curves_to_all_mounts(self) -> None:
        system = load_vehicle_profile(profile_path=ROOT / "config/profiles/f10_sequential.json")
        self.assertTrue(all(mount.motor is not None for mount in system.mounts))
        self.assertTrue(all(mount.engine == "Apogee/F10" for mount in system.mounts))
        self.assertAlmostEqual(system.mounts[0].thrust, system.mounts[0].motor.max_thrust_n)
        self.assertEqual(
            [mount.motor.ignition_delay_s for mount in system.mounts],
            [0.0, 0.5, 1.0],
        )

    def test_flight_sim_mass_falls_during_g12_burn(self) -> None:
        system = load_config(ROOT / "config/vehicles/tvc_ring3_g12.json")
        flight = FlightTestConfig(dt=0.05, duration=13.0)
        flight.vehicle = system.vehicle
        simulator = FlightSimulator(system, flight)
        simulator.reset()
        start = float(simulator.vehicle.mass)
        simulator.run(FlightScenario.hold())
        end = float(simulator.vehicle.mass)
        propellant = sum(
            float(mount.motor.propellant_mass_kg)
            for mount in system.mounts
            if mount.motor is not None
        )
        self.assertGreater(propellant, 0.1)
        self.assertAlmostEqual(start, 2.713 + 3 * 0.150, places=5)
        self.assertAlmostEqual(end, start - propellant, places=5)

    def test_flight_sim_uses_curve_thrusts(self) -> None:
        system = load_vehicle_profile(profile_path=ROOT / "config/profiles/f10_sequential.json")
        flight = FlightTestConfig(dt=0.05, duration=1.2, thrust_per_mount=999.0)
        simulator = FlightSimulator(system, flight)
        log = simulator.run(FlightScenario.hold())
        for sample in log.samples:
            expected = np.array(
                [mount.motor.thrust_at(sample.time) for mount in system.mounts],
                dtype=float,
            )
            self.assertTrue(np.allclose(sample.mount_thrusts, expected))
            # Constant thrust_per_mount must not override selected engines.
            self.assertFalse(np.allclose(sample.mount_thrusts, 999.0))
        # Sequential ignition: only M0 lit at t=0.25, M0+M1 at t=0.75, all by t=1.1.
        early = next(sample for sample in log.samples if abs(sample.time - 0.25) < 1e-9)
        mid = next(sample for sample in log.samples if abs(sample.time - 0.75) < 1e-9)
        late = next(sample for sample in log.samples if abs(sample.time - 1.1) < 1e-9)
        self.assertGreater(early.mount_thrusts[0], 0.0)
        self.assertEqual(early.mount_thrusts[1], 0.0)
        self.assertEqual(early.mount_thrusts[2], 0.0)
        self.assertGreater(mid.mount_thrusts[0], 0.0)
        self.assertGreater(mid.mount_thrusts[1], 0.0)
        self.assertEqual(mid.mount_thrusts[2], 0.0)
        self.assertTrue(np.all(late.mount_thrusts > 0.0))

    def test_mass_tracks_delivered_impulse(self) -> None:
        motor = MotorCurve(
            manufacturer="Test",
            designation="T10",
            times_s=np.array([0.0, 1.0, 2.0]),
            thrust_n=np.array([10.0, 10.0, 10.0]),
            dynamics_path=Path("."),
            initial_mass_kg=0.150,
            propellant_mass_kg=0.060,
        )
        self.assertAlmostEqual(motor.impulse_at(-0.1), 0.0, places=9)
        self.assertAlmostEqual(motor.impulse_at(0.0), 0.0, places=9)
        self.assertAlmostEqual(motor.impulse_at(1.0), 10.0, places=6)
        self.assertAlmostEqual(motor.impulse_at(2.0), 20.0, places=6)
        self.assertAlmostEqual(motor.impulse_at(5.0), 20.0, places=6)
        self.assertAlmostEqual(motor.mass_at(0.0), 0.150, places=9)
        self.assertAlmostEqual(motor.mass_at(1.0), 0.120, places=9)
        self.assertAlmostEqual(motor.mass_at(2.0), 0.090, places=9)
        self.assertAlmostEqual(motor.mass_at(8.0), 0.090, places=9)

    def test_mass_waits_for_ignition_delay(self) -> None:
        motor = MotorCurve(
            manufacturer="Test",
            designation="T10",
            times_s=np.array([0.0, 2.0]),
            thrust_n=np.array([5.0, 5.0]),
            dynamics_path=Path("."),
            initial_mass_kg=0.100,
            propellant_mass_kg=0.040,
            ignition_delay_s=0.5,
        )
        self.assertAlmostEqual(motor.mass_at(0.5), 0.100, places=9)
        self.assertAlmostEqual(motor.mass_at(1.5), 0.080, places=6)
        self.assertAlmostEqual(motor.mass_at(2.5), 0.060, places=6)

    def test_g12_spec_has_propellant_and_burns_to_casing(self) -> None:
        motor = load_motor("AeroTech/G12")
        self.assertGreater(motor.initial_mass_kg, 0.0)
        self.assertGreater(motor.propellant_mass_kg, 0.0)
        self.assertLess(motor.propellant_mass_kg, motor.initial_mass_kg)
        self.assertAlmostEqual(motor.mass_at(0.0), motor.initial_mass_kg, places=9)
        dry = motor.initial_mass_kg - motor.propellant_mass_kg
        self.assertAlmostEqual(motor.mass_at(motor.burn_time_s + 1.0), dry, places=6)

    def test_with_performance_scales_thrust_and_burn_time(self) -> None:
        motor = MotorCurve(
            manufacturer="Test",
            designation="T10",
            times_s=np.array([0.0, 1.0, 2.0]),
            thrust_n=np.array([10.0, 10.0, 10.0]),
            dynamics_path=Path("."),
            initial_mass_kg=0.150,
            propellant_mass_kg=0.060,
        )
        hot = motor.with_performance(thrust_scale=1.05, time_scale=1.10)
        self.assertAlmostEqual(hot.thrust_at(0.5), 10.5, places=9)
        self.assertAlmostEqual(hot.thrust_at(2.0), 10.5, places=9)
        self.assertAlmostEqual(hot.thrust_at(2.2), 10.5, places=9)
        self.assertAlmostEqual(hot.thrust_at(2.21), 0.0, places=9)
        self.assertAlmostEqual(hot.burn_time_s, 2.2, places=9)
        self.assertAlmostEqual(hot.impulse_at(2.2), 20.0 * 1.05 * 1.10, places=6)
        self.assertAlmostEqual(hot.mass_at(0.0), 0.150, places=9)
        self.assertAlmostEqual(hot.mass_at(1.1), 0.120, places=6)
        self.assertAlmostEqual(hot.mass_at(2.2), 0.090, places=6)

    def test_g12_profile_applies_independent_engine_scatter(self) -> None:
        system = load_vehicle_profile(
            profile_path=ROOT / "config/profiles/real_landing_g12.json"
        )
        assert system.profile is not None
        variance = system.profile.engine_variance
        self.assertIsNotNone(variance)
        assert variance is not None
        self.assertAlmostEqual(variance.thrust, 0.05)
        self.assertAlmostEqual(variance.burn_time, 0.05)
        self.assertEqual(len(variance.draws), 3)
        self.assertEqual(len(system.mounts), 3)
        lo, hi = 0.95, 1.05
        for mount, draw in zip(system.mounts, variance.draws):
            thrust_scale, time_scale = float(draw[0]), float(draw[1])
            self.assertGreaterEqual(thrust_scale, lo)
            self.assertLessEqual(thrust_scale, hi)
            self.assertGreaterEqual(time_scale, lo)
            self.assertLessEqual(time_scale, hi)
            assert mount.motor is not None
            self.assertAlmostEqual(mount.motor.thrust_scale, thrust_scale, places=12)
            self.assertAlmostEqual(mount.motor.time_scale, time_scale, places=12)
            self.assertAlmostEqual(mount.thrust, mount.motor.max_thrust_n, places=9)
        # Independent draws: all three motors matching exactly is vanishingly rare.
        self.assertGreater(len(set(variance.draws)), 1)

    def test_engine_variance_seed_is_reproducible(self) -> None:
        system = load_config(ROOT / "config/vehicles/tvc_ring3_g12.json")
        first, draws_a = apply_engine_variance(
            system.mounts, EngineVariance(thrust=0.05, burn_time=0.05, seed=42)
        )
        second, draws_b = apply_engine_variance(
            system.mounts, EngineVariance(thrust=0.05, burn_time=0.05, seed=42)
        )
        self.assertEqual(draws_a, draws_b)
        self.assertEqual(len(draws_a), 3)
        for draw, mount in zip(draws_a, first):
            thrust_scale, time_scale = float(draw[0]), float(draw[1])
            self.assertGreaterEqual(thrust_scale, 0.95)
            self.assertLessEqual(thrust_scale, 1.05)
            self.assertGreaterEqual(time_scale, 0.95)
            self.assertLessEqual(time_scale, 1.05)
            assert mount.motor is not None
            self.assertAlmostEqual(mount.motor.thrust_scale, thrust_scale)
        self.assertEqual(
            [m.motor.thrust_scale for m in first],
            [m.motor.thrust_scale for m in second],
        )

    def test_engine_variance_ignition_delay_draw(self) -> None:
        system = load_config(ROOT / "config/vehicles/tvc_ring3_g12.json")
        mounts, draws = apply_engine_variance(
            system.mounts,
            EngineVariance(thrust=0.0, burn_time=0.0, ignition_delay_s=1.0, seed=7),
        )
        self.assertEqual(len(draws), 3)
        for draw, mount in zip(draws, mounts):
            ign = float(draw[2])
            self.assertGreaterEqual(ign, 0.0)
            self.assertLessEqual(ign, 1.0)
            assert mount.motor is not None
            self.assertAlmostEqual(mount.motor.ignition_delay_s, ign, places=12)
        self.assertGreater(max(d[2] for d in draws), 0.0)

    def test_thrusts_at_helper_keeps_fallback_for_missing_motors(self) -> None:
        motor = load_motor("Estes/D12")
        values = thrusts_at([motor, None], 0.4, np.array([0.0, 7.5]))
        self.assertAlmostEqual(values[0], motor.thrust_at(0.4), places=9)
        self.assertAlmostEqual(values[1], 7.5, places=9)


if __name__ == "__main__":
    unittest.main()
