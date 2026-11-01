import unittest
from pathlib import Path

import numpy as np

from tvc.dynamics import compose_loaded_properties, point_mass_inertia
from tvc.parameters import load_config


ROOT = Path(__file__).resolve().parents[1]


class ComposeLoadedPropertiesTests(unittest.TestCase):
    def test_no_point_masses_returns_unloaded(self) -> None:
        inertia = np.diag([0.75, 0.75, 0.02])
        mass, com, i_out = compose_loaded_properties(
            2.713,
            (0.0, 0.0, 0.4445),
            inertia,
            [],
        )
        self.assertAlmostEqual(mass, 2.713)
        np.testing.assert_allclose(com, [0.0, 0.0, 0.4445])
        np.testing.assert_allclose(i_out, inertia)

    def test_symmetric_ring_motors_shift_com_down(self) -> None:
        unloaded_mass = 2.713
        unloaded_com = np.array([0.0, 0.0, 0.4445])
        unloaded_inertia = np.diag([0.75, 0.75, 0.02])
        motor_mass = 0.150
        positions = [
            np.array([0.098, 0.0, 0.0]),
            np.array([-0.049, 0.084870489570875, 0.0]),
            np.array([-0.049, -0.084870489570875, 0.0]),
        ]
        mass, com, inertia = compose_loaded_properties(
            unloaded_mass,
            unloaded_com,
            unloaded_inertia,
            [(motor_mass, p) for p in positions],
        )
        expected_mass = unloaded_mass + 3.0 * motor_mass
        expected_z = unloaded_mass * 0.4445 / expected_mass
        self.assertAlmostEqual(mass, expected_mass, places=9)
        np.testing.assert_allclose(com, [0.0, 0.0, expected_z], atol=1e-9)
        # Pitch inertia must increase vs unloaded (parallel-axis + motor offsets).
        self.assertGreater(float(inertia[0, 0]), 0.75)
        self.assertGreater(float(inertia[1, 1]), 0.75)
        # Roll gains motor radial contribution.
        self.assertGreater(float(inertia[2, 2]), 0.02)

    def test_point_mass_inertia_formula(self) -> None:
        i = point_mass_inertia(2.0, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(i, np.diag([0.0, 2.0, 2.0]))


class VehicleConfigCompositionTests(unittest.TestCase):
    def test_g12_vehicle_composes_motor_masses(self) -> None:
        system = load_config(ROOT / "config/vehicles/tvc_ring3_g12.json")
        self.assertEqual(len(system.catch_arms), 2)
        self.assertAlmostEqual(system.catch_arms[0].length_m, 0.5)
        np.testing.assert_allclose(system.catch_arms[0].position[2], 51.5 * 0.0254)
        # Unloaded 2.713 kg + 3 × G12 (0.150 kg).
        self.assertAlmostEqual(system.vehicle.mass, 2.713 + 3 * 0.150, places=6)
        self.assertAlmostEqual(system.center_of_mass[2], 2.713 * 0.4445 / system.vehicle.mass, places=6)
        # Loaded pitch inertia is SI-scaled from 750 g·m² plus motor contribution.
        self.assertGreater(float(system.vehicle.inertia[0, 0]), 0.75)
        self.assertGreater(float(system.vehicle.inertia[2, 2]), 0.02)

    def test_offset_motor_cg_sits_opposite_thrust(self) -> None:
        from tvc.dynamics import compose_offset_mass_properties, offset_motor_cgs
        from tvc.geometry import thrust_direction

        system = load_config(ROOT / "config/vehicles/tvc_ring3_g12.json")
        rest = np.vstack([mount.rest_angles for mount in system.mounts])
        cgs = offset_motor_cgs(
            system.mounts, rest, standoff_m=system.motor_pivot_standoff_m
        )
        # G12 length 156 mm → offset = 55 mm + 78 mm = 133 mm below the ring at rest.
        expected_z = -(0.055 + 0.5 * 0.156)
        np.testing.assert_allclose(cgs[:, 2], expected_z, atol=1e-6)
        for mount, cg, pose in zip(system.mounts, cgs, rest):
            direction = thrust_direction(pose[0], pose[1], mount)
            np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-9)
            np.testing.assert_allclose(cg, mount.position - direction * abs(expected_z), atol=1e-9)
        mass, com, _inertia, _cgs = compose_offset_mass_properties(
            system.unloaded_vehicle,
            system.unloaded_center_of_mass,
            system.mounts,
            rest,
            standoff_m=system.motor_pivot_standoff_m,
        )
        self.assertAlmostEqual(mass, system.vehicle.mass, places=6)
        # Offset CGs are below the ring, so plant CoM is below pivot-composed CoM.
        self.assertLess(float(com[2]), float(system.center_of_mass[2]))

    def test_g12_mass_drops_as_propellant_burns(self) -> None:
        from tvc.dynamics import compose_offset_mass_properties

        system = load_config(ROOT / "config/vehicles/tvc_ring3_g12.json")
        rest = np.vstack([mount.rest_angles for mount in system.mounts])
        mass0, com0, _, _ = compose_offset_mass_properties(
            system.unloaded_vehicle,
            system.unloaded_center_of_mass,
            system.mounts,
            rest,
            standoff_m=system.motor_pivot_standoff_m,
            time_s=0.0,
        )
        mass_end, com_end, _, _ = compose_offset_mass_properties(
            system.unloaded_vehicle,
            system.unloaded_center_of_mass,
            system.mounts,
            rest,
            standoff_m=system.motor_pivot_standoff_m,
            time_s=20.0,
        )
        propellant = sum(
            float(mount.motor.propellant_mass_kg)
            for mount in system.mounts
            if mount.motor is not None
        )
        self.assertAlmostEqual(mass0, system.vehicle.mass, places=6)
        self.assertAlmostEqual(mass_end, mass0 - propellant, places=6)
        self.assertGreater(propellant, 0.1)
        # Spent grain is below the ring, so the CoM rises as mass leaves.
        self.assertGreater(float(com_end[2]), float(com0[2]))

    def test_inertia_g_m2_converts_to_si(self) -> None:
        system = load_config(ROOT / "config/vehicles/tvc_ring3.json")
        # Constant-thrust mounts still carry mass_g=150, so inertia > 0.75.
        self.assertGreater(float(system.vehicle.inertia[0, 0]), 0.74)
        self.assertLess(float(system.vehicle.inertia[2, 2]), 0.1)


if __name__ == "__main__":
    unittest.main()
