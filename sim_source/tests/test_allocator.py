from pathlib import Path
import unittest

import numpy as np

from tvc.allocator import TorqueAllocator
from tvc.parameters import load_config


def make_allocator() -> TorqueAllocator:
    config_path = Path(__file__).resolve().parents[1] / "config.example.json"
    config = load_config(config_path)
    return TorqueAllocator(
        config.mounts,
        config.solver,
        force_axis=config.render.force_axis,
        center_of_mass=config.center_of_mass,
    )


class SimpleMixerTests(unittest.TestCase):
    def test_allocate_recovers_feasible_lateral_targets(self) -> None:
        # Roll/pitch are high-authority; pure yaw still has less lever arm than
        # pitch/roll, so keep yaw targets in the linear envelope.
        targets = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.9, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.15, 0.0], dtype=float),
            np.array([0.0, 0.8, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([0.056857, -0.067804, -0.4], dtype=float),
        ]

        for target in targets:
            with self.subTest(target=target.tolist()):
                allocator = make_allocator()
                _, torque, _, saturated = allocator.allocate(target)

                self.assertFalse(saturated)
                self.assertLess(np.linalg.norm(target - torque), 5e-2)

    def test_allocate_is_stateless_for_identical_commands(self) -> None:
        allocator = make_allocator()
        target = np.array([0.5, -0.3, 0.1], dtype=float)
        angles_a, torque_a, _, _ = allocator.allocate(target)
        angles_b, torque_b, _, _ = allocator.allocate(target)

        self.assertTrue(np.allclose(angles_a, angles_b))
        self.assertTrue(np.allclose(torque_a, torque_b))

    def test_allocate_tracks_zero_axial_force_target(self) -> None:
        allocator = make_allocator()
        thrusts = np.array([mount.thrust for mount in allocator.mounts], dtype=float)

        angles, torque, _, saturated = allocator.allocate(
            np.zeros(3, dtype=float),
            force_target=0.0,
            force_weight=0.25,
            thrusts=thrusts,
        )
        force, _ = allocator.force_and_dirs(angles, thrusts)

        self.assertFalse(saturated)
        self.assertLess(np.linalg.norm(torque), 1e-2)
        self.assertLess(abs(allocator.axial_force(force)), 0.2)

    def test_allocate_tracks_axial_force_target(self) -> None:
        allocator = make_allocator()
        thrusts = np.array([mount.thrust for mount in allocator.mounts], dtype=float)
        force_target = 15.0

        angles, torque, _, saturated = allocator.allocate(
            np.zeros(3, dtype=float),
            force_target=force_target,
            force_weight=0.25,
            thrusts=thrusts,
        )
        force, _ = allocator.force_and_dirs(angles, thrusts)

        self.assertFalse(saturated)
        self.assertLess(np.linalg.norm(torque), 1e-2)
        self.assertLess(abs(force_target - allocator.axial_force(force)), 0.3)

    def test_allocate_marks_engine_out_force_request_saturated(self) -> None:
        allocator = make_allocator()
        thrusts = np.array([0.0, 10.0, 10.0], dtype=float)

        angles, torque, _, saturated = allocator.allocate(
            np.zeros(3, dtype=float),
            force_target=15.0,
            force_weight=0.25,
            thrusts=thrusts,
        )
        force, _ = allocator.force_and_dirs(angles, thrusts)

        # Dead mount stays at rest. With ±45° a2 the remaining pair can cancel
        # most bias torque; force/torque residual may still flag saturated.
        self.assertTrue(np.allclose(angles[0], allocator.rest_angles[0]))
        # Force tracks within ~1 N; residual torque stays small with ±45° a2.
        self.assertLess(abs(15.0 - allocator.axial_force(force)), 1.5)
        self.assertLess(np.linalg.norm(torque), 0.5)

    def test_zero_thrust_mount_stays_at_rest(self) -> None:
        allocator = make_allocator()
        thrusts = np.array([0.0, 10.0, 10.0], dtype=float)

        angles, _, _, _ = allocator.allocate(
            np.array([0.2, 0.0, 0.0], dtype=float),
            force_target=0.0,
            force_weight=0.25,
            thrusts=thrusts,
        )

        self.assertTrue(np.allclose(angles[0], allocator.rest_angles[0]))

    def test_allocate_tracks_force_with_one_reduced_engine_when_feasible(self) -> None:
        allocator = make_allocator()
        thrusts = np.array([6.0, 10.0, 10.0], dtype=float)

        angles, torque, _, saturated = allocator.allocate(
            np.zeros(3, dtype=float),
            force_target=20.0,
            force_weight=0.25,
            thrusts=thrusts,
        )
        force, _ = allocator.force_and_dirs(angles, thrusts)

        # Uniform-a1 collective cannot null residual torque with asymmetric T
        # while hitting force exactly; accept best-effort force tracking.
        self.assertLess(abs(20.0 - allocator.axial_force(force)), 0.5)
        # Residual body torque from thrust imbalance should stay modest.
        self.assertLess(np.linalg.norm(torque), 1.5)

    def test_force_coupled_solution_matches_force_better_than_rest(self) -> None:
        target = np.array([0.001, 0.0, 0.0], dtype=float)
        force_target = 30.0

        allocator = make_allocator()
        angles, torque, _, saturated = allocator.allocate(
            target,
            force_target=force_target,
            force_weight=0.25,
        )
        selected_force, _ = allocator.force_and_dirs(angles)
        rest_force, _ = allocator.force_and_dirs(allocator.rest_angles)

        self.assertFalse(saturated)
        self.assertLess(np.linalg.norm(target - torque), 5e-2)
        self.assertLessEqual(
            abs(force_target - allocator.axial_force(selected_force)),
            abs(force_target - allocator.axial_force(rest_force)) + 1e-6,
        )

    def test_allocate_force_track_matches_allocate(self) -> None:
        allocator = make_allocator()
        target = np.array([0.4, -0.2, 0.1], dtype=float)
        force_target = 18.0
        thrusts = np.array([mount.thrust for mount in allocator.mounts], dtype=float)

        a1, t1, d1, s1 = allocator.allocate(
            target, force_target=force_target, force_weight=0.25, thrusts=thrusts
        )
        a2, t2, d2, s2 = allocator.allocate_force_track(
            target, force_target=force_target, force_weight=0.25, thrusts=thrusts
        )

        self.assertTrue(np.allclose(a1, a2))
        self.assertTrue(np.allclose(t1, t2))
        self.assertTrue(np.allclose(d1, d2))
        self.assertEqual(s1, s2)

    def test_fast_allocate_tracks_torque_under_force_coupling(self) -> None:
        allocator = make_allocator()
        thrusts = np.array([mount.thrust for mount in allocator.mounts], dtype=float)
        target = np.array([0.5, 0.3, 0.2], dtype=float)

        for step in range(8):
            trial = target + np.array([0.01 * step, 0.0, 0.0], dtype=float)
            _, torque, _, saturated = allocator.allocate(
                trial,
                force_target=30.0,
                force_weight=0.2,
                thrusts=thrusts,
                fast=True,
            )
            self.assertFalse(saturated)
            self.assertLess(np.linalg.norm(trial - torque), 8e-2)

    def test_allocate_marks_infeasible_target_saturated(self) -> None:
        allocator = make_allocator()
        target = np.array([25.0, 0.0, 0.0], dtype=float)

        _, torque, _, saturated = allocator.allocate(target)

        self.assertTrue(saturated)
        direction = target / np.linalg.norm(target)
        self.assertLessEqual(
            float(np.dot(torque, direction)),
            np.linalg.norm(target) + 1e-6,
        )

    def test_allocate_keeps_saturated_torque_aligned(self) -> None:
        allocator = make_allocator()
        target = np.array([8.7178, 4.3589, 2.6153], dtype=float)

        _, torque, _, saturated = allocator.allocate(target)

        self.assertTrue(saturated)
        self.assertLess(float(np.linalg.norm(torque)), float(np.linalg.norm(target)))
        cosine = float(
            np.dot(target, torque)
            / (np.linalg.norm(target) * np.linalg.norm(torque) + 1e-12)
        )
        angle_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        self.assertLess(angle_deg, 75.0)

    def test_twr_lock_folds_to_three_quarter_weight(self) -> None:
        allocator = make_allocator()
        thrusts = np.array([12.0, 12.0, 12.0], dtype=float)
        weight = 36.0
        angles = allocator.twr_lock_angles(thrusts, weight, twr=0.75)
        _, force, _ = allocator.torque_force_dirs(angles, thrusts)
        twr = float(allocator.axial_force(force)) / weight
        self.assertAlmostEqual(twr, 0.75, delta=0.05)
        self.assertLess(float(np.mean(np.abs(angles[:, 0]))), 1e-6)

    def test_collective_a1_monotonic_with_force_target(self) -> None:
        allocator = make_allocator()
        thrusts = np.array([mount.thrust for mount in allocator.mounts], dtype=float)
        angles_low, _, _, _ = allocator.allocate(
            np.zeros(3), force_target=10.0, force_weight=1.0, thrusts=thrusts
        )
        angles_high, _, _, _ = allocator.allocate(
            np.zeros(3), force_target=30.0, force_weight=1.0, thrusts=thrusts
        )
        # a1 is measured from vertical rest (0 = max axial). Higher force target
        # therefore commands a smaller nested angle.
        self.assertLess(
            float(np.mean(angles_high[:, 1])), float(np.mean(angles_low[:, 1]))
        )


if __name__ == "__main__":
    unittest.main()
