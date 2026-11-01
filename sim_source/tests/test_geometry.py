import unittest

import numpy as np

from tvc.geometry import (
    _angle_limits_from_deg,
    axis_angle_rotation,
    build_ring_mounts,
    make_mount,
    mount_local_frame,
    thrust_direction,
    thrust_envelope_mesh,
    unit,
)


class AngleLimitTests(unittest.TestCase):
    def test_angle_limits_from_dict(self) -> None:
        # Storage order is (a2, a1).
        mins, maxs = _angle_limits_from_deg({"a1": [0.0, 90.0], "a2": [-45.0, 45.0]})
        np.testing.assert_allclose(np.rad2deg(mins), [-45.0, 0.0])
        np.testing.assert_allclose(np.rad2deg(maxs), [45.0, 90.0])

    def test_make_mount_stores_limits_in_a2_a1_order(self) -> None:
        mount = make_mount(
            position=[0.098, 0.0, 0.0],
            thrust=10.0,
            base_dir=[-1.0, 0.0, 0.0],
            axis1=[0.0, 1.0, 0.0],
            axis2=[-1.0, 0.0, 0.0],
            angle_limits_deg={"a1": [0.0, 90.0], "a2": [-45.0, 45.0]},
        )
        np.testing.assert_allclose(np.rad2deg(mount.min_angles), [-45.0, 0.0])
        np.testing.assert_allclose(np.rad2deg(mount.max_angles), [45.0, 90.0])
        np.testing.assert_allclose(np.rad2deg(mount.rest_angles), [0.0, 0.0])


class MountGeometryTests(unittest.TestCase):
    def test_mount_local_frame_faces_origin_with_nested_axes(self) -> None:
        radius = 0.098
        mounts = build_ring_mounts(count=3, radius=radius, thrust=10.0)

        self.assertEqual(len(mounts), 3)
        for mount in mounts:
            radial = mount.position / np.linalg.norm(mount.position)
            self.assertAlmostEqual(float(np.linalg.norm(mount.position)), radius, places=9)
            np.testing.assert_allclose(mount.base_dir, -radial, atol=1e-12)
            np.testing.assert_allclose(
                mount.axis1,
                np.array([-radial[1], radial[0], 0.0], dtype=float),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                mount.axis2,
                -radial,
                atol=1e-12,
            )
            self.assertAlmostEqual(float(np.dot(mount.base_dir, mount.axis1)), 0.0, places=12)
            self.assertAlmostEqual(float(np.dot(mount.axis1, mount.axis2)), 0.0, places=12)
            np.testing.assert_allclose(mount.axis2, mount.base_dir, atol=1e-12)

    def test_rest_pose_uses_configured_default_angles(self) -> None:
        mounts = build_ring_mounts(count=3, radius=0.098, thrust=10.0)
        for mount in mounts:
            np.testing.assert_allclose(np.rad2deg(mount.rest_angles), [0.0, 0.0])
            direction = thrust_direction(mount.rest_angles[0], mount.rest_angles[1], mount)
            np.testing.assert_allclose(direction, np.array([0.0, 0.0, 1.0]), atol=1e-9)
            np.testing.assert_allclose(
                direction,
                thrust_direction(0.0, 0.0, mount),
                atol=1e-12,
            )

    def test_nested_axis1_follows_primary_rotation(self) -> None:
        position = np.array([0.098, 0.0, 0.0], dtype=float)
        base_dir, axis1, axis2 = mount_local_frame(position)
        mount = build_ring_mounts(count=1, radius=0.098, thrust=10.0)[0]
        self.assertTrue(np.allclose(mount.base_dir, base_dir))
        self.assertTrue(np.allclose(mount.axis1, axis1))
        self.assertTrue(np.allclose(mount.axis2, axis2))

        direction = thrust_direction(0.0, np.deg2rad(10.0), mount)
        self.assertGreater(abs(direction[2]), 0.0)

        rot_primary = axis_angle_rotation(mount.axis2, np.deg2rad(10.0))
        nested_axis1 = unit(rot_primary @ mount.axis1)
        self.assertGreater(
            float(np.linalg.norm(nested_axis1 - mount.axis1)),
            0.1,
        )

    def test_thrust_envelope_lies_on_sphere_shell(self) -> None:
        mount = make_mount(
            position=[0.098, 0.0, 0.0],
            thrust=10.0,
            base_dir=[-1.0, 0.0, 0.0],
            axis1=[0.0, 1.0, 0.0],
            axis2=[-1.0, 0.0, 0.0],
            angle_limits_deg={"a1": [0.0, 90.0], "a2": [-45.0, 45.0]},
        )
        radius = 0.2
        for invert in (False, True):
            faces = thrust_envelope_mesh(
                mount, radius, samples_per_axis=8, invert=invert
            )
            self.assertGreater(len(faces), 0)
            for face in faces:
                for vertex in face:
                    dist = float(np.linalg.norm(vertex - mount.position))
                    self.assertAlmostEqual(dist, radius, places=9)


if __name__ == "__main__":
    unittest.main()
