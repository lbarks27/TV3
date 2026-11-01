from __future__ import annotations

import unittest

import numpy as np

from tvc.dynamics import WindField, quadratic_drag_force_inertial
from tvc.parameters import _build_wind


class WindModelTests(unittest.TestCase):
    def test_hover_crosswind_pushes_with_wind(self) -> None:
        force = quadratic_drag_force_inertial(
            np.zeros(3),
            np.array([0.0, 3.0, 0.0]),
            drag_coeff=1.0,
            ref_area_m2=0.12,
        )
        self.assertGreater(float(force[1]), 0.0)
        self.assertAlmostEqual(float(force[0]), 0.0, places=9)
        self.assertAlmostEqual(float(force[2]), 0.0, places=9)

    def test_wind_config_roundtrip(self) -> None:
        cfg = _build_wind(
            {
                "velocity_m_s": [1.0, 0.0, 0.0],
                "gust_std_m_s": 0.5,
                "ref_area_m2": 0.1,
                "seed": 3,
            }
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertTrue(cfg.enabled())
        field = WindField(
            cfg.velocity_m_s,
            gust_std_m_s=cfg.gust_std_m_s,
            gust_tau_s=cfg.gust_tau_s,
            seed=cfg.seed,
        )
        field.reset()
        wind = field.step(0.01)
        self.assertEqual(wind.shape, (3,))

    def test_zero_wind_disabled(self) -> None:
        self.assertIsNone(_build_wind({"velocity_m_s": [0, 0, 0], "gust_std_m_s": 0}))


if __name__ == "__main__":
    unittest.main()
