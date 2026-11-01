#!/usr/bin/env python3
"""Regression for the frozen low 30 m east transit.

The profile is config/profiles/real_landing_g12_launch30_low.json. It must
catch the sticks without climbing to the 30 m loft.

    PROTOTYPE_ROOT=.../Prototype .venv/bin/python sih/tests/test_launch30_low.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SIH = Path(__file__).resolve().parents[1]
TV3 = SIH.parent
os.environ.setdefault("PROTOTYPE_ROOT", str(TV3 / "sim_source"))
sys.path.insert(0, str(SIH))

from run_sih import run_once  # noqa: E402


class Launch30LowTransitTest(unittest.TestCase):
    def test_catches_without_a_high_loft(self) -> None:
        result = run_once(
            "real_landing_g12_launch30_low",
            SIH / "logs" / "real_landing_g12_launch30_low.csv",
        )
        landing = result["landing"]
        self.assertTrue(landing["success"], landing)
        self.assertEqual(landing["reason"], "caught")
        z_max = float(result["z_max_m"])
        self.assertGreater(z_max, 1.0)
        self.assertLess(z_max, 6.0, f"low transit climbed to {z_max:.2f} m")


if __name__ == "__main__":
    unittest.main()
