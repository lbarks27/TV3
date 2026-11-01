#!/usr/bin/env python3
"""Replay a SIH / Prototype CSV log with the Prototype 3D viewer.

Examples::

    python tools/replay_sih_log.py sih/logs/real_landing_g12.csv
    python tools/replay_sih_log.py sih/logs/real_landing_g12.csv --plot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TV3 = Path(__file__).resolve().parents[1]
PROTOTYPE = Path(os.environ.get("PROTOTYPE_ROOT", TV3 / "sim_source"))
sys.path.insert(0, str(PROTOTYPE))


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay SIH log with Prototype 3D viewer")
    ap.add_argument("log", type=Path, help="CSV flight log")
    ap.add_argument(
        "--vehicle",
        type=Path,
        default=TV3 / "config/vehicles/tvc_ring3_g12.json",
        help="Vehicle JSON for mount geometry",
    )
    ap.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional profile JSON",
    )
    ap.add_argument("--plot", action="store_true", help="Also open 2D plot panels")
    args = ap.parse_args()

    from tvc.flight_log import FlightLog
    from tvc.flight_sim import plot_flight_log, summarize_log
    from tvc.parameters import load_vehicle_profile
    from tvc.replay_view import replay_3d

    log = FlightLog.load(args.log)
    print(summarize_log(log))

    system = load_vehicle_profile(
        vehicle_path=args.vehicle if args.vehicle else None,
        profile_path=args.profile if args.profile else None,
    )

    if args.plot:
        plot_flight_log(log, title=f"SIH: {args.log.name}")

    replay_3d(log, config=system, title=f"3D replay: {args.log.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
