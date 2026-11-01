#!/usr/bin/env python3
"""Attitude-control lab: stress tracking and gain sweeps (const-thrust plant).

Examples::

    python sih/envs/attitude/run.py --profile att_step_roll
    python sih/envs/attitude/run.py --profile att_stress --plot
    python sih/envs/attitude/run.py --profile att_doublet_yaw --tune
    python sih/envs/attitude/run.py --suite
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ENV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENV_DIR.parent))

from common import (  # noqa: E402
    SIH_ROOT,
    print_score,
    resolve_profile,
    run_sim,
    score_attitude,
)


PROFILES = (
    "att_step_roll",
    "att_step_pitch",
    "att_step_yaw",
    "att_doublet_roll",
    "att_doublet_pitch",
    "att_doublet_yaw",
    "att_stress",
    "att_a1_force_hold",
)


def _run_one(profile: str, *, plot: bool, view_3d: bool, log: Path | None) -> dict:
    path = resolve_profile(profile, search=[ENV_DIR / "profiles"])
    log_path = log or (SIH_ROOT / "logs" / "envs" / f"{path.stem}.csv")
    flight_log, _sim, info = run_sim(path, log_path=log_path)
    meta = json.loads(path.read_text()).get("env") or {}
    score = score_attitude(
        flight_log,
        settle_tol_deg=float(meta.get("settle_tol_deg", 3.0)),
        eval_start_s=float(meta.get("step_time_s", 1.0)),
    )
    print(info["summary"])
    print_score(path.stem, score)
    print(f"  log={info['log']}")

    if plot or view_3d:
        from tvc.flight_sim import plot_flight_log
        from tvc.parameters import load_vehicle_profile
        from tvc.replay_view import replay_3d

        if plot:
            plot_flight_log(flight_log, title=f"attitude: {path.stem}")
        if view_3d:
            system = load_vehicle_profile(profile_path=path)
            replay_3d(flight_log, config=system, title=f"attitude: {path.stem}")

    return {"profile": path.stem, **score, "log": info["log"]}


def _tune(profile: str) -> dict:
    """Coarse grid on roll/pitch kp,kd (yaw scaled)."""
    path = resolve_profile(profile, search=[ENV_DIR / "profiles"])
    meta = json.loads(path.read_text()).get("env") or {}
    kp_xy = meta.get("tune_kp_xy", [12.0, 18.0, 22.0, 28.0])
    kd_xy = meta.get("tune_kd_xy", [4.0, 6.0, 9.0, 12.0])
    yaw_scale = float(meta.get("tune_yaw_scale", 0.15))

    results = []
    best = None
    for kp, kd in itertools.product(kp_xy, kd_xy):
        att = {
            "kp": [float(kp), float(kp), float(kp) * yaw_scale],
            "kd": [float(kd), float(kd), float(kd) * yaw_scale],
            "ki": [0.0, 0.0, 0.0],
        }
        tag = f"{path.stem}_kp{kp:g}_kd{kd:g}"
        log_path = SIH_ROOT / "logs" / "envs" / "tune" / f"{tag}.csv"
        flight_log, _sim, info = run_sim(
            path, log_path=log_path, attitude_override=att
        )
        score = score_attitude(
            flight_log,
            settle_tol_deg=float(meta.get("settle_tol_deg", 3.0)),
            eval_start_s=float(meta.get("step_time_s", 1.0)),
        )
        # Lower is better.
        settle = score["settle_s"] if score["settle_s"] is not None else 9.0
        cost = (
            3.0 * settle
            + 0.15 * float(score["peak_err_deg"])
            + 2.0 * float(score["rmse_deg"])
            + 8.0 * float(score["sat_frac"])
            + (0.0 if score["ok"] else 20.0)
        )
        row = {
            "kp_xy": float(kp),
            "kd_xy": float(kd),
            "cost": float(cost),
            **{k: score[k] for k in score if k != "axis_final_deg"},
            "log": info["log"],
        }
        results.append(row)
        mark = "✓" if score["ok"] else "·"
        print(
            f"  {mark} kp={kp:g} kd={kd:g} cost={cost:.2f} "
            f"settle={settle:.2f}s rmse={score['rmse_deg']:.2f}° sat={score['sat_frac']:.2f}"
        )
        if best is None or cost < best["cost"]:
            best = row

    out = {
        "profile": path.stem,
        "best": best,
        "results": results,
    }
    out_path = SIH_ROOT / "logs" / "envs" / f"tune_{path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"best: kp_xy={best['kp_xy']} kd_xy={best['kd_xy']} cost={best['cost']:.2f}")
    print(f"wrote {out_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Attitude controller lab (SIH)")
    ap.add_argument("--profile", default="att_step_roll")
    ap.add_argument("--suite", action="store_true", help="Run all stock attitude profiles")
    ap.add_argument("--tune", action="store_true", help="Grid-search attitude gains")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--view-3d", action="store_true")
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    if args.suite:
        rows = []
        for name in PROFILES:
            print(f"\n=== {name} ===")
            rows.append(_run_one(name, plot=False, view_3d=False, log=None))
        n_ok = sum(1 for r in rows if r.get("ok"))
        print(f"\nsuite: {n_ok}/{len(rows)} PASS")
        summary = SIH_ROOT / "logs" / "envs" / "attitude_suite.json"
        summary.write_text(json.dumps(rows, indent=2))
        print(f"wrote {summary}")
        return 0 if n_ok == len(rows) else 1

    if args.tune:
        _tune(args.profile)
        return 0

    result = _run_one(
        args.profile, plot=args.plot, view_3d=args.view_3d, log=args.log
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
