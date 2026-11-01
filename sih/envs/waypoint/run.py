#!/usr/bin/env python3
"""Waypoint / position-control lab: track timed/lerp/capture courses and tune.

Examples::

    python sih/envs/waypoint/run.py --profile wp_translate
    python sih/envs/waypoint/run.py --profile wp_square --plot
    python sih/envs/waypoint/run.py --profile wp_lerp_box --tune
    python sih/envs/waypoint/run.py --suite
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENV_DIR.parent))

from common import (  # noqa: E402
    SIH_ROOT,
    print_score,
    resolve_profile,
    run_sim,
    score_waypoint,
)


PROFILES = (
    "wp_hover_hold",
    "wp_translate",
    "wp_square",
    "wp_lerp_box",
    "wp_capture_gate",
)


def _run_one(profile: str, *, plot: bool, view_3d: bool, log: Path | None) -> dict:
    path = resolve_profile(profile, search=[ENV_DIR / "profiles"])
    log_path = log or (SIH_ROOT / "logs" / "envs" / f"{path.stem}.csv")
    flight_log, _sim, info = run_sim(path, log_path=log_path)
    meta = json.loads(path.read_text()).get("env") or {}
    score = score_waypoint(
        flight_log,
        xy_tol_m=float(meta.get("xy_tol_m", 0.75)),
        z_tol_m=float(meta.get("z_tol_m", 0.75)),
        eval_start_s=float(meta.get("eval_start_s", 1.0)),
        path_rmse_tol_m=float(meta.get("path_rmse_tol_m", 1.5)),
    )
    print(info["summary"])
    print_score(path.stem, score)
    print(f"  log={info['log']}")

    if plot or view_3d:
        from tvc.flight_sim import plot_flight_log
        from tvc.parameters import load_vehicle_profile
        from tvc.replay_view import replay_3d

        if plot:
            plot_flight_log(flight_log, title=f"waypoint: {path.stem}")
        if view_3d:
            system = load_vehicle_profile(profile_path=path)
            replay_3d(flight_log, config=system, title=f"waypoint: {path.stem}")

    return {"profile": path.stem, **score, "log": info["log"]}


def _tune(profile: str) -> dict:
    path = resolve_profile(profile, search=[ENV_DIR / "profiles"])
    meta = json.loads(path.read_text()).get("env") or {}
    kp_xy = meta.get("tune_kp_xy", [0.6, 0.9, 1.2, 1.6])
    kd_xy = meta.get("tune_kd_xy", [3.0, 5.0, 7.0, 10.0])
    tilts = meta.get("tune_max_tilt_deg", [8.0, 12.0, 18.0])

    results = []
    best = None
    for kp, kd, tilt in itertools.product(kp_xy, kd_xy, tilts):
        pos = {
            "kp": [float(kp), float(kp), 0.0],
            "kd": [float(kd), float(kd), 0.0],
            "ki": [0.0, 0.0, 0.0],
            "max_tilt_deg": float(tilt),
            "velocity_cascade": bool(meta.get("velocity_cascade", True)),
        }
        tag = f"{path.stem}_kp{kp:g}_kd{kd:g}_t{tilt:g}"
        log_path = SIH_ROOT / "logs" / "envs" / "tune" / f"{tag}.csv"
        flight_log, _sim, info = run_sim(
            path, log_path=log_path, position_override=pos
        )
        score = score_waypoint(
            flight_log,
            xy_tol_m=float(meta.get("xy_tol_m", 0.75)),
            z_tol_m=float(meta.get("z_tol_m", 0.75)),
            eval_start_s=float(meta.get("eval_start_s", 1.0)),
            path_rmse_tol_m=float(meta.get("path_rmse_tol_m", 1.5)),
        )
        cost = (
            2.0 * float(score["path_rmse_m"])
            + 1.5 * float(score["final_xy_m"])
            + 1.0 * float(score["final_z_m"])
            + 4.0 * float(score["sat_frac"])
            + (0.0 if score["ok"] else 15.0)
        )
        row = {
            "kp_xy": float(kp),
            "kd_xy": float(kd),
            "max_tilt_deg": float(tilt),
            "cost": float(cost),
            **{k: score[k] for k in score if k not in {"final_pos_m", "final_ref_m"}},
            "log": info["log"],
        }
        results.append(row)
        mark = "✓" if score["ok"] else "·"
        print(
            f"  {mark} kp={kp:g} kd={kd:g} tilt={tilt:g} cost={cost:.2f} "
            f"rmse={score['path_rmse_m']:.2f} final_xy={score['final_xy_m']:.2f}"
        )
        if best is None or cost < best["cost"]:
            best = row

    out = {"profile": path.stem, "best": best, "results": results}
    out_path = SIH_ROOT / "logs" / "envs" / f"tune_{path.stem}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"best: kp={best['kp_xy']} kd={best['kd_xy']} "
        f"tilt={best['max_tilt_deg']} cost={best['cost']:.2f}"
    )
    print(f"wrote {out_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Waypoint / position controller lab (SIH)")
    ap.add_argument("--profile", default="wp_translate")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--tune", action="store_true")
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
        summary = SIH_ROOT / "logs" / "envs" / "waypoint_suite.json"
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
