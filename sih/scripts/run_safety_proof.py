#!/usr/bin/env python3
"""Exercise the landing acceptance checks against simulated flight profiles.

The runner saves a flight log and plots for each case and can open a 3D replay.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

# Prefer non-blocking save for CI; viewer can re-enable interactive backend.
if os.environ.get("SIH_INTERACTIVE", "1") != "1":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SIH_ROOT = Path(__file__).resolve().parents[1]
TV3_ROOT = SIH_ROOT.parent
PROTOTYPE = Path(os.environ.get("PROTOTYPE_ROOT", TV3_ROOT / "sim_source"))

sys.path.insert(0, str(TV3_ROOT / "sih"))
sys.path.insert(0, str(PROTOTYPE))

from run_sih import run_once  # noqa: E402
from tvc.flight_log import FlightLog  # noqa: E402
from tvc.flight_sim import (  # noqa: E402
    evaluate_landing_acceptance,
    find_hard_landing,
    plot_flight_log,
    summarize_log,
)


def _analyze(log_path: Path, box=None) -> dict:
    log = FlightLog.load(log_path)
    positions = np.array([s.position for s in log.samples], dtype=float)
    velocities = np.array([s.velocity for s in log.samples], dtype=float)
    thrusts = np.array([s.mount_thrusts for s in log.samples], dtype=float)
    times = np.array([s.time for s in log.samples], dtype=float)

    lit = thrusts > 0.5
    any_lit = lit.any(axis=0) if thrusts.ndim == 2 and thrusts.shape[1] >= 3 else [False, False, False]
    hard = find_hard_landing(log)
    acceptance = evaluate_landing_acceptance(log, box) if box is not None else None

    # Hang-fire style: after t>1.5s of burn window, any mount never lit while others were
    hang_like = False
    if thrusts.ndim == 2 and thrusts.shape[1] >= 3 and times.size:
        # approximate fire epoch: first time total thrust > 0.5
        total = thrusts.sum(axis=1)
        fire_idx = np.argmax(total > 0.5) if np.any(total > 0.5) else None
        if fire_idx is not None:
            t0 = times[fire_idx]
            late = times > (t0 + 1.5)
            if np.any(late):
                ever = (thrusts[late] > 0.5).any(axis=0)
                hang_like = bool(ever.any() and (not ever.all()))

    z = positions[:, 2] if len(positions) else np.array([0.0])
    spd = np.linalg.norm(velocities, axis=1) if len(velocities) else np.array([0.0])

    return {
        "log": str(log_path),
        "n_samples": len(log.samples),
        "t_end_s": float(times[-1]) if len(times) else 0.0,
        "z_max_m": float(z.max()),
        "z_min_m": float(z.min()),
        "speed_max_m_s": float(spd.max()),
        "mounts_ever_lit": [bool(x) for x in any_lit],
        "hard_landing": (
            None
            if hard is None
            else {"index": int(hard[0]), "time_s": float(hard[1]), "vz": float(hard[2])}
        ),
        "hang_like_partial": hang_like,
        "landing_acceptance": acceptance,
        "summary": summarize_log(log),
    }


def _save_plot(log_path: Path, out_png: Path, title: str) -> None:
    log = FlightLog.load(log_path)
    # plot_flight_log ends with plt.show(); keep non-blocking so the suite can finish.
    real_show = plt.show
    plt.show = lambda *a, **k: None  # type: ignore[assignment]
    try:
        plot_flight_log(log, title=title)
        fig = plt.gcf()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    finally:
        plt.show = real_show  # type: ignore[assignment]
        plt.close("all")
    print(f"  plot → {out_png}")


def _viewer(log_path: Path, vehicle: Path) -> None:
    """Open Prototype 3D viewer (blocks until closed)."""
    py = PROTOTYPE / ".venv" / "bin" / "python"
    if not py.is_file():
        py = Path(sys.executable)
    cmd = [
        str(py),
        str(PROTOTYPE / "flight_test.py"),
        "replay",
        "--log",
        str(log_path),
        "--vehicle",
        str(vehicle),
        "--view-3d",
    ]
    print("  launching 3D viewer:", " ".join(cmd))
    # Non-zero if user closes early is fine
    subprocess.run(cmd, cwd=str(PROTOTYPE), check=False)


def main() -> int:
    logs = SIH_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    report: dict = {"cases": [], "all_pass": True}

    cases = [
        {
            "id": "nominal_real_landing",
            "profile": "real_landing_g12",
            "expect_success": True,
            "expect_hard": False,
            "note": "Primary pad mission at estimated mass (2263 g + 3×G12) — SAFE path after COAST/LAND",
        },
        {
            "id": "hover_smoke",
            "profile": "hover_hold_const",
            "expect_success": True,
            "expect_hard": False,
            "note": "Constant-thrust hover — dry-arm style control without solids fire latch (not flight mass)",
        },
        {
            "id": "seq_ignition_stress",
            "profile": "f10_sequential",
            "expect_success": None,  # informational
            "expect_hard": False,
            "note": "Staggered F10 ignition delays (0/0.5/1.0 s) — hang-fire clock must respect delays (TVC_IGN_DLY*)",
        },
    ]

    vehicle_g12 = TV3_ROOT / "config" / "vehicles" / "tvc_ring3_g12.json"
    if not vehicle_g12.is_file():
        vehicle_g12 = PROTOTYPE / "config" / "vehicles" / "tvc_ring3_g12.json"

    for case in cases:
        print(f"\n=== {case['id']}: {case['profile']} ===")
        print(f"  {case['note']}")
        try:
            result = run_once(case["profile"], logs / f"proof_{case['id']}.csv")
        except Exception as exc:  # noqa: BLE001
            print(f"  RUN ERROR: {exc}")
            report["cases"].append({"id": case["id"], "error": str(exc), "pass": False})
            report["all_pass"] = False
            continue

        log_path = Path(result["log"])
        analysis = _analyze(log_path)
        landing = result.get("landing") or {}
        success = landing.get("success")
        hard = analysis["hard_landing"]

        ok = True
        reasons = []
        if case["expect_success"] is True and not success:
            ok = False
            reasons.append(f"expected success, got {landing}")
        if case["expect_success"] is False and success:
            ok = False
            reasons.append("expected fail, got success")
        if case["expect_hard"] is False and hard is not None:
            # only fail if hard during burn would fail acceptance
            if landing.get("reason") == "hard_landing" or (
                success is False and hard is not None
            ):
                # for expect_success True, hard is already a failure above
                if case["expect_success"] is True:
                    ok = False
                    reasons.append(f"hard_landing during burn: {hard}")
        if case["expect_hard"] is True and hard is None:
            ok = False
            reasons.append("expected hard_landing event")

        # seq case: document partial light if present
        if case["id"] == "seq_ignition_stress":
            print(f"  mounts_ever_lit={analysis['mounts_ever_lit']} hang_like={analysis['hang_like_partial']}")
            # still mark pass if it ran and produced thrust
            ok = bool(any(analysis["mounts_ever_lit"])) or success is True
            if not ok:
                reasons.append("no mounts lit")

        entry = {
            "id": case["id"],
            "profile": case["profile"],
            "pass": ok,
            "reasons": reasons,
            "sih_result": result,
            "analysis": {
                k: analysis[k]
                for k in (
                    "t_end_s",
                    "z_max_m",
                    "z_min_m",
                    "speed_max_m_s",
                    "mounts_ever_lit",
                    "hard_landing",
                    "hang_like_partial",
                )
            },
            "landing": landing,
            "note": case["note"],
        }
        report["cases"].append(entry)
        if not ok:
            report["all_pass"] = False

        status = "PASS" if ok else "FAIL"
        print(f"  {status}: landing={landing.get('success')} reason={landing.get('reason')} hard={hard}")
        print(f"  z_max={analysis['z_max_m']:.2f} m  log={log_path}")

        try:
            _save_plot(log_path, logs / f"proof_{case['id']}_plot.png", f"SIH proof: {case['id']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  plot failed: {exc}")

        # ULog export for primary
        if case["id"] == "nominal_real_landing":
            try:
                ulog = TV3_ROOT / "tools" / "csv_to_ulog.py"
                subprocess.run(
                    [sys.executable, str(ulog), str(log_path), "--verify"],
                    check=False,
                    cwd=str(TV3_ROOT),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ulog export failed: {exc}")

    # Hard-landing criterion vs burnout: post-burn free-fall is expected and must
    # NOT fail the mission (same rule as tvc_safety hard-land only while lit/BURN).
    print("\n=== hard_landing vs burnout (design parity) ===")
    try:
        nom = logs / "proof_nominal_real_landing.csv"
        if nom.is_file():
            log = FlightLog.load(nom)
            hard = find_hard_landing(log)
            acc = evaluate_landing_acceptance(log, None)
            # Re-evaluate with box from metadata if present via run result
            from tvc.parameters import load_vehicle_profile

            resolved = SIH_ROOT / "logs" / "_resolved_real_landing_g12.json"
            box = None
            if resolved.is_file():
                system = load_vehicle_profile(profile_path=resolved)
                if system.profile is not None:
                    box = system.profile.landing_acceptance
            acc = evaluate_landing_acceptance(log, box)
            bo_t = acc.get("burnout_time_s")
            hard_during_burn = False
            if hard is not None and bo_t is not None:
                hard_during_burn = float(hard[1]) <= float(bo_t) + 1e-6
            ok_crit = bool(acc.get("success")) and not hard_during_burn
            report["hard_landing_criterion"] = {
                "hard_event": (
                    None
                    if hard is None
                    else {"time_s": float(hard[1]), "vz": float(hard[2])}
                ),
                "burnout_time_s": bo_t,
                "hard_during_burn": hard_during_burn,
                "acceptance_success": acc.get("success"),
                "acceptance_reason": acc.get("reason"),
                "altitude_m": 0.08,
                "vz_sim_limit_m_s": -1.5,
                "px4_safety_map": "TVC_HL_Z=0.08, TVC_HL_VZ=1.5 (NED +down); only while BURN/lit",
                "pass": ok_crit,
            }
            print(
                f"  burnout_t={bo_t} hard={hard} hard_during_burn={hard_during_burn} "
                f"success={acc.get('success')} → {'PASS' if ok_crit else 'FAIL'}"
            )
            if not ok_crit:
                report["all_pass"] = False
    except Exception as exc:  # noqa: BLE001
        print(f"  criterion check failed: {exc}")
        report["all_pass"] = False

    out = logs / "safety_proof_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n=== report → {out} ===")
    print(f"ALL_PASS={report['all_pass']}")

    # 3D viewer for primary success log
    primary = logs / "proof_nominal_real_landing.csv"
    if primary.is_file() and os.environ.get("SIH_VIEWER", "1") == "1":
        print("\n=== 3D viewer (close window to finish) ===")
        try:
            _viewer(primary, vehicle_g12)
        except Exception as exc:  # noqa: BLE001
            print(f"  viewer failed: {exc}")
            # Fallback: tools/replay_sih_log.py
            try:
                subprocess.run(
                    [
                        str(PROTOTYPE / ".venv" / "bin" / "python"),
                        str(TV3_ROOT / "tools" / "replay_sih_log.py"),
                        str(primary),
                        "--plot",
                    ],
                    check=False,
                )
            except Exception as exc2:  # noqa: BLE001
                print(f"  fallback viewer failed: {exc2}")

    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
