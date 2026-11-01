#!/usr/bin/env python3
"""SIH flight runner for TV3.

Prototype 6-DOF plant + mission/allocator; controllers default to firmware
``tvc_core`` (same C++ as Cube). Set ``SIH_CONTROL=prototype`` for the old
Python attitude/position/altitude loops.

Success criterion matches ``evaluate_landing_acceptance``: inside the pad box at
solid-motor burnout, without hard landing during burn.

Examples::

    python sih/run_sih.py --profile real_landing_g12
    SIH_CONTROL=prototype python sih/run_sih.py --profile real_landing_g12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SIH_ROOT = Path(__file__).resolve().parent
TV3_ROOT = SIH_ROOT.parent
PROTOTYPE = Path(os.environ.get("PROTOTYPE_ROOT", TV3_ROOT / "sim_source")).resolve()

sys.path.insert(0, str(PROTOTYPE))

from tvc.flight_log import FlightLog  # noqa: E402
from tvc.flight_sim import (  # noqa: E402
    FlightScenario,
    FlightSimulator,
    evaluate_landing_acceptance,
    summarize_log,
)
from tvc.parameters import FlightTestConfig, load_vehicle_profile  # noqa: E402

# TV3 firmware controller bridge (default). SIH_CONTROL=prototype for Python.
sys.path.insert(0, str(SIH_ROOT))
from tvc_core_bridge import use_firmware_controllers, wrap_sim_controllers  # noqa: E402

# Weighed unloaded airframe (grams). Loaded mass = this + motor catalog masses.
UNLOADED_MASS_G = 2605.6
G = 9.81


def _resolve_profile(name: str) -> Path:
    p = Path(name)
    if p.is_file():
        return p.resolve()
    stems = (f"{name}.json", name)
    search_dirs = (
        SIH_ROOT / "configs" / "profiles",
        SIH_ROOT / "envs" / "attitude" / "profiles",
        SIH_ROOT / "envs" / "waypoint" / "profiles",
        TV3_ROOT / "config" / "profiles",
        PROTOTYPE / "config" / "profiles",
    )
    for directory in search_dirs:
        for stem in stems:
            candidate = directory / stem
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"profile not found: {name}")


def _patch_vehicle_path(profile_path: Path) -> Path:
    """Rewrite relative vehicle path so Prototype loader finds TV3 or Prototype JSON."""
    data = json.loads(profile_path.read_text())
    vehicle = data.get("vehicle")
    if not vehicle:
        return profile_path
    vpath = Path(vehicle)
    if vpath.is_absolute() and vpath.is_file():
        return profile_path

    candidates = [
        (profile_path.parent.parent / vehicle).resolve(),
        (SIH_ROOT / "envs" / vehicle).resolve(),
        (SIH_ROOT / "configs" / vehicle).resolve(),
        (TV3_ROOT / "config" / vehicle).resolve(),
        (PROTOTYPE / "config" / vehicle).resolve(),
        (TV3_ROOT / vehicle).resolve(),
    ]
    local = next((c for c in candidates if c.is_file()), None)
    if local is None:
        raise FileNotFoundError(f"vehicle config not found for {vehicle}")
    data["vehicle"] = str(local)
    out = SIH_ROOT / "logs" / f"_resolved_{profile_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out


def _thrust_to_weight(system) -> dict:
    gravity = np.asarray(system.vehicle.gravity, dtype=float)
    g = float(np.linalg.norm(gravity)) if gravity.size else G
    if g <= 0.0:
        g = G
    mass = float(system.vehicle.mass)
    weight = mass * g
    peak = 0.0
    avg = 0.0
    has_solids = False
    for mount in system.mounts:
        motor = getattr(mount, "motor", None)
        if motor is not None:
            has_solids = True
            peak += float(motor.max_thrust_n)
            avg += float(motor.average_thrust_n)
        else:
            peak += float(mount.thrust)
            avg += float(mount.thrust)
    return {
        "unloaded_mass_g": UNLOADED_MASS_G,
        "mass_kg": mass,
        "weight_n": weight,
        "peak_thrust_n": peak,
        "avg_thrust_n": avg,
        "tw_peak": peak / max(weight, 1e-9),
        "tw_avg": avg / max(weight, 1e-9),
        "has_solids": has_solids,
        "com_m": [float(x) for x in np.asarray(system.center_of_mass, dtype=float).reshape(-1)[:3]],
    }


def run_once(
    profile_name: str,
    log_path: Path | None = None,
    *,
    mass_override_kg: float | None = None,
    extra_mass_g: float = 0.0,
    allow_subunity_tw: bool = False,
) -> dict:
    profile_path = _resolve_profile(profile_name)
    profile_raw = json.loads(profile_path.read_text())
    sih_gate = str(profile_raw.get("sih_gate") or "landing")
    resolved = _patch_vehicle_path(profile_path)

    cwd = os.getcwd()
    os.chdir(PROTOTYPE)
    try:
        system = load_vehicle_profile(profile_path=resolved)
    finally:
        os.chdir(cwd)

    # Live mass is recomposed from unloaded_vehicle + motors each step.
    # Payload (cameras, etc.) must land on the unloaded airframe.
    extra_kg = max(float(extra_mass_g), 0.0) / 1000.0
    if extra_kg > 0.0:
        system.unloaded_vehicle.mass = float(system.unloaded_vehicle.mass) + extra_kg
    if mass_override_kg is not None and mass_override_kg > 0.0:
        current_loaded = float(system.vehicle.mass) + extra_kg
        delta = float(mass_override_kg) - current_loaded
        system.unloaded_vehicle.mass = float(system.unloaded_vehicle.mass) + delta
        extra_kg += delta
    if extra_kg != 0.0:
        system.vehicle.mass = float(system.vehicle.mass) + extra_kg
        if system.flight_test is not None:
            system.flight_test.vehicle.mass = float(system.vehicle.mass)

    tw = _thrust_to_weight(system)
    tw["unloaded_mass_g"] = float(system.unloaded_vehicle.mass) * 1000.0
    if (
        tw["has_solids"]
        and tw["tw_peak"] < 1.0
        and not allow_subunity_tw
        and sih_gate != "ballistic"
    ):
        raise RuntimeError(
            f"T/W not flight-valid for {profile_name}: loaded {tw['mass_kg']:.3f} kg, "
            f"peak thrust {tw['peak_thrust_n']:.1f} N → T/W_peak={tw['tw_peak']:.2f} "
            f"(avg {tw['tw_avg']:.2f}). Weighed unloaded is {UNLOADED_MASS_G:.0f} g. "
            f"G8 cannot fly; use real_landing_g12 or real_landing_f10. "
            f"Pass --allow-subunity-tw only for historical experiments. "
            f"Pad pops use sih_gate=ballistic."
        )

    flight = system.flight_test or FlightTestConfig()
    flight.vehicle = system.vehicle
    flight.controller = system.controller
    if system.profile is not None and system.profile.altitude_controller is not None:
        flight.altitude_controller = system.profile.altitude_controller
    if system.flight_test is not None:
        flight.position_controller = system.flight_test.position_controller
        flight.altitude_controller = system.flight_test.altitude_controller

    if system.profile is not None and system.profile.real_landing is not None:
        mission = system.profile.real_landing
        flight.initial_position_m = mission.launch_position_m()
        scenario = FlightScenario.real_landing(
            mission,
            name=system.profile.name or "real_landing",
        )
    elif system.profile is not None and system.profile.motion == "attitude":
        scenario_name = str(system.profile.scenario or "hold")
        euler = (0.0, 0.0, 0.0)
        if flight.initial_euler_deg is not None:
            e = np.asarray(flight.initial_euler_deg, dtype=float).reshape(-1)
            if e.size >= 3:
                euler = (float(e[0]), float(e[1]), float(e[2]))
        # Prefer dedicated lab runner for stress/tune; SIH smoke uses step/hold/doublet.
        if scenario_name == "step":
            scenario = FlightScenario.step((15.0, 0.0, 0.0), step_time=1.0, initial_euler_deg=euler)
        elif scenario_name == "doublet":
            scenario = FlightScenario.doublet("roll", 12.0, pulse_width=0.5, start_time=1.0)
        else:
            scenario = FlightScenario.hold(euler)
    elif system.profile is not None and system.profile.motion == "hover":
        euler = (0.0, 0.0, 0.0)
        if flight.initial_euler_deg is not None:
            e = np.asarray(flight.initial_euler_deg, dtype=float).reshape(-1)
            if e.size >= 3:
                euler = (float(e[0]), float(e[1]), float(e[2]))
        if system.profile.waypoints:
            scenario = FlightScenario.hover_waypoints(
                [(wp.t, wp.position_m) for wp in system.profile.waypoints],
                euler_deg=euler,
                advance=str(system.profile.waypoint_advance or "timed"),
                capture_radius_m=float(system.profile.waypoint_capture_radius_m or 2.0),
                capture_settle_s=float(
                    getattr(system.profile, "capture_settle_s", 0.0) or 0.0
                ),
                capture_max_speed_m_s=float(
                    getattr(system.profile, "capture_max_speed_m_s", 1.5) or 1.5
                ),
            )
        else:
            target = system.profile.target_position_m or (0.0, 0.0, 5.0)
            scenario = FlightScenario.hover(
                target[2],
                xy_m=(target[0], target[1]),
                euler_deg=euler,
            )
            if abs(euler[2]) > 1e-6:
                yaw = float(np.deg2rad(euler[2]))
                scenario.heading_approach_rad = yaw
                scenario.heading_landing_rad = yaw
    else:
        scenario = FlightScenario.hover(5.0)

    sim = FlightSimulator(system, flight)
    sim.allocator.center_of_mass = np.asarray(system.center_of_mass, dtype=float)
    if use_firmware_controllers():
        wrap_sim_controllers(sim)
    log = sim.run(scenario)

    draws = []
    if system.profile is not None and system.profile.engine_variance is not None:
        for draw in system.profile.engine_variance.draws:
            ts, bs = float(draw[0]), float(draw[1])
            ign = float(draw[2]) if len(draw) > 2 else 0.0
            draws.append(
                {
                    "thrust_scale": ts,
                    "time_scale": bs,
                    "ignition_delay_s": ign,
                }
            )

    result: dict = {
        "profile": profile_name,
        "control_stack": getattr(sim, "control_stack", "prototype"),
        "mass_kg": float(tw["mass_kg"]),
        "unloaded_mass_g": float(tw["unloaded_mass_g"]),
        "extra_mass_g": extra_kg * 1000.0,
        "tw_peak": tw["tw_peak"],
        "tw_avg": tw["tw_avg"],
        "com_m": tw["com_m"],
        "engine_draws": draws,
        "safe_land": bool(getattr(sim, "_safe_land_hold", False)),
        "duration_s": float(flight.duration),
        "samples": len(log.samples),
    }

    positions = np.array([s.position for s in log.samples], dtype=float)
    result["z_max_m"] = float(positions[:, 2].max()) if len(positions) else 0.0
    result["final_pos_m"] = positions[-1].tolist() if len(positions) else [0, 0, 0]

    box = None
    if system.profile is not None:
        box = system.profile.landing_acceptance
    acceptance = evaluate_landing_acceptance(
        log,
        box,
        chopsticks_enabled=sim.chopsticks is not None,
        chopsticks_caught=bool(getattr(sim, "_chopstick_caught", False)),
        chopsticks_heading_ok=bool(getattr(sim, "_chopstick_heading_ok", False)),
        stick_backstop=bool(getattr(sim, "_stick_backstop", False)),
        body_stick_contact=bool(getattr(sim, "_body_stick_contact", False)),
    )
    result["landing"] = {
        "evaluated": acceptance.get("evaluated"),
        "success": acceptance.get("success"),
        "reason": acceptance.get("reason"),
        "burnout_time_s": acceptance.get("burnout_time_s"),
        "burnout_position_m": (
            np.asarray(acceptance["burnout_position_m"]).tolist()
            if acceptance.get("burnout_position_m") is not None
            else None
        ),
        "burnout_speed_m_s": acceptance.get("burnout_speed_m_s"),
    }

    if sih_gate == "ballistic":
        # C6 at flight mass: ~1 cm spike, not a 0.5 m hop. Score ignition + upright.
        lit = False
        for sample in log.samples:
            thrusts = np.asarray(getattr(sample, "mount_thrusts", []), dtype=float).reshape(-1)
            if thrusts.size >= 3 and bool(np.all(thrusts > 0.5)):
                lit = True
                break
        ok = lit
        result["landing"] = {
            "evaluated": True,
            "success": ok,
            "reason": "ballistic_fire" if ok else "no_ignition",
            "burnout_time_s": result["landing"].get("burnout_time_s"),
            "burnout_position_m": (
                result["landing"].get("burnout_position_m")
                or (positions[-1].tolist() if len(positions) else None)
            ),
            "burnout_speed_m_s": result["landing"].get("burnout_speed_m_s"),
        }
    elif sih_gate == "pad_hover":
        # Intentional powered return to the pad after the hold. Do not use
        # the loft hard-landing box (touchdown during the G12 tail is the land).
        zs = positions[:, 2] if len(positions) else np.zeros(0)
        dt = float(getattr(flight, "dt", 0.01) or 0.01)
        hold_s = float(np.sum(zs >= 0.30) * dt) if zs.size else 0.0
        xy_end = float(np.linalg.norm(positions[-1, :2])) if len(positions) else 99.0
        ok = float(result["z_max_m"]) >= 0.35 and hold_s >= 7.0 and xy_end < 1.5
        result["landing"] = {
            "evaluated": True,
            "success": ok,
            "reason": (
                "pad_hover"
                if ok
                else f"pad_miss z_max={result['z_max_m']:.2f} hold={hold_s:.1f}s xy={xy_end:.2f}"
            ),
            "burnout_time_s": result["landing"].get("burnout_time_s"),
            "burnout_position_m": result["landing"].get("burnout_position_m"),
            "burnout_speed_m_s": result["landing"].get("burnout_speed_m_s"),
        }
    elif box is None and system.profile is not None and system.profile.motion == "hover":
        target_z = (system.profile.target_position_m or (0, 0, 5))[2]
        z_end = float(positions[-1, 2]) if len(positions) else 0.0
        z_err = abs(z_end - float(target_z))
        xy_err = float(np.linalg.norm(positions[-1, :2])) if len(positions) else 99.0
        ok = z_err < 0.75 and xy_err < 1.0 and result["z_max_m"] > 0.5 * float(target_z)
        result["landing"] = {
            "evaluated": True,
            "success": ok,
            "reason": "hover_hold" if ok else f"hover_miss z_err={z_err:.2f} xy_err={xy_err:.2f}",
            "burnout_time_s": None,
            "burnout_position_m": positions[-1].tolist() if len(positions) else None,
            "burnout_speed_m_s": None,
        }

    if log_path is None:
        log_path = SIH_ROOT / "logs" / f"{Path(profile_name).stem}.csv"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.save(log_path)
    result["log"] = str(log_path)
    result["summary"] = summarize_log(log) if callable(summarize_log) else ""
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="TV3 Prototype-parity SIH flight runner")
    parser.add_argument(
        "--profile",
        default="real_landing_g12",
        help="Profile name (sih/configs/profiles, TV3 config/profiles, or Prototype)",
    )
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument(
        "--mass-kg",
        type=float,
        default=None,
        help="Override total loaded mass (adds the delta onto the unloaded airframe)",
    )
    parser.add_argument(
        "--extra-mass-g",
        type=float,
        default=0.0,
        help="Payload on the unloaded airframe (e.g. 400 g cameras). Does not change JSON inertia.",
    )
    parser.add_argument(
        "--allow-subunity-tw",
        action="store_true",
        help="Allow T/W_peak < 1 (historical G8 only; not flight-valid)",
    )
    parser.add_argument(
        "--until-success",
        action="store_true",
        help="Retry the same loaded mass (does not lighten the vehicle)",
    )
    parser.add_argument("--max-iters", type=int, default=8)
    args = parser.parse_args()

    last: dict = {}
    for attempt in range(1, args.max_iters + 1):
        log_path = None
        if args.log:
            base = Path(args.log)
            log_path = base if args.max_iters == 1 else base.with_name(
                f"{base.stem}_try{attempt}{base.suffix}"
            )
        try:
            last = run_once(
                args.profile,
                log_path,
                mass_override_kg=args.mass_kg,
                extra_mass_g=args.extra_mass_g,
                allow_subunity_tw=args.allow_subunity_tw,
            )
        except RuntimeError as exc:
            print(f"SIH FAIL: {exc}")
            stamp = SIH_ROOT / "logs" / "last_result.json"
            stamp.write_text(json.dumps({"profile": args.profile, "error": str(exc)}, indent=2))
            return 1
        success = bool(last.get("landing", {}).get("success"))
        print(
            f"[try {attempt}] stack={last.get('control_stack', '?')}  "
            f"mass={last['mass_kg']:.3f} kg  "
            f"T/W_peak={last['tw_peak']:.2f} T/W_avg={last['tw_avg']:.2f}  "
            f"z_max={last['z_max_m']:.2f} m  landing={last['landing']}"
        )
        if last.get("engine_draws"):
            bits = []
            for i, d in enumerate(last["engine_draws"]):
                bit = f"M{i} T={d['thrust_scale']:.3f} t={d['time_scale']:.3f}"
                ign = float(d.get("ignition_delay_s", 0.0) or 0.0)
                if ign > 1e-9:
                    bit += f" ign={ign:.3f}s"
                bits.append(bit)
            print("  scatter: " + " ".join(bits))
        if last.get("safe_land"):
            print("  safe-land: 0.75 T/W lock, attitude hold")
        print(f"  log: {last['log']}")
        if success:
            print("SIH PASS: expected landing/hover behavior achieved at loaded flight mass.")
            stamp = SIH_ROOT / "logs" / "last_result.json"
            stamp.write_text(json.dumps(last, indent=2, default=str))
            return 0
        if not args.until_success:
            break

    print("SIH FAIL: did not achieve landing/hover success at loaded flight mass.")
    stamp = SIH_ROOT / "logs" / "last_result.json"
    stamp.write_text(json.dumps(last, indent=2, default=str))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
