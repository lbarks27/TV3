"""Shared helpers for SIH controller lab environments (attitude + waypoint)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ENVS_ROOT = Path(__file__).resolve().parent
SIH_ROOT = ENVS_ROOT.parent
TV3_ROOT = SIH_ROOT.parent
PROTOTYPE = Path(os.environ.get("PROTOTYPE_ROOT", TV3_ROOT / "sim_source")).resolve()

if str(PROTOTYPE) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE))

from tvc.attitude import (  # noqa: E402
    normalize_quaternion,
    quaternion_error_vector,
    quaternion_from_euler,
)
from tvc.controller import PositionPIDConfig  # noqa: E402
from tvc.flight_log import FlightLog  # noqa: E402
from tvc.flight_sim import FlightScenario, FlightSimulator, summarize_log  # noqa: E402
from tvc.parameters import FlightTestConfig, load_vehicle_profile  # noqa: E402

sys.path.insert(0, str(SIH_ROOT))
from tvc_core_bridge import use_firmware_controllers, wrap_sim_controllers  # noqa: E402


def resolve_profile(name: str | Path, *, search: list[Path] | None = None) -> Path:
    p = Path(name)
    if p.is_file():
        return p.resolve()
    stems = (f"{name}.json", name)
    dirs = search or [
        ENVS_ROOT / "attitude" / "profiles",
        ENVS_ROOT / "waypoint" / "profiles",
        SIH_ROOT / "configs" / "profiles",
        TV3_ROOT / "config" / "profiles",
    ]
    for directory in dirs:
        for stem in stems:
            candidate = directory / stem
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"profile not found: {name}")


def patch_vehicle_path(profile_path: Path) -> Path:
    data = json.loads(profile_path.read_text())
    vehicle = data.get("vehicle")
    if not vehicle:
        return profile_path
    vpath = Path(vehicle)
    if vpath.is_absolute() and vpath.is_file():
        return profile_path
    candidates = [
        (profile_path.parent.parent / vehicle).resolve(),  # envs/vehicles/...
        (ENVS_ROOT / vehicle).resolve(),
        (SIH_ROOT / "configs" / vehicle).resolve(),
        (TV3_ROOT / "config" / vehicle).resolve(),
        (PROTOTYPE / "config" / vehicle).resolve(),
    ]
    local = next((c for c in candidates if c.is_file()), None)
    if local is None:
        raise FileNotFoundError(f"vehicle config not found for {vehicle}")
    data["vehicle"] = str(local)
    out = SIH_ROOT / "logs" / "envs" / f"_resolved_{profile_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out


def load_system(profile_path: Path):
    resolved = patch_vehicle_path(profile_path)
    cwd = os.getcwd()
    os.chdir(PROTOTYPE)
    try:
        return load_vehicle_profile(profile_path=resolved)
    finally:
        os.chdir(cwd)


def apply_controller_overrides(
    system,
    *,
    attitude: dict[str, Any] | None = None,
    position: dict[str, Any] | None = None,
) -> None:
    """Patch live PID configs on a loaded SystemConfig / FlightTestConfig."""
    if attitude:
        cfg = system.controller
        if "kp" in attitude:
            cfg.kp = np.asarray(attitude["kp"], dtype=float)
        if "kd" in attitude:
            cfg.kd = np.asarray(attitude["kd"], dtype=float)
        if "ki" in attitude:
            cfg.ki = np.asarray(attitude["ki"], dtype=float)
        if "torque_limit" in attitude:
            cfg.torque_limit = float(attitude["torque_limit"])
        if "integral_limit" in attitude:
            cfg.integral_limit = float(attitude["integral_limit"])
        if system.flight_test is not None:
            system.flight_test.controller = cfg

    if position and system.flight_test is not None:
        pc = system.flight_test.position_controller or PositionPIDConfig()
        if "kp" in position:
            pc.kp = np.asarray(position["kp"], dtype=float)
        if "kd" in position:
            pc.kd = np.asarray(position["kd"], dtype=float)
        if "ki" in position:
            pc.ki = np.asarray(position["ki"], dtype=float)
        if "max_tilt_deg" in position:
            pc.max_tilt_deg = float(position["max_tilt_deg"])
        if "velocity_cascade" in position:
            pc.velocity_cascade = bool(position["velocity_cascade"])
        if "lateral_priority_m" in position:
            pc.lateral_priority_m = float(position["lateral_priority_m"])
        if "integral_limit" in position:
            pc.integral_limit = float(position["integral_limit"])
        system.flight_test.position_controller = pc


def build_scenario(system) -> FlightScenario:
    """Mirror Prototype flight_test._build_scenario for profile-driven labs."""
    profile = system.profile
    flight = system.flight_test or FlightTestConfig()
    euler = (0.0, 0.0, 0.0)
    if flight.initial_euler_deg is not None:
        e = np.asarray(flight.initial_euler_deg, dtype=float).reshape(-1)
        if e.size >= 3:
            euler = (float(e[0]), float(e[1]), float(e[2]))

    motion = "hover" if profile is None else str(profile.motion or "hover")
    raw = getattr(profile, "_raw", None)  # unused; read from file fields via profile

    if motion == "attitude":
        scenario = str(getattr(profile, "scenario", "hold") or "hold")
        # Optional profile knobs via JSON extras on flight_test / notes — use
        # dedicated builders from attitude.env_meta if present.
        meta = _profile_meta(system)
        if scenario == "hold":
            return FlightScenario.hold(tuple(meta.get("hold_euler_deg", euler)))
        if scenario == "step":
            target = meta.get("step_euler_deg", [15.0, 0.0, 0.0])
            return FlightScenario.step(
                (float(target[0]), float(target[1]), float(target[2])),
                step_time=float(meta.get("step_time_s", 1.0)),
                initial_euler_deg=euler,
            )
        if scenario == "doublet":
            return FlightScenario.doublet(
                str(meta.get("doublet_axis", "roll")),
                float(meta.get("doublet_deg", 12.0)),
                pulse_width=float(meta.get("doublet_width_s", 0.5)),
                start_time=float(meta.get("step_time_s", 1.0)),
            )
        if scenario == "stress":
            return _attitude_stress_scenario(meta)
        raise ValueError(f"unsupported attitude scenario={scenario!r}")

    if motion == "position":
        target = profile.target_position_m if profile is not None else (2.0, 0.0, 5.0)
        meta = _profile_meta(system)
        return FlightScenario.position_step(
            (float(target[0]), float(target[1]), float(target[2])),
            step_time=float(meta.get("step_time_s", 1.0)),
        )

    if motion == "hover":
        if profile is not None and profile.waypoints:
            return FlightScenario.hover_waypoints(
                [(wp.t, wp.position_m) for wp in profile.waypoints],
                euler_deg=euler,
                advance=str(profile.waypoint_advance or "timed"),
                capture_radius_m=float(profile.waypoint_capture_radius_m or 2.0),
                capture_settle_s=float(getattr(profile, "capture_settle_s", 0.0) or 0.0),
                capture_max_speed_m_s=float(
                    getattr(profile, "capture_max_speed_m_s", 1.5) or 1.5
                ),
            )
        target = (profile.target_position_m if profile is not None else None) or (
            0.0,
            0.0,
            5.0,
        )
        return FlightScenario.hover(
            float(target[2]),
            xy_m=(float(target[0]), float(target[1])),
            euler_deg=euler,
        )

    raise ValueError(f"unsupported motion={motion!r}")


def _profile_meta(system) -> dict[str, Any]:
    """Optional lab knobs stored under profile JSON key ``env``."""
    # Re-read resolved profile path from flight notes is awkward; stash on load.
    meta = getattr(system, "_env_meta", None)
    return dict(meta or {})


def attach_env_meta(system, profile_path: Path) -> None:
    data = json.loads(profile_path.read_text())
    system._env_meta = dict(data.get("env") or {})


def _attitude_stress_scenario(meta: dict[str, Any]) -> FlightScenario:
    """Roll → pitch → yaw steps with dwell, then return to level."""
    amp = float(meta.get("stress_amp_deg", 12.0))
    dwell = float(meta.get("stress_dwell_s", 2.0))
    t0 = float(meta.get("step_time_s", 1.0))
    # schedule: level, +roll, level, +pitch, level, +yaw, level
    segs = [
        (t0, (0.0, 0.0, 0.0)),
        (t0 + dwell, (amp, 0.0, 0.0)),
        (t0 + 2 * dwell, (0.0, 0.0, 0.0)),
        (t0 + 3 * dwell, (0.0, amp, 0.0)),
        (t0 + 4 * dwell, (0.0, 0.0, 0.0)),
        (t0 + 5 * dwell, (0.0, 0.0, amp)),
        (t0 + 6 * dwell, (0.0, 0.0, 0.0)),
    ]

    def attitude_reference_fn(time: float) -> np.ndarray:
        euler = (0.0, 0.0, 0.0)
        for t_switch, eul in segs:
            if time >= t_switch:
                euler = eul
            else:
                break
        return quaternion_from_euler(*np.deg2rad(euler))

    return FlightScenario(
        name="stress",
        motion_mode="attitude",
        attitude_reference_fn=attitude_reference_fn,
        freeze_translation=True,
    )


def run_sim(
    profile_path: Path,
    *,
    log_path: Path | None = None,
    attitude_override: dict[str, Any] | None = None,
    position_override: dict[str, Any] | None = None,
) -> tuple[FlightLog, FlightSimulator, dict[str, Any]]:
    system = load_system(profile_path)
    attach_env_meta(system, profile_path)
    apply_controller_overrides(
        system, attitude=attitude_override, position=position_override
    )

    flight = system.flight_test or FlightTestConfig()
    flight.vehicle = system.vehicle
    flight.controller = system.controller
    if system.profile is not None and system.profile.altitude_controller is not None:
        flight.altitude_controller = system.profile.altitude_controller
    if system.flight_test is not None:
        flight.position_controller = system.flight_test.position_controller
        flight.altitude_controller = system.flight_test.altitude_controller

    scenario = build_scenario(system)
    sim = FlightSimulator(system, flight)
    sim.allocator.center_of_mass = np.asarray(system.center_of_mass, dtype=float)
    if use_firmware_controllers():
        wrap_sim_controllers(sim)

    meta = _profile_meta(system)
    schedule = meta.get("force_schedule")
    if schedule:
        knots = sorted(
            (float(item["t"]), float(item["force_n"])) for item in schedule
        )

        def _force_at(time: float) -> float:
            value = knots[0][1]
            for t_switch, force_n in knots:
                if time >= t_switch:
                    value = force_n
                else:
                    break
            return value

        _orig_step = sim.step

        def _step_with_force(state, scenario, time):
            sim.flight.force_target = _force_at(float(time))
            return _orig_step(state, scenario, time)

        sim.step = _step_with_force  # type: ignore[method-assign]

    log = sim.run(scenario)

    if log_path is None:
        log_path = SIH_ROOT / "logs" / "envs" / f"{profile_path.stem}.csv"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.save(log_path)

    info = {
        "profile": profile_path.stem,
        "log": str(log_path),
        "summary": summarize_log(log),
        "samples": len(log.samples),
        "duration_s": float(flight.duration),
        "motion": str(getattr(system.profile, "motion", "")),
        "scenario": str(getattr(system.profile, "scenario", "")),
    }
    return log, sim, info


def _sample_att_err_deg(sample) -> np.ndarray:
    # Prefer logged attitude error vector when present.
    err = getattr(sample, "attitude_error", None)
    if err is not None:
        e = np.asarray(err, dtype=float).reshape(-1)
        if e.size >= 3:
            return np.rad2deg(e[:3])
    q = normalize_quaternion(np.asarray(sample.quaternion, dtype=float))
    q_ref = normalize_quaternion(np.asarray(sample.quaternion_reference, dtype=float))
    return np.rad2deg(quaternion_error_vector(q, q_ref))


def score_attitude(
    log: FlightLog,
    *,
    settle_tol_deg: float = 3.0,
    eval_start_s: float = 1.0,
) -> dict[str, Any]:
    if not log.samples:
        return {"ok": False, "reason": "empty_log"}

    times = np.array([float(s.time) for s in log.samples], dtype=float)
    errs = np.vstack([_sample_att_err_deg(s) for s in log.samples])
    err_norm = np.linalg.norm(errs, axis=1)
    sat = np.array(
        [bool(getattr(s, "saturated", False)) for s in log.samples], dtype=bool
    )
    omega = np.vstack([np.asarray(s.omega, dtype=float).reshape(3) for s in log.samples])
    tau = np.vstack(
        [
            np.asarray(getattr(s, "torque_command", np.zeros(3)), dtype=float).reshape(3)
            for s in log.samples
        ]
    )

    mask = times >= float(eval_start_s)
    if not np.any(mask):
        mask = np.ones_like(times, dtype=bool)

    # Settling: first time after eval_start that |e| stays below tol for 0.5 s.
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    hold_n = max(1, int(round(0.5 / max(dt, 1e-6))))
    settle_s = None
    for i in np.where(mask)[0]:
        if i + hold_n >= len(err_norm):
            break
        if float(np.max(err_norm[i : i + hold_n])) <= settle_tol_deg:
            settle_s = float(times[i] - eval_start_s)
            break

    post = err_norm[mask]
    peak = float(np.max(post)) if post.size else 99.0
    final = float(err_norm[-1])
    rmse = float(np.sqrt(np.mean(np.square(post)))) if post.size else 99.0
    sat_frac = float(np.mean(sat[mask])) if np.any(mask) else 1.0
    # Overshoot vs first commanded step: peak / |err at step|
    step_i = int(np.argmin(np.abs(times - eval_start_s)))
    e0 = float(err_norm[min(step_i + 1, len(err_norm) - 1)])
    overshoot = float(max(0.0, (peak - abs(e0)) / max(abs(e0), 1e-3)))

    ok = (
        settle_s is not None
        and settle_s <= 2.5
        and final <= settle_tol_deg
        and sat_frac <= 0.35
    )
    return {
        "ok": bool(ok),
        "reason": "attitude_track" if ok else "attitude_miss",
        "settle_s": settle_s,
        "final_err_deg": final,
        "peak_err_deg": peak,
        "rmse_deg": rmse,
        "overshoot_frac": overshoot,
        "sat_frac": sat_frac,
        "peak_omega_rad_s": float(np.max(np.linalg.norm(omega, axis=1))),
        "peak_tau_nm": float(np.max(np.linalg.norm(tau, axis=1))),
        "axis_final_deg": errs[-1].tolist(),
    }


def score_waypoint(
    log: FlightLog,
    *,
    xy_tol_m: float = 0.75,
    z_tol_m: float = 0.75,
    eval_start_s: float = 1.0,
    path_rmse_tol_m: float = 1.5,
) -> dict[str, Any]:
    if not log.samples:
        return {"ok": False, "reason": "empty_log"}

    times = np.array([float(s.time) for s in log.samples], dtype=float)
    pos = np.vstack([np.asarray(s.position, dtype=float).reshape(3) for s in log.samples])
    pref = np.vstack(
        [
            np.asarray(getattr(s, "position_reference", s.position), dtype=float).reshape(3)
            for s in log.samples
        ]
    )
    # NaN refs (vertical hold style) → skip those samples.
    finite = np.isfinite(pref).all(axis=1)
    mask = (times >= float(eval_start_s)) & finite
    if not np.any(mask):
        mask = finite
    if not np.any(mask):
        return {"ok": False, "reason": "no_position_ref"}

    err = pos[mask] - pref[mask]
    err[:, 2] = pos[mask, 2] - pref[mask, 2]
    xy_err = np.linalg.norm(err[:, :2], axis=1)
    z_err = np.abs(err[:, 2])
    path_rmse = float(np.sqrt(np.mean(np.sum(np.square(err), axis=1))))
    xy_rmse = float(np.sqrt(np.mean(np.square(xy_err))))
    z_rmse = float(np.sqrt(np.mean(np.square(z_err))))
    peak_xy = float(np.max(xy_err))
    final_xy = float(np.linalg.norm(pos[-1, :2] - pref[-1, :2])) if finite[-1] else 99.0
    final_z = float(abs(pos[-1, 2] - pref[-1, 2])) if finite[-1] else 99.0
    sat = np.array(
        [bool(getattr(s, "saturated", False)) for s in log.samples], dtype=bool
    )
    sat_frac = float(np.mean(sat[mask]))

    ok = (
        final_xy <= xy_tol_m
        and final_z <= z_tol_m
        and path_rmse <= float(path_rmse_tol_m)
    )
    return {
        "ok": bool(ok),
        "reason": "waypoint_track" if ok else "waypoint_miss",
        "path_rmse_m": path_rmse,
        "xy_rmse_m": xy_rmse,
        "z_rmse_m": z_rmse,
        "peak_xy_m": peak_xy,
        "final_xy_m": final_xy,
        "final_z_m": final_z,
        "sat_frac": sat_frac,
        "final_pos_m": pos[-1].tolist(),
        "final_ref_m": pref[-1].tolist() if finite[-1] else None,
    }


def print_score(title: str, score: dict[str, Any]) -> None:
    status = "PASS" if score.get("ok") else "FAIL"
    print(f"[{status}] {title}: {score.get('reason')}")
    for key, value in score.items():
        if key in {"ok", "reason"}:
            continue
        if isinstance(value, float):
            print(f"  {key}={value:.4g}")
        else:
            print(f"  {key}={value}")
