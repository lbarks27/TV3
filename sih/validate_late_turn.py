#!/usr/bin/env python3
"""Reproducible late-turn flight checks; never connects to hardware.

Cases: nominal, archive (motor draws from the saved successful flight), or
seed:N (the profile's motor scatter with a fixed RNG seed). All cases retain
the simulator's existing catch, contact, backstop, and hard-landing checks.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_sih import TV3_ROOT, run_once
from tvc.attitude import quaternion_to_euler
from tvc.flight_log import FlightLog
import tvc.parameters as parameters


def evaluate(profile_path: str, case: str, output_dir: str) -> dict:
    source = Path(profile_path).resolve()
    out = Path(output_dir).resolve() / case.replace(":", "_")
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(source.read_text())
    vehicle = Path(data["vehicle"])
    if not vehicle.is_absolute():
        candidates = [source.parent / vehicle, TV3_ROOT / "config" / vehicle]
        vehicle = next(p for p in candidates if p.is_file()).resolve()
    data["vehicle"] = str(vehicle)
    if case == "nominal":
        data["engine_variance"] = {"thrust": 0.0, "burn_time": 0.0}
    elif case.startswith("seed:"):
        scatter = data.get("engine_variance", {})
        if not any(float(scatter.get(k, 0)) > 0 for k in ["thrust", "burn_time", "ignition_delay_s"]):
            raise ValueError("Seeded checks require nonzero motor scatter in the profile")
        data.setdefault("engine_variance", {})["seed"] = int(case.split(":")[1])
    elif case == "archive":
        # The loader omits the variance object when every range is zero.
        # These ranges only activate loading; recorded_variance supplies
        # the exact draws rather than sampling from them.
        data["engine_variance"] = {"thrust": 0.03, "burn_time": 0.03}
    elif case != "archive":
        raise ValueError(f"Unknown case: {case}")
    resolved = out / "profile.json"
    resolved.write_text(json.dumps(data, indent=2) + "\n")

    if case == "archive":
        archive = TV3_ROOT / "sih/fixtures/archive-catch.json"
        draws = json.loads(archive.read_text())["engine_draws"]

        def recorded_variance(mounts, variance, *, rng=None):
            if len(mounts) != len(draws):
                raise ValueError("Archived motor count does not match vehicle")
            updated, values = [], []
            for mount, draw in zip(mounts, draws):
                thrust = draw["thrust_scale"]
                burn = draw["time_scale"]
                delay = draw["ignition_delay_s"]
                motor = mount.motor.with_performance(thrust_scale=thrust, time_scale=burn)
                if delay:
                    motor = motor.with_ignition_delay(motor.ignition_delay_s + delay)
                updated.append(replace(mount, motor=motor, thrust=float(motor.max_thrust_n)))
                values.append((thrust, burn, delay))
            return updated, values

        with patch.object(parameters, "apply_engine_variance", recorded_variance):
            result = run_once(str(resolved), out / "flight.csv")
        if result["engine_draws"] != draws:
            raise RuntimeError("Simulation did not use the archived motor draws")
        result["motor_draw_source"] = str(archive)
    else:
        result = run_once(str(resolved), out / "flight.csv")

    log = FlightLog.load(out / "flight.csv")
    approach = np.deg2rad(data["mission"]["approach_yaw_deg"])
    landing = np.deg2rad(data["mission"]["landing_yaw_deg"])
    wrap = lambda x: (x + np.pi) % (2.0 * np.pi) - np.pi
    turns = [s for s in log.samples if abs(wrap(s.heading_ref_rad - approach)) > 1e-4]
    complete = [s for s in turns if abs(wrap(s.heading_ref_rad - landing)) < 1e-4]
    final = [s for s in log.samples if s.mission_phase == "landing"]
    pre = [s for s in log.samples if not turns or s.time < turns[0].time]
    catch_sample = None
    if log.metadata.chopstick_caught == "true":
        # The plant freezes on a two-bar catch. Locate the beginning of
        # the terminal stationary interval, rather than calling burnout
        # the touchdown time.
        last = log.samples[-1]
        index = len(log.samples) - 1
        while index > 0:
            s = log.samples[index - 1]
            if not (np.allclose(s.position, last.position, atol=1e-10, rtol=0)
                    and np.allclose(s.velocity, 0, atol=1e-10, rtol=0)):
                break
            index -= 1
        if index < len(log.samples) - 2:
            catch_sample = log.samples[index]
    result["case"] = case
    result["source_profile"] = str(source)
    result["heading"] = {
        "landing_phase_s": final[0].time if final else None,
        "turn_start_s": turns[0].time if turns else None,
        "turn_end_s": complete[0].time if complete else None,
        "turn_start_position_m": turns[0].position.tolist() if turns else None,
        "max_pre_turn_yaw_error_deg": max(
            (abs(float(np.rad2deg(wrap(quaternion_to_euler(s.quaternion)[2] - approach))))
             for s in pre), default=0.0
        ),
        "final_yaw_deg": float(np.rad2deg(quaternion_to_euler(log.samples[-1].quaternion)[2])),
    }
    result["catch_reported"] = log.metadata.chopstick_caught == "true"
    result["catch_time_s"] = catch_sample.time if catch_sample is not None else None
    result["heading_schedule_ok"] = bool(
        final and turns and complete
        and turns[0].time >= final[0].time
        and result["heading"]["max_pre_turn_yaw_error_deg"] < 5.0
        and (catch_sample is None or complete[0].time <= catch_sample.time)
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _worker(args):
    return evaluate(*args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=TV3_ROOT / "config/profiles/real_landing_g12_late_turn.json")
    parser.add_argument("--cases", nargs="+", default=["nominal", "archive", "seed:0", "seed:1", "seed:2", "seed:42"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.profile.resolve()), c, str(out)) for c in args.cases]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for result in pool.map(_worker, jobs):
            results.append(result)
            print(json.dumps({k: result[k] for k in ["case", "landing", "heading"]}), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(r["landing"]["success"] and r["heading_schedule_ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
