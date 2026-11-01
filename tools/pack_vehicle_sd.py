#!/usr/bin/env python3
"""Pack vehicle + mission + motor curves + logger topics for Cube Orange+ SD boot.

Writes under ``--out`` (the SD ``/tvc`` directory):

  vehicle.json / vehicle.crc     geometry (arm is fail-closed on these)
  mission.json                   flattened pad mission (tvc_mission / tvc_pos_alt)
  motor0.csv motor1.csv motor2.csv   time_s,thrust_n (tvc_thrust)

and beside the SD root (parent of ``tvc``):

  etc/logging/logger_topics.txt  extra ULog topics including flight_log_vl @ 100 Hz

Fail-closed on the FC: missing vehicle.json, missing CRC, or mismatch blocks arm.
Mission / curves are optional at boot (params / nominal fallback) but should be
packed for pad.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import zlib
from pathlib import Path

TV3_ROOT = Path(__file__).resolve().parents[1]

PROFILE_BY_VEHICLE = {
    "tvc_ring3_g12.json": "real_landing_g12_aligned.json",
    "tvc_ring3_c6.json": "pad_hop_c6.json",
    "tvc_ring3_b6.json": "pad_hop_b6.json",
    "tvc_ring3_a3.json": "pad_hop_a3.json",
    "tvc_ring3_f10.json": "real_landing_f10.json",
    "tvc_ring3_f10_sync.json": "real_landing_f10.json",
    "tvc_ring3_g8.json": "real_landing_g8.json",
    "tvc_ring3.json": "hover_10m.json",
}

LOGGER_TOPICS_SRC = TV3_ROOT / "config" / "logger_topics.txt"
MAX_MOTOR_SAMPLES = 256


def crc32_ieee(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def prototype_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("PROTOTYPE_ROOT")
    if env:
        roots.append(Path(env).expanduser().resolve())
    roots.append((TV3_ROOT.parent / "Prototype").resolve())
    roots.append((TV3_ROOT / "sim_source").resolve())
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r in seen or not r.is_dir():
            continue
        seen.add(r)
        out.append(r)
    return out


def find_dynamics_csv(manufacturer: str, designation: str) -> Path | None:
    rel = Path("vendor") / "Thrust-Curves-Apogee" / manufacturer / designation
    for root in prototype_roots():
        folder = root / rel
        if not folder.is_dir():
            continue
        cands = sorted(folder.glob("*_dynamics.csv"))
        if cands:
            return cands[0]
    return None


def load_dynamics_samples(path: Path) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"no header in {path}")
        field_map = {name.lower(): name for name in reader.fieldnames}
        time_key = field_map.get("time_s") or field_map.get("time")
        thrust_key = field_map.get("thrust_n") or field_map.get("thrust")
        if time_key is None or thrust_key is None:
            raise ValueError(f"{path} needs time_s and thrust_N columns")
        for row in reader:
            samples.append((float(row[time_key]), float(row[thrust_key])))
    if len(samples) < 2:
        raise ValueError(f"{path} has fewer than 2 samples")
    if len(samples) > MAX_MOTOR_SAMPLES:
        # Keep first/last; take evenly spaced interior points.
        idxs = [round(i * (len(samples) - 1) / (MAX_MOTOR_SAMPLES - 1)) for i in range(MAX_MOTOR_SAMPLES)]
        samples = [samples[i] for i in idxs]
    return samples


def write_motor_csv(path: Path, samples: list[tuple[float, float]], delay_s: float) -> None:
    with path.open("w", newline="") as handle:
        handle.write("time_s,thrust_n\n")
        for t, th in samples:
            handle.write(f"{t + delay_s:.6f},{th:.6f}\n")


def mount_engines(vehicle: dict) -> list[dict]:
    shared = dict(vehicle.get("engine") or {})
    mounts = vehicle.get("mounts") or []
    out: list[dict] = []
    for i in range(3):
        eng = dict(shared)
        if i < len(mounts) and isinstance(mounts[i], dict):
            local = mounts[i].get("engine") or {}
            if isinstance(local, dict):
                eng.update(local)
        out.append(eng)
    return out


def flatten_profile(profile: dict) -> dict:
    mission = profile.get("mission") or {}
    mtype = str(mission.get("type") or "")
    motion = str(profile.get("motion") or "")
    if mtype == "real_landing":
        mode = 2
    elif mtype == "hover" or motion == "hover":
        mode = 1
    else:
        mode = 0

    wp = mission.get("first_waypoint_offset_m") or [5.0, 0.0, 5.0]
    target = profile.get("target_position_m") or [0.0, 0.0, 10.0]
    pos = profile.get("position_controller") or {}
    alt = profile.get("altitude_controller") or {}
    ft = profile.get("flight_test") or {}

    kp = pos.get("kp") or [0.8, 0.8, 0.0]
    kd = pos.get("kd") or [1.4, 1.4, 0.0]
    ki = pos.get("ki") or [0.0, 0.015, 0.0]
    speed = pos.get("max_speed_m_s")
    if isinstance(speed, (list, tuple)):
        speed_x = float(speed[0]) if speed else 0.0
        speed_y = float(speed[1]) if len(speed) > 1 else speed_x
    elif speed is not None:
        speed_x = speed_y = float(speed)
    else:
        speed_x = speed_y = 0.0
    tilt = float(pos.get("max_tilt_deg", 20.0))
    tilt_y = pos.get("max_tilt_y_deg")
    err = float(pos.get("max_position_error_m", 3.0))
    err_y = float(pos.get("max_position_error_y_m", 0.0))

    hover_z = float(target[2]) if len(target) > 2 else 10.0
    if mode == 2 and len(wp) > 2:
        # real_landing uses the approach waypoint altitude, not a hover hold.
        hover_z = float(wp[2])

    launch = mission.get("launch_offset_m") or [0.0, 0.0, 0.0]
    voff = mission.get("vertical_hold_offset_m")
    wp_n = float(wp[1]) if len(wp) > 1 else 0.0

    heading_deg = None
    if "landing_yaw_deg" in mission:
        land_yaw = float(mission["landing_yaw_deg"])
        appr_yaw = float(mission.get("approach_yaw_deg", land_yaw))
        # Hold a heading only when the mission does not slew (pad tests).
        if abs(land_yaw - appr_yaw) < 1e-3:
            heading_deg = land_yaw
    else:
        euler = ft.get("initial_euler_deg") or []
        if len(euler) > 2:
            heading_deg = float(euler[2])

    flat = {
        "name": str(profile.get("name") or ""),
        "mode": mode,
        "hover_z": hover_z,
        "wp_e": float(wp[0]),
        "wp_n": wp_n,
        "wp_u": float(wp[2]) if len(wp) > 2 else 5.0,
        "capture_radius_m": float(mission.get("capture_radius_m", 1.5)),
        "capture_max_speed_m_s": float(mission.get("capture_max_speed_m_s", 2.0)),
        "capture_settle_s": float(mission.get("capture_settle_s", 0.0)),
        "stage_radius_m": float(mission.get("stage_capture_radius_m", 1.0)),
        "stage_speed_m_s": 1.2,
        "vertical_hold_s": float(mission.get("vertical_hold_s", 0.0) or 0.0),
        "descent_duration_s": float(mission.get("descent_duration_s", 7.0)),
        "min_approach_time_s": float(mission.get("min_approach_time_s", 2.0)),
        "safe_z": float(
            mission.get("path_end_altitude_m")
            or mission.get("touchdown_altitude_m")
            or 1.3081
        ),
        "land_z": float(mission.get("touchdown_altitude_m", 0.0)),
        "safe_half_m": 0.40,
        "safe_vmax": 1.0,
        "pos_kp": float(kp[0]),
        "pos_kd": float(kd[0]),
        "pos_kp_y": float(kp[1] if len(kp) > 1 else kp[0]),
        "pos_kd_y": float(kd[1] if len(kd) > 1 else kd[0]),
        "pos_ki": float(ki[0]),
        "pos_ki_y": float(ki[1] if len(ki) > 1 else ki[0]),
        "pos_i_limit": 0.08 if pos.get("integral_limit") is None else float(pos.get("integral_limit")),
        "pos_i_start": float(pos.get("integral_start_m", 1.5)),
        "lat_priority": float(pos.get("lateral_priority_m", 0.05)),
        "lat_lpf_hz": float(pos.get("lateral_accel_lpf_hz", 0.0)),
        "pos_tilt_deg": tilt,
        "pos_tilt_y_deg": float(tilt_y) if tilt_y is not None else tilt,
        "max_speed_m_s": speed_x,
        "max_speed_y_m_s": speed_y,
        "max_position_error_m": err,
        "max_position_error_y_m": err_y,
        "prefer_lateral_y": 1 if pos.get("prefer_lateral_y") else 0,
        "alt_kp": float(alt.get("kp", 12.0)),
        "alt_kd": float(alt.get("kd", 8.0)),
        "alt_ki": float(alt.get("ki", 1.5)),
        "alt_i_limit": 20.0 if alt.get("integral_limit") is None else float(alt.get("integral_limit")),
        "throttle_min": float(alt.get("throttle_min", 0.0) or 0.0),
        "land_tilt_deg": -1.0 if mission.get("landing_max_tilt_deg") is None else float(mission.get("landing_max_tilt_deg")),
        "force_weight": float(ft.get("force_weight", 0.25)),
    }
    if voff is not None:
        flat["has_loft"] = 1
        flat["loft_e"] = float(launch[0]) + float(voff[0])
        flat["loft_n"] = float(launch[1]) + float(voff[1])
        flat["loft_u"] = float(max(0.0, launch[2])) + float(voff[2])
    remain = mission.get("pad_transition_burn_remaining_s")
    if remain is not None:
        flat["pad_remain_s"] = float(remain)
    if heading_deg is not None:
        flat["heading_deg"] = heading_deg
    return flat


def default_profile_path(vehicle_json: Path) -> Path | None:
    name = PROFILE_BY_VEHICLE.get(vehicle_json.name)
    if not name:
        return None
    path = TV3_ROOT / "config" / "profiles" / name
    return path if path.is_file() else None


def pack(
    vehicle_json: Path,
    out: Path,
    profile_json: Path | None,
    pack_mission: bool,
    pack_curves: bool,
    pack_logger: bool,
) -> int:
    src = vehicle_json.expanduser().resolve()
    if not src.is_file():
        print(f"missing {src}", file=sys.stderr)
        return 1

    payload = json.loads(src.read_text())
    required = ("center_of_mass", "vehicle", "mounts", "angle_limits_deg", "rest_angles_deg")
    for key in required:
        if key not in payload:
            print(f"{src} missing required key {key}", file=sys.stderr)
            return 1
    if float(payload["vehicle"].get("mass_g", 0)) <= 0:
        print("vehicle.mass_g must be positive", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    dest = out / "vehicle.json"
    shutil.copyfile(src, dest)
    data = dest.read_bytes()
    crc = crc32_ieee(data)
    crc_path = out / "vehicle.crc"
    crc_path.write_text(f"crc32={crc:08x}\n")
    print(f"wrote {dest} ({len(data)} bytes)")
    print(f"wrote {crc_path} crc32={crc:08x}")

    if pack_mission:
        if profile_json is None:
            profile_json = default_profile_path(src)
            if profile_json:
                print(f"auto profile {profile_json}")

        if profile_json is not None:
            psrc = profile_json.expanduser().resolve()
            if not psrc.is_file():
                print(f"missing profile {psrc}", file=sys.stderr)
                return 1
            profile = json.loads(psrc.read_text())
            flat = flatten_profile(profile)
            if pack_curves:
                ends = []
                for eng in mount_engines(payload):
                    des = str(eng.get("designation") or "").strip()
                    if not des:
                        continue
                    dyn = find_dynamics_csv(str(eng.get("manufacturer") or "").strip(), des)
                    if dyn is None:
                        continue
                    samples = load_dynamics_samples(dyn)
                    if samples:
                        ends.append(samples[-1][0] + float(eng.get("ignition_delay_s") or 0.0))
                if ends and flat.get("pad_remain_s") is not None:
                    flat["burn_end_s"] = max(ends)
            mission_path = out / "mission.json"
            mission_path.write_text(json.dumps(flat, indent=2) + "\n")
            print(f"wrote {mission_path} name={flat['name']} mode={flat['mode']}")
        else:
            print("no profile — skip mission.json (tvc_mission will use TVC_MIS_* params)")

    if pack_curves:
        engines = mount_engines(payload)
        for i, eng in enumerate(engines):
            mfg = str(eng.get("manufacturer") or "").strip()
            des = str(eng.get("designation") or "").strip()
            delay = float(eng.get("ignition_delay_s") or 0.0)
            if not des:
                print(f"mount {i}: no engine designation — skip motor{i}.csv", file=sys.stderr)
                continue
            dyn = find_dynamics_csv(mfg, des)
            if dyn is None:
                print(
                    f"mount {i}: no dynamics CSV for {mfg}/{des} "
                    "(set PROTOTYPE_ROOT or vendor/Thrust-Curves-Apogee)",
                    file=sys.stderr,
                )
                return 1
            samples = load_dynamics_samples(dyn)
            csv_path = out / f"motor{i}.csv"
            write_motor_csv(csv_path, samples, delay)
            print(
                f"wrote {csv_path} {mfg}/{des} n={len(samples)} delay={delay:.3f}s from {dyn.name}"
            )

    if pack_logger:
        if not LOGGER_TOPICS_SRC.is_file():
            print(f"missing {LOGGER_TOPICS_SRC}", file=sys.stderr)
            return 1
        sd_root = out.parent if out.name == "tvc" else out
        log_dir = sd_root / "etc" / "logging"
        log_dir.mkdir(parents=True, exist_ok=True)
        dest_topics = log_dir / "logger_topics.txt"
        shutil.copyfile(LOGGER_TOPICS_SRC, dest_topics)
        print(f"wrote {dest_topics} (flight_log_vl @ 100 Hz)")

    print("Copy onto the SD card as /tvc/ (and /etc/logging/logger_topics.txt)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vehicle_json", type=Path, help="config/vehicles/tvc_ring3_g12.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sd/tvc"),
        help="Destination directory (copied onto the SD card as /tvc)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="config/profiles/real_landing_g12.json (auto from vehicle name if omitted)",
    )
    parser.add_argument(
        "--no-mission",
        action="store_true",
        help="Do not write mission.json",
    )
    parser.add_argument(
        "--no-curves",
        action="store_true",
        help="Do not write motor*.csv",
    )
    parser.add_argument(
        "--no-logger",
        action="store_true",
        help="Do not write etc/logging/logger_topics.txt",
    )
    args = parser.parse_args()
    return pack(
        args.vehicle_json,
        args.out,
        args.profile,
        pack_mission=not args.no_mission,
        pack_curves=not args.no_curves,
        pack_logger=not args.no_logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
