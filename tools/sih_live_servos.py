#!/usr/bin/env python3
"""Run Prototype SIH and stream gimbal angles to live Cube MAIN PWM.

Live path: MAV_CMD_ACTUATOR_TEST / DO_SET_SERVO (not param writes).
Rest restore uses a1=0° (1500 µs midpoint). Does not arm. Fire stays off.

Logical angles [M0a2, M0a1, M1a2, M1a1, M2a2, M2a1] match tvc_actuators.
Pin map is airframe PWM_MAIN_FUNC* (MAIN 1/2/3 = a2, MAIN 5/6/7 = a1).

Usage::

    "$PROTOTYPE_ROOT/.venv/bin/python" tools/sih_live_servos.py
    "$PROTOTYPE_ROOT/.venv/bin/python" tools/sih_live_servos.py --profile real_landing_g12
    "$PROTOTYPE_ROOT/.venv/bin/python" tools/sih_live_servos.py --replay sih/logs/real_landing_g12.csv
"""

from __future__ import annotations

import argparse
import atexit
import glob
import math
import os
import signal
import struct
import sys
import time
from pathlib import Path

import numpy as np
from pymavlink import mavutil

TV3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TV3_ROOT / "sih"))

from run_sih import run_once  # noqa: E402

PROTOTYPE = Path(os.environ.get("PROTOTYPE_ROOT", TV3_ROOT / "sim_source")).resolve()
sys.path.insert(0, str(PROTOTYPE))

from tvc.flight_log import FlightLog  # noqa: E402

# a2 hardware stop ±30°. PWM 1500±250 µs (800–2200 looked like ±90° on these servos).
A2_RANGE_RAD = math.radians(30.0)
A1_REST_RAD = math.radians(0.0)
A1_RANGE_RAD = math.radians(90.0)
PWM_MID = 1500
PWM_HALF_A2 = 250
PWM_HALF_A1 = 700
PWM_MIN = 800
PWM_MAX = 2200

# Physical DIS params for the current harness.
# Logical servo 201..206 → MAIN pins via PWM_MAIN_FUNC*.
DIS_PARAM = {
    0: "PWM_MAIN_DIS1",  # M0 a2 → MAIN 1
    1: "PWM_MAIN_DIS5",  # M0 a1 → MAIN 5
    2: "PWM_MAIN_DIS2",  # M1 a2 → MAIN 2
    3: "PWM_MAIN_DIS6",  # M1 a1 → MAIN 6
    4: "PWM_MAIN_DIS3",  # M2 a2 → MAIN 3
    5: "PWM_MAIN_DIS7",  # M2 a1 → MAIN 7
}

_mav = None
_restored = False
_rev = 47
_trims = [0.0] * 6


def find_port(preferred: str | None) -> str:
    if preferred:
        return preferred
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not ports:
        raise SystemExit("No /dev/cu.usbmodem* — Cube unplugged or QGC holding USB?")
    return ports[0]


def send_hb(m) -> None:
    m.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        0,
    )


def cmd_long(m, command: int, *params) -> None:
    p = list(params) + [0.0] * 7
    m.mav.command_long_send(m.target_system or 1, 1, command, 0, *p[:7])


def param_id(msg) -> str:
    pid = msg.param_id
    if not isinstance(pid, str):
        pid = pid.decode("utf-8", errors="ignore")
    return pid.strip("\x00")


def get_param_float(m, name: str, timeout: float = 2.5) -> float | None:
    m.mav.param_request_read_send(m.target_system or 1, 1, name.encode("ascii"), -1)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.35)
        except Exception:
            continue
        if msg and name in param_id(msg):
            return float(msg.param_value)
    return None


def get_param_int(m, name: str, timeout: float = 2.5) -> int | None:
    m.mav.param_request_read_send(m.target_system or 1, 1, name.encode("ascii"), -1)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.35)
        except Exception:
            continue
        if msg and name in param_id(msg):
            return struct.unpack("<i", struct.pack("<f", msg.param_value))[0]
    return None


def set_param_int(m, name: str, value: int, wait: bool = False) -> None:
    fval = struct.unpack("<f", struct.pack("<i", int(value)))[0]
    m.mav.param_set_send(
        m.target_system or 1,
        1,
        name.encode("ascii"),
        fval,
        mavutil.mavlink.MAV_PARAM_TYPE_INT32,
    )
    if not wait:
        return
    t0 = time.time()
    while time.time() - t0 < 2.0:
        try:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.3)
        except Exception:
            continue
        if msg and name in param_id(msg):
            return


def connect(port: str):
    print(f"Connecting {port}")
    m = mavutil.mavlink_connection(port, baud=57600, source_system=255, source_component=190)
    t0 = time.time()
    while time.time() - t0 < 15:
        send_hb(m)
        try:
            msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        except Exception:
            continue
        if msg and getattr(msg, "autopilot", None) == mavutil.mavlink.MAV_AUTOPILOT_PX4:
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            hil = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_HIL_ENABLED)
            print(f"  PX4 armed={armed} hil={hil} base_mode={msg.base_mode}")
            if armed:
                raise SystemExit("Cube is ARMED — disarm before live SIH playback")
            return m
    raise SystemExit("no PX4 HEARTBEAT")


def angles_to_us(
    angles: np.ndarray,
    scale: float,
    rev: int = 63,
    trims_deg: list[float] | None = None,
) -> list[int]:
    """angles (3, 2)=(a2, a1) rad → logical [6] µs. rev bits match TVC_SERVO_REV."""
    pose = np.asarray(angles, dtype=float).reshape(3, 2) * float(scale)
    trims = list(_trims if trims_deg is None else trims_deg)
    while len(trims) < 6:
        trims.append(0.0)
    out: list[int] = []
    for mount in range(3):
        a2 = float(pose[mount, 0]) + math.radians(max(-15.0, min(15.0, trims[mount * 2])))
        a1 = float(pose[mount, 1]) + math.radians(max(-15.0, min(15.0, trims[mount * 2 + 1])))
        a2 = max(-A2_RANGE_RAD, min(A2_RANGE_RAD, a2))
        a1 = max(A1_REST_RAD, min(A1_RANGE_RAD, a1))
        n2 = max(-1.0, min(1.0, a2 / A2_RANGE_RAD))
        n1 = max(-1.0, min(1.0, -a1 / A1_RANGE_RAD))
        halves = (PWM_HALF_A2, PWM_HALF_A1)
        pair = (n2, n1)
        for j, n in enumerate(pair):
            ch = mount * 2 + j
            if rev & (1 << ch):
                n = -n
            us = int(round(PWM_MID + n * halves[j]))
            out.append(max(PWM_MIN, min(PWM_MAX, us)))
    return out


def us_to_norm(us: int) -> float:
    # ACTUATOR_TEST is mixer-normalized: PX4 maps [-1,1] to PWM_MAIN_MIN/MAX
    # (default 800–2200, half 700). angles_to_us already used ±250 µs for ±30°.
    return max(-1.0, min(1.0, (float(us) - PWM_MID) / 700.0))


def emit_us(m, us6: list[int], *, persist_dis: bool = False) -> None:
    """Drive MAIN via Commander ACTUATOR_TEST (works while disarmed).

    PX4 Commander::handleCommandActuatorTest:
      param1 = value [-1, 1]
      param2 = timeout [s]  (must be > 0 or it releases)
      param5 = MAVLink function: servo1 = 33  (or 1000+201)
    COM_MOT_TEST_EN must be 1. Denied if commander-armed.
    """
    for logical, us in enumerate(us6):
        cmd_long(m, 310, us_to_norm(us), 0.35, 0.0, 0.0, 33.0 + logical)
    if persist_dis:
        for logical, us in enumerate(us6):
            set_param_int(m, DIS_PARAM[logical], us, wait=False)


def restore_rest(m) -> None:
    global _restored
    if _restored or m is None:
        return
    _restored = True
    rest_pose = np.zeros((3, 2))
    rest_pose[:, 1] = A1_REST_RAD
    rest = angles_to_us(rest_pose, 1.0, _rev)
    print(f"Restoring rest {rest} µs on MAIN 1/2/3/5/6/7 (trim={_trims})", flush=True)
    emit_us(m, rest, persist_dis=True)
    time.sleep(0.15)
    for logical, name in DIS_PARAM.items():
        set_param_int(m, name, rest[logical], wait=True)
    # Release actuator-test overrides so DIS rest remains.
    for logical in range(6):
        cmd_long(m, 310, 0.0, 0.0, 0.0, 0.0, 33.0 + logical)  # timeout 0 = release
    cmd_long(m, mavutil.mavlink.MAV_CMD_PREFLIGHT_STORAGE, 1)
    time.sleep(0.3)


def prepare_fc(m) -> int:
    global _rev, _trims
    print("=== Safety ===")
    for name, want in (
        ("SYS_HITL", 0),
        ("TVC_ALLOW_FIRE", 0),
        ("TVC_IGN_GPIO_EN", 0),
        ("COM_PREARM_MODE", 0),  # stay disarmed — ACTUATOR_TEST is denied if armed
        ("COM_MOT_TEST_EN", 1),
        ("TVC_SERVO_REV", 47),
    ):
        cur = get_param_int(m, name)
        print(f"  {name}={cur}")
        if cur != want:
            set_param_int(m, name, want, wait=True)
            print(f"    -> {want}")
    # Confirm map still matches the harness.
    for name, want in (
        ("PWM_MAIN_FUNC1", 201),
        ("PWM_MAIN_FUNC2", 203),
        ("PWM_MAIN_FUNC3", 205),
        ("PWM_MAIN_FUNC4", 0),
        ("PWM_MAIN_FUNC5", 202),
        ("PWM_MAIN_FUNC6", 204),
        ("PWM_MAIN_FUNC7", 206),
    ):
        got = get_param_int(m, name)
        if got != want:
            raise SystemExit(f"{name}={got} want {want} — refuse to stream")
    print("  pin map ok, fire inhibited, HITL off")
    rev = get_param_int(m, "TVC_SERVO_REV")
    if rev is None:
        rev = 63
    print(f"  TVC_SERVO_REV={rev} (bits 0..5 = M0a2 M0a1 M1a2 M1a1 M2a2 M2a1)")
    _rev = int(rev)
    trims: list[float] = []
    for i in range(6):
        v = get_param_float(m, f"TVC_TRIM_{i}")
        trims.append(0.0 if v is None else max(-15.0, min(15.0, float(v))))
    _trims = trims
    print(f"  TVC_TRIM={_trims} deg (M0a2 M0a1 M1a2 M1a1 M2a2 M2a1)")
    return rev


def play(m, log: FlightLog, *, scale: float, rate_hz: float, rev: int) -> None:
    samples = log.samples
    if not samples:
        raise SystemExit("empty SIH log")
    dt_out = 1.0 / max(rate_hz, 1.0)
    t_end = float(samples[-1].time)
    print(
        f"Playback {t_end:.1f}s  {len(samples)} samples  "
        f"{rate_hz:.0f} Hz  scale={scale:.2f}",
        flush=True,
    )
    print("Watch the gimbals. Ctrl-C returns to rest.", flush=True)
    time.sleep(1.0)

    t_wall0 = time.perf_counter()
    idx = 0
    last_print = -1.0
    while True:
        now = time.perf_counter() - t_wall0
        if now > t_end + 0.05:
            break
        while idx + 1 < len(samples) and float(samples[idx + 1].time) <= now:
            idx += 1
        sample = samples[idx]
        us6 = angles_to_us(sample.gimbal_angles, scale, rev)
        # Always refresh ACTUATOR_TEST (timeout 0.35 s) — skip-if-unchanged
        # lets Commander release and the horns snap to DIS rest.
        emit_us(m, us6)
        send_hb(m)
        try:
            while m.recv_match(blocking=False) is not None:
                pass
        except Exception:
            pass
        if now - last_print >= 0.5:
            last_print = now
            pos = np.asarray(sample.position, dtype=float).reshape(3)
            deg = np.rad2deg(np.asarray(sample.gimbal_angles, dtype=float).reshape(3, 2))
            print(
                f"  t={now:5.2f}s  z={pos[2]:5.2f}m  "
                f"a2=[{deg[0,0]:+5.1f},{deg[1,0]:+5.1f},{deg[2,0]:+5.1f}]  "
                f"a1=[{deg[0,1]:4.1f},{deg[1,1]:4.1f},{deg[2,1]:4.1f}]  "
                f"us={us6}",
                flush=True,
            )
        sleep = dt_out - (time.perf_counter() - t_wall0 - now)
        if sleep > 0:
            time.sleep(min(sleep, dt_out))


def main() -> int:
    global _mav, _rev
    ap = argparse.ArgumentParser(description="SIH → live Cube gimbal PWM")
    ap.add_argument("--profile", default="real_landing_g12")
    ap.add_argument("--replay", type=str, default=None, help="Skip SIH; play this CSV")
    ap.add_argument("--port", default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="Angle scale (1=full SIH)")
    ap.add_argument("--rate", type=float, default=40.0, help="PWM update Hz")
    ap.add_argument("--skip-sih", action="store_true", help="Use last log for --profile")
    ap.add_argument(
        "--rev",
        type=int,
        default=None,
        help="TVC_SERVO_REV bitmask (default 47 = invert all except M2 a2)",
    )
    args = ap.parse_args()

    if args.replay:
        log_path = Path(args.replay)
        print(f"Replay {log_path}")
        log = FlightLog.load(log_path)
        sih_info = {"log": str(log_path), "profile": log_path.stem}
    elif args.skip_sih:
        log_path = TV3_ROOT / "sih" / "logs" / f"{Path(args.profile).stem}.csv"
        print(f"Skip SIH, load {log_path}")
        log = FlightLog.load(log_path)
        sih_info = {"log": str(log_path), "profile": args.profile}
    else:
        print(f"=== SIH {args.profile} ===")
        sih_info = run_once(args.profile)
        print(
            f"  mass={sih_info['mass_kg']:.3f} kg  "
            f"T/W_peak={sih_info['tw_peak']:.2f}  "
            f"z_max={sih_info['z_max_m']:.2f} m  "
            f"landing={sih_info['landing']}"
        )
        log = FlightLog.load(sih_info["log"])

    port = find_port(args.port)
    m = connect(port)
    _mav = m

    def _cleanup(*_a):
        restore_rest(_mav)

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(143)))

    rev = prepare_fc(m)
    if args.rev is not None:
        set_param_int(m, "TVC_SERVO_REV", int(args.rev), wait=True)
        rev = int(args.rev)
        _rev = rev
        print(f"  TVC_SERVO_REV override {rev}")
    try:
        play(m, log, scale=args.scale, rate_hz=args.rate, rev=rev)
    finally:
        restore_rest(m)
        try:
            m.close()
        except Exception:
            pass
    print("Done. Servos commanded back to rest. Fire was never enabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
