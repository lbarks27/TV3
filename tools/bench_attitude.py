#!/usr/bin/env python3
"""Live bench attitude hold: Cube IMU → Prototype controller → MAIN PWM.

Pretends the motors are producing constant hover-class thrust so the
allocator has authority. Torque-only (no collective a1 fold). Holds
world-level with the heading captured at start.

Does not arm. Forces fire inhibits off. Ctrl-C returns servos to rest.

Usage::

    "$PROTOTYPE_ROOT/.venv/bin/python" tools/bench_attitude.py
    "$PROTOTYPE_ROOT/.venv/bin/python" tools/bench_attitude.py --gain-scale 0.5
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from pymavlink import mavutil

TV3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TV3_ROOT / "sih"))
sys.path.insert(0, str(TV3_ROOT / "tools"))

from run_sih import _patch_vehicle_path, _resolve_profile  # noqa: E402
import sih_live_servos as live  # noqa: E402

PROTOTYPE = Path(os.environ.get("PROTOTYPE_ROOT", TV3_ROOT / "sim_source")).resolve()
sys.path.insert(0, str(PROTOTYPE))

from tvc.allocator import TorqueAllocator  # noqa: E402
from tvc.attitude import (  # noqa: E402
    normalize_quaternion,
    quaternion_from_euler,
    quaternion_from_rotation_matrix,
    quaternion_from_thrust_and_heading,
    quaternion_to_euler,
)
from tvc.controller import AttitudePIDController  # noqa: E402
from tvc.parameters import load_vehicle_profile  # noqa: E402

R_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
R_FRD_FLU = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def q_px4_to_sim(q_px4: np.ndarray) -> np.ndarray:
    """NED←FRD → ENU←FLU (same contract as tvc_core frames.cpp)."""
    r_ned_frd = quat_to_dcm(q_px4)
    r_enu_flu = R_NED_ENU @ r_ned_frd @ R_FRD_FLU
    return quaternion_from_rotation_matrix(r_enu_flu)


def frd_to_flu(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], -v[1], -v[2]], dtype=float)


def slew_angles(current: np.ndarray, target: np.ndarray, dps: float, dt: float) -> np.ndarray:
    max_step = math.radians(dps) * max(dt, 1e-4)
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    return current + np.clip(delta, -max_step, max_step)


def load_plant(profile_name: str):
    profile_path = _resolve_profile(profile_name)
    resolved = _patch_vehicle_path(profile_path)
    cwd = os.getcwd()
    os.chdir(PROTOTYPE)
    try:
        system = load_vehicle_profile(profile_path=resolved)
    finally:
        os.chdir(cwd)
    alloc = TorqueAllocator(
        system.mounts,
        system.solver,
        force_axis=system.render.force_axis,
        center_of_mass=system.center_of_mass,
    )
    return system, alloc


def request_imu(m) -> None:
    # ATTITUDE=30, ATTITUDE_QUATERNION=31
    live.cmd_long(m, 511, 30, 20000)  # 50 Hz
    live.cmd_long(m, 511, 31, 20000)


def read_imu(m, timeout: float = 0.05):
    att = None
    quat = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            msg = m.recv_match(
                type=["ATTITUDE", "ATTITUDE_QUATERNION"],
                blocking=True,
                timeout=0.02,
            )
        except Exception:
            continue
        if msg is None:
            continue
        if msg.get_type() == "ATTITUDE":
            att = msg
        elif msg.get_type() == "ATTITUDE_QUATERNION":
            quat = msg
        if att is not None:
            break
    return att, quat


def sim_state(att, quat) -> tuple[np.ndarray, np.ndarray]:
    if quat is not None:
        q_px4 = np.array([quat.q1, quat.q2, quat.q3, quat.q4], dtype=float)
    else:
        q_px4 = quaternion_from_euler(float(att.roll), float(att.pitch), float(att.yaw))
    if att is not None:
        omega_frd = np.array([att.rollspeed, att.pitchspeed, att.yawspeed], dtype=float)
    else:
        omega_frd = np.zeros(3, dtype=float)
    return q_px4_to_sim(q_px4), frd_to_flu(omega_frd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live bench attitude → gimbal PWM")
    ap.add_argument("--profile", default="real_landing_g12")
    ap.add_argument("--port", default=None)
    ap.add_argument("--gain-scale", type=float, default=0.5, help="1.0 = flight gains")
    ap.add_argument("--thrust", type=float, default=12.0, help="Fake thrust per mount [N]")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--slew-dps", type=float, default=400.0)
    ap.add_argument("--scale", type=float, default=1.0, help="Angle scale to PWM")
    args = ap.parse_args()

    system, allocator = load_plant(args.profile)
    cfg = system.controller
    gs = float(args.gain_scale)
    cfg.kp = np.asarray(cfg.kp, dtype=float) * gs
    cfg.kd = np.asarray(cfg.kd, dtype=float) * gs
    cfg.ki = np.asarray(cfg.ki, dtype=float) * gs
    pid = AttitudePIDController(cfg)
    pid.reset()

    mass = float(system.vehicle.mass)
    thrusts = np.full(3, float(args.thrust), dtype=float)
    print(
        f"Plant {args.profile}  mass={mass:.3f} kg  "
        f"T_fake={args.thrust:.1f} N×3  gains×{gs:.2f}  "
        f"kp={cfg.kp} kd={cfg.kd}",
        flush=True,
    )

    port = live.find_port(args.port)
    m = live.connect(port)
    live._mav = m

    def _cleanup(*_a):
        live.restore_rest(m)

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(143)))

    rev = live.prepare_fc(m)
    request_imu(m)

    # Capture heading once we have IMU.
    q_ref = None
    heading0 = 0.0
    print("Waiting for ATTITUDE...", flush=True)
    t_wait = time.time()
    while time.time() - t_wait < 5.0:
        live.send_hb(m)
        att, quat = read_imu(m, 0.2)
        if att is not None or quat is not None:
            q_sim, _om = sim_state(att, quat)
            _, _, heading0 = quaternion_to_euler(q_sim)
            q_ref = quaternion_from_thrust_and_heading(np.array([0.0, 0.0, 1.0]), heading0)
            rpy = np.rad2deg(quaternion_to_euler(q_sim))
            print(
                f"  q_ref = level @ heading {math.degrees(heading0):.1f} deg  "
                f"(now rpy={rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f})",
                flush=True,
            )
            break
    if q_ref is None:
        raise SystemExit("no ATTITUDE from Cube — is HITL off and IMU healthy?")

    print("Bench stability LIVE. Tilt the airframe; gimbals should restore.", flush=True)
    print("Torque-only (a1 stays near rest). Ctrl-C → rest. Fire off.", flush=True)

    angles = allocator.rest_angles.copy()
    t_prev = time.perf_counter()
    last_print = 0.0
    loops = 0
    t_hz = time.perf_counter()
    dt_nom = 1.0 / max(args.rate, 10.0)

    while True:
        live.send_hb(m)
        att, quat = read_imu(m, 0.04)
        now = time.perf_counter()
        dt = min(max(now - t_prev, 0.002), 0.05)
        t_prev = now
        if att is None and quat is None:
            time.sleep(0.005)
            continue

        q_sim, omega_sim = sim_state(att, quat)

        tau = pid.update(q_sim, omega_sim, q_ref, dt)
        ang_cmd, tau_act, _dirs, sat = allocator.allocate_force_track(
            tau, force_target=0.0, force_weight=0.0, thrusts=thrusts
        )
        angles = slew_angles(angles, ang_cmd, args.slew_dps, dt)
        us6 = live.angles_to_us(angles, args.scale, rev)
        live.emit_us(m, us6)
        loops += 1
        # Drop command ACKs so the USB CDC does not stall.
        try:
            while m.recv_match(blocking=False) is not None:
                pass
        except Exception:
            pass

        if now - last_print >= 0.25:
            hz = loops / max(now - t_hz, 1e-3)
            loops = 0
            t_hz = now
            last_print = now
            rpy = np.rad2deg(quaternion_to_euler(q_sim))
            deg = np.rad2deg(angles)
            print(
                f"  {hz:5.1f} Hz  rpy=[{rpy[0]:+6.1f},{rpy[1]:+6.1f},{rpy[2]:+6.1f}]  "
                f"tau=[{tau[0]:+5.2f},{tau[1]:+5.2f},{tau[2]:+5.2f}]  "
                f"a2=[{deg[0,0]:+5.1f},{deg[1,0]:+5.1f},{deg[2,0]:+5.1f}]  "
                f"sat={int(sat)}",
                flush=True,
            )

        sleep = dt_nom - (time.perf_counter() - now)
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    raise SystemExit(main())
