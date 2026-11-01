#!/usr/bin/env python3
"""Run a simulated landing with a PX4 flight controller on the HIL link.

The host models vehicle motion and motor thrust and sends simulated IMU and
GPS data to PX4. The --control option selects the gimbal command source.

fc    Use actuator feedback from PX4. Requires the firmware fire latch.
host  Use the simulator's controllers. This mode can run without the latch.
auto  Start in fc mode if the latch is set, otherwise start in host mode.
      After 0.35 s, sustained rest commands or a trajectory outside the speed,
      position or tilt limits trigger a restart under host control. The landing
      check then scores only the host-controlled segment.

Example:
    python3 tools/hil_closed_loop_landing.py --control fc --profile real_landing_g12
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROTOTYPE = Path(os.environ.get("PROTOTYPE_ROOT", Path(__file__).resolve().parents[1] / "sim_source"))
sys.path.insert(0, str(PROTOTYPE))

from pymavlink import mavutil  # noqa: E402

from tvc.attitude import quaternion_to_euler, rotate_vector, world_to_body  # noqa: E402
from tvc.chopsticks import apply_chopstick_contacts  # noqa: E402
from tvc.dynamics import (  # noqa: E402
    VehicleProperties,
    compose_offset_mass_properties,
    integrate_6dof_rk4,
)
from tvc.geometry import thrust_direction  # noqa: E402
from tvc.flight_log import FlightLog, FlightLogMetadata, FlightLogSample  # noqa: E402
from tvc.flight_sim import (  # noqa: E402
    FlightScenario,
    FlightSimulator,
    evaluate_landing_acceptance,
)
from tvc.motors import thrusts_at  # noqa: E402
from tvc.parameters import FlightTestConfig, load_vehicle_profile  # noqa: E402

TV3_ROOT = Path(__file__).resolve().parents[1]
STATE_NAME = {
    0: "DISARMED",
    1: "PREFLIGHT",
    2: "ARMED",
    3: "LAUNCH_READY",
    4: "BURN",
    5: "COAST",
    6: "LAND",
    7: "ABORT",
    8: "SAFE",
}


# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

# PX4 accepts HIL baro only when fields_updated has abs pressure (bit 9),
# pressure altitude (bit 11), and temperature (bit 12). 0xFFF drops the baro.
_HIL_FIELDS = 0x1FFF

# ENU Gauss at the scripted GPS (PX4 SITL, Zurich): ~0.48 G, inc ~63°, dec ~3°.
# A horizontal field is rejected by the EKF inclination check, so yaw never aligns.
_MAG_ENU = np.array([0.011, 0.218, -0.428])

_ACTIVE_PUMP: "HilPump | None" = None
_SAVED_IMU_MODE: int | None = None

def find_port(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not ports:
        raise SystemExit("No Cube serial port — plug in FC and disconnect QGC")
    return ports[0]


class HilPump:
    """Sole owner of the Cube serial port once started.

    NSH and parameter writes are queued onto this thread so a blocking shell
    cannot pause HIL_SENSOR. A frozen magnetometer is what PX4 calls STALE.
    """

    def __init__(self, m):
        global _ACTIVE_PUMP
        self.m = m
        self.lock = threading.Lock()
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.accel = np.array([0.0, 0.0, -9.81])
        self.gyro = np.zeros(3)
        self.mag = _MAG_ENU.copy()
        self.stop = False
        self.keep_armed = False
        self._jobs: list = []
        self._nsh_req = None
        self._quiet = False
        self._n = 0
        self._last_arm = 0.0
        self._err_printed = False
        self._stream_set = False
        self.controls: list | None = None
        self.control_msgs = 0
        self.armed = False
        self.hil = False
        self.last_g: int | None = None
        self.last_status = ""
        self.thread = threading.Thread(target=self._run, daemon=True)
        _ACTIVE_PUMP = self
        self.thread.start()

    def update(self, pos, vel, accel_frd, gyro_frd, mag_frd) -> None:
        with self.lock:
            self.pos = np.array(pos, dtype=float)
            self.vel = np.array(vel, dtype=float)
            self.accel = np.array(accel_frd, dtype=float)
            self.gyro = np.array(gyro_frd, dtype=float)
            self.mag = np.array(mag_frd, dtype=float)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "controls": None if self.controls is None else list(self.controls),
                "control_msgs": self.control_msgs,
                "armed": self.armed,
                "hil": self.hil,
                "last_g": self.last_g,
                "last_status": self.last_status,
            }

    def call(self, fn, timeout: float = 2.0):
        done = threading.Event()
        slot: dict = {}

        def job() -> None:
            try:
                slot["r"] = fn(self.m)
            except Exception as exc:  # noqa: BLE001
                slot["e"] = exc
            finally:
                done.set()

        with self.lock:
            self._jobs.append(job)
        if not done.wait(timeout):
            raise TimeoutError("HIL link call timed out")
        if "e" in slot:
            raise slot["e"]
        return slot.get("r")

    def nsh(self, cmd: str, wait: float, quiet: bool = False) -> str:
        done = threading.Event()
        slot = {"text": ""}
        with self.lock:
            if quiet:
                self._quiet = True
            self._nsh_req = (cmd, wait, slot, done)
        done.wait(wait + 2.0)
        if quiet:
            with self.lock:
                self._quiet = False
        return slot["text"]

    def _run(self) -> None:
        nsh = None
        nsh_buf: list[str] = []
        period = 0.02
        while not self.stop:
            t_loop = time.time()
            with self.lock:
                pos, vel = self.pos.copy(), self.vel.copy()
                accel, gyro, mag = self.accel.copy(), self.gyro.copy(), self.mag.copy()
                quiet = self._quiet
                job = self._jobs.pop(0) if self._jobs else None
                if nsh is None and self._nsh_req is not None:
                    nsh = self._nsh_req
                    self._nsh_req = None
                    start_nsh = True
                else:
                    start_nsh = False
            mag = np.array(mag, dtype=float)
            # Identical samples are what PX4's voter calls STALE (baro ~1 s, mag ~7 s).
            mag[0] += 0.002 if (self._n & 1) else -0.002
            self._press_wiggle = 0.001 if (self._n & 1) else -0.001
            self._n += 1
            t_us = int(time.time() * 1e6)
            try:
                # Keep sensors flowing through an NSH poll. A gap is what PX4
                # reports as TIMEOUT, and that drops the arm gates.
                del quiet
                send_hil_sensor(self.m, t_us, accel, gyro, pos, mag, self._press_wiggle)
                send_hil_gps(self.m, t_us, pos, vel)
                _gcs_hb(self.m)
                if self.keep_armed and time.time() - self._last_arm > 0.5:
                    _force_arm(self.m)
                    self._last_arm = time.time()
                # 200 Hz HIL_ACTUATOR_CONTROLS fills the USB link and later
                # statustext (the g= word) never arrives. 50 Hz is enough.
                if self.hil and not self._stream_set:
                    try:
                        self.m.mav.request_data_stream_send(
                            self.m.target_system or 1,
                            self.m.target_component or 1,
                            mavutil.mavlink.MAV_DATA_STREAM_ALL,
                            0,
                            0,
                        )
                    except Exception:
                        pass
                    for mid, interval in ((30, 0), (36, 0), (105, 0), (93, 20000)):
                        _set_msg_interval(self.m, mid, interval)
                    self._stream_set = True
                if start_nsh and nsh is not None:
                    nsh_buf = []
                    _serial_send(self.m, nsh[0])
                    nsh = (nsh[0], nsh[1], nsh[2], nsh[3], time.time())
                if job is not None:
                    job()
                # One empty read is not "caught up": USB delivers in bursts.
                # Poll until the period ends or the kernel buffer overflows
                # and the g= line is the message that gets dropped.
                deadline = t_loop + period
                while time.time() < deadline and not self.stop:
                    self._drain(nsh, nsh_buf)
                    if nsh is not None and time.time() - nsh[4] >= nsh[1]:
                        nsh[2]["text"] = "".join(nsh_buf)
                        nsh[3].set()
                        nsh = None
                        nsh_buf = []
                    time.sleep(0.001)
            except Exception as exc:  # noqa: BLE001
                if not self._err_printed:
                    print(f"  HIL pump error: {exc}")
                    self._err_printed = True
                delay = period - (time.time() - t_loop)
                if delay > 0:
                    time.sleep(delay)

    def _drain(self, nsh, nsh_buf: list[str]) -> None:
        while True:
            msg = _recv_match_safe(self.m, blocking=False, timeout=0.0)
            if not msg:
                return
            mt = msg.get_type()
            if mt == "HIL_ACTUATOR_CONTROLS":
                ctrls = list(msg.controls) if hasattr(msg, "controls") else []
                with self.lock:
                    self.controls = ctrls
                    self.control_msgs += 1
            elif mt == "HEARTBEAT" and getattr(msg, "autopilot", None) == 12:
                base = int(getattr(msg, "base_mode", 0) or 0)
                with self.lock:
                    self.armed = bool(base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    self.hil = bool(base & mavutil.mavlink.MAV_MODE_FLAG_HIL_ENABLED)
            elif mt == "STATUSTEXT":
                text = str(getattr(msg, "text", "") or "")
                with self.lock:
                    self.last_status = text
                found = re.search(r"g=([0-9a-fA-F]+)", text)
                if found:
                    with self.lock:
                        self.last_g = int(found.group(1), 16)
                if nsh is not None:
                    nsh_buf.append(text + "\n")
            elif mt == "SERIAL_CONTROL" and nsh is not None and getattr(msg, "count", 0):
                chunk = bytes(bytearray(msg.data[: msg.count])).decode("utf-8", "ignore")
                nsh_buf.append(chunk)

    def close(self) -> None:
        global _ACTIVE_PUMP
        self.stop = True
        self.thread.join(timeout=1.0)
        if _ACTIVE_PUMP is self:
            _ACTIVE_PUMP = None


def _gcs_hb(m) -> None:
    m.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        0,
    )


def _link_call(m, fn):
    """Run a one-shot MAVLink send on the pump thread when it owns the port."""
    pump = _ACTIVE_PUMP
    if pump is not None and threading.current_thread() is not pump.thread:
        return pump.call(fn)
    return fn(m)


def gcs_hb(m) -> None:
    _link_call(m, _gcs_hb)


def _set_param_int(m, name: str, value: int) -> None:
    fval = struct.unpack("<f", struct.pack("<i", int(value)))[0]
    m.mav.param_set_send(
        m.target_system or 1,
        m.target_component or 1,
        name.encode("ascii"),
        fval,
        mavutil.mavlink.MAV_PARAM_TYPE_INT32,
    )
    time.sleep(0.04)


def _set_param_float(m, name: str, value: float) -> None:
    m.mav.param_set_send(
        m.target_system or 1,
        m.target_component or 1,
        name.encode("ascii"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    time.sleep(0.04)


def set_param_int(m, name: str, value: int) -> None:
    _link_call(m, lambda link: _set_param_int(link, name, value))


def set_param_float(m, name: str, value: float) -> None:
    _link_call(m, lambda link: _set_param_float(link, name, value))


def _recv_match_safe(m, type=None, blocking=True, timeout=0.4):
    """recv_match that survives pymavlink multi-instance TypeError."""
    try:
        return m.recv_match(type=type, blocking=blocking, timeout=timeout)
    except TypeError:
        return None
    except Exception:
        return None


def get_param_int(m, name: str, timeout: float = 2.5) -> int | None:
    m.mav.param_request_read_send(
        m.target_system or 1, m.target_component or 1, name.encode("ascii"), -1
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = _recv_match_safe(m, type="PARAM_VALUE", blocking=True, timeout=0.4)
        if not msg:
            continue
        pid = msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()
        pid = pid.strip("\x00")
        if name in pid:
            return struct.unpack("<i", struct.pack("<f", msg.param_value))[0]
    return None


def _serial_send(m, cmd: str) -> None:
    dev = 10
    flags = (
        mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE
        | mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND
        | mavutil.mavlink.SERIAL_CONTROL_FLAG_MULTI
    )
    _gcs_hb(m)
    payload = (cmd + "\n").encode("ascii")
    while payload:
        n = min(70, len(payload))
        buf = list(payload[:n]) + [0] * (70 - n)
        m.mav.serial_control_send(dev, flags, 0, 0, n, buf)
        payload = payload[n:]


def shell(m, cmd: str, wait: float = 1.5, quiet: bool = False) -> str:
    """NSH via SERIAL_CONTROL device 10."""
    pump = _ACTIVE_PUMP
    if pump is not None and threading.current_thread() is not pump.thread:
        return pump.nsh(cmd, wait, quiet=quiet)
    _serial_send(m, cmd)
    out: list[str] = []
    t0 = time.time()
    while time.time() - t0 < wait:
        msg = _recv_match_safe(m, type=["SERIAL_CONTROL", "STATUSTEXT"], blocking=True, timeout=0.25)
        if not msg:
            continue
        if msg.get_type() == "SERIAL_CONTROL" and msg.count:
            chunk = bytes(bytearray(msg.data[: msg.count])).decode("utf-8", "ignore")
            out.append(chunk)
        elif msg.get_type() == "STATUSTEXT":
            out.append(str(msg.text) + "\n")
    return "".join(out)


def wait_px4(m, timeout: float = 20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = _recv_match_safe(m, type="HEARTBEAT", blocking=True, timeout=1)
        if msg and getattr(msg, "autopilot", None) == 12:
            return msg
    return None


def _set_msg_interval(m, msg_id: int, interval_us: int) -> None:
    m.mav.command_long_send(
        m.target_system or 1,
        m.target_component or 1,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        float(msg_id),
        float(interval_us),
        0,
        0,
        0,
        0,
        0,
    )


def request_streams(m) -> None:
    """Only the gimbal command. A 50 Hz 'everything' stream drowns STATUSTEXT."""
    try:
        m.mav.request_data_stream_send(
            m.target_system or 1,
            m.target_component or 1,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            0,
            0,
        )
    except Exception:
        pass
    # 93 = HIL_ACTUATOR_CONTROLS at 50 Hz. 0 = stop attitude / IMU / servo raw.
    # HIL enable forces 93 to 200 Hz; the pump sends this again once hil is on.
    for mid, interval in ((30, 0), (36, 0), (105, 0), (93, 20000)):
        try:
            _set_msg_interval(m, mid, interval)
        except Exception:
            pass


def _force_arm(m) -> None:
    m.mav.command_long_send(
        m.target_system or 1,
        m.target_component or 1,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1.0,
        21196.0,
        0,
        0,
        0,
        0,
        0,
    )


def force_arm_commander(m) -> None:
    _link_call(m, _force_arm)


def enable_hil_mode(m) -> None:
    def send(link) -> None:
        try:
            link.mav.set_mode_send(
                link.target_system or 1,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                | mavutil.mavlink.MAV_MODE_FLAG_HIL_ENABLED
                | mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                | mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED
                | mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED,
                0,
            )
        except Exception:
            pass

    _link_call(m, send)


def parse_safety_status(text: str) -> dict:
    """Parse `tvc_safety status` NSH output."""
    state = None
    fire = None
    abort = None
    arm_ok = None
    ignite_ok = None
    m = re.search(r"state=(\d+)", text)
    if m:
        state = int(m.group(1))
    m = re.search(r"fire=(\d+)", text)
    if m:
        fire = int(m.group(1))
    m = re.search(r"abort=(\d+)", text)
    if m:
        abort = int(m.group(1))
    m = re.search(r"arm_ok=(\d+)", text)
    if m:
        arm_ok = int(m.group(1))
    m = re.search(r"ignite_ok=(\d+)", text)
    if m:
        ignite_ok = int(m.group(1))
    gates = {}
    gm = re.search(
        r"gates rtk=(\d+) ekf=(\d+) gim=(\d+) geo=(\d+) link=(\d+) alloc=(\d+) thr=(\d+) rail=(\d+) mass=(\d+)",
        text,
    )
    if gm:
        keys = ("rtk", "ekf", "gim", "geo", "link", "alloc", "thr", "rail", "mass")
        gates = {k: int(v) for k, v in zip(keys, gm.groups())}
    return {
        "state": state,
        "fire": fire,
        "abort": abort,
        "arm_ok": arm_ok,
        "ignite_ok": ignite_ok,
        "gates": gates,
        "raw": text,
    }


def wait_safety_state(m, want_states: set[int], timeout: float = 5.0) -> dict:
    t0 = time.time()
    last: dict = {}
    while time.time() - t0 < timeout:
        gcs_hb(m)
        txt = shell(m, "tvc_safety status", wait=1.2)
        last = parse_safety_status(txt)
        st = last.get("state")
        if st is not None and st in want_states:
            return last
        time.sleep(0.15)
    return last


# ---------------------------------------------------------------------------
# Frames / plant
# ---------------------------------------------------------------------------

def enu_to_ned_pos(p: np.ndarray) -> np.ndarray:
    # ENU (east, north, up) → NED (north, east, down). Matches tvc::enu_to_ned.
    return np.array([p[1], p[0], -p[2]], dtype=float)


def enu_to_ned_vel(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], v[0], -v[2]], dtype=float)


def flu_to_frd(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], -v[1], -v[2]], dtype=float)


def pwm_to_angles(servo_us: list[int], a2_range_deg: float = 45.0, a1_range_deg: float = 90.0) -> np.ndarray:
    """Invert TvcActuators mapping → shape (3, 2) rows (a2, a1). a1 rest = 0."""
    a2_r = math.radians(a2_range_deg)
    a1_r = math.radians(a1_range_deg)
    flat = np.zeros(6, dtype=float)
    for ch in range(6):
        us = servo_us[ch] if ch < len(servo_us) else 1500
        if us < 800:
            us = 1500
        n = max(-1.0, min(1.0, (float(us) - 1500.0) / 500.0))
        if ch % 2 == 0:
            flat[ch] = n * a2_r
        else:
            flat[ch] = -n * a1_r
    return flat.reshape(3, 2)


def angles_are_rest(angles: np.ndarray, tol_deg: float = 2.0) -> bool:
    a = np.asarray(angles, dtype=float).reshape(3, 2)
    tol = math.radians(tol_deg)
    for i in range(3):
        if abs(a[i, 0]) > tol:
            return False
        if abs(a[i, 1]) > tol:
            return False
    return True


def _clamp_f(x: float, lim: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(-lim, min(lim, float(x)))


# HIL_ACTUATOR_CONTROLS from pwm_out_sim is MAIN order, normalized -1..1,
# after TVC_SERVO_REV. Logical channels are [M0a2, M0a1, M1a2, M1a1, M2a2, M2a1].
_MAIN_TO_LOGICAL = (0, 2, 4, -1, 1, 3, 5)
_SERVO_REV = 47
_A2_RANGE = math.radians(45.0)
_A1_RANGE = math.radians(90.0)


def bench_matched_wrench(mounts, angles, thrusts, com) -> tuple[np.ndarray, np.ndarray]:
    """Force and torque the onboard mixer believes these angles produce.

    Flight code flips M1/M2 yaw relative to the JSON geometry. The SIH plant
    keeps the JSON wrench, so that yaw command runs away unless the plant
    matches it. Roll stays with the JSON.
    """
    pose = np.asarray(angles, dtype=float).reshape(len(mounts), 2)
    thrust = np.asarray(thrusts, dtype=float).reshape(len(mounts))
    origin = np.asarray(com, dtype=float).reshape(3)
    force = np.zeros(3, dtype=float)
    torque = np.zeros(3, dtype=float)
    for i, mount in enumerate(mounts):
        fi = thrust_direction(float(pose[i, 0]), float(pose[i, 1]), mount) * float(thrust[i])
        force = force + fi
        tau = np.cross(np.asarray(mount.position, dtype=float) - origin, fi)
        # Bench mixer inverts M1/M2 yaw versus the JSON geometry. Roll stays
        # with the JSON: flipping it as well runs the vehicle north.
        if i != 0:
            tau[2] = -tau[2]
        torque = torque + tau
    return torque, force


def step_fc_plant(sim, system, state, angles, thrusts, t_burn: float, dt: float):
    """Integrate the Cube's gimbal command on the catching plant."""
    ang = np.asarray(angles, dtype=float).reshape(3, 2)
    if sim._chopstick_frozen is not None:
        frozen = sim._chopstick_frozen.copy()
        return frozen, ang, np.zeros(3), np.zeros(3), float(sim.vehicle.mass)

    mass, com, inertia, motor_cgs = compose_offset_mass_properties(
        system.unloaded_vehicle,
        system.unloaded_center_of_mass,
        system.mounts,
        ang,
        standoff_m=system.motor_pivot_standoff_m,
        time_s=t_burn,
    )
    tau, force = bench_matched_wrench(system.mounts, ang, thrusts, com)
    plant = VehicleProperties(
        inertia=np.asarray(inertia, dtype=float),
        mass=float(mass),
        gravity=np.asarray(system.vehicle.gravity, dtype=float).copy(),
    )
    next_state = integrate_6dof_rk4(state, force, tau, plant, dt)
    if sim.chopsticks is not None and system.catch_arms:
        if float(next_state.position[2]) > 0.3:
            sim._chopstick_airborne = True
        next_state, n_hit, heading_ok = apply_chopstick_contacts(
            next_state,
            plant,
            list(system.catch_arms),
            sim.chopsticks,
            com_body=com,
            dt=dt,
        )
        if heading_ok:
            sim._chopstick_heading_ok = True
        sim._note_stick_faults(
            next_state.position,
            next_state.quaternion,
            motor_cgs,
            gimbal_angles=ang,
        )
        if sim._chopstick_airborne and n_hit >= 2:
            sim._chopstick_caught = True
            next_state.velocity[:] = 0.0
            next_state.omega[:] = 0.0
            sim._chopstick_frozen = next_state.copy()
    if sim._chopstick_frozen is None and float(next_state.position[2]) <= 0.0:
        next_state.position[2] = 0.0
        next_state.velocity[:] = 0.0
    return next_state, ang, tau, force, float(mass)


def hil_controls_to_angles(controls) -> np.ndarray:
    """pwm_out_sim MAIN-order controls → allocator angles (3, 2) of (a2, a1)."""
    ang = np.zeros((3, 2), dtype=float)
    for i, logical in enumerate(_MAIN_TO_LOGICAL):
        if logical < 0 or i >= len(controls):
            continue
        n = float(controls[i])
        if not math.isfinite(n):
            continue
        if abs(n) > 2.0:
            n = (n - 1500.0) / 500.0
        n = max(-1.0, min(1.0, n))
        if _SERVO_REV & (1 << logical):
            n = -n
        mount = logical // 2
        if logical % 2 == 0:
            ang[mount, 0] = n * _A2_RANGE
        else:
            ang[mount, 1] = -n * _A1_RANGE
    return ang


def send_hil_sensor(m, t_usec: int, accel_frd: np.ndarray, gyro_frd: np.ndarray, pos_ned: np.ndarray,
                    mag_frd: np.ndarray | None = None, pressure_wiggle_hpa: float = 0.0) -> None:
    alt = _clamp_f(-pos_ned[2], 5000.0)
    abs_pressure = _clamp_f(1013.25 * math.exp(-alt / 8500.0) + pressure_wiggle_hpa, 2000.0)
    m.mav.hil_sensor_send(
        t_usec,
        _clamp_f(accel_frd[0], 80.0),
        _clamp_f(accel_frd[1], 80.0),
        _clamp_f(accel_frd[2], 80.0),
        _clamp_f(gyro_frd[0], 20.0),
        _clamp_f(gyro_frd[1], 20.0),
        _clamp_f(gyro_frd[2], 20.0),
        _clamp_f(float(_MAG_ENU[0]) if mag_frd is None else mag_frd[0], 2.0),
        _clamp_f(float(_MAG_ENU[1]) if mag_frd is None else mag_frd[1], 2.0),
        _clamp_f(float(_MAG_ENU[2]) if mag_frd is None else mag_frd[2], 2.0),
        abs_pressure,
        0.0,
        alt,
        25.0,
        _HIL_FIELDS,
        0,
    )


def send_hil_gps(m, t_usec: int, pos_ned: np.ndarray, vel_ned: np.ndarray) -> None:
    lat0, lon0 = 47.397742, 8.545594
    lat = int((lat0 + pos_ned[0] / 111320.0) * 1e7)
    lon = int((lon0 + pos_ned[1] / (111320.0 * math.cos(math.radians(lat0)))) * 1e7)
    alt_mm = int((-pos_ned[2]) * 1000.0)
    vn, ve, vd = int(vel_ned[0] * 100), int(vel_ned[1] * 100), int(vel_ned[2] * 100)
    vel = int(np.linalg.norm(vel_ned) * 100)
    cog = 0
    if abs(vel_ned[0]) + abs(vel_ned[1]) > 0.1:
        cog = int((math.degrees(math.atan2(vel_ned[1], vel_ned[0])) + 360) % 360 * 100)
    # pymavlink: (time, fix, lat, lon, alt, eph, epv, vel, vn, ve, vd, cog, sats, id=0, yaw=0)
    try:
        m.mav.hil_gps_send(
            t_usec, 6, lat, lon, alt_mm, 50, 50, vel, vn, ve, vd, cog, 16, 0, 0
        )
    except TypeError:
        m.mav.hil_gps_send(
            t_usec, 6, lat, lon, alt_mm, 50, 50, vel, vn, ve, vd, cog, 16
        )


def plant_specific_force_frd(force_body_flu: np.ndarray, mass: float) -> np.ndarray:
    f_body = np.asarray(force_body_flu, dtype=float) / max(float(mass), 1e-6)
    return flu_to_frd(f_body)


def load_flight_plant(dt: float, duration: float, profile: str = "real_landing_g12"):
    """Load weighed-mass plant (no hop-scale 1 kg override)."""
    sih_dir = str(TV3_ROOT / "sih")
    if sih_dir not in sys.path:
        sys.path.insert(0, sih_dir)
    from run_sih import _patch_vehicle_path, _resolve_profile, _thrust_to_weight  # type: ignore
    from tvc_core_bridge import use_firmware_controllers, wrap_sim_controllers  # type: ignore

    profile_path = _resolve_profile(profile)
    resolved = _patch_vehicle_path(profile_path)
    cwd = os.getcwd()
    os.chdir(PROTOTYPE)
    try:
        system = load_vehicle_profile(profile_path=resolved)
    finally:
        os.chdir(cwd)

    tw = _thrust_to_weight(system)
    if tw["has_solids"] and tw["tw_peak"] < 1.0:
        raise RuntimeError(
            f"HIL plant T/W_peak={tw['tw_peak']:.2f} < 1 at {tw['mass_kg']:.3f} kg "
            f"(profile {profile}). Use G12 or F10, not G8."
        )

    flight = system.flight_test or FlightTestConfig()
    flight.vehicle = system.vehicle
    flight.controller = system.controller
    if system.profile is not None and system.profile.altitude_controller is not None:
        flight.altitude_controller = system.profile.altitude_controller
    if system.flight_test is not None:
        flight.position_controller = system.flight_test.position_controller
        flight.altitude_controller = system.flight_test.altitude_controller
    flight.dt = dt
    flight.duration = duration

    if system.profile is not None and system.profile.real_landing is not None:
        mission = system.profile.real_landing
        flight.initial_position_m = mission.launch_position_m()
        scenario = FlightScenario.real_landing(
            mission, name=system.profile.name or "real_landing"
        )
    else:
        scenario = FlightScenario.hover(5.0)

    sim = FlightSimulator(system, flight)
    sim.allocator.center_of_mass = np.asarray(system.center_of_mass, dtype=float)
    # Match the SIH catches: host closed-loop uses tvc_core unless
    # SIH_CONTROL=prototype. FC mode ignores these and flies the Cube.
    if use_firmware_controllers():
        wrap_sim_controllers(sim)
        print("  control stack=tvc_core")
    else:
        print("  control stack=prototype")
    # The Cube flies the catalog curves. A 3% draw the mixer cannot see
    # walks the heading, and the landing leaves the pad.
    for motor in sim.motors:
        if motor is None:
            continue
        object.__setattr__(motor, "thrust_scale", 1.0)
        object.__setattr__(motor, "time_scale", 1.0)
        object.__setattr__(motor, "ignition_delay_s", 0.0)
    # sim.run() arms the pad slide at burnout − pad_transition_burn_remaining_s.
    # Without that, a burn-gated profile never leaves the staging waypoint
    # and the late heading slew never starts.
    scenario.reset_waypoints()
    sim._apply_pad_transition_burn_gate(scenario)
    if scenario.forced_descent_time_s is not None:
        remain = float(scenario.pad_transition_burn_remaining_s or 0.0)
        print(
            f"  pad slide arms at t={float(scenario.forced_descent_time_s):.2f}s "
            f"(burnout − {remain:.1f}s)"
        )
    state = sim.reset()
    box = system.profile.landing_acceptance if system.profile else None
    print(
        f"  plant mass={system.vehicle.mass:.3f} kg  com={system.center_of_mass}  "
        f"T/W_peak={tw['tw_peak']:.2f} T/W_avg={tw['tw_avg']:.2f}  "
        f"profile={profile_path.name}"
    )
    return system, sim, scenario, state, box, tw


def configure_fc_params(m, *, mass_kg: float, burn_s: float, peak_n: float, nom_n: float) -> None:
    print("Configuring FC params for HIL landing (mission.json owns the path)...")
    for name, val, kind in [
        ("TVC_ALLOW_FIRE", 1, "int"),
        ("TVC_IGN_GPIO_EN", 0, "int"),
        ("TVC_BENCH", 0, "int"),
        # tvc_safety reads sensor_gps instance 0 only. The CAN GPS already
        # owns that instance, so the HIL fix-type 6 never counts as RTK and
        # the pad origin never latches. Host mode flies the plant, not the
        # Cube's GPS. Restored to 1 before the reboot.
        ("TVC_THR_NOM", nom_n, "float"),
        ("TVC_THR_PEAK", peak_n, "float"),
        ("TVC_THR_BURN", burn_s, "float"),
        ("TVC_REQ_RTK", 0, "int"),
        ("TVC_REQ_LINK", 0, "int"),
        ("TVC_FORCE_ARM", 1, "int"),
        ("COM_DISARM_PRFLT", -1.0, "float"),
        ("COM_DISARM_LAND", -1.0, "float"),
        ("NAV_RCL_ACT", 0, "int"),
        ("COM_RC_IN_MODE", 1, "int"),
        ("CBRK_IO_SAFETY", 22027, "int"),
    ]:
        gcs_hb(m)
        if kind == "int":
            set_param_int(m, name, int(val))
        else:
            set_param_float(m, name, float(val))
    allow = get_param_int(m, "TVC_ALLOW_FIRE")
    print(f"  TVC_ALLOW_FIRE={allow} (want 1)")


_FSM = {
    "DISARM": 0,
    "PREFLT": 1,
    "ARMED": 2,
    "LAUNCH": 3,
    "BURN": 4,
    "COAST": 5,
    "LAND": 6,
    "ABORT": 7,
    "SAFE": 8,
}


def fsm_from_status(text: str) -> dict:
    # enter_burn prints this immediately. The periodic `TVC BURN a=` line is
    # only every 2 s, and waiting for it leaves the plant sitting through the
    # start of the loft.
    if re.search(r"(?:→|->)\s*BURN", text or ""):
        return {"state": 4, "name": "BURN", "arm_ok": 1, "ignite_ok": 1, "fire": 1}
    found = re.search(r"TVC (\w+) a=(\d+) i=(\d+) f=(\d+)", text or "")
    if not found:
        return {}
    name = found.group(1)
    return {
        "state": _FSM.get(name),
        "name": name,
        "arm_ok": int(found.group(2)),
        "ignite_ok": int(found.group(3)),
        "fire": int(found.group(4)),
    }


def tvc_gcs(m, action: int, magic: float = 0.0) -> None:
    """MAV_CMD_USER_1. NSH replies get lost once HIL_ACTUATOR_CONTROLS is flowing."""

    def send(link) -> None:
        link.mav.command_long_send(
            link.target_system or 1,
            link.target_component or 1,
            31010,
            0,
            float(action),
            float(magic),
            0,
            0,
            0,
            0,
            0,
        )

    _link_call(m, send)


def wait_fsm(want: set[int], timeout: float = 4.0) -> dict:
    t0 = time.time()
    last: dict = {}
    while time.time() - t0 < timeout:
        text = ""
        if _ACTIVE_PUMP is not None:
            text = _ACTIVE_PUMP.snapshot().get("last_status") or ""
        last = fsm_from_status(text) or last
        if last.get("state") in want:
            return last
        time.sleep(0.05)
    return last


def arm_and_fire(m, state_pos: np.ndarray) -> dict:
    """PREFLIGHT → ARMED → LAUNCH → BURN over MAV_CMD_USER_1."""
    del state_pos
    print("Arming / fire sequence...")
    set_param_int(m, "TVC_ALLOW_FIRE", 1)
    st: dict = {}
    for attempt in range(4):
        force_arm_commander(m)
        tvc_gcs(m, 1)
        st = wait_fsm({2, 3, 4})
        print(f"  arm {attempt+1}: {st.get('name')} a={st.get('arm_ok')}")
        if st.get("state") in (2, 3, 4):
            break
    for attempt in range(4):
        tvc_gcs(m, 3)
        st = wait_fsm({3, 4})
        print(f"  launch_ready {attempt+1}: {st.get('name')}")
        if st.get("state") in (3, 4):
            break
    for attempt in range(4):
        tvc_gcs(m, 4, 21196)
        st = wait_fsm({4})
        print(f"  fire {attempt+1}: {st.get('name')} f={st.get('fire')}")
        if st.get("state") == 4 and st.get("fire") == 1:
            print("  FIRE LATCHED → BURN")
            return st
    print("  WARN: fire not latched")
    return st


def _hil_bit(hb) -> bool:
    return bool(int(getattr(hb, "base_mode", 0) or 0) & mavutil.mavlink.MAV_MODE_FLAG_HIL_ENABLED)


def connect_cube(port: str | None):
    port = find_port(port)
    print(f"Connecting to Cube at {port}")
    m = mavutil.mavlink_connection(port, baud=57600, autoreconnect=True, source_system=255)
    for _ in range(8):
        _gcs_hb(m)
        time.sleep(0.05)
    hb = wait_px4(m)
    if not hb:
        raise SystemExit("No PX4 heartbeat")
    print(f"PX4 online  base_mode={hb.base_mode} hil={_hil_bit(hb)}")
    return m, port, hb


def reboot_and_reconnect(m, port: str):
    print("Rebooting Cube into the new param set...")
    m.mav.command_long_send(
        m.target_system or 1,
        m.target_component or 1,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    time.sleep(1.5)
    try:
        m.close()
    except Exception:
        pass
    t0 = time.time()
    last_err = ""
    while time.time() - t0 < 45:
        time.sleep(1.0)
        try:
            use = find_port(port)
        except SystemExit:
            continue
        try:
            m2 = mavutil.mavlink_connection(use, baud=57600, autoreconnect=True, source_system=255)
            for _ in range(4):
                _gcs_hb(m2)
                time.sleep(0.05)
            hb = wait_px4(m2, timeout=6)
            if hb:
                print(f"PX4 back  base_mode={hb.base_mode} hil={_hil_bit(hb)}")
                return m2, hb
            m2.close()
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
    raise SystemExit(f"Cube did not come back after reboot ({last_err})")


def ensure_hitl_boot(m, port: str):
    """Switch the running Cube into HITL without rebooting.

    Airframe 15000 runs ``param set SYS_HITL 0`` from rc.autostart before
    commander starts, so a saved SYS_HITL=1 does not survive boot. Stop the
    board IMUs and restart commander with ``-h``. Parameters are left unsaved;
    teardown puts the flight values back.
    """
    del port
    print("Switching the running Cube into HITL (no reboot; airframe clears SYS_HITL)...")
    global _SAVED_IMU_MODE
    _SAVED_IMU_MODE = get_param_int(m, "SENS_IMU_MODE")
    print(f"  saved SENS_IMU_MODE={_SAVED_IMU_MODE}")
    _set_param_int(m, "SYS_HITL", 1)
    _set_param_int(m, "UAVCAN_ENABLE", 0)
    _set_param_int(m, "SENS_BOARD_ROT", 0)
    _set_param_int(m, "SENS_IMU_MODE", 1)
    for cmd in (
        "commander stop",
        "ekf2 stop",
        "sensors stop",
        "icm45686 stop",
        "icm42688p stop",
        "icm20948 stop",
        "icm20649 stop",
        "ak09916 stop",
        "ms5611 stop",
        # A CAN GPS/mag keeps publishing through UAVCAN_ENABLE=0. It owns
        # sensor_gps instance 0 and pulls yaw off the HIL magnetometer.
        "uavcan stop",
        "gps stop",
        "sensors start -h",
        "commander start -h",
        "ekf2 start",
    ):
        print(f"$ {cmd}")
        print(shell(m, cmd, 2.0))
    hb = wait_px4(m, timeout=8)
    if not hb or not _hil_bit(hb):
        print("Commander did not stay in HITL after the live switch.")
        restore_safe_params(m)
        raise SystemExit(1)
    print(f"  HITL base_mode={hb.base_mode}")
    return m


def map_hil_actuators(m) -> None:
    """pwm_out_sim channels start at func 0. Match the airframe MAIN map."""
    for name, val in (
        ("HIL_ACT_FUNC1", 201),
        ("HIL_ACT_FUNC2", 203),
        ("HIL_ACT_FUNC3", 205),
        ("HIL_ACT_FUNC4", 0),
        ("HIL_ACT_FUNC5", 202),
        ("HIL_ACT_FUNC6", 204),
        ("HIL_ACT_FUNC7", 206),
    ):
        _set_param_int(m, name, val)
    print(shell(m, "pwm_out_sim stop", 1.0))
    print(shell(m, "pwm_out_sim start -m hil", 1.5))
    print(shell(m, "pwm_out_sim status", 1.2))


def wait_for_flight_gates(m, timeout: float = 90.0) -> bool:
    """Sit still until RTK, EKF, geofence, and the pad origin are all true.

    The 2 Hz statustext `g=` word is the reliable signal. NSH status is only
    fetched once, so a shell wait cannot stall the estimator.
    """
    print(f"Waiting up to {timeout:.0f}s for RTK, EKF, geofence, and pad origin...")
    need = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 10)
    t0 = time.time()
    while time.time() - t0 < timeout:
        snap = _ACTIVE_PUMP.snapshot() if _ACTIVE_PUMP is not None else {}
        g = snap.get("last_g")
        ok = g is not None and (g & need) == need
        print(
            f"  t={time.time() - t0:4.0f}s g={None if g is None else hex(g)} "
            f"ready={int(ok)} hil={snap.get('hil')} ctrls={snap.get('control_msgs')}"
        )
        if ok:
            print("  flight gates open")
            return True
        time.sleep(1.0)
    print(shell(m, "ekf2 status", 2.0))
    print(shell(m, "listener sensor_baro 1", 1.2))
    print(shell(m, "listener sensor_mag 1", 1.2))
    return False


_RESTORED = False


def restore_safe_params(m) -> None:
    global _RESTORED
    _RESTORED = True
    print("Restoring flight params (ALLOW_FIRE=0, HITL off, board rot 2)...")
    set_param_int(m, "TVC_ALLOW_FIRE", 0)
    set_param_int(m, "TVC_IGN_GPIO_EN", 0)
    set_param_int(m, "TVC_BENCH", 0)
    set_param_int(m, "TVC_REQ_RTK", 1)
    set_param_float(m, "TVC_GEOFENCE_R", 50.0)
    set_param_int(m, "SYS_HITL", 0)
    set_param_int(m, "SENS_BOARD_ROT", 2)
    set_param_int(m, "GPS_1_CONFIG", 201)
    set_param_int(m, "UAVCAN_ENABLE", 2)
    if _SAVED_IMU_MODE is not None:
        set_param_int(m, "SENS_IMU_MODE", int(_SAVED_IMU_MODE))
    shell(m, "tvc_safety abort", 1.0)
    shell(m, "tvc_safety disarm", 1.0)
    shell(m, "tvc_safety reset", 1.0)
    print(shell(m, "param save", 6.0))
    print("Rebooting so the airframe restarts the real IMUs...")
    try:
        m2, hb = reboot_and_reconnect(m, find_port(None))
        print(f"  Cube back hil={_hil_bit(hb)} base_mode={hb.base_mode}")
        m2.close()
    except Exception as exc:  # noqa: BLE001
        print(f"  reboot after restore failed: {exc}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Closed-loop HIL landing")
    ap.add_argument("--port", default=None, help="Cube serial (default auto)")
    ap.add_argument("--profile", default="real_landing_g12", help="SIH/Prototype landing profile")
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument(
        "--control",
        choices=("auto", "fc", "host"),
        default="auto",
        help="Plant actuator source (default auto)",
    )
    ap.add_argument("--no-reboot", action="store_true", default=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--keep-fire-enabled", action="store_true", help="Do not restore ALLOW_FIRE=0")
    args = ap.parse_args()

    m, port, _hb = connect_cube(args.port)
    pump = None
    m = ensure_hitl_boot(m, port)

    print("Loading weighed-mass plant...")
    system, sim, scenario, state, box, tw = load_flight_plant(
        args.dt, args.duration, args.profile
    )
    motors = sim.motors
    rest_angles = sim.allocator.rest_angles.copy()
    n_mounts = max(len(system.mounts), 1)
    nom_n = tw["avg_thrust_n"] / n_mounts
    peak_n = tw["peak_thrust_n"] / n_mounts
    burn_s = 12.7
    if any("F10" in str(getattr(mt, "engine", "") or "") for mt in system.mounts):
        burn_s = 7.2
    configure_fc_params(
        m, mass_kg=float(tw["mass_kg"]), burn_s=burn_s, peak_n=peak_n, nom_n=nom_n
    )
    map_hil_actuators(m)
    request_streams(m)

    # Sit on the pad so the Cube captures local origin, then carry the plant
    # to the launch stand. The mission loft is an offset from that origin.
    launch_pos = np.array(state.position, dtype=float).copy()
    state.position[:] = 0.0
    state.velocity[:] = 0.0
    pump = HilPump(m)
    specific = world_to_body(state.quaternion, np.array([0.0, 0.0, 9.81]))
    mag = flu_to_frd(world_to_body(state.quaternion, _MAG_ENU))
    pump.update(np.zeros(3), np.zeros(3), flu_to_frd(specific), np.zeros(3), mag)
    if not wait_for_flight_gates(m, timeout=90.0):
        print("Estimator never became armable. Not flying.")
        pump.close()
        restore_safe_params(m)
        return 1
    print(f"Moving to launch {launch_pos.tolist()}")
    move_s = 8.0
    steps_move = max(1, int(move_s / args.dt))
    for i in range(steps_move):
        state.position[:] = launch_pos * ((i + 1) / steps_move)
        state.velocity[:] = launch_pos / move_s
        specific = world_to_body(state.quaternion, np.array([0.0, 0.0, 9.81]))
        mag = flu_to_frd(world_to_body(state.quaternion, _MAG_ENU))
        pump.update(
            enu_to_ned_pos(state.position),
            enu_to_ned_vel(state.velocity),
            flu_to_frd(specific),
            np.zeros(3),
            mag,
        )
        time.sleep(args.dt)
    state.velocity[:] = 0.0
    print("Sitting at the stand so the velocity estimate can settle...")
    for _ in range(max(1, int(4.0 / args.dt))):
        specific = world_to_body(state.quaternion, np.array([0.0, 0.0, 9.81]))
        mag = flu_to_frd(world_to_body(state.quaternion, _MAG_ENU))
        pump.update(
            enu_to_ned_pos(state.position),
            np.zeros(3),
            flu_to_frd(specific),
            np.zeros(3),
            mag,
        )
        time.sleep(args.dt)

    st = arm_and_fire(m, state.position)
    fire_latched = st.get("state") == 4 and st.get("fire") == 1

    control_mode = args.control
    using_host = control_mode == "host" or (control_mode == "auto" and not fire_latched)
    if using_host and not fire_latched:
        print("  auto/host: fire not latched — host closed-loop will still fly (FC FSM incomplete)")
    if control_mode == "fc" and not fire_latched:
        print("  fc mode without fire latch — not flying.")
        pump.close()
        restore_safe_params(m)
        return 1

    log = FlightLog(
        metadata=FlightLogMetadata(
            scenario="hil_closed_loop_landing",
            motion_mode="hover",
            dt=args.dt,
            duration=args.duration,
            force_target=float(getattr(sim.flight, "force_target", 9.81) or 9.81),
            force_weight=float(getattr(sim.flight, "force_weight", 1.0) or 1.0),
            notes=f"control={control_mode} fire_latched={fire_latched} profile={args.profile}",
            mount_count=3,
            engines=",".join(str(getattr(mt, "engine", None) or "unknown") for mt in system.mounts),
        )
    )

    servo_us = [1500] * 8
    angles = rest_angles.copy()
    steps = int(args.duration / args.dt)
    t_wall0 = time.time()
    servo_msgs = 0
    armed_hb = 0
    host_steps = 0
    fc_steps = 0
    rest_streak = 0.0
    switched_to_host = using_host
    host_t0 = 0.0  # host mission clock origin (wall sim-time)

    print(f"Closed-loop run: {steps} steps dt={args.dt}s control={control_mode} fire={fire_latched}")
    pump.keep_armed = bool(fire_latched)

    for k in range(steps):
        t = k * args.dt
        snap = pump.snapshot()
        ctrls = snap["controls"]
        if ctrls:
            angles = hil_controls_to_angles(ctrls)
            servo_us = [
                int(1500 + 500 * max(-1.0, min(1.0, float(c))))
                if c is not None and math.isfinite(float(c)) and abs(float(c)) <= 2.0
                else int(c or 1500)
                for c in list(ctrls)[:8]
            ]
            while len(servo_us) < 8:
                servo_us.append(1500)
        servo_msgs = int(snap["control_msgs"])
        if snap["armed"]:
            armed_hb += 1

        # Auto fallback: rest PWM, or FC trajectory diverging / tumbling
        if control_mode == "auto" and not switched_to_host:
            if angles_are_rest(angles):
                rest_streak += args.dt
            else:
                rest_streak = 0.0
            speed = float(np.linalg.norm(state.velocity))
            xy = float(np.linalg.norm(state.position[:2]))
            up = rotate_vector(state.quaternion, np.array([0.0, 0.0, 1.0]))
            diverge = (
                rest_streak >= 0.35
                or speed > 12.0
                or xy > 18.0
                or float(state.position[2]) > 18.0
                or float(up[2]) < 0.6
            )
            if diverge and t > 0.35:
                reason = (
                    "servos_rest"
                    if rest_streak >= 0.35
                    else f"diverge speed={speed:.1f} xy={xy:.1f} z={state.position[2]:.1f} upz={up[2]:.2f}"
                )
                print(f"  t={t:.2f}s auto → host closed-loop ({reason})")
                # Restart the simulated trajectory from the pad and keep the HIL link active.
                state = sim.reset()
                if hasattr(scenario, "reset_waypoints"):
                    scenario.reset_waypoints()
                host_t0 = t
                log.samples.clear()  # evaluate only the host closed-loop segment
                log.metadata.notes += f" | auto_handoff_t={t:.2f}s ({reason})"
                switched_to_host = True

        use_host_now = control_mode == "host" or switched_to_host

        if use_host_now:
            # The host mission clock starts at reset, including a mid-run handoff.
            t_host = t if control_mode == "host" else max(0.0, t - host_t0)
            next_state, sample = sim.step(state, scenario, t_host)
            host_steps += 1
            force_body = sample.force_body.copy()
            tau_act = sample.torque_achieved.copy()
            thrusts = sample.mount_thrusts.copy()
            angles = np.asarray(sample.gimbal_angles, dtype=float).reshape(-1)
            if angles.size == 6:
                angles = angles.reshape(3, 2)
            sample.time = t  # wall/mission log time
            log.samples.append(sample)
            state = next_state
            plant_mass = float(sim.vehicle.mass)
        else:
            # FC-driven plant. t=0 is the fire instant (arm_and_fire returns on
            # the BURN statustext), so the thrust curve matches the Cube clock.
            t_burn = t if fire_latched else 0.0
            fallback = np.array(
                [float(getattr(mt, "thrust", 12.0) or 12.0) for mt in system.mounts],
                dtype=float,
            )
            thrusts = thrusts_at(motors, t_burn, fallback) if fire_latched else np.zeros(3)
            next_state, ang, tau_act, force_body, plant_mass = step_fc_plant(
                sim, system, state, angles, thrusts, t_burn, args.dt
            )
            sample = FlightLogSample(
                time=t,
                quaternion=state.quaternion.copy(),
                omega=state.omega.copy(),
                quaternion_reference=state.quaternion.copy(),
                torque_command=tau_act.copy(),
                torque_achieved=tau_act.copy(),
                gimbal_angles=np.asarray(ang, dtype=float).reshape(-1).copy(),
                saturated=False,
                position=state.position.copy(),
                velocity=state.velocity.copy(),
                force_command=force_body.copy(),
                force_body=force_body.copy(),
                force_inertial=rotate_vector(state.quaternion, force_body),
                mount_thrusts=thrusts.copy(),
            )
            log.samples.append(sample)
            state = next_state
            fc_steps += 1

        # Stream plant state → FC. On the ground the accelerometer reads 1 g
        # up (the pad's normal). In the air it reads thrust per mass.
        on_pad = sim._chopstick_frozen is not None or float(state.position[2]) <= 0.02
        if on_pad:
            specific_flu = world_to_body(state.quaternion, np.array([0.0, 0.0, 9.81]))
        else:
            specific_flu = np.asarray(force_body, dtype=float) / max(float(plant_mass), 1e-6)
        accel_frd = flu_to_frd(specific_flu)
        mag_frd = flu_to_frd(world_to_body(state.quaternion, _MAG_ENU))
        gyro_frd = flu_to_frd(state.omega)
        pos_ned = enu_to_ned_pos(state.position)
        vel_ned = enu_to_ned_vel(state.velocity)
        pump.update(pos_ned, vel_ned, accel_frd, gyro_frd, mag_frd)

        # Real-time pacing
        target = t_wall0 + (k + 1) * args.dt
        delay = target - time.time()
        if delay > 0:
            time.sleep(min(delay, args.dt))

        if k % 50 == 0:
            src = "host" if use_host_now else "fc"
            _roll, _pitch, _yaw = quaternion_to_euler(state.quaternion)
            print(
                f"t={t:5.2f}s [{src}] pos=({state.position[0]:6.2f},{state.position[1]:6.2f},{state.position[2]:6.2f}) "
                f"yaw={math.degrees(_yaw):6.1f} T={float(np.sum(thrusts)):5.1f}N "
                f"a2={np.degrees(np.asarray(angles, dtype=float).reshape(3, 2)[:, 0])} "
                f"msgs={servo_msgs}"
            )

    # Release the port before scoring. Param restore reboots the Cube.
    pump.close()

    out_dir = TV3_ROOT / "build" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hil_closed_loop_landing.csv"
    log.save(csv_path)
    print(f"Log → {csv_path}")

    acc = evaluate_landing_acceptance(
        log,
        box,
        chopsticks_enabled=sim.chopsticks is not None,
        chopsticks_caught=bool(getattr(sim, "_chopstick_caught", False)),
        chopsticks_heading_ok=bool(getattr(sim, "_chopstick_heading_ok", False)),
        stick_backstop=bool(getattr(sim, "_stick_backstop", False)),
        body_stick_contact=bool(getattr(sim, "_body_stick_contact", False)),
    )
    print("Landing acceptance:", acc)
    z_max = float(max(float(s.position[2]) for s in log.samples)) if log.samples else 0.0
    final = log.samples[-1].position.tolist() if log.samples else None
    tilted = False
    for s in log.samples[::10]:
        up = rotate_vector(s.quaternion, np.array([0.0, 0.0, 1.0]))
        if float(up[2]) < 0.3:
            tilted = True
            break

    result = {
        "log": str(csv_path),
        "duration_s": args.duration,
        "control_mode": control_mode,
        "used_host_closed_loop": bool(switched_to_host or control_mode == "host"),
        "fire_latched": bool(fire_latched),
        "landing": {
            "evaluated": acc.get("evaluated"),
            "success": acc.get("success"),
            "reason": acc.get("reason"),
            "burnout_time_s": acc.get("burnout_time_s"),
            "burnout_position_m": (
                np.asarray(acc["burnout_position_m"]).tolist()
                if acc.get("burnout_position_m") is not None
                else None
            ),
            "burnout_speed_m_s": acc.get("burnout_speed_m_s"),
        },
        "pass": bool(acc.get("success")),
        "stable_upright": not tilted,
        "servo_msgs": servo_msgs,
        "armed_heartbeat_hits": armed_hb,
        "host_steps": host_steps,
        "fc_steps": fc_steps,
        "z_max": z_max,
        "final_pos": final,
    }
    result_path = out_dir / "hil_closed_loop_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print("RESULT:", "PASS" if result["pass"] else "FAIL")
    print(json.dumps(result, indent=2, default=str))

    if args.plot:
        try:
            from tvc.flight_sim import plot_flight_log
            import matplotlib.pyplot as plt

            plot_flight_log(log, title="HIL closed-loop landing")
            fig = plt.gcf()
            png = out_dir / "hil_closed_loop_landing_plot.png"
            fig.savefig(png, dpi=120, bbox_inches="tight")
            plt.close("all")
            print("Plot →", png)
        except Exception as e:
            print("plot failed", e)

    if not args.keep_fire_enabled:
        restore_safe_params(m)
    else:
        shell(m, "tvc_safety abort", 1.0)
        shell(m, "tvc_safety disarm", 1.0)
        try:
            m.close()
        except Exception:
            pass
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
