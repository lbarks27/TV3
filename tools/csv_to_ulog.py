#!/usr/bin/env python3
"""Convert Prototype / SIH flight_log CSV (flight_log_v1) to a PX4 ULog (.ulg).

Writes uORB-style topics that mirror the TV3 out-of-tree messages plus a few
stock-shaped topics so Flight Review / pyulog / QGC log tools can open them.

Topics:
  - flight_log_vl          (aggregated flight_log_v1 snapshot)
  - tvc_control_debug
  - tvc_gimbal_cmd
  - tvc_mount_thrust
  - tvc_setpoint
  - tvc_force_sp
  - vehicle_local_position  (NED, converted from ENU-sim)
  - vehicle_attitude        (NED←FRD, converted from ENU-sim / FLU-sim)

Usage::

    python tools/csv_to_ulog.py sih/logs/real_landing_g12.csv -o sih/logs/real_landing_g12.ulg
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# Allow importing Prototype flight log + frames via TV3 sim_source / PROTOTYPE_ROOT
TV3 = Path(__file__).resolve().parents[1]
PROTOTYPE = Path(
    __import__("os").environ.get("PROTOTYPE_ROOT", TV3 / "sim_source")
).resolve()
sys.path.insert(0, str(PROTOTYPE))
sys.path.insert(0, str(TV3 / "src" / "lib" / "tvc_core" / "include"))  # not python

from tvc.flight_log import FlightLog  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal ULog writer (little-endian), format version 1
# Spec: https://docs.px4.io/main/en/dev_log/ulog_file_format.html
# ---------------------------------------------------------------------------


def _hdr(msg_type: bytes, payload: bytes) -> bytes:
    assert len(msg_type) == 1
    return struct.pack("<HB", len(payload), msg_type[0]) + payload


class ULogWriter:
    def __init__(self, path: Path, start_us: int = 0) -> None:
        self.path = Path(path)
        self.start_us = int(start_us)
        self._defs = bytearray()
        self._data = bytearray()
        self._next_msg_id = 0
        self._formats: dict[str, int] = {}  # name -> msg_id
        self._wrote_flag_bits = False

    def _write_flag_bits(self) -> None:
        # Must be first message after header
        payload = bytes(8) + bytes(8) + struct.pack("<QQQ", 0, 0, 0)
        self._defs += _hdr(b"B", payload)
        self._wrote_flag_bits = True

    def add_format(self, format_string: str) -> None:
        if not self._wrote_flag_bits:
            self._write_flag_bits()
        self._defs += _hdr(b"F", format_string.encode("utf-8"))

    def add_info_string(self, key: str, value: str) -> None:
        if not self._wrote_flag_bits:
            self._write_flag_bits()
        # key is "char[N] name"
        val_b = value.encode("utf-8")
        key_s = f"char[{len(val_b)}] {key}"
        key_b = key_s.encode("utf-8")
        payload = bytes([len(key_b)]) + key_b + val_b
        self._defs += _hdr(b"I", payload)

    def subscribe(self, message_name: str, multi_id: int = 0) -> int:
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        name_b = message_name.encode("utf-8")
        payload = struct.pack("<BH", multi_id, msg_id) + name_b
        self._data += _hdr(b"A", payload)
        self._formats[message_name] = msg_id
        return msg_id

    def write_data(self, msg_id: int, data: bytes) -> None:
        payload = struct.pack("<H", msg_id) + data
        self._data += _hdr(b"D", payload)

    def write_log_string(self, timestamp_us: int, message: str, level: int = ord("6")) -> None:
        # LOG_INFO = 6
        msg_b = message.encode("utf-8")
        payload = struct.pack("<BQ", level, timestamp_us) + msg_b
        self._data += _hdr(b"L", payload)

    def finish(self) -> None:
        magic = b"ULog\x01\x12\x35"
        version = bytes([1])
        header = magic + version + struct.pack("<Q", self.start_us)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            f.write(header)
            f.write(self._defs)
            f.write(self._data)


def _enu_to_ned(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], v[0], -v[2]], dtype=np.float32)


def _q_sim_to_px4(q: np.ndarray) -> np.ndarray:
    """ENU←FLU wxyz → NED←FRD wxyz via DCM product (matches tvc_core frames)."""
    # Build DCM from quaternion (body→world)
    w, x, y, z = [float(c) for c in q]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    R_ned_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    R_frd_flu = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    R_flu_frd = R_frd_flu.T
    R_ned_frd = R_ned_enu @ R @ R_flu_frd
    # DCM → quat wxyz
    m = R_ned_frd
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    qout = np.array([qw, qx, qy, qz], dtype=np.float32)
    n = float(np.linalg.norm(qout))
    return qout / n if n > 0 else np.array([1, 0, 0, 0], dtype=np.float32)


def _phase_u8(phase: str) -> int:
    p = (phase or "").lower()
    if p == "approach":
        return 1
    if p == "landing":
        return 2
    return 0


def convert_csv_to_ulog(csv_path: Path, ulg_path: Path) -> dict:
    log = FlightLog.load(csv_path)
    if not log.samples:
        raise ValueError(f"empty log: {csv_path}")

    t0_us = 0
    writer = ULogWriter(ulg_path, start_us=t0_us)

    # Formats (must match packing order below)
    writer.add_format(
        "tvc_control_debug:uint64_t timestamp;float[3] tau_cmd;float[3] tau_act;"
        "float[3] force_body;float[4] q;float[4] q_ref;float[3] omega;"
        "float[3] att_integral;float alt_integral;uint8_t mission_phase;"
    )
    writer.add_format(
        "tvc_gimbal_cmd:uint64_t timestamp;float[6] angles_rad;uint8_t saturated;"
        "uint8_t timed_out;uint32_t solve_us;float residual;"
    )
    writer.add_format(
        "tvc_mount_thrust:uint64_t timestamp;float[3] thrust;uint8_t[3] valid;"
        "uint8_t[3] source;float burn_time_s;uint8_t[3] lit;uint8_t[3] burned_out;"
    )
    writer.add_format(
        "tvc_setpoint:uint64_t timestamp;float[3] pos;float[3] vel;float heading_rad;"
        "uint8_t phase;float path_speed_ff;"
    )
    writer.add_format(
        "tvc_force_sp:uint64_t timestamp;float f_axial_n;float force_weight;uint8_t mix_armed;"
    )
    writer.add_format(
        "vehicle_local_position:uint64_t timestamp;uint64_t timestamp_sample;"
        "float x;float y;float z;float vx;float vy;float vz;"
        "uint8_t xy_valid;uint8_t z_valid;uint8_t v_xy_valid;uint8_t v_z_valid;"
    )
    writer.add_format(
        "vehicle_attitude:uint64_t timestamp;uint64_t timestamp_sample;float[4] q;"
    )
    writer.add_format(
        "flight_log_vl:uint64_t timestamp;float time_s;float[4] q;float[4] q_ref;"
        "float[3] omega;float[3] tau_cmd;float[3] tau_act;float[6] gimbal_angles;"
        "float[3] position;float[3] velocity;float[3] position_ref;float[3] velocity_ref;"
        "float[3] force_body;float[3] mount_thrusts;float force_target_axial;"
        "float force_weight;float[3] att_integral;float alt_integral;float heading_ref_rad;"
        "uint8_t mission_phase;uint8_t vehicle_state;uint8_t abort_reason;"
        "uint8_t saturated;uint8_t mix_armed;uint8_t fire_latched;"
    )

    meta = log.metadata
    writer.add_info_string("sys_name", "TV3-SIH")
    writer.add_info_string("ver_hw", "sih_prototype_plant")
    writer.add_info_string("ver_sw", "tv3-sih")
    writer.add_info_string("log_type", "sih_flight_log_v1")
    writer.add_info_string("tvc_scenario", str(getattr(meta, "scenario", "") or ""))
    writer.add_info_string(
        "tvc_landing_success", str(getattr(meta, "landing_success", "") or "")
    )
    writer.add_info_string("tvc_source_csv", str(csv_path.name))

    id_dbg = writer.subscribe("tvc_control_debug")
    id_gim = writer.subscribe("tvc_gimbal_cmd")
    id_thr = writer.subscribe("tvc_mount_thrust")
    id_sp = writer.subscribe("tvc_setpoint")
    id_fsp = writer.subscribe("tvc_force_sp")
    id_lpos = writer.subscribe("vehicle_local_position")
    id_att = writer.subscribe("vehicle_attitude")
    id_vl = writer.subscribe("flight_log_vl")

    writer.write_log_string(
        0,
        f"TV3 SIH export from {csv_path.name} samples={len(log.samples)}",
    )

    for s in log.samples:
        ts = int(round(float(s.time) * 1e6))
        q = np.asarray(s.quaternion, dtype=float)
        q_ref = np.asarray(s.quaternion_reference, dtype=float)
        omega = np.asarray(s.omega, dtype=float)
        tau_cmd = np.asarray(s.torque_command, dtype=np.float32)
        tau_act = np.asarray(s.torque_achieved, dtype=np.float32)
        f_body = np.asarray(s.force_body, dtype=np.float32)
        angles = np.asarray(s.gimbal_angles, dtype=np.float32).reshape(-1)
        if angles.size < 6:
            angles = np.pad(angles, (0, 6 - angles.size))
        angles = angles[:6]
        thrusts = np.asarray(s.mount_thrusts, dtype=np.float32).reshape(-1)
        if thrusts.size < 3:
            thrusts = np.pad(thrusts, (0, 3 - thrusts.size))
        thrusts = thrusts[:3]
        pos = np.asarray(s.position, dtype=float)
        vel = np.asarray(s.velocity, dtype=float)
        pref = np.asarray(s.position_reference, dtype=float)
        vref = np.asarray(s.velocity_reference, dtype=float)

        # tvc_control_debug
        phase = _phase_u8(s.mission_phase)
        att_int = np.asarray(s.att_integral, dtype=np.float32)
        data = struct.pack("<Q", ts)
        data += tau_cmd.astype(np.float32).tobytes()
        data += tau_act.astype(np.float32).tobytes()
        data += f_body.astype(np.float32).tobytes()
        data += np.asarray(q, dtype=np.float32).tobytes()
        data += np.asarray(q_ref, dtype=np.float32).tobytes()
        data += np.asarray(omega, dtype=np.float32).tobytes()
        data += att_int.astype(np.float32).tobytes()
        data += struct.pack("<fB", float(s.alt_integral), phase)
        writer.write_data(id_dbg, data)

        # tvc_gimbal_cmd
        data = struct.pack("<Q", ts)
        data += angles.astype(np.float32).tobytes()
        data += struct.pack("<BBIf", 1 if s.saturated else 0, 0, 0, 0.0)
        writer.write_data(id_gim, data)

        # tvc_mount_thrust
        lit = (thrusts > 0.5).astype(np.uint8)
        burned = (thrusts < 0.25).astype(np.uint8)
        valid = np.ones(3, dtype=np.uint8)
        source = np.ones(3, dtype=np.uint8)  # curve
        data = struct.pack("<Q", ts)
        data += thrusts.astype(np.float32).tobytes()
        data += valid.tobytes() + source.tobytes()
        data += struct.pack("<f", float(s.time))
        data += lit.tobytes() + burned.tobytes()
        writer.write_data(id_thr, data)

        # tvc_setpoint (sim ENU)
        heading = float(s.heading_ref_rad)
        if not np.isfinite(heading):
            heading = float("nan")
        data = struct.pack("<Q", ts)
        data += np.asarray(pref, dtype=np.float32).tobytes()
        data += np.asarray(vref, dtype=np.float32).tobytes()
        data += struct.pack("<fBf", heading, phase, 0.0)
        writer.write_data(id_sp, data)

        # tvc_force_sp
        data = struct.pack(
            "<QffB",
            ts,
            float(s.force_target_axial),
            float(s.force_weight),
            1 if s.axial_mix_authorized else 0,
        )
        writer.write_data(id_fsp, data)

        # vehicle_local_position NED
        p_ned = _enu_to_ned(pos)
        v_ned = _enu_to_ned(vel)
        data = struct.pack("<QQ", ts, ts)
        data += struct.pack(
            "<ffffff",
            float(p_ned[0]),
            float(p_ned[1]),
            float(p_ned[2]),
            float(v_ned[0]),
            float(v_ned[1]),
            float(v_ned[2]),
        )
        data += struct.pack("<BBBB", 1, 1, 1, 1)
        writer.write_data(id_lpos, data)

        # vehicle_attitude NED/FRD
        q_px4 = _q_sim_to_px4(q)
        data = struct.pack("<QQ", ts, ts) + q_px4.astype(np.float32).tobytes()
        writer.write_data(id_att, data)

        # flight_log_vl (ENU-sim / FLU-sim snapshot, same schema as onboard)
        heading = float(s.heading_ref_rad)
        if not np.isfinite(heading):
            heading = float("nan")
        data = struct.pack("<Q", ts)
        data += struct.pack("<f", float(s.time))
        data += np.asarray(q, dtype=np.float32).tobytes()
        data += np.asarray(q_ref, dtype=np.float32).tobytes()
        data += np.asarray(omega, dtype=np.float32).tobytes()
        data += tau_cmd.astype(np.float32).tobytes()
        data += tau_act.astype(np.float32).tobytes()
        data += angles.astype(np.float32).tobytes()
        data += np.asarray(pos, dtype=np.float32).tobytes()
        data += np.asarray(vel, dtype=np.float32).tobytes()
        data += np.asarray(pref, dtype=np.float32).tobytes()
        data += np.asarray(vref, dtype=np.float32).tobytes()
        data += f_body.astype(np.float32).tobytes()
        data += thrusts.astype(np.float32).tobytes()
        data += struct.pack("<ff", float(s.force_target_axial), float(s.force_weight))
        data += att_int.astype(np.float32).tobytes()
        data += struct.pack("<ff", float(s.alt_integral), heading)
        data += struct.pack(
            "<BBBBBB",
            phase,
            0,
            0,
            1 if s.saturated else 0,
            1 if s.axial_mix_authorized else 0,
            1 if s.lift_authorized else 0,
        )
        writer.write_data(id_vl, data)

    writer.finish()

    return {
        "csv": str(csv_path),
        "ulg": str(ulg_path),
        "samples": len(log.samples),
        "duration_s": float(log.samples[-1].time) if log.samples else 0.0,
        "topics": [
            "flight_log_vl",
            "tvc_control_debug",
            "tvc_gimbal_cmd",
            "tvc_mount_thrust",
            "tvc_setpoint",
            "tvc_force_sp",
            "vehicle_local_position",
            "vehicle_attitude",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CSV flight_log_v1 → PX4 ULog")
    ap.add_argument("csv", type=Path, help="Input SIH/Prototype CSV log")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .ulg path (default: same stem as CSV)",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Parse the written ULog with pyulog if available",
    )
    args = ap.parse_args()
    out = args.output or args.csv.with_suffix(".ulg")
    info = convert_csv_to_ulog(args.csv, out)
    print(f"Wrote {info['ulg']} ({info['samples']} samples, {info['duration_s']:.2f} s)")
    print("Topics:", ", ".join(info["topics"]))

    if args.verify:
        try:
            from pyulog import ULog

            u = ULog(str(out))
            names = sorted({d.name for d in u.data_list})
            print("pyulog topics:", ", ".join(names))
            for d in u.data_list:
                n = len(d.data.get("timestamp", []))
                print(f"  {d.name}: {n} rows")
        except Exception as exc:
            print(f"verify failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
