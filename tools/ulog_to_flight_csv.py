#!/usr/bin/env python3
"""Convert a TV3 SIH-exported ULog back toward flight_log_v1 CSV (partial).

Reads tvc_* topics written by ``csv_to_ulog.py`` and emits a CSV that
``FlightLog.load`` / Prototype replay can open.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    from pyulog import ULog
except ImportError as exc:
    raise SystemExit("pyulog required: python3 -m pip install --user pyulog") from exc


def _get_topic(ulog: ULog, name: str):
    for d in ulog.data_list:
        if d.name == name:
            return d
    return None


def _arr(topic, name: str, i: int, n: int) -> list[float]:
    return [float(topic.data[f"{name}[{k}]"][i]) for k in range(n)]


def convert_from_vl(vl, csv_path: Path, ulg_name: str) -> int:
    n = len(vl.data["timestamp"])
    rows = []
    t0 = float(vl.data["timestamp"][0]) * 1e-6 if n else 0.0
    for i in range(n):
        ts = float(vl.data["timestamp"][i]) * 1e-6
        time_s = float(vl.data["time_s"][i]) if "time_s" in vl.data else (ts - t0)
        q = _arr(vl, "q", i, 4)
        q_ref = _arr(vl, "q_ref", i, 4)
        omega = _arr(vl, "omega", i, 3)
        tau_cmd = _arr(vl, "tau_cmd", i, 3)
        tau_act = _arr(vl, "tau_act", i, 3)
        angles = _arr(vl, "gimbal_angles", i, 6)
        pos = _arr(vl, "position", i, 3)
        vel = _arr(vl, "velocity", i, 3)
        pref = _arr(vl, "position_ref", i, 3)
        f_body = _arr(vl, "force_body", i, 3)
        thrusts = _arr(vl, "mount_thrusts", i, 3)
        f_ax = float(vl.data["force_target_axial"][i])
        fw = float(vl.data["force_weight"][i])
        sat = int(vl.data["saturated"][i])
        rows.append(
            {
                "time": time_s,
                "px": pos[0],
                "py": pos[1],
                "pz": pos[2],
                "vx": vel[0],
                "vy": vel[1],
                "vz": vel[2],
                "px_ref": pref[0],
                "py_ref": pref[1],
                "pz_ref": pref[2],
                "qw": q[0],
                "qx": q[1],
                "qy": q[2],
                "qz": q[3],
                "wx": omega[0],
                "wy": omega[1],
                "wz": omega[2],
                "qw_ref": q_ref[0],
                "qx_ref": q_ref[1],
                "qy_ref": q_ref[2],
                "qz_ref": q_ref[3],
                "tx_cmd": tau_cmd[0],
                "ty_cmd": tau_cmd[1],
                "tz_cmd": tau_cmd[2],
                "tx_act": tau_act[0],
                "ty_act": tau_act[1],
                "tz_act": tau_act[2],
                "fx_cmd": 0.0,
                "fy_cmd": 0.0,
                "fz_cmd": f_ax,
                "fx_body": f_body[0],
                "fy_body": f_body[1],
                "fz_body": f_body[2],
                "fx": f_body[0],
                "fy": f_body[1],
                "fz": f_body[2],
                "saturated": sat,
                "m0_a2": angles[0],
                "m0_a1": angles[1],
                "m1_a2": angles[2],
                "m1_a1": angles[3],
                "m2_a2": angles[4],
                "m2_a1": angles[5],
                "m0_thrust": thrusts[0],
                "m1_thrust": thrusts[1],
                "m2_thrust": thrusts[2],
                "force_weight": fw,
            }
        )
    return _write_csv(csv_path, rows, ulg_name, source_topic="flight_log_vl")


def _write_csv(csv_path: Path, rows: list[dict], ulg_name: str, source_topic: str) -> int:
    fields = list(rows[0].keys()) if rows else []
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        f.write("# schema=flight_log_v1\n")
        f.write("# frame=enu_sim_body_flu\n")
        f.write(f"# source_ulog={ulg_name}\n")
        f.write(f"# source_topic={source_topic}\n")
        f.write("# notes=converted from TV3 ULog\n")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def convert(ulg_path: Path, csv_path: Path) -> int:
    ulog = ULog(str(ulg_path))
    vl = _get_topic(ulog, "flight_log_vl")
    if vl is not None:
        return convert_from_vl(vl, csv_path, ulg_path.name)

    dbg = _get_topic(ulog, "tvc_control_debug")
    gim = _get_topic(ulog, "tvc_gimbal_cmd")
    thr = _get_topic(ulog, "tvc_mount_thrust")
    sp = _get_topic(ulog, "tvc_setpoint")
    fsp = _get_topic(ulog, "tvc_force_sp")
    if dbg is None:
        raise SystemExit("no flight_log_vl or tvc_control_debug in ulog")

    n = len(dbg.data["timestamp"])
    rows = []
    for i in range(n):
        t = float(dbg.data["timestamp"][i]) * 1e-6
        q = [float(dbg.data[f"q[{k}]"][i]) for k in range(4)]
        q_ref = [float(dbg.data[f"q_ref[{k}]"][i]) for k in range(4)]
        omega = [float(dbg.data[f"omega[{k}]"][i]) for k in range(3)]
        tau_cmd = [float(dbg.data[f"tau_cmd[{k}]"][i]) for k in range(3)]
        tau_act = [float(dbg.data[f"tau_act[{k}]"][i]) for k in range(3)]
        f_body = [float(dbg.data[f"force_body[{k}]"][i]) for k in range(3)]

        pos = [0.0, 0.0, 0.0]
        vel = [0.0, 0.0, 0.0]
        pref = [0.0, 0.0, 0.0]
        vref = [0.0, 0.0, 0.0]
        if sp is not None and i < len(sp.data["timestamp"]):
            pref = [float(sp.data[f"pos[{k}]"][i]) for k in range(3)]
            vref = [float(sp.data[f"vel[{k}]"][i]) for k in range(3)]

        # Prefer vehicle_local_position if present (NED → ENU)
        lpos = _get_topic(ulog, "vehicle_local_position")
        if lpos is not None and i < len(lpos.data["timestamp"]):
            # NED x,y,z → ENU
            xn, ye, zd = float(lpos.data["x"][i]), float(lpos.data["y"][i]), float(lpos.data["z"][i])
            vxn, vye, vzd = float(lpos.data["vx"][i]), float(lpos.data["vy"][i]), float(lpos.data["vz"][i])
            pos = [ye, xn, -zd]
            vel = [vye, vxn, -vzd]

        angles = [0.0] * 6
        sat = 0
        if gim is not None and i < len(gim.data["timestamp"]):
            angles = [float(gim.data[f"angles_rad[{k}]"][i]) for k in range(6)]
            sat = int(gim.data["saturated"][i])

        thrusts = [0.0, 0.0, 0.0]
        if thr is not None and i < len(thr.data["timestamp"]):
            thrusts = [float(thr.data[f"thrust[{k}]"][i]) for k in range(3)]

        f_ax = 0.0
        fw = 0.0
        if fsp is not None and i < len(fsp.data["timestamp"]):
            f_ax = float(fsp.data["f_axial_n"][i])
            fw = float(fsp.data["force_weight"][i])

        rows.append(
            {
                "time": t,
                "px": pos[0],
                "py": pos[1],
                "pz": pos[2],
                "vx": vel[0],
                "vy": vel[1],
                "vz": vel[2],
                "px_ref": pref[0],
                "py_ref": pref[1],
                "pz_ref": pref[2],
                "qw": q[0],
                "qx": q[1],
                "qy": q[2],
                "qz": q[3],
                "wx": omega[0],
                "wy": omega[1],
                "wz": omega[2],
                "qw_ref": q_ref[0],
                "qx_ref": q_ref[1],
                "qy_ref": q_ref[2],
                "qz_ref": q_ref[3],
                "tx_cmd": tau_cmd[0],
                "ty_cmd": tau_cmd[1],
                "tz_cmd": tau_cmd[2],
                "tx_act": tau_act[0],
                "ty_act": tau_act[1],
                "tz_act": tau_act[2],
                "fx_cmd": 0.0,
                "fy_cmd": 0.0,
                "fz_cmd": f_ax,
                "fx_body": f_body[0],
                "fy_body": f_body[1],
                "fz_body": f_body[2],
                "fx": f_body[0],
                "fy": f_body[1],
                "fz": f_body[2],
                "saturated": sat,
                "m0_a2": angles[0],
                "m0_a1": angles[1],
                "m1_a2": angles[2],
                "m1_a1": angles[3],
                "m2_a2": angles[4],
                "m2_a1": angles[5],
                "m0_thrust": thrusts[0],
                "m1_thrust": thrusts[1],
                "m2_thrust": thrusts[2],
                "force_weight": fw,
            }
        )

    return _write_csv(csv_path, rows, ulg_path.name, source_topic="tvc_control_debug")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or args.ulg.with_name(args.ulg.stem + "_from_ulog.csv")
    n = convert(args.ulg, out)
    print(f"Wrote {out} ({n} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
