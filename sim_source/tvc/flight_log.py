from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _to_list(value: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    return np.asarray(value, dtype=float).tolist()


def _zeros3() -> np.ndarray:
    return np.zeros(3, dtype=float)


def _nan_heading() -> float:
    return float("nan")


@dataclass
class FlightLogSample:
    time: float
    quaternion: np.ndarray
    omega: np.ndarray
    quaternion_reference: np.ndarray
    torque_command: np.ndarray
    torque_achieved: np.ndarray
    gimbal_angles: np.ndarray
    saturated: bool
    position: np.ndarray = field(default_factory=_zeros3)
    velocity: np.ndarray = field(default_factory=_zeros3)
    position_reference: np.ndarray = field(default_factory=_zeros3)
    force_command: np.ndarray = field(default_factory=_zeros3)
    force_body: np.ndarray = field(default_factory=_zeros3)
    force_inertial: np.ndarray = field(default_factory=_zeros3)
    mount_thrusts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    # --- Module / state-machine telemetry (optional on older logs) ---
    # Mission state machine: "", "approach", "landing", or capture waypoint index context.
    mission_phase: str = ""
    waypoint_index: int = 0
    waypoint_count: int = 0
    waypoint_advance: str = ""
    # Attitude outer-loop mode: level | vel_aim | pos_hold | att_ref | idle
    att_mode: str = ""
    lift_authorized: bool = False
    axial_mix_authorized: bool = False
    force_target_axial: float = 0.0
    force_weight: float = 0.0
    attitude_error: np.ndarray = field(default_factory=_zeros3)
    att_integral: np.ndarray = field(default_factory=_zeros3)
    alt_integral: float = 0.0
    velocity_reference: np.ndarray = field(default_factory=_zeros3)
    heading_ref_rad: float = field(default_factory=_nan_heading)
    # Plant-only motor-offset mass properties (absent on older logs).
    com_plant: np.ndarray = field(default_factory=_zeros3)
    com_alloc: np.ndarray = field(default_factory=_zeros3)
    motor_cgs: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    torque_mass: np.ndarray = field(default_factory=_zeros3)
    force_mass: np.ndarray = field(default_factory=_zeros3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": float(self.time),
            "quaternion": _to_list(self.quaternion),
            "omega": _to_list(self.omega),
            "quaternion_reference": _to_list(self.quaternion_reference),
            "torque_command": _to_list(self.torque_command),
            "torque_achieved": _to_list(self.torque_achieved),
            "gimbal_angles": _to_list(self.gimbal_angles),
            "saturated": bool(self.saturated),
            "position": _to_list(self.position),
            "velocity": _to_list(self.velocity),
            "position_reference": _to_list(self.position_reference),
            "force_command": _to_list(self.force_command),
            "force_body": _to_list(self.force_body),
            "force_inertial": _to_list(self.force_inertial),
            "mount_thrusts": _to_list(self.mount_thrusts),
            "mission_phase": str(self.mission_phase),
            "waypoint_index": int(self.waypoint_index),
            "waypoint_count": int(self.waypoint_count),
            "waypoint_advance": str(self.waypoint_advance),
            "att_mode": str(self.att_mode),
            "lift_authorized": bool(self.lift_authorized),
            "axial_mix_authorized": bool(self.axial_mix_authorized),
            "force_target_axial": float(self.force_target_axial),
            "force_weight": float(self.force_weight),
            "attitude_error": _to_list(self.attitude_error),
            "att_integral": _to_list(self.att_integral),
            "alt_integral": float(self.alt_integral),
            "velocity_reference": _to_list(self.velocity_reference),
            "heading_ref_rad": float(self.heading_ref_rad),
            "com_plant": _to_list(self.com_plant),
            "com_alloc": _to_list(self.com_alloc),
            "motor_cgs": np.asarray(self.motor_cgs, dtype=float).reshape(-1, 3).tolist(),
            "torque_mass": _to_list(self.torque_mass),
            "force_mass": _to_list(self.force_mass),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlightLogSample:
        heading = data.get("heading_ref_rad", float("nan"))
        try:
            heading_val = float(heading)
        except (TypeError, ValueError):
            heading_val = float("nan")
        return cls(
            time=float(data["time"]),
            quaternion=np.asarray(data["quaternion"], dtype=float),
            omega=np.asarray(data["omega"], dtype=float),
            quaternion_reference=np.asarray(data["quaternion_reference"], dtype=float),
            torque_command=np.asarray(data["torque_command"], dtype=float),
            torque_achieved=np.asarray(data["torque_achieved"], dtype=float),
            gimbal_angles=np.asarray(data["gimbal_angles"], dtype=float),
            saturated=bool(data.get("saturated", False)),
            position=np.asarray(data.get("position", _zeros3()), dtype=float),
            velocity=np.asarray(data.get("velocity", _zeros3()), dtype=float),
            position_reference=np.asarray(data.get("position_reference", _zeros3()), dtype=float),
            force_command=np.asarray(data.get("force_command", _zeros3()), dtype=float),
            force_body=np.asarray(data.get("force_body", _zeros3()), dtype=float),
            force_inertial=np.asarray(data.get("force_inertial", _zeros3()), dtype=float),
            mount_thrusts=np.asarray(data.get("mount_thrusts", []), dtype=float),
            mission_phase=str(data.get("mission_phase", "")),
            waypoint_index=int(data.get("waypoint_index", 0)),
            waypoint_count=int(data.get("waypoint_count", 0)),
            waypoint_advance=str(data.get("waypoint_advance", "")),
            att_mode=str(data.get("att_mode", "")),
            lift_authorized=bool(data.get("lift_authorized", False)),
            axial_mix_authorized=bool(data.get("axial_mix_authorized", False)),
            force_target_axial=float(data.get("force_target_axial", 0.0)),
            force_weight=float(data.get("force_weight", 0.0)),
            attitude_error=np.asarray(data.get("attitude_error", _zeros3()), dtype=float),
            att_integral=np.asarray(data.get("att_integral", _zeros3()), dtype=float),
            alt_integral=float(data.get("alt_integral", 0.0)),
            velocity_reference=np.asarray(data.get("velocity_reference", _zeros3()), dtype=float),
            heading_ref_rad=heading_val,
            com_plant=np.asarray(data.get("com_plant", _zeros3()), dtype=float),
            com_alloc=np.asarray(data.get("com_alloc", _zeros3()), dtype=float),
            motor_cgs=np.asarray(data.get("motor_cgs", []), dtype=float).reshape(-1, 3)
            if data.get("motor_cgs") is not None
            else np.zeros((0, 3), dtype=float),
            torque_mass=np.asarray(data.get("torque_mass", _zeros3()), dtype=float),
            force_mass=np.asarray(data.get("force_mass", _zeros3()), dtype=float),
        )


@dataclass
class FlightLogMetadata:
    scenario: str = "hold"
    motion_mode: str = "attitude"
    dt: float = 0.01
    duration: float = 5.0
    force_target: float = 0.0
    force_weight: float = 0.0
    notes: str = ""
    mount_count: int = 0
    engines: str = ""
    # Landing acceptance (solid-motor success gate). Empty strings mean unevaluated.
    landing_success: str = ""  # "true" | "false" | ""
    burnout_time_s: float = -1.0
    burnout_position_m: str = ""  # "x,y,z"
    burnout_speed_m_s: float = -1.0
    landing_box_center_m: str = ""  # "x,y,z"
    landing_box_half_extents_m: str = ""  # "hx,hy,hz"
    landing_box_max_speed_m_s: str = ""
    chopstick_caught: str = ""
    chopstick_heading_ok: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlightLogMetadata:
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


_CSV_BASE_FIELDS = [
    "time",
    "px",
    "py",
    "pz",
    "vx",
    "vy",
    "vz",
    "px_ref",
    "py_ref",
    "pz_ref",
    "qw",
    "qx",
    "qy",
    "qz",
    "wx",
    "wy",
    "wz",
    "qw_ref",
    "qx_ref",
    "qy_ref",
    "qz_ref",
    "tx_cmd",
    "ty_cmd",
    "tz_cmd",
    "tx_act",
    "ty_act",
    "tz_act",
    "fx_cmd",
    "fy_cmd",
    "fz_cmd",
    "fx_body",
    "fy_body",
    "fz_body",
    "fx",
    "fy",
    "fz",
    "saturated",
]

# Per-sample control / state-machine telemetry (appended before gimbal columns).
_CSV_TELEMETRY_FIELDS = [
    "mission_phase",
    "waypoint_index",
    "waypoint_count",
    "waypoint_advance",
    "att_mode",
    "lift_auth",
    "mix_auth",
    "force_target_axial",
    "force_weight",
    "att_err_x",
    "att_err_y",
    "att_err_z",
    "att_int_x",
    "att_int_y",
    "att_int_z",
    "alt_int",
    "vx_ref",
    "vy_ref",
    "vz_ref",
    "heading_ref",
]


def _gimbal_field_names(mount_count: int) -> list[str]:
    names: list[str] = []
    for index in range(mount_count):
        names.append(f"m{index}_a2")
        names.append(f"m{index}_a1")
    return names


def _thrust_field_names(mount_count: int) -> list[str]:
    return [f"m{index}_thrust" for index in range(mount_count)]


def _mass_field_names(mount_count: int) -> list[str]:
    names = [
        "com_plant_x",
        "com_plant_y",
        "com_plant_z",
        "com_alloc_x",
        "com_alloc_y",
        "com_alloc_z",
        "tx_mass",
        "ty_mass",
        "tz_mass",
        "fx_mass",
        "fy_mass",
        "fz_mass",
    ]
    for index in range(mount_count):
        names.extend([f"m{index}_cg_x", f"m{index}_cg_y", f"m{index}_cg_z"])
    return names


def _csv_fieldnames(mount_count: int) -> list[str]:
    return (
        _CSV_BASE_FIELDS
        + _CSV_TELEMETRY_FIELDS
        + _gimbal_field_names(mount_count)
        + _thrust_field_names(mount_count)
        + _mass_field_names(mount_count)
    )


def _parse_metadata_comment(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text.startswith("#"):
        return None
    payload = text[1:].strip()
    if "=" not in payload:
        return None
    key, value = payload.split("=", 1)
    return key.strip(), value.strip()


def _coerce_metadata_value(key: str, value: str) -> Any:
    if key in {
        "dt",
        "duration",
        "force_target",
        "force_weight",
        "burnout_time_s",
        "burnout_speed_m_s",
    }:
        if value in ("", "None", "none"):
            return -1.0
        return float(value)
    if key == "mount_count":
        return int(value)
    return value


def _row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    if key not in row or row[key] in (None, ""):
        return default
    return float(row[key])


@dataclass
class FlightLog:
    metadata: FlightLogMetadata = field(default_factory=FlightLogMetadata)
    samples: list[FlightLogSample] = field(default_factory=list)

    def append(self, sample: FlightLogSample) -> None:
        self.samples.append(sample)
        if self.metadata.mount_count <= 0 and sample.gimbal_angles.size:
            self.metadata.mount_count = int(np.asarray(sample.gimbal_angles).shape[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() == ".json":
            self._save_json(path)
            return
        self._save_csv(path)

    def _save_json(self, path: Path) -> None:
        payload = {
            "version": 2,
            "metadata": self.metadata.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
        }
        path.write_text(json.dumps(payload, indent=2))

    def _save_csv(self, path: Path) -> None:
        mount_count = self.metadata.mount_count
        if mount_count <= 0 and self.samples:
            mount_count = int(np.asarray(self.samples[0].gimbal_angles).shape[0])
            self.metadata.mount_count = mount_count

        fieldnames = _csv_fieldnames(mount_count)
        with path.open("w", newline="") as handle:
            for key, value in self.metadata.to_dict().items():
                handle.write(f"# {key}={value}\n")
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(self._sample_to_csv_row(sample, mount_count))

    @staticmethod
    def _sample_to_csv_row(sample: FlightLogSample, mount_count: int) -> dict[str, Any]:
        q = np.asarray(sample.quaternion, dtype=float)
        omega = np.asarray(sample.omega, dtype=float)
        q_ref = np.asarray(sample.quaternion_reference, dtype=float)
        tau_cmd = np.asarray(sample.torque_command, dtype=float)
        tau_act = np.asarray(sample.torque_achieved, dtype=float)
        position = np.asarray(sample.position, dtype=float)
        velocity = np.asarray(sample.velocity, dtype=float)
        position_ref = np.asarray(sample.position_reference, dtype=float)
        force_cmd = np.asarray(sample.force_command, dtype=float)
        force_body = np.asarray(sample.force_body, dtype=float)
        force_inertial = np.asarray(sample.force_inertial, dtype=float)
        angles = np.asarray(sample.gimbal_angles, dtype=float).reshape(-1, 2)
        row: dict[str, Any] = {
            "time": float(sample.time),
            "px": float(position[0]),
            "py": float(position[1]),
            "pz": float(position[2]),
            "vx": float(velocity[0]),
            "vy": float(velocity[1]),
            "vz": float(velocity[2]),
            "px_ref": float(position_ref[0]),
            "py_ref": float(position_ref[1]),
            "pz_ref": float(position_ref[2]),
            "qw": float(q[0]),
            "qx": float(q[1]),
            "qy": float(q[2]),
            "qz": float(q[3]),
            "wx": float(omega[0]),
            "wy": float(omega[1]),
            "wz": float(omega[2]),
            "qw_ref": float(q_ref[0]),
            "qx_ref": float(q_ref[1]),
            "qy_ref": float(q_ref[2]),
            "qz_ref": float(q_ref[3]),
            "tx_cmd": float(tau_cmd[0]),
            "ty_cmd": float(tau_cmd[1]),
            "tz_cmd": float(tau_cmd[2]),
            "tx_act": float(tau_act[0]),
            "ty_act": float(tau_act[1]),
            "tz_act": float(tau_act[2]),
            "fx_cmd": float(force_cmd[0]),
            "fy_cmd": float(force_cmd[1]),
            "fz_cmd": float(force_cmd[2]),
            "fx_body": float(force_body[0]),
            "fy_body": float(force_body[1]),
            "fz_body": float(force_body[2]),
            "fx": float(force_inertial[0]),
            "fy": float(force_inertial[1]),
            "fz": float(force_inertial[2]),
            "saturated": int(bool(sample.saturated)),
        }
        att_err = np.asarray(sample.attitude_error, dtype=float).reshape(3)
        att_int = np.asarray(sample.att_integral, dtype=float).reshape(3)
        vel_ref = np.asarray(sample.velocity_reference, dtype=float).reshape(3)
        heading = float(sample.heading_ref_rad)
        row.update(
            {
                "mission_phase": str(sample.mission_phase),
                "waypoint_index": int(sample.waypoint_index),
                "waypoint_count": int(sample.waypoint_count),
                "waypoint_advance": str(sample.waypoint_advance),
                "att_mode": str(sample.att_mode),
                "lift_auth": int(bool(sample.lift_authorized)),
                "mix_auth": int(bool(sample.axial_mix_authorized)),
                "force_target_axial": float(sample.force_target_axial),
                "force_weight": float(sample.force_weight),
                "att_err_x": float(att_err[0]),
                "att_err_y": float(att_err[1]),
                "att_err_z": float(att_err[2]),
                "att_int_x": float(att_int[0]),
                "att_int_y": float(att_int[1]),
                "att_int_z": float(att_int[2]),
                "alt_int": float(sample.alt_integral),
                "vx_ref": float(vel_ref[0]),
                "vy_ref": float(vel_ref[1]),
                "vz_ref": float(vel_ref[2]),
                "heading_ref": "" if not np.isfinite(heading) else float(heading),
            }
        )
        for index in range(mount_count):
            if index < angles.shape[0]:
                row[f"m{index}_a2"] = float(angles[index, 0])
                row[f"m{index}_a1"] = float(angles[index, 1])
            else:
                row[f"m{index}_a2"] = 0.0
                row[f"m{index}_a1"] = 0.0
        thrusts = np.asarray(sample.mount_thrusts, dtype=float).reshape(-1)
        for index in range(mount_count):
            row[f"m{index}_thrust"] = float(thrusts[index]) if index < thrusts.size else 0.0
        com_plant = np.asarray(sample.com_plant, dtype=float).reshape(3)
        com_alloc = np.asarray(sample.com_alloc, dtype=float).reshape(3)
        tau_mass = np.asarray(sample.torque_mass, dtype=float).reshape(3)
        f_mass = np.asarray(sample.force_mass, dtype=float).reshape(3)
        row.update(
            {
                "com_plant_x": float(com_plant[0]),
                "com_plant_y": float(com_plant[1]),
                "com_plant_z": float(com_plant[2]),
                "com_alloc_x": float(com_alloc[0]),
                "com_alloc_y": float(com_alloc[1]),
                "com_alloc_z": float(com_alloc[2]),
                "tx_mass": float(tau_mass[0]),
                "ty_mass": float(tau_mass[1]),
                "tz_mass": float(tau_mass[2]),
                "fx_mass": float(f_mass[0]),
                "fy_mass": float(f_mass[1]),
                "fz_mass": float(f_mass[2]),
            }
        )
        cgs = np.asarray(sample.motor_cgs, dtype=float).reshape(-1, 3)
        for index in range(mount_count):
            if index < cgs.shape[0]:
                row[f"m{index}_cg_x"] = float(cgs[index, 0])
                row[f"m{index}_cg_y"] = float(cgs[index, 1])
                row[f"m{index}_cg_z"] = float(cgs[index, 2])
            else:
                row[f"m{index}_cg_x"] = 0.0
                row[f"m{index}_cg_y"] = 0.0
                row[f"m{index}_cg_z"] = 0.0
        return row

    @classmethod
    def load(cls, path: str | Path) -> FlightLog:
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls._load_json(path)
        return cls._load_csv(path)

    @classmethod
    def _load_json(cls, path: Path) -> FlightLog:
        payload = json.loads(path.read_text())
        metadata = FlightLogMetadata.from_dict(payload.get("metadata", {}))
        samples = [FlightLogSample.from_dict(item) for item in payload.get("samples", [])]
        if metadata.mount_count <= 0 and samples:
            metadata.mount_count = int(np.asarray(samples[0].gimbal_angles).shape[0])
        return cls(metadata=metadata, samples=samples)

    @classmethod
    def _load_csv(cls, path: Path) -> FlightLog:
        metadata_data: dict[str, Any] = {}
        data_lines: list[str] = []
        with path.open("r", newline="") as handle:
            for line in handle:
                parsed = _parse_metadata_comment(line)
                if parsed is not None:
                    key, value = parsed
                    metadata_data[key] = _coerce_metadata_value(key, value)
                    continue
                data_lines.append(line)

        reader = csv.DictReader(data_lines)
        if reader.fieldnames is None:
            return cls(metadata=FlightLogMetadata.from_dict(metadata_data), samples=[])

        mount_count = int(metadata_data.get("mount_count", 0))
        if mount_count <= 0:
            angle_fields = [
                name for name in reader.fieldnames if name.startswith("m") and name.endswith("_a2")
            ]
            mount_count = len(angle_fields)
            metadata_data["mount_count"] = mount_count

        has_motion_fields = "px" in reader.fieldnames
        has_force_cmd = "fx_cmd" in reader.fieldnames
        has_thrust_fields = any(
            name.startswith("m") and name.endswith("_thrust") for name in reader.fieldnames
        )
        has_telemetry = "att_mode" in reader.fieldnames or "mission_phase" in reader.fieldnames
        samples: list[FlightLogSample] = []
        for row in reader:
            angles = np.zeros((mount_count, 2), dtype=float)
            thrusts = np.zeros(mount_count, dtype=float)
            for index in range(mount_count):
                angles[index, 0] = _row_float(row, f"m{index}_a2")
                angles[index, 1] = _row_float(row, f"m{index}_a1")
                if has_thrust_fields:
                    thrusts[index] = _row_float(row, f"m{index}_thrust")
            heading_raw = row.get("heading_ref", "") if has_telemetry else ""
            if heading_raw in (None, ""):
                heading_val = float("nan")
            else:
                heading_val = float(heading_raw)
            samples.append(
                FlightLogSample(
                    time=float(row["time"]),
                    position=np.array(
                        [
                            _row_float(row, "px"),
                            _row_float(row, "py"),
                            _row_float(row, "pz"),
                        ],
                        dtype=float,
                    ),
                    velocity=np.array(
                        [
                            _row_float(row, "vx"),
                            _row_float(row, "vy"),
                            _row_float(row, "vz"),
                        ],
                        dtype=float,
                    ),
                    position_reference=np.array(
                        [
                            _row_float(row, "px_ref"),
                            _row_float(row, "py_ref"),
                            _row_float(row, "pz_ref"),
                        ],
                        dtype=float,
                    ),
                    quaternion=np.array(
                        [row["qw"], row["qx"], row["qy"], row["qz"]], dtype=float
                    ),
                    omega=np.array([row["wx"], row["wy"], row["wz"]], dtype=float),
                    quaternion_reference=np.array(
                        [row["qw_ref"], row["qx_ref"], row["qy_ref"], row["qz_ref"]],
                        dtype=float,
                    ),
                    torque_command=np.array(
                        [row["tx_cmd"], row["ty_cmd"], row["tz_cmd"]], dtype=float
                    ),
                    torque_achieved=np.array(
                        [row["tx_act"], row["ty_act"], row["tz_act"]], dtype=float
                    ),
                    force_command=np.array(
                        [
                            _row_float(row, "fx_cmd"),
                            _row_float(row, "fy_cmd"),
                            _row_float(row, "fz_cmd"),
                        ],
                        dtype=float,
                    )
                    if has_force_cmd
                    else _zeros3(),
                    force_body=np.array(
                        [
                            _row_float(row, "fx_body"),
                            _row_float(row, "fy_body"),
                            _row_float(row, "fz_body"),
                        ],
                        dtype=float,
                    )
                    if has_motion_fields
                    else _zeros3(),
                    force_inertial=np.array(
                        [
                            _row_float(row, "fx"),
                            _row_float(row, "fy"),
                            _row_float(row, "fz"),
                        ],
                        dtype=float,
                    )
                    if has_motion_fields
                    else _zeros3(),
                    gimbal_angles=angles,
                    mount_thrusts=thrusts,
                    saturated=bool(int(float(row.get("saturated", 0)))),
                    mission_phase=str(row.get("mission_phase", "") or "") if has_telemetry else "",
                    waypoint_index=int(_row_float(row, "waypoint_index", 0.0)) if has_telemetry else 0,
                    waypoint_count=int(_row_float(row, "waypoint_count", 0.0)) if has_telemetry else 0,
                    waypoint_advance=str(row.get("waypoint_advance", "") or "") if has_telemetry else "",
                    att_mode=str(row.get("att_mode", "") or "") if has_telemetry else "",
                    lift_authorized=bool(int(_row_float(row, "lift_auth", 0.0))) if has_telemetry else False,
                    axial_mix_authorized=bool(int(_row_float(row, "mix_auth", 0.0))) if has_telemetry else False,
                    force_target_axial=_row_float(row, "force_target_axial") if has_telemetry else 0.0,
                    force_weight=_row_float(row, "force_weight") if has_telemetry else 0.0,
                    attitude_error=np.array(
                        [
                            _row_float(row, "att_err_x"),
                            _row_float(row, "att_err_y"),
                            _row_float(row, "att_err_z"),
                        ],
                        dtype=float,
                    )
                    if has_telemetry
                    else _zeros3(),
                    att_integral=np.array(
                        [
                            _row_float(row, "att_int_x"),
                            _row_float(row, "att_int_y"),
                            _row_float(row, "att_int_z"),
                        ],
                        dtype=float,
                    )
                    if has_telemetry
                    else _zeros3(),
                    alt_integral=_row_float(row, "alt_int") if has_telemetry else 0.0,
                    velocity_reference=np.array(
                        [
                            _row_float(row, "vx_ref"),
                            _row_float(row, "vy_ref"),
                            _row_float(row, "vz_ref"),
                        ],
                        dtype=float,
                    )
                    if has_telemetry
                    else _zeros3(),
                    heading_ref_rad=heading_val,
                    com_plant=np.array(
                        [
                            _row_float(row, "com_plant_x"),
                            _row_float(row, "com_plant_y"),
                            _row_float(row, "com_plant_z"),
                        ],
                        dtype=float,
                    ),
                    com_alloc=np.array(
                        [
                            _row_float(row, "com_alloc_x"),
                            _row_float(row, "com_alloc_y"),
                            _row_float(row, "com_alloc_z"),
                        ],
                        dtype=float,
                    ),
                    motor_cgs=np.array(
                        [
                            [
                                _row_float(row, f"m{index}_cg_x"),
                                _row_float(row, f"m{index}_cg_y"),
                                _row_float(row, f"m{index}_cg_z"),
                            ]
                            for index in range(mount_count)
                        ],
                        dtype=float,
                    )
                    if f"m0_cg_x" in row
                    else np.zeros((0, 3), dtype=float),
                    torque_mass=np.array(
                        [
                            _row_float(row, "tx_mass"),
                            _row_float(row, "ty_mass"),
                            _row_float(row, "tz_mass"),
                        ],
                        dtype=float,
                    ),
                    force_mass=np.array(
                        [
                            _row_float(row, "fx_mass"),
                            _row_float(row, "fy_mass"),
                            _row_float(row, "fz_mass"),
                        ],
                        dtype=float,
                    ),
                )
            )
        metadata = FlightLogMetadata.from_dict(metadata_data)
        metadata.mount_count = mount_count
        return cls(metadata=metadata, samples=samples)

    def times(self) -> np.ndarray:
        return np.asarray([sample.time for sample in self.samples], dtype=float)

    def field_array(self, name: str) -> np.ndarray:
        return np.vstack([getattr(sample, name) for sample in self.samples])
