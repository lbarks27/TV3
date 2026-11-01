"""ctypes bridge: firmware tvc_core controllers as drop-ins for Prototype SIH.

Builds against ``build/host_tests/libtvc_core_host.*`` (see
``tools/build_host_tests.sh``). Plant + allocator stay on Prototype; attitude,
position, and altitude force loops run the same C++ as Cube ``tvc_pos_alt`` /
``tvc_control``.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

TV3_ROOT = Path(__file__).resolve().parents[1]


def _lib_candidates() -> list[Path]:
	env = os.environ.get("TVC_CORE_HOST_LIB")
	names = (
		"libtvc_core_host.dylib",
		"libtvc_core_host.so",
		"tvc_core_host.dll",
	)
	roots = [
		TV3_ROOT / "build" / "host_tests",
		TV3_ROOT / "build" / "host_tests" / "tvc_core_build",
		TV3_ROOT / "build" / "tvc_core_host",
	]
	out: list[Path] = []
	if env:
		out.append(Path(env))
	for root in roots:
		for name in names:
			out.append(root / name)
	return out


def load_lib() -> ctypes.CDLL:
	last_err: Exception | None = None
	for path in _lib_candidates():
		if not path.is_file():
			continue
		try:
			return ctypes.CDLL(str(path))
		except OSError as exc:
			last_err = exc
	raise FileNotFoundError(
		"libtvc_core_host not found. Run ./tools/build_host_tests.sh "
		f"(looked under build/host_tests). Last error: {last_err}"
	)


_LIB: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
	global _LIB
	if _LIB is None:
		lib = load_lib()
		# Attitude
		lib.tvc_att_create.restype = ctypes.c_void_p
		lib.tvc_att_destroy.argtypes = [ctypes.c_void_p]
		lib.tvc_att_reset.argtypes = [ctypes.c_void_p]
		lib.tvc_att_set_gains.argtypes = [
			ctypes.c_void_p,
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.c_float,
			ctypes.c_float,
		]
		lib.tvc_att_update.argtypes = [
			ctypes.c_void_p,
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.c_float,
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
		]
		# Position
		lib.tvc_pos_create.restype = ctypes.c_void_p
		lib.tvc_pos_destroy.argtypes = [ctypes.c_void_p]
		lib.tvc_pos_reset.argtypes = [ctypes.c_void_p]
		lib.tvc_pos_set_gains.argtypes = [
			ctypes.c_void_p,
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_int,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_int,
			ctypes.c_float,
		]
		lib.tvc_pos_set_body_up.argtypes = [
			ctypes.c_void_p,
			ctypes.POINTER(ctypes.c_float),
		]
		lib.tvc_pos_attitude_reference.argtypes = [
			ctypes.c_void_p,
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
			ctypes.c_float,
			ctypes.c_int,
			ctypes.c_float,
			ctypes.POINTER(ctypes.c_float),
		]
		# Altitude
		lib.tvc_alt_create.restype = ctypes.c_void_p
		lib.tvc_alt_destroy.argtypes = [ctypes.c_void_p]
		lib.tvc_alt_reset.argtypes = [ctypes.c_void_p]
		lib.tvc_alt_set_gains.argtypes = [
			ctypes.c_void_p,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
		]
		lib.tvc_alt_update.argtypes = [
			ctypes.c_void_p,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.c_float,
			ctypes.POINTER(ctypes.c_float),
			ctypes.POINTER(ctypes.c_float),
		]
		lib.tvc_alt_get_integral.restype = ctypes.c_float
		lib.tvc_alt_get_integral.argtypes = [ctypes.c_void_p]
		lib.tvc_alt_set_integral.argtypes = [ctypes.c_void_p, ctypes.c_float]
		_LIB = lib
	return _LIB


def _f3(arr: Any) -> ctypes.Array[ctypes.c_float]:
	a = np.asarray(arr, dtype=np.float32).reshape(3)
	return (ctypes.c_float * 3)(float(a[0]), float(a[1]), float(a[2]))


def _f4(arr: Any) -> ctypes.Array[ctypes.c_float]:
	a = np.asarray(arr, dtype=np.float32).reshape(4)
	return (ctypes.c_float * 4)(
		float(a[0]), float(a[1]), float(a[2]), float(a[3])
	)


def _lim(v: Any, default: float = -1.0) -> float:
	if v is None:
		return default
	return float(v)


class FirmwareAttitudePID:
	"""Drop-in for Prototype ``AttitudePIDController``."""

	def __init__(self, config: Any = None) -> None:
		self._lib = _lib()
		self._h = self._lib.tvc_att_create()
		if not self._h:
			raise RuntimeError("tvc_att_create failed")
		self.config = config
		self.integral = np.zeros(3, dtype=float)
		self._push_gains()

	def __del__(self) -> None:
		h = getattr(self, "_h", None)
		lib = getattr(self, "_lib", None)
		if h and lib:
			lib.tvc_att_destroy(h)

	def _push_gains(self) -> None:
		cfg = self.config
		if cfg is None:
			kp = np.array([12.0, 12.0, 2.5], dtype=float)
			ki = np.zeros(3, dtype=float)
			kd = np.array([6.0, 6.0, 1.4], dtype=float)
			tlim, ilim = 15.0, 2.0
		else:
			kp = np.asarray(cfg.kp, dtype=float).reshape(3)
			ki = np.asarray(cfg.ki, dtype=float).reshape(3)
			kd = np.asarray(cfg.kd, dtype=float).reshape(3)
			tlim = _lim(getattr(cfg, "torque_limit", 15.0), 15.0)
			ilim = _lim(getattr(cfg, "integral_limit", 2.0), 2.0)
		self._lib.tvc_att_set_gains(
			self._h, _f3(kp), _f3(ki), _f3(kd), float(tlim), float(ilim)
		)

	def reset(self) -> None:
		self._lib.tvc_att_reset(self._h)
		self.integral[:] = 0.0

	def update(
		self,
		quaternion: np.ndarray,
		omega: np.ndarray,
		quaternion_reference: np.ndarray,
		dt: float,
		*,
		omega_reference: np.ndarray | None = None,
	) -> np.ndarray:
		self._push_gains()
		tau = (ctypes.c_float * 3)()
		wref = None if omega_reference is None else _f3(omega_reference)
		self._lib.tvc_att_update(
			self._h,
			_f4(quaternion),
			_f3(omega),
			_f4(quaternion_reference),
			float(dt),
			wref,
			tau,
		)
		return np.array([tau[0], tau[1], tau[2]], dtype=float)


class FirmwarePositionPID:
	"""Drop-in for Prototype ``PositionPIDController`` (core cascade path)."""

	def __init__(self, config: Any = None, body_up: Any = None) -> None:
		self._lib = _lib()
		self._h = self._lib.tvc_pos_create()
		if not self._h:
			raise RuntimeError("tvc_pos_create failed")
		self.config = config
		self.integral = np.zeros(3, dtype=float)
		self._body_up = (
			np.array([0.0, 0.0, 1.0], dtype=float)
			if body_up is None
			else np.asarray(body_up, dtype=float).reshape(3)
		)
		self._push_gains()
		self._push_body_up()

	def __del__(self) -> None:
		h = getattr(self, "_h", None)
		lib = getattr(self, "_lib", None)
		if h and lib:
			lib.tvc_pos_destroy(h)

	def set_body_up(self, body_up: Any) -> None:
		self._body_up = np.asarray(body_up, dtype=float).reshape(3)
		self._push_body_up()

	def _push_body_up(self) -> None:
		self._lib.tvc_pos_set_body_up(self._h, _f3(self._body_up))

	def _push_gains(self) -> None:
		cfg = self.config
		if cfg is None:
			kp = np.array([0.55, 0.70, 0.0])
			ki = np.array([0.0, 0.015, 0.0])
			kd = np.array([4.0, 5.0, 0.0])
			max_tilt, min_up = 15.0, 2.0
			ilim, cascade = 0.08, 1
			e_max, v_max, i_start, lat = 3.0, 3.5, 1.5, 0.08
			tilt_y, e_y, v_y = -1.0, 0.0, -1.0
			prefer_y = 0
			lpf_hz = 0.0
		else:
			kp = np.asarray(cfg.kp, dtype=float).reshape(3)
			ki = np.asarray(getattr(cfg, "ki", [0, 0, 0]), dtype=float).reshape(3)
			kd = np.asarray(cfg.kd, dtype=float).reshape(3)
			max_tilt = float(cfg.max_tilt_deg)
			min_up = float(getattr(cfg, "min_upward_accel", 2.0))
			ilim = _lim(getattr(cfg, "integral_limit", 0.08), 0.08)
			cascade = 1 if bool(getattr(cfg, "velocity_cascade", True)) else 0
			e_max = float(getattr(cfg, "max_position_error_m", 3.0) or 0.0)
			raw_v = getattr(cfg, "max_speed_m_s", 3.5)
			if isinstance(raw_v, (tuple, list)):
				v_max = float(raw_v[0])
				v_y = float(raw_v[1]) if len(raw_v) > 1 else -1.0
			else:
				v_max = float(raw_v or 0.0)
				v_y = -1.0
			i_start = float(getattr(cfg, "integral_start_m", 1.5) or 0.0)
			lat = float(getattr(cfg, "lateral_priority_m", 0.08) or 0.0)
			raw_ty = getattr(cfg, "max_tilt_y_deg", None)
			tilt_y = -1.0 if raw_ty is None else float(raw_ty)
			e_y = float(getattr(cfg, "max_position_error_y_m", 0.0) or 0.0)
			prefer_y = 1 if bool(getattr(cfg, "prefer_lateral_y", False)) else 0
			lpf_hz = float(getattr(cfg, "lateral_accel_lpf_hz", 0.0) or 0.0)
		self._lib.tvc_pos_set_gains(
			self._h,
			_f3(kp),
			_f3(ki),
			_f3(kd),
			float(max_tilt),
			float(min_up),
			float(ilim),
			int(cascade),
			float(e_max),
			float(v_max),
			float(i_start),
			float(lat),
			float(tilt_y),
			float(e_y),
			float(v_y),
			int(prefer_y),
			float(lpf_hz),
		)

	def reset(self) -> None:
		self._lib.tvc_pos_reset(self._h)
		self.integral[:] = 0.0

	def attitude_reference(
		self,
		position: np.ndarray,
		velocity: np.ndarray,
		position_reference: np.ndarray,
		velocity_reference: np.ndarray | None = None,
		*,
		accel_reference: np.ndarray | None = None,
		gravity: np.ndarray | None = None,
		heading_rad: float | None = None,
		dt: float = 0.0,
	) -> np.ndarray:
		# Firmware path ignores accel_reference (no IMU/mission accel FF on tilt).
		_ = accel_reference
		self._push_gains()
		vref = (
			np.zeros(3, dtype=float)
			if velocity_reference is None
			else np.asarray(velocity_reference, dtype=float)
		)
		g = (
			np.array([0.0, 0.0, -9.81], dtype=float)
			if gravity is None
			else np.asarray(gravity, dtype=float)
		)
		heading_valid = heading_rad is not None and np.isfinite(float(heading_rad))
		heading = float(heading_rad) if heading_valid else 0.0
		q_out = (ctypes.c_float * 4)()
		self._lib.tvc_pos_attitude_reference(
			self._h,
			_f3(position),
			_f3(velocity),
			_f3(position_reference),
			_f3(vref),
			_f3(g),
			float(heading),
			1 if heading_valid else 0,
			float(dt),
			q_out,
		)
		return np.array([q_out[0], q_out[1], q_out[2], q_out[3]], dtype=float)

	def attitude_reference_velocity_aim(self, *args: Any, **kwargs: Any) -> np.ndarray:
		raise NotImplementedError(
			"velocity-aim is Prototype-only; firmware SIH uses attitude_reference"
		)


class FirmwareAltitudeForce:
	"""Drop-in for Prototype ``AltitudeForceController`` (az cascade)."""

	def __init__(self, config: Any = None) -> None:
		self._lib = _lib()
		self._h = self._lib.tvc_alt_create()
		if not self._h:
			raise RuntimeError("tvc_alt_create failed")
		self.config = config
		self.last_az_cmd = 0.0
		self._push_gains()

	def __del__(self) -> None:
		h = getattr(self, "_h", None)
		lib = getattr(self, "_lib", None)
		if h and lib:
			lib.tvc_alt_destroy(h)

	@property
	def integral(self) -> float:
		return float(self._lib.tvc_alt_get_integral(self._h))

	@integral.setter
	def integral(self, value: float) -> None:
		self._lib.tvc_alt_set_integral(self._h, float(value))

	def _push_gains(self) -> None:
		cfg = self.config
		if cfg is None:
			kp, ki, kd = 12.0, 1.5, 8.0
			ilim, fmin, fmax = 20.0, 0.0, -1.0
			az_min, az_max = -9.81, 3.924
		else:
			kp = float(cfg.kp)
			ki = float(cfg.ki)
			kd = float(cfg.kd)
			ilim = _lim(getattr(cfg, "integral_limit", 20.0), 20.0)
			fmin = float(getattr(cfg, "force_min", 0.0))
			fmax = _lim(getattr(cfg, "force_max", None), -1.0)
			az_min = float(getattr(cfg, "az_min", -9.81))
			az_max = float(getattr(cfg, "az_max", 3.924))
		self._lib.tvc_alt_set_gains(
			self._h, kp, ki, kd, ilim, fmin, fmax, az_min, az_max
		)

	def reset(self) -> None:
		self._lib.tvc_alt_reset(self._h)
		self.last_az_cmd = 0.0

	def update(
		self,
		altitude: float,
		vertical_velocity: float,
		altitude_reference: float,
		dt: float,
		*,
		mass: float,
		gravity: np.ndarray | None = None,
		vertical_velocity_reference: float | None = None,
		vertical_accel_reference: float | None = None,
	) -> float:
		self._push_gains()
		g = (
			np.array([0.0, 0.0, -9.81], dtype=float)
			if gravity is None
			else np.asarray(gravity, dtype=float)
		)
		g_abs = float(np.linalg.norm(g))
		vz_ref = 0.0 if vertical_velocity_reference is None else float(vertical_velocity_reference)
		az_ff = 0.0 if vertical_accel_reference is None else float(vertical_accel_reference)
		force = ctypes.c_float()
		az = ctypes.c_float()
		self._lib.tvc_alt_update(
			self._h,
			float(altitude),
			float(vertical_velocity),
			float(altitude_reference),
			float(dt),
			float(mass),
			float(g_abs),
			float(vz_ref),
			float(az_ff),
			ctypes.byref(force),
			ctypes.byref(az),
		)
		self.last_az_cmd = float(az.value)
		return float(force.value)


def wrap_sim_controllers(sim: Any) -> None:
	"""Replace attitude / position / altitude controllers on a FlightSimulator."""
	att_cfg = getattr(sim.attitude_controller, "config", None)
	pos_cfg = getattr(sim.position_controller, "config", None)
	alt_cfg = getattr(sim.altitude_controller, "config", None)
	com = getattr(sim, "allocator", None)
	body_up = None
	if com is not None and getattr(com, "center_of_mass", None) is not None:
		body_up = np.asarray(com.center_of_mass, dtype=float).reshape(3)
	elif getattr(sim, "system", None) is not None:
		sys = sim.system
		if getattr(sys, "center_of_mass", None) is not None:
			body_up = np.asarray(sys.center_of_mass, dtype=float).reshape(3)
	sim.attitude_controller = FirmwareAttitudePID(att_cfg)
	sim.position_controller = FirmwarePositionPID(pos_cfg, body_up=body_up)
	sim.altitude_controller = FirmwareAltitudeForce(alt_cfg)
	sim.control_stack = "tvc_core"


def use_firmware_controllers() -> bool:
	"""Default on. Set SIH_CONTROL=prototype to keep Python controllers."""
	raw = os.environ.get("SIH_CONTROL", "firmware").strip().lower()
	return raw not in {"prototype", "python", "proto", "0", "off", "false"}


# Ensure TV3 sih/ is importable when run from Prototype cwd.
if str(TV3_ROOT / "sih") not in sys.path:
	sys.path.insert(0, str(TV3_ROOT / "sih"))
