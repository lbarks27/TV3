from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_MOTOR_ROOT = (
    Path(__file__).resolve().parents[1] / "vendor" / "Thrust-Curves-Apogee"
)
INVENTORY_NAME = "Apogee_motor_inventory.csv"


@dataclass(frozen=True)
class MotorCurve:
    """Sampled solid-motor thrust curve from the Apogee vendor data set."""

    manufacturer: str
    designation: str
    times_s: np.ndarray
    thrust_n: np.ndarray
    dynamics_path: Path
    specs_path: Path | None = None
    burn_time_s: float = 0.0
    max_thrust_n: float = 0.0
    average_thrust_n: float = 0.0
    total_impulse_ns: float = 0.0
    initial_mass_kg: float = 0.0
    propellant_mass_kg: float = 0.0
    ignition_delay_s: float = 0.0
    length_m: float = 0.0
    diameter_m: float = 0.0
    # Independent performance scatter. 1.0 is the catalog curve.
    thrust_scale: float = 1.0
    time_scale: float = 1.0

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float).reshape(-1)
        thrust = np.asarray(self.thrust_n, dtype=float).reshape(-1)
        if times.size == 0 or thrust.size == 0:
            raise ValueError(f"Motor {self.key} has an empty thrust curve.")
        if times.size != thrust.size:
            raise ValueError(f"Motor {self.key} time/thrust lengths differ.")
        if np.any(np.diff(times) < 0.0):
            raise ValueError(f"Motor {self.key} times must be non-decreasing.")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "thrust_n", thrust)
        if self.burn_time_s <= 0.0:
            object.__setattr__(self, "burn_time_s", float(times[-1]))
        if self.max_thrust_n <= 0.0:
            object.__setattr__(self, "max_thrust_n", float(np.max(thrust)))
        prefix = np.zeros(times.size, dtype=float)
        if times.size >= 2:
            prefix[1:] = np.cumsum(0.5 * (thrust[1:] + thrust[:-1]) * np.diff(times))
        object.__setattr__(self, "_impulse_prefix_ns", prefix)
        curve_impulse = float(prefix[-1]) if prefix.size else 0.0
        if self.average_thrust_n <= 0.0 and times[-1] > times[0]:
            object.__setattr__(
                self,
                "average_thrust_n",
                curve_impulse / float(times[-1] - times[0]),
            )
        if self.total_impulse_ns <= 0.0:
            object.__setattr__(self, "total_impulse_ns", curve_impulse)

    @property
    def key(self) -> str:
        return f"{self.manufacturer}/{self.designation}"

    def _burn_tau(self, time_s: float) -> float:
        """Catalog-curve time for a simulation clock, after ignition and time scatter."""
        scale = max(float(self.time_scale), 1e-9)
        return (float(time_s) - float(self.ignition_delay_s)) / scale

    def thrust_at(self, time_s: float) -> float:
        """Return thrust (N) at simulation time. Strictly follows the curve samples.

        Before ignition and after the last sample the thrust is zero. Between
        samples the value is linearly interpolated. ``thrust_scale`` multiplies
        the catalog force; ``time_scale`` stretches the burn clock.
        """
        burn_t = self._burn_tau(time_s)
        if burn_t < float(self.times_s[0]) or burn_t > float(self.times_s[-1]):
            return 0.0
        return float(np.interp(burn_t, self.times_s, self.thrust_n)) * float(
            self.thrust_scale
        )

    def impulse_at(self, time_s: float) -> float:
        """Delivered impulse (N·s) from ignition through ``time_s``."""
        times = self.times_s
        if times.size == 0:
            return 0.0
        burn_t = self._burn_tau(time_s)
        prefix = np.asarray(getattr(self, "_impulse_prefix_ns", np.zeros(0)), dtype=float)
        if prefix.size != times.size:
            prefix = np.zeros(times.size, dtype=float)
            if times.size >= 2:
                prefix[1:] = np.cumsum(
                    0.5 * (self.thrust_n[1:] + self.thrust_n[:-1]) * np.diff(times)
                )
        if burn_t <= float(times[0]):
            return 0.0
        scale = float(self.thrust_scale) * max(float(self.time_scale), 1e-9)
        if burn_t >= float(times[-1]):
            return float(prefix[-1]) * scale
        index = int(np.searchsorted(times, burn_t, side="right") - 1)
        index = int(np.clip(index, 0, times.size - 2))
        t0 = float(times[index])
        t1 = float(times[index + 1])
        f0 = float(self.thrust_n[index])
        f1 = float(self.thrust_n[index + 1])
        span = t1 - t0
        frac = 0.0 if span <= 1e-12 else (burn_t - t0) / span
        thrust = f0 + frac * (f1 - f0)
        return float(prefix[index] + 0.5 * (f0 + thrust) * (burn_t - t0)) * scale

    def mass_at(self, time_s: float) -> float:
        """Remaining motor mass (kg). Propellant is spent with delivered impulse.

        Measured thrust already includes exhaust momentum, so only inertial mass
        is reduced. Before ignition the motor is fully loaded; after the curve
        ends only the casing remains. Missing propellant data leaves mass fixed.
        """
        initial = float(self.initial_mass_kg)
        if initial <= 0.0:
            return 0.0
        propellant = min(max(float(self.propellant_mass_kg), 0.0), initial)
        if propellant <= 0.0:
            return initial
        total = float(
            self.impulse_at(
                float(self.ignition_delay_s)
                + float(self.times_s[-1]) * max(float(self.time_scale), 1e-9)
            )
        )
        if total <= 1e-12:
            return initial
        burned = float(np.clip(self.impulse_at(time_s) / total, 0.0, 1.0))
        return initial - propellant * burned

    def _clone(self, **overrides: object) -> MotorCurve:
        payload = {
            "manufacturer": self.manufacturer,
            "designation": self.designation,
            "times_s": self.times_s,
            "thrust_n": self.thrust_n,
            "dynamics_path": self.dynamics_path,
            "specs_path": self.specs_path,
            "burn_time_s": self.burn_time_s,
            "max_thrust_n": self.max_thrust_n,
            "average_thrust_n": self.average_thrust_n,
            "total_impulse_ns": self.total_impulse_ns,
            "initial_mass_kg": self.initial_mass_kg,
            "propellant_mass_kg": self.propellant_mass_kg,
            "ignition_delay_s": self.ignition_delay_s,
            "length_m": self.length_m,
            "diameter_m": self.diameter_m,
            "thrust_scale": self.thrust_scale,
            "time_scale": self.time_scale,
        }
        payload.update(overrides)
        return MotorCurve(**payload)  # type: ignore[arg-type]

    def with_ignition_delay(self, delay_s: float) -> MotorCurve:
        return self._clone(ignition_delay_s=float(delay_s))

    def with_performance(
        self, *, thrust_scale: float = 1.0, time_scale: float = 1.0
    ) -> MotorCurve:
        """Scatter catalog thrust and burn duration. 1.0 is nominal."""
        ts = float(thrust_scale)
        bs = float(time_scale)
        if ts <= 0.0 or bs <= 0.0:
            raise ValueError("Motor performance scales must be positive.")
        catalog_burn = float(self.burn_time_s) / max(float(self.time_scale), 1e-9)
        catalog_peak = float(self.max_thrust_n) / max(float(self.thrust_scale), 1e-9)
        catalog_avg = float(self.average_thrust_n) / max(float(self.thrust_scale), 1e-9)
        catalog_impulse = float(self.total_impulse_ns) / (
            max(float(self.thrust_scale), 1e-9) * max(float(self.time_scale), 1e-9)
        )
        return self._clone(
            thrust_scale=ts,
            time_scale=bs,
            burn_time_s=catalog_burn * bs,
            max_thrust_n=catalog_peak * ts,
            average_thrust_n=catalog_avg * ts,
            total_impulse_ns=catalog_impulse * ts * bs,
        )


def default_motor_root() -> Path:
    return DEFAULT_MOTOR_ROOT


@lru_cache(maxsize=1)
def _load_inventory(root: str) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    inventory_path = Path(root) / INVENTORY_NAME
    by_key: dict[str, dict[str, str]] = {}
    by_designation: dict[str, list[dict[str, str]]] = {}
    if not inventory_path.exists():
        return by_key, by_designation
    with inventory_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            manufacturer = (row.get("manufacturer") or "").strip()
            designation = (row.get("designation") or "").strip()
            if not manufacturer or not designation:
                continue
            key = f"{manufacturer}/{designation}".upper()
            by_key[key] = row
            by_designation.setdefault(designation.upper(), []).append(row)
    return by_key, by_designation


def _parse_engine_spec(
    engine: str | dict | None,
) -> tuple[str | None, str | None, Path | None, float]:
    if engine is None:
        return None, None, None, 0.0
    if isinstance(engine, str):
        text = engine.strip()
        if not text:
            return None, None, None, 0.0
        path = Path(text)
        if path.suffix.lower() == ".csv" or "/" in text and text.lower().endswith(".csv"):
            return None, None, Path(text), 0.0
        if "/" in text:
            manufacturer, designation = text.split("/", 1)
            return manufacturer.strip(), designation.strip(), None, 0.0
        return None, text, None, 0.0
    if not isinstance(engine, dict):
        raise TypeError("engine must be a string, mapping, or None")
    manufacturer = engine.get("manufacturer")
    designation = engine.get("designation") or engine.get("motor") or engine.get("name")
    dynamics = engine.get("dynamics_file") or engine.get("path")
    delay = float(engine.get("ignition_delay_s", 0.0))
    path = Path(dynamics) if dynamics else None
    return (
        str(manufacturer).strip() if manufacturer else None,
        str(designation).strip() if designation else None,
        path,
        delay,
    )


def _read_specs(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return {key: (value or "") for key, value in row.items()}
    return {}


def _load_dynamics_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    thrusts: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dynamics CSV has no header: {path}")
        field_map = {name.lower(): name for name in reader.fieldnames}
        time_key = field_map.get("time_s") or field_map.get("time")
        thrust_key = field_map.get("thrust_n") or field_map.get("thrust")
        if time_key is None or thrust_key is None:
            raise ValueError(
                f"Dynamics CSV must include time_s and thrust_N columns: {path}"
            )
        for row in reader:
            times.append(float(row[time_key]))
            thrusts.append(float(row[thrust_key]))
    return np.asarray(times, dtype=float), np.asarray(thrusts, dtype=float)


def _resolve_dynamics_path(
    *,
    root: Path,
    manufacturer: str | None,
    designation: str | None,
    dynamics_path: Path | None,
) -> tuple[Path, str, str, Path | None]:
    if dynamics_path is not None:
        path = dynamics_path if dynamics_path.is_absolute() else root / dynamics_path
        if not path.exists():
            # Also allow repo-relative paths.
            alt = Path(dynamics_path)
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"Motor dynamics file not found: {dynamics_path}")
        mfg = manufacturer or path.parent.parent.name
        des = designation or path.parent.name
        specs = path.with_name(path.name.replace("_dynamics.csv", "_specs.csv"))
        return path, mfg, des, specs if specs.exists() else None

    if designation is None:
        raise ValueError("engine designation is required")

    by_key, by_designation = _load_inventory(str(root))
    row: dict[str, str] | None = None
    if manufacturer:
        row = by_key.get(f"{manufacturer}/{designation}".upper())
    else:
        matches = by_designation.get(designation.upper(), [])
        if len(matches) == 1:
            row = matches[0]
        elif len(matches) > 1:
            options = ", ".join(
                f"{item['manufacturer']}/{item['designation']}" for item in matches
            )
            raise ValueError(
                f"Ambiguous engine '{designation}'. Specify manufacturer. Options: {options}"
            )

    if row is not None:
        rel = row.get("dynamics_file") or ""
        if not rel:
            raise FileNotFoundError(
                f"Inventory entry for {row.get('manufacturer')}/{row.get('designation')} "
                "has no dynamics_file."
            )
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Motor dynamics file missing: {path}")
        specs_rel = row.get("specs_file") or ""
        specs = root / specs_rel if specs_rel else None
        return (
            path,
            row.get("manufacturer") or manufacturer or "",
            row.get("designation") or designation,
            specs if specs and specs.exists() else None,
        )

    # Fallback: manufacturer/designation folder layout used by the vendor tree.
    if manufacturer:
        folder = root / manufacturer / designation
        candidates = sorted(folder.glob("*_dynamics.csv"))
        if candidates:
            path = candidates[0]
            specs = path.with_name(path.name.replace("_dynamics.csv", "_specs.csv"))
            return path, manufacturer, designation, specs if specs.exists() else None

    raise FileNotFoundError(
        f"Unknown engine '{manufacturer + '/' if manufacturer else ''}{designation}'. "
        f"Check {root / INVENTORY_NAME}."
    )


def load_motor(
    engine: str | dict | None,
    *,
    root: str | Path | None = None,
) -> MotorCurve:
    """Load a motor thrust curve from the Apogee vendor data set.

    ``engine`` may be:
    - ``"AeroTech/G80T"``
    - ``"G80T"`` when the designation is unique
    - a path to a ``*_dynamics.csv`` file
    - a mapping with ``manufacturer`` / ``designation`` / ``dynamics_file`` /
      ``ignition_delay_s``
    """
    motor_root = Path(root) if root is not None else default_motor_root()
    manufacturer, designation, dynamics_path, ignition_delay = _parse_engine_spec(engine)
    path, mfg, des, specs_path = _resolve_dynamics_path(
        root=motor_root,
        manufacturer=manufacturer,
        designation=designation,
        dynamics_path=dynamics_path,
    )
    times, thrusts = _load_dynamics_csv(path)
    specs = _read_specs(specs_path) if specs_path is not None else {}

    def _spec_float(*keys: str) -> float:
        for key in keys:
            raw = specs.get(key, "")
            if raw not in ("", None):
                try:
                    return float(raw)
                except ValueError:
                    continue
        return 0.0

    curve = MotorCurve(
        manufacturer=mfg,
        designation=des,
        times_s=times,
        thrust_n=thrusts,
        dynamics_path=path,
        specs_path=specs_path,
        burn_time_s=_spec_float("burn_time_curve_s", "burn_time_certified_s"),
        max_thrust_n=_spec_float("max_thrust_certified_N"),
        average_thrust_n=_spec_float("average_thrust_certified_N"),
        total_impulse_ns=_spec_float("total_impulse_curve_Ns", "total_impulse_certified_Ns"),
        initial_mass_kg=_spec_float("initial_mass_g") / 1000.0,
        propellant_mass_kg=_spec_float("propellant_mass_g") / 1000.0,
        ignition_delay_s=ignition_delay,
        length_m=_spec_float("length_mm") / 1000.0,
        diameter_m=_spec_float("diameter_mm") / 1000.0,
    )
    return curve


def thrusts_at(motors: list[MotorCurve | None], time_s: float, fallback: np.ndarray) -> np.ndarray:
    """Sample per-mount thrusts. Mounts without a motor keep ``fallback`` values."""
    values = np.asarray(fallback, dtype=float).copy()
    if values.shape != (len(motors),):
        raise ValueError("fallback thrust vector must match motor list length")
    for index, motor in enumerate(motors):
        if motor is not None:
            values[index] = motor.thrust_at(time_s)
    return values


def masses_at(motors: list[MotorCurve | None], time_s: float, fallback: np.ndarray) -> np.ndarray:
    """Sample remaining per-mount motor mass. Mounts without a curve keep ``fallback``."""
    values = np.asarray(fallback, dtype=float).copy()
    if values.shape != (len(motors),):
        raise ValueError("fallback mass vector must match motor list length")
    for index, motor in enumerate(motors):
        if motor is not None and float(motor.initial_mass_kg) > 0.0:
            values[index] = motor.mass_at(time_s)
    return values
