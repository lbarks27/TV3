#!/usr/bin/env python3
"""3D matplotlib preview of a TVC ring-3 vehicle (FLU-sim).

Reads vehicle JSON and draws the gimbal ring, motors, mast, catch arms,
landing legs, and body axes. Mounts are labeled m0, m1, m2.

Examples::

    python3 tools/preview_vehicle.py
    python3 tools/preview_vehicle.py --vehicle config/vehicles/tvc_ring3_g12.json
    python3 tools/preview_vehicle.py --save build/artifacts/vehicle_preview.png
    python3 tools/preview_vehicle.py --a1-deg 0,30,0 --a2-deg 15,0,-10
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

TV3 = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE = TV3 / "config/vehicles/tvc_ring3.json"

# Dark canvas so the three mounts stay distinct against structure.
BG = "#0b0d10"
TEXT = "#e8eaed"
MUTED = "#9aa0a6"
PANE = (0.04, 0.05, 0.06, 1.0)
PANE_EDGE = (0.18, 0.20, 0.22, 0.55)
GRID = (0.22, 0.24, 0.26, 0.35)

MOUNT_COLORS = ("#ff6b6b", "#69db7c", "#74c0fc")
AXIS_COLORS = {"x": "#ff6b6b", "y": "#51cf66", "z": "#4dabf7"}
DEFAULT_MOTOR_LENGTH_M = 0.156
DEFAULT_MOTOR_DIAMETER_M = 0.029
DEFAULT_STANDOFF_M = 0.055


@dataclass
class Mount:
    name: str
    position: np.ndarray
    base_dir: np.ndarray
    axis1: np.ndarray
    axis2: np.ndarray
    rest_a2: float
    rest_a1: float
    length_m: float
    diameter_m: float


@dataclass
class CatchArm:
    position: np.ndarray
    direction: np.ndarray
    length_m: float

    def tip(self) -> np.ndarray:
        return self.position + unit(self.direction) * self.length_m


@dataclass
class Vehicle:
    name: str
    mounts: list[Mount]
    catch_arms: list[CatchArm]
    landing_legs: list[CatchArm]
    com: np.ndarray
    standoff_m: float
    mass_g: float
    notes: str
    mast_top_m: float | None


def unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=float)
    return v / n


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit(axis)
    if float(np.linalg.norm(axis)) < 1e-12:
        return np.eye(3)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def thrust_direction(a2: float, a1: float, mount: Mount) -> np.ndarray:
    """Match tvc_core / Prototype: a1 = 0 is +Z rest; nested a2 then a1."""
    a1_kin = 0.5 * math.pi - float(a1)
    rot_primary = axis_angle_rotation(mount.axis2, a2)
    nested_axis1 = unit(rot_primary @ mount.axis1)
    rot_nested = axis_angle_rotation(nested_axis1, a1_kin)
    return unit(rot_nested @ rot_primary @ mount.base_dir)


def _vec3(value, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError("missing vec3")
        return np.asarray(default, dtype=float)
    return np.asarray(value, dtype=float).reshape(3)


def _deg_pair(block: dict | None, key: str, fallback: tuple[float, float]) -> tuple[float, float]:
    src = (block or {}).get(key) or {}
    a1 = float(src.get("a1", fallback[1]))
    a2 = float(src.get("a2", fallback[0]))
    return math.radians(a2), math.radians(a1)


def load_vehicle(path: Path) -> Vehicle:
    data = json.loads(path.read_text())
    rest_a2, rest_a1 = _deg_pair(data, "rest_angles_deg", (0.0, 0.0))
    standoff = float((data.get("motor_cg") or {}).get("pivot_standoff_m", DEFAULT_STANDOFF_M))
    mounts: list[Mount] = []
    for idx, raw in enumerate(data.get("mounts") or []):
        m_rest_a2, m_rest_a1 = _deg_pair(raw, "rest_angles_deg", (math.degrees(rest_a2), math.degrees(rest_a1)))
        length_mm = raw.get("length_mm")
        length_m = float(length_mm) / 1000.0 if length_mm is not None else DEFAULT_MOTOR_LENGTH_M
        mounts.append(
            Mount(
                name=str(raw.get("name") or f"M{idx}"),
                position=_vec3(raw["position"]),
                base_dir=unit(_vec3(raw["base_dir"])),
                axis1=unit(_vec3(raw["axis1"])),
                axis2=unit(_vec3(raw["axis2"])),
                rest_a2=m_rest_a2,
                rest_a1=m_rest_a1,
                length_m=length_m,
                diameter_m=DEFAULT_MOTOR_DIAMETER_M,
            )
        )
    def _members(key: str, default_length: float) -> list[CatchArm]:
        return [
            CatchArm(
                position=_vec3(raw["position"]),
                direction=unit(_vec3(raw["direction"])),
                length_m=float(raw.get("length_m", default_length)),
            )
            for raw in (data.get(key) or [])
        ]

    arms = _members("catch_arms", 0.5)
    legs = _members("landing_legs", 1.0)
    raw_top = data.get("mast_top_m")
    mast_top = float(raw_top) if raw_top is not None else None
    veh = data.get("vehicle") or {}
    return Vehicle(
        name=str(data.get("name") or path.stem),
        mounts=mounts,
        catch_arms=arms,
        landing_legs=legs,
        com=_vec3(data.get("center_of_mass"), np.array([0.0, 0.0, 0.5329])),
        standoff_m=standoff,
        mass_g=float(veh.get("mass_g", 0.0)),
        notes=str(data.get("notes") or ""),
        mast_top_m=mast_top,
    )


def parse_angle_list(text: str | None, count: int, default_deg: float) -> np.ndarray:
    if not text:
        return np.full(count, math.radians(default_deg), dtype=float)
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if len(parts) == 1:
        return np.full(count, math.radians(float(parts[0])), dtype=float)
    if len(parts) != count:
        raise SystemExit(f"expected 1 or {count} comma-separated angles, got {len(parts)}")
    return np.array([math.radians(float(p)) for p in parts], dtype=float)


def _basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n1 = unit(np.cross(axis, a))
    n2 = unit(np.cross(axis, n1))
    return n1, n2


def add_cylinder(ax, p0, p1, radius: float, color: str, *, n: int = 22, alpha: float = 0.95) -> None:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9 or radius <= 0.0:
        return
    axis = axis / length
    n1, n2 = _basis_from_axis(axis)
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    ring = np.outer(np.cos(theta), n1) + np.outer(np.sin(theta), n2)
    start = p0 + radius * ring
    end = p1 + radius * ring
    x = np.vstack([start[:, 0], end[:, 0]])
    y = np.vstack([start[:, 1], end[:, 1]])
    z = np.vstack([start[:, 2], end[:, 2]])
    ax.plot_surface(x, y, z, color=color, linewidth=0, antialiased=True, shade=True, alpha=alpha, rasterized=True)
    for cap, shade in ((start, 0.72), (end, 0.55)):
        poly = Poly3DCollection([cap], color=color, alpha=alpha * shade, linewidths=0)
        ax.add_collection3d(poly)


def add_cone(ax, base, tip, radius: float, color: str, *, n: int = 18, alpha: float = 0.95) -> None:
    base = np.asarray(base, dtype=float)
    tip = np.asarray(tip, dtype=float)
    axis = tip - base
    if float(np.linalg.norm(axis)) < 1e-9 or radius <= 0.0:
        return
    n1, n2 = _basis_from_axis(unit(axis))
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    ring = base + radius * (np.outer(np.cos(theta), n1) + np.outer(np.sin(theta), n2))
    faces = [np.vstack([ring[i], ring[(i + 1) % (n - 1)], tip]) for i in range(n - 1)]
    coll = Poly3DCollection(faces, color=color, alpha=alpha, linewidths=0)
    ax.add_collection3d(coll)


def add_ring(ax, radius: float, z: float, color: str, *, tube: float = 0.004, n: int = 48) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)])
    for i in range(n):
        add_cylinder(ax, pts[i], pts[(i + 1) % n], tube, color, n=8, alpha=0.9)


def style_axes(ax, title: str) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_xlabel("X forward (m)", color=MUTED, labelpad=6)
    ax.set_ylabel("Y left (m)", color=MUTED, labelpad=6)
    ax.set_zlabel("Z up (m)", color=MUTED, labelpad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE)
        axis.pane.set_edgecolor(PANE_EDGE)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.55
    ax.set_title(title, color=TEXT, fontsize=11, pad=10)


def set_equal_aspect(ax, points: np.ndarray, pad: float = 0.04) -> None:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    span = 0.5 * float((hi - lo).max()) + pad
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def draw_vehicle(
    ax,
    vehicle: Vehicle,
    a2: np.ndarray,
    a1: np.ndarray,
    *,
    closeup: bool,
    elev: float,
    azim: float,
) -> list[np.ndarray]:
    mounts = vehicle.mounts
    interest: list[np.ndarray] = [np.zeros(3)]
    if not closeup:
        interest.append(vehicle.com)

    # Mast, catch arms, and landing legs only in the full airframe view.
    if not closeup:
        mast_z = vehicle.mast_top_m
        if mast_z is None and vehicle.catch_arms:
            mast_z = float(vehicle.catch_arms[0].position[2])
        if mast_z is not None:
            mast_top = np.array([0.0, 0.0, float(mast_z)])
            add_cylinder(ax, np.array([0.0, 0.0, min(0.0, mast_top[2])]), mast_top, 0.012, "#6c757d", n=14)
            interest.append(mast_top)
        for arm in vehicle.catch_arms:
            tip = arm.tip()
            add_cylinder(ax, arm.position, tip, 0.008, "#e9ecef", n=10)
            interest.append(arm.position)
            interest.append(tip)
        for leg in vehicle.landing_legs:
            tip = leg.tip()
            add_cylinder(ax, leg.position, tip, 0.010, "#f8f9fa", n=10)
            ax.scatter([tip[0]], [tip[1]], [tip[2]], s=22, c="#f8f9fa", depthshade=False)
            interest.append(leg.position)
            interest.append(tip)

    radii = [float(np.linalg.norm(m.position[:2])) for m in mounts]
    ring_r = float(np.mean(radii)) if radii else 0.098
    add_ring(ax, ring_r, 0.0, "#adb5bd", tube=0.005)

    # Body axes at the origin (FLU-sim).
    axis_len = 0.09 if closeup else 0.22
    origin = np.zeros(3)
    for key, vec in (("x", np.array([axis_len, 0.0, 0.0])), ("y", np.array([0.0, axis_len, 0.0])), ("z", np.array([0.0, 0.0, axis_len]))):
        tip = origin + vec
        ax.plot([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]], color=AXIS_COLORS[key], lw=1.8)
    if not closeup:
        ax.text(axis_len, 0.0, 0.01, "+X", color=AXIS_COLORS["x"], fontsize=8)
        ax.text(0.0, axis_len, 0.01, "+Y", color=AXIS_COLORS["y"], fontsize=8)
        ax.text(0.01, 0.0, axis_len, "+Z", color=AXIS_COLORS["z"], fontsize=8)

    ax.scatter([vehicle.com[0]], [vehicle.com[1]], [vehicle.com[2]], s=70, c="#ffd43b", marker="+", linewidths=1.8)
    if not closeup:
        ax.text(vehicle.com[0] + 0.03, vehicle.com[1] + 0.03, vehicle.com[2], "CoM", color="#ffd43b", fontsize=8)

    for idx, mount in enumerate(mounts):
        color = MOUNT_COLORS[idx % len(MOUNT_COLORS)]
        direction = thrust_direction(float(a2[idx]), float(a1[idx]), mount)
        pivot = mount.position
        face = pivot - direction * vehicle.standoff_m
        nozzle = face - direction * mount.length_m
        radius = 0.5 * mount.diameter_m

        # Spoke from ring center to gimbal pivot.
        ax.plot([0.0, pivot[0]], [0.0, pivot[1]], [0.0, pivot[2]], color="#495057", lw=1.2, alpha=0.9)

        visual_r = radius * (1.15 if closeup else 1.35)
        add_cylinder(ax, pivot, face, visual_r * 0.45, "#ced4da", n=12)
        add_cylinder(ax, face, nozzle, visual_r, color, n=18)
        add_cone(ax, nozzle, nozzle - direction * (visual_r * 1.6), visual_r * 0.72, color)

        # Force on the vehicle is +thrust_dir; exhaust is the opposite way.
        plume = nozzle - direction * (0.05 if closeup else 0.08)
        ax.plot([nozzle[0], plume[0]], [nozzle[1], plume[1]], [nozzle[2], plume[2]], color="#fcc419", lw=2.0, alpha=0.9)

        ax.scatter([pivot[0]], [pivot[1]], [pivot[2]], s=48 if closeup else 36, c=color, depthshade=False, edgecolors="white", linewidths=0.4)

        # Labels sit on the ring plane, radially outside each gimbal, so no tube hides them.
        radial = unit(np.array([pivot[0], pivot[1], 0.0]))
        if float(np.linalg.norm(radial)) < 1e-9:
            radial = np.array([1.0, 0.0, 0.0])
        label_off = 0.055 if closeup else 0.08
        label_pos = pivot + radial * label_off + np.array([0.0, 0.0, 0.018])
        tag = f"m{idx}"
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            tag,
            color=color,
            fontsize=15 if closeup else 10,
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.plot(
            [pivot[0], label_pos[0]],
            [pivot[1], label_pos[1]],
            [pivot[2], label_pos[2]],
            color=color,
            lw=0.8,
            alpha=0.7,
        )
        interest.extend([pivot, face, nozzle, plume, label_pos])

    style_axes(ax, "Gimbal ring" if closeup else "Full airframe")
    ax.view_init(elev=elev, azim=azim)
    return interest


def build_figure(
    vehicle: Vehicle,
    a2: np.ndarray,
    a1: np.ndarray,
    *,
    elev: float,
    azim: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(12.4, 6.6), facecolor=BG)
    ax_full = fig.add_subplot(1, 2, 1, projection="3d")
    ax_ring = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.04, top=0.82, wspace=0.06)

    full_pts = draw_vehicle(ax_full, vehicle, a2, a1, closeup=False, elev=elev, azim=azim)
    ring_pts = draw_vehicle(ax_ring, vehicle, a2, a1, closeup=True, elev=28.0, azim=azim)
    set_equal_aspect(ax_full, np.vstack(full_pts), pad=0.08)
    set_equal_aspect(ax_ring, np.vstack(ring_pts), pad=0.02)

    lines = [
        f"{vehicle.name}   unloaded {vehicle.mass_g:.0f} g   FLU-sim (+X = m0 azimuth)",
        "m0 +X  ·  m1 120°  ·  m2 240°   |   tubes hang opposite thrust   |   gold = exhaust",
    ]
    if vehicle.landing_legs:
        length = float(vehicle.landing_legs[0].length_m)
        root_z = float(vehicle.landing_legs[0].position[2])
        lines.append(
            f"{len(vehicle.landing_legs)} landing legs, {length:.2f} m, roots at z={root_z:.3f} m"
        )
    fig.text(0.03, 0.97, lines[0], color=TEXT, fontsize=12, fontweight="bold", va="top")
    fig.text(0.03, 0.925, lines[1], color=MUTED, fontsize=9, va="top")
    if len(lines) > 2:
        fig.text(0.03, 0.888, lines[2], color=MUTED, fontsize=9, va="top")
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description="3D matplotlib preview of a TVC ring-3 vehicle")
    ap.add_argument("--vehicle", type=Path, default=DEFAULT_VEHICLE, help="Vehicle JSON")
    ap.add_argument("--save", type=Path, default=None, help="Write a PNG instead of (or as well as) showing")
    ap.add_argument("--show", action="store_true", help="Open an interactive window")
    ap.add_argument("--a1-deg", default=None, help="Collective a1 for all mounts, or m0,m1,m2")
    ap.add_argument("--a2-deg", default=None, help="Differential a2 for all mounts, or m0,m1,m2")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    args = ap.parse_args()

    path = args.vehicle.expanduser()
    if not path.is_file():
        raise SystemExit(f"vehicle JSON not found: {path}")
    vehicle = load_vehicle(path)
    if len(vehicle.mounts) < 3:
        raise SystemExit(f"{path} has {len(vehicle.mounts)} mounts; need m0, m1, m2")

    n = len(vehicle.mounts)
    a1 = parse_angle_list(args.a1_deg, n, 0.0)
    a2 = parse_angle_list(args.a2_deg, n, 0.0)
    for i, mount in enumerate(vehicle.mounts):
        if args.a1_deg is None:
            a1[i] = mount.rest_a1
        if args.a2_deg is None:
            a2[i] = mount.rest_a2

    save_path = args.save
    show = args.show or save_path is None
    if save_path is not None:
        plt.switch_backend("Agg")

    fig = build_figure(vehicle, a2, a1, elev=args.elev, azim=args.azim)

    if save_path is not None:
        save_path = save_path.expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, facecolor=fig.get_facecolor())
        print(f"wrote {save_path}")

    if show and save_path is None:
        plt.show()
    elif show:
        # With both --save and --show, rebuild the figure after saving the image.
        plt.close(fig)
        plt.switch_backend(plt.rcParams.get("backend", "macosx"))
        fig = build_figure(vehicle, a2, a1, elev=args.elev, azim=args.azim)
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
