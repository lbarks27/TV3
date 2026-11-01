from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider

from .attitude import quaternion_to_euler, rotate_vector, world_to_body
from .flight_log import FlightLog
from .geometry import CatchArm, Mount, motor_aft_position, thrust_direction
from .parameters import LandingAcceptanceBox, SystemConfig, load_config
from . import viz_theme as theme


@dataclass
class _VehicleArtists:
    """Vehicle overlay shared by world-frame and body-frame panels."""

    body_x: object
    body_y: object
    body_z: object
    force_cmd: object
    force_act: object
    torque_cmd: object
    torque_act: object
    mount_scatter: object
    com_marker: object
    com_plant_marker: object
    motor_cg_scatter: object
    mass_force: object
    mass_torque: object
    ring_line: object
    catch_arm_lines: list
    mast_line: object
    motor_lines: list
    thrust_lines: list
    force_segments: list
    torque_segments: list
    setpoint_marker: object
    setpoint_x: object
    setpoint_y: object
    setpoint_z: object


@dataclass
class _SceneExtras:
    trail_line: object
    ref_trail_line: object
    world_x: object
    world_y: object
    world_z: object


def _draw_arrow(
    ax,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    color: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
    alpha: float = 1.0,
):
    tip = origin + direction
    return ax.plot(
        [origin[0], tip[0]],
        [origin[1], tip[1]],
        [origin[2], tip[2]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )[0]


def _style_axes(
    ax,
    *,
    title: str,
    span: float,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    center: np.ndarray | None = None,
) -> None:
    center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
    theme.style_3d_axes(ax, title=title)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_zlabel(zlabel, labelpad=4)


def _style_button(button: Button) -> None:
    theme.style_button(button, active=False)
    button.label.set_fontsize(10)


def _style_slider(slider: Slider, *, track: str = theme.SPINE, handle: str = theme.ACCENT) -> None:
    theme.style_slider(slider, active=True)
    slider.valtext.set_fontsize(9)
    if hasattr(slider, "track"):
        slider.track.set_color(track)
    if hasattr(slider, "poly"):
        slider.poly.set_color(handle)
    if hasattr(slider, "hline"):
        slider.hline.set_color(track)


def _set_arrow(line, origin: np.ndarray, direction: np.ndarray) -> None:
    tip = origin + direction
    line.set_data_3d([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]])


def _fmt_vec3(vec: np.ndarray, *, prec: int = 2, deg: bool = False) -> str:
    v = np.asarray(vec, dtype=float).reshape(3)
    if deg:
        v = np.rad2deg(v)
    if not np.all(np.isfinite(v)):
        return "(—)"
    return f"({v[0]:+.{prec}f},{v[1]:+.{prec}f},{v[2]:+.{prec}f})"


def _fmt_bool(flag: bool) -> str:
    return "ON " if flag else "off"


def _format_telemetry_hud(
    sample,
    *,
    metadata,
    index: int,
    n_samples: int,
    direction_txt: str,
    speed: float,
    force_axis: np.ndarray,
) -> str:
    """Multi-line status strip for replay: state machines + module telemetry."""
    origin = np.asarray(sample.position, dtype=float)
    velocity = np.asarray(sample.velocity, dtype=float)
    pos_ref = np.asarray(sample.position_reference, dtype=float)
    vel_ref = np.asarray(sample.velocity_reference, dtype=float)
    quaternion = np.asarray(sample.quaternion, dtype=float)
    q_ref = np.asarray(sample.quaternion_reference, dtype=float)
    omega = np.asarray(sample.omega, dtype=float)
    tau_cmd = np.asarray(sample.torque_command, dtype=float)
    tau_act = np.asarray(sample.torque_achieved, dtype=float)
    force_cmd = np.asarray(sample.force_command, dtype=float)
    force_body = np.asarray(sample.force_body, dtype=float)
    force_inertial = np.asarray(sample.force_inertial, dtype=float)
    att_err = np.asarray(sample.attitude_error, dtype=float)
    att_int = np.asarray(sample.att_integral, dtype=float)
    mount_thrusts = np.asarray(sample.mount_thrusts, dtype=float).reshape(-1)
    angles = np.asarray(sample.gimbal_angles, dtype=float).reshape(-1, 2)

    # Older logs may lack force_command; fall back to metadata axial target.
    if float(np.linalg.norm(force_cmd)) < 1e-12 and abs(metadata.force_target) > 1e-12:
        force_cmd = force_axis * float(metadata.force_target)

    roll, pitch, yaw = quaternion_to_euler(quaternion)
    roll_r, pitch_r, yaw_r = quaternion_to_euler(q_ref)
    pos_err = pos_ref - origin
    if not np.all(np.isfinite(pos_ref)):
        pos_err = np.full(3, np.nan)
    speed_mps = float(np.linalg.norm(velocity))
    omega_deg = np.rad2deg(omega)
    tau_err = tau_cmd - tau_act
    force_axis_u = force_axis / max(float(np.linalg.norm(force_axis)), 1e-9)
    f_axial_cmd = float(sample.force_target_axial)
    if abs(f_axial_cmd) < 1e-12:
        f_axial_cmd = float(np.dot(force_cmd, force_axis_u))
    f_axial_act = float(np.dot(force_body, force_axis_u))
    f_axial_err = f_axial_cmd - f_axial_act
    total_thrust = float(np.sum(mount_thrusts)) if mount_thrusts.size else 0.0

    phase = str(sample.mission_phase or "").strip() or "—"
    att_mode = str(sample.att_mode or "").strip() or "—"
    wp_adv = str(sample.waypoint_advance or "").strip() or "—"
    wp_idx = int(sample.waypoint_index)
    wp_cnt = int(sample.waypoint_count)
    if wp_cnt > 0:
        wp_txt = f"{wp_idx + 1}/{wp_cnt} ({wp_adv})"
    elif att_mode == "vertical":
        wp_txt = "none"
    elif wp_adv == "real_landing":
        wp_txt = "gate→pad"
    elif wp_adv != "—":
        wp_txt = wp_adv
    else:
        wp_txt = "—"

    heading = float(sample.heading_ref_rad)
    if np.isfinite(heading):
        heading_txt = f"{np.rad2deg(heading):+.1f}°"
    else:
        heading_txt = "free"

    landing_txt = ""
    if metadata.landing_success:
        label = "SUCCESS" if metadata.landing_success == "true" else "FAIL"
        landing_txt = f"  land={label}"
        if metadata.burnout_time_s >= 0.0:
            landing_txt += f"@t={metadata.burnout_time_s:.2f}s"

    thrust_txt = ",".join(f"{value:.1f}" for value in mount_thrusts) if mount_thrusts.size else "—"
    if angles.size:
        gimbal_bits = []
        for idx, (a2, a1) in enumerate(angles):
            gimbal_bits.append(
                f"M{idx}[{np.rad2deg(a2):+5.1f},{np.rad2deg(a1):+5.1f}]"
            )
        gimbal_txt = " ".join(gimbal_bits)
    else:
        gimbal_txt = "—"

    # Prefer logged attitude_error; fall back to euler delta for older logs.
    if float(np.linalg.norm(att_err)) > 1e-12:
        att_err_deg = np.rad2deg(att_err)
        att_err_txt = _fmt_vec3(att_err_deg, prec=1)
    else:
        att_err_txt = (
            f"({np.rad2deg(roll - roll_r):+.1f},"
            f"{np.rad2deg(pitch - pitch_r):+.1f},"
            f"{np.rad2deg(yaw - yaw_r):+.1f})"
        )

    line1 = (
        f"t={sample.time:6.2f}s  {direction_txt} {speed:.2f}x  "
        f"[{index + 1}/{n_samples}]  mode={metadata.motion_mode}  "
        f"scenario={metadata.scenario}{landing_txt}"
    )
    line2 = (
        f"SM  phase={phase:<8}  wp={wp_txt:<16}  att_mode={att_mode:<8}  "
        f"lift={_fmt_bool(sample.lift_authorized)}  "
        f"mix={_fmt_bool(sample.axial_mix_authorized)}  "
        f"w={sample.force_weight:4.2f}  sat={sample.saturated}"
    )
    line3 = (
        f"NAV pos={_fmt_vec3(origin)}  ref={_fmt_vec3(pos_ref)}  "
        f"err={_fmt_vec3(pos_err)}  |v|={speed_mps:5.2f}  "
        f"v={_fmt_vec3(velocity)}  vref={_fmt_vec3(vel_ref)}"
    )
    line4 = (
        f"ATT rpy=({np.rad2deg(roll):+6.1f},{np.rad2deg(pitch):+6.1f},{np.rad2deg(yaw):+6.1f})°  "
        f"ref=({np.rad2deg(roll_r):+6.1f},{np.rad2deg(pitch_r):+6.1f},{np.rad2deg(yaw_r):+6.1f})°  "
        f"err={att_err_txt}°  ω={_fmt_vec3(omega_deg, prec=1)}°/s  "
        f"hdg_ref={heading_txt}"
    )
    line5 = (
        f"CTL τcmd={_fmt_vec3(tau_cmd)}  τact={_fmt_vec3(tau_act)}  "
        f"τerr={_fmt_vec3(tau_err)}  "
        f"∫att={_fmt_vec3(att_int, prec=3)}  ∫alt={sample.alt_integral:+7.3f}"
    )
    line6 = (
        f"MIX F*ax={f_axial_cmd:6.2f}  Fax={f_axial_act:6.2f}  "
        f"Ferr={f_axial_err:+6.2f}  "
        f"Fcmd={_fmt_vec3(force_cmd, prec=1)}  "
        f"Fbody={_fmt_vec3(force_body, prec=1)}  "
        f"Finert={_fmt_vec3(force_inertial, prec=1)}  ΣT={total_thrust:5.1f}N"
    )
    line7 = f"ACT T=[{thrust_txt}]N  gimbal_deg {gimbal_txt}"
    com_plant = np.asarray(sample.com_plant, dtype=float).reshape(3)
    com_alloc = np.asarray(sample.com_alloc, dtype=float).reshape(3)
    tau_mass = np.asarray(sample.torque_mass, dtype=float).reshape(3)
    f_mass = np.asarray(sample.force_mass, dtype=float).reshape(3)
    cgs = np.asarray(sample.motor_cgs, dtype=float).reshape(-1, 3)
    if cgs.size:
        cg_txt = " ".join(
            f"M{i}{_fmt_vec3(cgs[i], prec=3)}" for i in range(cgs.shape[0])
        )
    else:
        cg_txt = "—"
    line8 = (
        f"MASS CoMplant={_fmt_vec3(com_plant, prec=3)}  "
        f"CoMalloc={_fmt_vec3(com_alloc, prec=3)}  "
        f"τmass={_fmt_vec3(tau_mass)}  Fmass={_fmt_vec3(f_mass, prec=2)}  "
        f"CG {cg_txt}"
    )
    return "\n".join((line1, line2, line3, line4, line5, line6, line7, line8))


def _load_system(
    config: SystemConfig | None,
    config_path: str | Path | None,
) -> tuple[list[Mount], np.ndarray, np.ndarray, list[CatchArm], list[CatchArm], float | None]:
    """Mounts, CoM, force axis, catch arms, landing legs, mast top.

    Landing legs replace catch arms in the drawing. A vehicle with legs does
    not also draw its catch arms.
    """
    loaded = config
    if loaded is None and config_path is not None:
        loaded = load_config(config_path)
    if loaded is None:
        default_path = Path("config.example.json")
        if default_path.exists():
            loaded = load_config(default_path)
    if loaded is None:
        return (
            [],
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
            [],
            [],
            None,
        )
    mast = getattr(loaded, "mast_top_m", None)
    return (
        loaded.mounts,
        np.asarray(loaded.center_of_mass, dtype=float),
        np.asarray(loaded.render.force_axis, dtype=float),
        list(loaded.catch_arms),
        list(getattr(loaded, "landing_legs", []) or []),
        None if mast is None else float(mast),
    )


def _scene_bounds(
    positions: np.ndarray,
    ref_positions: np.ndarray,
    extra_points: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    chunks = [np.zeros((1, 3), dtype=float)]
    if positions.size:
        chunks.append(np.asarray(positions, dtype=float).reshape(-1, 3))
    if ref_positions.size:
        chunks.append(np.asarray(ref_positions, dtype=float).reshape(-1, 3))
    if extra_points is not None and np.asarray(extra_points).size:
        chunks.append(np.asarray(extra_points, dtype=float).reshape(-1, 3))
    cloud = np.vstack(chunks)
    finite = np.isfinite(cloud).all(axis=1)
    if not np.any(finite):
        return np.zeros(3, dtype=float), 0.8
    cloud = cloud[finite]
    mins = np.min(cloud, axis=0)
    maxs = np.max(cloud, axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins)
    half_span = max(0.8, float(np.max(half)) * 1.2 + 0.35)
    return center, half_span


def _landing_box_from_sources(
    log: FlightLog,
    config: SystemConfig | None,
) -> LandingAcceptanceBox | None:
    if config is not None and config.profile is not None and config.profile.landing_acceptance is not None:
        return config.profile.landing_acceptance
    return LandingAcceptanceBox.from_metadata(log.metadata)


def _draw_landing_acceptance_box(
    ax,
    box: LandingAcceptanceBox,
    *,
    success: str = "",
) -> list:
    """Draw an axis-aligned wireframe acceptance box in world coordinates."""
    corners = box.corners()
    # Edge connectivity for a box (pairs of corner indices).
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    if success == "true":
        color = "#40c057"
        alpha = 0.95
    elif success == "false":
        color = "#ff6b6b"
        alpha = 0.9
    else:
        color = "#fcc419"
        alpha = 0.85
    artists = []
    for i0, i1 in edges:
        a = corners[i0]
        b = corners[i1]
        line = ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=color,
            linewidth=2.0,
            linestyle="-",
            alpha=alpha,
        )[0]
        artists.append(line)
    # Center marker for the pad.
    center = np.asarray(box.center_m, dtype=float)
    marker = ax.scatter(
        [center[0]],
        [center[1]],
        [center[2]],
        s=60,
        c=color,
        marker="s",
        depthshade=False,
        alpha=0.9,
    )
    artists.append(marker)
    return artists


def _draw_chopsticks(ax, chopsticks) -> list:
    """Draw the two X-parallel pad catch bars."""
    artists = []
    for (p0, p1) in chopsticks.segments():
        line = ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#fab005",
            linewidth=3.4,
            solid_capstyle="round",
            alpha=0.95,
        )[0]
        artists.append(line)
    return artists


def _create_vehicle_artists(
    ax,
    mounts: list[Mount],
    axis_scale: float,
    catch_arms: list[CatchArm] | None = None,
) -> _VehicleArtists:
    force_seg_colors = ("#5eead4", "#2dd4bf", "#14b8a6")
    torque_seg_colors = ("#86efac", "#4ade80", "#22c55e")
    arms = list(catch_arms or [])
    return _VehicleArtists(
        body_x=_draw_arrow(ax, np.zeros(3), np.array([axis_scale, 0.0, 0.0]), color="#ff6b6b"),
        body_y=_draw_arrow(ax, np.zeros(3), np.array([0.0, axis_scale, 0.0]), color="#51cf66"),
        body_z=_draw_arrow(ax, np.zeros(3), np.array([0.0, 0.0, axis_scale]), color="#4dabf7"),
        force_cmd=_draw_arrow(
            ax, np.zeros(3), np.zeros(3), color="#5eead4", linewidth=1.2, linestyle="--", alpha=0.95
        ),
        force_act=_draw_arrow(ax, np.zeros(3), np.zeros(3), color="#2dd4bf", linewidth=1.3),
        torque_cmd=_draw_arrow(
            ax, np.zeros(3), np.zeros(3), color="#c084fc", linewidth=1.2, linestyle="--", alpha=0.95
        ),
        torque_act=_draw_arrow(ax, np.zeros(3), np.zeros(3), color="#a855f7", linewidth=1.3),
        mount_scatter=ax.scatter([], [], [], s=44, c="#adb5bd", depthshade=False),
        com_marker=ax.scatter([], [], [], s=72, c="#ffd43b", marker="+", linewidths=2.0),
        com_plant_marker=ax.scatter(
            [], [], [], s=56, c="#ff922b", marker="o", linewidths=1.2, depthshade=False
        ),
        motor_cg_scatter=ax.scatter(
            [], [], [], s=36, c="#ffa94d", marker="d", linewidths=0.8, depthshade=False
        ),
        mass_force=_draw_arrow(
            ax, np.zeros(3), np.zeros(3), color="#ff922b", linewidth=1.4, linestyle=":"
        ),
        mass_torque=_draw_arrow(
            ax, np.zeros(3), np.zeros(3), color="#e8590c", linewidth=1.4, linestyle=":"
        ),
        ring_line=ax.plot([], [], [], color="#868e96", linewidth=1.2, alpha=0.85)[0],
        catch_arm_lines=[
            ax.plot([], [], [], color="#e9ecef", linewidth=2.2, alpha=0.95)[0] for _ in arms
        ],
        mast_line=ax.plot([], [], [], color="#868e96", linewidth=1.4, linestyle="--", alpha=0.7)[0],
        motor_lines=[
            ax.plot([], [], [], color="#ff922b", linewidth=2.6, solid_capstyle="round", alpha=0.95)[0]
            for _ in mounts
        ],
        thrust_lines=[
            _draw_arrow(ax, np.zeros(3), np.zeros(3), color="#fcc419", linewidth=2.4)
            for _ in mounts
        ],
        force_segments=[
            _draw_arrow(
                ax,
                np.zeros(3),
                np.zeros(3),
                color=force_seg_colors[index % len(force_seg_colors)],
                linewidth=0.9,
                alpha=0.8,
            )
            for index in range(len(mounts))
        ],
        torque_segments=[
            _draw_arrow(
                ax,
                np.zeros(3),
                np.zeros(3),
                color=torque_seg_colors[index % len(torque_seg_colors)],
                linewidth=0.9,
                alpha=0.8,
            )
            for index in range(len(mounts))
        ],
        setpoint_marker=ax.scatter(
            [], [], [], s=90, c="#ff6b6b", marker="x", linewidths=2.2, depthshade=False
        ),
        setpoint_x=_draw_arrow(ax, np.zeros(3), np.zeros(3), color="#ff6b6b", linewidth=1.6, alpha=0.9),
        setpoint_y=_draw_arrow(ax, np.zeros(3), np.zeros(3), color="#51cf66", linewidth=1.6, alpha=0.9),
        setpoint_z=_draw_arrow(ax, np.zeros(3), np.zeros(3), color="#4dabf7", linewidth=1.6, alpha=0.9),
    )


def _create_scene_extras(ax, axis_len: float) -> _SceneExtras:
    return _SceneExtras(
        trail_line=ax.plot([], [], [], color="#dee2e6", linewidth=0.9, alpha=0.9)[0],
        ref_trail_line=ax.plot([], [], [], color="#495057", linewidth=0.7, linestyle="--", alpha=0.75)[0],
        world_x=_draw_arrow(ax, np.zeros(3), np.array([axis_len, 0.0, 0.0]), color="#ff6b6b", linewidth=1.8),
        world_y=_draw_arrow(ax, np.zeros(3), np.array([0.0, axis_len, 0.0]), color="#51cf66", linewidth=1.8),
        world_z=_draw_arrow(ax, np.zeros(3), np.array([0.0, 0.0, axis_len]), color="#4dabf7", linewidth=1.8),
    )


def _update_vehicle_artists(
    artists: _VehicleArtists,
    *,
    mounts: list[Mount],
    angles: np.ndarray,
    mount_thrusts: np.ndarray,
    com_body: np.ndarray,
    force_command: np.ndarray,
    force_body: np.ndarray,
    torque_command: np.ndarray,
    torque_achieved: np.ndarray,
    position_world: np.ndarray,
    quaternion: np.ndarray,
    position_reference: np.ndarray | None,
    axis_scale: float,
    thrust_draw_scale: float,
    force_scale: float,
    tau_scale: float,
    setpoint_axis_len: float,
    frame: str,
    catch_arms: list[CatchArm] | None = None,
    mast_top_m: float | None = None,
    motor_standoff_m: float = 0.055,
    com_plant: np.ndarray | None = None,
    motor_cgs: np.ndarray | None = None,
    force_mass: np.ndarray | None = None,
    torque_mass: np.ndarray | None = None,
) -> None:
    """Draw the full vehicle overlay in world or body coordinates."""
    if frame not in {"world", "body"}:
        raise ValueError("frame must be 'world' or 'body'")

    def to_frame(vector_body: np.ndarray) -> np.ndarray:
        if frame == "body":
            return np.asarray(vector_body, dtype=float)
        return rotate_vector(quaternion, vector_body)

    origin = np.zeros(3, dtype=float) if frame == "body" else np.asarray(position_world, dtype=float)
    com = origin + to_frame(com_body)
    plant = np.asarray(com_body if com_plant is None else com_plant, dtype=float)
    com_p = origin + to_frame(plant)

    _set_arrow(artists.body_x, origin, to_frame(np.array([axis_scale, 0.0, 0.0])))
    _set_arrow(artists.body_y, origin, to_frame(np.array([0.0, axis_scale, 0.0])))
    _set_arrow(artists.body_z, origin, to_frame(np.array([0.0, 0.0, axis_scale])))
    _set_arrow(artists.torque_cmd, com, to_frame(np.asarray(torque_command, dtype=float) * tau_scale))
    _set_arrow(artists.torque_act, com, to_frame(np.asarray(torque_achieved, dtype=float) * tau_scale))
    _set_arrow(artists.force_cmd, com, to_frame(np.asarray(force_command, dtype=float) * force_scale))
    _set_arrow(artists.force_act, com, to_frame(np.asarray(force_body, dtype=float) * force_scale))

    mount_body = np.vstack([mount.position for mount in mounts])
    mount_frame = np.vstack([origin + to_frame(point) for point in mount_body])
    artists.mount_scatter._offsets3d = (mount_frame[:, 0], mount_frame[:, 1], mount_frame[:, 2])
    artists.com_marker._offsets3d = ([com[0]], [com[1]], [com[2]])
    artists.com_plant_marker._offsets3d = ([com_p[0]], [com_p[1]], [com_p[2]])
    if motor_cgs is not None and np.asarray(motor_cgs).size:
        cg_frame = np.vstack([origin + to_frame(p) for p in np.asarray(motor_cgs, dtype=float).reshape(-1, 3)])
        artists.motor_cg_scatter._offsets3d = (cg_frame[:, 0], cg_frame[:, 1], cg_frame[:, 2])
    else:
        artists.motor_cg_scatter._offsets3d = ([], [], [])
    f_mass = np.zeros(3, dtype=float) if force_mass is None else np.asarray(force_mass, dtype=float)
    t_mass = np.zeros(3, dtype=float) if torque_mass is None else np.asarray(torque_mass, dtype=float)
    _set_arrow(artists.mass_force, com_p, to_frame(f_mass * force_scale))
    _set_arrow(artists.mass_torque, com_p, to_frame(t_mass * tau_scale))
    if len(mount_frame) >= 2:
        closed = np.vstack([mount_frame, mount_frame[0]])
        artists.ring_line.set_data_3d(closed[:, 0], closed[:, 1], closed[:, 2])

    arms = list(catch_arms or [])
    mast_root = None
    for index, arm in enumerate(arms):
        if index >= len(artists.catch_arm_lines):
            break
        root_body = np.asarray(arm.position, dtype=float)
        tip_body = arm.tip()
        root = origin + to_frame(root_body)
        tip = origin + to_frame(tip_body)
        artists.catch_arm_lines[index].set_data_3d(
            [root[0], tip[0]], [root[1], tip[1]], [root[2], tip[2]]
        )
        if mast_root is None and float(np.linalg.norm(root_body[:2])) < 1e-9:
            mast_root = root_body
    if mast_top_m is not None:
        mast_root = np.array([0.0, 0.0, float(mast_top_m)], dtype=float)
    if mast_root is not None:
        mast_tip = origin + to_frame(mast_root)
        artists.mast_line.set_data_3d(
            [origin[0], mast_tip[0]],
            [origin[1], mast_tip[1]],
            [origin[2], mast_tip[2]],
        )
    else:
        artists.mast_line.set_data_3d([], [], [])

    for mount_index, mount in enumerate(mounts):
        if mount_index >= len(artists.motor_lines) or mount_index >= angles.shape[0]:
            continue
        a2, a1 = angles[mount_index]
        aft_body = motor_aft_position(
            mount, float(a2), float(a1), standoff_m=float(motor_standoff_m)
        )
        pivot = mount_frame[mount_index]
        aft = origin + to_frame(aft_body)
        artists.motor_lines[mount_index].set_data_3d(
            [pivot[0], aft[0]], [pivot[1], aft[1]], [pivot[2], aft[2]]
        )

    thrusts = np.asarray(mount_thrusts, dtype=float).reshape(-1)
    force_cumulative = np.zeros(3, dtype=float)
    torque_cumulative = np.zeros(3, dtype=float)
    for mount_index, mount in enumerate(mounts):
        if mount_index >= angles.shape[0]:
            _set_arrow(artists.thrust_lines[mount_index], mount_frame[mount_index], np.zeros(3))
            _set_arrow(artists.force_segments[mount_index], com, np.zeros(3))
            _set_arrow(artists.torque_segments[mount_index], com, np.zeros(3))
            continue
        a2, a1 = angles[mount_index]
        thrust_n = float(thrusts[mount_index]) if mount_index < thrusts.size else float(mount.thrust)
        direction_body = thrust_direction(a2, a1, mount)
        if thrust_n <= 1e-9:
            _set_arrow(artists.thrust_lines[mount_index], mount_frame[mount_index], np.zeros(3))
            force_vec = np.zeros(3, dtype=float)
        else:
            _set_arrow(
                artists.thrust_lines[mount_index],
                mount_frame[mount_index],
                to_frame(direction_body * (thrust_n * thrust_draw_scale)),
            )
            force_vec = direction_body * thrust_n

        # Cumulative force chain from CoM (each segment = one engine's force contribution).
        force_start = com + to_frame(force_cumulative * force_scale)
        force_cumulative = force_cumulative + force_vec
        force_end = com + to_frame(force_cumulative * force_scale)
        artists.force_segments[mount_index].set_data_3d(
            [force_start[0], force_end[0]],
            [force_start[1], force_end[1]],
            [force_start[2], force_end[2]],
        )

        # Cumulative torque chain from CoM (each segment = one engine's torque contribution).
        torque_seg = np.cross(mount.position - com_body, force_vec)
        torque_start = com + to_frame(torque_cumulative * tau_scale)
        torque_cumulative = torque_cumulative + torque_seg
        torque_end = com + to_frame(torque_cumulative * tau_scale)
        artists.torque_segments[mount_index].set_data_3d(
            [torque_start[0], torque_end[0]],
            [torque_start[1], torque_end[1]],
            [torque_start[2], torque_end[2]],
        )

    if position_reference is None:
        artists.setpoint_marker._offsets3d = ([], [], [])
        _set_arrow(artists.setpoint_x, origin, np.zeros(3))
        _set_arrow(artists.setpoint_y, origin, np.zeros(3))
        _set_arrow(artists.setpoint_z, origin, np.zeros(3))
        return

    # Setpoint triad is always world-aligned (RGB = world XYZ), expressed in the panel frame.
    setpoint_world = np.asarray(position_reference, dtype=float)
    world_axis_x = np.array([setpoint_axis_len, 0.0, 0.0])
    world_axis_y = np.array([0.0, setpoint_axis_len, 0.0])
    world_axis_z = np.array([0.0, 0.0, setpoint_axis_len])
    if frame == "world":
        setpoint = setpoint_world
        axis_x, axis_y, axis_z = world_axis_x, world_axis_y, world_axis_z
    else:
        setpoint = world_to_body(quaternion, setpoint_world - position_world)
        axis_x = world_to_body(quaternion, world_axis_x)
        axis_y = world_to_body(quaternion, world_axis_y)
        axis_z = world_to_body(quaternion, world_axis_z)

    artists.setpoint_marker._offsets3d = ([setpoint[0]], [setpoint[1]], [setpoint[2]])
    _set_arrow(artists.setpoint_x, setpoint, axis_x)
    _set_arrow(artists.setpoint_y, setpoint, axis_y)
    _set_arrow(artists.setpoint_z, setpoint, axis_z)


def _points_to_body(
    points_world: np.ndarray,
    *,
    position_world: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=float).reshape(-1, 3)
    if points.size == 0:
        return np.zeros((0, 3), dtype=float)
    relative = points - np.asarray(position_world, dtype=float)
    return np.vstack([world_to_body(quaternion, row) for row in relative])


def _update_scene_extras(
    extras: _SceneExtras,
    *,
    positions: np.ndarray,
    ref_positions: np.ndarray,
    index: int,
    position_world: np.ndarray | None = None,
    quaternion: np.ndarray | None = None,
    frame: str = "world",
    axis_len: float = 0.4,
) -> None:
    """Update path / reference trail / world-origin axes in world or body coordinates."""
    trail = np.asarray(positions[: index + 1], dtype=float).reshape(-1, 3)
    ref_trail = np.asarray(ref_positions[: index + 1], dtype=float).reshape(-1, 3)

    if frame == "body":
        if position_world is None or quaternion is None:
            raise ValueError("body-frame extras need position_world and quaternion")
        trail = _points_to_body(
            trail, position_world=position_world, quaternion=quaternion
        )
        if ref_trail.size:
            ref_trail = _points_to_body(
                ref_trail, position_world=position_world, quaternion=quaternion
            )
        origin_body = world_to_body(quaternion, -np.asarray(position_world, dtype=float))
        axis_x = world_to_body(quaternion, np.array([axis_len, 0.0, 0.0]))
        axis_y = world_to_body(quaternion, np.array([0.0, axis_len, 0.0]))
        axis_z = world_to_body(quaternion, np.array([0.0, 0.0, axis_len]))
        _set_arrow(extras.world_x, origin_body, axis_x)
        _set_arrow(extras.world_y, origin_body, axis_y)
        _set_arrow(extras.world_z, origin_body, axis_z)
    else:
        _set_arrow(extras.world_x, np.zeros(3), np.array([axis_len, 0.0, 0.0]))
        _set_arrow(extras.world_y, np.zeros(3), np.array([0.0, axis_len, 0.0]))
        _set_arrow(extras.world_z, np.zeros(3), np.array([0.0, 0.0, axis_len]))

    extras.trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
    if ref_trail.size:
        ok = np.isfinite(ref_trail).all(axis=1)
        ref_trail = ref_trail[ok]
    if ref_trail.size:
        extras.ref_trail_line.set_data_3d(ref_trail[:, 0], ref_trail[:, 1], ref_trail[:, 2])
    else:
        extras.ref_trail_line.set_data_3d([], [], [])


def _body_view_span(
    *,
    vehicle_span: float,
    positions: np.ndarray,
    ref_positions: np.ndarray,
) -> float:
    """Half-span so body pane can show the vehicle, world origin, and waypoint."""
    distances = [float(vehicle_span)]
    if positions.size:
        distances.append(float(np.nanmax(np.linalg.norm(positions, axis=1))))
    if ref_positions.size:
        refs = np.asarray(ref_positions, dtype=float).reshape(-1, 3)
        ok = np.isfinite(refs).all(axis=1)
        if np.any(ok):
            distances.append(float(np.max(np.linalg.norm(refs[ok], axis=1))))
            if positions.size:
                pos = np.asarray(positions, dtype=float).reshape(-1, 3)
                n = min(len(pos), len(refs))
                both = ok[:n] & np.isfinite(pos[:n]).all(axis=1)
                if np.any(both):
                    distances.append(
                        float(np.max(np.linalg.norm(pos[:n][both] - refs[:n][both], axis=1)))
                    )
    return max(distances) * 1.2 + 0.5


def replay_3d(
    log: FlightLog,
    *,
    config: SystemConfig | None = None,
    config_path: str | Path | None = None,
    title: str | None = None,
    interval_ms: float | None = None,
) -> None:
    if not log.samples:
        raise ValueError("Cannot replay an empty flight log.")

    mounts, com_body, force_axis, catch_arms, landing_legs, mast_top_m = _load_system(
        config, config_path
    )
    # Legs replace the catch arms in the drawing. Chopsticks stay on catch arms only.
    structure = landing_legs if landing_legs else catch_arms
    motor_standoff_m = 0.055
    if config is not None:
        motor_standoff_m = float(getattr(config, "motor_pivot_standoff_m", 0.055) or 0.055)
    default_com = np.asarray(com_body, dtype=float).reshape(3).copy()
    if not mounts:
        raise ValueError("3D replay needs mount geometry from --config or config.example.json.")
    force_axis = force_axis / max(float(np.linalg.norm(force_axis)), 1e-9)
    times = log.times()
    positions = log.field_array("position")
    ref_positions = log.field_array("position_reference")
    torque_commands = log.field_array("torque_command")
    torque_achieved = log.field_array("torque_achieved")
    force_commands = log.field_array("force_command")
    forces = log.field_array("force_body")
    landing_box = _landing_box_from_sources(log, config)
    chopsticks = None
    pad_xy = None
    if config is not None and config.profile is not None and config.profile.real_landing is not None:
        pad = config.profile.real_landing.landing_end_m()
        pad_xy = (float(pad[0]), float(pad[1]))
    elif landing_box is not None:
        pad_xy = (float(landing_box.center_m[0]), float(landing_box.center_m[1]))
    if pad_xy is not None and catch_arms and not landing_legs:
        from tvc.chopsticks import chopsticks_for_pad

        chopsticks = chopsticks_for_pad(pad_xy, list(catch_arms))
    extra_chunks = []
    if landing_box is not None:
        extra_chunks.append(np.asarray(landing_box.corners(), dtype=float))
    if chopsticks is not None:
        extra_chunks.append(np.asarray(chopsticks.points(), dtype=float))
    landing_extra = np.vstack(extra_chunks) if extra_chunks else None

    if log.metadata.dt > 0:
        sample_dt = float(log.metadata.dt)
    elif times.size > 1:
        sample_dt = float(np.median(np.diff(times)))
    else:
        sample_dt = 0.01
    frame_interval_ms = (
        float(interval_ms) if interval_ms is not None else max(1.0, sample_dt * 1000.0)
    )
    poll_interval_ms = max(1.0, min(frame_interval_ms, 16.0))

    mount_reach = float(np.max(np.linalg.norm([mount.position for mount in mounts], axis=1)))
    arm_reach = 0.0
    for arm in structure:
        arm_reach = max(
            arm_reach,
            float(np.linalg.norm(arm.position)),
            float(np.linalg.norm(arm.tip())),
        )
    arrow_ref_length = max(0.08, mount_reach * 0.95)
    axis_scale = arrow_ref_length * 0.75
    vehicle_span = max(
        0.35,
        mount_reach * 3.0,
        arm_reach * 1.15,
        float(np.linalg.norm(com_body)) * 1.5,
        arrow_ref_length * 2.5,
    )
    scene_center, scene_span = _scene_bounds(positions, ref_positions, landing_extra)
    # Leave room around the vehicle for thrust/force overlays in the world view.
    scene_span = max(scene_span, vehicle_span * 1.35)
    body_span = _body_view_span(
        vehicle_span=vehicle_span,
        positions=positions,
        ref_positions=ref_positions,
    )
    world_axis_len = max(0.4, scene_span * 0.12)
    setpoint_axis_len = max(0.35, world_axis_len * 0.55)

    max_tau = max(
        float(np.max(np.linalg.norm(torque_commands, axis=1), initial=0.0)),
        float(np.max(np.linalg.norm(torque_achieved, axis=1), initial=0.0)),
    )
    tau_scale = 0.12 / max(max_tau, 1e-3)
    max_force = max(
        float(np.max(np.linalg.norm(forces, axis=1), initial=0.0)),
        float(np.max(np.linalg.norm(force_commands, axis=1), initial=0.0)),
    )
    force_scale = 0.12 / max(max_force, 1e-3)

    logged_thrusts = []
    for sample in log.samples:
        thrusts = np.asarray(sample.mount_thrusts, dtype=float).reshape(-1)
        if thrusts.size:
            logged_thrusts.append(thrusts)
    if logged_thrusts:
        max_mount_thrust = float(np.max(np.vstack(logged_thrusts), initial=0.0))
    else:
        max_mount_thrust = float(max((mount.thrust for mount in mounts), default=0.0))
    thrust_draw_scale = arrow_ref_length / max(max_mount_thrust, 1e-3)

    fig = plt.figure(figsize=(14.0, 8.6))
    theme.apply_figure_theme(fig)

    grid = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        height_ratios=[1.0, 0.02],
        width_ratios=[1.35, 1.0],
        left=0.04,
        right=0.99,
        bottom=0.30,
        top=0.94,
        wspace=0.14,
        hspace=0.04,
    )

    ax_scene = fig.add_subplot(grid[0, 0], projection="3d")
    ax_body = fig.add_subplot(grid[0, 1], projection="3d")

    fig.suptitle(
        title or f"Flight replay: {log.metadata.scenario}",
        color=theme.TEXT,
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    _style_axes(
        ax_scene,
        title="World frame",
        span=scene_span,
        center=scene_center,
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
    )
    _style_axes(
        ax_body,
        title="Body frame",
        span=body_span,
        xlabel="Body X",
        ylabel="Body Y",
        zlabel="Body Z",
    )

    scene_vehicle = _create_vehicle_artists(ax_scene, mounts, axis_scale, catch_arms=structure)
    body_vehicle = _create_vehicle_artists(ax_body, mounts, axis_scale, catch_arms=structure)
    scene_extras = _create_scene_extras(ax_scene, world_axis_len)
    body_extras = _create_scene_extras(ax_body, world_axis_len)
    if landing_box is not None:
        _draw_landing_acceptance_box(
            ax_scene,
            landing_box,
            success=str(log.metadata.landing_success or ""),
        )
    if chopsticks is not None:
        _draw_chopsticks(ax_scene, chopsticks)

    # Multi-line telemetry HUD (state machines + control modules).
    info = fig.text(
        0.04,
        0.275,
        "",
        family="monospace",
        fontsize=8.2,
        va="top",
        ha="left",
        color=theme.TEXT,
        linespacing=1.35,
    )
    legend_bits = [
        "F cmd (-- teal) / F act (teal)",
        "τ cmd (-- purple) / τ act (purple)",
        "thrust (gold)",
        "CoM alloc (+) / plant (o)  motor CG (diamond)",
        "mass F/τ (dotted orange) — gimbal-offset motors, sim plant only",
        "world | body",
        "SM=mission  NAV=guidance  ATT=attitude  CTL=PID  MIX=allocator  ACT=servos",
    ]
    if landing_legs:
        legend_bits.append("landing legs")
    if landing_box is not None:
        status = str(log.metadata.landing_success or "")
        if status == "true":
            legend_bits.append("box=SUCCESS")
        elif status == "false":
            legend_bits.append("box=FAIL")
        else:
            legend_bits.append("acceptance box")
    fig.text(
        0.04,
        0.012,
        "  |  ".join(legend_bits),
        fontsize=8.0,
        color=theme.MUTED,
    )

    state = {
        "index": 0,
        "playing": True,
        "direction": 1,  # +1 forward, -1 reverse
        "speed": 1.0,
        "updating_slider": False,
        "updating_speed": False,
        "wall_anchor": None,
        "sim_anchor": float(times[0]),
    }

    def _sync_clock(index: int) -> None:
        state["wall_anchor"] = time.perf_counter()
        state["sim_anchor"] = float(times[index])

    def _index_for_time(sim_time: float) -> int:
        sim_time = float(np.clip(sim_time, times[0], times[-1]))
        index = int(np.searchsorted(times, sim_time, side="right") - 1)
        return int(np.clip(index, 0, len(times) - 1))

    def render_frame(index: int) -> None:
        sample = log.samples[index]
        origin = np.asarray(sample.position, dtype=float)
        quaternion = np.asarray(sample.quaternion, dtype=float)
        angles = np.asarray(sample.gimbal_angles, dtype=float).reshape(-1, 2)
        tau_cmd = np.asarray(sample.torque_command, dtype=float)
        tau_act = np.asarray(sample.torque_achieved, dtype=float)
        force_cmd = np.asarray(sample.force_command, dtype=float)
        # Older logs may lack force_command; fall back to metadata axial target.
        if float(np.linalg.norm(force_cmd)) < 1e-12 and abs(log.metadata.force_target) > 1e-12:
            force_cmd = force_axis * float(log.metadata.force_target)
        force_body = np.asarray(sample.force_body, dtype=float)
        position_ref = np.asarray(sample.position_reference, dtype=float)
        if not np.all(np.isfinite(position_ref)):
            position_ref = None
        mount_thrusts = np.asarray(sample.mount_thrusts, dtype=float).reshape(-1)
        if mount_thrusts.size == 0:
            mount_thrusts = np.array([mount.thrust for mount in mounts], dtype=float)
        # Capture the config CoM via default_com; do not read the name
        # ``com_body`` here (an assignment later in this closure made it
        # UnboundLocalError on the 3D replay).
        frame_com = np.asarray(default_com, dtype=float).reshape(3)
        alloc_com = np.asarray(sample.com_alloc, dtype=float).reshape(3)
        if float(np.linalg.norm(alloc_com)) > 1e-12:
            frame_com = alloc_com

        shared = dict(
            mounts=mounts,
            angles=angles,
            mount_thrusts=mount_thrusts,
            com_body=frame_com,
            force_command=force_cmd,
            force_body=force_body,
            torque_command=tau_cmd,
            torque_achieved=tau_act,
            position_world=origin,
            quaternion=quaternion,
            position_reference=position_ref,
            axis_scale=axis_scale,
            thrust_draw_scale=thrust_draw_scale,
            force_scale=force_scale,
            tau_scale=tau_scale,
            setpoint_axis_len=setpoint_axis_len,
            catch_arms=structure,
            mast_top_m=mast_top_m,
            motor_standoff_m=motor_standoff_m,
            com_plant=np.asarray(sample.com_plant, dtype=float),
            motor_cgs=np.asarray(sample.motor_cgs, dtype=float),
            force_mass=np.asarray(sample.force_mass, dtype=float),
            torque_mass=np.asarray(sample.torque_mass, dtype=float),
        )
        _update_scene_extras(
            scene_extras,
            positions=positions,
            ref_positions=ref_positions,
            index=index,
            frame="world",
            axis_len=world_axis_len,
        )
        _update_scene_extras(
            body_extras,
            positions=positions,
            ref_positions=ref_positions,
            index=index,
            position_world=origin,
            quaternion=quaternion,
            frame="body",
            axis_len=world_axis_len,
        )
        _update_vehicle_artists(scene_vehicle, frame="world", **shared)
        _update_vehicle_artists(body_vehicle, frame="body", **shared)

        direction_txt = "FWD" if state["direction"] > 0 else "REV"
        info.set_text(
            _format_telemetry_hud(
                sample,
                metadata=log.metadata,
                index=index,
                n_samples=len(log.samples),
                direction_txt=direction_txt,
                speed=float(state["speed"]),
                force_axis=force_axis,
            )
        )

        state["updating_slider"] = True
        slider_stride = max(1, int(round(0.05 / max(sample_dt, 1e-3))))
        if index % slider_stride == 0 or index in {0, len(log.samples) - 1}:
            slider.set_val(sample.time)
        state["updating_slider"] = False
        fig.canvas.draw_idle()

    # Transport bar under the telemetry HUD.
    # Row 1: full-width time scrubber
    # Row 2: play, direction, speed
    fig.text(0.04, 0.108, "time", fontsize=10, color=theme.TEXT, va="center", ha="left", fontweight="bold")
    ax_slider = fig.add_axes([0.11, 0.096, 0.84, 0.026])
    slider = Slider(
        ax_slider,
        "",
        float(times[0]),
        float(times[-1]),
        valinit=float(times[0]),
        valstep=sample_dt,
        **theme.slider_kwargs(),
    )
    _style_slider(slider)

    ax_play = fig.add_axes([0.11, 0.040, 0.09, 0.036])
    play_button = Button(ax_play, "Pause")
    _style_button(play_button)
    ax_dir = fig.add_axes([0.22, 0.040, 0.09, 0.036])
    dir_button = Button(ax_dir, "Fwd")
    _style_button(dir_button)
    fig.text(0.34, 0.057, "speed", fontsize=10, color=theme.TEXT, va="center", ha="left", fontweight="bold")
    ax_speed = fig.add_axes([0.41, 0.046, 0.54, 0.026])
    speed_slider = Slider(
        ax_speed,
        "",
        0.25,
        4.0,
        valinit=1.0,
        valstep=0.25,
        valfmt="%.2fx",
        **theme.slider_kwargs(),
    )
    _style_slider(speed_slider)

    def on_slider(value: float) -> None:
        if state["updating_slider"] or not times.size:
            return
        index = _index_for_time(value)
        if index != state["index"]:
            state["index"] = index
            _sync_clock(index)
            render_frame(index)

    def on_play(_event) -> None:
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            _sync_clock(state["index"])

    def on_direction(_event) -> None:
        state["direction"] = -1 if state["direction"] > 0 else 1
        dir_button.label.set_text("Fwd" if state["direction"] > 0 else "Rev")
        _sync_clock(state["index"])
        render_frame(state["index"])

    def on_speed(value: float) -> None:
        if state["updating_speed"]:
            return
        state["speed"] = max(0.25, float(value))
        _sync_clock(state["index"])
        render_frame(state["index"])

    slider.on_changed(on_slider)
    play_button.on_clicked(on_play)
    dir_button.on_clicked(on_direction)
    speed_slider.on_changed(on_speed)

    def animate(_frame: int):
        if not state["playing"]:
            return ()
        if state["wall_anchor"] is None:
            _sync_clock(state["index"])

        elapsed = time.perf_counter() - state["wall_anchor"]
        target_sim = state["sim_anchor"] + state["direction"] * state["speed"] * elapsed

        if target_sim > float(times[-1]):
            # Wrap forward past the end.
            overflow = target_sim - float(times[-1])
            target_sim = float(times[0]) + overflow
            state["index"] = _index_for_time(target_sim)
            _sync_clock(state["index"])
            render_frame(state["index"])
            return ()
        if target_sim < float(times[0]):
            # Wrap reverse past the start.
            underflow = float(times[0]) - target_sim
            target_sim = float(times[-1]) - underflow
            state["index"] = _index_for_time(target_sim)
            _sync_clock(state["index"])
            render_frame(state["index"])
            return ()

        next_index = _index_for_time(target_sim)
        if next_index != state["index"]:
            state["index"] = next_index
            render_frame(next_index)
        return ()

    _sync_clock(0)
    render_frame(0)
    anim = FuncAnimation(
        fig, animate, interval=poll_interval_ms, blit=False, cache_frame_data=False
    )
    fig._tvc_replay_anim = anim  # type: ignore[attr-defined]
    # Keep button refs alive for the interactive session.
    fig._tvc_replay_controls = (play_button, dir_button, slider, speed_slider)  # type: ignore[attr-defined]
    plt.show()
