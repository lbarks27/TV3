import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, Slider

from .allocator import TorqueAllocator
from .geometry import CatchArm, Mount, axis_angle_rotation, thrust_envelope_mesh, unit
from .parameters import RenderConfig
from . import viz_theme as theme


def format_angles_deg(angles: np.ndarray) -> str:
    lines = []
    for idx, (ax, ay) in enumerate(angles):
        lines.append(
            f"M{idx}: a2={np.rad2deg(ax):6.1f} deg  a1={np.rad2deg(ay):6.1f} deg"
        )
    return "\n".join(lines)


def format_vector(vec: np.ndarray) -> str:
    return f"[{vec[0]:6.2f}, {vec[1]:6.2f}, {vec[2]:6.2f}]"


def format_scalar(value: float) -> str:
    return f"{value:6.2f}"


def setup_plot(
    mounts: list[Mount],
    allocator: TorqueAllocator,
    render: RenderConfig,
    catch_arms: list[CatchArm] | None = None,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    theme.apply_figure_theme(fig)
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.05, right=0.78, bottom=0.14, top=0.95)
    theme.style_3d_axes(ax)
    text_color = theme.TEXT

    def slider_theme_kwargs() -> dict:
        return theme.slider_kwargs(active=True)

    positions = np.array([m.position for m in mounts])
    mount_scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=35,
        c="tab:blue",
    )

    torque_limit = (
        render.torque_limit
        if render.torque_limit is not None
        else max(1.0, allocator.torque_upper_bound)
    )
    radius = float(np.max(np.linalg.norm(positions, axis=1))) if len(positions) else 1.0
    max_mount_thrust = max(mount.thrust for mount in mounts) if mounts else 1.0
    force_scale = (
        render.force_scale if render.force_scale is not None else max(0.2, radius * 0.7)
    )
    force_axis = unit(np.asarray(allocator.force_axis, dtype=float))
    force_vector_scale = force_scale / max_mount_thrust
    axis_scale = max(0.06, radius * 0.45)
    envelope_scale = force_scale * 0.55
    line_alpha = 0.82
    plane_half_span = max(radius * 1.35, 0.12)

    def mount_axis_end(mount: Mount, direction: np.ndarray) -> np.ndarray:
        return mount.position + direction * axis_scale

    axis_specs = (
        ("base_dir", "#6ee7b7", (0, (2, 2)), 1.4, line_alpha * 0.88),
        ("axis2", "#f472b6", "-", 2.0, line_alpha),
        ("axis1", "#67b7ff", "-", 2.0, line_alpha),
    )
    mount_axis_lines: list[list] = []
    for mount in mounts:
        mount_lines = []
        for _name, color, linestyle, linewidth, alpha in axis_specs:
            start = mount.position
            end = mount_axis_end(mount, mount.base_dir)
            (line,) = ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )
            mount_lines.append(line)
        mount_axis_lines.append(mount_lines)

    thrust_lines = []
    envelope_collections: list[Poly3DCollection] = []
    envelope_colors = ("#fb923c", "#fbbf24", "#f97316")
    for idx, mount in enumerate(mounts):
        start = mount.position
        end = start + np.array([0.0, 0.0, force_scale])
        (line,) = ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="tab:orange",
            linewidth=2,
            alpha=line_alpha,
        )
        thrust_lines.append(line)

        envelope = Poly3DCollection(
            thrust_envelope_mesh(
                mount, envelope_scale, invert=True, samples_per_axis=10
            ),
            color=envelope_colors[idx % len(envelope_colors)],
            alpha=line_alpha * 0.22,
            zorder=0,
        )
        envelope.set_edgecolor("#fdba74")
        envelope.set_linewidth(0.35)
        envelope.set_linestyle((0, (1, 2)))
        ax.add_collection3d(envelope)
        envelope_collections.append(envelope)

    segment_lines = []
    for _ in mounts:
        (line,) = ax.plot(
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            color="tab:green",
            linewidth=0.8,
            alpha=line_alpha * 0.65,
            zorder=1,
        )
        segment_lines.append(line)

    (actual_line,) = ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        color="tab:green",
        linewidth=1.3,
        linestyle="-",
        alpha=line_alpha,
        zorder=2,
        label="actual torque",
    )
    target_line = ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color="tab:red",
        s=34,
        marker="o",
        alpha=line_alpha,
        zorder=4,
        label="target torque",
    )
    (actual_force_line,) = ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        color="#5eead4",
        linewidth=1.3,
        linestyle="-",
        alpha=line_alpha,
        zorder=2,
        label="net force",
    )
    target_force_plane = Poly3DCollection(
        [[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]],
        color="#c77dff",
        alpha=line_alpha * 0.18,
        zorder=4,
        label="target axial force",
    )
    target_force_plane.set_edgecolor("#c77dff")
    target_force_plane.set_linewidth(1.4)
    target_force_plane.set_linestyle((0, (1, 3)))
    ax.add_collection3d(target_force_plane)
    actual_force_segment_lines = []
    for _ in mounts:
        (line,) = ax.plot(
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            color="#5eead4",
            linewidth=0.8,
            alpha=line_alpha * 0.7,
            zorder=1,
        )
        actual_force_segment_lines.append(line)

    text = fig.text(
        0.02,
        0.98,
        "",
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        color=text_color,
    )

    slider_left = 0.06
    slider_width = 0.67
    slider_height = 0.028
    slider_gap = 0.012
    slider_top = 0.11
    slider_y = [
        slider_top - idx * (slider_height + slider_gap) for idx in range(4)
    ]

    ax_tx = fig.add_axes([slider_left, slider_y[0], slider_width, slider_height])
    ax_ty = fig.add_axes([slider_left, slider_y[1], slider_width, slider_height])
    ax_tz = fig.add_axes([slider_left, slider_y[2], slider_width, slider_height])
    ax_force = fig.add_axes([slider_left, slider_y[3], slider_width, slider_height])
    for slider_ax in (ax_tx, ax_ty, ax_tz, ax_force):
        theme.style_widget_axes(slider_ax)

    # Live dragging uses a 60 fps preview timer; settled drag triggers full solve/redraw.
    slider_tx = Slider(
        ax_tx, "Tx", -torque_limit, torque_limit, valinit=0.0, **slider_theme_kwargs()
    )
    slider_ty = Slider(
        ax_ty, "Ty", -torque_limit, torque_limit, valinit=0.0, **slider_theme_kwargs()
    )
    slider_tz = Slider(
        ax_tz, "Tz", -torque_limit, torque_limit, valinit=0.0, **slider_theme_kwargs()
    )
    slider_force = Slider(
        ax_force,
        "Fmag",
        0.0,
        allocator.force_upper_bound,
        valinit=allocator.rest_axial_force(
            np.array([mount.thrust for mount in mounts], dtype=float)
        ),
        **slider_theme_kwargs(),
    )

    for slider in (slider_tx, slider_ty, slider_tz, slider_force):
        theme.style_slider(slider, active=True)

    auto_sliders = [slider_tx, slider_ty, slider_tz, slider_force]
    mount_count = len(mounts)
    panel_left = 0.795
    panel_width = 0.19
    panel_btn_h = 0.030
    panel_btn_gap = 0.010
    section_label_h = 0.022
    display_btn_top = 0.808
    display_header_y = display_btn_top + panel_btn_h + 0.006
    vector_button_y_positions = [
        display_btn_top - idx * (panel_btn_h + panel_btn_gap) for idx in range(6)
    ]
    display_section_bottom = vector_button_y_positions[-1] - 0.012
    alloc_header_y = display_section_bottom - section_label_h
    mode_button_y = alloc_header_y - panel_btn_h - 0.006
    jog_header_y = mode_button_y - section_label_h - 0.004
    jog_area_top = jog_header_y - 0.020
    jog_slider_left = panel_left
    jog_slider_width = panel_width
    jog_slider_height = 0.014
    jog_slider_gap = 0.007
    jog_block_pad = 0.012
    jog_block_height = 3 * (jog_slider_height + jog_slider_gap) + jog_block_pad

    def style_slider_axes(slider_ax) -> None:
        theme.style_widget_axes(slider_ax)
        slider_ax.tick_params(labelsize=7)

    jog_ax1_sliders: list[Slider] = []
    jog_ax2_sliders: list[Slider] = []
    jog_thrust_sliders: list[Slider] = []
    for idx, mount in enumerate(mounts):
        block_top = jog_area_top - idx * jog_block_height
        slider_ys = [
            block_top - j * (jog_slider_height + jog_slider_gap) for j in range(3)
        ]
        max_ax2_min = float(np.rad2deg(allocator.min_angles[idx, 0]))
        max_ax2_max = float(np.rad2deg(allocator.max_angles[idx, 0]))
        max_ax1_min = float(np.rad2deg(allocator.min_angles[idx, 1]))
        max_ax1_max = float(np.rad2deg(allocator.max_angles[idx, 1]))

        rest_a2 = float(np.rad2deg(mount.rest_angles[0]))
        rest_a1 = float(np.rad2deg(mount.rest_angles[1]))

        ax_jog_a2 = fig.add_axes(
            [jog_slider_left, slider_ys[0], jog_slider_width, jog_slider_height]
        )
        ax_jog_a1 = fig.add_axes(
            [jog_slider_left, slider_ys[1], jog_slider_width, jog_slider_height]
        )
        ax_jog_f = fig.add_axes(
            [jog_slider_left, slider_ys[2], jog_slider_width, jog_slider_height]
        )
        for slider_ax in (ax_jog_a2, ax_jog_a1, ax_jog_f):
            style_slider_axes(slider_ax)

        slider_a2 = Slider(
            ax_jog_a2,
            f"M{idx} a2",
            max_ax2_min,
            max_ax2_max,
            valinit=rest_a2,
            **slider_theme_kwargs(),
        )
        slider_a1 = Slider(
            ax_jog_a1,
            f"M{idx} a1",
            max_ax1_min,
            max_ax1_max,
            valinit=rest_a1,
            **slider_theme_kwargs(),
        )
        slider_f = Slider(
            ax_jog_f,
            f"M{idx} F",
            0.0,
            mount.thrust,
            valinit=mount.thrust,
            **slider_theme_kwargs(),
        )
        for slider in (slider_a1, slider_a2, slider_f):
            theme.style_slider(slider, active=True)
            slider.label.set_fontsize(8)
            slider.valtext.set_fontsize(8)
        jog_ax1_sliders.append(slider_a1)
        jog_ax2_sliders.append(slider_a2)
        jog_thrust_sliders.append(slider_f)

    jog_angle_sliders = jog_ax2_sliders + jog_ax1_sliders
    jog_sliders = jog_angle_sliders + jog_thrust_sliders
    control_mode = "auto"
    updating_jog_sliders = False
    slider_axes = {
        ax_tx,
        ax_ty,
        ax_tz,
        ax_force,
    }
    for slider in jog_sliders:
        slider_axes.add(slider.ax)

    def style_slider(slider: Slider, active: bool) -> None:
        slider.set_active(active)
        theme.style_slider(slider, active=active)
        marker_line = getattr(slider, "vline", None) or getattr(slider, "hline", None)
        if marker_line is not None:
            fill = theme.ACCENT if active else theme.INACTIVE_FILL
            marker_line.set_color(fill)
            marker_line.set_alpha(0.85 if active else 0.3)
        if hasattr(slider, "track"):
            slider.track.set_alpha(1.0 if active else 0.55)
        if hasattr(slider, "poly"):
            slider.poly.set_alpha(0.9 if active else 0.35)

    def set_slider_group(sliders: list[Slider], active: bool) -> None:
        for slider in sliders:
            style_slider(slider, active)
        if sliders:
            sliders[0].ax.figure.canvas.draw_idle()

    def read_jog_angles() -> np.ndarray:
        angles = np.zeros((mount_count, 2), dtype=float)
        for idx in range(mount_count):
            angles[idx, 0] = np.deg2rad(jog_ax2_sliders[idx].val)
            angles[idx, 1] = np.deg2rad(jog_ax1_sliders[idx].val)
        return angles

    def read_jog_thrusts() -> np.ndarray:
        return np.array([slider.val for slider in jog_thrust_sliders], dtype=float)

    def sync_jog_angles(angles: np.ndarray) -> None:
        nonlocal updating_jog_sliders
        updating_jog_sliders = True
        for idx in range(mount_count):
            jog_ax2_sliders[idx].set_val(float(np.rad2deg(angles[idx, 0])))
            jog_ax1_sliders[idx].set_val(float(np.rad2deg(angles[idx, 1])))
        updating_jog_sliders = False

    force_weight = render.force_weight
    torque_scale = 0.7 / max(1.0, allocator.torque_upper_bound)
    preview_fps = 60
    preview_interval_ms = max(1, int(round(1000.0 / preview_fps)))
    settle_interval_ms = 120
    preview_pending = False
    preview_scheduled = False
    render_in_progress = False
    preview_timer = fig.canvas.new_timer(interval=preview_interval_ms)
    preview_timer.single_shot = True
    settle_timer = fig.canvas.new_timer(interval=settle_interval_ms)
    settle_timer.single_shot = True

    def refresh_canvas(*, fast: bool) -> None:
        fig.canvas.draw_idle()
        if fast:
            fig.canvas.flush_events()

    center_of_mass = np.asarray(allocator.center_of_mass, dtype=float)
    com_marker = ax.scatter(
        [center_of_mass[0]],
        [center_of_mass[1]],
        [center_of_mass[2]],
        s=135,
        marker="X",
        c="#ffd166",
        edgecolors="#f4f1de",
        linewidths=1.2,
        alpha=line_alpha,
        depthshade=False,
        label="center of mass",
    )

    arm_points = [center_of_mass]
    mast_drawn = False
    for arm in catch_arms or []:
        root = np.asarray(arm.position, dtype=float)
        tip = arm.tip()
        arm_points.extend([root, tip])
        ax.plot(
            [root[0], tip[0]],
            [root[1], tip[1]],
            [root[2], tip[2]],
            color="#e9ecef",
            linewidth=2.4,
            alpha=line_alpha,
            solid_capstyle="round",
        )
        if (
            not mast_drawn
            and float(np.linalg.norm(root[:2])) < 1e-9
            and abs(float(root[2])) > 1e-9
        ):
            ax.plot(
                [0.0, root[0]],
                [0.0, root[1]],
                [0.0, root[2]],
                color="#868e96",
                linewidth=1.4,
                alpha=0.7 * line_alpha,
                linestyle="--",
            )
            mast_drawn = True

    extent = max(
        float(np.max(np.linalg.norm(positions, axis=1))) if len(positions) else 1.0,
        float(np.linalg.norm(center_of_mass)),
        *(float(np.linalg.norm(p)) for p in arm_points),
    ) + force_scale
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("TVC allocation", color=text_color, pad=12)
    ax.view_init(elev=25, azim=35)
    active_button_color = theme.BUTTON_ACTIVE
    active_button_hovercolor = theme.BUTTON_ACTIVE_HOVER
    inactive_button_color = theme.BUTTON
    inactive_button_hovercolor = theme.BUTTON_HOVER

    def style_panel_button(btn: Button, active: bool) -> None:
        face = active_button_color if active else inactive_button_color
        btn.color = face
        btn.hovercolor = active_button_hovercolor if active else inactive_button_hovercolor
        btn.ax.set_facecolor(face)
        btn.ax.figure.canvas.draw_idle()

    def panel_button_rect(y: float) -> list[float]:
        return [panel_left, y, panel_width, panel_btn_h]

    vector_button_specs = [
        ("Target torque", vector_button_y_positions[0]),
        ("Displayed torque", vector_button_y_positions[1]),
        ("Target Fz", vector_button_y_positions[2]),
        ("Net force", vector_button_y_positions[3]),
        ("Mount axes", vector_button_y_positions[4]),
        ("Thrust range", vector_button_y_positions[5]),
    ]

    fig.text(
        panel_left,
        display_header_y,
        "Torque / force display",
        color=text_color,
        fontsize=9,
        weight="bold",
        ha="left",
    )
    mode_button_ax = fig.add_axes(panel_button_rect(mode_button_y))
    theme.style_widget_axes(mode_button_ax)
    mode_button = Button(
        mode_button_ax,
        "Manual jog",
        color=inactive_button_color,
        hovercolor=inactive_button_hovercolor,
    )
    mode_button.label.set_fontsize(8)
    mode_button.label.set_color(text_color)
    fig.text(
        panel_left,
        alloc_header_y,
        "Allocation",
        color=text_color,
        fontsize=8,
        weight="bold",
        ha="left",
    )
    fig.text(
        panel_left,
        jog_header_y,
        "Mount jog (deg / N)",
        color=text_color,
        fontsize=8,
        weight="bold",
        ha="left",
    )
    vector_lines = {
        "Target torque": target_line,
        "Displayed torque": actual_line,
        "Target Fz": target_force_plane,
        "Net force": actual_force_line,
    }
    axis_line_refs = [line for mount_lines in mount_axis_lines for line in mount_lines]
    vector_lines["Mount axes"] = axis_line_refs[0]
    vector_lines["Thrust range"] = envelope_collections[0]
    vector_buttons: dict[str, Button] = {}
    vector_state: dict[str, bool] = {label: True for label in vector_lines}

    def style_toggle_button(label: str) -> None:
        btn = vector_buttons[label]
        active = vector_state[label]
        style_panel_button(btn, active)
        btn.label.set_color(text_color if active else theme.MUTED)
        btn.label.set_text(label)

    def target_fz_visible() -> bool:
        return vector_state["Target Fz"]

    def update_target_fz_display() -> None:
        target_force_plane.set_visible(target_fz_visible())
        if "Target Fz" in vector_buttons:
            style_toggle_button("Target Fz")

    def toggle_vector(label: str) -> None:
        vector_state[label] = not vector_state[label]
        if label == "Mount axes":
            visible = vector_state[label]
            for line in axis_line_refs:
                line.set_visible(visible)
        elif label == "Thrust range":
            visible = vector_state[label]
            for collection in envelope_collections:
                collection.set_visible(visible)
        else:
            vector_lines[label].set_visible(vector_state[label])
        if label == "Target Fz":
            update_target_fz_display()
        else:
            style_toggle_button(label)

    for label, y in vector_button_specs:
        btn_ax = fig.add_axes(panel_button_rect(y))
        theme.style_widget_axes(btn_ax)
        btn = Button(
            btn_ax,
            label,
            color=inactive_button_color,
            hovercolor=inactive_button_hovercolor,
        )
        btn.label.set_fontsize(7.2)
        btn.on_clicked(lambda _event, lab=label: toggle_vector(lab))
        vector_buttons[label] = btn
        style_toggle_button(label)

    def style_mode_button() -> None:
        manual = control_mode == "manual"
        style_panel_button(mode_button, True)
        mode_button.label.set_text("Manual jog" if manual else "Auto solve")
        mode_button.label.set_color(text_color)

    def set_control_mode(mode: str) -> None:
        nonlocal control_mode
        control_mode = mode
        auto_active = mode == "auto"
        set_slider_group(auto_sliders, auto_active)
        set_slider_group(jog_angle_sliders, not auto_active)
        set_slider_group(jog_thrust_sliders, True)
        set_slider_group([slider_force], auto_active)
        style_mode_button()
        render_state()

    def toggle_control_mode(_event=None) -> None:
        set_control_mode("manual" if control_mode == "auto" else "auto")

    mode_button.on_clicked(toggle_control_mode)

    def render_state(*, fast: bool = False) -> None:
        nonlocal render_in_progress
        if render_in_progress:
            schedule_preview()
            return
        render_in_progress = True
        try:
            target = np.array([slider_tx.val, slider_ty.val, slider_tz.val], dtype=float)
            force_target = slider_force.val
            if control_mode == "auto":
                thrusts = read_jog_thrusts()
                angles, torque, dirs, saturated = allocator.allocate(
                    target,
                    force_target=force_target,
                    force_weight=force_weight,
                    thrusts=thrusts,
                    fast=fast,
                )
                if not fast and not updating_jog_sliders:
                    sync_jog_angles(angles)
            else:
                angles = read_jog_angles()
                thrusts = read_jog_thrusts()
                torque, net_force_vec, dirs = allocator.torque_force_dirs(angles, thrusts)
                saturated = False
            target_norm = np.linalg.norm(target)
            target_dir = target / target_norm if target_norm > 0.0 else None
            for i, line in enumerate(thrust_lines):
                start = mounts[i].position
                thrust_vis = force_scale * (thrusts[i] / max_mount_thrust)
                end = start + dirs[i] * thrust_vis
                line.set_data([start[0], end[0]], [start[1], end[1]])
                line.set_3d_properties([start[2], end[2]])

            if not fast:
                for i, mount in enumerate(mounts):
                    primary_angle = angles[i][0]
                    rot_primary = axis_angle_rotation(mount.axis2, primary_angle)
                    nested_axis1 = unit(rot_primary @ mount.axis1)
                    axis_directions = (mount.base_dir, mount.axis2, nested_axis1)
                    for line, direction in zip(mount_axis_lines[i], axis_directions):
                        start = mount.position
                        end = mount_axis_end(mount, direction)
                        line.set_data([start[0], end[0]], [start[1], end[1]])
                        line.set_3d_properties([start[2], end[2]])
                for collection in envelope_collections:
                    if vector_state.get("Thrust range", True):
                        collection.set_visible(True)
            else:
                for collection in envelope_collections:
                    collection.set_visible(False)

            com = center_of_mass
            target_end = target * torque_scale
            displayed_torque = torque
            if saturated and target_dir is not None:
                displayed_torque = target_dir * float(np.dot(torque, target_dir))
            actual_end = displayed_torque * torque_scale
            actual_line.set_data(
                [com[0], com[0] + actual_end[0]],
                [com[1], com[1] + actual_end[1]],
            )
            actual_line.set_3d_properties([com[2], com[2] + actual_end[2]])
            target_line._offsets3d = (
                [com[0] + target_end[0]],
                [com[1] + target_end[1]],
                [com[2] + target_end[2]],
            )

            force_target_mag = float(force_target)

            cumulative = np.zeros(3, dtype=float)
            net_force = np.zeros(3, dtype=float)
            force_cumulative = np.zeros(3, dtype=float)
            if fast:
                for mount, direction, thrust in zip(mounts, dirs, thrusts):
                    net_force += thrust * direction
            else:
                for torque_line, force_line, mount, direction, thrust in zip(
                    segment_lines, actual_force_segment_lines, mounts, dirs, thrusts
                ):
                    force_vec = thrust * direction
                    net_force += force_vec
                    torque_seg = np.cross(mount.position - com, force_vec)
                    start = com + cumulative * torque_scale
                    cumulative = cumulative + torque_seg
                    end = com + cumulative * torque_scale
                    torque_line.set_data([start[0], end[0]], [start[1], end[1]])
                    torque_line.set_3d_properties([start[2], end[2]])
                    force_start = com + force_cumulative * force_vector_scale
                    force_cumulative = force_cumulative + force_vec
                    force_end = com + force_cumulative * force_vector_scale
                    force_line.set_data(
                        [force_start[0], force_end[0]], [force_start[1], force_end[1]]
                    )
                    force_line.set_3d_properties([force_start[2], force_end[2]])

            if not fast:
                target_plane_z = force_target_mag * force_vector_scale
                plane_vertices = [[
                    (
                        com[0] - plane_half_span,
                        com[1] - plane_half_span,
                        com[2] + target_plane_z,
                    ),
                    (
                        com[0] + plane_half_span,
                        com[1] - plane_half_span,
                        com[2] + target_plane_z,
                    ),
                    (
                        com[0] + plane_half_span,
                        com[1] + plane_half_span,
                        com[2] + target_plane_z,
                    ),
                    (
                        com[0] - plane_half_span,
                        com[1] + plane_half_span,
                        com[2] + target_plane_z,
                    ),
                ]]
                target_force_plane.set_verts(plane_vertices)
                target_force_plane.set_visible(target_fz_visible())
            actual_axial_force = float(np.dot(net_force, force_axis))
            actual_force_end = net_force * force_vector_scale
            actual_force_line.set_data(
                [com[0], com[0] + actual_force_end[0]],
                [com[1], com[1] + actual_force_end[1]],
            )
            actual_force_line.set_3d_properties(
                [com[2], com[2] + actual_force_end[2]]
            )

            if not fast:
                torque_error = target - torque
                torque_error_norm = float(np.linalg.norm(torque_error))
                force_actual_mag = float(np.dot(net_force, force_axis))
                force_error_norm = abs(force_target_mag - force_actual_mag)
                force_tolerance = max(1e-3, 0.01 * max(abs(force_target_mag), 1.0))
                force_coupling_active = force_weight > 0.0
                force_saturated = (
                    force_coupling_active and force_error_norm > force_tolerance
                )
                force_mode = "ACTIVE" if force_coupling_active else "OFF"
                mode_label = "MANUAL JOG" if control_mode == "manual" else "AUTO SOLVE"
                force_line = (
                    "Target axial F: "
                    f"{format_scalar(force_target_mag)}\n"
                    "Measured axial F: "
                    f"{format_scalar(force_actual_mag)}\n"
                    "Net force vector: "
                    f"{format_vector(net_force)}\n"
                )
                thrust_line = "Per-mount thrust: " + "  ".join(
                    f"M{idx}={format_scalar(thrusts[idx])}" for idx in range(mount_count)
                )
                message = f"Mode: {mode_label}\n"
                message += (
                    "Target torque: "
                    f"{format_vector(target)}\n"
                    "Measured torque: "
                    f"{format_vector(torque)}\n"
                )
                if saturated and target_dir is not None:
                    message += (
                        "Displayed torque: "
                        f"{format_vector(displayed_torque)}\n"
                    )
                message += force_line
                message += thrust_line + "\n"
                message += f"Torque error |ΔT|: {torque_error_norm:6.3f}\n"
                if control_mode == "auto":
                    message += f"Force error  |ΔF|: {force_error_norm:6.3f}\n"
                    message += (
                        f"Force coupling: {force_mode}  weight={force_weight:5.3f}\n"
                    )
                    message += f"Torque saturated: {'YES' if saturated else 'NO'}\n"
                    message += f"Force saturated: {'YES' if force_saturated else 'NO'}\n"
                message += "\n"
                message += format_angles_deg(angles)
                text.set_text(message)
            refresh_canvas(fast=fast)
        finally:
            render_in_progress = False

    fig.text(
        panel_left,
        0.10,
        "Auto: Tx/Ty/Tz + Fmag + per-mount F\n"
        "Manual: all jog sliders\n\n"
        "Mount axes\n"
        "  base_dir  mint dashed\n"
        "  axis2     pink (primary)\n"
        "  axis1     blue (nested)\n"
        "  thrust range orange sphere patch (inward)\n"
        "  target torque red tip point\n"
        "  measured torque green solid\n"
        "  target Fz purple dotted plane\n"
        "  net force cyan solid + chain",
        color=theme.MUTED,
        fontsize=7.2,
        ha="left",
        va="bottom",
    )

    def schedule_preview(_=None) -> None:
        nonlocal preview_pending, preview_scheduled
        preview_pending = True
        if not preview_scheduled:
            preview_scheduled = True
            preview_timer.start()
        settle_timer.stop()
        settle_timer.start()

    def run_preview() -> None:
        nonlocal preview_pending, preview_scheduled
        preview_scheduled = False
        if not preview_pending:
            return
        preview_pending = False
        render_state(fast=True)
        if preview_pending and not preview_scheduled:
            preview_scheduled = True
            preview_timer.start()

    def run_settled_render() -> None:
        if preview_pending:
            settle_timer.start()
            return
        render_state(fast=False)

    def on_slider_release(event) -> None:
        if event.inaxes not in slider_axes:
            return
        nonlocal preview_pending, preview_scheduled
        preview_pending = False
        preview_scheduled = False
        preview_timer.stop()
        settle_timer.stop()
        render_state(fast=False)

    preview_timer.add_callback(run_preview)
    settle_timer.add_callback(run_settled_render)
    fig.canvas.mpl_connect("button_release_event", on_slider_release)

    def on_angle_jog_changed(_=None) -> None:
        if updating_jog_sliders or control_mode != "manual":
            return
        schedule_preview()

    def on_thrust_jog_changed(_=None) -> None:
        if updating_jog_sliders:
            return
        schedule_preview()

    slider_tx.on_changed(schedule_preview)
    slider_ty.on_changed(schedule_preview)
    slider_tz.on_changed(schedule_preview)
    slider_force.on_changed(schedule_preview)
    for slider in jog_angle_sliders:
        slider.on_changed(on_angle_jog_changed)
    for slider in jog_thrust_sliders:
        slider.on_changed(on_thrust_jog_changed)
    update_target_fz_display()
    set_control_mode("auto")
    plt.show()
